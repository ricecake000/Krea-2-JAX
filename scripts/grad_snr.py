"""Gradient signal / noise per timestep bin of the full fine-tune (base Raw,
B = 1, one GPU): does reweighting / stratifying the training timestep
distribution raise the gradient S/N enough to matter?

Design (all numbers per parameter set; P = --params, P' = --cross-params, a
subset of P):
  * Bins b = 0..K-1 (--bins K, default 8): equal-probability quantiles of
    the ACTUAL training t distribution of the chosen bucket. train_full draws
    u = logit_mean + logit_std * z, z ~ N(0, 1), t = shift_time(sigmoid(u),
    mu) with mu = per_sample_mu(n_img) (constant within a bucket); t is
    monotone in z, so bin b is z in [Phi^-1(b/K), Phi^-1((b+1)/K)) and a bin
    draws v ~ U(b/K, (b+1)/K), z = Phi^-1(v): the equal-weight mixture of the
    bins IS the training distribution.
  * Paired images: the same N images (one bucket, --samples from `select`)
    in every bin, each once per bin with fresh noise (key = fold_in(seed,
    bin, image)); per-sample gradients g (B = 1) of exactly train_full's loss
    (masked_flow_loss, its text template and online text states, bf16
    forward/backward from bf16 weights = train_full's per-step bf16 cast;
    the gradient of the bf16 cast equals train_full's f32 gradient values).
    Only P gets a gradient; every other weight is a non-differentiated jit
    argument.
  * Image i goes to group i mod G (--groups G, default 8; interleaved, the
    same in every bin). Per bin: G group-sum accumulators S_g (f32, on
    device) and per sample and tensor ||g||^2. At the end of a bin the
    per-tensor Gram H_gh = <S_g, S_h> (G x G) is formed on device
    (elementwise products, f32 row partials, f64 on the host: no TF32 dot)
    and the P' part of the S_g goes to a host memmap; vectors never leave
    the VM, only Gram matrices and scalars.
  * Estimators (unbiased, no ||S||^2 - Q cancellation):
      ||mu_b||^2 = sum_{g!=h} H_gh / sum_{g!=h} n_g n_h   (image-disjoint pairs)
      E||g_b||^2 = Q / N,  tr Sigma_b = E||g_b||^2 - ||mu_b||^2,
      per-sample S/N_b = ||mu_b||^2 / tr Sigma_b;
    split-half cos(S_A, S_B) with A = even groups, B = odd groups (= the
    interleaved halves); detectability = the exact sign-flip null over the
    G group sums (2^(G-1) patterns) for the split-half cos and for the
    U statistic sum_{g!=h} H_gh; CIs = delete-one-group jackknife (t_{G-1}).
  * Cross-bin (P'): <mu_a, mu_b> = sum_{g!=h} <S_a,g, S_b,h> / sum_{g!=h}
    n_g n_h (image-disjoint, so unbiased although the images are paired) ->
    cos(mu_a, mu_b), the pooled mean ||mu||^2 = mean_ab <mu_a, mu_b>, and the
    payoff: predicted per-sample S/N of the training estimator under
      (a) the current t distribution (uniform over bins):
          V = mean_b E||g_b||^2 - ||mu||^2
      (b) proportional stratification (one t per bin per K samples):
          V = mean_b tr Sigma_b                 (between-bin term removed)
      (c1) importance sampling, objective unchanged: q_b ~ sqrt(E||g_b||^2),
          weight 1/(K q_b): V = (mean_b sqrt(E||g_b||^2))^2 - ||mu||^2
      (c2) stratified Neyman allocation, objective unchanged: n_b ~
          sqrt(tr Sigma_b): V = (mean_b sqrt(tr Sigma_b))^2
      (d) objective reweighting w >= 0 (changes the objective), equal
          allocation: S/N(w) = w'Mw / (K sum_b w_b^2 tr Sigma_b), maximized
          exactly over the nonnegative orthant (every support enumerated,
          reweight_nonneg), w normalized to sum 1;
    S/N = ||mu||^2 / V, gains = S/N / S/N(a), all with jackknife CIs. When
    the per-sample S/N is << 1 the gains of (b)/(c) are ratios of second
    moments, i.e. well measured even where ||mu||^2 itself is noisy.
    ||mu||^2 and tr Sigma are unbiased, their ratio (S/N) is not: biased
    upward at small N (mean 1.07x the truth at N = 32, G = 8, Gaussian);
    the jackknife CIs are approximate (t_{G-1}, G - 1 = 7 df).
  * Detectability rule (written into the report): E cos(S_A, S_B) ~
    r / (1 + r), r = (N/2) S/N_b, and the null sd of the cos does not depend
    on N (~1/sqrt(d_eff)); a bin is detectable when the sign-flip p of both
    the split-half cos and U is <= 0.05 AND the jackknife 95% lower bound of
    ||mu_b||^2 is > 0. The sign-flip null assumes the group sums are about
    sign-symmetric under mu = 0: true for symmetric heavy tails and
    per-sample scale mixtures, close for skewness spread over many
    coordinates; a skewed shared direction is the weak case (the jackknife
    bound keeps it at <= 5% false positives in the scratch calibration), and
    a rare-event null whose samples usually coincide with a real-mean twin
    cannot be controlled by any finite-N test without giving up most of the
    power (P_null(reject) >= P(no rare draw) x P_twin(reject)). A GO is
    therefore conditional on that assumption: a reason to run phase 2, not
    5% evidence of signal. The N needed for detection at
    z = 2 null sd is N* = max(2c / ((1 - c) S/N_b) with c = 2 sd_null,
    N t_{G-1} / (||mu||^2 / se_jackknife)) (both statistics grow ~ N) (sd_null
    from the sign-flip null; a present signal inflates it, so N* overstates
    the N of an already detectable bin; exact in the noise-dominated case).

Modes:
  select   (laptop, pyarrow only) metadata.parquet -> samples json: M ids of
           one bucket from a few latent shards (fixed seed, shuffled), one
           caption per image drawn like training (long/medium/short
           0.5/0.3/0.2, ", drawn by <artist>" with p 0.5), the shard digests
  smoke    (phase 1, the go/no-go) platform/version/memory report and must-
           fail checks (GPU compute capability 12.x unless --allow-cpu);
           AOT compile of the per-sample step for each --remat-candidates
           (peak memory; the first that fits is used); text states of the N
           captions with the JAX Qwen3 tower (then freed); Raw loaded tensor
           by tensor straight to the device (bf16); s/sample; optional cuDNN
           attention timing + gradient cosine vs ref (--try-cudnn); the
           extreme bins (--run-bins, default 0 and K-1) with N each; report
           incl. the projected N and GO / NO-GO
  measure  (phase 2) the same pipeline for --run-bins (default all K)
  auto     smoke, gate and phase 2 in ONE process (one billed session): the
           inputs of all shipped images are loaded up front (a fresh N_m up
           to that needs no reload), bins run in the order 0, K-1, 1, ...,
           K-2, all at --n-per-bin N; after 0 and K-1 the gate
           (gate_decision) picks, pre-registered (N_RULE):
             NO-GO -> stop (exit 0, the report says NO-GO);
             GO because a gate bin is detectable -> N_m = N, the K-2 other
               bins continue in the SAME state (0 and K-1 reused);
             GO only by projection (N* <= n_fit, nothing detectable) ->
               N_m = min(N* rounded up to G, n_fit rounded down to G, the
               shipped images); N_m > N: a FRESH measurement of all K bins
               at N_m (new noise and t draws: seed + 1, so the result does
               not reuse the draws the gate was decided on; the images are
               the first N_m, the gate's N among them); N_m <= N: as the
               detectable case;
           n_fit = min(shipped images, n of --budget-cu (the CU allowed
           after the gate) minus --overhead-min, n of the time left before
           --deadline-min minus --overhead-min), each for K fresh bins at
           bin_est (conservative for the K-2 bins of the reuse case: one
           number, no circularity). bin_est uses the measured s/sample of
           the chosen attention (or the gate bins' own rate if slower), the
           same estimate the per-bin deadline check uses, so a GO never
           plans bins the deadline would cut: a GO whose bins (K-2 at N or
           K at N_m) do not fit stops instead (verdict GO, action stop).
           cuDNN attention (--try-cudnn) is compiled in a background thread
           while the ref step compiles and the inputs download / load (after
           the ref timing it is waited for at most --cudnn-wait-s, never into
           the reserve of the first bins; still compiling -> ref); it
           is used for ALL bins iff its gradient cos vs ref >= 0.999 AND its
           median s/sample <= 0.90 x ref (decided before the first bin,
           recorded in the config; any cuDNN failure -> ref). Output:
           out/smoke/ = the gate snapshot (state / report of bins 0, K-1
           + the decision, frozen at the gate); out/measure/ = the phase-2
           result (reuse: the continuing state, bins 0 and K-1 included;
           fresh: the new state); the final answer is always
           out/measure/report.txt. One `GRADSNR_GATE {json}` line
  report   re-print the report of --out (state.json; out/measure or
           out/smoke of an auto run) or of a job.log (--from-log: the last
           GRADSNR_STATE line, of --stage smoke|measure if given)
After every bin: state.json + report.json + report.txt in --out (auto: the
stage dir) and one `GRADSNR_STATE stage=<stage> <base64 gzip json>` line on
stdout (the job log is fetched even when the runner hits its wall cap; lines
without the stage token are from older runs and still decode). The host
part of a bin (P' memmap write, cross-bin products, state, report) runs on a
single worker thread while the GPU starts the next bin; a worker error
fails the run. --resume (smoke / measure) skips bins already in state.json
(same config); --deadline-min stops before a bin that would not finish in
time (exit 6, PARTIAL). --wait-inputs / --wait-raw: ready markers of a
background downloader (grad_snr_vm.sh) waited on before reading the inputs
/ Raw (a `.failed` file next to a marker or a dead --downloader-pid fails at
once), so the compiles overlap the downloads.

    python scripts/grad_snr.py --mode select --metadata metadata.parquet \\
        --manifest latents/manifest.json --out samples.json
    python scripts/grad_snr.py --mode auto --base-ckpt raw.safetensors \\
        --text-encoder <Qwen3-VL-4B dir> --data-dir <latents dir> \\
        --samples samples.json --out out/ --scratch /content/scratch
"""

import argparse
import base64
import gzip
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# run1/run2's text encoder (Qwen/Qwen3-VL-4B-Instruct @ ebb281ec, train_full
# run config "text_encoder_id")
TRAIN_TE_ID = "ec5d13e03e6114b9658324d1620708a76248cf4a1cf9af5a41df31be19d38645"
STATE_TAG = "GRADSNR_STATE"
GATE_TAG = "GRADSNR_GATE"
CU_PER_H = {"G4": 8.90}

# two-sided 97.5% Student t quantiles (df -> t); df > 30 -> 1.96
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
         13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
         19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t975(df):
    if df > 30:
        return 1.96
    return _T975[max(k for k in _T975 if k <= df)]


# ---------------------------------------------------------------------------
# estimators (pure numpy; Gram matrices of group sums)
# ---------------------------------------------------------------------------


def _fin(x):
    return x is not None and np.isfinite(x)


def _div(a, b):
    return a / b if b > 0 else float("nan")


def offdiag_mean(H, n):
    """sum_{g != h} H_gh / sum_{g != h} n_g n_h: unbiased <mu_a, mu_b> from a
    Gram of group sums over image-disjoint group pairs."""
    n = np.asarray(n, np.float64)
    off = ~np.eye(len(n), dtype=bool)
    return float(np.asarray(H, np.float64)[off].sum()
                 / np.outer(n, n)[off].sum())


def jackknife(fn, G):
    """Delete-one-group jackknife of fn(keep) -> {name: float} (keep = bool
    mask over the G groups) -> {name: {est, se, lo, hi}} (t_{G-1} CI around
    the full-sample estimate; se/lo/hi None when a replicate is not
    finite)."""
    full = fn(np.ones(G, bool))
    reps = [fn(np.arange(G) != g) for g in range(G)]
    out = {}
    for k, v in full.items():
        r = np.array([x.get(k, float("nan")) for x in reps], np.float64)
        if not _fin(v) or not np.all(np.isfinite(r)):
            out[k] = {"est": v if _fin(v) else None, "se": None, "lo": None,
                      "hi": None}
            continue
        se = math.sqrt((G - 1) / G * float(np.sum((r - r.mean()) ** 2)))
        t = t975(G - 1)
        out[k] = {"est": float(v), "se": se, "lo": float(v) - t * se,
                  "hi": float(v) + t * se}
    return out


def _split_cos(H, a):
    b = ~a
    haa, hbb, hab = H[np.ix_(a, a)].sum(), H[np.ix_(b, b)].sum(), H[np.ix_(a, b)].sum()
    return float(hab / math.sqrt(haa * hbb)) if haa > 0 and hbb > 0 else float("nan")


def bin_stats(H, Q, n, z=2.0, alpha=0.05):
    """Per-bin statistics from the group Gram H (G, G) (summed over a
    parameter set), Q (G,) per-group sums of per-sample ||g||^2 and n (G,)
    group sizes. Returns a JSON-able dict (see the module docstring)."""
    H = np.asarray(H, np.float64)
    Q = np.asarray(Q, np.float64)
    n = np.asarray(n, np.float64)
    G = len(n)

    def fn(keep):
        mu2 = offdiag_mean(H[np.ix_(keep, keep)], n[keep])
        e2 = float(Q[keep].sum() / n[keep].sum())
        tr = e2 - mu2
        return {"mu2": mu2, "E2": e2, "trS": tr, "snr": _div(mu2, tr)}

    jk = jackknife(fn, G)
    a = np.arange(G) % 2 == 0
    cos_obs = _split_cos(H, a)
    off = ~np.eye(G, dtype=bool)
    u_obs = float(H[off].sum())
    eps = np.array([(1,) + s for s in itertools.product((1, -1), repeat=G - 1)],
                   np.float64)
    He = eps[:, :, None] * eps[:, None, :] * H[None]
    cos_null = np.array([_split_cos(h, a) for h in He])
    u_null = He[:, off].sum(axis=1)
    sd_cos = float(np.std(cos_null[1:])) if len(cos_null) > 1 else float("nan")
    sd_u = float(np.std(u_null))
    p_cos = float(np.mean(cos_null >= cos_obs - 1e-15))
    p_u = float(np.mean(u_null >= u_obs - 1e-12 * abs(u_obs)))
    splits = [np.isin(np.arange(G), (0,) + c)
              for c in itertools.combinations(range(1, G), G // 2 - 1)]
    cos_splits = np.array([_split_cos(H, s) for s in splits])
    iu = np.triu_indices(G, 1)
    tr_s2 = float(np.mean(H[iu] ** 2 / (n[iu[0]] * n[iu[1]])))
    tr = jk["trS"]["est"]
    snr = jk["snr"]["est"]
    c = z * sd_cos
    n_cos = (int(math.ceil(2 * c / ((1 - c) * snr)))
             if _fin(snr) and snr > 0 and c < 1 else None)
    # the jackknife bound: mu2 / se grows ~ N in the noise-dominated regime
    m2, se = jk["mu2"]["est"], jk["mu2"]["se"]
    n_jk = (int(math.ceil(n.sum() * t975(G - 1) / (m2 / se)))
            if _fin(m2) and m2 > 0 and se else None)
    n_needed = (max(n_cos, n_jk) if n_cos is not None and n_jk is not None
                else None)
    return {
        "N": int(n.sum()), "groups": G, **jk,
        "split_cos": cos_obs,
        "split_cos_resplits": {"mean": float(np.mean(cos_splits)),
                               "min": float(np.min(cos_splits)),
                               "max": float(np.max(cos_splits)),
                               "count": len(splits)},
        "null_cos_sd": sd_cos,
        "null_cos_q95": float(np.quantile(cos_null[1:], 0.95)) if G > 1 else None,
        "null_cos_sd_analytic": (math.sqrt(tr_s2) / tr if _fin(tr) and tr > 0
                                 else None),
        "p_cos": p_cos, "U": u_obs, "z_U": _div(u_obs, sd_u), "p_U": p_u,
        "patterns": int(len(eps)),
        # sign-flip (assumes ~sign-symmetric group sums) AND the jackknife
        # lower bound of ||mu||^2 > 0 (scratch calibration: false positives
        # <= 5% on Gaussian / t2.5 / lognormal-scaled / skewed nulls)
        "detectable": bool(p_cos <= alpha and p_u <= alpha
                           and _fin(jk["mu2"]["lo"]) and jk["mu2"]["lo"] > 0),
        "n_needed": n_needed,
    }


def reweight_nonneg(M, D):
    """Exact max over w >= 0, w != 0 of w'Mw / sum_b D_b w_b^2: with
    u = sqrt(D) w it is the Rayleigh quotient of A = D^-1/2 M D^-1/2 on the
    nonnegative orthant, attained (KKT) at an eigenvector of some principal
    submatrix A_SS with all entries of one sign on its support S; every
    support is enumerated (K <= 12). Bins with D_b <= 0 and M_bb <= 0 (no
    noise, no signal) are left out of every support; a bin with D_b <= 0 but
    M_bb > 0 (signal without noise: unbounded S/N) is reported instead of a
    number. -> (w normalized to sum 1 or None, max or nan, [zero-noise
    signal bins])."""
    M = np.asarray(M, np.float64)
    D = np.asarray(D, np.float64)
    K = len(D)
    zero_noise_signal = [b for b in range(K) if D[b] <= 0 and M[b, b] > 0]
    ok = [b for b in range(K) if D[b] > 0]
    if K > 12 or zero_noise_signal or not ok:
        return None, float("nan"), zero_noise_signal
    dm = np.zeros(K)
    dm[ok] = 1.0 / np.sqrt(D[ok])
    A = dm[:, None] * M * dm[None, :]
    best, best_u = -np.inf, None
    for r in range(1, len(ok) + 1):
        for sup in itertools.combinations(ok, r):
            idx = list(sup)
            lam, vec = np.linalg.eigh(A[np.ix_(idx, idx)])
            for j in range(r):
                v = vec[:, j] * (1.0 if vec[:, j].sum() >= 0 else -1.0)
                if np.all(v > 1e-12) and lam[j] > best:
                    best, best_u = float(lam[j]), (idx, v)
    w = np.zeros(K)
    w[best_u[0]] = best_u[1] * dm[best_u[0]]
    return w / w.sum(), best, []


def cross_stats(C, Q, n):
    """Cross-bin statistics and the payoff from C (K, G, K, G) = <S_a,g,
    S_b,h> over P' (all bins, all groups), Q (K, G) per-bin per-group sums
    of ||g||^2 over P', n (G,) group sizes (the same images in every bin)."""
    C = np.asarray(C, np.float64)
    Q = np.asarray(Q, np.float64)
    n = np.asarray(n, np.float64)
    K, G = Q.shape

    def mats(keep):
        nk = n[keep]
        off = ~np.eye(int(keep.sum()), dtype=bool)
        Ck = C[:, keep][:, :, :, keep]
        M = np.einsum("agbh,gh->ab", Ck, off) / np.outer(nk, nk)[off].sum()
        M = 0.5 * (M + M.T)
        e2 = Q[:, keep].sum(axis=1) / nk.sum()
        return M, e2

    def fn(keep):
        M, e2 = mats(keep)
        d = np.diag(M).copy()
        tr = e2 - d
        mix = float(M.mean())
        out = {"mix_mu2": mix, "between": float(d.mean() - mix)}
        v = {"a_uniform": float(e2.mean() - mix),
             "b_stratified": float(tr.mean()),
             "c1_importance": float(np.mean(np.sqrt(e2)) ** 2 - mix),
             "c2_neyman": float(np.mean(np.sqrt(np.clip(tr, 0, None))) ** 2)}
        for k, x in v.items():
            out["snr_" + k] = _div(mix, x)
            out["gain_" + k] = _div(v["a_uniform"], x)
        # stable key set: undefined -> NaN (jackknife replicates may differ)
        w, snr_d, _ = reweight_nonneg(M, K * tr)
        out["snr_d_reweight"] = snr_d
        out["gain_d_reweight"] = _div(snr_d, out["snr_a_uniform"])
        for b in range(K):
            out[f"w_d_{b}"] = float(w[b]) if w is not None else float("nan")
        for b in range(K):
            out[f"q_c1_{b}"] = float(np.sqrt(e2[b]) / np.sqrt(e2).sum())
            out[f"n_c2_{b}"] = _div(float(np.sqrt(max(tr[b], 0))),
                                    float(np.sqrt(np.clip(tr, 0, None)).sum()))
        return out

    jk = jackknife(fn, G)
    M, e2 = mats(np.ones(G, bool))
    d = np.diag(M)
    cos = [[float(M[a, b] / math.sqrt(d[a] * d[b])) if d[a] > 0 and d[b] > 0
            else None for b in range(K)] for a in range(K)]
    zns = reweight_nonneg(M, K * (e2 - d))[2]
    return {"K": K, "M": M.tolist(), "cos": cos, "E2": e2.tolist(),
            "d_zero_noise_signal_bins": zns, **jk}


def p_gain_bounds(per_bin):
    """Gains of (b), (c1), (c2) on P from per-bin {mu2, E2, trS} without the
    cross-bin term: the pooled ||mu||^2 lies in [0, mean_b ||mu_b||^2]
    (Cauchy-Schwarz), so each gain is given at both ends."""
    d = np.array([s["mu2"]["est"] for s in per_bin])
    e2 = np.array([s["E2"]["est"] for s in per_bin])
    tr = e2 - d
    out = {}
    for name, mix in (("mix0", 0.0), ("mixmax", float(max(d.mean(), 0.0)))):
        va = e2.mean() - mix
        out[name] = {
            "gain_b_stratified": _div(va, tr.mean()),
            "gain_c1_importance": _div(va, np.mean(np.sqrt(e2)) ** 2 - mix),
            "gain_c2_neyman": _div(va, np.mean(np.sqrt(np.clip(tr, 0, None))) ** 2),
        }
    return out


# ---------------------------------------------------------------------------
# state -> report
# ---------------------------------------------------------------------------


def state_arrays(state, b):
    """(H (L, G, G), Q (L, G)) of bin b."""
    r = state["bins"][str(b)]
    return np.asarray(r["H"], np.float64), np.asarray(r["Q"], np.float64)


def cross_matrix(state, bins, cross_mask):
    """(C (K', G, K', G), Q (K', G)) over P' for the given done bins."""
    G = state["groups"]
    Kp = len(bins)
    C = np.zeros((Kp, G, Kp, G))
    Qc = np.zeros((Kp, G))
    for i, b in enumerate(bins):
        H, Q = state_arrays(state, b)
        C[i, :, i, :] = H[cross_mask].sum(axis=0)
        Qc[i] = Q[cross_mask].sum(axis=0)
    for i, a in enumerate(bins):
        for j, b in enumerate(bins):
            if a < b:
                X = np.asarray(state["cross"][f"{a},{b}"], np.float64)
                C[i, :, j, :] = X
                C[j, :, i, :] = X.T
    return C, Qc


def build_report(state, z=2.0, alpha=0.05):
    cfg = state["config"]
    leaves = state["leaves"]
    n = np.asarray(state["n_per_group"], np.float64)
    cross_mask = np.array([l["cross"] for l in leaves])
    bins = sorted(int(b) for b in state["bins"])
    K = cfg["bins"]
    rep = {"config": cfg, "bins_done": bins, "bins": {},
           "rule": ("detectable: sign-flip p <= %.2f for split-half cos AND "
                    "U AND jackknife 95%% lower bound of ||mu||^2 > 0 "
                    "(assumes ~sign-symmetric group sums under mu = 0; see "
                    "the script docstring); n_needed = 2c/((1-c) snr), c = "
                    "%.1f null sd of the split-half cos (the sign-flip sd is "
                    "inflated by a present signal, so n_needed overstates N "
                    "for a bin that is already detectable), maxed with N * "
                    "t / (mu2 / se) for the jackknife bound. GO is CONDITIONAL "
                    "on sign-symmetric group sums / no unobserved heavy tail: "
                    "a rare-event zero-mean null can give a false GO at any "
                    "finite N (controlling size there would cost most of the "
                    "power), so GO is a reason to run phase 2, not 5%% "
                    "evidence of signal; S/N is biased "
                    "upward at small N (~1.07x at N 32), CIs approximate"
                    % (alpha, z))}
    for b in bins:
        H, Q = state_arrays(state, b)
        r = state["bins"][str(b)]
        per_tensor = {}
        for li, leaf in enumerate(leaves):
            s = bin_stats(H[li], Q[li], n, z, alpha)
            per_tensor[leaf["path"]] = {k: s[k] for k in ("mu2", "trS", "snr")}
            per_tensor[leaf["path"]]["p_U"] = s["p_U"]
        rep["bins"][str(b)] = {
            "t_range": r["t_range"], "t_mean": float(np.mean(r["t"])),
            "loss_mean": float(np.mean(r["loss"])), "seconds": r["seconds"],
            # outlier diagnostic for the sign-symmetry assumption
            "sq_max_over_median": (float(np.max(r["sq_total"]) / np.median(r["sq_total"]))
                                   if r.get("sq_total") else None),
            "P": bin_stats(H.sum(axis=0), Q.sum(axis=0), n, z, alpha),
            "Pcross": bin_stats(H[cross_mask].sum(axis=0),
                                Q[cross_mask].sum(axis=0), n, z, alpha),
            "per_tensor": per_tensor,
        }
    if len(bins) >= 2:
        C, Qc = cross_matrix(state, bins, cross_mask)
        cs = cross_stats(C, Qc, n)
        cs["bins"] = bins
        if bins != list(range(K)):
            # payoff needs every bin of the mixture: keep only M / cos
            cs = {k: cs[k] for k in ("K", "M", "cos", "E2", "bins")}
        rep["cross"] = cs
    if bins == list(range(K)):
        rep["P_gain_bounds"] = p_gain_bounds([rep["bins"][str(b)]["P"]
                                              for b in bins])
    return rep


def _g(x, spec=".3e"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else format(x, spec)


def _ci(d, spec=".3e"):
    return f"{_g(d['est'], spec)} [{_g(d['lo'], spec)}, {_g(d['hi'], spec)}]"


def report_text(rep):
    cfg = rep["config"]
    lines = [f"grad S/N: bucket {cfg['bucket']} mu {cfg['mu']:.4f}, K {cfg['bins']} "
             f"bins, N {cfg['n_per_bin']}, G {cfg['groups']} groups, seed "
             f"{cfg['seed']}, attn {cfg['attn']}, remat "
             f"{cfg['remat']}, |P| {cfg['p_size']:,} ({cfg['params']}), "
             f"|P'| {cfg['cross_size']:,} ({cfg['cross_params']})",
             f"rule: {rep['rule']}"]
    for scope in ("P", "Pcross"):
        lines.append(f"-- per bin, {scope} (estimate [95% CI, jackknife over groups])")
        lines.append(f"{'bin':>3} {'t range':>15} {'N':>4} {'||mu||^2':>34} "
                     f"{'tr Sigma':>11} {'S/N per sample':>34} {'cos':>7} "
                     f"{'null sd':>8} {'p_cos':>6} {'p_U':>6} {'det':>4} {'N*':>7}")
        for b, r in rep["bins"].items():
            s = r[scope]
            lo, hi = r["t_range"]
            lines.append(
                f"{b:>3} {lo:7.4f}-{hi:7.4f} {s['N']:>4} {_ci(s['mu2']):>34} "
                f"{_g(s['trS']['est']):>11} {_ci(s['snr']):>34} "
                f"{_g(s['split_cos'], '.4f'):>7} {_g(s['null_cos_sd'], '.4f'):>8} "
                f"{s['p_cos']:6.3f} {s['p_U']:6.3f} {'yes' if s['detectable'] else 'no':>4} "
                f"{s['n_needed'] if s['n_needed'] is not None else 'inf':>7}")
    lines.append("per-sample ||g||^2 on P, max / median per bin (outlier check "
                 "of the sign-flip assumption): " + ", ".join(
                     f"{b}: {_g(r.get('sq_max_over_median'), '.2f')}"
                     for b, r in rep["bins"].items()))
    cs = rep.get("cross")
    if cs:
        lines.append("-- cross-bin cos(mu_a, mu_b) on P' (bins " +
                     ",".join(map(str, cs["bins"])) + ")")
        for a, row in zip(cs["bins"], cs["cos"]):
            lines.append(f"{a:>3} " + " ".join(f"{_g(x, '7.3f')}" for x in row))
        if "mix_mu2" in cs:
            lines.append(f"pooled ||mu||^2 {_ci(cs['mix_mu2'])}; between-bin "
                         f"(1/K) sum ||mu_b - mu||^2 {_ci(cs['between'])}")
            lines.append("-- payoff on P': per-sample S/N of the training "
                         "estimator, gain vs (a)")
            for k, label in (("a_uniform", "(a) current t distribution"),
                             ("b_stratified", "(b) proportional stratification"),
                             ("c1_importance", "(c1) importance q_b ~ sqrt(E|g_b|^2)"),
                             ("c2_neyman", "(c2) stratified Neyman n_b ~ sqrt(trS_b)"),
                             ("d_reweight", "(d) objective reweight w* (changes objective)")):
                if k == "d_reweight" and cs.get("d_zero_noise_signal_bins"):
                    lines.append(f"  {label:<48} unbounded: bins "
                                 f"{cs['d_zero_noise_signal_bins']} have signal "
                                 "but zero estimated noise")
                elif "snr_" + k in cs:
                    lines.append(f"  {label:<48} S/N {_ci(cs['snr_' + k])}  gain "
                                 f"{_ci(cs['gain_' + k], '.4f')}")
            K = cs["K"]
            for pre, label in (("q_c1", "q_b (c1)"), ("n_c2", "n_b (c2)"),
                               ("w_d", "w_b (d)")):
                if f"{pre}_0" in cs:
                    lines.append(f"  {label:<10}" + " ".join(
                        _g(cs[f"{pre}_{b}"]["est"], '.3f') for b in range(K)))
    if "P_gain_bounds" in rep:
        gb = rep["P_gain_bounds"]
        lines.append("-- gains on P without the cross-bin term (pooled ||mu||^2 "
                     "at 0 / at its Cauchy-Schwarz max)")
        for k in ("gain_b_stratified", "gain_c1_importance", "gain_c2_neyman"):
            lines.append(f"  {k:<22} {_g(gb['mix0'][k], '.4f')} / "
                         f"{_g(gb['mixmax'][k], '.4f')}")
    if "go" in rep:
        go = rep["go"]
        lines.append(f"-- {go['verdict']}: {go['why']}")
        if "action" in go:
            p = go["n_fit_parts"]
            lines.append(f"   gate -> {go['action']}, N_m {go['N_m']}; N* "
                         f"{go['n_star'] if go['n_star'] is not None else 'inf'}, "
                         f"n_fit {go['n_fit']} = min(images {p['images']}, "
                         f"budget {p['budget']}, time "
                         f"{p['time'] if p['time'] is not None else 'no limit'}) at "
                         f"{go['s_per_sample']:.3f} s/sample")
            lines.append(f"   phase-2 N rule: {go['n_rule']}")
        if go["verdict"] == "GO":
            lines.append("   (conditional on the sign-symmetry / no-unobserved-"
                         "tail assumption of the rule above: a reason to run "
                         "phase 2, not 5% evidence of signal)")
    if "budget" in rep:
        bu = rep["budget"]
        lines.append(f"-- post-gate budget {bu['budget_cu']:g} CU: bins admitted until "
                     f"{bu['budget_end_after_gate_s']:.0f} s after the gate (reserve "
                     f"{bu['reserve_s']:.0f} s); spent {bu['spent_after_gate_s']:.0f} s "
                     f"= {bu['spent_after_gate_cu']:.2f} CU after the gate; "
                     + (bu["stop"] or "no stop"))
    return "\n".join(lines)


def go_no_go(rep, s_per_sample, budget_cu, overhead_min, max_images,
             rate=CU_PER_H["G4"]):
    """Phase-2 decision: GO iff at least one smoke bin is detectable at the
    phase-1 N, or the smallest projected N* fits the budget (K bins x N*
    samples at s_per_sample, plus overhead_min, at most budget_cu, and N* <=
    the shipped images)."""
    K = rep["config"]["bins"]
    det = [b for b, r in rep["bins"].items() if r["P"]["detectable"]]
    ns = [r["P"]["n_needed"] for r in rep["bins"].values()
          if r["P"]["n_needed"] is not None]
    n_star = min(ns) if ns else None
    n_budget = int(max((budget_cu / rate * 60 - overhead_min) * 60
                       / (K * s_per_sample), 0))
    n_fit = min(n_budget, max_images)
    if det:
        return {"verdict": "GO", "why": f"bins {det} detectable at N "
                f"{rep['bins'][det[0]]['P']['N']}", "n_star": n_star,
                "n_budget": n_fit}
    if n_star is not None and n_star <= n_fit:
        return {"verdict": "GO", "why": f"projected N* {n_star} <= {n_fit} "
                f"(budget {budget_cu} CU, {max_images} images)",
                "n_star": n_star, "n_budget": n_fit}
    return {"verdict": "NO-GO", "why": f"no bin detectable; projected N* "
            f"{n_star if n_star is not None else 'inf'} > {n_fit} (budget "
            f"{budget_cu} CU at {s_per_sample:.2f} s/sample, {max_images} "
            "images)", "n_star": n_star, "n_budget": n_fit}


N_RULE = ("NO-GO -> stop. GO because a gate bin is detectable -> N_m = N, "
          "the K-2 other bins continue in the same state. GO only by "
          "projection (N* <= n_fit) -> N_m = min(N* rounded up to G, n_fit "
          "rounded down to G, shipped images); N_m > N -> fresh measurement "
          "of all K bins at N_m (seed + 1), else as detectable. n_fit = "
          "min(shipped images, n of the CU budget after the gate, n of the "
          "time left), both minus the overhead, for K fresh bins at bin_est; "
          "a GO whose bins do not fit the budget and the time left stops")


def bin_est(sps, n, K, bin_overhead_s):
    """Conservative seconds of one bin of n samples (the per-bin deadline
    check and the gate): 1.25 x the samples + Grams / P' transfer / host
    cross products (~3 s per earlier bin) + bin_overhead_s; inf without a
    valid timing (fail closed)."""
    if not (np.isfinite(sps) and sps > 0):
        return float("inf")
    return 1.25 * n * sps + bin_overhead_s + 3 * K


def n_within(seconds, n_bins, sps, K, bin_overhead_s):
    """Largest n with n_bins x bin_est(sps, n) <= seconds (None: no limit)."""
    if not np.isfinite(seconds):
        return None
    if not (np.isfinite(sps) and sps > 0):
        return 0
    return max(int(math.floor((seconds / n_bins - bin_overhead_s - 3 * K)
                              / (1.25 * sps))), 0)


def gate_decision(det, n_star, N, K, G, sps, n_ship, budget_cu, overhead_min,
                  left_s, bin_overhead_s, rate=CU_PER_H["G4"]):
    """--mode auto: the pre-registered phase-2 rule (N_RULE) from the gate
    bins (det: the detectable ones, n_star: the smallest N*) at N, the
    planning s/sample, the shipped images, the CU budget after the gate and
    the seconds left before the deadline -> {verdict GO / NO-GO, action
    reuse / fresh / stop, N_m (None when stopping), why, n_star, n_fit and
    its parts}."""
    over = overhead_min * 60
    budget_s = budget_cu / rate * 3600 - over
    time_s = left_s - over
    parts = {"images": n_ship,
             "budget": n_within(budget_s, K, sps, K, bin_overhead_s),
             "time": n_within(time_s, K, sps, K, bin_overhead_s)}
    n_fit = min(v for v in parts.values() if v is not None)
    d = {"verdict": "NO-GO", "action": "stop", "N_m": None, "n_star": n_star,
         "n_fit": n_fit, "n_fit_parts": parts, "s_per_sample": sps,
         "budget_cu": budget_cu, "overhead_min": overhead_min,
         "left_s": left_s if np.isfinite(left_s) else None, "n_rule": N_RULE}
    if det:
        d.update(verdict="GO", N_m=N, why=f"bins {det} detectable at N {N}")
    elif n_star is not None and n_star <= n_fit:
        n_m = min(-(-n_star // G) * G, n_fit // G * G, n_ship)
        d.update(verdict="GO", N_m=max(n_m, N),
                 why=f"projected N* {n_star} <= n_fit {n_fit}")
    else:
        d["why"] = (f"no bin detectable; projected N* "
                    f"{n_star if n_star is not None else 'inf'} > n_fit {n_fit}")
        return d
    m = K - 2 if d["N_m"] == N else K
    need = m * bin_est(sps, d["N_m"], K, bin_overhead_s)
    if need <= budget_s * (1 + 1e-9) and need <= time_s * (1 + 1e-9):
        d["action"] = "reuse" if m == K - 2 else "fresh"
    else:
        d["why"] += (f"; but {m} bins at N {d['N_m']} need {need:.0f} s > "
                     f"budget {budget_s:.0f} s / time left {time_s:.0f} s "
                     "(after the overhead): stop")
        d["N_m"] = None
    return d


_OUT = threading.Lock()


def emit(line):
    """One locked write per line: the worker thread prints the (long)
    GRADSNR_STATE lines while the main thread logs."""
    with _OUT:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def encode_state(state):
    return base64.b64encode(gzip.compress(json.dumps(state).encode())).decode()


def decode_state_from_log(path, stage=None):
    """The last GRADSNR_STATE line (of `stage` if given) of a log; lines
    without the stage token (older runs) match only stage=None."""
    last = None
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith(STATE_TAG + " "):
                continue
            parts = line.split()
            st = parts[1][6:] if parts[1].startswith("stage=") else None
            if stage is None or st == stage:
                last = parts[-1]
    if last is None:
        raise SystemExit(f"{path}: no {STATE_TAG} line"
                         + (f" of stage {stage}" if stage else ""))
    return json.loads(gzip.decompress(base64.b64decode(last)))


def _write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def write_outputs(out, state, extra=None, stage=None):
    """Persist the raw state FIRST (state.json + the GRADSNR_STATE log
    line), then the derived report; a failing report never loses a bin."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "state.json", state)
    emit(f"{STATE_TAG} " + (f"stage={stage} " if stage else "")
         + encode_state(state))
    try:
        rep = build_report(state)
        if extra:
            rep.update(extra)
        _write_json(out / "report.json", rep)
        txt = report_text(rep)
    except Exception as e:  # the raw state is safe; report offline later
        import traceback
        traceback.print_exc()
        rep, txt = {"error": repr(e)}, f"REPORT_FAILED {e!r} (state.json is saved)"
    (out / "report.txt").write_text(txt + "\n")
    return rep, txt


# ---------------------------------------------------------------------------
# select (laptop)
# ---------------------------------------------------------------------------


def select_samples(args):
    import pyarrow.parquet as pq

    sys.path.insert(0, str(REPO))
    from krea2_jax.data import ARTIST_SUFFIX, CAPTION_FIELDS

    w, h = (int(x) for x in args.bucket.lower().split("x"))
    cols = ["id", "bucket_w", "bucket_h", "latent_shard", "artist_name",
            *CAPTION_FIELDS]
    t = pq.read_table(args.metadata, columns=cols).to_pydict()
    rows = [dict(zip(cols, v)) for v in zip(*(t[c] for c in cols))]
    rows = [r for r in rows if r["bucket_w"] == w and r["bucket_h"] == h
            and str(r["latent_shard"]).startswith("train-")]
    by_shard = {}
    for r in rows:
        by_shard.setdefault(r["latent_shard"], []).append(r)
    rng = np.random.default_rng(args.seed)
    names = sorted(by_shard)
    per = -(-args.images // args.shards)
    pick = [names[i] for i in sorted(rng.choice(len(names), args.shards,
                                                replace=False))]
    man = json.loads(Path(args.manifest).read_text())
    chosen = []
    for s in pick:
        cand = sorted(by_shard[s], key=lambda r: r["id"])
        if len(cand) < per:
            raise SystemExit(f"{s}: {len(cand)} ids of {args.bucket} < {per}")
        chosen += [cand[i] for i in sorted(rng.choice(len(cand), per,
                                                      replace=False))]
    chosen = [chosen[i] for i in rng.permutation(len(chosen))][:args.images]
    probs = np.asarray([float(x) for x in args.caption_probs.split(",")])
    entries = []
    for r in chosen:
        var = int(rng.choice(3, p=probs / probs.sum()))
        art = bool(rng.random() < args.artist_prob) and bool(r["artist_name"])
        text = (r[CAPTION_FIELDS[var]] or "") + (
            ARTIST_SUFFIX.format(r["artist_name"]) if art else "")
        entries.append({"id": int(r["id"]), "shard": r["latent_shard"],
                        "variant": CAPTION_FIELDS[var], "artist": art,
                        "caption": text})
    out = {"bucket": args.bucket, "seed": args.seed,
           "revision": args.revision,
           "metadata_sha256": hashlib.sha256(
               Path(args.metadata).read_bytes()).hexdigest(),
           "shards": {s: man["digests"][s] for s in pick},
           "entries": entries}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"selected {len(entries)} ids of {args.bucket} from shards "
          f"{pick} ({sum(len(by_shard[s]) for s in pick)} candidates); "
          f"variants " + str({v: sum(e['variant'] == v for e in entries)
                              for v in CAPTION_FIELDS})
          + f", artist {sum(e['artist'] for e in entries)} -> {args.out}")


# ---------------------------------------------------------------------------
# JAX part (smoke / measure)
# ---------------------------------------------------------------------------


def dotted_paths(tree):
    import jax
    from krea2_jax.checkpoint import _path_keys

    flat, treedef = jax.tree_util.tree_flatten_with_path(tree)
    return [".".join(_path_keys(kp)) for kp, _ in flat], treedef


def select_leaves(paths, spec):
    """Indices of the leaves whose dotted path equals or lies under one of
    the comma-separated prefixes in spec; every prefix must match."""
    prefixes = [s.strip() for s in spec.split(",") if s.strip()]
    hit = {q: 0 for q in prefixes}
    idx = []
    for i, p in enumerate(paths):
        m = [q for q in prefixes if p == q or p.startswith(q + ".")]
        for q in m:
            hit[q] += 1
        if m:
            idx.append(i)
    bad = [q for q, c in hit.items() if c == 0]
    if bad:
        raise SystemExit(f"parameter prefixes match nothing: {bad}")
    return idx


def load_dit_bf16(path, cfg):
    """Raw safetensors -> bf16 param tree on the default device with blocks
    as a per-layer list (dit_forward layer_loop="unroll"), read and placed one
    tensor at a time (host peak = one tensor; stacked txtfusion blocks: one
    field). F32 tensors are rounded to bf16 once (RNE), BF16 ones kept: the
    same values as train_full's per-step bf16 cast of its f32 master."""
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    from safetensors import safe_open

    from krea2_jax.checkpoint import _official_entries, abstract_krea2_params

    abstract = abstract_krea2_params(cfg, dtype=jnp.bfloat16, unstack=("blocks",))
    entries, abs_leaves = _official_entries(abstract)
    by_leaf = {}
    for name, li, layer, shape in entries:
        by_leaf.setdefault(li, []).append((name, layer, shape))
    bf16 = ml_dtypes.bfloat16
    leaves = [None] * len(abs_leaves)
    with safe_open(str(path), framework="numpy") as f:
        have = set(f.keys())
        want = {e[0] for e in entries}
        if have != want:
            raise SystemExit(f"{path}: keys differ from the model config: "
                             f"missing {sorted(want - have)[:5]}, extra "
                             f"{sorted(have - want)[:5]}")
        for li, items in sorted(by_leaf.items()):
            parts = []
            for name, layer, shape in sorted(items, key=lambda e: e[1] or 0):
                t = f.get_tensor(name)
                if tuple(t.shape) != tuple(shape):
                    raise SystemExit(f"{name}: shape {t.shape} != {shape}")
                parts.append(t.astype(bf16))
            arr = parts[0] if items[0][1] is None else np.stack(parts)
            leaves[li] = jax.device_put(arr)
            del parts, arr
    return jax.tree_util.tree_unflatten(jax.tree.structure(abstract), leaves)


def load_samples(args, cfg, n):
    """First n entries of --samples -> (ids, captions, x0 (n, img_tokens,
    feat) bf16 numpy, hw (2,), bucket str); latents read with safe_open and
    patchified like BucketDataset.read_latent_into (model.patchify layout)."""
    import ml_dtypes
    from safetensors import safe_open

    from krea2_jax.model import patchify

    spec = json.loads(Path(args.samples).read_text())
    ents = spec["entries"][:n]
    if len(ents) < n:
        raise SystemExit(f"--samples has {len(spec['entries'])} entries < N {n}")
    w_px, h_px = (int(x) for x in spec["bucket"].split("x"))
    unit = cfg.vae_compression * cfg.patch
    hw = (h_px // unit, w_px // unit)
    feat = cfg.channels * cfg.patch ** 2
    x0 = np.zeros((n, args.img_tokens, feat), ml_dtypes.bfloat16)
    lat_dir = Path(args.data_dir) / "latents"
    handles = {}
    try:
        for j, e in enumerate(ents):
            if e["shard"] not in handles:
                handles[e["shard"]] = safe_open(str(lat_dir / e["shard"]),
                                                framework="numpy")
            lat = handles[e["shard"]].get_tensor(str(e["id"]))
            exp = (cfg.channels, h_px // cfg.vae_compression,
                   w_px // cfg.vae_compression)
            if tuple(lat.shape) != exp:
                raise SystemExit(f"latent {e['id']}: shape {lat.shape} != {exp}")
            x0[j, :hw[0] * hw[1]] = patchify(lat[None], cfg.patch)[0]
    finally:
        handles.clear()
    return ([e["id"] for e in ents], [e["caption"] for e in ents], x0, hw,
            spec["bucket"])


def verify_shards(args, spec_path):
    """sha256 of every shard the samples use == the samples' digests == the
    data-dir manifest digests (must-fail)."""
    from krea2_jax.hashing import file_sha256

    spec = json.loads(Path(spec_path).read_text())
    man = json.loads((Path(args.data_dir) / "latents" / "manifest.json").read_text())
    for s, dig in spec["shards"].items():
        got = file_sha256(Path(args.data_dir) / "latents" / s)
        if not (got == dig == man["digests"].get(s)):
            raise SystemExit(f"shard {s}: sha256 {got} vs samples {dig} vs "
                             f"manifest {man['digests'].get(s)}")
    emit(f"latent shards verified: {sorted(spec['shards'])}")


def text_contexts(args, cfg, captions):
    """Online text states of the captions exactly like train_full (bf16
    tower, CaptionTemplate, text_states, context cast to bf16) -> list of
    (ctx (1, T, S, H) bf16, mask (1, T)) device arrays; the tower is freed."""
    import jax
    import jax.numpy as jnp

    from krea2_jax.qwen3_text import (from_hf_config, load_qwen3_text_params,
                                      load_tokenizer, resolve_text_encoder,
                                      text_encoder_identity, text_states)
    from krea2_jax.text_template import CaptionTemplate

    te_dir = resolve_text_encoder(args.text_encoder)
    over = json.loads(args.text_model_config) if args.text_model_config else {}
    txt_cfg = from_hf_config(te_dir, **over)
    t0 = time.time()
    te_id = text_encoder_identity(te_dir, txt_cfg)
    emit(f"text encoder identity {te_id} ({time.time() - t0:.0f} s)")
    if args.expect_te_id != "any" and te_id != args.expect_te_id:
        raise SystemExit(f"text encoder identity {te_id} != expected "
                         f"{args.expect_te_id} (train_full's run config)")
    template = CaptionTemplate(load_tokenizer(te_dir), text_rows=args.text_rows)
    if (txt_cfg.prefix_len != template.prefix_len or txt_cfg.hidden_size != cfg.txtdim
            or len(txt_cfg.select_layers) != cfg.txtlayers):
        raise SystemExit("text encoder does not fit the DiT config")
    t0 = time.time()
    tp = load_qwen3_text_params(te_dir, jnp.bfloat16, cfg=txt_cfg)
    enc = jax.jit(text_states, static_argnums=(1,))
    out = []
    bs = 8
    for s in range(0, len(captions), bs):
        chunk = captions[s:s + bs]
        ids, tl = template.encode_captions(chunk + [chunk[-1]] * (bs - len(chunk)))
        ctx, mask = enc(tp, txt_cfg, ids, tl)
        ctx = ctx.astype(jnp.bfloat16)
        for j in range(len(chunk)):
            out.append((ctx[j:j + 1], mask[j:j + 1]))
    jax.block_until_ready(out)
    for a in jax.tree.leaves(tp):
        a.delete()
    del tp
    lens = [int(m.sum()) for _, m in out]
    emit(f"text states of {len(out)} captions in {time.time() - t0:.0f} s "
          f"(valid rows {min(lens)}..{max(lens)} of {args.text_rows}); tower "
          "freed")
    return out, te_id


def make_sample_fn(cfg, hp, treedef, p_idx, n_leaves, img_tokens):
    """jit(p, rest, acc, x0, hw, ctx, txt_mask, key, v_lo, v_hi) ->
    (acc + g (f32, donated), per-leaf ||g||^2 (L,), loss, t): one sample of
    train_full's step_fn with t drawn from the bin v in [v_lo, v_hi) of the
    training z quantiles, gradient w.r.t. the P leaves only."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import ndtri

    import train_full as tf
    from krea2_jax.model import build_positions_and_mask_bucketed, dit_forward
    from krea2_jax.sampling import shift_time

    remat = tf.REMAT_MODES[hp.remat_policy]
    pset = set(p_idx)
    rest_idx = [i for i in range(n_leaves) if i not in pset]

    def fn(p, rest, acc, x0, hw, ctx, txt_mask, key, v_lo, v_hi):
        k_t, k_noise = jax.random.split(key)
        B = x0.shape[0]
        hw = hw.astype(jnp.int32)
        n_img = hw[:, 0] * hw[:, 1]
        mu = tf.per_sample_mu(cfg, n_img, hp.timestep_mu)
        x0f = x0.astype(jnp.float32)
        v = v_lo + (v_hi - v_lo) * jax.random.uniform(k_t, (B,))
        z = ndtri(jnp.clip(v, 1e-7, 1.0 - 1e-7))
        u = hp.logit_mean + hp.logit_std * z
        t = shift_time(jax.nn.sigmoid(u), mu)
        noise = jax.random.normal(k_noise, x0f.shape, dtype=jnp.float32)
        tt = t[:, None, None]
        xt = ((1.0 - tt) * x0f + tt * noise).astype(jnp.bfloat16)
        target = noise - x0f
        pos, mask = build_positions_and_mask_bucketed(txt_mask, hw, img_tokens)

        def loss_fn(p):
            leaves = [None] * n_leaves
            for i, a in zip(p_idx, p):
                leaves[i] = a
            for i, a in zip(rest_idx, rest):
                leaves[i] = a
            p16 = jax.tree_util.tree_unflatten(treedef, leaves)
            vel = dit_forward(
                p16, cfg, xt, ctx, t, pos, mask, lora=None, qchunk=hp.qchunk,
                remat=remat, attn_impl=hp.attn, attn_mesh=None,
                attn_block=hp.attn_block, act_sharding=None,
                layer_loop="unroll")
            return tf.masked_flow_loss(vel, target, n_img)

        loss, g = jax.value_and_grad(loss_fn)(p)
        g32 = [x.astype(jnp.float32) for x in g]
        acc = [a + x for a, x in zip(acc, g32)]
        sq = jnp.stack([jnp.sum(jnp.square(x)) for x in g32])
        return acc, sq, loss, t[0]

    return jax.jit(fn, donate_argnums=(2,))


def _sha256_array(x):
    """sha256 of an array's C-order bytes (memmap slices included), read in
    64 MiB pieces"""
    x = np.ascontiguousarray(x).reshape(-1)
    h = hashlib.sha256()
    step = (64 << 20) // max(x.itemsize, 1)
    for s in range(0, x.size, step):
        h.update(np.ascontiguousarray(x[s:s + step]).tobytes())
    return h.hexdigest()


def _pair_rows(a, b):
    import jax.numpy as jnp
    x = a * b
    return jnp.sum(x.reshape(x.shape[0], -1) if x.ndim > 1 else x[None], axis=-1)


class Grams:
    """Per-leaf inner products of f32 device arrays without dot_general
    (no TF32): elementwise product, f32 row sums on device, f64 on host."""

    def __init__(self):
        import jax
        self.f = jax.jit(_pair_rows)

    def dot(self, a, b):
        return float(np.asarray(self.f(a, b), np.float64).sum())

    def gram(self, accs, leaf):
        G = len(accs)
        H = np.zeros((G, G))
        for g in range(G):
            for h in range(g, G):
                H[g, h] = H[h, g] = self.dot(accs[g][leaf], accs[h][leaf])
        return H


def env_report(args):
    import importlib.metadata as md

    import jax

    dev = jax.devices()[0]
    info = {"jax": jax.__version__, "platform": dev.platform,
            "device_kind": getattr(dev, "device_kind", "?"),
            "compute_capability": getattr(dev, "compute_capability", None),
            "n_devices": jax.device_count()}
    for p in ("jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt",
              "jax-cuda12-plugin", "jax-cuda12-pjrt"):
        try:
            info[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    try:
        st = dev.memory_stats() or {}
        info["device_bytes_limit"] = st.get("bytes_limit")
    except Exception:
        info["device_bytes_limit"] = None
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable"):
            mem[k] = int(v.split()[0]) * 1024
    info["host_mem_total"] = mem.get("MemTotal")
    info["host_mem_available"] = mem.get("MemAvailable")
    du = shutil.disk_usage(args.scratch)
    info["scratch_free"] = du.free
    emit("env: " + json.dumps(info))
    if dev.platform == "cpu" and not args.allow_cpu:
        raise SystemExit("CPU backend: the measurement needs the GPU "
                         "(--allow-cpu for the tiny end-to-end check)")
    if dev.platform == "gpu" and not str(info["compute_capability"]).startswith("12"):
        if not args.allow_any_gpu:
            raise SystemExit(f"GPU compute capability "
                             f"{info['compute_capability']} is not sm_120 "
                             "(--allow-any-gpu to run anyway)")
    return info


_COMPILE_THREADS = []  # background cuDNN compiles (daemon threads)


def exit_past_compiles(code):
    """os._exit(code) while an abandoned cuDNN compile is still inside XLA:
    interpreter finalization under a running compile could crash or hang and
    turn the exit code; a no-op otherwise"""
    if any(th.is_alive() for th in _COMPILE_THREADS):
        emit(f"abandoned cuDNN compile still running: hard exit {code}")
        sys.stderr.flush()
        os._exit(code)


def _ready(path):
    try:
        return Path(path).read_text().strip() == "ok"
    except OSError:
        return False


def _alive(pid):
    """the process exists and is not a zombie"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def download_failed(markers):
    """the reason in a `.failed` file next to one of the ready markers"""
    for m in markers:
        f = Path(m).parent / ".failed"
        if f.exists():
            return f"{f}: {f.read_text(errors='replace').strip()[:500]}"
    return None


def run_jax(args):
    sys.path[:0] = [str(REPO), str(REPO / "scripts")]
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import dataclasses

    import jax
    import jax.numpy as jnp

    import train_full  # noqa: F401  (imported here, before the compile thread)
    from krea2_jax.checkpoint import abstract_krea2_params
    from krea2_jax.config import KREA2_CONFIG

    t_start = time.time()

    def left_s():
        """seconds left before --deadline-min (inf without one)"""
        if not args.deadline_min:
            return float("inf")
        return args.deadline_min * 60 - (time.time() - t_start)

    def deadline_check(where, need_s=0.0):
        """exit 6 (PARTIAL, like a per-bin stop) when less than need_s is
        left before `where`"""
        if not left_s() > need_s:
            emit(f"DEADLINE_STOP before {where} ({left_s():.0f} s left, "
                 f"need {need_s:.0f} s)")
            sys.exit(6)

    def wait_ready(markers, what):
        """poll (~2 s) until every ready marker says ok; a .failed file next
        to one or a dead --downloader-pid fails at once"""
        t0 = last = time.time()
        while markers:
            bad = download_failed(markers)
            if bad:
                raise SystemExit(f"DOWNLOAD_FAILED ({what}) {bad}")
            if all(_ready(m) for m in markers):
                emit(f"{what} ready (waited {time.time() - t0:.0f} s)")
                return
            if args.downloader_pid and not _alive(args.downloader_pid):
                if all(_ready(m) for m in markers):
                    continue
                raise SystemExit(
                    f"DOWNLOAD_FAILED ({what}) downloader pid "
                    f"{args.downloader_pid} exited without "
                    f"{[m for m in markers if not _ready(m)]}"
                    + (f": {download_failed(markers)}" if download_failed(markers) else ""))
            deadline_check(f"waiting for {what}", 300)
            if time.time() - last >= 60:
                emit(f"waiting for {what}: {[m for m in markers if not _ready(m)]} "
                     f"({time.time() - t0:.0f} s)")
                last = time.time()
            time.sleep(2)

    auto = args.mode == "auto"
    if auto and (args.resume or args.run_bins):
        raise SystemExit("--mode auto: --resume / --run-bins do not apply (the "
                         "gate and phase 2 run in one process, bins in a fixed order)")
    bad = download_failed(args.wait_inputs + ([args.wait_raw] if args.wait_raw else []))
    if bad:
        raise SystemExit(f"DOWNLOAD_FAILED {bad}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.scratch).mkdir(parents=True, exist_ok=True)
    env = env_report(args)
    cfg = KREA2_CONFIG
    if args.model_config:
        cfg = dataclasses.replace(cfg, **json.loads(args.model_config))
    K, G = args.bins, args.groups
    run_bins = ([int(x) for x in args.run_bins.split(",")] if args.run_bins
                else [0, K - 1] if args.mode == "smoke"
                else [0, K - 1, *range(1, K - 1)] if auto else list(range(K)))
    if any(not 0 <= b < K for b in run_bins) or G < 3 or G % 2 or K < 2:
        raise SystemExit("--run-bins must lie in [0, K); --groups even >= 4; "
                         "--bins >= 2")
    N = args.n_per_bin
    if N < 2 * G:
        raise SystemExit(f"--n-per-bin {N} < 2 * groups {G}")
    if args.timing_samples < 1:
        raise SystemExit("--timing-samples must be >= 1 (the deadline "
                         "estimate needs a measured s/sample)")
    overhead_min = (args.overhead_min if args.overhead_min is not None
                    else 5.0 if auto else 25.0)

    # --- parameter subsets on the abstract tree (no weights yet) ---
    abs_p = abstract_krea2_params(cfg, dtype=jnp.bfloat16, unstack=("blocks",))
    paths, treedef = dotted_paths(abs_p)
    abs_leaves = jax.tree.leaves(abs_p)
    p_idx = select_leaves(paths, args.params)
    c_idx = set(select_leaves(paths, args.cross_params))
    if not c_idx <= set(p_idx):
        raise SystemExit("--cross-params must be a subset of --params")
    sizes = [int(np.prod(abs_leaves[i].shape)) for i in p_idx]
    p_size = sum(sizes)
    c_size = sum(s for i, s in zip(p_idx, sizes) if i in c_idx)
    acc_bytes = G * p_size * 4
    mm_bytes = K * G * c_size * 4
    emit(f"P: {len(p_idx)} tensors, {p_size:,} params ({args.params}); P': "
         f"{len(c_idx)} tensors, {c_size:,} ({args.cross_params}); group "
         f"accumulators {acc_bytes / 2**30:.2f} GiB on device, P' memmap "
         f"{mm_bytes / 2**30:.2f} GiB in {args.scratch} (free "
         f"{env['scratch_free'] / 2**30:.0f} GiB)")
    if mm_bytes > 0.8 * env["scratch_free"]:
        raise SystemExit("P' memmap does not fit the scratch disk: shrink "
                         "--cross-params")

    hp = argparse.Namespace(
        remat_policy=None, attn=args.attn, qchunk=args.qchunk,
        attn_block=256, logit_mean=args.logit_mean, logit_std=args.logit_std,
        timestep_mu=args.timestep_mu)
    spec = json.loads(Path(args.samples).read_text())
    n_ship = len(spec["entries"])
    if N > n_ship:
        raise SystemExit(f"--n-per-bin {N} > the {n_ship} shipped images")
    # auto: inputs of every shipped image up front (a fresh N_m needs no reload)
    n_load = n_ship if auto else N
    w_px, h_px = (int(x) for x in spec["bucket"].split("x"))
    unit = cfg.vae_compression * cfg.patch
    hw = (h_px // unit, w_px // unit)
    n_img = hw[0] * hw[1]
    if n_img > args.img_tokens:
        raise SystemExit(f"bucket {spec['bucket']} has {n_img} tokens > "
                         f"--img-tokens {args.img_tokens}")
    mu = (args.timestep_mu if args.timestep_mu is not None
          else float(cfg.mu_for_seq_len(n_img)))
    feat = cfg.channels * cfg.patch ** 2

    # --- abstract args + AOT compile (memory) ---
    S = jax.ShapeDtypeStruct
    a_p = [S(abs_leaves[i].shape, jnp.bfloat16) for i in p_idx]
    a_rest = [S(l.shape, jnp.bfloat16) for i, l in enumerate(abs_leaves)
              if i not in set(p_idx)]
    a_acc = [S(abs_leaves[i].shape, jnp.float32) for i in p_idx]
    a_args = (a_p, a_rest, a_acc, S((1, args.img_tokens, feat), jnp.bfloat16),
              S((1, 2), jnp.int32),
              S((1, args.text_rows, cfg.txtlayers, cfg.txtdim), jnp.bfloat16),
              S((1, args.text_rows), jnp.bool_),
              jax.eval_shape(lambda: jax.random.PRNGKey(0)),
              S((), jnp.float32), S((), jnp.float32))
    limit = env.get("device_bytes_limit")
    # resident outside the executable: the other G - 1 group accumulators and
    # the cached text states + latents of every loaded image
    outside = ((G - 1) * p_size * 4 + n_load * (args.text_rows * cfg.txtlayers
               * cfg.txtdim * 2 + args.img_tokens * feat * 2))

    def compile_for(remat, attn):
        hp2 = argparse.Namespace(**{**vars(hp), "remat_policy": remat,
                                    "attn": attn})
        fn = make_sample_fn(cfg, hp2, treedef, p_idx, len(abs_leaves),
                            args.img_tokens)
        t0 = time.time()
        comp = fn.lower(*a_args).compile()
        ma = comp.memory_analysis()
        peak = None
        if ma is not None:
            peak = (ma.argument_size_in_bytes + ma.output_size_in_bytes
                    - ma.alias_size_in_bytes + ma.temp_size_in_bytes)
        row = {"remat": remat, "attn": attn, "compile_s": round(time.time() - t0, 1),
               "peak_bytes": peak,
               "temp_bytes": getattr(ma, "temp_size_in_bytes", None),
               "argument_bytes": getattr(ma, "argument_size_in_bytes", None),
               "outside_bytes": outside}
        emit("aot: " + json.dumps(row))
        return comp, row

    def fits(row):
        return (limit is None or row["peak_bytes"] is None
                or row["peak_bytes"] + outside <= args.mem_margin * limit)

    def cud_compile(remat):
        try:
            c, row = compile_for(remat, "cudnn")
        except Exception as e:  # optional speed-up only
            return None, {"remat": remat, "attn": "cudnn", "error": repr(e)[:500]}
        if not fits(row):
            return None, {**row, "error": f"does not fit {args.mem_margin} x "
                          f"the device limit {limit}"}
        return c, row

    def cud_bg(remat, after=None):
        """cud_compile on a daemon thread (a failing run exits without
        waiting for an optional compile) -> (thread, result box)"""
        box = {}

        def run():
            if after is not None:
                after[0].join()  # one cuDNN compile at a time
            box["r"] = cud_compile(remat)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        _COMPILE_THREADS.append(th)
        return th, box

    # the cuDNN candidate compiles on a background thread (shapes only)
    # while the ref step compiles and the inputs download / load; joined
    # before any timing so it never overlaps a measured sample
    remats = args.remat_candidates.split(",")
    try_cud = (args.try_cudnn and args.attn != "cudnn"
               and args.mode in ("smoke", "auto"))
    cud_futs = {remats[0]: cud_bg(remats[0])} if try_cud else {}
    aot = []
    comp = None
    for remat in remats:
        c, row = compile_for(remat, args.attn)
        aot.append(row)
        if fits(row):
            comp, hp.remat_policy = c, remat
            break
        del c
    if comp is None:
        raise SystemExit(f"no remat candidate fits {args.mem_margin} x the "
                         f"device limit {limit}: {aot}")
    if try_cud and hp.remat_policy not in cud_futs:
        cud_futs[hp.remat_policy] = cud_bg(hp.remat_policy, cud_futs[remats[0]])
    emit(f"using remat {hp.remat_policy}, attn {hp.attn}")

    # --- inputs ---
    deadline_check("loading the inputs", 300)
    wait_ready(args.wait_inputs, "latents + text encoder")
    if args.verify_shards:
        verify_shards(args, args.samples)
    ids, caps, x0_np, hw_s, bucket = load_samples(args, cfg, n_load)
    ctxs, te_id = text_contexts(args, cfg, caps)
    deadline_check("loading Raw")
    wait_ready([args.wait_raw] if args.wait_raw else [], "Raw")
    t0 = time.time()
    params = load_dit_bf16(args.base_ckpt, cfg)
    leaves = jax.tree.leaves(params)
    p_leaves = [leaves[i] for i in p_idx]
    rest = [l for i, l in enumerate(leaves) if i not in set(p_idx)]
    del params, leaves
    emit(f"Raw loaded as bf16 in {time.time() - t0:.0f} s")
    x0_dev = [jax.device_put(x0_np[j:j + 1]) for j in range(n_load)]
    hw_dev = jax.device_put(np.array([hw_s], np.int32))
    grams = Grams()
    c_pos = [k for k, i in enumerate(p_idx) if i in c_idx]

    def zeros_acc():
        return [jnp.zeros(abs_leaves[i].shape, jnp.float32) for i in p_idx]

    def edges(b):
        return float(b) / K, float(b + 1) / K

    def t_of(v):
        import math as m
        from statistics import NormalDist
        z = NormalDist().inv_cdf(min(max(v, 1e-12), 1 - 1e-12))
        u = args.logit_mean + args.logit_std * z
        s = 1.0 / (1.0 + m.exp(-u))
        return m.exp(mu) / (m.exp(mu) + (1.0 / s - 1.0))

    def one(f, acc, b, j, key0):
        lo, hi = edges(b)
        key = jax.random.fold_in(jax.random.fold_in(key0, b), j)
        ctx, tm = ctxs[j]
        return f(p_leaves, rest, acc, x0_dev[j], hw_dev, ctx, tm, key,
                 np.float32(lo), np.float32(hi))

    # --- timing (every mode: the per-bin time estimate of the deadline) ---
    timing = {}
    extra = {"env": env, "aot": aot, "te_id": te_id}
    base_key = jax.random.PRNGKey(args.seed)

    def median_s(f, warm):
        """median s/sample of --timing-samples samples after `warm` warm-up
        samples (bin K//2)"""
        a, ts = zeros_acc(), []
        for j in range(warm + args.timing_samples):
            t0 = time.time()
            a, _, loss, _ = one(f, a, K // 2, j % n_load, base_key)
            jax.block_until_ready(loss)
            ts.append(time.time() - t0)
        del a
        return float(np.median(ts[warm:]))

    deadline_check("timing")
    busy = any(th.is_alive() for th, _ in cud_futs.values())
    ref_sps = timing["ref_s_per_sample"] = median_s(comp, 1)
    if not (np.isfinite(ref_sps) and ref_sps > 0):
        raise SystemExit(f"invalid timing {ref_sps} s/sample")
    deadline_check("the bins")
    emit(f"timing ({hp.attn}, remat {hp.remat_policy}): {ref_sps:.3f} "
         f"s/sample (median of {args.timing_samples})")
    n_first = 2 if auto else len(run_bins)
    cud_est = 8 * ref_sps + 60
    c2 = None
    if try_cud:
        # bounded: --cudnn-wait-s, and never into the reserve of the trial +
        # the first bins; a compile still running is abandoned (daemon
        # thread, its result never read; main() exits past it)
        wait = max(0.0, min(args.cudnn_wait_s, left_s() - cud_est - n_first
                            * bin_est(ref_sps, N, K, args.bin_overhead_s)))
        t0 = time.time()
        for th, _ in cud_futs.values():
            th.join(max(0.0, t0 + wait - time.time()))
        if any(th.is_alive() for th, _ in cud_futs.values()):
            timing["cudnn_skipped"] = (f"compile not done after "
                                       f"{time.time() - t0:.0f} s (wait cap "
                                       f"{wait:.0f} s): ref")
            emit("cudnn: " + timing["cudnn_skipped"])
        else:
            res = {r: box.get("r", (None, {"remat": r, "attn": "cudnn",
                                            "error": "compile thread died"}))
                   for r, (_, box) in cud_futs.items()}
            aot.extend(row for _, row in res.values())
            c2, row = res[hp.remat_policy]
            del res
            if c2 is None:
                timing["cudnn_error"] = row["error"]
            emit(f"cudnn compile joined (waited {time.time() - t0:.0f} s): "
                 + ("ok" if c2 is not None else row["error"]))
            if c2 is not None and busy:
                # the compile ran during the ref timing: time ref again so the
                # switch rule compares undisturbed numbers
                ref_sps = timing["ref_s_per_sample"] = median_s(comp, 0)
                emit(f"timing ({hp.attn}, re-timed): {ref_sps:.3f} s/sample")
    if c2 is not None and (left_s() - cud_est
                           < n_first * bin_est(ref_sps, N, K, args.bin_overhead_s)):
        timing["cudnn_skipped"] = (f"deadline: {left_s():.0f} s left, the "
                                   f"trial ~{cud_est:.0f} s + {n_first} bins")
        c2 = None
    elif c2 is not None:
        try:
            g_ref, _, _, _ = one(comp, zeros_acc(), K // 2, 0, base_key)
            g_cud, _, _, _ = one(c2, zeros_acc(), K // 2, 0, base_key)
            cosv = (sum(grams.dot(a, b) for a, b in zip(g_ref, g_cud))
                    / math.sqrt(sum(grams.dot(a, a) for a in g_ref)
                                * sum(grams.dot(b, b) for b in g_cud)))
            del g_ref, g_cud
            timing["cudnn_s_per_sample"] = median_s(c2, 1)
            timing["cudnn_grad_cos_vs_ref"] = float(cosv)
        except Exception as e:  # optional speed-up only
            timing["cudnn_error"] = repr(e)[:500]
            c2 = None
    sps = ref_sps
    if auto:
        # pre-registered: cuDNN for ALL bins iff cos >= 0.999 AND <= 0.90 x ref
        use = (c2 is not None
               and timing["cudnn_grad_cos_vs_ref"] >= 0.999
               and timing["cudnn_s_per_sample"] <= 0.90 * ref_sps)
        timing["attn_rule"] = ("cudnn for all bins iff grad cos vs ref >= 0.999 "
                               "AND median s/sample <= 0.90 x ref, else ref "
                               "(any cuDNN failure -> ref)")
        timing["attn_choice"] = "cudnn" if use else "ref"
        if use:
            comp, hp.attn, sps = c2, "cudnn", timing["cudnn_s_per_sample"]
    c2 = None
    if try_cud:
        emit("cudnn: " + json.dumps(timing))
    extra["timing"] = timing

    # --- state ---
    def new_state(n, seed):
        config = {"bins": K, "groups": G, "n_per_bin": n, "seed": seed,
                  "bucket": bucket, "mu": mu, "hw": list(hw_s),
                  "img_tokens": args.img_tokens, "text_rows": args.text_rows,
                  "logit_mean": args.logit_mean, "logit_std": args.logit_std,
                  "attn": hp.attn, "remat": hp.remat_policy, "qchunk": args.qchunk,
                  "params": args.params, "cross_params": args.cross_params,
                  "p_size": p_size, "cross_size": c_size, "ids": ids[:n],
                  "base_ckpt": Path(args.base_ckpt).name, "te_id": te_id,
                  "model_config": args.model_config}
        return {"config": config, "groups": G,
                "n_per_group": [len(range(g, n, G)) for g in range(G)],
                "leaves": [{"path": paths[i], "size": s, "cross": i in c_idx}
                           for i, s in zip(p_idx, sizes)],
                "bins": {}, "cross": {}}

    mm_path = Path(args.scratch) / "cross_sums.npy"

    def new_mm():
        # one P' memmap on the disk at a time (a fresh measurement replaces it)
        mm_path.unlink(missing_ok=True)
        return np.lib.format.open_memmap(str(mm_path), mode="w+",
                                         dtype=np.float32, shape=(K, G, c_size))

    def run_stage(stage, out_dir, state, mm, bins, n, key0, sps, extra_s,
                  budget_end=None):
        """The bins (in order; those already in state skipped) at n samples
        into state / out_dir; a bin starts only if its estimate fits before
        the deadline and before budget_end (epoch s; auto: the post-gate CU
        budget). GPU part here; the host part of a bin (memmap
        row, cross products with every earlier bin, state, outputs) on ONE
        serial worker thread while the next bin runs, <= 2 host copies of
        the P' sums in flight. A worker error is re-raised here (before the
        next bin, at the latest at the end). -> (the stop line or None,
        (rep, txt) of the last bin or None)."""
        L = len(p_idx)
        ex = ThreadPoolExecutor(max_workers=1)
        slots = threading.Semaphore(2)
        futs = []
        done = {int(x) for x in state["bins"]}  # main-thread view
        bin_s = None
        stopped = None

        def host_part(b, X, H, Q, losses, tvals, sq_tot, secs):
            try:
                t0 = time.time()
                mm[b] = X
                mm.flush()
                # serial worker: every bin already in state has its mm row
                for a in sorted(int(x) for x in state["bins"]):
                    Xa = np.asarray(mm[a])
                    Cab = np.zeros((G, G))
                    step = 1 << 22
                    for s in range(0, c_size, step):
                        Cab += (Xa[:, s:s + step].astype(np.float64)
                                @ X[:, s:s + step].astype(np.float64).T)
                    lo_b, hi_b = sorted((a, b))
                    state["cross"][f"{lo_b},{hi_b}"] = (Cab if a < b else Cab.T).tolist()
                lo, hi = edges(b)
                entry = {"H": H.tolist(), "Q": Q.tolist(), "loss": losses,
                         "t": tvals, "t_range": [t_of(lo), t_of(hi)],
                         "seconds": secs, "sq_total": sq_tot,
                         "cross_sha256": _sha256_array(X)}
                entry["host_s"] = round(time.time() - t0, 1)
                state["bins"][str(b)] = entry
                return write_outputs(out_dir, state, extra_s, stage)
            finally:
                slots.release()

        try:
            for b in bins:
                if b in done:
                    continue
                for f in futs:
                    if f.done():
                        f.result()  # a failed host part stops the run here
                need = max(bin_est(sps, n, K, args.bin_overhead_s),
                           1.1 * (bin_s or 0))
                if not left_s() > need:
                    stopped = (f"DEADLINE_STOP before bin {b}: {left_s():.0f} s "
                               f"left < {need:.0f} s per bin")
                elif budget_end is not None and not budget_end - time.time() > need:
                    stopped = (f"BUDGET_STOP before bin {b}: "
                               f"{budget_end - time.time():.0f} s of the post-gate "
                               f"CU budget left < {need:.0f} s per bin")
                if stopped:
                    emit(stopped)
                    break
                t0 = time.time()
                accs = [zeros_acc() for _ in range(G)]
                Q = np.zeros((L, G))
                losses, tvals, sq_tot = [], [], []

                def take(j, g, sq, loss, t):
                    sq = np.asarray(sq, np.float64)
                    Q[:, g] += sq
                    sq_tot.append(float(sq.sum()))
                    losses.append(float(loss))
                    tvals.append(float(t))
                    if not np.isfinite(losses[-1]) or not np.all(np.isfinite(Q[:, g])):
                        raise SystemExit(f"bin {b} sample {j}: non-finite loss/grad")

                prev = None
                for j in range(n):
                    g = j % G
                    accs[g], sq, loss, t = one(comp, accs[g], b, j, key0)
                    if prev is not None:
                        take(*prev)  # sample j-1, read while sample j runs
                    prev = (j, g, sq, loss, t)
                take(*prev)
                H = np.stack([grams.gram(accs, k) for k in range(L)])
                slots.acquire()  # waits while 2 P' copies are in flight
                try:
                    X = np.empty((G, c_size), np.float32)  # filled in place
                    for g in range(G):
                        o = 0
                        for k in c_pos:
                            a = np.asarray(accs[g][k]).ravel()
                            X[g, o:o + a.size] = a
                            o += a.size
                except BaseException:
                    slots.release()
                    raise
                del accs
                bin_s = time.time() - t0  # GPU part incl. Grams + P' transfer
                futs.append(ex.submit(host_part, b, X, H, Q, losses, tvals,
                                      sq_tot, bin_s))
                done.add(b)
                del X
                emit(f"bin {b} done in {bin_s:.0f} s ({bin_s / n:.2f} s/sample); "
                     f"t {min(tvals):.4f}..{max(tvals):.4f}, loss "
                     f"{np.mean(losses):.4f}")
            last = None
            for f in futs:
                last = f.result()
        finally:
            ex.shutdown(wait=True)
        return stopped, last

    def finish(stage, out_dir, state, last, extra_s, bins):
        rep, txt = last if last is not None else write_outputs(
            out_dir, state, extra_s, stage)
        emit(txt)
        emit(f"total {time.time() - t_start:.0f} s")
        return rep, [b for b in bins if str(b) not in state["bins"]]

    if not auto:
        state = new_state(N, args.seed)
        state_path = out / "state.json"
        if args.resume and state_path.exists():
            old = json.loads(state_path.read_text())
            if old["config"] != state["config"]:
                raise SystemExit("--resume: state.json has a different config")
            state = old
            emit(f"resumed: bins done {sorted(state['bins'])}")
        if state["bins"]:
            # resumed bins: their P' group sums must be in this scratch
            # memmap, else every later cross-bin product with them would be
            # silently 0
            if not mm_path.exists():
                raise SystemExit(f"--resume: {mm_path} is missing (the P' sums of "
                                 f"bins {sorted(state['bins'])} live only there); "
                                 "start a fresh --out instead")
            mm = np.load(str(mm_path), mmap_mode="r+")
            if mm.shape != (K, G, c_size) or mm.dtype != np.float32:
                raise SystemExit(f"--resume: {mm_path} has shape {mm.shape} "
                                 f"{mm.dtype}, expected {(K, G, c_size)} float32")
            for b in sorted(state["bins"]):
                want = state["bins"][b].get("cross_sha256")
                got = _sha256_array(mm[int(b)])
                if got != want:
                    raise SystemExit(f"--resume: {mm_path} bin {b} sha256 {got} != "
                                     f"state.json {want} (wrong or stale scratch)")
            emit(f"resume: {mm_path} matches bins {sorted(state['bins'])}")
        else:
            mm = new_mm()
        stopped, last = run_stage(args.mode, out, state, mm, run_bins, N,
                                  base_key, sps, extra)
        if args.mode == "smoke":
            rep = last[0] if last is not None else build_report(state)
            go = go_no_go(rep, sps, args.budget_cu, overhead_min, n_ship)
            last = write_outputs(out, state, {**extra, "go": go}, "smoke")
        rep, missing = finish(args.mode, out, state, last, extra, run_bins)
        if stopped or missing:
            emit(f"PARTIAL: bins {missing} not measured ({stopped or 'deadline'})")
            sys.exit(6)
        return

    # --- auto: gate bins, gate, phase 2 ---
    sm_dir, me_dir = out / "smoke", out / "measure"
    state = new_state(N, args.seed)
    mm = new_mm()
    stopped, last = run_stage("smoke", sm_dir, state, mm, run_bins[:2], N,
                              base_key, sps, extra)
    if stopped:
        finish("smoke", sm_dir, state, last, extra, run_bins[:2])
        emit(f"PARTIAL: the gate bins did not finish ({stopped})")
        sys.exit(6)
    rep = build_report(state)
    det = [b for b, r in rep["bins"].items() if r["P"]["detectable"]]
    ns = [r["P"]["n_needed"] for r in rep["bins"].values()
          if r["P"]["n_needed"] is not None]
    # planning s/sample: the timing of the chosen attention, or the gate
    # bins' own rate if that is slower (the per-bin deadline uses the same)
    sps_plan = max([sps] + [state["bins"][str(b)]["seconds"] / N
                            for b in run_bins[:2]])
    t_gate = time.time()
    gate = gate_decision(det, min(ns) if ns else None, N, K, G, sps_plan,
                         n_ship, args.budget_cu, overhead_min, left_s(),
                         args.bin_overhead_s)
    emit(f"{GATE_TAG} {json.dumps(gate)}")
    gx = {**extra, "go": gate}
    _, txt = write_outputs(sm_dir, state, gx, "smoke")
    emit(txt)
    if gate["action"] == "stop":
        emit(f"GATE_STOP {gate['verdict']}: no phase 2 (total "
             f"{time.time() - t_start:.0f} s)")
        return
    if gate["action"] == "reuse":
        n_m, key_m = N, base_key
        # the continuing state lives in out/measure from here on
        write_outputs(me_dir, state, gx, "measure")
    else:
        n_m, key_m = gate["N_m"], jax.random.PRNGKey(args.seed + 1)
        mm.flush()
        del mm
        mm = new_mm()
        state = new_state(n_m, args.seed + 1)
    emit(f"phase 2: {gate['action']} at N {n_m} (bins "
         f"{[b for b in range(K) if str(b) not in state['bins']]})")
    # the post-gate CU budget is enforced, not only planned: no bin starts
    # unless it fits before gate time + budget - the overhead reserve (the
    # same reserve gate_decision sized the plan with: worker drain, reports)
    budget_end = t_gate + args.budget_cu / CU_PER_H["G4"] * 3600 - overhead_min * 60
    stopped, last = run_stage("measure", me_dir, state, mm, list(range(K)), n_m,
                              key_m, sps_plan, gx, budget_end)
    spent = time.time() - t_gate
    budget = {"budget_cu": args.budget_cu, "reserve_s": overhead_min * 60,
              "budget_end_epoch": budget_end,
              "budget_end_after_gate_s": budget_end - t_gate,
              "spent_after_gate_s": spent,
              "spent_after_gate_cu": spent / 3600 * CU_PER_H["G4"],
              "stop": stopped}
    _, txt = write_outputs(me_dir, state, {**gx, "budget": budget}, "measure")
    emit(txt)
    emit(f"total {time.time() - t_start:.0f} s")
    missing = [b for b in range(K) if str(b) not in state["bins"]]
    if stopped or missing:
        emit(f"PARTIAL: bins {missing} not measured ({stopped or 'deadline'})")
        sys.exit(6)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True,
                    choices=["select", "smoke", "measure", "auto", "report"])
    ap.add_argument("--out", required=True,
                    help="select: the samples json; else the output dir")
    # select
    ap.add_argument("--metadata", help="select: metadata.parquet")
    ap.add_argument("--manifest", help="select: latents/manifest.json")
    ap.add_argument("--bucket", default="832x1216")
    ap.add_argument("--images", type=int, default=128)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--caption-probs", default="0.5,0.3,0.2")
    ap.add_argument("--artist-prob", type=float, default=0.5)
    ap.add_argument("--revision", default="8c8af9b53259d175621b08e6ad1572bd23596eaf")
    # smoke / measure
    ap.add_argument("--base-ckpt")
    ap.add_argument("--text-encoder")
    ap.add_argument("--text-model-config", default=None)
    ap.add_argument("--model-config", default=None)
    ap.add_argument("--data-dir", help="dir holding latents/manifest.json + "
                    "the shards the samples use")
    ap.add_argument("--samples", help="samples json from --mode select")
    ap.add_argument("--scratch", default="/content/gradsnr_scratch",
                    help="local dir for the P' memmap (never downloaded)")
    ap.add_argument("--expect-te-id", default=TRAIN_TE_ID,
                    help="must-fail text encoder identity ('any' skips)")
    ap.add_argument("--verify-shards", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--run-bins", default=None,
                    help="comma list (default: smoke 0,K-1; measure all; "
                         "auto: fixed)")
    ap.add_argument("--n-per-bin", type=int, default=64)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--params", default="blocks.4,blocks.18,tmlp,last,first")
    ap.add_argument("--cross-params",
                    default="blocks.18.attn.wq,blocks.18.attn.wo,blocks.18.mod,tmlp,last")
    ap.add_argument("--img-tokens", type=int, default=4096)
    ap.add_argument("--text-rows", type=int, default=256)
    ap.add_argument("--attn", choices=["ref", "cudnn"], default="ref")
    ap.add_argument("--qchunk", type=int, default=256)
    ap.add_argument("--remat-candidates", default="dots,full",
                    help="AOT-compiled in order; the first whose peak + the "
                         "other G-1 accumulators + cached inputs fit "
                         "--mem-margin x the device limit is used")
    ap.add_argument("--mem-margin", type=float, default=0.92)
    ap.add_argument("--logit-mean", type=float, default=0.0)
    ap.add_argument("--logit-std", type=float, default=1.0)
    ap.add_argument("--timestep-mu", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--try-cudnn", action=argparse.BooleanOptionalAction,
                    default=True, help="smoke: also time --attn cudnn; auto: "
                    "and use it iff cos >= 0.999 and <= 0.90 x ref s/sample")
    ap.add_argument("--timing-samples", type=int, default=3)
    ap.add_argument("--cudnn-wait-s", type=float, default=300.0,
                    help="max seconds to wait (after the ref timing) for the "
                         "background cuDNN compile; still running -> ref")
    ap.add_argument("--budget-cu", type=float, default=15.0,
                    help="smoke: phase-2 CU budget of the go/no-go rule; "
                         "auto: the CU allowed after the gate")
    ap.add_argument("--overhead-min", type=float, default=None,
                    help="phase-2 fixed minutes in the go/no-go budget "
                         "(default: smoke 25 = a new session's setup, "
                         "downloads, compile, load; auto 5 = continuing: "
                         "memmap re-create, worker drain, reports)")
    ap.add_argument("--deadline-min", type=float, default=0,
                    help="minutes of the script: refuse to load inputs with "
                         "< 5 min left, skip the cuDNN trial and stop before "
                         "any bin (also the first) that would not finish "
                         "(1.25 x N x measured s/sample + overhead); exit 6 "
                         "when bins are left out (0: no limit)")
    ap.add_argument("--bin-overhead-s", type=float, default=60.0,
                    help="fixed seconds per bin in the deadline estimate")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--from-log", default=None, help="report: a job.log")
    ap.add_argument("--stage", choices=["smoke", "measure"], default=None,
                    help="report: the stage of an auto run (default: the last "
                         "GRADSNR_STATE line / out/measure, else out/smoke)")
    ap.add_argument("--wait-inputs", nargs="*", default=[],
                    help="ready markers (content 'ok') to wait for before "
                         "reading the latents and the text encoder")
    ap.add_argument("--wait-raw", default=None,
                    help="ready marker of Raw (downloaded AND sha256 "
                         "verified) to wait for before loading it")
    ap.add_argument("--downloader-pid", type=int, default=None,
                    help="pid of the background downloader: exiting without "
                         "the markers fails the wait at once")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "select":
        return select_samples(args)
    if args.mode == "report":
        if args.from_log:
            print(report_text(build_report(decode_state_from_log(
                args.from_log, args.stage))))
            return None
        d = Path(args.out)
        cands = [d / args.stage] if args.stage else [d, d / "measure", d / "smoke"]
        sp = next((c / "state.json" for c in cands
                   if (c / "state.json").exists()), None)
        if sp is None:
            raise SystemExit(f"no state.json in {[str(c) for c in cands]}")
        rep = build_report(json.loads(sp.read_text()))
        rj = sp.with_name("report.json")
        if rj.exists():  # the gate decision lives only in the report
            go = json.loads(rj.read_text()).get("go")
            if go:
                rep["go"] = go
        print(report_text(rep))
        return None
    for k in ("base_ckpt", "text_encoder", "data_dir", "samples"):
        if getattr(args, k) is None:
            raise SystemExit(f"--{k.replace('_', '-')} is required")
    try:
        run_jax(args)
    except SystemExit as e:
        if any(th.is_alive() for th in _COMPILE_THREADS):
            if e.code is not None and not isinstance(e.code, int):
                print(e.code, file=sys.stderr)
            exit_past_compiles(e.code if isinstance(e.code, int) else 1)
        raise
    except BaseException:
        if any(th.is_alive() for th in _COMPILE_THREADS):
            import traceback
            traceback.print_exc()
            exit_past_compiles(1)
        raise
    exit_past_compiles(0)


if __name__ == "__main__":
    main()
