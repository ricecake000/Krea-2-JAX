"""Modal app (CPU only): update-coherence analysis of a full fine-tuning
checkpoint on R2 -- did the weights move along a consistent direction, or
only random-walk? Exact f32 numbers from the train state (params, EMA and
the AdamW moments), not from bf16 exports.

    modal run scripts/modal_coherence.py --run run1 --step 1513
    .venv/bin/python scripts/modal_coherence.py --sim-only   # null model only,
                                                             # local, no Modal
    python scripts/modal_coherence.py --help

Options (local entrypoint, argparse):
  --run NAME        run directory on R2 (default run1)
  --step N          checkpoint step (required unless --sim-only; with
                    --sim-only the simulated T, default 1513)
  --prefix P        R2 prefix of the runs (default krea2-fullft)
  --out PATH        local JSON (default
                    /home/mj/krea2-ops/eval/coherence_<run>_<step>.json)
  --sim-only        only the null / calibration simulation table (local, no
                    Modal, no R2; needs numpy + ml_dtypes, i.e. the repo
                    venv, not the Modal CLI's Python); settings default to
                    run1's (--lr 1e-5 --warmup-steps 150 --beta1 0.9 --beta2
                    0.999 --ema-decay 0.999, T 1513)
  --sim-n N         simulated elements per SNR (default 20000)
  --sim-seed S      simulation RNG seed (default 0)

One function (`coherence_remote`, COH_CPU cores, COH_MEM_GIB GiB) on the
Volume `krea2-eval-cache` of scripts/modal_eval_full.py, with the `krea2-r2`
secret:
  1. download the params, ema and opt_state trees of
     <prefix>/<run>/ckpt/<step>/ into /vol/ckpt/<run>/<tree>/<step>/ (one
     orbax dir per tree, the layout and the ETag/size-checked, resumable
     download of scripts/modal_export_full.py; an already complete tree,
     e.g. the params of an earlier export, downloads nothing).
  2. analysis, one group of leaves at a time (a DiT block, or one stacked
     leaf for a `--layer-loop scan` checkpoint, or one top-level module):
     only that group's leaves of params, ema, mu and nu are read (orbax
     partial restore: every other leaf of the target tree is
     ocp.PLACEHOLDER and is never deserialized), so the peak RAM is ~16
     B/param of the largest group (~6.5 GiB for an unrolled block, ~42 GiB for
     the largest stacked MLP leaf) instead of 4 x 51 GB for the four trees.
     Per tensor (official key, per layer), float64 sums over chunks:
       base = bf16 base as f32, P = params, E = ema, m = mu, v = nu, T = the
       Adam count; mhat = m/(1-b1^T), vhat = v/(1-b2^T),
       u = mhat/(sqrt(vhat)+eps) (current Adam direction; the update is
       -lr*u - lr*wd*P); D = P - base, De = E - base.
       drift ||D||, ||D||/||base||, rms(D) (same for De), ||De||/||D||,
       cos(De, D); coherence cos(-u, D), cos(-m, D), sign agreement
       frac(sign(-u) == sign(D) | D != 0); step-size calibration
       rms(D)/sum(lr_t), rms(D)/sqrt(sum lr_t^2) (lr_t = sched(count) for
       count 0..T-1, the optax semantics), mean|u|, rms(u); bf16 export
       loss: Pb = bf16(P) (round to nearest even), frac of changed elements
       rounded back to the base, ||Pb - base||/||D||, cos(Pb - base, D)
       (same for E).
     Aggregated globally, per tensor kind and per DiT block (the helpers of
     modal_export_full).
  3. the null / calibration simulation with the run's own schedule, betas,
     EMA decay and T (pure numpy, seconds; in the container because the
     Modal CLI's local Python has no numpy): per element g_t = s + xi_t, xi
     ~ N(0, 1), exact optax Adam (bias correction, eps 0, no weight decay),
     EMA from 0, for a grid of SNRs s; the same observables plus the
     coherent fraction ||E[D]||^2 / E||D||^2; the measured global / per-kind
     cos(-u, D) and ||De||/||D|| inverted into an implied s and coherent
     fraction (the real SNR is heterogeneous, so these are approximate).
     Saved to /vol/coherence/<run>_<step>.json, returned, printed as tables
     and written locally (--out).

Secrets: `krea2-r2` (R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY); values (incl. the bucket name) are never printed.

Importable without the Modal SDK; the restore, metrics, simulation and flow
are plain functions (tests/test_modal_coherence.py).
"""

import argparse
import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import modal
except ImportError:  # importable (pyflakes, tests) without the Modal SDK
    modal = None

LOCAL_REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/repo"
# scripts/ locally; /repo/scripts (modal_export_full.py + krea2_jax attached
# by the image) in the container, where this file is mounted on its own
for _repo in (str(LOCAL_REPO), REMOTE_REPO):
    _scripts = os.path.join(_repo, "scripts")
    if (_scripts not in sys.path
            and os.path.isfile(os.path.join(_scripts, "modal_export_full.py"))):
        sys.path.insert(0, _scripts)


def _import_mxf():
    """scripts/modal_export_full.py as a helper library. `modal` is hidden
    while it is imported for the first time, so it does not define its own
    Modal app (krea2-export-full) inside this one."""
    if "modal_export_full" in sys.modules:
        return sys.modules["modal_export_full"]
    missing = object()
    saved = sys.modules.get("modal", missing)
    sys.modules["modal"] = None   # `import modal` raises ImportError
    try:
        import modal_export_full
    finally:
        if saved is missing:
            del sys.modules["modal"]
        else:
            sys.modules["modal"] = saved
    return modal_export_full


mxf = _import_mxf()

from krea2_jax.r2 import r2_client  # noqa: E402 - needs the sys.path above

LOCAL_OUT_ROOT = mxf.LOCAL_OUT_ROOT
R2_PREFIX = mxf.R2_PREFIX
VOLUME_NAME, VOL, BASE_REL = mxf.VOLUME_NAME, mxf.VOL, mxf.BASE_REL
CKPT_TREES = ("params", "ema", "opt_state")

# --- Modal resources ---
COH_CPU = 8
COH_WORKERS = 4           # metric threads (the bf16 casts hold the GIL:
#                           more threads do not scale, measured)
# Peak RAM = 16 B/param (params + ema + mu + nu, f32) of the largest leaf
# group + ~2 GiB (one base tensor, chunk temporaries, runtime): 6.5 + 2 GiB
# for a --layer-loop unroll checkpoint (one block = 434M params), 42 + 2 GiB
# for a --layer-loop scan one (a stacked MLP leaf [28, 16384, 6144] = 2.8B
# params is one group). The partial restore adds no transient copy for
# checkpoints saved FSDP-sharded (orbax chunk = one device's shard; tested).
# 56 GiB covers both layouts with ~25% headroom over the scan case.
COH_MEM_GIB = 56
COH_TIMEOUT_S = 3 * 3600
OVERHEAD_GIB = 2.0        # base tensor + chunk temporaries + runtime
CHUNK = 1 << 20           # elements per float64 chunk (per thread)

# --- simulation ---
SIM_SNRS = (0.0, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0)
SIM_N = 20000
# run1's optimizer settings (scripts/train_full.py defaults + --warmup-steps
# 150), for --sim-only
RUN1_SIM = {"lr": 1e-5, "warmup_steps": 150, "cosine_decay": False,
            "steps": 1513, "min_lr": 0.0, "beta1": 0.9, "beta2": 0.999,
            "ema_decay": 0.999}
RUN1_T = 1513
SIM_NOTE = ("null model: per element g_t = s + N(0, 1) iid, exact Adam (bias "
            "correction, eps 0, weight decay 0, no clipping), EMA from 0; the "
            "real per-element SNR is heterogeneous, so the implied s / "
            "coherent fraction are approximate (an 'effective' SNR)")


# --------------------------------------------------------------- schedule
def lr_at(cfg, count):
    """The LR scripts/train_full.py's make_schedule (optax) gives at update
    count `count` (numpy, float64): linear warmup 0 -> lr over warmup_steps,
    then constant, or cosine to min_lr at `steps` (--cosine-decay).
    cfg: the run config of the checkpoint meta (lr, warmup_steps,
    cosine_decay, steps, min_lr)."""
    import numpy as np

    c = np.asarray(count, np.float64)
    lr, w = float(cfg["lr"]), int(cfg["warmup_steps"])
    if cfg.get("cosine_decay"):
        decay = max(int(cfg["steps"]), w + 1) - w
        alpha = 0.0 if lr == 0.0 else float(cfg.get("min_lr", 0.0)) / lr
        warm = lr * np.clip(c, 0, w) / w if w > 0 else np.full_like(c, 0.0)
        k = np.minimum(c - w, decay)
        cos = 0.5 * (1.0 + np.cos(np.pi * k / decay))
        after = lr * ((1.0 - alpha) * cos + alpha)
    else:
        ts = max(w, 1)
        warm = lr * np.clip(c, 0, ts) / ts
        after = np.full_like(c, lr)
    return np.where(c < w, warm, after)


def lr_schedule(cfg, T):
    """lr_t of updates t = 1..T: sched(count) for count 0..T-1 (optax
    scale_by_learning_rate reads its count before incrementing it)."""
    import numpy as np

    return lr_at(cfg, np.arange(int(T)))


def schedule_sums(cfg, T):
    import numpy as np

    lrs = lr_schedule(cfg, T)
    return {"T": int(T), "lr_sum": float(lrs.sum()),
            "lr_l2": float(np.sqrt(np.dot(lrs, lrs))),
            "lr_max": float(lrs.max()) if lrs.size else 0.0}


# --------------------------------------------------------------- restore
def path_keys(path):
    """jax key path -> tuple of strings (dict keys, list indices)."""
    from krea2_jax.checkpoint import _path_keys

    return tuple(_path_keys(path))


def tree_metadata(item_dir):
    """The on-disk tree of one orbax PyTree item dir (e.g.
    <step>/opt_state): plain dicts / lists with ArrayMetadata leaves
    (namedtuples such as optax's ScaleByAdamState appear as dicts)."""
    import orbax.checkpoint as ocp

    md = ocp.PyTreeCheckpointer().metadata(str(item_dir))
    md = getattr(md, "item_metadata", md)
    return getattr(md, "tree", md)


def flat_metadata(tree):
    """{key tuple: ArrayMetadata} of a metadata tree."""
    import jax

    return {path_keys(p): m
            for p, m in jax.tree_util.tree_flatten_with_path(tree)[0]}


def restore_leaves(item_dir, wanted, meta_tree=None, ckptr=None):
    """Partial restore of the orbax PyTree item at item_dir: {key tuple:
    np.ndarray} of exactly the leaves in `wanted` (key tuples). Every other
    leaf of the target is ocp.PLACEHOLDER, which orbax skips (not read, not
    allocated), so only the wanted leaves cost memory and I/O."""
    import jax
    import numpy as np
    import orbax.checkpoint as ocp

    tree = tree_metadata(item_dir) if meta_tree is None else meta_tree
    wanted = set(wanted)

    def item(p, m):
        if path_keys(p) in wanted:
            return jax.ShapeDtypeStruct(tuple(m.shape), m.dtype)
        return ocp.PLACEHOLDER

    def rargs(p, m):
        if path_keys(p) in wanted:
            return ocp.RestoreArgs(restore_type=np.ndarray)
        return ocp.RestoreArgs()

    ckptr = ckptr or ocp.PyTreeCheckpointer()
    out = ckptr.restore(str(item_dir), args=ocp.args.PyTreeRestore(
        item=jax.tree_util.tree_map_with_path(item, tree),
        restore_args=jax.tree_util.tree_map_with_path(rargs, tree)))
    got = {}
    for p, x in jax.tree_util.tree_flatten_with_path(out)[0]:
        k = path_keys(p)
        if k in wanted:
            if x is ocp.PLACEHOLDER:
                raise RuntimeError(f"{item_dir}: {'.'.join(k)} not restored")
            got[k] = np.asarray(x)
    missing = wanted - set(got)
    if missing:
        raise KeyError(f"{item_dir}: no leaves "
                       f"{sorted('.'.join(k) for k in missing)[:5]}")
    return got


def find_adam_state(opt_tree):
    """Key tuple of the optax ScaleByAdamState in an opt_state metadata tree
    (the dict node with count / mu / nu). Exactly one is expected."""
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            if {"count", "mu", "nu"} <= set(node):
                found.append(path)
            for k, v in node.items():
                walk(v, path + (str(k),))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, path + (str(i),))

    walk(opt_tree, ())
    if len(found) != 1:
        raise ValueError(f"expected one Adam state (count/mu/nu) in "
                         f"opt_state, found {len(found)}: {found}")
    return found[0]


class TrainState:
    """The params / ema / opt_state item dirs of one checkpoint step and
    how their leaves line up: params leaf `k` (key tuple) is ema leaf `k`
    and opt_state leaves adam + ('mu',) + k / adam + ('nu',) + k."""

    def __init__(self, params_dir, ema_dir, opt_dir):
        import orbax.checkpoint as ocp

        self.dirs = {"params": Path(params_dir), "ema": Path(ema_dir),
                     "opt_state": Path(opt_dir)}
        self.trees = {k: tree_metadata(d) for k, d in self.dirs.items()}
        self.params_tree = self.trees["params"]
        self.leaves = flat_metadata(self.params_tree)   # params key -> meta
        self.adam = find_adam_state(self.trees["opt_state"])
        opt_flat = flat_metadata(self.trees["opt_state"])
        for name, flat in (("ema", flat_metadata(self.trees["ema"])),
                           ("mu", None), ("nu", None)):
            if flat is None:
                pre = self.adam + (name,)
                flat = {k[len(pre):]: m for k, m in opt_flat.items()
                        if k[:len(pre)] == pre}
            if set(flat) != set(self.leaves):
                raise ValueError(f"{name} leaves differ from params: "
                                 f"{sorted(set(flat) ^ set(self.leaves))[:5]}")
            bad = [k for k in flat
                   if tuple(flat[k].shape) != tuple(self.leaves[k].shape)]
            if bad:
                raise ValueError(f"{name} shapes differ from params: {bad[:5]}")
        self.scalars = [k for k, m in opt_flat.items() if not tuple(m.shape)]
        self.ckptr = ocp.PyTreeCheckpointer()

    def opt_scalars(self):
        """{dotted key: int} of the opt_state scalars (step counters)."""
        got = restore_leaves(self.dirs["opt_state"], self.scalars,
                             self.trees["opt_state"], self.ckptr)
        return {".".join(k): int(v) for k, v in got.items()}

    def adam_count(self):
        k = self.adam + ("count",)
        got = restore_leaves(self.dirs["opt_state"], [k],
                             self.trees["opt_state"], self.ckptr)
        return int(got[k])

    def restore_group(self, keys):
        """{"params", "ema", "mu", "nu"}: {param key: np.ndarray} of the
        param leaves `keys` only."""
        keys = list(keys)
        out = {}
        # one restore call per tree: a call's transient read buffers (a
        # whole decoded chunk per leaf in flight) then never span two trees
        for name in ("params", "ema"):
            out[name] = restore_leaves(self.dirs[name], keys,
                                       self.trees[name], self.ckptr)
        for n in ("mu", "nu"):
            pre = self.adam + (n,)
            got = restore_leaves(self.dirs["opt_state"],
                                 [pre + k for k in keys],
                                 self.trees["opt_state"], self.ckptr)
            out[n] = {k[len(pre):]: v for k, v in got.items()}
        return out


def plan_groups(leaves):
    """Leaf groups restored together: {group name: [param key tuples]}.
    Per-layer lists (--layer-loop unroll: blocks.<i>.*) -> one group per
    block; a stacked leaf under `blocks` (--layer-loop scan) -> a group of
    its own; everything else -> one group per top-level module. Every leaf
    is in exactly one group."""
    groups = {}
    for k in sorted(leaves):
        if k[0] == "blocks" and len(k) > 1 and k[1].isdigit():
            g = f"blocks.{int(k[1]):03d}"
        elif k[0] == "blocks":
            g = "blocks:" + ".".join(k[1:])
        else:
            g = k[0]
        groups.setdefault(g, []).append(k)
    return groups


def group_bytes(leaves, keys):
    """Host bytes of one group: params + ema + mu + nu, float32."""
    import numpy as np

    return 16 * sum(int(np.prod(leaves[k].shape)) for k in keys)


# --------------------------------------------------------------- metrics
SUM_KEYS = ("n", "dd", "bb", "ee", "dde", "uu", "ud", "mm", "md", "au",
            "nz", "agree", "nz_e", "pbd2", "pbd", "lost_p", "chg_p",
            "ebd2", "ebde", "lost_e", "chg_e")


def chunk_sums(p, e, m, v, b, c1, c2, eps):
    """Float64 sums of one chunk (any float dtype in; b = the base as
    float32 values of its bf16). c1 = 1/(1-b1^T), c2 = 1/(1-b2^T)."""
    import ml_dtypes
    import numpy as np

    f8 = np.float64
    b64 = b.astype(f8)
    p64, e64 = p.astype(f8), e.astype(f8)
    d = p64 - b64
    de = e64 - b64
    m64 = m.astype(f8)
    u = (m64 * c1) / (np.sqrt(v.astype(f8) * c2) + eps)
    nzm = d != 0
    bf16 = ml_dtypes.bfloat16   # float32 -> bf16: round to nearest even
    pb = p.astype(np.float32, copy=False).astype(bf16).astype(f8) - b64
    eb = e.astype(np.float32, copy=False).astype(bf16).astype(f8) - b64
    return {
        "n": int(d.size), "dd": float(d @ d), "bb": float(b64 @ b64),
        "ee": float(de @ de), "dde": float(d @ de), "uu": float(u @ u),
        "ud": -float(u @ d), "mm": float(m64 @ m64), "md": -float(m64 @ d),
        "au": float(np.abs(u).sum()), "nz": int(nzm.sum()),
        "agree": int(np.count_nonzero(nzm & (np.sign(-u) == np.sign(d)))),
        "nz_e": int(np.count_nonzero(de)),
        "pbd2": float(pb @ pb), "pbd": float(pb @ d),
        "lost_p": int(np.count_nonzero(nzm & (pb == 0))),
        "chg_p": int(np.count_nonzero(pb)),
        "ebd2": float(eb @ eb), "ebde": float(eb @ de),
        "lost_e": int(np.count_nonzero((de != 0) & (eb == 0))),
        "chg_e": int(np.count_nonzero(eb)),
    }


def add_sums(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in SUM_KEYS}


def tensor_sums(p, e, m, v, b, c1, c2, eps, pool=None, chunk=CHUNK):
    """chunk_sums over a whole tensor (flattened, CHUNK elements per task,
    in `pool` when given)."""
    arrs = [x.reshape(-1) for x in (p, e, m, v, b)]
    n = arrs[0].size
    if any(x.size != n for x in arrs):
        raise ValueError(f"size mismatch {[x.size for x in arrs]}")

    def one(i):
        return chunk_sums(*(x[i:i + chunk] for x in arrs), c1, c2, eps)

    starts = range(0, n, chunk)
    parts = pool.map(one, starts) if pool is not None else map(one, starts)
    total = {k: 0 for k in SUM_KEYS}
    for s in parts:
        total = add_sums(total, s)
    return total


def _div(a, b):
    return a / b if b else None


def _sqrt(x):
    return None if x is None else math.sqrt(max(x, 0.0))


def _cos(xy, xx, yy):
    return xy / math.sqrt(xx * yy) if xx > 0 and yy > 0 else None


def derive(s, sched):
    """Sums -> the reported observables (None where undefined)."""
    n = s["n"]
    rms = _sqrt(_div(s["dd"], n))
    return {
        "n": n,
        "drift_norm": math.sqrt(s["dd"]),
        "drift_rel": _sqrt(_div(s["dd"], s["bb"])),
        "drift_rms": rms,
        "ema_drift_norm": math.sqrt(s["ee"]),
        "ema_drift_rel": _sqrt(_div(s["ee"], s["bb"])),
        "ema_drift_rms": _sqrt(_div(s["ee"], n)),
        "ema_ratio": _sqrt(_div(s["ee"], s["dd"])),
        "cos_ema": _cos(s["dde"], s["dd"], s["ee"]),
        "cos_u": _cos(s["ud"], s["uu"], s["dd"]),
        "cos_m": _cos(s["md"], s["mm"], s["dd"]),
        "sign_agree": _div(s["agree"], s["nz"]),
        "changed_frac": _div(s["nz"], n),
        "rms_over_lrsum": (None if rms is None
                           else _div(rms, sched["lr_sum"])),
        "rms_over_lrl2": (None if rms is None
                          else _div(rms, sched["lr_l2"])),
        "mean_abs_u": _div(s["au"], n),
        "rms_u": _sqrt(_div(s["uu"], n)),
        "bf16_params": {
            "lost_frac": _div(s["lost_p"], s["nz"]),
            "norm_ratio": _sqrt(_div(s["pbd2"], s["dd"])),
            "cos": _cos(s["pbd"], s["pbd2"], s["dd"]),
            "changed_frac": _div(s["chg_p"], n)},
        "bf16_ema": {
            "lost_frac": _div(s["lost_e"], s["nz_e"]),
            "norm_ratio": _sqrt(_div(s["ebd2"], s["ee"])),
            "cos": _cos(s["ebde"], s["ebd2"], s["ee"]),
            "changed_frac": _div(s["chg_e"], n)},
    }


def aggregate(tensors, sched):
    """{"global", "by_kind", "by_block"} of derived observables from the
    per-tensor sums ({official key: sums})."""
    total = {k: 0 for k in SUM_KEYS}
    kinds, blocks = {}, {}
    for key, s in tensors.items():
        total = add_sums(total, s)
        kk, bk = mxf.tensor_kind(key), mxf.tensor_group(key)
        kinds[kk] = add_sums(kinds.get(kk, {}), s)
        blocks[bk] = add_sums(blocks.get(bk, {}), s)
    return {
        "global": derive(total, sched),
        "by_kind": {k: derive(kinds[k], sched) for k in mxf.KINDS
                    if k in kinds},
        "by_block": {b: derive(blocks[b], sched) for b in sorted(blocks)},
    }


def analyze(state, base_path, cfg, T, workers=COH_WORKERS,
            mem_budget_gib=None, log=print):
    """The measured part of the report (see the module docstring).
    state: TrainState; cfg: the checkpoint's run config; T: the Adam count.
    mem_budget_gib: refuse (before reading anything) when the largest group
    would not fit. -> {"schedule", "groups", "tensors", "global", "by_kind",
    "by_block", ...}."""
    import ml_dtypes  # noqa: F401 - numpy 'bfloat16' for safe_open
    from safetensors import safe_open

    from krea2_jax.checkpoint import _official_entries

    t0 = time.time()
    b1, b2, eps = (float(cfg["beta1"]), float(cfg["beta2"]),
                   float(cfg["eps"]))
    c1, c2 = 1.0 / (1.0 - b1 ** T), 1.0 / (1.0 - b2 ** T)
    sched = schedule_sums(cfg, T)
    leaves = state.leaves
    groups = plan_groups(leaves)
    sizes = {g: group_bytes(leaves, ks) for g, ks in groups.items()}
    big = max(sizes, key=sizes.get)
    est = sizes[big] / 2**30 + OVERHEAD_GIB
    log(f"{len(leaves)} param leaves in {len(groups)} groups; largest "
        f"{big} {sizes[big] / 2**30:.2f} GiB (x4 trees) -> peak ~{est:.1f} "
        "GiB")
    if mem_budget_gib is not None and est > 0.9 * mem_budget_gib:
        raise RuntimeError(f"largest leaf group {big} needs ~{est:.1f} GiB, "
                           f"over 90% of the {mem_budget_gib} GiB budget")
    # official keys of every leaf (stacked leaves: one per layer)
    entries, _ = _official_entries(state.params_tree)
    flat_keys = list(flat_metadata(state.params_tree))
    by_leaf = {}
    for name, li, layer, shape in entries:
        by_leaf.setdefault(flat_keys[li], []).append((name, layer, shape))
    tensors = {}
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        with safe_open(str(base_path), framework="numpy") as base:
            base_keys = set(base.keys())
            want = {n for es in by_leaf.values() for n, _, _ in es}
            if want != base_keys:
                raise ValueError(
                    f"checkpoint vs base keys: missing in base "
                    f"{sorted(want - base_keys)[:5]}, not in checkpoint "
                    f"{sorted(base_keys - want)[:5]}")
            for gi, (g, keys) in enumerate(groups.items()):
                tg = time.time()
                # every array of the group lives only inside _group_sums:
                # none survives into the next group's restore
                tr = _group_sums(state, keys, by_leaf, base, tensors,
                                 (c1, c2, eps), pool)
                log(f"group {gi + 1}/{len(groups)} {g}: "
                    f"{sizes[g] / 2**30:.2f} GiB, restore {tr:.0f} s, "
                    f"metrics {time.time() - tg - tr:.0f} s "
                    f"({time.time() - t0:.0f} s total, peak RSS "
                    f"{peak_rss_gib():.1f} GiB)")
    finally:
        pool.shutdown(wait=True)
    return {
        "schedule": sched,
        "adam": {"T": int(T), "beta1": b1, "beta2": b2, "eps": eps,
                 "weight_decay": cfg.get("weight_decay"),
                 "bias_correction": [1.0 / c1, 1.0 / c2]},
        "groups": {g: {"leaves": len(groups[g]),
                       "gib": sizes[g] / 2**30} for g in groups},
        "tensors": tensors,
        **aggregate(tensors, sched),
        "seconds": time.time() - t0,
        "peak_rss_gib": peak_rss_gib(),
    }


def _group_sums(state, keys, by_leaf, base, tensors, consts, pool):
    """Restore one leaf group (params / ema / mu / nu) and add the sums of
    each of its official tensors to `tensors`. All references to the
    group's arrays are locals of this function, so they are released when
    it returns -- before the caller restores the next group (two groups of
    a scan checkpoint, 42 GiB each, must never be alive together).
    -> seconds spent restoring."""
    import numpy as np

    t0 = time.time()
    arrs = state.restore_group(keys)
    tr = time.time() - t0
    for k in keys:
        trees = [arrs[t][k] for t in ("params", "ema", "mu", "nu")]
        for name, layer, shape in by_leaf[k]:
            b = base.get_tensor(name)
            if tuple(b.shape) != tuple(shape):
                raise ValueError(f"{name}: base {b.shape}, checkpoint "
                                 f"{shape}")
            views = trees if layer is None else [x[layer] for x in trees]
            tensors[name] = tensor_sums(*views, b.astype(np.float32),
                                        *consts, pool=pool)
    return tr


def peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


# --------------------------------------------------------------- simulation
def simulate(snrs=SIM_SNRS, lrs=None, b1=0.9, b2=0.999, eps=0.0,
             ema_decay=0.999, n=SIM_N, seed=0):
    """Null / calibration model: n elements per SNR s, g_t = s + xi_t (xi ~
    N(0, 1)), exact optax Adam with bias correction (eps relative to sigma
    = 1; weight decay 0), lr_t = lrs[t-1], EMA of the params from 0 (the
    base) with ema_decay after every update. The observables are computed
    with the same code as the measurement (chunk_sums / derive), plus the
    coherent fraction ||E[D]||^2 / E||D||^2 (E[D] = the mean over the
    elements of one s). -> list of rows, one per s."""
    import numpy as np

    lrs = np.asarray(lrs, np.float64)
    T = lrs.size
    snrs = [float(s) for s in snrs]
    mu = np.repeat(np.asarray(snrs, np.float64), n)
    rng = np.random.default_rng(seed)
    p, e = np.zeros_like(mu), np.zeros_like(mu)
    m, v = np.zeros_like(mu), np.zeros_like(mu)
    for t in range(1, T + 1):
        g = mu + rng.standard_normal(mu.size)
        m = (1.0 - b1) * g + b1 * m
        v = (1.0 - b2) * (g * g) + b2 * v
        u = (m / (1.0 - b1 ** t)) / (np.sqrt(v / (1.0 - b2 ** t)) + eps)
        p -= lrs[t - 1] * u
        e += (1.0 - ema_decay) * (p - e)
    sched = {"T": T, "lr_sum": float(lrs.sum()),
             "lr_l2": float(np.sqrt(lrs @ lrs))}
    c1, c2 = 1.0 / (1.0 - b1 ** T), 1.0 / (1.0 - b2 ** T)
    rows = []
    for i, s in enumerate(snrs):
        sl = slice(i * n, (i + 1) * n)
        zero = np.zeros(n, np.float32)
        d = derive(chunk_sums(p[sl], e[sl], m[sl], v[sl], zero, c1, c2, eps),
                   sched)
        mean_d = float(p[sl].mean())
        rows.append({
            "s": s,
            **{k: d[k] for k in ("rms_over_lrsum", "rms_over_lrl2", "cos_u",
                                 "cos_m", "sign_agree", "ema_ratio",
                                 "cos_ema", "mean_abs_u", "rms_u")},
            "coherent_frac": mean_d * mean_d / float(np.mean(p[sl] ** 2)),
        })
    return rows


def null_cos_u(lrs, b1=0.9):
    """Analytic s = 0 limit of cos(-u, D): with vhat ~ 1, u_T ~ m_T and D ~
    -sum lr_t m_t, cos ~ 1/sqrt((1-b1)(1+b1) T_eff) (10/sqrt(19 T_eff) for
    b1 = 0.9), T_eff = sum lr_t^2 / lr_T^2."""
    import numpy as np

    lrs = np.asarray(lrs, np.float64)
    t_eff = float(lrs @ lrs) / float(lrs[-1]) ** 2
    return 1.0 / math.sqrt((1.0 - b1) * (1.0 + b1) * t_eff)


def invert(rows, metric, x):
    """The s (and coherent fraction) at which the simulated `metric`
    equals x, by linear interpolation on the grid segments that bracket x.
      one solution        -> range "inside", s / coherent_frac
      several solutions   -> range "ambiguous" (a non-monotone curve, e.g.
                             ||De||/||D|| near saturation, or a flat
                             segment equal to x): s / coherent_frac None,
                             "candidates" [{s, coherent_frac}] and their
                             s_min / s_max / coherent_frac_min / _max
      outside the curve   -> range "below" / "above", clamped to the
                             nearest end."""
    if x is None:
        return None
    pts = [(r["s"], r[metric], r["coherent_frac"]) for r in rows
           if r[metric] is not None]
    cands = []
    for (s0, y0, f0), (s1, y1, f1) in zip(pts, pts[1:]):
        if min(y0, y1) <= x <= max(y0, y1):
            if y1 == y0:   # flat at x: every s of the segment fits
                cands += [(s0, f0), (s1, f1)]
            else:
                w = (x - y0) / (y1 - y0)
                cands.append((s0 + w * (s1 - s0), f0 + w * (f1 - f0)))
    uniq = []   # x on a grid point is bracketed by both its segments
    for s, f in sorted(cands):
        if not uniq or abs(s - uniq[-1][0]) > 1e-12 * max(1.0, abs(s)):
            uniq.append((s, f))
    out = {"metric": metric, "measured": x}
    if len(uniq) == 1:
        return {**out, "s": uniq[0][0], "coherent_frac": uniq[0][1],
                "range": "inside"}
    if uniq:
        fs = [f for _, f in uniq]
        return {**out, "s": None, "coherent_frac": None,
                "range": "ambiguous",
                "candidates": [{"s": s, "coherent_frac": f}
                               for s, f in uniq],
                "s_min": uniq[0][0], "s_max": uniq[-1][0],
                "coherent_frac_min": min(fs), "coherent_frac_max": max(fs)}
    ys = [y for _, y, _ in pts]
    lo, hi = min(range(len(ys)), key=ys.__getitem__), max(
        range(len(ys)), key=ys.__getitem__)
    j = lo if x < ys[lo] else hi
    return {"metric": metric, "measured": x, "s": pts[j][0],
            "coherent_frac": pts[j][2],
            "range": "below" if x < ys[lo] else "above"}


def implied(report, rows):
    """Measured global / per-kind cos(-u, D) and ||De||/||D|| -> implied
    SNR and coherent fraction (via the simulation)."""
    out = {"note": SIM_NOTE}
    scopes = {"global": report["global"], **report["by_kind"]}
    for name, obs in scopes.items():
        out[name] = {m: invert(rows, m, obs.get(m))
                     for m in ("cos_u", "ema_ratio")}
    return out


def sim_config(cfg, T, n=SIM_N, seed=0):
    """Simulation settings from a run config + T."""
    return {"lr": float(cfg["lr"]), "warmup_steps": int(cfg["warmup_steps"]),
            "cosine_decay": bool(cfg.get("cosine_decay", False)),
            "steps": int(cfg.get("steps", T)),
            "min_lr": float(cfg.get("min_lr", 0.0)),
            "beta1": float(cfg["beta1"]), "beta2": float(cfg["beta2"]),
            "ema_decay": float(cfg["ema_decay"]), "T": int(T), "n": int(n),
            "seed": int(seed), "eps": 0.0, "weight_decay": 0.0}


def run_sim(sc, snrs=SIM_SNRS):
    """simulate() with sim_config settings -> {"config", "rows",
    "null_cos_u_analytic", "note"}."""
    lrs = lr_schedule(sc, sc["T"])
    rows = simulate(snrs, lrs, b1=sc["beta1"], b2=sc["beta2"], eps=0.0,
                    ema_decay=sc["ema_decay"], n=sc["n"], seed=sc["seed"])
    return {"config": sc, "rows": rows,
            "null_cos_u_analytic": null_cos_u(lrs, sc["beta1"]),
            "note": SIM_NOTE}


# --------------------------------------------------------------- tables
def _f(x, w=10, spec=".3e"):
    return f"{'-':>{w}}" if x is None else f"{x:>{w}{spec}}"


def sim_table(sim):
    sc = sim["config"]
    lines = [f"null / calibration simulation (T {sc['T']}, lr {sc['lr']:g}, "
             f"warmup {sc['warmup_steps']}, b1 {sc['beta1']}, b2 "
             f"{sc['beta2']}, EMA {sc['ema_decay']}, eps 0, wd 0, "
             f"{sc['n']} elements per s, seed {sc['seed']})",
             f"{'s':>7}{'rms/lrsum':>11}{'rms/lrl2':>10}{'cos(-u,D)':>11}"
             f"{'sign':>8}{'|De|/|D|':>10}{'cos(De,D)':>11}{'coh.frac':>10}"]
    for r in sim["rows"]:
        lines.append(f"{r['s']:>7.3f}{_f(r['rms_over_lrsum'], 11, '.4f')}"
                     f"{_f(r['rms_over_lrl2'], 10, '.3f')}"
                     f"{_f(r['cos_u'], 11, '.4f')}"
                     f"{_f(r['sign_agree'], 8, '.4f')}"
                     f"{_f(r['ema_ratio'], 10, '.4f')}"
                     f"{_f(r['cos_ema'], 11, '.4f')}"
                     f"{_f(r['coherent_frac'], 10, '.4f')}")
    lines.append(f"analytic s=0 cos(-u,D) ~ 1/sqrt((1-b1)(1+b1) T_eff) = "
                 f"{sim['null_cos_u_analytic']:.4f}")
    lines.append(sim["note"])
    return "\n".join(lines)


def report_table(rep):
    cols = (("drift_rel", "|D|/|W|", ".3e"), ("drift_rms", "rms(D)", ".3e"),
            ("ema_ratio", "|De|/|D|", ".4f"), ("cos_ema", "cos(De,D)", ".4f"),
            ("cos_u", "cos(-u,D)", ".4f"), ("cos_m", "cos(-m,D)", ".4f"),
            ("sign_agree", "sign", ".4f"),
            ("rms_over_lrsum", "rms/lrsum", ".4f"),
            ("rms_over_lrl2", "rms/lrl2", ".3f"), ("rms_u", "rms(u)", ".3f"))
    bcols = (("bf16_params", "lost_frac", "P lost", ".4f"),
             ("bf16_params", "norm_ratio", "P |r|", ".3f"),
             ("bf16_params", "cos", "P cos", ".4f"),
             ("bf16_ema", "lost_frac", "E lost", ".4f"),
             ("bf16_ema", "norm_ratio", "E |r|", ".3f"),
             ("bf16_ema", "cos", "E cos", ".4f"))
    s = rep["schedule"]
    lines = [f"update coherence of {rep.get('run')}/{rep.get('step')} "
             f"(Adam T {rep['adam']['T']}, lr_sum {s['lr_sum']:.4e}, "
             f"sqrt(sum lr^2) {s['lr_l2']:.4e}); D = P - base, De = E - "
             "base, u = Adam direction",
             f"{'':<12}" + "".join(f"{h:>11}" for _, h, _ in cols)]
    scopes = [("global", rep["global"])] + list(rep["by_kind"].items())
    for name, o in scopes:
        lines.append(f"{name:<12}" + "".join(_f(o[k], 11, sp)
                                             for k, _, sp in cols))
    lines.append("bf16 export loss (lost = changed elements rounded back to "
                 "the base; |r| = ||bf16 - base|| / ||D||; cos vs D):")
    lines.append(f"{'':<12}" + "".join(f"{h:>9}" for _, _, h, _ in bcols))
    for name, o in scopes:
        lines.append(f"{name:<12}" + "".join(_f(o[a][k], 9, sp)
                                             for a, k, _, sp in bcols))
    lines.append("by block: cos(-u,D) / sign / |De|/|D| / |D|/|W|")
    for b, o in rep["by_block"].items():
        lines.append(f"  {b:<10}{_f(o['cos_u'], 9, '.4f')}"
                     f"{_f(o['sign_agree'], 9, '.4f')}"
                     f"{_f(o['ema_ratio'], 9, '.4f')}"
                     f"{_f(o['drift_rel'], 11, '.3e')}")
    return "\n".join(lines)


def implied_table(imp):
    lines = ["implied effective SNR / coherent fraction of ||D||^2 (from the "
             "simulation):",
             f"{'':<12}{'cos(-u,D)':>10}{'s':>15}{'coh':>15}"
             f"{'|De|/|D|':>10}{'s':>15}{'coh':>15}"]
    for name, v in imp.items():
        if name == "note":
            continue
        row = f"{name:<12}"
        for m in ("cos_u", "ema_ratio"):
            r = v.get(m)
            if r is None:
                row += f"{'-':>10}{'-':>15}{'-':>15}"
                continue
            if r["range"] == "ambiguous":
                s = f"?{r['s_min']:.4f}-{r['s_max']:.4f}"
                f = (f"{r['coherent_frac_min']:.4f}-"
                     f"{r['coherent_frac_max']:.4f}")
            else:
                flag = {"inside": "", "below": "<", "above": ">"}[r["range"]]
                s, f = flag + format(r["s"], ".4f"), format(
                    r["coherent_frac"], ".4f")
            row += f"{r['measured']:>10.4f}{s:>15}{f:>15}"
        lines.append(row)
    lines.append("(< / >: measured value outside the simulated range, "
                 "clamped; ?a-b: ambiguous, the simulated curve reaches it "
                 "at several s) " + imp["note"])
    return "\n".join(lines)


# --------------------------------------------------------------- flow
def coherence_paths(vol, run, step):
    """{tree: modal_export_full.local_paths(...)} + the report path."""
    paths = {t: mxf.local_paths(vol, run, step, t) for t in CKPT_TREES}
    paths["report"] = Path(vol) / "coherence" / f"{run}_{int(step)}.json"
    return paths


def coherence_impl(run, step, prefix=R2_PREFIX, env=None, vol=VOL,
                   commit=lambda: None, client=None, transfer_config=None,
                   workers=COH_WORKERS, mem_budget_gib=None, sim_n=SIM_N,
                   sim_seed=0, log=print):
    """The remote flow: download params / ema / opt_state of the step (via
    modal_export_full's download, idempotent), analyze(), then
    add_simulation(). -> the report dict (also written to
    /vol/coherence/<run>_<step>.json)."""
    from krea2_jax import fullft_ckpt

    env = os.environ if env is None else env
    paths = coherence_paths(vol, run, step)
    base_path = Path(vol) / BASE_REL
    if not base_path.is_file():
        raise RuntimeError(f"{base_path} missing: run `modal run "
                           "scripts/modal_eval_full.py --limit 1` (its "
                           "prepare caches the HF base) first")
    if client is None:
        client = r2_client(env)
    bucket = env["R2_BUCKET"]
    sp = mxf.step_prefix(prefix, run, step)
    objects = mxf.list_objects(client, bucket, sp)
    sels, downloads = {}, {}
    for tree in CKPT_TREES:   # every selection first: fail before a transfer
        sels[tree] = mxf.select_ckpt_objects(objects, sp, tree,
                                             trees=CKPT_TREES)
    for tree in CKPT_TREES:
        downloads[tree] = mxf.download_ckpt_tree(
            client, bucket, sp, paths[tree]["step_dir"], tree,
            paths[tree]["marker"], transfer_config=transfer_config, log=log,
            sel=sels[tree])
        commit()
    merged = {}
    for sel in sels.values():
        merged.update(sel)
    meta = fullft_ckpt.read_meta_at(str(paths["params"]["ckpt_root"]), step)
    cfg = meta.get("config")
    if not cfg:
        raise RuntimeError(f"checkpoint {run}/{step} has no run config in "
                           "its meta (lr / betas / schedule unknown)")
    if not meta.get("has_ema"):
        raise RuntimeError(f"checkpoint {run}/{step} was saved without EMA")
    state = TrainState(*(paths[t]["step_dir"] / t for t in CKPT_TREES))
    T = state.adam_count()
    counts = state.opt_scalars()
    log(f"meta step {meta['step']}, Adam count {T}, opt_state counters "
        f"{counts}, layer_loop {cfg.get('layer_loop', 'scan')}")
    rep = analyze(state, base_path, cfg, T, workers=workers,
                  mem_budget_gib=mem_budget_gib, log=log)
    rep = {"run": run, "step": int(step), "meta_step": int(meta["step"]),
           "prefix": prefix, "ckpt_fingerprint": mxf.ckpt_fingerprint(merged),
           "config": cfg, "opt_counters": counts, "downloads": downloads,
           "base": BASE_REL, **rep}
    if T != int(meta["step"]):
        rep["warning"] = (f"Adam count {T} != meta step {meta['step']}: "
                          "bias correction and schedule use the count")
    log(report_table(rep))
    t0 = time.time()
    rep = add_simulation(rep, sim_n, sim_seed)
    log(f"simulation: {time.time() - t0:.0f} s")
    mxf._write_json(paths["report"], rep)
    commit()
    return rep


def add_simulation(rep, sim_n=SIM_N, sim_seed=0):
    """The null / calibration simulation with the run's own settings (its
    config + Adam count) and the implied values, added to a report."""
    sim = run_sim(sim_config(rep["config"], rep["adam"]["T"], sim_n,
                             sim_seed))
    rep = dict(rep)
    rep["sim"] = sim
    rep["implied"] = implied(rep, sim["rows"])
    return rep


# --------------------------------------------------------------- CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(
        prog="modal run scripts/modal_coherence.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="run1",
                    help="run directory under --prefix on R2")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (required unless --sim-only; with "
                         "--sim-only: the simulated T, default 1513)")
    ap.add_argument("--prefix", default=R2_PREFIX,
                    help="R2 prefix of the runs")
    ap.add_argument("--out", default="",
                    help="local JSON (default "
                         f"{LOCAL_OUT_ROOT}/coherence_<run>_<step>.json)")
    ap.add_argument("--sim-only", action="store_true",
                    help="print only the simulation table (no Modal / R2)")
    ap.add_argument("--sim-n", type=int, default=SIM_N,
                    help="simulated elements per SNR")
    ap.add_argument("--sim-seed", type=int, default=0)
    for k in ("lr", "warmup_steps", "beta1", "beta2", "ema_decay"):
        ap.add_argument("--" + k.replace("_", "-"), type=type(RUN1_SIM[k]),
                        default=RUN1_SIM[k],
                        help=f"--sim-only setting (default run1: "
                             f"{RUN1_SIM[k]})")
    return ap


def parse_args(argv):
    ap = build_arg_parser()
    args = ap.parse_args(list(argv))
    if not mxf.NAME_RE.match(args.run):
        ap.error(f"--run must match {mxf.NAME_RE.pattern}, got {args.run!r}")
    if not all(mxf.NAME_RE.match(p) for p in args.prefix.split("/")):
        ap.error(f"--prefix must be '/'-separated names, got {args.prefix!r}")
    if args.step is None and not args.sim_only:
        ap.error("--step is required (unless --sim-only)")
    if args.step is not None and args.step < (1 if args.sim_only else 0):
        ap.error(f"--step must be >= {1 if args.sim_only else 0}, got "
                 f"{args.step}")
    if args.sim_n < 1:
        ap.error("--sim-n must be >= 1")
    return args


def sim_only(args):
    cfg = {**RUN1_SIM, "lr": args.lr, "warmup_steps": args.warmup_steps,
           "beta1": args.beta1, "beta2": args.beta2,
           "ema_decay": args.ema_decay}
    T = args.step if args.step is not None else RUN1_T
    cfg["steps"] = max(int(cfg["steps"]), T)
    t0 = time.time()
    sim = run_sim(sim_config(cfg, T, args.sim_n, args.sim_seed))
    print(sim_table(sim))
    print(f"({time.time() - t0:.1f} s)")
    return sim


def write_local(rep, out=""):
    path = Path(out) if out else (
        LOCAL_OUT_ROOT / f"coherence_{rep['run']}_{rep['step']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1))
    return path


# ------------------------------------------------------------------ Modal app
if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*mxf.EXPORT_PIP)
        .env(mxf.EXPORT_ENV)
        .add_local_dir(LOCAL_REPO / "krea2_jax", f"{REMOTE_REPO}/krea2_jax",
                       ignore=["**/__pycache__"])
        .add_local_file(LOCAL_REPO / "scripts" / "modal_export_full.py",
                        f"{REMOTE_REPO}/scripts/modal_export_full.py")
    )
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    r2_secret = modal.Secret.from_name("krea2-r2")
    app = modal.App("krea2-coherence")

    @app.function(image=image, cpu=float(COH_CPU),
                  memory=COH_MEM_GIB * 1024, timeout=COH_TIMEOUT_S,
                  volumes={VOL: vol}, secrets=[r2_secret])
    def coherence_remote(run: str, step: int, prefix: str, sim_n: int,
                         sim_seed: int) -> dict:
        return coherence_impl(run, step, prefix=prefix, commit=vol.commit,
                              workers=COH_WORKERS, mem_budget_gib=COH_MEM_GIB,
                              sim_n=sim_n, sim_seed=sim_seed,
                              log=lambda m: print(m, flush=True))

    @app.local_entrypoint()
    def main(*argv: str):
        # runs in the Modal CLI's Python (no numpy there): only printing
        args = parse_args(argv)
        if args.sim_only:
            import importlib.util

            if not all(importlib.util.find_spec(m)
                       for m in ("numpy", "ml_dtypes")):
                raise SystemExit(
                    "--sim-only needs numpy + ml_dtypes in the local Python; "
                    "run it with the repo venv instead: .venv/bin/python "
                    "scripts/modal_coherence.py --sim-only")
            sim_only(args)
            return
        t0 = time.time()
        rep = coherence_remote.remote(args.run, args.step, args.prefix,
                                      args.sim_n, args.sim_seed)
        print(f"remote: {(time.time() - t0) / 60:.1f} min")
        print(report_table(rep))
        print()
        print(sim_table(rep["sim"]))
        print()
        print(implied_table(rep["implied"]))
        print(f"\nwrote {write_local(rep, args.out)}")


def local_main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.sim_only:
        return sim_only(args)
    raise SystemExit(f"this script runs on Modal: modal run "
                     f"scripts/modal_coherence.py --run {args.run} "
                     f"--step {args.step}")


if __name__ == "__main__":
    local_main()
