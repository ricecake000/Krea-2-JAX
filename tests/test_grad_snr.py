import numpy as np
import pytest
import grad_snr as gs


@pytest.mark.parametrize("snr", [0.0, 0.02])
def test_bin_stats_planted_mean(snr):
    N, G, d, rng = 64, 8, 4000, np.random.default_rng(1)
    sig = np.linspace(0.5, 2.0, d)
    mu = rng.standard_normal(d) * np.sqrt(snr * np.sum(sig ** 2) / d)
    g = mu + sig * rng.standard_normal((N, d))
    S = np.stack([g[k::G].sum(0) for k in range(G)])
    r = gs.bin_stats(S @ S.T, [np.sum(g[k::G] ** 2) for k in range(G)], [N // G] * G)
    true = mu @ mu / np.sum(sig ** 2)
    assert r["snr"]["lo"] <= true <= r["snr"]["hi"] and r["detectable"] == (snr > 0)


def test_reweight_nonneg_exact():  # clipping the top eigenvector gives [.5, .5, 0] (2.0)
    w, v, _ = gs.reweight_nonneg(np.outer([1, 1, -1.9], [1, 1, -1.9]), np.ones(3))
    assert np.allclose(w, [0, 0, 1]) and np.isclose(v, 3.61)


# sps 4, K = G = 8, 60 s/bin: bin_est(n) = 5n + 84; 15 CU - 5 min -> n_budget
# 127; left 4600 s -> n_time 90; left 2000 s: 6 reuse bins (2424 s) do not fit
@pytest.mark.parametrize("det,n_star,left,verdict,action,n_m,n_fit", [
    (["0"], None, float("inf"), "GO", "reuse", 64, 127),
    ([], 100, float("inf"), "GO", "fresh", 104, 127),   # N* up to G
    ([], 40, float("inf"), "GO", "reuse", 64, 127),     # N_m <= N -> reuse
    ([], 80, 4600, "GO", "fresh", 80, 90),              # time-limited n_fit
    ([], 100, 4600, "NO-GO", "stop", None, 90),         # N* > time-limited n_fit
    ([], 300, float("inf"), "NO-GO", "stop", None, 127),
    (["7"], None, 2000, "GO", "stop", None, 25),        # detectable, no time
])
def test_gate_decision(det, n_star, left, verdict, action, n_m, n_fit):
    r = gs.gate_decision(det, n_star, 64, 8, 8, 4.0, 128, 15.0, 5.0, left, 60.0)
    assert (r["verdict"], r["action"], r["N_m"], r["n_fit"]) == (verdict, action, n_m, n_fit)
