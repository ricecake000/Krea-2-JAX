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
