"""Sampler shortcuts on the tiny DiT: precomputed text tokens, compact_text and
cfg_batch leave the outputs unchanged."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import krea2_jax.sampling as smp
from helpers import TINY
from krea2_jax.checkpoint import abstract_krea2_params
from krea2_jax.config import Krea2Config
from krea2_jax.model import build_positions_and_mask, dit_forward, patchify, text_tokens_forward

CFG = Krea2Config(**TINY)
B, LTXT, LAT = 2, 128, 16
rng = np.random.default_rng(0)
TXT = rng.normal(size=(B, LTXT, 3, 64)).astype(np.float32)
UNTXT = rng.normal(size=(B, LTXT, 3, 64)).astype(np.float32)
MASK = np.zeros((B, LTXT), bool)
MASK[0, :9] = MASK[0, 30:34] = MASK[1, :5] = True
UNMASK = np.arange(LTXT)[None].repeat(B, 0) < 3


@pytest.fixture(scope="module")
def params():
    leaves, tree = jax.tree.flatten(abstract_krea2_params(CFG))
    keys = jax.random.split(jax.random.PRNGKey(1), len(leaves))
    return tree.unflatten([0.02 * jax.random.normal(k, l.shape) for k, l in zip(keys, leaves)])


def rel_err(a, b):
    return float(jnp.abs(a - b).max() / jnp.abs(a).max())


def test_text_tokens_and_compact_match_full_forward(params):
    x = patchify(jnp.asarray(rng.normal(size=(B, 4, LAT, LAT)), jnp.float32), 2)
    t = jnp.full((B,), 0.7)
    pos, mask = build_positions_and_mask(jnp.asarray(MASK), LAT // 2, LAT // 2)
    full = dit_forward(params, CFG, x, jnp.asarray(TXT), t, pos, mask, remat=False)
    tok = text_tokens_forward(params, CFG, jnp.asarray(TXT), jnp.asarray(MASK))
    assert rel_err(full, dit_forward(params, CFG, x, None, t, pos, mask, txt_tokens=tok, remat=False)) < 1e-6
    ctxt, cmask = smp.compact_text(TXT, MASK, multiple=4)
    assert ctxt.shape[1] == 16 and int(cmask.sum()) == int(MASK.sum())
    cpos, cm = build_positions_and_mask(cmask, LAT // 2, LAT // 2)
    assert rel_err(full, dit_forward(params, CFG, x, ctxt, t, cpos, cm, remat=False)) < 1e-5


def sample(params, nb, cfg_batch, attn="ref", un=(UNTXT, UNMASK), mask=MASK, compact=False, steps=3):
    return np.asarray(smp.sample_latents(
        params, CFG, jnp.asarray(TXT[:nb]), jnp.asarray(mask[:nb]), untxt=jnp.asarray(un[0][:nb]),
        untxt_mask=jnp.asarray(un[1][:nb]), height=LAT * 8, width=LAT * 8, steps=steps, guidance=4.5,
        seed=0, compact=compact, attn_impl=attn, cfg_batch=cfg_batch).astype(jnp.float32))


@pytest.mark.parametrize("attn", ["ref", "cudnn"])
@pytest.mark.parametrize("nb", [1, 2])
def test_cfg_batch_matches_two_calls(params, attn, nb):
    two, one = sample(params, nb, False, attn), sample(params, nb, True, attn)
    assert one.shape == (nb, 4, LAT, LAT) and np.abs(one - two).max() < 1e-5


def test_cfg_batch_traces_one_2b_forward(params, monkeypatch):
    seen, orig = [], smp.dit_forward
    monkeypatch.setattr(smp, "dit_forward", lambda *a, **k: (seen.append(a[2].shape[0]), orig(*a, **k))[1])
    for cfg_batch, want in ((True, [4]), (False, [2, 2])):
        smp._euler_step.clear_cache(), seen.clear()
        sample(params, 2, cfg_batch, steps=1)
        assert seen == want
    smp._euler_step.clear_cache()


def test_cfg_batch_fallback_bit_identical(params):
    short = (UNTXT[:, :64], UNMASK[:, :64])
    for nb in (1, 2):
        assert np.array_equal(sample(params, nb, True, un=short), sample(params, nb, False, un=short))
    long_mask = MASK | (np.arange(LTXT) < 70)
    assert smp.compact_text(TXT, long_mask)[0].shape[1] == 128
    assert np.array_equal(sample(params, 2, True, mask=long_mask, compact=True, steps=2),
                          sample(params, 2, False, mask=long_mask, compact=True, steps=2))
