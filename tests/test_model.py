"""dit_forward variants (attention impls, unrolled layer loop) vs the ref path, tiny model,
FSDP params on 8 emulated devices; full-size abstract params vs the official module."""

import functools
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import TINY
from krea2_jax.checkpoint import (
    abstract_krea2_params, batch_sharding, load_krea2_params, make_mesh,
    official_key_shapes, param_shardings, save_krea2_params, stack_blocks)
from krea2_jax.config import KREA2_CONFIG, Krea2Config
from krea2_jax.model import (
    build_positions_and_mask_bucketed, dit_forward, scaled_dot_product_attention,
    varlen_reference_kernel)

B, LTXT, NIMG = 8, 6, 64
HW = np.array([(8, 8), (6, 8), (8, 6), (4, 4)] * 2)


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def tree_rel_err(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    return max(map(rel_err, jax.tree.leaves(a), jax.tree.leaves(b)))


@functools.lru_cache(maxsize=None)
def setup(features):
    mesh, rng = make_mesh(), np.random.default_rng(features)
    cfg = Krea2Config(**{**TINY, "features": features})
    params = jax.tree.map(lambda s: 0.05 * rng.standard_normal(s.shape).astype(np.float32),
                          abstract_krea2_params(cfg))
    params = jax.device_put(params, param_shardings(params, mesh))
    txt_mask = np.arange(LTXT)[None] < (1 + np.arange(B) % LTXT)[:, None]
    pos, mask = build_positions_and_mask_bucketed(txt_mask, HW, NIMG)
    valid = np.arange(NIMG)[None] < HW.prod(1)[:, None]
    w = rng.standard_normal((B, NIMG, 16)).astype(np.float32) * valid[..., None]
    inp = (rng.standard_normal((B, NIMG, 16)).astype(np.float32),
           rng.standard_normal((B, LTXT, 3, 64)).astype(np.float32),
           rng.uniform(0.1, 0.9, B).astype(np.float32), pos, mask, w)
    return cfg, params, tuple(jax.device_put(jnp.asarray(a), batch_sharding(mesh)) for a in inp)


def run(cfg, params, inp, **kw):
    """(masked output, param grads) of sum(out * w); w is zero on padded image rows."""
    kw = {"remat": "dots", "qchunk": 512, "act_sharding": batch_sharding(make_mesh()), **kw}
    def loss(p, *a):
        out = dit_forward(p, cfg, *a[:5], **kw) * (a[5] != 0)
        return jnp.sum(out * a[5]), out
    (_, out), g = jax.jit(jax.value_and_grad(loss, has_aux=True))(params, *inp)
    return np.asarray(out), g


@functools.lru_cache(maxsize=None)
def ref_result(features):
    return run(*setup(features), remat=False)


@pytest.mark.parametrize("impl,features,tol", [
    ("cudnn", 128, 1e-5), ("aiter", 128, 1e-5), ("splash", 512, 1e-3)])
def test_attn_impl_matches_ref(impl, features, tol, mesh):
    """Outputs and parameter grads of each fused-attention path == attn_impl="ref"."""
    if impl == "cudnn" and "query_seq_lengths" not in inspect.signature(
            jax.nn.dot_product_attention).parameters:
        pytest.skip("jax.nn.dot_product_attention lacks seq lengths")
    kw = {"aiter": dict(attn_kernel=varlen_reference_kernel),
          "splash": dict(attn_block=128, attn_interpret=True)}.get(impl, {})
    out, g = run(*setup(features), attn_impl=impl, attn_mesh=mesh, **kw)
    out_ref, g_ref = ref_result(features)
    assert rel_err(out, out_ref) < tol
    assert tree_rel_err(g, g_ref) < tol


def test_unroll_list_blocks_matches_scan(tmp_path, mesh):
    """unroll over per-layer lists == scan over stacked blocks; same export bytes."""
    cfg, params, inp = setup(128)
    st, lst = str(tmp_path / "st.safetensors"), str(tmp_path / "ls.safetensors")
    save_krea2_params(st, params, dtype=jnp.float32)
    ls = load_krea2_params(st, mesh=mesh, dtype=jnp.float32, unstack=("blocks",))
    assert isinstance(ls["blocks"], list) and len(ls["blocks"]) == cfg.layers
    save_krea2_params(lst, ls, dtype=jnp.float32)
    assert open(lst, "rb").read() == open(st, "rb").read()
    out_s, g_s = run(cfg, params, inp, layer_loop="scan")
    out_u, g_u = run(cfg, ls, inp, layer_loop="unroll")
    assert rel_err(out_u, out_s) < 1e-5
    assert tree_rel_err(stack_blocks(g_u), g_s) < 1e-5
    with pytest.raises(TypeError, match="stack_blocks"):
        dit_forward(ls, cfg, *inp[:5], layer_loop="scan")
    with pytest.raises(ValueError, match="layer_loop"):
        dit_forward(params, cfg, *inp[:5], layer_loop="while")


def test_chunked_ref_attention_saves_no_attention_matrix():
    """Under grad the q-chunk map must not keep every chunk's softmax for the backward:
    stacked, that is the whole Hq x L x L matrix (5.5 GB per layer of the grad_snr step,
    a failed billed G4 AOT on 2026-10-29)."""
    q = jax.ShapeDtypeStruct((1, 4, 64, 8), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((1, 2, 64, 8), jnp.bfloat16)
    f = functools.partial(scaled_dot_product_attention, qchunk=16)
    res = jax.tree.leaves(jax.eval_shape(lambda *a: jax.vjp(f, *a)[1], q, kv, kv))
    assert max(r.size for r in res) < 4 * 64 * 64


def test_abstract_params_match_official_module(ref):
    """abstract_krea2_params(KREA2_CONFIG) == official state_dict keys/shapes (meta device)."""
    import torch
    fields = ("features", "tdim", "txtdim", "heads", "multiplier", "layers", "patch",
              "channels", "bias", "theta", "kvheads", "txtlayers", "txtheads", "txtkvheads")
    with torch.device("meta"):
        model = ref.SingleStreamDiT(ref.SingleMMDiTConfig(
            **{f: getattr(KREA2_CONFIG, f) for f in fields}))
    assert official_key_shapes(abstract_krea2_params(KREA2_CONFIG)) == {
        k: tuple(v.shape) for k, v in model.state_dict().items()}
