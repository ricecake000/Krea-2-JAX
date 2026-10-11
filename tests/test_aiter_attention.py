"""jax-aiter (varlen flash attention) path, tested on CPU via emulation.

The real kernel only runs on ROCm gfx950; here `varlen_reference_kernel`
(a pure-JAX emulation of its padded group mode) stands in for it, which
exercises everything around the kernel: the per-sample token reorder in
dit_forward, packing into fixed-stride segments with logical prefixes,
shard_map over a sharded batch, and gradients. Also checks that activation
sharding constraints do not change numerics. Runs entirely on CPU:

    python tests/test_aiter_attention.py
"""

import os, sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from krea2_jax.checkpoint import (
    abstract_krea2_params, batch_sharding, init_lora, make_mesh,
    param_shardings, replicated_sharding,
)
from krea2_jax.config import Krea2Config
from krea2_jax.model import (
    aiter_attention, build_positions_and_mask, dit_forward, patchify,
    scaled_dot_product_attention, varlen_reference_kernel,
)

assert len(jax.devices()) == 8, jax.devices()
rng = np.random.default_rng(0)


def rel_err(a, b):
    """max|a-b| / max|b|; inf if either operand has any non-finite element."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def max_err(*errs):
    """max that propagates inf/NaN (Python max() can drop a trailing NaN)."""
    arr = np.asarray(errs, np.float64)
    return float("inf") if not np.isfinite(arr).all() else float(arr.max())


def tree_rel_err(ta, tb):
    """max over leaves of max|a-b| / max|b| (leaf-normalized); inf if any
    leaf is non-finite."""
    la, lb = jax.tree.leaves(ta), jax.tree.leaves(tb)
    assert len(la) == len(lb) and la, (len(la), len(lb))
    return max_err(*(rel_err(a, b) for a, b in zip(la, lb)))


def within(err, tol):
    return bool(np.isfinite(err)) and err <= tol


# self-test: the helpers must not let NaN/inf through
assert rel_err(np.array([np.nan]), np.array([1.0])) == float("inf")
assert rel_err(np.array([1.0]), np.array([np.inf])) == float("inf")
assert tree_rel_err([np.ones(2), np.ones(2), np.array([np.nan])],
                    [np.ones(2), np.ones(2), np.ones(1)]) == float("inf")
assert max_err(1e-9, float("nan")) == float("inf")
assert not within(float("nan"), 1.0) and not within(float("inf"), 1.0)


# ---------------------------------------------------------------------------
# (a) kernel level: emulated varlen kernel == key-masked exact attention
# ---------------------------------------------------------------------------


def numpy_attention(q, k, v, keymask):
    """Independent float64 reference: key-masked softmax attention, GQA,
    zeros on invalid query rows. q (B,Hq,L,D), k/v (B,Hk,L,D) -> (B,L,Hq*D)."""
    B, Hq, L, D = q.shape
    G = Hq // k.shape[1]
    q, k, v = (np.asarray(a, np.float64) for a in (q, k, v))
    k = np.repeat(k, G, axis=1)
    v = np.repeat(v, G, axis=1)
    logits = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(D)
    logits = np.where(keymask[:, None, None, :], logits, -np.inf)
    logits -= logits.max(-1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(-1, keepdims=True)
    out = np.einsum("bhqk,bhkd->bhqd", w, v)
    out = np.where(keymask[:, None, :, None], out, 0.0)
    return out.transpose(0, 2, 1, 3).reshape(B, L, Hq * D)


B, L, D = 4, 128, 32
lens = np.array([128, 77, 1, 100])
keymask = np.arange(L)[None, :] < lens[:, None]  # prefix masks
for Hq, Hk in ((6, 3), (4, 4)):
    q = rng.standard_normal((B, Hq, L, D)).astype(np.float32)
    k = rng.standard_normal((B, Hk, L, D)).astype(np.float32)
    v = rng.standard_normal((B, Hk, L, D)).astype(np.float32)
    km = jnp.asarray(keymask)

    # direct kernel call in the packed layout
    q3 = jnp.asarray(q).transpose(0, 2, 1, 3).reshape(B * L, Hq, D)
    k3 = jnp.asarray(k).transpose(0, 2, 1, 3).reshape(B * L, Hk, D)
    v3 = jnp.asarray(v).transpose(0, 2, 1, 3).reshape(B * L, Hk, D)
    seqstart = jnp.arange(B + 1, dtype=jnp.int32) * L
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
    (out3,) = varlen_reference_kernel(
        q3, k3, v3, seqstart, seqstart, cu, cu, L, L,
        softmax_scale=1.0 / np.sqrt(D), causal=False, deterministic=False,
    )
    assert out3.shape == q3.shape and out3.dtype == q3.dtype
    emul = out3.reshape(B, L, Hq * D)

    sdpa = scaled_dot_product_attention(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), keymask=km,
        qchunk=1 << 30,
    )
    sdpa = jnp.where(km[:, :, None], sdpa, 0.0)  # zeroed padded query rows
    npref = numpy_attention(q, k, v, keymask)

    via_api = aiter_attention(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), keymask=km,
        kernel=varlen_reference_kernel,
    )
    e1 = float(jnp.abs(emul - sdpa).max())
    e2 = float(np.abs(np.asarray(emul) - npref).max())
    e3 = float(jnp.abs(via_api - sdpa).max())
    print(f"GQA {Hq}/{Hk}: emul vs sdpa {e1:.2e}, emul vs numpy {e2:.2e}, "
          f"aiter_attention vs sdpa {e3:.2e}")
    assert e1 < 1e-5 and e2 < 1e-5 and e3 < 1e-5, (e1, e2, e3)
    pad_rows = np.asarray(emul)[~keymask]
    assert np.all(pad_rows == 0.0), "padded query rows must be exactly zero"

    # a non-default softmax scale is honoured
    s = 0.37
    (o_s,) = varlen_reference_kernel(q3, k3, v3, seqstart, seqstart, cu, cu,
                                     L, L, softmax_scale=s)
    want = numpy_attention(q * (s * np.sqrt(D)), k, v, keymask)
    e4 = float(np.abs(np.asarray(o_s).reshape(B, L, -1) - want).max())
    assert e4 < 1e-5, e4

    # gradients of the packed path == sdpa with zeroed rows
    w = jnp.asarray(rng.standard_normal((B, L, Hq * D)).astype(np.float32))

    def f_api(q, k, v):
        return jnp.sum(aiter_attention(q, k, v, keymask=km,
                                       kernel=varlen_reference_kernel) * w)

    def f_ref(q, k, v):
        o = scaled_dot_product_attention(q, k, v, keymask=km, qchunk=1 << 30)
        return jnp.sum(jnp.where(km[:, :, None], o, 0.0) * w)

    args = tuple(jnp.asarray(a) for a in (q, k, v))
    ga = jax.grad(f_api, argnums=(0, 1, 2))(*args)
    gr = jax.grad(f_ref, argnums=(0, 1, 2))(*args)
    ge = max_err(*(rel_err(a, b) for a, b in zip(ga, gr)))
    print(f"GQA {Hq}/{Hk}: grad max rel err {ge:.2e}")
    assert within(ge, 1e-5), ge

# keymask=None == all valid
o_none = aiter_attention(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v),
                         kernel=varlen_reference_kernel)
o_full = scaled_dot_product_attention(jnp.asarray(q), jnp.asarray(k),
                                      jnp.asarray(v), qchunk=1 << 30)
assert float(jnp.abs(o_none - o_full).max()) < 1e-5

# the emulator verifies the arguments it cannot honour instead of ignoring
# them: wrong seqstart / cu_q != cu_k -> all-NaN output (also under jit),
# a static max_seqlen != L -> ValueError
zeros = jnp.zeros_like(seqstart)
cu_other = cu.at[1:].add(-1).at[-1].set(cu[-1])  # different lengths, same total
for desc, ss_q, ss_k, cq, ck in (
    ("seqstart_q zeros", zeros, seqstart, cu, cu),
    ("seqstart_k zeros", seqstart, zeros, cu, cu),
    ("cu_q != cu_k", seqstart, seqstart, cu, cu_other),
):
    for jitted in (False, True):
        call = lambda *a: varlen_reference_kernel(*a, L, L)[0]
        o_bad = (jax.jit(call) if jitted else call)(q3, k3, v3, ss_q, ss_k, cq, ck)
        assert bool(jnp.all(jnp.isnan(o_bad))), (desc, jitted)
(o_ok,) = jax.jit(lambda *a: varlen_reference_kernel(*a, L, L))(
    q3, k3, v3, seqstart, seqstart, cu, cu)
assert bool(jnp.all(jnp.isfinite(o_ok)))
for mq, mk in ((1, L), (L, 1), (np.int32(1), L)):
    try:
        varlen_reference_kernel(q3, k3, v3, seqstart, seqstart, cu, cu, mq, mk)
    except ValueError as e:
        assert "max_seqlen" in str(e), str(e)
    else:
        raise AssertionError(f"expected ValueError for max_seqlen {mq}, {mk}")
print("emulator: bad seqstart / cu_q != cu_k -> NaN, max_seqlen != L -> "
      "ValueError: ok")

# ---------------------------------------------------------------------------
# (b) dit_forward(attn_impl="aiter") == ref path: outputs + grads
# ---------------------------------------------------------------------------

TINY = dict(features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
            layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3)
cfg = Krea2Config(**TINY)

abstract = abstract_krea2_params(cfg)
keys = jax.random.split(jax.random.PRNGKey(1), len(jax.tree.leaves(abstract)))
params = jax.tree.unflatten(
    jax.tree.structure(abstract),
    [0.05 * jax.random.normal(kk, s.shape, s.dtype)
     for kk, s in zip(keys, jax.tree.leaves(abstract))],
)
lora = init_lora(params, jax.random.PRNGKey(0), rank=4, targets="all")
lora = jax.tree.map(lambda a: a + 0.01, lora)  # nonzero lora_B

Bm, ltxt, latH, latW = 8, 6, 16, 16
htok, wtok = latH // 2, latW // 2
latents = rng.normal(size=(Bm, 4, latH, latW)).astype(np.float32)
txt = jnp.asarray(rng.normal(size=(Bm, ltxt, 3, 64)).astype(np.float32))
# per-sample text lengths: padding sits in the middle of [txt, img, seq-pad]
txt_mask = np.arange(ltxt)[None, :] < (1 + np.arange(Bm) % ltxt)[:, None]
assert len({int(r.sum()) for r in txt_mask}) > 1
pos, mask = build_positions_and_mask(jnp.asarray(txt_mask), htok, wtok)
x0 = patchify(jnp.asarray(latents), 2)
t = jnp.asarray(rng.uniform(0.05, 0.95, size=(Bm,)).astype(np.float32))
target = jnp.asarray(
    rng.normal(size=(Bm, htok * wtok, 16)).astype(np.float32))


def make_loss(remat, **kw):
    def fn(params, lora, x0, txt, t, pos, mask):
        v = dit_forward(params, cfg, x0, txt, t, pos, mask, lora=lora,
                        lora_scale=1.0, remat=remat, **kw)
        return jnp.mean((v - target) ** 2), v
    return fn


def run(remat, inputs, **kw):
    fn = jax.jit(jax.value_and_grad(make_loss(remat, **kw), argnums=(0, 1),
                                    has_aux=True))
    (loss, v), (gp, gl) = fn(*inputs)
    return float(loss), v, gp, gl


local_inputs = (params, lora, x0, txt, t, pos, mask)
l_ref, v_ref, gp_ref, gl_ref = run(False, local_inputs)

mesh = make_mesh()
dsh = batch_sharding(mesh)
psh = param_shardings(params, mesh)
sharded_inputs = (
    jax.tree.map(jax.device_put, params, psh),
    jax.tree.map(lambda a: jax.device_put(a, replicated_sharding(mesh)), lora),
    *(jax.device_put(a, dsh) for a in (x0, txt, t, pos, mask)),
)

aiter_kw = dict(attn_impl="aiter", attn_kernel=varlen_reference_kernel)
for remat in (False, True, "dots"):
    for use_mesh in (False, True):
        inputs = sharded_inputs if use_mesh else local_inputs
        l_a, v_a, gp_a, gl_a = run(remat, inputs, attn_mesh=mesh if use_mesh else None,
                                   **aiter_kw)
        ev = rel_err(v_a, v_ref)
        el = abs(l_a - l_ref) / abs(l_ref)
        egp = tree_rel_err(gp_a, gp_ref)
        egl = tree_rel_err(gl_a, gl_ref)
        print(f"dit aiter vs ref (remat={remat!r}, mesh={use_mesh}): out {ev:.2e} "
              f"loss {el:.2e} dparams {egp:.2e} dlora {egl:.2e}")
        assert within(max_err(ev, el, egp, egl), 1e-5), (remat, use_mesh, ev, el, egp, egl)

# the reorder must not leak into the non-aiter impls: ref output unchanged
v_ref2 = dit_forward(params, cfg, x0, txt, t, pos, mask, lora=lora, remat=False)
assert float(jnp.abs(v_ref2 - v_ref).max()) < 1e-6

# ---------------------------------------------------------------------------
# (c) act_sharding on vs off: identical outputs and grads on 8 devices
# ---------------------------------------------------------------------------

for impl_kw in (dict(), dict(attn_mesh=mesh, **aiter_kw)):
    name = impl_kw.get("attn_impl", "ref")
    l_off, v_off, gp_off, gl_off = run("dots", sharded_inputs, **impl_kw)
    l_on, v_on, gp_on, gl_on = run("dots", sharded_inputs, act_sharding=dsh,
                                   **impl_kw)
    ev = rel_err(v_on, v_off)
    egp = tree_rel_err(gp_on, gp_off)
    egl = tree_rel_err(gl_on, gl_off)
    el = abs(l_on - l_off) / abs(l_off)
    print(f"act_sharding on vs off ({name}): out {ev:.2e} loss {el:.2e} "
          f"dparams {egp:.2e} dlora {egl:.2e}")
    # sharding constraints are numerically a no-op; the only differences
    # are fp32 reduction-order effects of the different partitioning, which
    # reach ~2e-6 (leaf-relative) on the tiny text-fusion grads
    assert within(max_err(ev, el), 1e-6), (name, ev, el)
    assert within(max_err(egp, egl), 1e-5), (name, egp, egl)
    assert v_on.sharding.is_equivalent_to(dsh, v_on.ndim), v_on.sharding

# ---------------------------------------------------------------------------
# (d) without jax-aiter, kernel=None fails with a clear ImportError
# ---------------------------------------------------------------------------

_saved = {m: sys.modules.get(m) for m in ("jax_aiter", "jax_aiter.mha")}
sys.modules["jax_aiter"] = None  # makes `import jax_aiter...` raise
sys.modules["jax_aiter.mha"] = None
try:
    for m in (None, mesh):
        try:
            aiter_attention(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v),
                            keymask=jnp.asarray(keymask), mesh=m)
        except ImportError as e:
            assert "jax-aiter" in str(e), str(e)
        else:
            raise AssertionError("expected ImportError without jax-aiter")
    print("missing jax-aiter -> ImportError: ok")
finally:
    for m, mod in _saved.items():
        if mod is None:
            sys.modules.pop(m, None)
        else:
            sys.modules[m] = mod

print("ALL AITER ATTENTION TESTS PASSED")
