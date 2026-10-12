"""cuDNN (jax.nn.dot_product_attention) attention path, tested on CPU.

attn_impl="cudnn" calls jax.nn.dot_product_attention with per-sample
query/key sequence lengths; off-GPU it picks the "xla" implementation, which
has the same masking semantics, so this exercises everything around the cuDNN
kernel: the per-sample token reorder in dit_forward, the BNTH -> BTNH layout,
GQA, the lengths, the batch-sharding constraint under a mesh, and gradients.
The cuDNN kernel itself only runs on an NVIDIA GPU (not covered here). Runs
entirely on CPU:

    python tests/test_cudnn_attention.py
"""

import os, sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from krea2_jax.checkpoint import (
    abstract_krea2_params, batch_sharding, init_lora, make_mesh,
    param_shardings, replicated_sharding,
)
from krea2_jax.config import Krea2Config
from krea2_jax.model import (
    attention_forward, build_positions_and_mask, cudnn_attention, dit_forward,
    patchify, scaled_dot_product_attention,
)

assert len(jax.devices()) == 8, jax.devices()
_params = inspect.signature(jax.nn.dot_product_attention).parameters
if not {"query_seq_lengths", "key_value_seq_lengths", "implementation"} <= set(_params):
    print(f"SKIP: jax {jax.__version__} dot_product_attention lacks "
          "query_seq_lengths / key_value_seq_lengths / implementation")
    sys.exit(0)
rng = np.random.default_rng(0)


def rel_err(a, b):
    """max|a-b| / max|b|; inf if either operand has any non-finite element."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def max_err(*errs):
    arr = np.asarray(errs, np.float64)
    return float("inf") if not np.isfinite(arr).all() else float(arr.max())


def tree_rel_err(ta, tb):
    la, lb = jax.tree.leaves(ta), jax.tree.leaves(tb)
    assert len(la) == len(lb) and la, (len(la), len(lb))
    return max_err(*(rel_err(a, b) for a, b in zip(la, lb)))


def within(err, tol):
    return bool(np.isfinite(err)) and err <= tol


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


# ---------------------------------------------------------------------------
# (a) kernel level: cudnn_attention ("xla" on CPU) == key-masked exact attn
# ---------------------------------------------------------------------------

mesh = make_mesh()
B, L, D = 8, 128, 32
lens = np.array([128, 77, 1, 100, 5, 128, 64, 33])
keymask = np.arange(L)[None, :] < lens[:, None]  # prefix masks
km = jnp.asarray(keymask)
for Hq, Hk in ((6, 3), (4, 4)):
    q = rng.standard_normal((B, Hq, L, D)).astype(np.float32)
    k = rng.standard_normal((B, Hk, L, D)).astype(np.float32)
    v = rng.standard_normal((B, Hk, L, D)).astype(np.float32)
    args = tuple(jnp.asarray(a) for a in (q, k, v))

    out = cudnn_attention(*args, keymask=km)
    out_x = cudnn_attention(*args, keymask=km, implementation="xla")
    assert out.shape == (B, L, Hq * D) and out.dtype == args[0].dtype
    sdpa = scaled_dot_product_attention(*args, keymask=km, qchunk=1 << 30)
    sdpa = jnp.where(km[:, :, None], sdpa, 0.0)  # zeroed padded query rows
    npref = numpy_attention(q, k, v, keymask)
    e1 = float(jnp.abs(out - sdpa).max())
    e2 = float(np.abs(np.asarray(out) - npref).max())
    e3 = float(jnp.abs(out - out_x).max())
    print(f"GQA {Hq}/{Hk}: cudnn_attention vs sdpa {e1:.2e}, vs numpy "
          f"{e2:.2e}, auto vs explicit xla {e3:.2e}")
    assert e1 < 1e-5 and e2 < 1e-5 and e3 == 0.0, (e1, e2, e3)
    assert np.all(np.asarray(out)[~keymask] == 0.0), \
        "padded query rows must be exactly zero"

    # under a mesh (batch-sharding constraint): same numbers
    out_m = jax.jit(lambda q, k, v, m: cudnn_attention(q, k, v, keymask=m,
                                                       mesh=mesh))(*args, km)
    e4 = float(jnp.abs(out_m - out).max())
    assert e4 < 1e-6, e4

    # gradients == sdpa with zeroed rows
    w = jnp.asarray(rng.standard_normal((B, L, Hq * D)).astype(np.float32))

    def f_cu(q, k, v):
        return jnp.sum(cudnn_attention(q, k, v, keymask=km) * w)

    def f_ref(q, k, v):
        o = scaled_dot_product_attention(q, k, v, keymask=km, qchunk=1 << 30)
        return jnp.sum(jnp.where(km[:, :, None], o, 0.0) * w)

    gc = jax.grad(f_cu, argnums=(0, 1, 2))(*args)
    gr = jax.grad(f_ref, argnums=(0, 1, 2))(*args)
    ge = max_err(*(rel_err(a, b) for a, b in zip(gc, gr)))
    print(f"GQA {Hq}/{Hk}: grad max rel err {ge:.2e}")
    assert within(ge, 1e-5), ge

# negative control: the lengths describe a PREFIX, so a mask with a hole
# (what dit_forward's reorder removes) is NOT honoured
holed = keymask.copy()
holed[0, 10:20] = False
o_h = cudnn_attention(*args, keymask=jnp.asarray(holed))
s_h = scaled_dot_product_attention(*args, keymask=jnp.asarray(holed),
                                   qchunk=1 << 30)
assert float(jnp.abs(o_h[0, :20] - s_h[0, :20]).max()) > 1e-3

# keymask=None == all valid (no lengths passed)
o_none = cudnn_attention(*args)
o_full = scaled_dot_product_attention(*args, qchunk=1 << 30)
assert float(jnp.abs(o_none - o_full).max()) < 1e-5

# bf16 inputs (the training dtype) keep their dtype and stay close
argsb = tuple(a.astype(jnp.bfloat16) for a in args)
ob = cudnn_attention(*argsb, keymask=km)
assert ob.dtype == jnp.bfloat16
eb = rel_err(ob.astype(jnp.float32), np.asarray(out))
print(f"bf16 vs f32 rel err {eb:.2e}")
assert within(eb, 3e-2), eb

# the real cuDNN kernel is not available on CPU: an explicit
# implementation="cudnn" must fail loudly rather than silently fall back
try:
    jax.block_until_ready(
        cudnn_attention(*argsb, keymask=km, implementation="cudnn"))
except Exception as e:  # RuntimeError "cuDNN is not detected" or similar
    print(f"implementation='cudnn' on CPU -> {type(e).__name__}: "
          f"{str(e).splitlines()[0][:100]}")
else:
    raise AssertionError("expected implementation='cudnn' to fail on CPU")

# ---------------------------------------------------------------------------
# (b) attention_forward level with a prefix keymask: valid rows == ref
# ---------------------------------------------------------------------------

F, H, KV = 64, 4, 2
hd = F // H
ka = jax.random.split(jax.random.PRNGKey(3), 8)
attn_p = {
    "wq": {"weight": 0.1 * jax.random.normal(ka[0], (H * hd, F))},
    "wk": {"weight": 0.1 * jax.random.normal(ka[1], (KV * hd, F))},
    "wv": {"weight": 0.1 * jax.random.normal(ka[2], (KV * hd, F))},
    "gate": {"weight": 0.1 * jax.random.normal(ka[3], (H * hd, F))},
    "wo": {"weight": 0.1 * jax.random.normal(ka[4], (F, H * hd))},
    "qknorm": {"qnorm": {"scale": 0.1 * jax.random.normal(ka[5], (hd,))},
               "knorm": {"scale": 0.1 * jax.random.normal(ka[6], (hd,))}},
}
xa = jax.random.normal(ka[7], (B, L, F))
af_ref = attention_forward(xa, attn_p, H, KV, keymask=km, attn_impl="ref",
                           qchunk=1 << 30)
af_cu = attention_forward(xa, attn_p, H, KV, keymask=km, attn_impl="cudnn")
ea = rel_err(np.asarray(af_cu)[keymask], np.asarray(af_ref)[keymask])
print(f"attention_forward cudnn vs ref (valid rows): {ea:.2e}")
assert within(ea, 1e-5), ea

# ---------------------------------------------------------------------------
# (c) dit_forward(attn_impl="cudnn") == ref path: outputs + grads
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
        # dit_forward returns only the image rows (all valid): padded text
        # rows and the sequence padding are sliced away
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

dsh = batch_sharding(mesh)
psh = param_shardings(params, mesh)
sharded_inputs = (
    jax.tree.map(jax.device_put, params, psh),
    jax.tree.map(lambda a: jax.device_put(a, replicated_sharding(mesh)), lora),
    *(jax.device_put(a, dsh) for a in (x0, txt, t, pos, mask)),
)

for remat in (False, True, "dots"):
    for use_mesh in (False, True):
        inputs = sharded_inputs if use_mesh else local_inputs
        extra = dict(attn_mesh=mesh, act_sharding=dsh) if use_mesh else {}
        l_c, v_c, gp_c, gl_c = run(remat, inputs, attn_impl="cudnn", **extra)
        ev = rel_err(v_c, v_ref)
        el = abs(l_c - l_ref) / abs(l_ref)
        egp = tree_rel_err(gp_c, gp_ref)
        egl = tree_rel_err(gl_c, gl_ref)
        print(f"dit cudnn vs ref (remat={remat!r}, mesh={use_mesh}): out "
              f"{ev:.2e} loss {el:.2e} dparams {egp:.2e} dlora {egl:.2e}")
        assert within(max_err(ev, el, egp, egl), 1e-5), \
            (remat, use_mesh, ev, el, egp, egl)
        if use_mesh:
            assert v_c.sharding.is_equivalent_to(dsh, v_c.ndim), v_c.sharding

# precomputed text tokens (the sampler path) go through the same reorder
from krea2_jax.model import text_tokens_forward  # noqa: E402

tt = text_tokens_forward(params, cfg, txt, jnp.asarray(txt_mask), lora=lora)
v_tt = dit_forward(params, cfg, x0, None, t, pos, mask, lora=lora,
                   remat=False, attn_impl="cudnn", txt_tokens=tt)
et = rel_err(v_tt, v_ref)
print(f"dit cudnn with txt_tokens vs ref: {et:.2e}")
assert within(et, 1e-5), et

print("ALL CUDNN ATTENTION TESTS PASSED")
