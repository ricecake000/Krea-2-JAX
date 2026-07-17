"""Splash-attention parity tests (CPU, Pallas interpret mode).

Verifies that the fused TPU attention path produces the same outputs and
gradients as the exact reference implementation, at the kernel level and
through the full DiT (including the sharded shard_map path and the remat
policies). Runs entirely on CPU:

    python tests/test_attention.py
"""

import os, sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from krea2_jax.model import scaled_dot_product_attention, splash_attention

# ---------------------------------------------------------------------------
# kernel-level parity
# ---------------------------------------------------------------------------

B, Hq, Hk, L, D = 2, 8, 2, 512, 128
rng = np.random.default_rng(0)
q = rng.standard_normal((B, Hq, L, D)).astype(np.float32)
k = rng.standard_normal((B, Hk, L, D)).astype(np.float32)
v = rng.standard_normal((B, Hk, L, D)).astype(np.float32)
keymask = np.ones((B, L), dtype=bool)
keymask[0, 400:] = False  # trailing padding
keymask[1, 100:130] = False  # a masked hole mid-sequence


def ref_fn(q, k, v):
    return scaled_dot_product_attention(
        q, k, v, keymask=jnp.asarray(keymask), qchunk=1 << 30
    )


def splash_fn(q, k, v):
    return splash_attention(
        q, k, v, keymask=jnp.asarray(keymask), block=128, interpret=True
    )


ref = ref_fn(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v))
spl = splash_fn(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v))
err = float(jnp.abs(ref - spl).max())
print(f"kernel forward: max abs err {err:.3e}")
assert err < 1e-4, err

# bf16 inputs (training dtype)
refb = ref_fn(*[jnp.asarray(a, jnp.bfloat16) for a in (q, k, v)])
splb = splash_fn(*[jnp.asarray(a, jnp.bfloat16) for a in (q, k, v)])
errb = float(jnp.abs(refb.astype(jnp.float32) - splb.astype(jnp.float32)).max())
print(f"kernel forward (bf16): max abs err {errb:.3e}")
assert errb < 5e-2, errb

# gradient parity (fp32; cotangent weighted to break symmetry)
w = jnp.asarray(rng.standard_normal((B, L, Hq * D)).astype(np.float32))
g_ref = jax.grad(lambda *a: jnp.sum(ref_fn(*a) * w), argnums=(0, 1, 2))(
    jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)
)
g_spl = jax.grad(lambda *a: jnp.sum(splash_fn(*a) * w), argnums=(0, 1, 2))(
    jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)
)
for name, a, b in zip("qkv", g_ref, g_spl):
    gerr = float(jnp.abs(a - b).max() / (jnp.abs(a).max() + 1e-9))
    print(f"kernel grad d{name}: max rel err {gerr:.3e}")
    assert gerr < 1e-3, (name, gerr)

# ---------------------------------------------------------------------------
# model-level parity: dit_forward ref vs splash, remat policies, shard_map
# ---------------------------------------------------------------------------

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from test_parity import get_reference, patch_reference_for_cpu  # noqa: E402

mmdit = get_reference()
patch_reference_for_cpu(mmdit, torch)

from krea2_jax.checkpoint import (  # noqa: E402
    batch_sharding, init_lora, load_krea2_params, make_mesh, replicate_tree,
)
from krea2_jax.config import Krea2Config  # noqa: E402
from krea2_jax.model import build_positions_and_mask, dit_forward, patchify  # noqa: E402

# features/heads = head_dim 128: the splash kernel requires a multiple of 128
# (the real model is 6144/48 = 128 as well)
TINY = dict(features=512, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
            layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3)
cfg = Krea2Config(**TINY)

model = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(**TINY))
g = torch.Generator().manual_seed(1)
sd = {k_: torch.randn(v_.shape, generator=g) * 0.02
      for k_, v_ in model.state_dict().items()}
with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
    save_file(sd, f.name)
    ckpt = f.name

params = load_krea2_params(ckpt, mesh=None, dtype=jnp.float32)

Bm, ltxt, latH, latW = 8, 6, 16, 16
htok, wtok = latH // 2, latW // 2
latents = rng.normal(size=(Bm, 4, latH, latW)).astype(np.float32)
txt = rng.normal(size=(Bm, ltxt, 3, 64)).astype(np.float32)
txt_mask = np.ones((Bm, ltxt), dtype=bool)
txt_mask[0, 4:] = False

pos, mask = build_positions_and_mask(jnp.asarray(txt_mask), htok, wtok)
x0 = patchify(jnp.asarray(latents), 2)
t = jnp.full((Bm,), 0.5)
lora = init_lora(params, jax.random.PRNGKey(0), rank=4)
# nonzero lora_B so the LoRA path contributes to outputs and gradients
lora = jax.tree.map(lambda a: a + 0.01, lora)

common = dict(lora=lora, lora_scale=1.0)
out_ref = dit_forward(params, cfg, x0, jnp.asarray(txt), t, pos, mask,
                      remat=False, **common)
out_spl = dit_forward(params, cfg, x0, jnp.asarray(txt), t, pos, mask,
                      remat=False, attn_impl="splash", attn_block=128,
                      attn_interpret=True, **common)
err = float(jnp.abs(out_ref - out_spl).max() / jnp.abs(out_ref).max())
print(f"dit ref vs splash: max rel err {err:.3e}")
assert err < 1e-3, err

# remat policies must not change outputs or LoRA gradients
target = jnp.asarray(rng.normal(size=out_ref.shape).astype(np.float32))


def loss_of(remat, **kw):
    def fn(lora):
        vpred = dit_forward(params, cfg, x0, jnp.asarray(txt), t, pos, mask,
                            lora=lora, lora_scale=1.0, remat=remat, **kw)
        return jnp.mean((vpred - target) ** 2)
    return fn


l_none, g_none = jax.value_and_grad(loss_of(False))(lora)
for mode in (True, "dots"):
    l_m, g_m = jax.value_and_grad(loss_of(mode))(lora)
    dl = abs(float(l_none - l_m))
    dg = max(
        float(jnp.abs(a - b).max())
        for a, b in zip(jax.tree.leaves(g_none), jax.tree.leaves(g_m))
    )
    print(f"remat {mode!r}: dloss {dl:.3e}, max dgrad {dg:.3e}")
    assert dl < 1e-6 and dg < 1e-5, (mode, dl, dg)

# splash under shard_map with a sharded batch (the real training path),
# and the replicated-base loading mode
mesh = make_mesh()
params_rep = load_krea2_params(ckpt, mesh=mesh, dtype=jnp.float32,
                               sharding="replicate")
assert all(a.sharding.is_fully_replicated for a in jax.tree.leaves(params_rep))
data_sh = batch_sharding(mesh)
lora_rep = replicate_tree(lora, mesh)

out_sm = dit_forward(
    params_rep, cfg, jax.device_put(x0, data_sh),
    jax.device_put(jnp.asarray(txt), data_sh), t,
    jax.device_put(pos, data_sh), jax.device_put(mask, data_sh),
    lora=lora_rep, lora_scale=1.0, remat=False,
    attn_impl="splash", attn_mesh=mesh, attn_block=128, attn_interpret=True,
)
err = float(jnp.abs(out_ref - out_sm).max() / jnp.abs(out_ref).max())
print(f"dit splash+shard_map+replicated-base: max rel err {err:.3e}")
assert err < 1e-3, err

# full production composition: jit(value_and_grad) over splash attention
# under scan + remat + shard_map, sharded batch, replicated base + LoRA —
# gradients must match the local reference path
@jax.jit
def sharded_loss_and_grad(lora):
    def fn(lora):
        vpred = dit_forward(
            params_rep, cfg, jax.device_put(x0, data_sh),
            jax.device_put(jnp.asarray(txt), data_sh), t,
            jax.device_put(pos, data_sh), jax.device_put(mask, data_sh),
            lora=lora, lora_scale=1.0, remat="dots",
            attn_impl="splash", attn_mesh=mesh, attn_block=128,
            attn_interpret=True,
        )
        return jnp.mean((vpred - target) ** 2)
    return jax.value_and_grad(fn)(lora)


l_sm, g_sm = sharded_loss_and_grad(lora_rep)
l_loc, g_loc = jax.value_and_grad(loss_of(False))(lora)
dl = abs(float(l_sm - l_loc))
dg = max(
    float(jnp.abs(a - b).max() / (jnp.abs(b).max() + 1e-9))
    for a, b in zip(jax.tree.leaves(g_sm), jax.tree.leaves(g_loc))
)
print(f"jit+scan+remat+shard_map+splash backward: dloss {dl:.3e}, "
      f"max rel dgrad {dg:.3e}")
assert dl < 1e-6 and dg < 1e-3, (dl, dg)

print("ALL ATTENTION TESTS PASSED")
