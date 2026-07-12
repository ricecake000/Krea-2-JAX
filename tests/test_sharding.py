import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import tempfile
from functools import partial

import numpy as np
import torch
from safetensors.torch import save_file

from test_parity import get_reference, patch_reference_for_cpu

mmdit = get_reference()
patch_reference_for_cpu(mmdit, torch)

import jax
import jax.numpy as jnp
import optax

from krea2_jax.checkpoint import (
    batch_sharding, init_lora, load_krea2_params, make_mesh, replicate_tree,
)
from krea2_jax.config import Krea2Config
from krea2_jax.model import build_positions_and_mask, dit_forward, patchify
from krea2_jax.sampling import shift_time

assert len(jax.devices()) == 8, jax.devices()

# tiny config with dims divisible by 8 for sharding
TINY = dict(features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
            layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3)
cfg = Krea2Config(**TINY)

model = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(**TINY))
g = torch.Generator().manual_seed(1)
sd = {k: torch.randn(v.shape, generator=g) * 0.02 for k, v in model.state_dict().items()}
with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
    save_file(sd, f.name)
    ckpt = f.name

mesh = make_mesh()
params_sharded = load_krea2_params(ckpt, mesh=mesh, dtype=jnp.float32)
params_local = load_krea2_params(ckpt, mesh=None, dtype=jnp.float32)

# check some leaves actually got sharded
sharded_leaves = [
    a for a in jax.tree.leaves(params_sharded)
    if not a.sharding.is_fully_replicated
]
print(f"sharded leaves: {len(sharded_leaves)}/{len(jax.tree.leaves(params_sharded))}")
assert sharded_leaves, "nothing was sharded"

B, ltxt, latH, latW = 8, 6, 16, 16
htok, wtok = latH // 2, latW // 2
rng = np.random.default_rng(0)
latents = rng.normal(size=(B, 4, latH, latW)).astype(np.float32)
txt = rng.normal(size=(B, ltxt, 3, 64)).astype(np.float32)
txt_mask = np.ones((B, ltxt), dtype=bool)
txt_mask[0, 4:] = False

pos, mask = build_positions_and_mask(jnp.asarray(txt_mask), htok, wtok)
x0 = patchify(jnp.asarray(latents), 2)
t = jnp.full((B,), 0.5)

# forward parity: sharded vs unsharded
data_sh = batch_sharding(mesh)
out_sharded = dit_forward(
    params_sharded, cfg, jax.device_put(x0, data_sh),
    jax.device_put(jnp.asarray(txt), data_sh), t,
    jax.device_put(pos, data_sh), jax.device_put(mask, data_sh), remat=False,
)
out_local = dit_forward(params_local, cfg, x0, jnp.asarray(txt), t, pos, mask, remat=False)
err = float(jnp.abs(out_sharded - out_local).max())
print(f"sharded vs local forward: max abs err {err:.3e}")
assert err < 1e-5

# one sharded train step
lora = replicate_tree(init_lora(params_sharded, jax.random.PRNGKey(0), rank=4), mesh)
tx = optax.adamw(1e-3)
opt_state = replicate_tree(tx.init(lora), mesh)
mu = 0.9

@partial(jax.jit, donate_argnums=(0, 1))
def train_step(lora, opt_state, params, blat, btxt, bmask, key):
    k_t, k_n = jax.random.split(key)
    x0 = patchify(blat, 2)
    tt = shift_time(jax.nn.sigmoid(jax.random.normal(k_t, (B,))), mu)
    noise = jax.random.normal(k_n, x0.shape)
    xt = (1 - tt[:, None, None]) * x0 + tt[:, None, None] * noise
    p, m = build_positions_and_mask(bmask, htok, wtok)

    def loss_fn(lora):
        v = dit_forward(params, cfg, xt, btxt, tt, p, m, lora=lora, remat=True)
        return jnp.mean((v - (noise - x0)) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(lora)
    updates, opt_state = tx.update(grads, opt_state, lora)
    return optax.apply_updates(lora, updates), opt_state, loss

for step in range(3):
    lora, opt_state, loss = train_step(
        lora, opt_state, params_sharded,
        jax.device_put(jnp.asarray(latents), data_sh),
        jax.device_put(jnp.asarray(txt), data_sh),
        jax.device_put(jnp.asarray(txt_mask), data_sh),
        jax.random.PRNGKey(step),
    )
    print(f"step {step}: loss {float(loss):.4f}")
    assert np.isfinite(float(loss))

print("ALL SHARDING TESTS PASSED")
