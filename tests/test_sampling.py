"""Sampler optimization tests (CPU; splash in Pallas interpret mode).

Checks that the inference-side shortcuts leave outputs unchanged:
  * precomputed text tokens (text branch run once per prompt) == full forward
  * dropping padded text tokens (`compact_text`) == padded forward
  * splash attention with heads split across devices == reference attention,
    for each head layout (KV split, KV expanded, replicated)
  * `sample_latents` end to end, compacted vs padded text, with CFG + LoRA

    python tests/test_sampling.py
"""

import os, sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import tempfile

import numpy as np
import torch
from safetensors.torch import save_file

from test_parity import get_reference, patch_reference_for_cpu

mmdit = get_reference()
patch_reference_for_cpu(mmdit, torch)

import jax
import jax.numpy as jnp

from krea2_jax.checkpoint import init_lora, load_krea2_params, make_mesh
from krea2_jax.config import Krea2Config
from krea2_jax.model import (
    build_positions_and_mask,
    dit_forward,
    patchify,
    scaled_dot_product_attention,
    splash_attention,
    text_tokens_forward,
)
from krea2_jax.sampling import compact_text, sample_latents

assert len(jax.devices()) == 8, jax.devices()
mesh = make_mesh()
rng = np.random.default_rng(0)


def rel_err(a, b):
    return float(jnp.abs(a - b).max() / jnp.abs(a).max())


# ---------------------------------------------------------------------------
# splash with heads split over the mesh (sampling path: batch replicated)
# ---------------------------------------------------------------------------

L, D = 512, 128
keymask = np.ones((1, L), dtype=bool)
keymask[0, 20:64] = False  # masked hole (text padding)
for Hq, Hk, layout in [(8, 8, "KV split"), (8, 2, "KV expanded"),
                       (4, 2, "replicated")]:
    q = jnp.asarray(rng.standard_normal((1, Hq, L, D)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((1, Hk, L, D)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((1, Hk, L, D)), jnp.float32)
    ref = scaled_dot_product_attention(
        q, k, v, keymask=jnp.asarray(keymask), qchunk=1 << 30
    )
    spl = jax.jit(lambda q, k, v: splash_attention(
        q, k, v, keymask=jnp.asarray(keymask), mesh=mesh, block=128,
        interpret=True, shard="heads",
    ))(q, k, v)
    err = float(jnp.abs(ref - spl).max())
    print(f"splash heads-sharded ({layout}, Hq={Hq}, Hk={Hk}): "
          f"max abs err {err:.3e}")
    assert err < 1e-4, (layout, err)

# ---------------------------------------------------------------------------
# tiny model (head_dim 128 for the splash kernel)
# ---------------------------------------------------------------------------

TINY = dict(features=1024, tdim=32, txtdim=64, heads=8, kvheads=2,
            multiplier=4, layers=2, patch=2, channels=4, txtheads=4,
            txtkvheads=4, txtlayers=3)
cfg = Krea2Config(**TINY)
model = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(**TINY))
g = torch.Generator().manual_seed(1)
sd = {k_: torch.randn(v_.shape, generator=g) * 0.02
      for k_, v_ in model.state_dict().items()}
with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
    save_file(sd, f.name)
    ckpt = f.name

params = load_krea2_params(ckpt, mesh=None, dtype=jnp.float32)
lora = jax.tree.map(lambda a: a + 0.01,
                    init_lora(params, jax.random.PRNGKey(0), rank=4,
                              targets="full"))
common = dict(lora=lora, lora_scale=0.5, remat=False)

B, ltxt, latH, latW = 2, 128, 16, 16
htok, wtok = latH // 2, latW // 2
x0 = patchify(jnp.asarray(rng.normal(size=(B, 4, latH, latW)), jnp.float32), 2)
txt = rng.normal(size=(B, ltxt, 3, 64)).astype(np.float32)
# encoder-style masks: a padding gap before the suffix, and trailing padding
txt_mask = np.zeros((B, ltxt), dtype=bool)
txt_mask[0, :9] = True
txt_mask[0, 30:34] = True
txt_mask[1, :5] = True
t = jnp.full((B,), 0.7)

pos, mask = build_positions_and_mask(jnp.asarray(txt_mask), htok, wtok)
out_full = dit_forward(params, cfg, x0, jnp.asarray(txt), t, pos, mask, **common)

# text branch computed once, passed to the step
tok = text_tokens_forward(params, cfg, jnp.asarray(txt), jnp.asarray(txt_mask),
                          lora=lora, lora_scale=0.5)
out_tok = dit_forward(params, cfg, x0, None, t, pos, mask, txt_tokens=tok,
                      **common)
err = rel_err(out_full, out_tok)
print(f"precomputed text tokens vs full forward: max rel err {err:.3e}")
assert err < 1e-6, err

# padded text tokens dropped
ctxt, cmask = compact_text(txt, txt_mask, multiple=4)
assert ctxt.shape[1] == 16 and int(cmask.sum()) == int(txt_mask.sum()), (
    ctxt.shape, cmask.sum())
cpos, cm = build_positions_and_mask(cmask, htok, wtok)
out_c = dit_forward(params, cfg, x0, ctxt, t, cpos, cm, **common)
err = rel_err(out_full, out_c)
print(f"compacted text ({ltxt} -> {ctxt.shape[1]} tokens) vs padded: "
      f"max rel err {err:.3e}")
assert err < 1e-5, err

# heads-sharded splash through the DiT with an FSDP-sharded base (as sample.py)
params_sh = load_krea2_params(ckpt, mesh=mesh, dtype=jnp.float32)
out_sh = dit_forward(
    params_sh, cfg, x0, jnp.asarray(txt), t, pos, mask, **common,
    attn_impl="splash", attn_mesh=mesh, attn_shard="heads", attn_block=128,
    attn_interpret=True,
)
err = rel_err(out_full, out_sh)
print(f"dit splash heads-sharded, FSDP base vs ref: max rel err {err:.3e}")
assert err < 1e-3, err

# ---------------------------------------------------------------------------
# sample_latents end to end: CFG + LoRA, compacted vs padded text
# ---------------------------------------------------------------------------

untxt = rng.normal(size=(B, ltxt, 3, 64)).astype(np.float32)
untxt_mask = np.zeros((B, ltxt), dtype=bool)
untxt_mask[:, :3] = True
run = lambda compact: sample_latents(
    params, cfg, jnp.asarray(txt), jnp.asarray(txt_mask),
    untxt=jnp.asarray(untxt), untxt_mask=jnp.asarray(untxt_mask),
    lora=lora, lora_scale=0.5, height=latH * 8, width=latW * 8, steps=3,
    guidance=4.5, seed=0, compact=compact,
).astype(jnp.float32)
lat_pad, lat_cmp = run(False), run(True)
# default bucket (64) actually shortens both text sequences here
assert compact_text(txt, txt_mask)[0].shape[1] == 64
assert compact_text(untxt, untxt_mask)[0].shape[1] == 64
err = rel_err(lat_pad, lat_cmp)
print(f"sample_latents compacted vs padded text: max rel err {err:.3e}")
# steps run in bf16 (the sampler's latent dtype), so allow bf16 rounding
assert err < 2e-2, err

print("ALL SAMPLING TESTS PASSED")
