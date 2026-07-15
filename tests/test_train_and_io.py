"""End-to-end smoke test on CPU with a tiny model:

  1. a few LoRA training steps reduce the flow-matching loss and leave the
     base params untouched,
  2. save_lora -> load_lora round-trips exactly,
  3. scripts/merge_lora.py produces a checkpoint whose plain forward equals
     the base-plus-LoRA forward.

Requires CPU PyTorch (to fabricate a tiny reference checkpoint) + einops.
"""

import os
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from test_parity import get_reference, patch_reference_for_cpu

TINY = dict(
    features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
    layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3,
)


def main():
    import torch

    mmdit = get_reference()
    patch_reference_for_cpu(mmdit, torch)

    import jax
    import jax.numpy as jnp
    import optax
    from safetensors.torch import save_file

    from krea2_jax.checkpoint import init_lora, load_krea2_params, load_lora, save_lora
    from krea2_jax.config import Krea2Config
    from krea2_jax.model import build_positions_and_mask, dit_forward, patchify
    from krea2_jax.sampling import shift_time

    cfg = Krea2Config(**TINY)
    tmp = Path(tempfile.mkdtemp())

    torch.manual_seed(0)
    model = mmdit.SingleStreamDiT(mmdit.SingleMMDiTConfig(**TINY))
    g = torch.Generator().manual_seed(1)
    sd = {k: torch.randn(v.shape, generator=g) * 0.02
          for k, v in model.state_dict().items()}
    base_path = tmp / "base.safetensors"
    save_file(sd, str(base_path))

    params = load_krea2_params(str(base_path), mesh=None, dtype=jnp.float32)
    params_snapshot = jax.tree.map(np.asarray, params)

    # --- fixed tiny batch ---
    rng = jax.random.PRNGKey(0)
    B, ltxt, latH, latW = 4, 6, 16, 16
    htok, wtok = latH // cfg.patch, latW // cfg.patch
    k1, k2, rng = jax.random.split(rng, 3)
    latents = jax.random.normal(k1, (B, cfg.channels, latH, latW))
    txt = jax.random.normal(k2, (B, ltxt, cfg.txtlayers, cfg.txtdim))
    txt_mask = jnp.ones((B, ltxt), dtype=bool).at[0, 4:].set(False)
    pos, mask = build_positions_and_mask(txt_mask, htok, wtok)
    mu = 0.9

    lora = init_lora(params, jax.random.PRNGKey(3), rank=4, targets="attn+mlp")
    tx = optax.adamw(1e-2)
    opt_state = tx.init(lora)

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(lora, opt_state, key):
        k_t, k_n = jax.random.split(key)
        x0 = patchify(latents, cfg.patch)
        t = shift_time(jax.nn.sigmoid(jax.random.normal(k_t, (B,))), mu)
        noise = jax.random.normal(k_n, x0.shape)
        tt = t[:, None, None]
        xt = (1 - tt) * x0 + tt * noise

        def loss_fn(lora):
            v = dit_forward(params, cfg, xt, txt, t, pos, mask,
                            lora=lora, lora_scale=1.0, remat=True)
            return jnp.mean((v - (noise - x0)) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(lora)
        updates, opt_state = tx.update(grads, opt_state, lora)
        return optax.apply_updates(lora, updates), opt_state, loss

    losses = []
    for step in range(30):
        lora, opt_state, loss = train_step(lora, opt_state, jax.random.fold_in(rng, step))
        losses.append(float(loss))
    print(f"loss first 3: {losses[:3]}, last 3: {losses[-3:]}")
    assert all(np.isfinite(losses)), "non-finite training loss"
    assert np.mean(losses[-5:]) < np.mean(losses[:5]), "loss did not decrease"

    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(params_snapshot)):
        np.testing.assert_array_equal(np.asarray(a), b)
    print("base params untouched: ok")

    # --- save / load round-trip ---
    lora_path = tmp / "lora.safetensors"
    save_lora(str(lora_path), lora, rank=4, alpha=8)
    lora2, meta = load_lora(str(lora_path))
    assert meta["rank"] == "4" and meta["alpha"] == "8"
    for a, b in zip(jax.tree.leaves(lora), jax.tree.leaves(lora2)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-7)
    print("LoRA save/load round-trip: ok")

    # --- optimizer-state save/resume: 15 + 15 steps == 30 straight steps ---
    from krea2_jax.checkpoint import load_train_state, save_train_state

    lora_b = init_lora(params, jax.random.PRNGKey(3), rank=4, targets="attn+mlp")
    opt_b = tx.init(lora_b)
    for step in range(15):
        lora_b, opt_b, _ = train_step(lora_b, opt_b, jax.random.fold_in(rng, step))

    mid_lora_path = tmp / "lora_step000015.safetensors"
    mid_state_path = tmp / "train_state_step000015.safetensors"
    save_lora(str(mid_lora_path), lora_b, rank=4, alpha=8)
    save_train_state(str(mid_state_path), opt_b, step=15)

    lora_c, _ = load_lora(str(mid_lora_path))
    opt_c, start_step = load_train_state(str(mid_state_path), tx.init(lora_c))
    assert start_step == 15
    for a, b in zip(jax.tree.leaves(opt_b), jax.tree.leaves(opt_c)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for step in range(start_step, 30):
        lora_c, opt_c, _ = train_step(lora_c, opt_c, jax.random.fold_in(rng, step))
    for a, b in zip(jax.tree.leaves(lora), jax.tree.leaves(lora_c)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6)
    print("optimizer-state save/resume matches uninterrupted training: ok")

    # --- merge_lora.py equivalence ---
    merged_path = tmp / "merged.safetensors"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "merge_lora.py"),
         "--base", str(base_path), "--lora", str(lora_path),
         "--output", str(merged_path)],
        check=True, env={**os.environ},
    )
    merged_params = load_krea2_params(str(merged_path), mesh=None, dtype=jnp.float32)

    x0 = patchify(latents, cfg.patch)
    t = jnp.full((B,), 0.5)
    out_lora = dit_forward(params, cfg, x0, txt, t, pos, mask,
                           lora=lora, lora_scale=2.0, remat=False)  # alpha/rank = 2
    out_merged = dit_forward(merged_params, cfg, x0, txt, t, pos, mask, remat=False)
    err = float(jnp.abs(out_lora - out_merged).max())
    print(f"merged-vs-lora forward: max abs err {err:.3e}")
    assert err < 1e-4

    print("ALL TRAIN/IO TESTS PASSED")


if __name__ == "__main__":
    main()
