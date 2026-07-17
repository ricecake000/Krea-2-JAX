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

    # --- find_latest_checkpoint picks the highest-step LoRA file ---
    sys.path.insert(0, str(REPO / "scripts"))
    from train_lora import find_latest_checkpoint

    assert find_latest_checkpoint(tmp) == mid_lora_path  # lora.safetensors has no step
    save_lora(str(tmp / "lora.safetensors"), lora, rank=4, alpha=8,
              extra_meta={"step": 30})
    assert find_latest_checkpoint(tmp) == tmp / "lora.safetensors"
    (tmp / "empty").mkdir()
    assert find_latest_checkpoint(tmp / "empty") is None
    print("find_latest_checkpoint: ok")

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

    # --- "full" targets: every Linear in the DiT is adapted and every
    # adapter actually reaches the forward pass (verified against merge) ---
    lora_f = init_lora(params, jax.random.PRNGKey(7), rank=4, targets="full")
    leaves, treedef = jax.tree.flatten(lora_f)
    # blocks (5 attn + 3 mlp) + txtfusion 2x(5 attn + 3 mlp) + 8 standalone
    # projections = 32 adapters, A and B each
    assert len(leaves) == 64, f"expected 64 full-target leaves, got {len(leaves)}"
    kb = jax.random.PRNGKey(11)
    leaves = [
        jax.random.normal(jax.random.fold_in(kb, i), l.shape, l.dtype) * 0.02
        for i, l in enumerate(leaves)
    ]
    lora_f = jax.tree.unflatten(treedef, leaves)

    def full_loss(lora_f):
        v = dit_forward(params, cfg, x0, txt, t, pos, mask,
                        lora=lora_f, lora_scale=2.0, remat=False)
        return jnp.mean(v**2)

    grads = jax.grad(full_loss)(lora_f)
    bad_grads = []
    for g in jax.tree.leaves(grads):
        g = np.asarray(g)
        if not np.all(np.isfinite(g)):
            bad_grads.append("non-finite")
        elif g.ndim == 3 and not np.all(np.abs(g).reshape(g.shape[0], -1).max(axis=1) > 0):
            bad_grads.append("zero layer slice")
        elif g.ndim == 2 and float(np.abs(g).max()) == 0.0:
            bad_grads.append("all zero")
    assert not bad_grads, f"bad full-target gradients: {bad_grads}"
    print("full targets: finite gradient reaches every adapter layer: ok")

    full_path = tmp / "lora_full.safetensors"
    save_lora(str(full_path), lora_f, rank=4, alpha=8)
    lora_f2, _ = load_lora(str(full_path))
    for a, b in zip(jax.tree.leaves(lora_f), jax.tree.leaves(lora_f2)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-7)
    print("full-target save/load round-trip: ok")

    merged_full = tmp / "merged_full.safetensors"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "merge_lora.py"),
         "--base", str(base_path), "--lora", str(full_path),
         "--output", str(merged_full)],
        check=True, env={**os.environ},
    )
    merged_f = load_krea2_params(str(merged_full), mesh=None, dtype=jnp.float32)
    out_lora_f = dit_forward(params, cfg, x0, txt, t, pos, mask,
                             lora=lora_f, lora_scale=2.0, remat=False)
    out_merged_f = dit_forward(merged_f, cfg, x0, txt, t, pos, mask, remat=False)
    err = float(jnp.abs(out_lora_f - out_merged_f).max())
    print(f"full-target merged-vs-lora forward: max abs err {err:.3e}")
    assert err < 1e-4

    # --- hardening (Codex review): zero-init equivalence, schema and
    # structural validation, resume metadata checks ---
    import re

    from krea2_jax.checkpoint import expected_lora_modules
    from safetensors.numpy import save_file as np_save_file

    # zero-init full LoRA must reproduce the base forward (up to XLA fusion
    # reassociation — the LoRA branch adds exact zeros but changes op order)
    lora_z = init_lora(params, jax.random.PRNGKey(9), rank=4, targets="full")
    out_base = dit_forward(params, cfg, x0, txt, t, pos, mask, remat=False)
    out_z = dit_forward(params, cfg, x0, txt, t, pos, mask,
                        lora=lora_z, lora_scale=2.0, remat=False)
    np.testing.assert_allclose(
        np.asarray(out_z), np.asarray(out_base), atol=1e-6
    )
    print("zero-init full LoRA == base forward: ok")

    # saved key sets must match the schema helper exactly (legacy + full)
    from safetensors import safe_open as st_open

    def module_set(p):
        with st_open(str(p), framework="numpy") as f:
            keys = f.keys()
        mods = set()
        for k in keys:
            mod = re.sub(r"\.(lora_A|lora_B)\.weight$", "", k)
            mods.add(re.sub(r"\.\d+\.", ".", mod, count=1))
        return mods

    assert module_set(lora_path) == expected_lora_modules("attn+mlp")
    assert module_set(full_path) == expected_lora_modules("full")
    print("saved key sets match the target schema: ok")

    # corrupted / mistyped checkpoints must be rejected at load time
    with st_open(str(lora_path), framework="numpy") as f:
        wq_a = f.get_tensor("blocks.0.attn.wq.lora_A.weight")
        wq_b = f.get_tensor("blocks.0.attn.wq.lora_B.weight")
    bad = tmp / "bad.safetensors"

    def expect_load_error(flat, why, metadata=None):
        np_save_file(flat, str(bad), metadata=metadata)
        try:
            load_lora(str(bad))
        except ValueError:
            return
        raise AssertionError(f"load_lora accepted a bad file: {why}")

    expect_load_error(
        {"blocks.0.attn.wq.lora_A.weight": wq_a}, "missing lora_B"
    )
    expect_load_error(
        {"blocks.0.attn.projetor.lora_A.weight": wq_a,
         "blocks.0.attn.projetor.lora_B.weight": wq_b}, "mistyped module path"
    )
    expect_load_error(
        {"blocks.0.attn.wq.lora_A.weight": wq_a,
         "blocks.1.attn.wq.lora_A.weight": wq_a,
         "blocks.0.attn.wq.lora_B.weight": wq_b}, "A/B layer-count mismatch"
    )
    expect_load_error(
        {"blocks.0.attn.wq.lora_A.weight": wq_a,
         "blocks.0.attn.wq.lora_B.weight": wq_b},
        "modules incomplete for declared targets",
        metadata={"targets": "attn"},
    )
    expect_load_error(
        {"blocks.0.attn.wq.lora_A.weight": wq_a,
         "blocks.0.attn.wq.lora_B.weight": wq_b},
        "tensor rank != metadata rank",
        metadata={"rank": "8"},
    )
    print("load_lora rejects corrupted checkpoints: ok")

    # resume flag / checkpoint metadata mismatches must fail fast
    from train_lora import check_resume_meta

    good = {"rank": "4", "alpha": "8.0", "targets": "full"}
    check_resume_meta(good, 4, 8.0, "full")
    check_resume_meta({}, 4, 8.0, "full")  # metadata-less file: no check
    for override in ({"rank": "8"}, {"alpha": "4.0"}, {"targets": "attn"}):
        try:
            check_resume_meta({**good, **override}, 4, 8.0, "full")
        except SystemExit:
            continue
        raise AssertionError(f"resume mismatch not caught: {override}")
    print("check_resume_meta catches CLI/checkpoint mismatches: ok")

    print("ALL TRAIN/IO TESTS PASSED")


if __name__ == "__main__":
    main()
