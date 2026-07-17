"""Gradient parity: JAX backward vs. the official PyTorch reference.

Extends test_parity.py (which checks the forward pass only): builds the same
tiny SingleStreamDiT in both frameworks, computes an MSE flow-matching loss
against a fixed target, and compares

  1. dLoss/dW for every base weight (full backward through attention, norms,
     RoPE, text fusion, modulation),
  2. dLoss/d(lora_A, lora_B) against the chain-rule values derived from the
     reference's dLoss/dW (dL/dB = s·dL/dW·Aᵀ, dL/dA = s·Bᵀ·dL/dW),
  3. (1) again under remat=True and remat="dots" — checkpointing must not
     change gradients.

Run: .venv/bin/python tests/test_grad_parity.py
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from test_parity import get_reference, patch_reference_for_cpu


def rel_err(a, b):
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-12)


def main():
    import torch

    mmdit = get_reference()
    patch_reference_for_cpu(mmdit, torch)

    import jax
    import jax.numpy as jnp
    from safetensors.torch import save_file

    from krea2_jax.checkpoint import init_lora, load_krea2_params
    from krea2_jax.config import Krea2Config
    from krea2_jax.model import build_positions_and_mask, dit_forward, patchify

    torch.manual_seed(0)
    tiny = dict(
        features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
        layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3,
    )
    ref_cfg = mmdit.SingleMMDiTConfig(**tiny)
    jx_cfg = Krea2Config(**tiny)

    model = mmdit.SingleStreamDiT(ref_cfg).float()
    g = torch.Generator().manual_seed(1)
    sd = {k: torch.randn(v.shape, generator=g) * 0.02
          for k, v in model.state_dict().items()}
    model.load_state_dict(sd)
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(True)

    B, ltxt, latH, latW = 2, 7, 16, 24
    htok, wtok = latH // 2, latW // 2
    lat = torch.randn(B, tiny["channels"], latH, latW, generator=g)
    context = torch.randn(B, ltxt, tiny["txtlayers"], tiny["txtdim"], generator=g)
    t = torch.tensor([0.3, 0.8])
    txtmask = torch.ones(B, ltxt, dtype=torch.bool)
    txtmask[0, 5:] = False
    txtmask[1, 6:] = False
    target = torch.randn(B, htok * wtok, tiny["channels"] * 4, generator=g)

    img_tokens = (
        lat.reshape(B, tiny["channels"], htok, 2, wtok, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(B, htok * wtok, -1)
    )
    imgids = torch.zeros(htok, wtok, 3)
    imgids[..., 1] = torch.arange(htok)[:, None]
    imgids[..., 2] = torch.arange(wtok)[None, :]
    pos = torch.cat(
        [torch.zeros(B, ltxt, 3), imgids.reshape(1, -1, 3).expand(B, -1, -1)], dim=1
    )
    mask = torch.cat([txtmask, torch.ones(B, htok * wtok, dtype=torch.bool)], dim=1)

    ref_out = model(img=img_tokens, context=context, t=t, pos=pos, mask=mask)
    ref_loss = ((ref_out - target) ** 2).mean()
    ref_loss.backward()
    ref_grads = {k: p.grad.detach().numpy() for k, p in model.named_parameters()
                 if p.grad is not None}
    print(f"reference loss {float(ref_loss):.6f}, {len(ref_grads)} grads")

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        ckpt_path = f.name
    save_file(sd, ckpt_path)
    params = load_krea2_params(ckpt_path, mesh=None, dtype=jnp.float32)

    jx_img = patchify(jnp.asarray(lat.numpy()), 2)
    jpos, jmask = build_positions_and_mask(jnp.asarray(txtmask.numpy()), htok, wtok)
    jx_ctx = jnp.asarray(context.numpy())
    jx_t = jnp.asarray(t.numpy())
    jx_target = jnp.asarray(target.numpy())

    def loss_fn(params, lora=None, lora_scale=1.0, remat=False):
        v = dit_forward(
            params, jx_cfg, jx_img, jx_ctx, jx_t, jpos, jmask,
            lora=lora, lora_scale=lora_scale, remat=remat, qchunk=64,
        )
        return ((v - jx_target) ** 2).mean()

    jx_loss, jx_grads = jax.value_and_grad(loss_fn)(params)
    print(f"jax loss {float(jx_loss):.6f} (diff {abs(float(jx_loss) - float(ref_loss)):.2e})")
    assert abs(float(jx_loss) - float(ref_loss)) < 1e-6

    # --- map JAX grad tree onto torch param names (same layout as save_lora's
    # stacked convention: our tree stacks the per-layer block weights) ---
    def flatten(tree, prefix=()):
        if isinstance(tree, dict):
            for k, v in tree.items():
                yield from flatten(v, prefix + (k,))
        else:
            yield prefix, tree

    jx_flat = {".".join(p): np.asarray(v) for p, v in flatten(jx_grads)}

    worst = ("", 0.0)
    n_checked = 0
    for name, ref_g in ref_grads.items():
        # torch name blocks.{i}.attn.wq.weight -> ours blocks.attn.wq.weight[i]
        parts = name.split(".")
        stacked_i = None
        if parts[0] in ("blocks",) and parts[1].isdigit():
            stacked_i = int(parts[1])
            jx_key = ".".join(["blocks"] + parts[2:])
        elif parts[0] == "txtfusion" and len(parts) > 2 and parts[1] in (
            "layerwise_blocks", "refiner_blocks"
        ) and parts[2].isdigit():
            stacked_i = int(parts[2])
            jx_key = ".".join(["txtfusion", parts[1]] + parts[3:])
        else:
            jx_key = name
        if jx_key not in jx_flat:
            raise AssertionError(f"missing JAX grad for {name} (looked for {jx_key})")
        jg = jx_flat[jx_key]
        if stacked_i is not None:
            jg = jg[stacked_i]
        e = rel_err(jg, ref_g)
        n_checked += 1
        if e > worst[1]:
            worst = (name, e)
        if e > 2e-4:
            print(f"  MISMATCH {name}: rel err {e:.3e}")
    print(f"base-weight grads: {n_checked} tensors checked, worst {worst[0]} rel {worst[1]:.3e}")
    assert worst[1] < 2e-4, f"gradient mismatch: {worst}"

    # --- LoRA gradient chain rule vs reference weight grads ---
    lora = init_lora(params, jax.random.PRNGKey(0), rank=4, targets="attn+mlp")
    lora = jax.tree.map(
        lambda a: jax.random.normal(jax.random.PRNGKey(2), a.shape) * 0.02, lora
    )
    scale = 2.0

    # merged torch model: W' = W + s*B@A; grads dL/dW' give the chain values
    merged = {k: v.clone() for k, v in sd.items()}
    for group, targets in lora["blocks"].items():
        for tname, ab in targets.items():
            a, b = np.asarray(ab["lora_A"]), np.asarray(ab["lora_B"])
            for i in range(tiny["layers"]):
                key = f"blocks.{i}.{group}.{tname}.weight"
                merged[key] = merged[key] + scale * torch.from_numpy(b[i] @ a[i])
    model2 = mmdit.SingleStreamDiT(ref_cfg).float()
    model2.load_state_dict(merged)
    for p in model2.parameters():
        p.requires_grad_(True)
    out2 = model2(img=img_tokens, context=context, t=t, pos=pos, mask=mask)
    loss2 = ((out2 - target) ** 2).mean()
    loss2.backward()
    ref2 = {k: p.grad.detach().numpy() for k, p in model2.named_parameters()}

    jx_loss2, jx_lgrads = jax.value_and_grad(
        lambda l: loss_fn(params, lora=l, lora_scale=scale)
    )(lora)
    assert abs(float(jx_loss2) - float(loss2)) < 1e-6, (float(jx_loss2), float(loss2))

    worst_l = ("", 0.0)
    for group, targets in lora["blocks"].items():
        for tname, ab in targets.items():
            a = np.asarray(ab["lora_A"])
            b = np.asarray(ab["lora_B"])
            ga = np.asarray(jx_lgrads["blocks"][group][tname]["lora_A"])
            gb = np.asarray(jx_lgrads["blocks"][group][tname]["lora_B"])
            for i in range(tiny["layers"]):
                gw = ref2[f"blocks.{i}.{group}.{tname}.weight"]
                want_gb = scale * gw @ a[i].T
                want_ga = scale * b[i].T @ gw
                for tag, got, want in (("A", ga[i], want_ga), ("B", gb[i], want_gb)):
                    e = rel_err(got, want)
                    if e > worst_l[1]:
                        worst_l = (f"blocks.{i}.{group}.{tname}.lora_{tag}", e)
    print(f"lora grads: worst {worst_l[0]} rel {worst_l[1]:.3e}")
    assert worst_l[1] < 2e-4, f"lora gradient mismatch: {worst_l}"

    # --- remat must not change gradients ---
    for policy in (True, "dots"):
        _, g2 = jax.value_and_grad(loss_fn)(params)
        _, g_remat = jax.value_and_grad(
            lambda p: loss_fn(p, remat=policy)
        )(params)
        worst_r = 0.0
        for (k1, v1), (k2, v2) in zip(flatten(g2), flatten(g_remat)):
            worst_r = max(worst_r, rel_err(np.asarray(v2), np.asarray(v1)))
        print(f"remat={policy!r} grad drift: {worst_r:.3e}")
        assert worst_r < 1e-5

    os.unlink(ckpt_path)
    print("ALL GRADIENT PARITY TESTS PASSED")


if __name__ == "__main__":
    main()
