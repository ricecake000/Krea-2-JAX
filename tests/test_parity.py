"""Numerical parity test: JAX DiT vs. the official PyTorch reference.

Builds a tiny randomly-initialized SingleStreamDiT with the reference code
(downloaded from github.com/krea-ai/krea-2 on first run, or set
KREA2_REF_DIR), round-trips the weights through our safetensors loader, and
checks that both implementations produce the same image-token outputs.

Requires CPU PyTorch + einops in addition to JAX:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install einops
    python tests/test_parity.py
"""

import contextlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # reference uses @torch.compile
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

REF_URL = "https://raw.githubusercontent.com/krea-ai/krea-2/main/mmdit.py"


def get_reference():
    ref_dir = os.environ.get("KREA2_REF_DIR")
    if ref_dir and (Path(ref_dir) / "mmdit.py").exists():
        path = Path(ref_dir)
    else:
        path = Path(tempfile.gettempdir()) / "krea2_ref"
        path.mkdir(exist_ok=True)
        if not (path / "mmdit.py").exists():
            print(f"downloading reference mmdit.py from {REF_URL}")
            urllib.request.urlretrieve(REF_URL, path / "mmdit.py")
    sys.path.insert(0, str(path))
    import mmdit

    return mmdit


def patch_reference_for_cpu(mmdit, torch):
    """Replace the cuDNN-only SDPA with an exact math version that (like the
    fused CUDA kernels) returns zeros for fully-masked query rows."""
    import torch.nn.functional as F
    from einops import rearrange

    def attention(q, k, v, mask=None, scale=None, gqa=False):
        if gqa:
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        d = q.shape[-1]
        s = scale if scale is not None else 1.0 / d**0.5
        logits = (q.float() @ k.float().transpose(-2, -1)) * s
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.softmax(logits, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)  # fully-masked rows -> zeros
        x = (w @ v.float()).to(v.dtype)
        return rearrange(x, "B H L D -> B L (H D)")

    mmdit.attention = attention
    mmdit.Attention.forward.__globals__["attention"] = attention


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

    model = mmdit.SingleStreamDiT(ref_cfg).eval().float()
    # Randomize every parameter (default init leaves norms/modulations at 0).
    sd = model.state_dict()
    g = torch.Generator().manual_seed(1)
    sd = {k: torch.randn(v.shape, generator=g) * 0.02 for k, v in sd.items()}
    model.load_state_dict(sd)

    # --- inputs ---
    B, ltxt, latH, latW = 2, 7, 16, 16
    htok, wtok = latH // 2, latW // 2
    lat = torch.randn(B, tiny["channels"], latH, latW, generator=g)
    context = torch.randn(B, ltxt, tiny["txtlayers"], tiny["txtdim"], generator=g)
    t = torch.tensor([0.3, 0.8])
    txtmask = torch.ones(B, ltxt, dtype=torch.bool)
    txtmask[0, 5:] = False  # exercise padding within the text region
    txtmask[1, 6:] = False

    # reference-style prepare
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
    mask = torch.cat(
        [txtmask, torch.ones(B, htok * wtok, dtype=torch.bool)], dim=1
    )

    with torch.no_grad():
        ref_out = model(
            img=img_tokens, context=context, t=t, pos=pos, mask=mask
        ).numpy()

    # --- ours ---
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        ckpt_path = f.name
    save_file(sd, ckpt_path)
    params = load_krea2_params(ckpt_path, mesh=None, dtype=jnp.float32)

    jx_img = patchify(jnp.asarray(lat.numpy()), 2)
    np.testing.assert_allclose(
        np.asarray(jx_img), img_tokens.numpy(), rtol=0, atol=0
    )  # patchify layout matches the reference rearrange

    jpos, jmask = build_positions_and_mask(jnp.asarray(txtmask.numpy()), htok, wtok)
    np.testing.assert_allclose(np.asarray(jpos), pos.numpy(), atol=0)

    jx_out = dit_forward(
        params, jx_cfg, jx_img, jnp.asarray(context.numpy()),
        jnp.asarray(t.numpy()), jpos, jmask, remat=False, qchunk=64,
    )
    jx_out = np.asarray(jx_out)

    err = np.abs(jx_out - ref_out).max() / (np.abs(ref_out).max() + 1e-8)
    print(f"forward parity: max abs err {np.abs(jx_out - ref_out).max():.3e} "
          f"(rel {err:.3e}), ref scale {np.abs(ref_out).max():.3f}")
    assert err < 2e-4, "JAX forward does not match the PyTorch reference"

    # --- LoRA: zero-init B keeps the output identical; nonzero B == merged ---
    lora = init_lora(params, jax.random.PRNGKey(0), rank=4, targets="attn+mlp")
    out_zero = np.asarray(
        dit_forward(
            params, jx_cfg, jx_img, jnp.asarray(context.numpy()),
            jnp.asarray(t.numpy()), jpos, jmask,
            lora=lora, lora_scale=2.0, remat=False,
        )
    )
    np.testing.assert_allclose(out_zero, jx_out, atol=1e-6)
    print("LoRA zero-init parity: ok")

    lora = jax.tree.map(
        lambda a: jax.random.normal(jax.random.PRNGKey(2), a.shape) * 0.02, lora
    )
    out_lora = np.asarray(
        dit_forward(
            params, jx_cfg, jx_img, jnp.asarray(context.numpy()),
            jnp.asarray(t.numpy()), jpos, jmask,
            lora=lora, lora_scale=2.0, remat=False,
        )
    )
    # merge deltas into the torch weights and re-run the reference
    merged = dict(sd)
    for group, targets in lora["blocks"].items():
        for name, ab in targets.items():
            a, b = np.asarray(ab["lora_A"]), np.asarray(ab["lora_B"])
            for i in range(tiny["layers"]):
                key = f"blocks.{i}.{group}.{name}.weight"
                merged[key] = merged[key] + 2.0 * torch.from_numpy(b[i] @ a[i])
    model.load_state_dict(merged)
    with torch.no_grad():
        ref_lora = model(
            img=img_tokens, context=context, t=t, pos=pos, mask=mask
        ).numpy()
    err = np.abs(out_lora - ref_lora).max() / (np.abs(ref_lora).max() + 1e-8)
    print(f"LoRA-vs-merged parity: rel err {err:.3e}")
    assert err < 2e-4

    os.unlink(ckpt_path)
    print("ALL PARITY TESTS PASSED")


if __name__ == "__main__":
    main()
