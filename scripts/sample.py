"""Generate images with the JAX Krea 2 DiT (optionally with a trained LoRA).

The DiT runs on TPU; the Qwen3-VL text encoder and Qwen-Image VAE decoder run
via PyTorch on the host CPU (they are small/one-shot relative to sampling).

Usage:
    python scripts/sample.py --base-ckpt raw.safetensors \
        --lora lora_out/lora.safetensors \
        --prompt "a photo of sks dog on the beach" --steps 28 --cfg 4.5

For the Turbo checkpoint use: --steps 8 --cfg 0 --mu 1.15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np


def encode_prompts(prompts, model_id, max_length, device="cpu"):
    import torch

    from cache_dataset import TextEncoder

    enc = TextEncoder(model_id, device, torch.float32, max_length)
    hiddens, masks = [], []
    for p in prompts:
        h, m = enc(p)
        hiddens.append(h.float().cpu().numpy())
        masks.append(m.cpu().numpy())
    del enc
    return np.stack(hiddens), np.stack(masks)


def decode_latents(latents_np, vae_id, device="cpu"):
    import torch
    from diffusers import AutoencoderKLQwenImage

    vae = (
        AutoencoderKLQwenImage.from_pretrained(vae_id, subfolder="vae")
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    z = torch.from_numpy(latents_np).float().unsqueeze(2)  # (b, c, 1, h, w)
    with torch.no_grad():
        z = z * std + mean
        img = vae.decode(z).sample[:, :, 0]
    img = (img.clamp(-1, 1) * 0.5 + 0.5) * 255.0
    return img.permute(0, 2, 3, 1).byte().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=None,
                    help="default: alpha/rank from the LoRA file")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--qchunk", type=int, default=512)
    ap.add_argument(
        "--attn", choices=["ref", "splash"], default="splash",
        help="attention kernel: 'splash' = fused Pallas TPU kernel (logits "
             "stay in VMEM; heads split across devices), 'ref' = exact "
             "query-chunked float32 softmax (materializes logits in HBM)",
    )
    ap.add_argument(
        "--no-compact-text", action="store_true",
        help="keep the text sequence padded to the encoder max length "
             "(default drops padded text tokens; outputs are unchanged)",
    )
    ap.add_argument("--output", default="sample")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from PIL import Image

    from krea2_jax.checkpoint import (
        load_krea2_params,
        load_lora,
        lora_scale_from_meta,
        make_mesh,
        replicate_tree,
    )
    from krea2_jax.config import KREA2_CONFIG
    from krea2_jax.sampling import sample_latents

    cfg = KREA2_CONFIG
    if args.width % 16 or args.height % 16:
        raise SystemExit("width/height must be multiples of 16")

    print("encoding prompts on CPU (PyTorch)...")
    use_cfg = args.cfg > 0
    prompts = [args.prompt] * args.num_images
    txt, txt_mask = encode_prompts(prompts, cfg.text_encoder_id, cfg.text_max_length)
    untxt = untxt_mask = None
    if use_cfg:
        untxt, untxt_mask = encode_prompts(
            [args.negative_prompt] * args.num_images,
            cfg.text_encoder_id,
            cfg.text_max_length,
        )

    if args.attn == "splash" and jax.default_backend() != "tpu":
        print(f"splash attention needs a TPU (backend: {jax.default_backend()}); "
              "using --attn ref")
        args.attn = "ref"

    mesh = make_mesh()
    print(f"loading DiT onto {mesh.size} device(s)...")
    params = load_krea2_params(args.base_ckpt, mesh=mesh, dtype=jnp.bfloat16)

    lora, lora_scale = None, 1.0
    if args.lora:
        lora, lmeta = load_lora(args.lora, mesh=mesh)
        lora_scale = (
            args.lora_scale
            if args.lora_scale is not None
            else lora_scale_from_meta(lmeta, default_alpha=16, default_rank=16)
        )
        print(f"applied LoRA {args.lora} (scale {lora_scale:.2f})")

    to_bf16 = lambda x: jnp.asarray(x, dtype=jnp.bfloat16)
    latents = sample_latents(
        params, cfg,
        to_bf16(txt), jnp.asarray(txt_mask),
        untxt=to_bf16(untxt) if use_cfg else None,
        untxt_mask=jnp.asarray(untxt_mask) if use_cfg else None,
        lora=lora, lora_scale=lora_scale,
        height=args.height, width=args.width,
        steps=args.steps, guidance=args.cfg, mu=args.mu,
        seed=args.seed, qchunk=args.qchunk,
        attn_impl=args.attn,
        attn_mesh=mesh if mesh.size > 1 else None,
        compact=not args.no_compact_text,
    )
    latents = np.asarray(jax.device_get(latents).astype(jnp.float32))

    print("decoding latents on CPU (PyTorch)...")
    images = decode_latents(latents, cfg.vae_id)
    for i in range(len(images)):
        out = f"{args.output}_{i}.png"
        Image.fromarray(images[i]).save(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
