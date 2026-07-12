"""Precompute VAE latents and Qwen3-VL text states for LoRA training.

The TPU training loop runs only the DiT; the (frozen) Qwen-Image VAE and the
Qwen3-VL-4B text encoder are run once here, on CPU or GPU, and their outputs
cached to disk. Padding tokens are dropped before saving (the DiT treats text
positions identically and masks padding everywhere, so re-padding with zeros
at batch time is exact).

Dataset layout: a directory of images with same-basename caption files:
    data/dog1.jpg  data/dog1.txt  data/dog2.png  data/dog2.txt ...
Images without a .txt use --default-caption.

Usage:
    python scripts/cache_dataset.py --data-dir ./data --cache-dir ./cache \
        --resolution 1024 [--device cuda] [--max-length 512]
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Prompt template constants from the official Qwen3VLConditioner.
PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_IDX = 34
SUFFIX_START_IDX = 5
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


class TextEncoder:
    def __init__(self, model_id, device, dtype, max_length=512):
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def __call__(self, text: str):
        """Returns (hiddens (L, 12, 2560), mask (L,)) with L = max_length."""
        suffix = self.tokenizer(SUFFIX, return_tensors="pt")
        suffix_ids = suffix["input_ids"].to(self.device)
        suffix_mask = suffix["attention_mask"].bool().to(self.device)

        inputs = self.tokenizer(
            PREFIX + text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length + PREFIX_IDX - SUFFIX_START_IDX,
            return_tensors="pt",
        ).to(self.device)
        input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)

        states = self.model(
            input_ids=input_ids, attention_mask=mask, output_hidden_states=True
        )
        hiddens = torch.stack(
            [states.hidden_states[i] for i in SELECT_LAYERS], dim=2
        )
        return hiddens[0, PREFIX_IDX:], mask[0, PREFIX_IDX:]


class VaeEncoder:
    def __init__(self, model_id, device, dtype):
        from diffusers import AutoencoderKLQwenImage

        self.vae = (
            AutoencoderKLQwenImage.from_pretrained(
                model_id, subfolder="vae", torch_dtype=dtype
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.mean = torch.tensor(self.vae.config.latents_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        self.std = torch.tensor(self.vae.config.latents_std, device=device).view(
            1, -1, 1, 1, 1
        )
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: (1, 3, H, W) in [-1, 1] -> normalized latent (16, H/8, W/8)."""
        x = img.to(self.device, self.dtype).unsqueeze(2)  # (b, c, 1, h, w)
        z = self.vae.encode(x).latent_dist.sample()
        z = (z - self.mean) / self.std
        return z[0, :, 0]


def load_image(path, resolution):
    """RGB load, resize shorter side to `resolution`, center crop, to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = resolution / min(w, h)
    img = img.resize(
        (max(resolution, round(w * scale)), max(resolution, round(h * scale))),
        Image.LANCZOS,
    )
    w, h = img.size
    left, top = (w - resolution) // 2, (h - resolution) // 2
    img = img.crop((left, top, left + resolution, top + resolution))
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(resolution, resolution, 3).permute(2, 0, 1).float()
    return (t / 127.5 - 1.0).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--default-caption", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text-encoder", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--vae", default="Qwen/Qwen-Image")
    args = ap.parse_args()

    if args.resolution % 16 != 0:
        raise SystemExit("--resolution must be a multiple of 16 (VAE f8 × patch 2)")

    data_dir, cache_dir = Path(args.data_dir), Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise SystemExit(f"no images found in {data_dir}")
    print(f"found {len(images)} images")

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    text_encoder = TextEncoder(args.text_encoder, args.device, dtype, args.max_length)
    vae = VaeEncoder(args.vae, args.device, dtype)

    samples = []
    for i, path in enumerate(images):
        cap_path = path.with_suffix(".txt")
        caption = (
            cap_path.read_text().strip() if cap_path.exists() else args.default_caption
        )
        latent = vae(load_image(path, args.resolution))
        hiddens, mask = text_encoder(caption)
        n_valid = int(mask.sum().item())
        # Compact away padding: valid tokens are (prompt..., suffix) — text
        # RoPE positions are all zero and padding is masked everywhere, so
        # dropping padded rows and re-padding later is mathematically exact.
        hiddens = hiddens[mask].contiguous()  # (n_valid, 12, 2560)

        out = cache_dir / f"{i:06d}.safetensors"
        save_file(
            {
                "latent": latent.to(torch.bfloat16).cpu(),
                "txt": hiddens.to(torch.bfloat16).cpu(),
            },
            str(out),
            metadata={"caption": caption, "image": path.name},
        )
        samples.append({"file": out.name, "txt_len": n_valid, "image": path.name})
        print(f"[{i + 1}/{len(images)}] {path.name} (txt tokens: {n_valid})")

    meta = {
        "resolution": args.resolution,
        "latent_channels": 16,
        "txt_dim": 2560,
        "txt_layers": 12,
        "max_length": args.max_length,
        "text_encoder": args.text_encoder,
        "vae": args.vae,
        "samples": samples,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"cached {len(samples)} samples -> {cache_dir}")


if __name__ == "__main__":
    main()
