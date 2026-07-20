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
from safetensors import safe_open
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
        self.dtype = dtype
        self.max_length = max_length

    def to_dtype(self, dtype):
        # Module.to converts parameter-by-parameter, so peak memory is
        # model + largest tensor, not 2x model.
        self.model.to(dtype)
        self.dtype = dtype

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

    def to_dtype(self, dtype):
        self.vae.to(dtype)
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: (1, 3, H, W) in [-1, 1] -> normalized latent (16, H/8, W/8)."""
        x = img.to(self.device, self.dtype).unsqueeze(2)  # (b, c, 1, h, w)
        z = self.vae.encode(x).latent_dist.sample()
        z = (z - self.mean) / self.std
        return z[0, :, 0]


def load_image(path, width, height):
    """RGB load, cover-resize to (width, height), center crop, to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(width / w, height / h)
    img = img.resize(
        (max(width, round(w * scale)), max(height, round(h * scale))),
        Image.LANCZOS,
    )
    w, h = img.size
    left, top = (w - width) // 2, (h - height) // 2
    img = img.crop((left, top, left + width, top + height))
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(height, width, 3).permute(2, 0, 1).float()
    return (t / 127.5 - 1.0).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--resolution", type=int, default=1024,
                    help="square training resolution (shorthand)")
    ap.add_argument("--width", type=int, default=None,
                    help="fixed training width; overrides --resolution")
    ap.add_argument("--height", type=int, default=None,
                    help="fixed training height; overrides --resolution")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--default-caption", default="")
    ap.add_argument(
        "--caption-prefix", default="",
        help="prepended to every caption before encoding (e.g. a trigger "
             'phrase: "PJSK_Style. "); existing cache entries whose stored '
             "caption differs are re-encoded, reusing their latents",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text-encoder", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--vae", default="Qwen/Qwen-Image")
    ap.add_argument(
        "--num-shards", type=int, default=1,
        help="split the run across N independent processes (e.g. one per "
             "GPU); file numbering stays global, so shards never collide. "
             "Sharded runs skip meta.json — run once more without shard "
             "args afterwards: every entry is reused (no model load) and "
             "the full meta.json is written",
    )
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument(
        "--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
        help="compute dtype; auto = bf16 on GPU, fp32 on CPU. fp16 uses the "
             "tensor cores on pre-Ampere GPUs (V100/T4, which lack bf16 "
             "hardware); non-finite outputs fall back to bf16 per encoder",
    )
    args = ap.parse_args()

    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else args.resolution
    if width % 16 or height % 16:
        raise SystemExit("width/height must be multiples of 16 (VAE f8 × patch 2)")

    data_dir, cache_dir = Path(args.data_dir), Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise SystemExit(f"no images found in {data_dir}")
    print(f"found {len(images)} images")

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    # Interleaved split balances content across shards; enumerate before
    # slicing so cache filenames keep their global index.
    jobs = [
        (i, p) for i, p in enumerate(images)
        if i % args.num_shards == args.shard_index
    ]
    if args.num_shards > 1:
        print(f"shard {args.shard_index}/{args.num_shards}: "
              f"{len(jobs)} images")

    def cached_txt_len(out, caption, name):
        """txt token count if `out` is a valid cache entry for exactly this
        (image, caption, resolution, max_length) job, else None.

        Outputs are deterministic per input+settings, so an entry may only be
        reused when everything that influenced it still matches; a truncated
        file (previous interruption) fails safe_open and is recomputed.
        """
        try:
            with safe_open(str(out), framework="pt") as f:
                m = f.metadata() or {}
                if m.get("image") != name or m.get("caption") != caption:
                    return None
                lat_shape = tuple(f.get_slice("latent").get_shape())
                txt_shape = tuple(f.get_slice("txt").get_shape())
        except Exception:
            return None
        if lat_shape != (16, height // 8, width // 8):
            return None
        if len(txt_shape) != 3 or txt_shape[1:] != (12, 2560):
            return None
        if txt_shape[0] > args.max_length:
            return None
        return txt_shape[0]

    # Encoders load lazily: a fully cached rerun never touches the models.
    if args.dtype == "auto":
        dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    else:
        dtype = {
            "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32
        }[args.dtype]
    text_encoder = vae = None

    def ensure_finite(tensor, encoder, recompute, label, name):
        """fp16 can overflow (max 65504) where bf16 does not; on a non-finite
        result, permanently switch that encoder to bf16 and redo the sample."""
        if torch.isfinite(tensor).all():
            return tensor
        if encoder.dtype == torch.bfloat16:
            raise SystemExit(f"non-finite {label} for {name} even in bf16")
        print(f"WARNING: non-finite {label} for {name}; "
              f"switching {type(encoder).__name__} to bf16")
        encoder.to_dtype(torch.bfloat16)
        tensor = recompute()
        if not torch.isfinite(tensor).all():
            raise SystemExit(f"non-finite {label} for {name} even in bf16")
        return tensor

    def cached_latent(out, name):
        """Reuse the stored latent when only the caption changed: the image,
        resolution, and VAE are unchanged, so the latent is identical."""
        try:
            with safe_open(str(out), framework="pt") as f:
                if (f.metadata() or {}).get("image") != name:
                    return None
                lat = f.get_tensor("latent")
        except Exception:
            return None
        return lat if lat.shape == (16, height // 8, width // 8) else None

    samples = []
    for i, path in jobs:
        cap_path = path.with_suffix(".txt")
        caption = (
            cap_path.read_text().strip() if cap_path.exists() else args.default_caption
        )
        if args.caption_prefix:
            caption = args.caption_prefix + caption
        out = cache_dir / f"{i:06d}.safetensors"
        latent = None
        if out.exists():
            n_valid = cached_txt_len(out, caption, path.name)
            if n_valid is not None:
                samples.append(
                    {"file": out.name, "txt_len": n_valid, "image": path.name}
                )
                print(f"[{i + 1}/{len(images)}] {path.name} (already cached)")
                continue
            latent = cached_latent(out, path.name)
        if text_encoder is None:
            text_encoder = TextEncoder(
                args.text_encoder, args.device, dtype, args.max_length
            )
        if latent is None:
            if vae is None:
                vae = VaeEncoder(args.vae, args.device, dtype)
            latent = ensure_finite(
                vae(load_image(path, width, height)), vae,
                lambda: vae(load_image(path, width, height)),
                "latent", path.name,
            )

        def encode_txt():
            hiddens, mask = text_encoder(caption)
            # Compact away padding: valid tokens are (prompt..., suffix) —
            # text RoPE positions are all zero and padding is masked
            # everywhere, so dropping padded rows and re-padding later is
            # mathematically exact. (Padded rows are also excluded from the
            # finite check: they may legitimately overflow and are dropped.)
            return hiddens[mask].contiguous(), mask  # (n_valid, 12, 2560)

        hiddens, mask = encode_txt()
        hiddens = ensure_finite(
            hiddens, text_encoder, lambda: encode_txt()[0], "txt", path.name
        )
        n_valid = int(mask.sum().item())

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

    if args.num_shards > 1:
        print(f"shard {args.shard_index}/{args.num_shards} done: "
              f"{len(samples)} samples (meta.json left to the merge pass)")
        return

    meta = {
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "latent_channels": 16,
        "txt_dim": 2560,
        "txt_layers": 12,
        "max_length": args.max_length,
        "compute_dtype": args.dtype,
        "caption_prefix": args.caption_prefix,
        "text_encoder": args.text_encoder,
        "vae": args.vae,
        "samples": samples,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"cached {len(samples)} samples -> {cache_dir}")


if __name__ == "__main__":
    main()
