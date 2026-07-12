"""Merge a trained LoRA into a Krea 2 base checkpoint (PyTorch, CPU).

Produces a plain full-weight safetensors loadable by the official inference
code, ComfyUI, or any other Krea 2 runtime — useful when a tool doesn't
understand our LoRA key format. Merging into Turbo is the officially
recommended deployment path for LoRAs trained on RAW.

Usage:
    python scripts/merge_lora.py --base turbo.safetensors \
        --lora lora_out/lora.safetensors --output turbo_merged.safetensors \
        [--scale 1.0]
"""

import argparse
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=float, default=None,
                    help="LoRA strength multiplier on top of alpha/rank "
                         "(default 1.0)")
    args = ap.parse_args()

    loras = {}
    with safe_open(args.lora, framework="pt") as f:
        meta = f.metadata() or {}
        for key in f.keys():
            m = re.match(r"^(.+)\.(lora_A|lora_B)\.weight$", key)
            if not m:
                raise SystemExit(f"unrecognized LoRA key: {key}")
            loras.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(key).float()

    alpha = float(meta.get("alpha", 16))
    rank = float(meta.get("rank", 16))
    scale = (alpha / rank) * (args.scale if args.scale is not None else 1.0)
    print(f"merging {len(loras)} LoRA modules (alpha {alpha}, rank {rank}, "
          f"effective scale {scale:.3f})")

    merged = {}
    with safe_open(args.base, framework="pt") as f:
        for key in f.keys():
            w = f.get_tensor(key)
            mod = key[: -len(".weight")] if key.endswith(".weight") else None
            if mod in loras:
                ab = loras.pop(mod)
                delta = scale * (ab["lora_B"] @ ab["lora_A"])
                w = (w.float() + delta).to(w.dtype)
            merged[key] = w

    if loras:
        raise SystemExit(f"LoRA modules not found in base: {sorted(loras)[:5]} ...")
    save_file(merged, args.output, metadata={"merged_lora": args.lora})
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
