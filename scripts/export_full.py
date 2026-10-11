"""Export full fine-tuned weights from an orbax train-state checkpoint.

Restores one tree (EMA or raw params) from `<output-dir>/ckpt/<step>/` on the
host CPU — no accelerator or matching mesh needed — and writes it as an
official-layout Krea 2 safetensors (per-layer keys, bf16), loadable by
scripts/sample.py --base-ckpt, the official PyTorch code and ComfyUI.

Usage:
    python scripts/export_full.py --ckpt-dir full_out/ckpt \
        --output model.safetensors [--step N] [--source ema|params] \
        [--base-ckpt raw.safetensors]

Runs on CPU by default (JAX_PLATFORMS=cpu unless already set), so it can run
next to a live training job without touching its accelerators. Host RAM
needed: about one fp32 copy of the chosen tree (~52 GB for the 12.9B DiT).
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_against_base(tree, base_path):
    """The keys and shapes the export would write (official_key_shapes of the
    restored tree) must equal the base checkpoint's. Runs before anything is
    written, so a rejected export never touches the destination."""
    from safetensors import safe_open

    from krea2_jax.checkpoint import official_key_shapes

    got = official_key_shapes(tree)
    try:
        with safe_open(base_path, framework="numpy") as f:
            want = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    except Exception as e:
        raise SystemExit(f"cannot read --base-ckpt {base_path}: {e}")
    if got != want:
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        bad = sorted(k for k in set(got) & set(want) if got[k] != want[k])
        raise SystemExit(
            f"exported keys do not match {base_path}: missing {missing[:5]}, "
            f"unexpected {extra[:5]}, shape mismatch "
            f"{[(k, got[k], want[k]) for k in bad[:5]]}; nothing written"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt-dir", required=True,
                    help="orbax checkpoint dir (<train output-dir>/ckpt)")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to export (default: newest)")
    ap.add_argument("--output", required=True, help="output .safetensors path")
    ap.add_argument("--source", choices=("ema", "params"), default=None,
                    help="which weights to export (default: ema when the "
                         "checkpoint has one, else params)")
    ap.add_argument("--base-ckpt", default=None,
                    help="optional base checkpoint: verify (before writing) "
                         "that the export has exactly its keys and shapes")
    args = ap.parse_args()

    import jax.numpy as jnp

    from krea2_jax.checkpoint import save_krea2_params
    from krea2_jax.fullft_ckpt import restore_tree_on_host

    print(f"restoring {args.source or 'ema/params'} from {args.ckpt_dir} "
          f"step {args.step if args.step is not None else 'latest'}")
    try:
        tree, meta = restore_tree_on_host(args.ckpt_dir, args.step, item=args.source)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))
    source = meta["item"]
    print(f"restored {source} of step {meta['step']}")
    metadata = {"step": meta["step"], "source": source,
                "format": "krea2-full-ft"}
    if args.base_ckpt:
        check_against_base(tree, args.base_ckpt)
        print(f"keys and shapes match {args.base_ckpt}")
        metadata["base"] = os.path.basename(args.base_ckpt)
    save_krea2_params(args.output, tree, dtype=jnp.bfloat16, metadata=metadata)
    del tree
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
