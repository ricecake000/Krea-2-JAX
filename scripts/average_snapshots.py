"""Average fp32 raw-param snapshots of scripts/train_full.py (--snapshot-steps)
into one bf16 official-layout Krea 2 checkpoint (post-hoc weight averaging).

Usage:
    python scripts/average_snapshots.py \
        full_out/snapshots/model_step007700_f32.safetensors \
        full_out/snapshots/model_step008000_f32.safetensors ... \
        --out avg.safetensors [--weights 1,1,2]

Every input must hold the same keys and shapes, all F32, from the same base
(metadata "base"); nothing is written otherwise. Streams tensor by tensor:
the inputs are memory-mapped and only one tensor (one input's copy plus an
f64 accumulator) is in RAM at a time, whatever the number of inputs.

Arithmetic: mean = (sum_i w_i * x_i) / (sum_i w_i), accumulated in float64
(--weights default all 1), rounded once to float32 (the averaged fp32 model),
then written by krea2_jax.checkpoint.save_krea2_params with dtype bf16 and
train_full.export_metadata — the same writer, key layout, f32 -> bf16 cast
(round to nearest even) and metadata code as the trainer's bf16 export, so
the output is a drop-in replacement for model.safetensors. Equivalence:
identical inputs average to exactly the input (the f64 sum of N equal f32
values and its division by N are exact), so the tensor data equals the
trainer's bf16 export of those params byte for byte (numpy/ml_dtypes and
XLA convert f32 -> bf16 identically). A single input is a pure f32 -> bf16
conversion and keeps the input's metadata (step, base, resolution, source
"params"); several inputs record the highest input step as "step", source
"params-avg", and "avg_steps" / "avg_weights" (normalized).

Runs on CPU (JAX_PLATFORMS=cpu unless already set); writes atomically.
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from safetensors import safe_open  # noqa: E402

RESOLUTION_KEYS = ("resolution", "img_tokens")


class _Mean:
    """Lazy averaged tensor: save_krea2_params reads `.shape` to lay out
    the file and np.asarray() of it once, when writing this tensor."""

    def __init__(self, key, shape, handles, weights):
        self.key, self.shape = key, tuple(shape)
        self._handles, self._weights = handles, weights

    def __array__(self, dtype=None, copy=None):
        acc = np.zeros(self.shape, np.float64)
        for f, w in zip(self._handles, self._weights):
            x = f.get_tensor(self.key)
            acc += x.astype(np.float64) * w
            del x
        acc /= float(sum(self._weights))
        out = acc.astype(np.float32)
        return out if dtype is None else out.astype(dtype, copy=False)


def parse_weights(text, n):
    if text is None:
        return [1.0] * n
    try:
        w = [float(x) for x in text.split(",")]
    except ValueError:
        w = []
    if len(w) != n or any(not np.isfinite(x) or x < 0 for x in w) or sum(w) <= 0:
        raise SystemExit(f"--weights must be {n} non-negative numbers with a "
                         f"positive sum, got {text!r}")
    return w


def header(f, path):
    """{key: shape} of an input; raises SystemExit unless every tensor is F32."""
    shapes, bad = {}, []
    for k in f.keys():
        sl = f.get_slice(k)
        shapes[k] = tuple(sl.get_shape())
        if sl.get_dtype() != "F32":
            bad.append(f"{k}:{sl.get_dtype()}")
    if bad:
        raise SystemExit(f"{path}: not an fp32 snapshot ({len(bad)} non-F32 "
                         f"tensors, e.g. {bad[:3]})")
    return shapes


def output_metadata(metas, weights):
    """The output __metadata__ (train_full.export_metadata)."""
    from train_full import export_metadata

    bases = {m.get("base") for m in metas}
    if len(bases) != 1:
        raise SystemExit(f"inputs come from different bases: {sorted(map(str, bases))}")
    try:
        steps = [int(m["step"]) for m in metas]
    except (KeyError, ValueError) as e:
        raise SystemExit(f"an input records no integer step: {e!r}")
    last = metas[int(np.argmax(steps))]
    resolution = {k: last[k] for k in RESOLUTION_KEYS if k in last}
    if len(metas) == 1:
        return export_metadata(steps[0], last.get("base"), resolution,
                               last.get("source", "params"))
    total = float(sum(weights))
    return {**export_metadata(max(steps), last.get("base"), resolution,
                              "params-avg"),
            "avg_steps": ",".join(map(str, steps)),
            "avg_weights": ",".join(repr(w / total) for w in weights)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="fp32 snapshot .safetensors files")
    ap.add_argument("--out", required=True, help="output bf16 .safetensors")
    ap.add_argument("--weights", default=None,
                    help="comma-separated non-negative weights, one per "
                         "input (normalized; default: uniform)")
    args = ap.parse_args(argv)
    weights = parse_weights(args.weights, len(args.inputs))
    out = Path(args.out).resolve()
    if any(Path(p).resolve() == out for p in args.inputs):
        raise SystemExit(f"--out {out} is one of the inputs")

    import jax.numpy as jnp

    from krea2_jax.checkpoint import save_krea2_params

    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(p), framework="numpy"))
                   for p in args.inputs]
        shapes = [header(f, p) for f, p in zip(handles, args.inputs)]
        for p, s in zip(args.inputs[1:], shapes[1:]):
            if s != shapes[0]:
                diff = sorted(set(s) ^ set(shapes[0])) or sorted(
                    k for k in s if s[k] != shapes[0][k])
                raise SystemExit(f"{p}: keys/shapes differ from "
                                 f"{args.inputs[0]} (e.g. {diff[:3]})")
        metadata = output_metadata([f.metadata() or {} for f in handles],
                                   weights)
        tree = {k: _Mean(k, shp, handles, weights)
                for k, shp in shapes[0].items()}
        save_krea2_params(str(out), tree, dtype=jnp.bfloat16, metadata=metadata)
    print(f"wrote {out} ({len(args.inputs)} input(s), bf16, "
          f"{out.stat().st_size} bytes, step {metadata['step']})")


if __name__ == "__main__":
    main()
