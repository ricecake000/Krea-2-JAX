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
f64 accumulator) is in RAM at a time, whatever the number of inputs. Large
tensors are accumulated in element chunks on a thread pool. The core,
average(), takes any objects with the safe_open interface and an optional
concurrent fetch (krea2_jax.r2.R2Safetensors / R2Fetch:
scripts/modal_average_full.py averages snapshots on R2 without a download).

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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from safetensors import safe_open  # noqa: E402

RESOLUTION_KEYS = ("resolution", "img_tokens")


PAR_ELEMS = 1 << 24    # tensors with more elements: chunks on a thread pool
CHUNK_ELEMS = 1 << 20  # elements per chunk (8 MiB of f64)


def _chunks(pool, n, fn):
    """fn(a, b) over [0, n) in CHUNK_ELEMS slices, on `pool` when n is large
    (numpy releases the GIL), inline otherwise."""
    if pool is None or n <= PAR_ELEMS:
        fn(0, n)
        return
    for _ in pool.map(lambda a: fn(a, min(a + CHUNK_ELEMS, n)),
                      range(0, n, CHUNK_ELEMS)):
        pass


def sequential_fetch(handles):
    """Default fetch: key -> the inputs' arrays in input order, read lazily
    one at a time (one input copy in RAM)."""
    def fetch(key):
        for f in handles:
            yield f.get_tensor(key)
    return fetch


class _Mean:
    """Lazy averaged tensor: save_krea2_params reads `.shape` to lay out
    the file and np.asarray() of it once, when writing this tensor.

    acc[e] = sum_i w_i * float64(x_i[e]) in input order, then / sum(w),
    rounded once to float32. w == 1 adds float64(x) directly (x * 1.0 is
    exact) and otherwise the product is computed in float64, so the bits
    equal `acc += x.astype(np.float64) * w`; large tensors are processed in
    element chunks on a thread pool (the per-element order over inputs is
    unchanged)."""

    def __init__(self, key, shape, fetch, weights, pool=None, timing=None,
                 on_tensor=None):
        self.key, self.shape = key, tuple(shape)
        self._fetch, self._weights, self._pool = fetch, weights, pool
        self._timing = timing if timing is not None else {}
        self._on_tensor = on_tensor

    def __array__(self, dtype=None, copy=None):
        t, clock = self._timing, time.perf_counter
        acc = np.zeros(self.shape, np.float64)
        flat = acc.reshape(-1)
        n = flat.size
        it = iter(self._fetch(self.key))
        for w in self._weights:
            t0 = clock()
            x = next(it, None)
            t1 = clock()
            if x is None:
                raise ValueError(f"{self.key}: fetch yielded fewer arrays "
                                 "than weights")
            x = x.reshape(-1)
            if x.size != n:
                raise ValueError(f"{self.key}: input has {x.size} elements, "
                                 f"expected {n}")
            if w == 1.0:
                def add(a, b, x=x):
                    np.add(flat[a:b], x[a:b], out=flat[a:b], dtype=np.float64)
            else:
                def add(a, b, x=x, w=w):
                    np.add(flat[a:b], np.multiply(x[a:b], w, dtype=np.float64),
                           out=flat[a:b])
            _chunks(self._pool, n, add)
            del x, add
            t["fetch_s"] = t.get("fetch_s", 0.0) + t1 - t0
            t["accumulate_s"] = t.get("accumulate_s", 0.0) + clock() - t1
        if next(it, None) is not None:
            raise ValueError(f"{self.key}: fetch yielded more arrays than "
                             "weights")
        t1 = clock()
        total = float(sum(self._weights))
        out = np.empty(self.shape, np.float32)
        oflat = out.reshape(-1)

        def finish(a, b):
            np.divide(flat[a:b], total, out=flat[a:b])
            oflat[a:b] = flat[a:b]          # f64 -> f32, round to nearest
        _chunks(self._pool, n, finish)
        t["accumulate_s"] = t.get("accumulate_s", 0.0) + clock() - t1
        t["tensors"] = t.get("tensors", 0) + 1
        if self._on_tensor is not None:
            self._on_tensor(self.key)
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

    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(p), framework="numpy"))
                   for p in args.inputs]
        res = average(handles, args.inputs, out, weights)
    print(f"wrote {out} ({len(args.inputs)} input(s), bf16, "
          f"{out.stat().st_size} bytes, step {res['metadata']['step']})")


def check_inputs(handles, names, weights):
    """Header checks of opened inputs (keys/shapes equal, all F32, one base)
    -> ({key: shape} of the first input, output metadata). Reads no tensor
    data; raises SystemExit on a mismatch."""
    shapes = [header(f, p) for f, p in zip(handles, names)]
    for p, s in zip(names[1:], shapes[1:]):
        if s != shapes[0]:
            diff = sorted(set(s) ^ set(shapes[0])) or sorted(
                k for k in s if s[k] != shapes[0][k])
            raise SystemExit(f"{p}: keys/shapes differ from "
                             f"{names[0]} (e.g. {diff[:3]})")
    metadata = output_metadata([f.metadata() or {} for f in handles], weights)
    return shapes[0], metadata


def average(handles, names, out, weights, fetch=None, threads=None,
            on_tensor=None, timing=None):
    """Average opened inputs (objects with the safe_open interface: keys,
    metadata, get_slice, get_tensor) into the bf16 file `out` (written
    atomically by save_krea2_params, tensors in sorted key order).
    fetch: callable (key, order) -> iterator of the inputs' arrays of `key` in
    input order, where order is the key sequence the writer will request
    (sorted); default: sequential_fetch(handles), lazy, one input at a time.
    threads: accumulate thread pool size (default: the usable CPUs,
    sched_getaffinity when available, capped at 32). on_tensor(key): called after each tensor is averaged. timing: a
    dict updated live with "tensors", "fetch_s", "accumulate_s".
    -> {"metadata", "tensors", "seconds", "fetch_s", "accumulate_s",
    "write_s"} (write_s: the rest of the time -- bf16 cast and file write)."""
    import jax.numpy as jnp

    from krea2_jax.checkpoint import save_krea2_params

    if len(handles) != len(weights) or len(handles) != len(names):
        raise ValueError("handles, names and weights differ in length")
    shapes, metadata = check_inputs(handles, names, weights)
    order = sorted(shapes)
    if fetch is None:
        get = sequential_fetch(handles)
    else:
        def get(key):
            return fetch(key, order)
    if not threads:
        try:
            threads = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            threads = os.cpu_count() or 1
        threads = min(32, threads)
    timing = {} if timing is None else timing
    timing.update(fetch_s=0.0, accumulate_s=0.0, tensors=0)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        tree = {k: _Mean(k, shapes[k], get, weights, pool=pool,
                         timing=timing, on_tensor=on_tensor) for k in order}
        save_krea2_params(str(out), tree, dtype=jnp.bfloat16,
                          metadata=metadata)
    seconds = time.perf_counter() - t0
    return {"metadata": metadata, "tensors": timing["tensors"],
            "seconds": seconds, "fetch_s": timing["fetch_s"],
            "accumulate_s": timing["accumulate_s"],
            "write_s": seconds - timing["fetch_s"] - timing["accumulate_s"]}

if __name__ == "__main__":
    main()
