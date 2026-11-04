"""Transplant a fine-tune delta onto Krea-2-Turbo:

    out = turbo + alpha * (ft - raw)

per tensor, over three official-layout Krea 2 checkpoints with the same keys
and shapes (Turbo, the Raw base the fine-tune started from, the fine-tune's
raw params, e.g. modal_export_full --dtype f32).

Usage:
    python scripts/build_transplant.py --turbo turbo.safetensors \\
        --raw raw.safetensors --ft run3_9192_params_f32.safetensors \\
        --alpha 0.7 --out turbo_run3_9192_a0.7.safetensors

Streams tensor by tensor (safe_open; only one tensor of each input plus its
float32 working copies in RAM at a time); large tensors are processed in
element chunks on a thread pool. Arithmetic in float32: BF16 inputs are
upcast exactly, the delta ft - raw and alpha * delta are float32 (alpha
rounded to float32), the sum is rounded once to Turbo's stored dtype of that
key (BF16 round to nearest even as ml_dtypes does; F32 stays F32). Where
alpha * delta is zero the Turbo value is kept as is, so -0.0 survives
(-0.0 + 0.0 would give +0.0) and alpha = 0 reproduces Turbo's tensors byte
for byte through the same path.

Checks before anything is written: equal key sets and shapes in the three
headers, supported dtypes (BF16 / F32), then one streamed pass that every
input value is finite (check_inputs; build(checked=True) skips the pass when
the caller ran it, e.g. once for several alphas). Every output value is
checked finite as well. The file is written by
krea2_jax.checkpoint.save_krea2_params (sorted keys, deterministic header,
atomic tmp + fsync + rename) with per-key dtypes and the metadata
{source "turbo-transplant", alpha, turbo, raw, ft}.

Runs on CPU (JAX_PLATFORMS=cpu unless already set).
"""

import argparse
import contextlib
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
from safetensors import safe_open  # noqa: E402

ST_NP = {"BF16": np.dtype(ml_dtypes.bfloat16), "F32": np.dtype(np.float32)}
INPUTS = ("turbo", "raw", "ft")
PAR_ELEMS = 1 << 24    # tensors with more elements: chunks on a thread pool
CHUNK_ELEMS = 1 << 22  # elements per chunk (16 MiB of f32)
PROGRESS_S = 30.0


def default_threads():
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, min(32, n))


def _chunks(pool, n, fn):
    """fn(a, b) over [0, n) in CHUNK_ELEMS slices, on `pool` when n is large
    (numpy releases the GIL), inline otherwise."""
    if pool is None or n <= PAR_ELEMS:
        fn(0, n)
        return
    for _ in pool.map(lambda a: fn(a, min(a + CHUNK_ELEMS, n)),
                      range(0, n, CHUNK_ELEMS)):
        pass


class _Every:
    """A log line at most every `every` s (and for the last item)."""

    def __init__(self, n, what, log=print, every=PROGRESS_S):
        self.n, self.what, self.log, self.every = n, what, log, every
        self.t0 = self.last = time.time()
        self.done, self.bytes = 0, 0

    def __call__(self, nbytes):
        self.done += 1
        self.bytes += nbytes
        now = time.time()
        if now - self.last >= self.every or self.done == self.n:
            self.last = now
            dt = max(now - self.t0, 1e-6)
            self.log(f"{self.what}: tensors {self.done}/{self.n}, "
                     f"{self.bytes / 2**30:.1f} GiB, "
                     f"{self.bytes / 2**20 / dt:.0f} MiB/s, {dt:.0f} s")


def check_alpha(alpha):
    a = float(alpha)
    if not math.isfinite(a):
        raise ValueError(f"alpha must be finite, got {alpha!r}")
    return 0.0 if a == 0 else a     # -0.0 -> 0.0


def format_alpha(alpha):
    """'0.4', '1.0', '0.0': the metadata value and the R2 key suffix."""
    return repr(check_alpha(alpha))


def check_headers(handles, names=INPUTS):
    """Header checks of opened (turbo, raw, ft) -> {key: (shape, Turbo's
    dtype name)}. Reads no tensor data; ValueError on a mismatch."""
    heads = []
    for f, name in zip(handles, names):
        h, bad = {}, []
        for k in f.keys():
            sl = f.get_slice(k)
            h[k] = (tuple(sl.get_shape()), sl.get_dtype())
            if h[k][1] not in ST_NP:
                bad.append(f"{k}:{h[k][1]}")
        if bad:
            raise ValueError(f"{name}: unsupported dtypes (BF16 / F32 only), "
                             f"e.g. {bad[:3]}")
        heads.append(h)
    for name, h in zip(names[1:], heads[1:]):
        if set(h) != set(heads[0]):
            diff = sorted(set(h) ^ set(heads[0]))
            raise ValueError(f"{name}: key set differs from {names[0]} "
                             f"({len(diff)} keys, e.g. {diff[:3]})")
        bad = sorted(k for k in h if h[k][0] != heads[0][k][0])
        if bad:
            raise ValueError(f"{name}: shapes differ from {names[0]} for "
                             f"{len(bad)} keys, e.g. {bad[0]} "
                             f"{h[bad[0]][0]} vs {heads[0][bad[0]][0]}")
    if not heads[0]:
        raise ValueError(f"{names[0]}: no tensors")
    return heads[0]


def _nbytes(shape, dtype_name):
    return math.prod(shape) * ST_NP[dtype_name].itemsize


def check_finite(handles, names=INPUTS, pool=None, log=print,
                 every=PROGRESS_S):
    """One streamed pass: every value of every input is finite, else
    ValueError naming the input and key."""
    for f, name in zip(handles, names):
        keys = sorted(f.keys())
        prog = _Every(len(keys), f"finite check {name}", log, every)
        for k in keys:
            x = f.get_tensor(k).reshape(-1)

            def chk(a, b, x=x, k=k):
                if not np.isfinite(x[a:b]).all():
                    raise ValueError(f"{name}: {k} has non-finite values")
            _chunks(pool, x.size, chk)
            prog(x.nbytes)
            del x, chk


class _Transplant:
    """Lazy output tensor: save_krea2_params reads `.shape` to lay out the
    file and np.asarray() of it once, when writing this tensor."""

    def __init__(self, key, shape, out_dtype, handles, alpha, pool=None,
                 on_tensor=None):
        self.key, self.shape = key, tuple(shape)
        self.out_dtype = np.dtype(out_dtype)
        self._handles, self._pool, self._on_tensor = handles, pool, on_tensor
        self._alpha = np.float32(alpha)

    def __array__(self, dtype=None, copy=None):
        t, r, f = (h.get_tensor(self.key).reshape(-1) for h in self._handles)
        out = np.empty(t.size, self.out_dtype)
        alpha, key = self._alpha, self.key

        def work(a, b):
            t32 = t[a:b].astype(np.float32)
            term = f[a:b].astype(np.float32)
            np.subtract(term, r[a:b].astype(np.float32), out=term)
            np.multiply(term, alpha, out=term)
            res = t32 + term
            np.copyto(res, t32, where=(term == 0))   # keeps -0.0
            out[a:b] = res                  # BF16: round to nearest even
            if not np.isfinite(out[a:b]).all():
                raise ValueError(f"{key}: non-finite output values")
        _chunks(self._pool, t.size, work)
        if self._on_tensor is not None:
            self._on_tensor(out.nbytes)
        out = out.reshape(self.shape)
        return out if dtype is None else out.astype(dtype, copy=False)


def default_labels(paths):
    """{turbo, raw, ft}: '<path> size <n>' (+ ' step <s>' from the ft
    metadata when it records one)."""
    labels = {n: f"{p} size {Path(p).stat().st_size}"
              for n, p in zip(INPUTS, paths)}
    with safe_open(str(paths[2]), framework="numpy") as f:
        step = (f.metadata() or {}).get("step")
    if step is not None:
        labels["ft"] += f" step {step}"
    return labels


def transplant_metadata(alpha, labels):
    return {"source": "turbo-transplant", "alpha": format_alpha(alpha),
            **{n: str(labels[n]) for n in INPUTS}}


@contextlib.contextmanager
def _opened(paths):
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(safe_open(str(p), framework="numpy"))
               for p in paths]


def check_inputs(turbo, raw, ft, threads=None, log=print, every=PROGRESS_S):
    """Headers + the finite pass over the three files -> {key: (shape,
    Turbo's dtype name)}; ValueError on any problem."""
    with _opened((turbo, raw, ft)) as hs, \
            ThreadPoolExecutor(threads or default_threads()) as pool:
        spec = check_headers(hs)
        log(f"headers ok: {len(spec)} tensors, keys and shapes equal")
        check_finite(hs, pool=pool, log=log, every=every)
    return spec


def build(turbo, raw, ft, alpha, out, labels=None, checked=False,
          threads=None, log=print, every=PROGRESS_S):
    """Write turbo + alpha * (ft - raw) to `out` (module docstring).
    checked: the caller already ran check_inputs on these files (skip the
    finite pass). labels: {turbo, raw, ft} metadata strings (default
    default_labels). -> {"metadata", "tensors", "bytes", "seconds"}."""
    alpha = check_alpha(alpha)
    paths = (turbo, raw, ft)
    out = Path(out).resolve()
    if any(Path(p).resolve() == out for p in paths):
        raise ValueError(f"--out {out} is one of the inputs")
    metadata = transplant_metadata(
        alpha, labels if labels is not None else default_labels(paths))
    from krea2_jax.checkpoint import save_krea2_params

    t0 = time.time()
    with _opened(paths) as hs, \
            ThreadPoolExecutor(threads or default_threads()) as pool:
        spec = check_headers(hs)
        if not checked:
            check_finite(hs, pool=pool, log=log, every=every)
        order = sorted(spec)
        prog = _Every(len(order), f"build a={metadata['alpha']}", log, every)
        tree = {k: _Transplant(k, spec[k][0], ST_NP[spec[k][1]], hs, alpha,
                               pool=pool, on_tensor=prog) for k in order}
        save_krea2_params(str(out), tree, dtype=np.float32, metadata=metadata,
                          key_dtypes={k: ST_NP[spec[k][1]] for k in order})
    return {"metadata": metadata, "tensors": len(order),
            "bytes": out.stat().st_size, "seconds": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for n in INPUTS:
        ap.add_argument(f"--{n}", required=True,
                        help=f"{n} .safetensors (official layout)")
    ap.add_argument("--alpha", type=float, required=True,
                    help="scale of the delta ft - raw")
    ap.add_argument("--out", required=True, help="output .safetensors")
    ap.add_argument("--threads", type=int, default=0,
                    help="worker threads (default: usable CPUs, max 32)")
    args = ap.parse_args(argv)
    try:
        res = build(args.turbo, args.raw, args.ft, args.alpha, args.out,
                    threads=args.threads or None)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"wrote {args.out} ({res['tensors']} tensors, {res['bytes']} "
          f"bytes, {res['seconds']:.0f} s, alpha {res['metadata']['alpha']})")


if __name__ == "__main__":
    main()
