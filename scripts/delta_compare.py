"""Direction of weight deltas against one base, over every tensor of a Krea 2
DiT: how does Δi = W_i - W_base relate to Δj = W_j - W_base?

    python scripts/delta_compare.py --base raw.safetensors \\
        --delta turbo=turbo.safetensors --delta run3=run3_f32.safetensors \\
        [--delta run2=run2.safetensors] --out report.json

The first --delta is the reference axis (e.g. turbo): projections and cos^2
are reported for every other delta onto it (and every pair both ways).

Every file is streamed tensor by tensor and, within a tensor, in row chunks
along axis 0 (safe_open get_slice, float64 sums), so memory stays at a few
chunks per file whatever the tensor size. Per tensor: ||W_base||^2 (b2),
||Δi||^2 (d2), Δi·Δj (dot) for every pair, plus derived rel_i = ||Δi|| /
||W_base||, cos_ij, ||Δi + Δj||, ||Δi - Δj|| (from d2_i + d2_j +- 2 dot).

Aggregates are energy-weighted global ratios over a set of tensors
(cos = Σdot / sqrt(ΣΔi² ΣΔj²), never a mean of cosines): ALL, each tensor
kind, each block group (block00..27, txtfusion, other) and each storage dtype
of the base tensor (BF16 / F32), with the per-tensor cos median / p10 / p90
and the random-direction noise floor 1/sqrt(n) (n = elements in the set).
For each pair: cos^2 (the fraction of Δj's energy along Δi) and the
projection coefficients dot / ||Δi||^2 (of Δj on Δi) and dot / ||Δj||^2.

Validation: keys missing / extra in a delta file, shape mismatches, storage
dtype differences vs the base and non-finite tensors are listed; a tensor
enters the per-tensor rows and aggregates only when every file has it with
the base's shape and all its values are finite.

Importable without numpy / safetensors (they load inside the functions):
scripts/modal_delta_compare.py runs delta_compare remotely and format_tables
in the Modal CLI's Python.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
for _repo in (_REPO, "/repo"):
    if (_repo not in sys.path
            and os.path.isfile(os.path.join(_repo, "krea2_jax",
                                            "tensor_groups.py"))):
        sys.path.insert(0, _repo)

from krea2_jax.tensor_groups import KINDS, tensor_group, tensor_kind  # noqa: E402

LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CHUNK = 1 << 23          # elements per row chunk (64 MiB as float64)
LOG_EVERY_S = 30.0
BF16_NOTE = (
    "{label}: {n} tensors are stored BF16 in this file where the base is "
    "F32; their delta includes the bf16 rounding of the stored value (up to "
    "~0.4 % relative per element, comparable to the true change of the F32 "
    "tensors), so its rel / cos on the F32 split are unreliable")


def pair_name(a, b):
    return f"{a}|{b}"


def parse_delta(text):
    """'LABEL=PATH' -> (label, path)."""
    label, sep, path = text.partition("=")
    if not sep or not LABEL_RE.match(label) or not path:
        raise ValueError(f"--delta must be LABEL=PATH (label "
                         f"{LABEL_RE.pattern}), got {text!r}")
    return label, path


def _sqrt(x):
    return math.sqrt(x) if x is not None and x >= 0 else None


def _div(a, b):
    return a / b if b else None


def _quantile(vals, q):
    """Linear-interpolated quantile (numpy's default) of a sorted list."""
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _row_chunks(shape, chunk):
    """[(start, stop)] row ranges along axis 0 of about `chunk` elements
    each; [None] for a 0-d tensor."""
    if not shape:
        return [None]
    per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows = max(1, chunk // max(per_row, 1))
    return [(a, min(a + rows, shape[0])) for a in range(0, shape[0], rows)]


def _tensor_sums(base_sl, delta_sls, shape, chunk):
    """float64 sums over one tensor: (b2, [d2_i], {(i, j): dot_ij},
    [finite_i]) for the base slice and the delta files' slices; b2 is
    non-finite when the base tensor is."""
    import numpy as np

    n = len(delta_sls)
    b2, d2, dot = 0.0, [0.0] * n, {}
    finite = [True] * n
    for rng in _row_chunks(shape, chunk):
        def read(sl):
            a = sl[...] if rng is None else sl[rng[0]:rng[1]]
            return np.asarray(a).reshape(-1).astype(np.float64)

        bb = read(base_sl)
        b2 += float(np.dot(bb, bb))
        ds = []
        for i, sl in enumerate(delta_sls):
            d = read(sl)
            d -= bb
            if not np.isfinite(d).all():
                finite[i] = False
            ds.append(d)
        del bb
        for i in range(n):
            d2[i] += float(np.dot(ds[i], ds[i]))
            for j in range(i + 1, n):
                dot[(i, j)] = dot.get((i, j), 0.0) + float(np.dot(ds[i], ds[j]))
        del ds
    return b2, d2, dot, finite


def tensor_row(key, numel, dtype, dtypes, labels, b2, d2, dot):
    """One per-tensor row: raw sums plus rel / cos / sum and diff norms."""
    row = {"key": key, "kind": tensor_kind(key), "group": tensor_group(key),
           "dtype": dtype, "dtypes": dtypes, "numel": numel, "b2": b2,
           "d2": dict(zip(labels, d2)), "dot": {}, "rel": {}, "cos": {},
           "sum_norm": {}, "diff_norm": {}}
    for i, a in enumerate(labels):
        row["rel"][a] = _sqrt(_div(d2[i], b2))
    for (i, j), v in sorted(dot.items()):
        p = pair_name(labels[i], labels[j])
        row["dot"][p] = v
        row["cos"][p] = _div(v, math.sqrt(d2[i] * d2[j]))
        row["sum_norm"][p] = _sqrt(d2[i] + d2[j] + 2 * v)
        row["diff_norm"][p] = _sqrt(d2[i] + d2[j] - 2 * v)
    return row


def aggregate(rows, labels):
    """Energy-weighted global ratios over rows (see the module docstring)."""
    numel = sum(r["numel"] for r in rows)
    b2 = sum(r["b2"] for r in rows)
    d2 = {a: sum(r["d2"][a] for r in rows) for a in labels}
    out = {"n_tensors": len(rows), "numel": numel,
           "noise_floor": 1 / math.sqrt(numel) if numel else None,
           "base_norm": math.sqrt(b2),
           "rel": {a: _sqrt(_div(d2[a], b2)) for a in labels},
           "delta_norm": {a: math.sqrt(d2[a]) for a in labels}, "pairs": {}}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            p = pair_name(a, b)
            dot = sum(r["dot"][p] for r in rows)
            cos = _div(dot, math.sqrt(d2[a] * d2[b]))
            per = sorted(r["cos"][p] for r in rows if r["cos"][p] is not None)
            out["pairs"][p] = {
                "dot": dot, "cos": cos,
                "cos2": None if cos is None else cos * cos,
                "proj_on_first": _div(dot, d2[a]),    # Δb ≈ c · Δa
                "proj_on_second": _div(dot, d2[b]),   # Δa ≈ c · Δb
                "sum_rel": _div(_sqrt(d2[a] + d2[b] + 2 * dot),
                                math.sqrt(b2)),
                "diff_rel": _div(_sqrt(d2[a] + d2[b] - 2 * dot),
                                 math.sqrt(b2)),
                "cos_median": _quantile(per, 0.5),
                "cos_p10": _quantile(per, 0.1),
                "cos_p90": _quantile(per, 0.9),
            }
    return out


def aggregates(rows, labels):
    """{"all", "by_kind", "by_group", "by_dtype"} of aggregate()."""
    def by(field, order=None):
        sets = {}
        for r in rows:
            sets.setdefault(r[field], []).append(r)
        names = [n for n in order if n in sets] if order else sorted(sets)
        return {n: aggregate(sets[n], labels) for n in names}

    return {"all": aggregate(rows, labels), "by_kind": by("kind", KINDS),
            "by_group": by("group"), "by_dtype": by("dtype")}


def delta_compare(base_path, deltas, chunk=CHUNK, log=print,
                  log_every=LOG_EVERY_S):
    """deltas: [(label, path)] (>= 2; the first is the reference axis).
    -> report dict (module docstring)."""
    import importlib
    from contextlib import ExitStack

    from safetensors import safe_open

    importlib.import_module("ml_dtypes")   # numpy 'bfloat16' for safe_open

    labels = [a for a, _ in deltas]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError(f"need >= 2 distinct delta labels, got {labels}")
    t0 = last = time.time()
    with ExitStack() as stack:
        base = stack.enter_context(safe_open(str(base_path), framework="numpy"))
        hs = [stack.enter_context(safe_open(str(p), framework="numpy"))
              for _, p in deltas]
        base_keys = sorted(base.keys())
        files = [{"label": a, "path": str(p), "keys": set(h.keys()),
                  "shape_mismatch": [], "dtype_mismatch": [],
                  "nonfinite": []} for (a, p), h in zip(deltas, hs)]
        rows, skipped, base_nonfinite, elems = [], [], [], 0
        for n, key in enumerate(base_keys):
            bsl = base.get_slice(key)
            shape, dtype = tuple(bsl.get_shape()), bsl.get_dtype()
            sls, dtypes, ok = [], {}, True
            for h, f in zip(hs, files):
                if key not in f["keys"]:
                    ok = False
                    continue
                sl = h.get_slice(key)
                dtypes[f["label"]] = sl.get_dtype()
                if tuple(sl.get_shape()) != shape:
                    f["shape_mismatch"].append(
                        [key, list(sl.get_shape()), list(shape)])
                    ok = False
                if sl.get_dtype() != dtype:
                    f["dtype_mismatch"].append([key, sl.get_dtype(), dtype])
                sls.append(sl)
            if not ok:
                skipped.append(key)
                continue
            b2, d2, dot, finite = _tensor_sums(bsl, sls, shape, chunk)
            numel = math.prod(shape)
            elems += numel
            for f, fin in zip(files, finite):
                if not fin:
                    f["nonfinite"].append(key)
            if not math.isfinite(b2):
                base_nonfinite.append(key)
            if not all(finite) or not math.isfinite(b2):
                skipped.append(key)
            else:
                rows.append(tensor_row(key, numel, dtype, dtypes, labels,
                                       b2, d2, dot))
            if time.time() - last >= log_every:
                last = time.time()
                log(f"delta: {n + 1}/{len(base_keys)} tensors, "
                    f"{elems / 1e9:.2f} G elements per file, "
                    f"{last - t0:.0f} s")
    notes = []
    for f in files:
        f["n_tensors"] = len(rows)
        f["missing"] = sorted(set(base_keys) - f["keys"])
        f["extra"] = sorted(f.pop("keys") - set(base_keys))
        n16 = sum(1 for _, got, want in f["dtype_mismatch"]
                  if got == "BF16" and want == "F32")
        if n16:
            notes.append(BF16_NOTE.format(label=f["label"], n=n16))
        n32 = sum(1 for _, got, want in f["dtype_mismatch"]
                  if got == "F32" and want == "BF16")
        if n32:
            notes.append(f"{f['label']}: {n32} tensors are stored F32 where "
                         "the base is BF16 (e.g. an fp32 export): expected, "
                         "their deltas are exact")
    log(f"delta: {len(rows)}/{len(base_keys)} tensors compared, "
        f"{len(skipped)} skipped, {time.time() - t0:.0f} s")
    if not rows:
        raise RuntimeError("no tensor is present in every file with the "
                           "base's shape and finite values: "
                           + json.dumps([{k: len(f[k]) for k in (
                               "missing", "shape_mismatch", "nonfinite")}
                               for f in files]))
    return {"base": str(base_path), "ref": labels[0], "labels": labels,
            "pairs": [pair_name(a, b) for i, a in enumerate(labels)
                      for b in labels[i + 1:]],
            "files": files, "n_base_tensors": len(base_keys),
            "skipped": skipped, "base_nonfinite": base_nonfinite,
            "notes": notes,
            "aggregates": aggregates(rows, labels), "tensors": rows,
            "seconds": time.time() - t0}


# ------------------------------------------------------------------ tables
def _pct(x):
    return "-" if x is None else f"{100 * x:.3f}%"


def _cos(x):
    return "-" if x is None else f"{x:+.4f}"


def _num(x):
    return "-" if x is None else f"{x:+.3e}"


def format_tables(report):
    """Compact text tables of a delta_compare report (pure Python)."""
    labels, pairs = report["labels"], report["pairs"]
    agg = report["aggregates"]
    lines = [f"delta directions vs {report['base']} (reference "
             f"{report['ref']}); {len(report['tensors'])}/"
             f"{report['n_base_tensors']} tensors compared"]
    for f in report["files"]:
        probs = [f"{len(f[k])} {k.replace('_', ' ')}"
                 for k in ("missing", "extra", "shape_mismatch",
                           "dtype_mismatch", "nonfinite") if f[k]]
        lines.append(f"  {f['label']}: {f['path']}"
                     + (f" ({', '.join(probs)})" if probs else ""))
        for k in ("missing", "extra", "shape_mismatch", "nonfinite"):
            for item in f[k][:5]:
                lines.append(f"    {k}: {item}")
    if report["base_nonfinite"]:
        lines.append(f"  base non-finite: {report['base_nonfinite'][:5]}")
    lines += [f"  note: {n}" for n in report["notes"]]

    def table(title, sets):
        head = (f"{title:<12}{'tensors':>8}{'floor':>9}"
                + "".join(f"{'rel ' + a:>12}" for a in labels)
                + "".join(f"{'cos ' + p:>16}" for p in pairs))
        out = [head]
        for name, a in sets.items():
            out.append(f"{name:<12}{a['n_tensors']:>8}"
                       f"{a['noise_floor']:>9.1e}"
                       + "".join(f"{_pct(a['rel'][x]):>12}" for x in labels)
                       + "".join(f"{_cos(a['pairs'][p]['cos']):>16}"
                                 for p in pairs))
        return out

    top = {"ALL": agg["all"], **agg["by_dtype"]}
    lines += [""] + table("set", top)
    lines += ["", "per pair (ALL / by dtype): cos^2 = fraction of the second "
              "delta's energy along the first; proj = dot / ||first||^2"]
    lines.append(f"{'pair':<16}{'set':<6}{'cos2':>10}{'proj':>11}"
                 f"{'proj_rev':>11}{'rel(a+b)':>10}{'rel(a-b)':>10}"
                 f"{'cos med':>9}{'p10':>9}{'p90':>9}")
    for p in pairs:
        for name, a in top.items():
            q = a["pairs"][p]
            lines.append(
                f"{p:<16}{name:<6}{_num(q['cos2']):>10}"
                f"{_num(q['proj_on_first']):>11}"
                f"{_num(q['proj_on_second']):>11}{_pct(q['sum_rel']):>10}"
                f"{_pct(q['diff_rel']):>10}{_cos(q['cos_median']):>9}"
                f"{_cos(q['cos_p10']):>9}{_cos(q['cos_p90']):>9}")
    lines += [""] + table("kind", agg["by_kind"])
    lines += [""] + table("group", agg["by_group"])
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base .safetensors")
    ap.add_argument("--delta", action="append", default=[],
                    metavar="LABEL=PATH",
                    help="a .safetensors to diff against --base; >= 2, the "
                         "first is the reference axis")
    ap.add_argument("--out", required=True, help="report JSON path")
    args = ap.parse_args(argv)
    try:
        deltas = [parse_delta(d) for d in args.delta]
    except ValueError as e:
        ap.error(str(e))
    if len(deltas) < 2 or len({a for a, _ in deltas}) != len(deltas):
        ap.error("need >= 2 --delta with distinct labels")
    for p in [args.base] + [p for _, p in deltas]:
        if not os.path.isfile(p):
            ap.error(f"not a file: {p}")
    report = delta_compare(args.base, deltas)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(format_tables(report))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
