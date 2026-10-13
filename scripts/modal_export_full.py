"""Modal app (CPU only): export one tree (raw params or EMA) of a full
fine-tuning train-state checkpoint on R2 as an official bf16 safetensors,
upload it next to the run on R2, and report how far exported weights moved
from the base (relative weight drift).

    modal run scripts/modal_export_full.py --run run1 --step 1513 --source params
    modal run scripts/modal_export_full.py --run run1 --step 1513 \\
        --compare /vol/ft/run1/model.safetensors      # + drift of the R2 EMA
    modal run scripts/modal_export_full.py --run run1 --step 1513 --drift-only \\
        --compare /vol/ft/run1/model.safetensors      # no download / export
    python scripts/modal_export_full.py --help

Options (local entrypoint, argparse):
  --run NAME        run directory on R2 (default run1)
  --step N          checkpoint step (required)
  --source S        params (default) or ema: the tree to export
  --compare PATH    another safetensors on the volume (/vol/...) to include in
                    the drift report, e.g. /vol/ft/run1/model.safetensors (the
                    R2 EMA export cached by scripts/modal_eval_full.py);
                    repeatable
  --drift-only      no download / export / upload: only the drift report of
                    the existing export (if any) and the --compare files
  --force           export and upload again even if a matching export exists
                    on the volume / a matching object exists on R2
  --prefix P        R2 prefix of the runs (default krea2-fullft)
  --out PATH        local drift JSON (default
                    /home/mj/krea2-ops/eval/drift_<run>_<step>.json)

One function (`export_full_remote`, EXPORT_CPU cores, EXPORT_MEM_GIB GiB:
scripts/export_full.py holds the fp32 tree, ~48 GiB for the 12.9B DiT, plus
the largest leaf's transient copies) on the Volume `krea2-eval-cache` of
scripts/modal_eval_full.py, with the `krea2-r2` secret:
  0. list <prefix>/<run>/ckpt/<step>/ (always): the checkpoint fingerprint
     is the sha256 of the selected objects' sorted (key, size, ETag). An
     existing export is reused only if its sidecar
     <export>.src.json records the same (bucket sha256, prefix, run, step,
     source, fingerprint) and the file's size; otherwise:
  1. download every object of <prefix>/<run>/ckpt/<step>/ that restoring ONE tree
     needs -- the top-level files (_CHECKPOINT_METADATA, commit marker),
     <source>/ and meta/ (never opt_state/ or the other tree) -- into
     /vol/ckpt/<run>/<source>/<step>/ (an orbax dir of its own per tree: a
     partial copy of another tree in the same step dir breaks orbax's
     metadata read). Objects download in parallel (boto3 multipart,
     max_concurrency 32), each is size-checked and its ETag re-checked after
     the download; /vol/ckpt/<run>/markers/<source>_<step>.json records the
     (ETag, size) of every complete object, so a re-run downloads only
     missing, changed or truncated objects and deletes local files the
     listing no longer has.
  2. scripts/export_full.py --ckpt-dir /vol/ckpt/<run>/<source> --step N
     --source S --base-ckpt /vol/base/raw.safetensors (cached by
     modal_eval_full's prepare; keys/shapes checked before writing) ->
     /vol/exports/<run>_<step>_<source>.safetensors (bf16, metadata step /
     source), then the sidecar (identity + size + the file's sha256).
  3. upload to <prefix>/<run>/exports/model_step<step>_<source>.safetensors
     with user metadata {ckpt-fingerprint, sha256}; skipped only when the
     object has this size and exactly this metadata (unless --force), any
     other object there is overwritten; size and metadata re-checked after
     the upload.
  4. drift report: the base and every compared file are streamed tensor by
     tensor (safe_open, float64 sums): per tensor ||W - W_base|| /
     ||W_base||; per file the global drift sqrt(sum ||dW||^2 / sum
     ||W_base||^2), median / p90 / max over tensors, and the same global
     ratio per DiT block (block00..block27, txtfusion, other) and per tensor
     kind (attn_q/k/v/o, attn_gate, mlp, norm, modulation, embed_final).
     Saved (with the export's sidecar as "export_source") to
     /vol/drift/<run>_<step>.json, returned, printed as tables and written
     locally (--out).

Secrets: `krea2-r2` (R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY); values (incl. the bucket name) are never printed.

Importable without the Modal SDK (the app is only defined when `modal`
imports); object selection, the checkpoint download, upload, drift report
and export flow are plain functions (tests/test_modal_export_full.py).
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import modal
except ImportError:  # importable (pyflakes, tests) without the Modal SDK
    modal = None

LOCAL_REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/repo"
# the repo locally; /repo (krea2_jax + export_full.py attached by the image)
# in the container, where this file is mounted on its own
for _repo in (str(LOCAL_REPO), REMOTE_REPO):
    if (_repo not in sys.path
            and os.path.isfile(os.path.join(_repo, "krea2_jax", "r2.py"))):
        sys.path.insert(0, _repo)

from krea2_jax.r2 import (  # noqa: E402 - needs the sys.path above
    head_or_none, make_transfer_config, r2_client, r2_identity,
)

LOCAL_OUT_ROOT = Path("/home/mj/krea2-ops/eval")
R2_PREFIX = "krea2-fullft"
SOURCES = ("params", "ema")

# --- volume layout (the volume of scripts/modal_eval_full.py) ---
VOLUME_NAME = "krea2-eval-cache"
VOL = "/vol"
BASE_REL = "base/raw.safetensors"   # modal_eval_full prepare's HF base

# --- Modal resources ---
EXPORT_CPU = 8
EXPORT_MEM_GIB = 98       # export_full: ~48 GiB fp32 tree + leaf copies
EXPORT_TIMEOUT_S = 3 * 3600
DOWNLOAD_WORKERS = 2      # objects in flight x max_concurrency 32 = the
#                           client's 64 pooled connections
EXPORT_PIP = [            # the stack of the run's checkpoints (orbax 0.12.7)
    "jax==0.11.0", "jaxlib==0.11.0", "orbax-checkpoint==0.12.7",
    "tensorstore==0.1.85", "safetensors==0.8.0", "ml_dtypes==0.6.0",
    "numpy==2.5.4", "boto3==1.43.112",
]
EXPORT_ENV = {"JAX_PLATFORMS": "cpu", "PYTHONUNBUFFERED": "1"}

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# --------------------------------------------------------------- layout
def step_prefix(prefix, run, step):
    """R2 prefix of one orbax step dir (trailing slash: step 15 does not
    match 150)."""
    return f"{prefix}/{run}/ckpt/{int(step)}/"


def export_key(prefix, run, step, source):
    return f"{prefix}/{run}/exports/model_step{int(step)}_{source}.safetensors"


def local_paths(vol, run, step, source):
    """Volume paths of one (run, step, source): {"ckpt_root" (orbax dir for
    export_full --ckpt-dir), "step_dir", "marker", "export", "drift"}."""
    vol = Path(vol)
    root = vol / "ckpt" / run / source
    return {
        "ckpt_root": root,
        "step_dir": root / str(int(step)),
        "marker": vol / "ckpt" / run / "markers" / f"{source}_{int(step)}.json",
        "export": vol / "exports" / f"{run}_{int(step)}_{source}.safetensors",
        "drift": vol / "drift" / f"{run}_{int(step)}.json",
    }


# --------------------------------------------------------------- download
def list_objects(client, bucket, prefix):
    """Every object under prefix (list_objects_v2, paginated) ->
    [{"key", "size", "etag"}]."""
    out, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = client.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            out.append({"key": o["Key"], "size": int(o["Size"]),
                        "etag": str(o.get("ETag", "")).strip('"')})
        if not resp.get("IsTruncated"):
            return out
        token = resp["NextContinuationToken"]


def select_ckpt_objects(objects, prefix, tree):
    """The objects of one orbax step dir (listed under `prefix`) needed to
    restore `tree` alone: the step's top-level files, <tree>/ and meta/.
    -> {relative path: {"key", "size", "etag"}}; ValueError when the step
    has no <tree>/ or no meta/ objects, or an unsafe relative path."""
    if tree not in SOURCES:
        raise ValueError(f"unknown tree {tree!r} (one of {SOURCES})")
    sel, tops = {}, set()
    for o in objects:
        if not o["key"].startswith(prefix):
            continue
        rel = o["key"][len(prefix):]
        if not rel or rel.endswith("/"):   # directory placeholder objects
            continue
        parts = rel.split("/")
        if any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"unsafe object key {o['key']!r}")
        tops.add(parts[0] if len(parts) > 1 else "")
        if len(parts) == 1 or parts[0] in (tree, "meta"):
            sel[rel] = {"key": o["key"], "size": o["size"], "etag": o["etag"]}
    for need in (tree, "meta"):
        if need not in tops:
            raise ValueError(
                f"r2:{prefix}: no {need}/ objects (has "
                f"{sorted(t + '/' for t in tops if t)}); "
                + ("saved without EMA?" if need == "ema" else
                   "not an orbax train-state step dir?"))
    return sel


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


def download_ckpt_tree(client, bucket, prefix, step_dir, tree, marker_path,
                       transfer_config=None, workers=DOWNLOAD_WORKERS,
                       log=print, sel=None):
    """Mirror select_ckpt_objects(<listing of prefix>, tree) into step_dir
    (sel: that selection when the caller already listed the prefix).

    An object is skipped when the marker records its listing (ETag, size)
    and the local file has that size; every other one is downloaded to
    <file>.part, size-checked, its head_object (ETag, size) compared with
    the listing (a replaced object is an error), renamed into place and
    recorded in the marker (written atomically after each object, so an
    interrupted run resumes). Local files not in the selection (stale
    objects, .part leftovers) are deleted first.
    -> {"objects", "downloaded", "skipped", "bytes", "seconds"}."""
    t0 = time.time()
    step_dir = Path(step_dir)
    if sel is None:
        sel = select_ckpt_objects(list_objects(client, bucket, prefix),
                                  prefix, tree)
    marker = _read_json(marker_path) or {}
    if marker.get("prefix") != prefix or marker.get("tree") != tree:
        marker = {}
    done = {rel: v for rel, v in (marker.get("objects") or {}).items()
            if rel in sel and v == {"etag": sel[rel]["etag"],
                                    "size": sel[rel]["size"]}}
    if step_dir.is_dir():
        for p in sorted(step_dir.rglob("*")):
            if p.is_file() and p.relative_to(step_dir).as_posix() not in sel:
                p.unlink()
    todo = [rel for rel in sel
            if not (rel in done and (step_dir / rel).is_file()
                    and (step_dir / rel).stat().st_size == sel[rel]["size"])]
    for rel in todo:
        done.pop(rel, None)
    state = {"prefix": prefix, "tree": tree, "objects": done,
             "complete": False}
    _write_json(marker_path, state)
    if transfer_config is None:
        transfer_config = make_transfer_config()
    lock = threading.Lock()
    total = sum(sel[rel]["size"] for rel in todo)
    log(f"r2:{prefix} ({tree}): {len(sel)} objects, {len(todo)} to download "
        f"({total / 2**30:.2f} GiB), {len(sel) - len(todo)} complete")

    def fetch(rel):
        o = sel[rel]
        dest = step_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        try:
            client.download_file(bucket, o["key"], str(part),
                                 Config=transfer_config)
            got = part.stat().st_size
            if got != o["size"]:
                raise RuntimeError(f"r2:{o['key']}: downloaded {got} bytes, "
                                   f"listing says {o['size']}")
            after = r2_identity(client.head_object(Bucket=bucket,
                                                   Key=o["key"]), o["key"])
            if (after["etag"], after["size"]) != (o["etag"], o["size"]):
                raise RuntimeError(
                    f"r2:{o['key']}: the object changed during the download "
                    f"(listed etag {o['etag']} / {o['size']} bytes, now "
                    f"{after['etag']} / {after['size']}); run again")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, dest)
        with lock:
            done[rel] = {"etag": o["etag"], "size": o["size"]}
            _write_json(marker_path, state)

    # largest first: the long multipart transfers start early
    order = sorted(todo, key=lambda r: -sel[r]["size"])
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        for fut in [pool.submit(fetch, rel) for rel in order]:
            fut.result()
    except BaseException:
        # first error: drop the queued objects, wait for the ones in flight
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    state["complete"] = True
    _write_json(marker_path, state)
    dt = time.time() - t0
    log(f"r2:{prefix} ({tree}): {len(todo)} objects, {total / 2**30:.2f} GiB "
        f"in {dt:.0f} s ({total / 2**20 / max(dt, 1e-6):.0f} MiB/s) -> "
        f"{step_dir}")
    return {"objects": len(sel), "downloaded": len(todo),
            "skipped": len(sel) - len(todo), "bytes": total, "seconds": dt}


# --------------------------------------------------------------- export
def export_metadata(path):
    """The safetensors __metadata__ of path, or None (missing/unreadable)."""
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="numpy") as f:
            return f.metadata() or {}
    except Exception:  # noqa: BLE001 - treated as "no usable export"
        return None


def ckpt_fingerprint(sel):
    """sha256 over the sorted (key, size, ETag) of the selected checkpoint
    objects: changes with the prefix/run/step and with any replaced object."""
    items = sorted((o["key"], o["size"], o["etag"]) for o in sel.values())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def file_sha256(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def export_source_path(export):
    """Sidecar of an export: the checkpoint identity it was made from plus
    the file's size and sha256."""
    export = Path(export)
    return export.with_name(export.name + ".src.json")


def source_identity(bucket, prefix, run, step, source, sel):
    """What an export is made from. The bucket name is a secret value, so
    only its sha256 is recorded."""
    return {"bucket_sha256": hashlib.sha256(bucket.encode()).hexdigest(),
            "prefix": prefix, "run": run, "step": int(step),
            "source": source, "ckpt_fingerprint": ckpt_fingerprint(sel)}


def export_reusable(path, ident):
    """True if path is a complete export whose sidecar records exactly
    `ident` and the file's current size."""
    side = _read_json(export_source_path(path))
    if not side or not Path(path).is_file():
        return False
    return ({k: side.get(k) for k in ident} == ident
            and side.get("size") == Path(path).stat().st_size
            and bool(side.get("sha256"))
            and export_matches(path, ident["step"], ident["source"]))


def export_matches(path, step, source):
    """True if path is an export_full output of this step and source."""
    meta = export_metadata(path)
    return (meta is not None and meta.get("step") == str(int(step))
            and meta.get("source") == source)


def export_script():
    """scripts/export_full.py: next to this file locally, under /repo in the
    container (this file is mounted on its own at /root)."""
    for p in (LOCAL_REPO / "scripts" / "export_full.py",
              Path(REMOTE_REPO, "scripts", "export_full.py")):
        if p.is_file():
            return p
    raise FileNotFoundError("scripts/export_full.py not found")


def run_export(ckpt_root, step, source, out_path, base_path,
               python=sys.executable, script=None, log=print):
    """scripts/export_full.py in a subprocess (its ~48 GiB host tree is
    freed when it exits); raises on a non-zero exit. script: default
    export_script()."""
    script = script or export_script()
    cmd = [python, str(script), "--ckpt-dir", str(ckpt_root),
           "--step", str(int(step)), "--source", source,
           "--output", str(out_path), "--base-ckpt", str(base_path)]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    log("+ " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, env={**os.environ, "JAX_PLATFORMS": "cpu"}).returncode
    if rc != 0:
        raise RuntimeError(f"export_full.py exited with {rc}")
    return time.time() - t0


def upload_export(client, bucket, key, path, metadata, force=False,
                  transfer_config=None, log=print):
    """Upload path to key with user metadata `metadata` (str -> str: the
    checkpoint fingerprint and the file's sha256), unless the object already
    has this size and exactly this metadata (and not force); any other
    existing object is overwritten. Size and metadata are re-checked after
    the upload. -> (status, identity)."""
    size = Path(path).stat().st_size
    metadata = {str(k).lower(): str(v) for k, v in metadata.items()}
    head = head_or_none(client, bucket, key)
    if (head is not None and int(head["ContentLength"]) == size
            and (head.get("Metadata") or {}) == metadata and not force):
        log(f"r2:{key}: exists with the same size and metadata, upload "
            "skipped (--force to replace)")
        return "skipped", r2_identity(head, key)
    if transfer_config is None:
        transfer_config = make_transfer_config()
    t0 = time.time()
    client.upload_file(str(path), bucket, key,
                       ExtraArgs={"Metadata": metadata},
                       Config=transfer_config)
    head = client.head_object(Bucket=bucket, Key=key)
    ident = r2_identity(head, key)
    if ident["size"] != size or (head.get("Metadata") or {}) != metadata:
        raise RuntimeError(f"r2:{key}: uploaded object has {ident['size']} "
                           f"bytes / metadata {head.get('Metadata')}, local "
                           f"file {size} bytes / {metadata}")
    dt = time.time() - t0
    log(f"r2:{key}: uploaded {size / 2**30:.2f} GiB in {dt:.0f} s "
        f"(etag {ident['etag']})")
    return "uploaded", ident


# --------------------------------------------------------------- drift
ATTN_KINDS = {"wq": "attn_q", "wk": "attn_k", "wv": "attn_v", "wo": "attn_o",
              "gate": "attn_gate", "qknorm": "norm"}
KINDS = ("attn_q", "attn_k", "attn_v", "attn_o", "attn_gate", "mlp", "norm",
         "modulation", "embed_final")
_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def tensor_kind(key):
    """Official Krea 2 key -> one of KINDS."""
    parts = key.split(".")
    if "attn" in parts[:-1]:
        kind = ATTN_KINDS.get(parts[parts.index("attn") + 1])
        if kind:
            return kind
    if "mlp" in parts:
        return "mlp"
    if parts[-1] == "scale" or any(p.endswith("norm") for p in parts):
        return "norm"
    if "mod" in parts or "modulation" in parts:
        return "modulation"
    return "embed_final"   # first, last.linear, tmlp, tproj, txtmlp, projector


def tensor_group(key):
    """'blockNN' for DiT block NN, 'txtfusion' for the text fusion blocks,
    'other' for the rest (embeddings, final layer)."""
    m = _BLOCK_RE.match(key)
    if m:
        return f"block{int(m.group(1)):02d}"
    return "txtfusion" if key.startswith("txtfusion.") else "other"


def _sq_sums(w, b, chunk=1 << 24):
    """(sum (w - b)^2, sum b^2) in float64, chunked (bounded temporaries)."""
    import numpy as np

    w, b = w.reshape(-1), b.reshape(-1)
    d2 = b2 = 0.0
    for i in range(0, b.size, chunk):
        bb = b[i:i + chunk].astype(np.float64)
        dd = w[i:i + chunk].astype(np.float64) - bb
        d2 += float(np.dot(dd, dd))
        b2 += float(np.dot(bb, bb))
    return d2, b2


def _ratio(d2, b2):
    import math

    return math.sqrt(d2 / b2) if b2 > 0 else None


def _summary(tensors):
    """Aggregate {key: {"d2", "b2", "rel"}} -> global / median / p90 / max
    and the per-group / per-kind global ratios."""
    import numpy as np

    rels = {k: t["rel"] for k, t in tensors.items()
            if t["rel"] is not None and np.isfinite(t["rel"])}
    vals = np.array(list(rels.values()), np.float64)

    def ratio(keys):
        keys = [k for k in keys if np.isfinite(tensors[k]["d2"])]
        return _ratio(sum(tensors[k]["d2"] for k in keys),
                      sum(tensors[k]["b2"] for k in keys))

    groups, kinds = {}, {}
    for k in tensors:
        groups.setdefault(tensor_group(k), []).append(k)
        kinds.setdefault(tensor_kind(k), []).append(k)
    max_key = max(rels, key=rels.get) if rels else None
    return {
        "global": ratio(tensors),
        "median": float(np.median(vals)) if vals.size else None,
        "p90": float(np.percentile(vals, 90)) if vals.size else None,
        "max": rels[max_key] if max_key else None,
        "max_key": max_key,
        "by_block": {g: ratio(groups[g]) for g in sorted(groups)},
        "by_kind": {k: ratio(kinds[k]) for k in KINDS if k in kinds},
    }


def drift_report(base_path, files, log=print):
    """Relative weight drift of each (label, path) in files vs base_path.

    Tensors are streamed one at a time (safe_open, any float dtype ->
    float64 sums): per tensor d2 = ||W - W_base||^2, b2 = ||W_base||^2, rel
    = sqrt(d2 / b2) (None when b2 == 0). Keys missing from a file, extra keys
    and shape mismatches are listed and left out; non-finite tensors are
    listed (their d2 is NaN/inf and is left out of the aggregates).
    -> {"base", "files": [{"label", "path", "n_tensors", "missing", "extra",
    "shape_mismatch", "nonfinite", "global", "median", "p90", "max",
    "max_key", "by_block", "by_kind", "tensors": {key: {d2, b2, rel}}}],
    "seconds"}."""
    import math
    from contextlib import ExitStack

    from safetensors import safe_open

    t0 = time.time()
    with ExitStack() as stack:
        base = stack.enter_context(safe_open(str(base_path),
                                             framework="numpy"))
        handles = [stack.enter_context(safe_open(str(p), framework="numpy"))
                   for _, p in files]
        base_keys = sorted(base.keys())
        per = [{"tensors": {}, "shape_mismatch": [], "nonfinite": [],
                "keys": set(h.keys())} for h in handles]
        for n, key in enumerate(base_keys):
            b = base.get_tensor(key)
            for h, st in zip(handles, per):
                if key not in st["keys"]:
                    continue
                w = h.get_tensor(key)
                if w.shape != b.shape:
                    st["shape_mismatch"].append(
                        [key, list(w.shape), list(b.shape)])
                    continue
                d2, b2 = _sq_sums(w, b)
                if not math.isfinite(d2):
                    st["nonfinite"].append(key)
                st["tensors"][key] = {"d2": d2, "b2": b2,
                                      "rel": _ratio(d2, b2)
                                      if math.isfinite(d2) else None}
                del w
            del b
            if (n + 1) % 50 == 0:
                log(f"drift: {n + 1}/{len(base_keys)} tensors "
                    f"({time.time() - t0:.0f} s)")
    out = []
    for (label, path), st in zip(files, per):
        out.append({
            "label": label, "path": str(path),
            "n_tensors": len(st["tensors"]),
            "missing": sorted(set(base_keys) - st["keys"]),
            "extra": sorted(st["keys"] - set(base_keys)),
            "shape_mismatch": st["shape_mismatch"],
            "nonfinite": st["nonfinite"],
            **_summary(st["tensors"]),
            "tensors": st["tensors"],
        })
    return {"base": str(base_path), "files": out,
            "seconds": time.time() - t0}


def _fmt(x):
    return "-" if x is None else f"{x:.3e}"


def drift_table(report):
    """Compact text tables of a drift_report (files as numbered columns)."""
    files = report["files"]
    lines = [f"relative drift ||W - W_base|| / ||W_base|| vs {report['base']}"]
    for i, f in enumerate(files):
        notes = [f"{len(f[k])} {k.replace('_', ' ')}"
                 for k in ("missing", "extra", "shape_mismatch", "nonfinite")
                 if f[k]]
        lines.append(f"  [{i + 1}] {f['label']} ({f['n_tensors']} tensors"
                     + (", " + ", ".join(notes) if notes else "") + ")")
    cols = "".join(f"{f'[{i + 1}]':>12}" for i in range(len(files)))
    lines.append(f"{'':<14}{cols}")
    for stat in ("global", "median", "p90", "max"):
        lines.append(f"{stat:<14}" + "".join(f"{_fmt(f[stat]):>12}"
                                             for f in files))
    for f_i, f in enumerate(files):
        if f["max_key"]:
            lines.append(f"  max [{f_i + 1}]: {f['max_key']}")
    for title, field in (("by kind", "by_kind"), ("by block", "by_block")):
        lines.append(f"{title}:")
        names = []
        for f in files:
            names += [n for n in f[field] if n not in names]
        for n in names:
            lines.append(f"  {n:<12}" + "".join(
                f"{_fmt(f[field].get(n)):>12}" for f in files))
    if len(files) > 1 and files[0]["global"]:
        ref = files[0]["global"]
        lines.append("global drift relative to [1]: " + ", ".join(
            f"[{i + 1}] {f['global'] / ref:.3f}" for i, f in enumerate(files)
            if f["global"] is not None))
    return "\n".join(lines)


# --------------------------------------------------------------- flow
def vol_label(path, vol=VOL):
    """Path relative to the volume root (display label)."""
    try:
        return Path(path).relative_to(vol).as_posix()
    except ValueError:
        return str(path)


def export_impl(run, step, source, compares=(), drift_only=False, force=False,
                prefix=R2_PREFIX, env=None, vol=VOL, commit=lambda: None,
                client=None, transfer_config=None, python=sys.executable,
                log=print):
    """The whole remote flow (see the module docstring). client: a
    boto3-style S3 client (default r2_client(env)). -> {"run", "step",
    "source", "export": {...} or None, "upload": {...} or None, "drift":
    drift_report}."""
    env = os.environ if env is None else env
    paths = local_paths(vol, run, step, source)
    base_path = Path(vol) / BASE_REL
    if not base_path.is_file():
        raise RuntimeError(f"{base_path} missing: run `modal run "
                           "scripts/modal_eval_full.py --limit 1` (its "
                           "prepare caches the HF base) first")
    missing = [c for c in compares if not Path(c).is_file()]
    if missing:
        raise RuntimeError(f"--compare files not on the volume: {missing}")
    out = {"run": run, "step": int(step), "source": source, "export": None,
           "upload": None}
    if not drift_only:
        if client is None:
            client = r2_client(env)
        bucket = env["R2_BUCKET"]
        exp = {"path": str(paths["export"])}
        # always list the checkpoint (cheap): the export is reused only if
        # it was made from exactly these objects
        sp = step_prefix(prefix, run, step)
        sel = select_ckpt_objects(list_objects(client, bucket, sp), sp,
                                  source)
        ident = source_identity(bucket, prefix, run, step, source, sel)
        side_path = export_source_path(paths["export"])
        if not force and export_reusable(paths["export"], ident):
            log(f"{paths['export']}: export of this checkpoint (fingerprint "
                f"{ident['ckpt_fingerprint'][:12]}) exists, skipped (--force "
                "to redo)")
            exp["status"] = "skipped"
        else:
            side_path.unlink(missing_ok=True)
            exp["download"] = download_ckpt_tree(
                client, bucket, sp, paths["step_dir"], source,
                paths["marker"], transfer_config=transfer_config, log=log,
                sel=sel)
            commit()
            exp["seconds"] = run_export(paths["ckpt_root"], step, source,
                                        paths["export"], base_path,
                                        python=python, log=log)
            if not export_matches(paths["export"], step, source):
                raise RuntimeError(f"{paths['export']}: not an export of step "
                                   f"{step} ({source}) after export_full")
            _write_json(side_path, {
                **ident, "size": paths["export"].stat().st_size,
                "sha256": file_sha256(paths["export"])})
            exp["status"] = "exported"
            commit()
        side = _read_json(side_path)
        exp["size"] = paths["export"].stat().st_size
        exp["source_identity"] = side
        out["export"] = exp
        status, up = upload_export(
            client, bucket, export_key(prefix, run, step, source),
            paths["export"], {"ckpt-fingerprint": side["ckpt_fingerprint"],
                              "sha256": side["sha256"]},
            force=force, transfer_config=transfer_config, log=log)
        out["upload"] = {"status": status, **up}
    files = []
    if paths["export"].is_file():
        files.append((vol_label(paths["export"], vol), paths["export"]))
    files += [(vol_label(c, vol), Path(c)) for c in compares
              if Path(c) != paths["export"]]
    if not files:
        raise RuntimeError(f"nothing to compare: no export at "
                           f"{paths['export']} and no --compare")
    report = drift_report(base_path, files, log=log)
    report["base"] = vol_label(base_path, vol)
    if paths["export"].is_file():   # what the export was made from
        report["export_source"] = _read_json(
            export_source_path(paths["export"]))
    _write_json(paths["drift"], report)
    commit()
    log(drift_table(report))
    out["drift"] = report
    out["drift_path"] = str(paths["drift"])
    return out


# --------------------------------------------------------------- CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(
        prog="modal run scripts/modal_export_full.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="run1",
                    help="run directory under --prefix on R2")
    ap.add_argument("--step", type=int, required=True,
                    help="checkpoint step (<prefix>/<run>/ckpt/<step>/)")
    ap.add_argument("--source", choices=SOURCES, default="params",
                    help="tree to export (default params)")
    ap.add_argument("--compare", action="append", default=[],
                    metavar="PATH",
                    help="another safetensors on the volume (/vol/...) for "
                         "the drift report; repeatable")
    ap.add_argument("--drift-only", action="store_true",
                    help="skip download / export / upload")
    ap.add_argument("--force", action="store_true",
                    help="export and upload again even when present")
    ap.add_argument("--prefix", default=R2_PREFIX,
                    help="R2 prefix of the runs")
    ap.add_argument("--out", default="",
                    help="local drift JSON (default "
                         f"{LOCAL_OUT_ROOT}/drift_<run>_<step>.json)")
    return ap


def parse_args(argv):
    """argparse + validation (names, step, /vol/ compare paths)."""
    ap = build_arg_parser()
    args = ap.parse_args(list(argv))
    if not NAME_RE.match(args.run):
        ap.error(f"--run must match {NAME_RE.pattern}, got {args.run!r}")
    if args.step < 0:
        ap.error(f"--step must be >= 0, got {args.step}")
    if not all(NAME_RE.match(p) for p in args.prefix.split("/")):
        ap.error(f"--prefix must be '/'-separated names, got {args.prefix!r}")
    for c in args.compare:
        parts = c.split("/")
        if not c.startswith(VOL + "/") or any(p in ("", ".", "..")
                                              for p in parts[1:]):
            ap.error(f"--compare must be a file on the volume ({VOL}/...), "
                     f"got {c!r}")
    return args


def write_local(result, out=""):
    """The returned drift report -> local JSON (--out or the default)."""
    path = Path(out) if out else (
        LOCAL_OUT_ROOT / f"drift_{result['run']}_{result['step']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return path


# ------------------------------------------------------------------ Modal app
if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*EXPORT_PIP)
        .env(EXPORT_ENV)
        .add_local_dir(LOCAL_REPO / "krea2_jax", f"{REMOTE_REPO}/krea2_jax",
                       ignore=["**/__pycache__"])
        .add_local_file(LOCAL_REPO / "scripts" / "export_full.py",
                        f"{REMOTE_REPO}/scripts/export_full.py")
    )
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    r2_secret = modal.Secret.from_name("krea2-r2")
    app = modal.App("krea2-export-full")

    @app.function(image=image, cpu=float(EXPORT_CPU),
                  memory=EXPORT_MEM_GIB * 1024, timeout=EXPORT_TIMEOUT_S,
                  volumes={VOL: vol}, secrets=[r2_secret])
    def export_full_remote(run: str, step: int, source: str, compares: list,
                           drift_only: bool, force: bool, prefix: str) -> dict:
        return export_impl(
            run, step, source, compares, drift_only=drift_only, force=force,
            prefix=prefix, commit=vol.commit,
            log=lambda m: print(m, flush=True),
            python=sys.executable)

    @app.local_entrypoint()
    def main(*argv: str):
        args = parse_args(argv)
        t0 = time.time()
        res = export_full_remote.remote(
            args.run, args.step, args.source, list(args.compare),
            args.drift_only, args.force, args.prefix)
        print(f"remote: {(time.time() - t0) / 60:.1f} min")
        if res["export"]:
            print(f"export: {res['export']['status']} "
                  f"{res['export']['path']} ({res['export']['size']} bytes)")
        if res["upload"]:
            u = res["upload"]
            print(f"upload: {u['status']} r2 {u['key']} (etag {u['etag']}, "
                  f"{u['size']} bytes)")
        print(drift_table(res["drift"]))
        print(f"\nwrote {write_local(res, args.out)}")


def local_main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    raise SystemExit(f"this script runs on Modal: modal run "
                     f"scripts/modal_export_full.py --run {args.run} "
                     f"--step {args.step} ...")


if __name__ == "__main__":
    local_main()
