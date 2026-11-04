"""Modal app (CPU only): average fp32 snapshots of a full fine-tuning run on
R2 (scripts/train_full.py --snapshot-steps) into one bf16 drop-in
model.safetensors (scripts/average_snapshots.py arithmetic and metadata) and
upload it next to the run on R2. The inputs are streamed tensor by tensor
with ranged GETs (krea2_jax.r2.R2Safetensors / R2Fetch); nothing of them is
downloaded to disk.

    modal run scripts/modal_average_full.py --run run3 \\
        --steps 7354,7814,8273,8733 --name avg_decay4 [--weights 1,1,2,2]
    modal run scripts/modal_average_full.py ... --dry-run   # plan, no reads
    python scripts/modal_average_full.py --help

Options (local entrypoint, argparse):
  --run NAME        run directory on R2 (required)
  --steps a,b,...   snapshot steps (unique, positive; the input order is the
                    summation order and is recorded in the output metadata)
  --name NAME       output <prefix>/<run>/averages/<NAME>.safetensors
  --weights w,...   non-negative weights, one per step (normalized; default
                    uniform)
  --prefix P        R2 prefix of the runs (default krea2-fullft)
  --force           replace an existing output object (otherwise an output
                    of the same inputs is skipped and any other one refused)
  --dry-run         list + headers only: checks and the plan, no tensor
                    reads, no upload

One function (`average_full_remote`, AVG_CPU cores, no memory / disk
reservation, the `krea2-r2` secret):
  1. list <prefix>/<run>/snapshots/, map each step to
     model_step<NNNNNN>_f32.safetensors (a missing step lists the available
     ones). Inputs fingerprint = sha256 of the JSON of the inputs in input
     order (key, size, ETag, normalized weight).
  2. head the output key: same "inputs-fingerprint" metadata -> "exists"
     (skipped, unless --force); any other object -> refused (unless --force).
  3. open an R2Safetensors reader per input (head ETag must equal the
     listing's; every ranged GET is pinned to it with IfMatch), check keys /
     shapes / F32 / base (average_snapshots.check_inputs). --dry-run stops
     here and returns the plan.
  4. average_snapshots.average into /tmp/<name>.safetensors (~25.6 GB),
     the readers' tensors prefetched across tensor boundaries within
     PREFETCH_BYTES (R2Fetch) over READ_WORKERS connections; progress
     (tensors, GiB read, MiB/s, seconds waiting on the network / accumulating
     / writing) every ~30 s.
  5. upload (boto3 upload_file, make_transfer_config) with user metadata
     {inputs-fingerprint, avg-steps, avg-weights} after every input is
     re-headed (ETag + size as pinned, else no upload); head_object: size and
     metadata must match. The temp file is deleted. Snapshot objects are
     only read.

Secrets: `krea2-r2`; values (incl. the bucket name) are never printed: any
remote error is re-raised with them replaced by placeholders (scrubbed_call).
Importable without the Modal SDK (the app is only defined when `modal`
imports).
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
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
# the repo locally; /repo (krea2_jax + scripts attached by the image) in the
# container, where this file is mounted on its own
for _repo in (str(LOCAL_REPO), REMOTE_REPO):
    if os.path.isfile(os.path.join(_repo, "krea2_jax", "r2.py")):
        for _p in (_repo, os.path.join(_repo, "scripts")):
            if _p not in sys.path:
                sys.path.insert(0, _p)

from krea2_jax.r2 import (  # noqa: E402 - needs the sys.path above
    R2Fetch, R2Safetensors, env_secrets, head_or_none, list_objects,
    make_transfer_config, r2_client, r2_identity, scrub,
)

AVG_CPU = 8
AVG_TIMEOUT_S = 3 * 3600
R2_PREFIX = "krea2-fullft"
AVG_PIP = [               # modal_export_full.EXPORT_PIP + optax (train_full)
    "jax==0.11.0", "jaxlib==0.11.0", "orbax-checkpoint==0.12.7",
    "tensorstore==0.1.85", "safetensors==0.8.0", "ml_dtypes==0.6.0",
    "numpy==2.5.4", "boto3==1.43.112", "optax==0.2.8",
]
AVG_ENV = {"JAX_PLATFORMS": "cpu", "PYTHONUNBUFFERED": "1"}
READ_WORKERS = 32          # ranged GETs in flight (client pool: 64)
PREFETCH_BYTES = 4 * 2**30  # fetched-but-unconsumed input bytes
PROGRESS_S = 30
SNAP_FMT = "model_step{:06d}_f32.safetensors"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# --------------------------------------------------------------- layout
def snapshots_prefix(prefix, run):
    return f"{prefix}/{run}/snapshots/"


def average_key(prefix, run, name):
    return f"{prefix}/{run}/averages/{name}.safetensors"


def select_snapshots(objects, prefix, run, steps):
    """[{"step", "key", "size", "etag"}] in `steps` order; RuntimeError
    listing the available steps when one is missing."""
    sp = snapshots_prefix(prefix, run)
    by_key = {o["key"]: o for o in objects}
    out, missing = [], []
    for s in steps:
        o = by_key.get(sp + SNAP_FMT.format(s))
        if o is None:
            missing.append(s)
        else:
            out.append({"step": s, **o})
    if missing:
        avail = sorted(int(o["key"][len(sp) + 10:len(sp) + 16])
                       for o in objects
                       if o["key"].startswith(sp + "model_step")
                       and o["key"].endswith("_f32.safetensors")
                       and o["key"][len(sp) + 10:len(sp) + 16].isdigit())
        raise RuntimeError(f"r2:{sp}: no snapshot of step(s) {missing}; "
                           f"available: {avail}")
    return out


def normalized(weights):
    total = float(sum(weights))
    return [repr(w / total) for w in weights]   # as output_metadata's


def inputs_fingerprint(inputs, weights):
    """sha256 over the inputs in input order (key, size, ETag, weight as
    given, i.e. as used in the arithmetic): the input order is the summation
    order (output bits)."""
    items = [[o["key"], int(o["size"]), o["etag"], repr(float(w))]
             for o, w in zip(inputs, weights)]
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


# --------------------------------------------------------------- flow
def upload_average(client, bucket, key, path, metadata, transfer_config=None,
                   log=print):
    """upload_file with user metadata; size and metadata re-checked."""
    size = Path(path).stat().st_size
    metadata = {str(k).lower(): str(v) for k, v in metadata.items()}
    t0 = time.time()
    client.upload_file(str(path), bucket, key,
                       ExtraArgs={"Metadata": metadata},
                       Config=transfer_config or make_transfer_config())
    head = client.head_object(Bucket=bucket, Key=key)
    ident = r2_identity(head, key)
    if ident["size"] != size or (head.get("Metadata") or {}) != metadata:
        raise RuntimeError(f"r2:{key}: uploaded object has {ident['size']} "
                           f"bytes / metadata {head.get('Metadata')}, local "
                           f"file {size} bytes / {metadata}")
    dt = time.time() - t0
    log(f"r2:{key}: uploaded {size / 2**30:.2f} GiB in {dt:.0f} s "
        f"(etag {ident['etag']})")
    return ident


class Progress:
    """on_tensor callback of average(): a line every PROGRESS_S seconds."""

    def __init__(self, stats, n_tensors, total_bytes, log=print,
                 every=PROGRESS_S):
        self.stats, self.n, self.total = stats, n_tensors, total_bytes
        self.log, self.every = log, every
        self.t0 = self.last = time.time()
        self.done = 0
        self.timing = {}       # updated live by average_snapshots.average

    def __call__(self, key):
        self.done += 1
        now = time.time()
        if now - self.last >= self.every or self.done == self.n:
            self.last = now
            self.log(self.line(now))

    def line(self, now=None):
        dt = max((now or time.time()) - self.t0, 1e-6)
        rb = self.stats.get("read_bytes", 0)
        f, a = self.timing.get("fetch_s", 0.0), self.timing.get(
            "accumulate_s", 0.0)
        return (f"tensors {self.done}/{self.n}, read {rb / 2**30:.1f}/"
                f"{self.total / 2**30:.1f} GiB, {rb / 2**20 / dt:.0f} MiB/s, "
                f"{dt:.0f} s (network wait {f:.0f} s, accumulate {a:.0f} s, "
                f"write {dt - f - a:.0f} s)")


def average_impl(run, steps, name, weights, prefix=R2_PREFIX, force=False,
                 dry_run=False, env=None, client=None, tmp_dir=None,
                 transfer_config=None, log=print):
    """The remote flow (module docstring). weights: list of floats (parsed
    locally). -> plan (dry run) or {"status", "key", "etag", "size",
    "read_bytes", "seconds", "read_mib_s", "fetch_s", "accumulate_s",
    "write_s", ...}."""
    import average_snapshots as avs

    env = os.environ if env is None else env
    client = client or r2_client(env)
    bucket = env["R2_BUCKET"]
    inputs = select_snapshots(
        list_objects(client, bucket, snapshots_prefix(prefix, run)),
        prefix, run, steps)
    fp = inputs_fingerprint(inputs, weights)
    key = average_key(prefix, run, name)
    out = {"key": key, "inputs_fingerprint": fp, "steps": list(steps),
           "weights": normalized(weights),
           "inputs": [{k: o[k] for k in ("step", "key", "size", "etag")}
                      for o in inputs]}
    head = head_or_none(client, bucket, key)
    if head is None:
        out["output"] = "absent"
    elif (head.get("Metadata") or {}).get("inputs-fingerprint") == fp:
        out["output"] = "exists"
    else:
        out["output"] = "conflict"
    if not force and not dry_run:
        if out["output"] == "exists":
            log(f"r2:{key}: exists, made from these inputs (fingerprint "
                f"{fp[:12]}); skipped (--force to redo)")
            return {**out, "status": "exists", **r2_identity(head, key)}
        if out["output"] == "conflict":
            raise RuntimeError(
                f"r2:{key}: exists with other inputs (inputs-fingerprint "
                f"{(head.get('Metadata') or {}).get('inputs-fingerprint')!r},"
                f" these {fp!r}); pick another --name or pass --force")

    stats, lock = {"read_bytes": 0}, threading.Lock()
    pool = ThreadPoolExecutor(READ_WORKERS)
    try:
        readers = []
        for o in inputs:
            r = R2Safetensors(client, bucket, o["key"], executor=pool,
                              stats=stats, lock=lock)
            if r.etag.strip('"') != o["etag"] or r.size != o["size"]:
                raise RuntimeError(f"r2:{o['key']}: changed since the listing "
                                   f"(etag {r.etag}, size {r.size}); re-run")
            readers.append(r)
        names = [o["key"] for o in inputs]
        shapes, metadata = avs.check_inputs(readers, names, weights)
        per_input = sum(readers[0].nbytes(k) for k in shapes)
        out.update(tensors=len(shapes), read_bytes_planned=per_input
                   * len(readers), metadata=metadata)
        log(f"plan: {len(readers)} input(s) {list(steps)} weights "
            f"{out['weights']} -> r2:{key} ({out['output']}); "
            f"{len(shapes)} tensors, {per_input * len(readers) / 2**30:.1f} "
            f"GiB to read; metadata {metadata}")
        if dry_run:
            return {**out, "status": "dry-run"}

        tmp = Path(tempfile.mkdtemp(dir=tmp_dir, prefix="avg_"))
        path = tmp / f"{name}.safetensors"
        try:
            progress = Progress(stats, len(shapes), out["read_bytes_planned"],
                                log=log)
            fetch = R2Fetch(readers, budget=PREFETCH_BYTES, log=log)
            res = avs.average(readers, names, path, weights, fetch=fetch,
                              threads=AVG_CPU, on_tensor=progress,
                              timing=progress.timing)
            rb = stats["read_bytes"]
            out.update(
                read_bytes=rb, seconds=round(res["seconds"], 1),
                read_mib_s=round(rb / 2**20 / max(res["seconds"], 1e-6), 1),
                fetch_s=round(res["fetch_s"], 1),
                accumulate_s=round(res["accumulate_s"], 1),
                write_s=round(res["write_s"], 1))
            log(f"averaged {res['tensors']} tensors in {out['seconds']} s: "
                f"read {rb / 2**30:.1f} GiB at {out['read_mib_s']} MiB/s; "
                f"waiting on the network {out['fetch_s']} s, accumulating "
                f"{out['accumulate_s']} s, bf16 cast + write {out['write_s']} "
                "s")
            # an input whose last GET finished before it was replaced is
            # caught only here: re-head every input before uploading
            for r in readers:
                h = client.head_object(Bucket=bucket, Key=r.key)
                if (h["ETag"], int(h["ContentLength"])) != (r.etag, r.size):
                    raise RuntimeError(f"input {r.key} changed during the run "
                                       f"(etag {r.etag} -> {h['ETag']}, size "
                                       f"{r.size} -> {h['ContentLength']}); "
                                       "not uploaded")
            ident = upload_average(
                client, bucket, key, path,
                {"inputs-fingerprint": fp,
                 "avg-steps": ",".join(map(str, steps)),
                 "avg-weights": ",".join(out["weights"])},
                transfer_config=transfer_config, log=log)
        finally:
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()
        return {**out, "status": "uploaded", **ident}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def scrubbed_call(fn, *args, env=None, **kwargs):
    """fn(*args, **kwargs); any exception is re-raised as RuntimeError with
    the `krea2-r2` secret values (bucket, account id / endpoint host, keys)
    replaced by placeholders, without the original chain (botocore errors
    can name the bucket or endpoint)."""
    env = os.environ if env is None else env
    try:
        return fn(*args, **kwargs)
    except (Exception, SystemExit) as e:   # check_inputs: SystemExit
        secrets = env_secrets(env)
        acct = env.get("R2_ACCOUNT_ID")
        if acct:
            secrets.append((f"{acct}.r2.cloudflarestorage.com", "<endpoint>"))
        msg = scrub(f"{type(e).__name__}: {e}", secrets)
    raise RuntimeError(msg) from None


# --------------------------------------------------------------- CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(
        prog="modal run scripts/modal_average_full.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="run directory under --prefix on R2")
    ap.add_argument("--steps", required=True,
                    help="comma-separated snapshot steps (input order)")
    ap.add_argument("--name", required=True,
                    help="output <prefix>/<run>/averages/<name>.safetensors")
    ap.add_argument("--weights", default=None,
                    help="comma-separated non-negative weights, one per step "
                         "(normalized; default uniform)")
    ap.add_argument("--prefix", default=R2_PREFIX,
                    help="R2 prefix of the runs")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing output object")
    ap.add_argument("--dry-run", action="store_true",
                    help="headers + plan only: no tensor reads, no upload")
    return ap


def parse_weights(text, n):
    """average_snapshots.parse_weights without numpy (the local Modal CLI's
    Python has none): None -> uniform, else n finite non-negative floats with
    a positive sum; SystemExit otherwise."""
    if text is None:
        return [1.0] * n
    try:
        w = [float(x) for x in text.split(",")]
    except ValueError:
        w = []
    if len(w) != n or any(not math.isfinite(x) or x < 0 for x in w) \
            or sum(w) <= 0:
        raise SystemExit(f"--weights must be {n} non-negative numbers with a "
                         f"positive sum, got {text!r}")
    return w


def parse_args(argv):
    """argparse + validation -> args with .steps (ints), .weights (floats)."""
    ap = build_arg_parser()
    args = ap.parse_args(list(argv))
    for opt, v in (("--run", args.run), ("--name", args.name)):
        if not NAME_RE.match(v):
            ap.error(f"{opt} must match {NAME_RE.pattern}, got {v!r}")
    if not all(NAME_RE.match(p) for p in args.prefix.split("/")):
        ap.error(f"--prefix must be '/'-separated names, got {args.prefix!r}")
    try:
        steps = [int(s) for s in args.steps.split(",")]
    except ValueError:
        ap.error(f"--steps must be comma-separated integers, got "
                 f"{args.steps!r}")
    if any(s <= 0 for s in steps) or len(set(steps)) != len(steps):
        ap.error(f"--steps must be unique positive integers, got {steps}")
    try:
        args.weights = parse_weights(args.weights, len(steps))
    except SystemExit as e:
        ap.error(str(e))
    args.steps = steps
    return args


# ------------------------------------------------------------------ Modal app
if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*AVG_PIP)
        .env(AVG_ENV)
        .add_local_dir(LOCAL_REPO / "krea2_jax", f"{REMOTE_REPO}/krea2_jax",
                       ignore=["**/__pycache__"])
        .add_local_file(LOCAL_REPO / "scripts" / "average_snapshots.py",
                        f"{REMOTE_REPO}/scripts/average_snapshots.py")
        .add_local_file(LOCAL_REPO / "scripts" / "train_full.py",
                        f"{REMOTE_REPO}/scripts/train_full.py")
    )
    r2_secret = modal.Secret.from_name("krea2-r2")
    app = modal.App("krea2-average-full")

    @app.function(image=image, cpu=float(AVG_CPU), timeout=AVG_TIMEOUT_S,
                  secrets=[r2_secret])
    def average_full_remote(run: str, steps: list, name: str, weights: list,
                            prefix: str, force: bool, dry_run: bool) -> dict:
        return scrubbed_call(
            average_impl, run, steps, name, weights, prefix=prefix,
            force=force, dry_run=dry_run, log=lambda m: print(m, flush=True))

    @app.local_entrypoint()
    def main(*argv: str):
        args = parse_args(argv)
        t0 = time.time()
        res = average_full_remote.remote(
            args.run, args.steps, args.name, args.weights, args.prefix,
            args.force, args.dry_run)
        print(f"remote: {(time.time() - t0) / 60:.1f} min")
        print(json.dumps(res, indent=1))
        if res["status"] in ("uploaded", "exists"):
            print(f"\neval: --models {args.name}={res['key']}")


def local_main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    raise SystemExit(f"this script runs on Modal: modal run "
                     f"scripts/modal_average_full.py --run {args.run} "
                     f"--steps {','.join(map(str, args.steps))} "
                     f"--name {args.name} ...")


if __name__ == "__main__":
    local_main()
