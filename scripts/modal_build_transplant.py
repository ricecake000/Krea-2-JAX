"""Modal app (CPU only): Krea-2-Turbo + alpha * (<run> - Raw) "transplant"
checkpoints (scripts/build_transplant.py), one per alpha, uploaded to R2 for
scripts/modal_eval_full.py --models.

    modal run scripts/modal_build_transplant.py --alphas 0,0.4,0.7,1.0
    modal run scripts/modal_build_transplant.py --alphas 0.7 --run run3 \\
        --step 9192 [--force] [--dry-run]
    python scripts/modal_build_transplant.py --help

Options (local entrypoint, argparse):
  --alphas a,b,...  comma-separated finite alphas (unique; built in order)
  --run NAME        run of the fp32 export (default run3)
  --step N          its checkpoint step (default 9192)
  --prefix P        R2 prefix of the runs (default krea2-fullft)
  --force           rebuild and replace existing output objects
  --dry-run         checks + plan only: no tensor reads, no upload

One function (`transplant_remote`, BUILD_CPU cores, no memory / disk
reservation) on the Volume `krea2-eval-cache` (read only) with the
`krea2-r2` secret, sequential per alpha in one container:
  1. inputs on the volume, nothing downloaded (the error names the command
     that creates a missing one):
       /vol/turbo/turbo.safetensors  complete marker (modal_delta_compare)
       /vol/base/raw.safetensors     complete marker (modal_eval_full)
       /vol/exports/<run>_<step>_params_f32.safetensors + its .src.json
                                     (modal_export_full --dtype f32
                                     --no-upload)
  2. per alpha: inputs fingerprint = sha256 of the JSON of {Turbo's marker
     identity, Raw's marker identity, the export's sidecar, alpha}; output
     <prefix>/<run>/transplant/turbo_<run>_<step>_a<alpha>.safetensors
     (alpha as repr: a0.0, a0.4, a1.0). Its "inputs-fingerprint" user
     metadata equal -> skipped; any other object -> refused (before any
     build) unless --force.
  3. headers of the three inputs checked (keys, shapes, dtypes). --dry-run
     returns the plan here.
  4. one finite pass over the inputs, then per alpha: build_transplant.build
     into a temp dir under /tmp (~26 GB), the inputs re-identified (same
     fingerprint, else no upload), upload_file (make_transfer_config) with
     {inputs-fingerprint, alpha} as user metadata, head-checked (size,
     metadata); the temp file is deleted in a finally. Progress every ~30 s.

The local entrypoint prints the R2 keys and the --models line for
scripts/modal_eval_full.py. Secret values (incl. the bucket name) are never
printed (modal_average_full.scrubbed_call). Importable without the Modal SDK
and without numpy (the local entrypoint runs in the Modal CLI's Python): the
helper scripts are imported with `modal` hidden, so they do not define their
own apps inside this one; build_transplant (numpy) only in the container.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

try:
    import modal
except ImportError:  # importable (pyflakes, tests) without the Modal SDK
    modal = None

LOCAL_REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/repo"
# the repo + scripts/ locally; /repo (attached by the image) in the
# container, where this file is mounted on its own
for _repo in (str(LOCAL_REPO), REMOTE_REPO):
    if os.path.isfile(os.path.join(_repo, "scripts", "build_transplant.py")):
        for _p in (_repo, os.path.join(_repo, "scripts")):
            if _p not in sys.path:
                sys.path.insert(0, _p)


def _import_hidden(name):
    """scripts/<name>.py as a helper library, `modal` hidden while it is
    first imported (modal_delta_compare's pattern)."""
    if name in sys.modules:
        return sys.modules[name]
    import importlib

    missing = object()
    saved = sys.modules.get("modal", missing)
    sys.modules["modal"] = None   # `import modal` raises ImportError
    try:
        return importlib.import_module(name)
    finally:
        if saved is missing:
            del sys.modules["modal"]
        else:
            sys.modules["modal"] = saved


mdc = _import_hidden("modal_delta_compare")
mav = _import_hidden("modal_average_full")
mxf, mef = mdc.mxf, mdc.mef

from krea2_jax.r2 import head_or_none, r2_client, r2_identity  # noqa: E402

VOLUME_NAME, VOL = mxf.VOLUME_NAME, mxf.VOL
R2_PREFIX = mxf.R2_PREFIX
DEFAULT_RUN, DEFAULT_STEP = mdc.DEFAULT_RUN, mdc.DEFAULT_STEP
SOURCE = "params"
METHOD = "turbo + alpha * (ft - raw), float32, Turbo dtypes, v1"

# --- Modal resources ---
BUILD_CPU = 8
BUILD_TIMEOUT_S = 3 * 3600
SCRIPTS = ("build_transplant.py", "modal_delta_compare.py", "delta_compare.py",
           "modal_export_full.py", "modal_eval_full.py",
           "modal_average_full.py")


# --------------------------------------------------------------- layout
def parse_alphas(text):
    """'0,0.4,0.7,1.0' -> [0.0, 0.4, 0.7, 1.0]; ValueError unless finite and
    unique (as formatted)."""
    try:
        alphas = [float(x) for x in str(text).split(",")]
    except ValueError:
        raise ValueError(f"--alphas must be comma-separated numbers, got "
                         f"{text!r}")
    if any(not math.isfinite(a) for a in alphas):
        raise ValueError(f"--alphas must be finite, got {text!r}")
    alphas = [0.0 if a == 0 else a for a in alphas]
    names = [format_alpha(a) for a in alphas]
    if len(set(names)) != len(names):
        raise ValueError(f"--alphas must be unique, got {names}")
    return alphas


def format_alpha(alpha):
    """build_transplant.format_alpha without numpy: repr, -0.0 -> 0.0."""
    a = float(alpha)
    return repr(0.0 if a == 0 else a)


def transplant_key(prefix, run, step, alpha):
    return (f"{prefix}/{run}/transplant/turbo_{run}_{int(step)}_"
            f"a{format_alpha(alpha)}.safetensors")


def eval_label(alpha):
    """--models label: 0.0 -> t00, 0.4 -> t04, 1.0 -> t10."""
    return "t" + format_alpha(alpha).replace(".", "").replace("-", "m")


def eval_models_line(items):
    return "--models " + ",".join(f"{eval_label(i['alpha'])}={i['key']}"
                                  for i in items)


def turbo_command(run, step):
    return (f"modal run scripts/modal_delta_compare.py --run {run} --step "
            f"{int(step)} --no-extra")


def input_identities(run, step, vol=VOL):
    """The three volume inputs -> ({turbo, raw, ft} paths, {turbo, raw, ft}
    identities); RuntimeError naming the command that creates a missing
    one. Reads markers, sidecars and safetensors metadata only."""
    turbo = Path(vol) / mdc.TURBO_DIR / mdc.TURBO_FILE
    turbo_id = mdc.turbo_identity(vol)
    if turbo_id is None:
        raise RuntimeError(f"{turbo} missing or its download marker "
                           "incomplete; run first (caches Krea-2-Turbo):\n  "
                           + turbo_command(run, step))
    raw, ft, _ = mdc.check_inputs(run, step, SOURCE, [], vol)
    raw_id = mef.hf_base_identity(str(raw.parent))
    side = mxf._read_json(mxf.export_source_path(ft))
    if (not isinstance(side, dict) or side.get("size") != ft.stat().st_size
            or side.get("step") != int(step) or side.get("run") != run):
        raise RuntimeError(f"{ft}: no matching sidecar "
                           f"{mxf.export_source_path(ft).name}; run first:\n"
                           "  " + mdc.export_command(run, step, SOURCE))
    return ({"turbo": turbo, "raw": raw, "ft": ft},
            {"turbo": turbo_id, "raw": raw_id, "ft": side})


def inputs_fingerprint(idents, alpha):
    obj = {"method": METHOD, "alpha": format_alpha(alpha),
           **{n: idents[n] for n in ("turbo", "raw", "ft")}}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def metadata_labels(paths, idents, step):
    """{turbo, raw, ft} strings of the output's safetensors metadata."""
    def hf(i):
        return f"{i.get('repo')}@{i.get('revision')}"
    return {"turbo": hf(idents["turbo"]), "raw": hf(idents["raw"]),
            "ft": f"{paths['ft']} size {idents['ft']['size']} step "
                  f"{int(step)}"}


# --------------------------------------------------------------- flow
def transplant_impl(alphas, run=DEFAULT_RUN, step=DEFAULT_STEP,
                    prefix=R2_PREFIX, force=False, dry_run=False, env=None,
                    client=None, vol=VOL, tmp_dir=None, transfer_config=None,
                    log=print):
    """The remote flow (module docstring) -> {"run", "step", "status",
    "items": [{"alpha", "key", "fingerprint", "output", "status", ...}]}."""
    env = os.environ if env is None else env
    paths, idents = input_identities(run, step, vol)
    client = client or r2_client(env)
    bucket = env["R2_BUCKET"]
    items = []
    for a in alphas:
        key, fp = transplant_key(prefix, run, step, a), inputs_fingerprint(
            idents, a)
        head = head_or_none(client, bucket, key)
        state = ("absent" if head is None else "exists"
                 if (head.get("Metadata") or {}).get("inputs-fingerprint")
                 == fp else "conflict")
        items.append({"alpha": a, "key": key, "fingerprint": fp,
                      "output": state, "head": head})
    conflicts = [i["key"] for i in items if i["output"] == "conflict"]
    if conflicts and not force and not dry_run:
        raise RuntimeError(f"r2: {conflicts} exist with other inputs "
                           "(inputs-fingerprint differs); pass --force to "
                           "replace them")
    import build_transplant as bt   # numpy: container only

    with bt._opened([paths[n] for n in bt.INPUTS]) as hs:
        spec = bt.check_headers(hs)
    out_bytes = sum(bt._nbytes(*v) for v in spec.values())
    labels = metadata_labels(paths, idents, step)
    log(f"plan: {len(spec)} tensors, {out_bytes / 2**30:.1f} GiB per output; "
        f"inputs {json.dumps({n: str(p) for n, p in paths.items()})}; "
        + "; ".join(f"a={format_alpha(i['alpha'])} -> r2:{i['key']} "
                    f"({i['output']})" for i in items))
    res = {"run": run, "step": int(step), "tensors": len(spec),
           "output_bytes": out_bytes, "labels": labels, "items": items}
    if dry_run:
        for i in items:
            i.pop("head")
            i["status"] = "dry-run"
        return {**res, "status": "dry-run"}

    todo = [i for i in items if force or i["output"] != "exists"]
    for i in items:
        if i not in todo:
            log(f"r2:{i['key']}: exists, made from these inputs "
                f"(fingerprint {i['fingerprint'][:12]}); skipped (--force to "
                "redo)")
            i.update(status="exists", **r2_identity(i.pop("head"), i["key"]))
    if todo:
        t0 = time.time()
        bt.check_inputs(paths["turbo"], paths["raw"], paths["ft"],
                        threads=BUILD_CPU, log=log)
        log(f"inputs finite ({time.time() - t0:.0f} s)")
    for i in todo:
        i.pop("head")
        tmp = Path(tempfile.mkdtemp(dir=tmp_dir, prefix="transplant_"))
        try:
            path = tmp / Path(i["key"]).name
            built = bt.build(paths["turbo"], paths["raw"], paths["ft"],
                             i["alpha"], path, labels=labels, checked=True,
                             threads=BUILD_CPU, log=log)
            log(f"built a={format_alpha(i['alpha'])}: {built['bytes']} bytes "
                f"in {built['seconds']:.0f} s")
            _, now = input_identities(run, step, vol)
            if inputs_fingerprint(now, i["alpha"]) != i["fingerprint"]:
                raise RuntimeError("the volume inputs changed during the "
                                   f"build of {i['key']}; not uploaded")
            ident = mav.upload_average(
                client, bucket, i["key"], path,
                {"inputs-fingerprint": i["fingerprint"],
                 "alpha": format_alpha(i["alpha"])},
                transfer_config=transfer_config, log=log)
            i.update(status="uploaded", seconds=round(built["seconds"], 1),
                     **ident)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return {**res, "status": "done"}


# --------------------------------------------------------------- CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(
        prog="modal run scripts/modal_build_transplant.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphas", required=True,
                    help="comma-separated alphas, e.g. 0,0.4,0.7,1.0")
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help=f"run of the fp32 export (default {DEFAULT_RUN})")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP,
                    help=f"checkpoint step (default {DEFAULT_STEP})")
    ap.add_argument("--prefix", default=R2_PREFIX,
                    help=f"R2 prefix of the runs (default {R2_PREFIX})")
    ap.add_argument("--force", action="store_true",
                    help="rebuild and replace existing output objects")
    ap.add_argument("--dry-run", action="store_true",
                    help="checks + plan only: no tensor reads, no upload")
    return ap


def parse_args(argv):
    """argparse + validation -> args with .alphas (floats)."""
    ap = build_arg_parser()
    args = ap.parse_args(list(argv))
    try:
        args.alphas = parse_alphas(args.alphas)
    except ValueError as e:
        ap.error(str(e))
    if not mxf.NAME_RE.match(args.run):
        ap.error(f"--run must match {mxf.NAME_RE.pattern}, got {args.run!r}")
    if not all(mxf.NAME_RE.match(p) for p in args.prefix.split("/")):
        ap.error(f"--prefix must be '/'-separated names, got {args.prefix!r}")
    if args.step < 0:
        ap.error(f"--step must be >= 0, got {args.step}")
    for a in args.alphas:
        if not mef.LABEL_RE.match(eval_label(a)):
            ap.error(f"alpha {a!r} gives no valid eval label")
    return args


# ------------------------------------------------------------------ Modal app
if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*mxf.EXPORT_PIP)
        .env(mxf.EXPORT_ENV)
        .add_local_dir(LOCAL_REPO / "krea2_jax", f"{REMOTE_REPO}/krea2_jax",
                       ignore=["**/__pycache__"])
    )
    for _s in SCRIPTS:
        image = image.add_local_file(LOCAL_REPO / "scripts" / _s,
                                     f"{REMOTE_REPO}/scripts/{_s}")
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    r2_secret = modal.Secret.from_name("krea2-r2")
    app = modal.App("krea2-build-transplant")

    @app.function(image=image, cpu=float(BUILD_CPU), timeout=BUILD_TIMEOUT_S,
                  volumes={VOL: vol}, secrets=[r2_secret])
    def transplant_remote(alphas: list, run: str, step: int, prefix: str,
                          force: bool, dry_run: bool) -> dict:
        return mav.scrubbed_call(
            transplant_impl, alphas, run, step, prefix=prefix, force=force,
            dry_run=dry_run, log=lambda m: print(m, flush=True))

    @app.local_entrypoint()
    def main(*argv: str):
        # runs in the Modal CLI's Python (no numpy there): only printing
        args = parse_args(argv)
        t0 = time.time()
        res = transplant_remote.remote(args.alphas, args.run, args.step,
                                       args.prefix, args.force, args.dry_run)
        print(f"remote: {(time.time() - t0) / 60:.1f} min")
        print(json.dumps(res, indent=1))
        for i in res["items"]:
            print(f"a={format_alpha(i['alpha'])}: {i['status']} r2:{i['key']}")
        print(f"\neval: {eval_models_line(res['items'])}")


def local_main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    raise SystemExit("this script runs on Modal: modal run "
                     "scripts/modal_build_transplant.py --alphas "
                     f"{','.join(format_alpha(a) for a in args.alphas)} "
                     f"--run {args.run} --step {args.step} ...")


if __name__ == "__main__":
    local_main()
