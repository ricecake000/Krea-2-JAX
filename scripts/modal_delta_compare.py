"""Modal app (CPU only): direction of ΔTurbo = Turbo - Raw against ΔFT =
<run> - Raw (fp32 export of the run's raw params) and optional extra deltas
(default run2's bf16 EMA export), over every tensor of the Krea 2 DiT.

    modal run scripts/modal_delta_compare.py                 # run3 9192 + run2
    modal run scripts/modal_delta_compare.py --run run3 --step 9192 --no-extra
    python scripts/modal_delta_compare.py --help

Options (local entrypoint, argparse):
  --run NAME        run of the fp32 export (default run3)
  --step N          its checkpoint step (default 9192)
  --source S        params (default) or ema: the exported tree
  --extra L=PATH    another delta, a safetensors on the volume (/vol/...);
                    repeatable; default run2=/vol/ft/run2/model.safetensors
  --no-extra        no extra deltas (only turbo and the run)
  --name NAME       report name (default turbo_<run>_<step>)
  --out PATH        local JSON (default
                    /home/mj/krea2-ops/eval/delta_<name>.json)

One function (`delta_remote`, DELTA_CPU cores, no memory / disk
reservation) on the Volume `krea2-eval-cache` of scripts/modal_eval_full.py,
with the `huggingface` secret:
  0. checks (before any download): /vol/base/raw.safetensors (cached by
     modal_eval_full's prepare), the fp32 export
     /vol/exports/<run>_<step>_<source>_f32.safetensors of
     scripts/modal_export_full.py (--dtype f32; the error names the command
     to run first) and every --extra file.
  1. krea/Krea-2-Turbo turbo.safetensors -> /vol/turbo/turbo.safetensors,
     cached with modal_eval_full's hf_fetch and download marker (MARKER,
     marker_ok; the repo's current commit, recorded in the marker); skipped
     when complete. Its own directory: the markers are per directory, and
     /vol/base/.eval_done.json belongs to raw.safetensors.
  2. scripts/delta_compare.py over base raw and the deltas turbo (reference
     axis), <run>, extras: streamed, chunked float64 per-tensor sums;
     energy-weighted aggregates (ALL, per kind, per block group, per base
     storage dtype BF16 / F32), cos^2, projection coefficients, noise floor;
     progress logged every ~30 s. Saved to /vol/delta/<name>.json (with the
     inputs' identities; on a late error the rows computed so far go to
     /vol/delta/<name>.json.partial), returned, written locally, printed.

Importable without the Modal SDK and without numpy (the local entrypoint
runs in the Modal CLI's Python): scripts/modal_export_full.py and
scripts/modal_eval_full.py are imported as helper libraries with `modal`
hidden, so they do not define their own apps inside this one.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

try:
    import modal
except ImportError:  # importable (pyflakes, tests) without the Modal SDK
    modal = None

LOCAL_REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/repo"
# scripts/ locally; /repo/scripts (attached by the image) in the container,
# where this file is mounted on its own
for _repo in (str(LOCAL_REPO), REMOTE_REPO):
    _scripts = os.path.join(_repo, "scripts")
    if (_scripts not in sys.path
            and os.path.isfile(os.path.join(_scripts, "delta_compare.py"))):
        sys.path.insert(0, _scripts)


def _import_hidden(name):
    """scripts/<name>.py as a helper library. `modal` is hidden while it is
    imported for the first time, so it does not define its own Modal app
    inside this one (the pattern of scripts/modal_coherence.py)."""
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


mxf = _import_hidden("modal_export_full")
mef = _import_hidden("modal_eval_full")
import delta_compare as dc  # noqa: E402 - needs the sys.path above

LOCAL_OUT_ROOT = mxf.LOCAL_OUT_ROOT
VOLUME_NAME, VOL, BASE_REL = mxf.VOLUME_NAME, mxf.VOL, mxf.BASE_REL
TURBO_REPO, TURBO_FILE = "krea/Krea-2-Turbo", "turbo.safetensors"
TURBO_DIR = "turbo"                  # /vol/turbo/turbo.safetensors
DELTA_DIR = "delta"                  # /vol/delta/<name>.json
DEFAULT_RUN, DEFAULT_STEP = "run3", 9192
DEFAULT_EXTRA = (f"run2={VOL}/ft/run2/model.safetensors",)

# --- Modal resources ---
DELTA_CPU = 8
DELTA_TIMEOUT_S = 3 * 3600
HEARTBEAT_S = 30.0
_PINNED = ("safetensors", "ml_dtypes", "numpy")
DELTA_PIP = ([p for p in mxf.EXPORT_PIP if p.split("==")[0] in _PINNED]
             + [p for p in mef.CPU_PIP
                if p.split("==")[0].split("[")[0] in ("huggingface_hub",
                                                      "hf_xet")])
DELTA_ENV = {k: mef.CPU_ENV[k]
             for k in ("HF_HOME", "HF_XET_CACHE", "PYTHONUNBUFFERED")}


# --------------------------------------------------------------- layout
def f32_export_path(run, step, source, vol=VOL):
    """The fp32 export of scripts/modal_export_full.py --dtype f32."""
    return mxf.local_paths(vol, run, step, source, "f32")["export"]


def export_command(run, step, source):
    return (f"modal run scripts/modal_export_full.py --run {run} --step "
            f"{int(step)} --source {source} --dtype f32 --no-upload")


def check_inputs(run, step, source, extras, vol=VOL):
    """The volume inputs that need no download: -> (base, f32 export,
    [(label, path)] extras); RuntimeError naming what to run first."""
    base = Path(vol) / BASE_REL
    if not (base.is_file() and mef.marker_ok(base.parent)):
        raise RuntimeError(f"{base} missing or incomplete: run `modal run "
                           "scripts/modal_eval_full.py --limit 1` (its "
                           "prepare caches the HF base) first")
    exp = f32_export_path(run, step, source, vol)
    if not mxf.export_matches(exp, step, source, "f32"):
        raise RuntimeError(f"{exp}: no fp32 export of {run} step {step} "
                           f"({source}); run first:\n  "
                           + export_command(run, step, source))
    extras = [dc.parse_delta(e) for e in extras]
    missing = [p for _, p in extras if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"--extra files not on the volume: {missing}")
    return base, exp, extras


# --------------------------------------------------------------- turbo
def _dir_bytes(path):
    """Bytes under path, partial downloads included."""
    total = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:   # renamed / removed while walking
            pass
    return total


def ensure_turbo(vol=VOL, env=None, commit=lambda: None, log=print,
                 every=HEARTBEAT_S):
    """krea/Krea-2-Turbo turbo.safetensors in <vol>/turbo/ (hf_fetch of
    modal_eval_full: skipped when its marker is complete), logging the
    bytes on disk every `every` s while it downloads. -> (path, marker)."""
    env = os.environ if env is None else env
    dest = Path(vol) / TURBO_DIR
    _, token = mef.pick_hf_token(env)
    stop = threading.Event()

    def beat():
        t0 = time.time()
        while not stop.wait(every):
            log(f"{TURBO_REPO}: {_dir_bytes(dest) / 2**30:.1f} GiB in {dest} "
                f"({time.time() - t0:.0f} s)")

    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        status = mef.hf_fetch(TURBO_REPO, dest, [TURBO_FILE], token, log=log)
    finally:
        stop.set()
        th.join()
    if status != "skipped":
        commit()
    marker = json.loads((dest / mef.MARKER).read_text())
    if TURBO_FILE not in marker.get("files", {}):
        raise RuntimeError(f"{dest}: download marker does not list "
                           f"{TURBO_FILE}")
    return dest / TURBO_FILE, {k: marker.get(k)
                               for k in ("repo", "revision", "files")}


# --------------------------------------------------------------- flow
def delta_impl(run, step, source, extras, name, vol=VOL, env=None,
               commit=lambda: None, log=print):
    """The whole remote flow (see the module docstring) -> the report."""
    base, exp, extras = check_inputs(run, step, source, extras, vol)
    labels = ["turbo", run] + [a for a, _ in extras]
    if len(set(labels)) != len(labels):
        raise RuntimeError(f"delta labels must be distinct, got {labels}")
    turbo, turbo_marker = ensure_turbo(vol, env, commit, log)
    deltas = [("turbo", turbo), (run, exp)] + extras
    out = Path(vol) / DELTA_DIR / f"{name}.json"
    try:   # a late error keeps the computed rows in <name>.json.partial
        report = dc.delta_compare(
            base, deltas, log=log,
            partial_path=out.with_name(out.name + ".partial"))
    except BaseException:
        commit()
        raise
    report["name"] = name
    report["inputs"] = {
        "base": mef.hf_base_identity(str(base.parent)),
        "turbo": turbo_marker,
        run: mxf._read_json(mxf.export_source_path(exp)),
    }
    mxf._write_json(out, report)
    commit()
    log(dc.format_tables(report))
    report["report_path"] = str(out)
    return report


# --------------------------------------------------------------- CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(
        prog="modal run scripts/modal_delta_compare.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help=f"run of the fp32 export (default {DEFAULT_RUN})")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP,
                    help=f"checkpoint step (default {DEFAULT_STEP})")
    ap.add_argument("--source", choices=mxf.SOURCES, default="params",
                    help="exported tree (default params)")
    ap.add_argument("--extra", action="append", default=None,
                    metavar="LABEL=PATH",
                    help="another delta on the volume (/vol/...); repeatable "
                         f"(default {' '.join(DEFAULT_EXTRA)})")
    ap.add_argument("--no-extra", action="store_true",
                    help="no extra deltas")
    ap.add_argument("--name", default="",
                    help="report name (default turbo_<run>_<step>)")
    ap.add_argument("--out", default="",
                    help="local JSON (default "
                         f"{LOCAL_OUT_ROOT}/delta_<name>.json)")
    return ap


def parse_args(argv):
    """argparse + validation (names, step, /vol/ extra paths, labels)."""
    ap = build_arg_parser()
    args = ap.parse_args(list(argv))
    if args.no_extra and args.extra:
        ap.error("--no-extra and --extra are exclusive")
    args.extra = [] if args.no_extra else list(args.extra or DEFAULT_EXTRA)
    args.name = args.name or f"turbo_{args.run}_{args.step}"
    for what, v in (("--run", args.run), ("--name", args.name)):
        if not mxf.NAME_RE.match(v):
            ap.error(f"{what} must match {mxf.NAME_RE.pattern}, got {v!r}")
    if args.step < 0:
        ap.error(f"--step must be >= 0, got {args.step}")
    labels = ["turbo", args.run]
    for e in args.extra:
        try:
            label, path = dc.parse_delta(e)
        except ValueError as err:
            ap.error(str(err).replace("--delta", "--extra"))
        parts = path.split("/")
        if not path.startswith(VOL + "/") or any(p in ("", ".", "..")
                                                 for p in parts[1:]):
            ap.error(f"--extra must be a file on the volume ({VOL}/...), "
                     f"got {path!r}")
        labels.append(label)
    if len(set(labels)) != len(labels):
        ap.error(f"delta labels must be distinct, got {labels}")
    return args


def write_local(report, out=""):
    path = Path(out) if out else LOCAL_OUT_ROOT / f"delta_{report['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    return path


# ------------------------------------------------------------------ Modal app
if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*DELTA_PIP)
        .env(DELTA_ENV)
        .add_local_dir(LOCAL_REPO / "krea2_jax", f"{REMOTE_REPO}/krea2_jax",
                       ignore=["**/__pycache__"])
        .add_local_file(LOCAL_REPO / "scripts" / "delta_compare.py",
                        f"{REMOTE_REPO}/scripts/delta_compare.py")
        .add_local_file(LOCAL_REPO / "scripts" / "modal_export_full.py",
                        f"{REMOTE_REPO}/scripts/modal_export_full.py")
        .add_local_file(LOCAL_REPO / "scripts" / "modal_eval_full.py",
                        f"{REMOTE_REPO}/scripts/modal_eval_full.py")
    )
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    hf_secret = modal.Secret.from_name("huggingface")
    app = modal.App("krea2-delta-compare")

    @app.function(image=image, cpu=float(DELTA_CPU), timeout=DELTA_TIMEOUT_S,
                  volumes={VOL: vol}, secrets=[hf_secret])
    def delta_remote(run: str, step: int, source: str, extras: list,
                     name: str) -> dict:
        return delta_impl(run, step, source, extras, name, commit=vol.commit,
                          log=lambda m: print(m, flush=True))

    @app.local_entrypoint()
    def main(*argv: str):
        # runs in the Modal CLI's Python (no numpy there): only printing
        args = parse_args(argv)
        t0 = time.time()
        rep = delta_remote.remote(args.run, args.step, args.source,
                                  args.extra, args.name)
        print(f"remote: {(time.time() - t0) / 60:.1f} min")
        local = write_local(rep, args.out)   # before any formatting
        print(dc.format_tables(rep))
        print(f"\nvolume: {rep['report_path']}")
        print(f"wrote {local}")


def local_main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    raise SystemExit("this script runs on Modal: modal run "
                     f"scripts/modal_delta_compare.py --run {args.run} "
                     f"--step {args.step} ...")


if __name__ == "__main__":
    local_main()
