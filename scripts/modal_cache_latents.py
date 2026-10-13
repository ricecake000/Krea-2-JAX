"""Modal app: benchmark Modal GPU types for the VAE latent cache, then build
the `krea2-latents-v1` cache (scripts/cache_latents.py) on the chosen type and
upload it to a private HF dataset repo the training node can `hf download`.

    modal run scripts/modal_cache_latents.py --bench                  # all GPU types
    modal run scripts/modal_cache_latents.py --bench --gpus L4,L40S,B200 --n-decode 128
    modal run --detach scripts/modal_cache_latents.py --build --gpu B200 \\
        --containers 4 --batch 16 --cpu 12 --worker-timeout-h 4
    modal run scripts/modal_cache_latents.py --finalize-only          # manifest + upload again
    modal run scripts/modal_cache_latents.py --finalize-only --reset-upload-cache

Mode 1 (--bench): `prep` (CPU) puts metadata.parquet + data/train-0000.tar of
ricecake/danbooru-curated-v2 into /vol/bench and the Qwen-Image VAE into
HF_HOME=/vol/hf, and reports whether the HF token can write. Then one
`bench_gpu` container per --gpus type runs concurrently: decode --n-decode
832x1216 samples (the most common bucket, 32% of the data; every bucket is
~1 MP so the per-image cost is near-constant) with cache_latents.decode, time
single-threaded decode, then a batch ladder 4/8/16/32 of
cache_latents.vae_encode_batch on the CPU float32 tensors exactly like
production (stack + H2D + encode + bf16 D2H), each >= max(64 images, 3
batches) and >= 10 s, stopping at the first OOM. A GPU type that is
unavailable or fails becomes an error row; the others still report. Locally:
cost per image (GPU + the CPU cores needed to keep it fed + 32 GiB, Modal
prices of PRICES_DATE), the selection (lowest $/img; anything within 5% of
it is a tie, broken by higher img/s) and the exact build command.
Output: /home/mj/krea2-ops/mi350x/results/modal/cache_bench_<ts>/{report.md,bench.json}.
Rough cost of a full bench: ~5 min per GPU type, ~$3 for all ten.

Mode 2 (--build): `prep` (VAE only; aborts before any GPU starts when the
token is clearly read-only), then --containers `build_worker` containers on
--gpu (Modal timeout --worker-timeout-h), worker i running `cache_latents.py
--num-workers N --worker-index i` into /vol/krea2 (shards i, i+N, ...), then
`finalize` (CPU): HF write check + create the repo FIRST and refuse a repo
that is not private, `cache_latents.py --write-manifest` (verifies every
shard, sha256 digests), upload_large_folder to
ricecake/danbooru-curated-v2-latents and a content check of every uploaded
file (shard LFS sha256 == manifest digest; parquet/manifest sha256 or git
blob id == local). A failed worker stops before finalize. Everything is
resumable: cache_latents skips finished shards, --write-manifest re-verifies
the digests, upload_large_folder resumes from its state in
/vol/krea2/.cache/huggingface (--reset-upload-cache deletes that state
first, e.g. after a failed remote content check: everything is re-hashed
and re-sent). The orchestration runs in the local entrypoint, so if the
laptop disconnects during a --detach run the workers finish but finalize
does not start: re-run with --finalize-only (or --build, which skips the
done shards). Never start a second --build while one is still running (two
workers would write the same shards). Output:
/home/mj/krea2-ops/mi350x/results/modal/cache_build_<ts>.md.

Volume `krea2-latents` (Volumes v2, created on first use):
    /vol/hf       HF_HOME (VAE weights downloaded once)
    /vol/bench    metadata.parquet, data/train-0000.tar (bench input)
    /vol/krea2    latents/train-XXXX.safetensors, then metadata.parquet +
                  latents/manifest.json (the cache; uploaded as the repo root)
Modal secret `huggingface`: the HF token under HF_TOKEN,
HUGGING_FACE_HUB_TOKEN or HUGGINGFACE_TOKEN (exported as HF_TOKEN).

Importable without the Modal SDK (the app is only defined when `modal`
imports); pricing, cost model, selection, report rendering, worker command,
streaming/commit loop and upload checks are plain functions
(tests/test_scripts_smoke.py::modal_cache_latents_flow).
"""

import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

try:
    import modal
except ImportError:  # importable (pyflakes, tests) without the Modal SDK
    modal = None

LOCAL_REPO = Path(__file__).resolve().parent.parent
LOCAL_RESULTS = Path("/home/mj/krea2-ops/mi350x/results/modal")
REMOTE_REPO = "/repo"

DATASET_REPO = "ricecake/danbooru-curated-v2"
OUT_OWNER = "ricecake"
OUT_REPO = f"{OUT_OWNER}/danbooru-curated-v2-latents"
VAE_REPO = "Qwen/Qwen-Image"
VAE_PATTERNS = ["vae/*", "model_index.json"]
N_IMAGES = 96_848

VOLUME_NAME = "krea2-latents"
VOLUME_VERSION = 2   # Volumes v2 (beta; the cache is rebuildable): many
#                      concurrent writers, faster commits
VOL = "/vol"
VOL_HF = f"{VOL}/hf"
VOL_BENCH = f"{VOL}/bench"
VOL_OUT = f"{VOL}/krea2"
REMOTE_DL = "/tmp/dl"   # tar downloads: container-local disk, never the volume
PARQUET = "metadata.parquet"
BENCH_TAR = "data/train-0000.tar"
BENCH_BUCKET = (832, 1216)   # (bucket_w, bucket_h): 143 samples in train-0000
LAT_CHANNELS = 16
VAE_SCALE = 8

# bench ladder
LADDER = (4, 8, 16, 32)
MIN_IMAGES = 64
MIN_SECONDS = 10.0
DECODE_TIMING_N = 32     # single-threaded decode timing sample
DECODE_POOL = 16         # threads decoding the bench set (8 physical cores)
BENCH_CPU = 8
BENCH_WAIT_S = 45 * 60   # local wait for all bench containers (incl. queueing)
DEFAULT_GPUS = ("L4,A10,L40S,A100-40GB,A100-80GB,RTX-PRO-6000,H100!,H200,"
                "B200,B300")
TIE_FRAC = 0.05
DEFAULT_OVERHEAD_MIN = 3.0   # startup + VAE load per container
CONTAINER_COUNTS = (1, 4)    # projected cost/wall-clock columns
DEFAULT_CONTAINERS = 4
DEFAULT_WORKER_TIMEOUT_H = 4
MAX_WORKER_TIMEOUT_H = 24    # Modal's maximum function timeout

# Modal pricing (modal.com/pricing, fetched 2026-10-13); $/h = per-second x
# 3600. Keys are Modal gpu strings (upper case; "H100!" = H100 without the
# H200 upgrade, "B200+" = B200 or B300 billed as B200). T4 excluded: no bf16.
PRICES_DATE = "2026-10-13"
GPU_PRICE_USD_H = {
    "B300": 7.10, "B200": 6.25, "H200": 4.54, "H100": 3.95,
    "RTX-PRO-6000": 3.03, "A100-80GB": 2.50, "A100-40GB": 2.10,
    "L40S": 1.95, "A10": 1.10, "L4": 0.80,
}
CPU_CORE_USD_H = 0.0472   # per physical core (Modal `cpu=` counts physical)
MEM_GIB_USD_H = 0.00799
MEM_GIB = 32
MEM_MB = MEM_GIB * 1024

# image pins: torch cu132 (CUDA 13.2 >= the 13.1 B300 needs; Blackwell
# sm_100/sm_103/sm_120 kernels; runs on Modal's 13.0 host driver through
# CUDA minor-version compatibility). diffusers 0.40 has
# AutoencoderKLQwenImage; transformers is not installed (the VAE path never
# imports it, and 4.57.x pins huggingface_hub<1.0).
TORCH_PIN = "torch==2.13.0+cu132"
TORCH_INDEX = "https://download.pytorch.org/whl/cu132"
PIP_PACKAGES = [
    "diffusers==0.40.0", "accelerate==1.15.0", "safetensors==0.8.0",
    "pillow==12.3.0", "pyarrow==26.0.0", "numpy==2.5.1",
    "huggingface_hub[hf_xet]==1.24.0", "hf_xet==1.5.2",
]
CONTAINER_ENV = {
    "HF_HOME": VOL_HF,
    "HF_XET_CACHE": "/tmp/hf_xet",   # keep xet chunk caches off the volume
    "PYTHONUNBUFFERED": "1",
}

# upload: fnmatch patterns on the path relative to /vol/krea2 (same semantics
# as huggingface_hub's filter_repo_objects: `*` crosses `/`)
UPLOAD_IGNORE = ["_download/*", "*.tmp", ".cache/*"]
UPLOAD_WORKERS = 8
TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
WRITE_HELP = (
    "the HF token in the Modal secret `huggingface` cannot write "
    f"{OUT_REPO} ({{reason}}). Put a WRITE token for user {OUT_OWNER} in it: "
    "`modal secret create huggingface HF_TOKEN=hf_... --force`, then re-run "
    "with --finalize-only (finished shards are kept on the volume).")
# cache_latents.py's per-shard progress line, e.g.
# "[w0/4] 3/25 train-0008: 968 imgs in 101.2s (9.57 img/s), ..."
SHARD_LINE_RE = re.compile(r"^\[w\d+/\d+\] \d+/\d+ \S+: \d+ imgs in ")


# ------------------------------------------------------------- pricing / cost
def gpu_key(gpu):
    """Modal gpu string -> GPU_PRICE_USD_H key ('h100!' -> 'H100',
    'B200+' -> 'B200'). Raises SystemExit for unknown types or counts."""
    key = str(gpu).strip().upper().rstrip("!+")
    if key not in GPU_PRICE_USD_H:
        raise SystemExit(f"unknown GPU {gpu!r} (one GPU per container; "
                         f"known: {', '.join(GPU_PRICE_USD_H)})")
    return key


def parse_gpus(s):
    """'L4, A10,H100!' -> ['L4', 'A10', 'H100!'] (validated, de-duplicated)."""
    out = []
    for g in str(s).split(","):
        g = g.strip().upper()
        if g:
            gpu_key(g)
            if g not in out:
                out.append(g)
    if not out:
        raise SystemExit(f"--gpus is empty: {s!r}")
    return out


def cores_needed(img_s, decode_img_s_per_thread):
    """Physical cores that keep the GPU fed: 1.5x the decode threads needed
    for img_s, +1 for the main/download thread, at least 4."""
    return max(4, math.ceil(1.5 * img_s / decode_img_s_per_thread) + 1)


def hourly_usd(gpu, cores, mem_gib=MEM_GIB):
    return (GPU_PRICE_USD_H[gpu_key(gpu)] + cores * CPU_CORE_USD_H
            + mem_gib * MEM_GIB_USD_H)


def best_of(ladder):
    """Fastest successful ladder row (error/OOM rows ignored) -> {best_batch,
    best_img_s, best_peak_gib} or {} when none succeeded."""
    ok = [r for r in ladder or [] if "error" not in r and r.get("img_s")]
    if not ok:
        return {}
    b = max(ok, key=lambda r: r["img_s"])
    return {"best_batch": b["batch"], "best_img_s": b["img_s"],
            "best_peak_gib": b.get("peak_gib")}


def cost_row(res, n_images=N_IMAGES, overhead_min=DEFAULT_OVERHEAD_MIN,
             containers=CONTAINER_COUNTS, mem_gib=MEM_GIB):
    """One bench_gpu result -> cost row, or None when it has no usable
    measurement (error, no successful batch, no decode rate).

    usd_per_1k = hourly / (img_s * 3600) * 1000; for k containers the
    projection is wall = N / img_s / 3600 / k + overhead and usd = hourly *
    (N / img_s / 3600 + k * overhead) (every container pays the overhead)."""
    img_s, dec = res.get("best_img_s"), res.get("decode_img_s_per_thread")
    if res.get("error") or not img_s or not dec:
        return None
    cores = cores_needed(img_s, dec)
    hourly = hourly_usd(res["gpu_requested"], cores, mem_gib)
    hours = n_images / img_s / 3600
    over = overhead_min / 60
    return {
        "gpu": res["gpu_requested"], "device_name": res.get("device_name"),
        "batch": res.get("best_batch"), "img_s": img_s,
        "peak_gib": res.get("best_peak_gib"), "decode": dec, "cores": cores,
        "usd_h": hourly, "usd_per_1k": hourly / (img_s * 3600) * 1000,
        "hours_1": hours, "overhead_h": over,
        "proj": {k: {"usd": hourly * (hours + k * over),
                     "wall_h": hours / k + over} for k in containers},
    }


def select_gpu(rows, tie_frac=TIE_FRAC):
    """Lowest $/img wins; every row within tie_frac of the lowest is a tie,
    broken by the higher img/s. None rows (failed GPUs) are ignored."""
    ok = [r for r in rows if r is not None]
    if not ok:
        return None
    lo = min(r["usd_per_1k"] for r in ok)
    ties = [r for r in ok if r["usd_per_1k"] <= lo * (1 + tie_frac)]
    return max(ties, key=lambda r: r["img_s"])


def worker_timeout_h(row, containers=DEFAULT_CONTAINERS):
    """Modal timeout for one build worker: ceil(1.5 x its projected hours
    (share of the images + startup overhead) + 0.5), at least 1."""
    per_worker = row["hours_1"] / containers + row["overhead_h"]
    return max(1, math.ceil(1.5 * per_worker + 0.5))


def build_plan(row, containers=DEFAULT_CONTAINERS):
    """-> (containers, worker timeout h): `containers` unless a worker would
    need more than Modal's MAX_WORKER_TIMEOUT_H; then the smallest count
    whose worker_timeout_h fits (the timeout is capped there)."""
    k = containers
    while worker_timeout_h(row, k) > MAX_WORKER_TIMEOUT_H:
        k += 1
    return k, min(worker_timeout_h(row, k), MAX_WORKER_TIMEOUT_H)


def build_command(row, containers=DEFAULT_CONTAINERS):
    k, t = build_plan(row, containers)
    return (f"modal run --detach scripts/modal_cache_latents.py --build "
            f"--gpu {row['gpu']} --containers {k} "
            f"--batch {row['batch']} --cpu {row['cores']} "
            f"--worker-timeout-h {t}")


def _ladder_str(ladder):
    return " ".join(f"{r['batch']}:{r['error']}" if "error" in r
                    else f"{r['batch']}:{r['img_s']:.2f}"
                    for r in ladder or [])


def _err_line(err):
    lines = [ln for ln in str(err).strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else "?"


def render_bench_report(results, rows, choice, prep=None,
                        overhead_min=DEFAULT_OVERHEAD_MIN,
                        containers=CONTAINER_COUNTS, n_images=N_IMAGES):
    """Markdown report: cost table, failed GPU types, the choice and the
    exact build command."""
    out = [f"# VAE latent cache: Modal GPU bench ({time.strftime('%Y-%m-%d %H:%M')})",
           "",
           f"Bucket {BENCH_BUCKET[0]}x{BENCH_BUCKET[1]}, {n_images} images, "
           f"prices of {PRICES_DATE} (cpu ${CPU_CORE_USD_H}/core-h, memory "
           f"{MEM_GIB} GiB at ${MEM_GIB_USD_H}/GiB-h), {overhead_min:g} min "
           "startup + VAE load per container. cores = max(4, ceil(1.5 * "
           "img/s / decode img/s/thread) + 1).", ""]
    if prep:
        out += [f"HF token: user {prep.get('whoami')!r}, can write "
                f"{prep.get('can_write')} ({prep.get('write_reason')})", ""]
    head = ["gpu", "device", "batch", "img/s", "peak GiB", "decode img/s/thr",
            "cores", "$/h", "$/1k img"]
    for k in containers:
        head += [f"{k} ctr $", f"{k} ctr wall h"]
    head.append("ladder (batch:img/s)")
    out += ["| " + " | ".join(head) + " |",
            "|" + "---|" * len(head)]
    by_gpu = {r["gpu"]: r for r in rows if r is not None}
    ordered = sorted(by_gpu.values(), key=lambda r: r["usd_per_1k"])
    ladders = {res["gpu_requested"]: res.get("ladder") for res in results}
    for r in ordered:
        peak = "-" if r["peak_gib"] is None else f"{r['peak_gib']:.1f}"
        cells = [r["gpu"], str(r["device_name"]), str(r["batch"]),
                 f"{r['img_s']:.2f}", peak, f"{r['decode']:.2f}",
                 str(r["cores"]), f"{r['usd_h']:.2f}",
                 f"{r['usd_per_1k']:.4f}"]
        for k in containers:
            p = r["proj"][k]
            cells += [f"{p['usd']:.2f}", f"{p['wall_h']:.2f}"]
        cells.append(_ladder_str(ladders.get(r["gpu"])))
        out.append("| " + " | ".join(cells) + " |")
    failed = [res for res in results if res["gpu_requested"] not in by_gpu]
    if failed:
        out += ["", "Not usable:"]
        for res in failed:
            why = (_err_line(res["error"]) if res.get("error")
                   else "no successful batch / decode rate")
            out.append(f"- {res['gpu_requested']} "
                       f"({res.get('device_name', '-')}): {why}"
                       + (f" [ladder {_ladder_str(res.get('ladder'))}]"
                          if res.get("ladder") else ""))
    out.append("")
    if choice is None:
        out.append("**No GPU type produced a usable measurement.**")
    else:
        lo = min(r["usd_per_1k"] for r in by_gpu.values())
        ties = [r["gpu"] for r in ordered
                if r["usd_per_1k"] <= lo * (1 + TIE_FRAC)]
        out += [f"**Chosen: {choice['gpu']}** ({choice['device_name']}): "
                f"${choice['usd_per_1k']:.4f}/1k img, {choice['img_s']:.2f} "
                f"img/s at batch {choice['batch']}, {choice['cores']} cores. "
                f"Lowest $/1k img {lo:.4f}; within {TIE_FRAC:.0%}: "
                f"{', '.join(ties)} (tie -> highest img/s).", "",
                "Build:", "", "    " + build_command(choice)]
        k, t = build_plan(choice)
        if k != DEFAULT_CONTAINERS:
            out += ["", f"(--containers raised from {DEFAULT_CONTAINERS} to "
                    f"{k}: with {DEFAULT_CONTAINERS} a worker would need "
                    f"{worker_timeout_h(choice)} h > Modal's "
                    f"{MAX_WORKER_TIMEOUT_H} h cap; now {t} h each)"]
    return "\n".join(out)


def write_bench_results(results, rows, choice, md, prep=None,
                        results_dir=LOCAL_RESULTS):
    d = Path(results_dir) / f"cache_bench_{time.strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.md").write_text(md + "\n")
    (d / "bench.json").write_text(json.dumps({
        "prices_date": PRICES_DATE, "gpu_price_usd_h": GPU_PRICE_USD_H,
        "cpu_core_usd_h": CPU_CORE_USD_H, "mem_gib_usd_h": MEM_GIB_USD_H,
        "prep": prep, "results": results, "rows": rows, "choice": choice,
    }, indent=2, default=str))
    return d


# --------------------------------------------------------------- HF token
def pick_hf_token(env):
    """-> (env var name, token) of the first non-empty HF token variable, or
    (None, None)."""
    for name in TOKEN_ENV_NAMES:
        if env.get(name):
            return name, env[name]
    return None, None


def setup_hf_token(env=None):
    """Export the secret's token as HF_TOKEN (for subprocesses too)."""
    env = os.environ if env is None else env
    name, token = pick_hf_token(env)
    if token:
        env["HF_TOKEN"] = token
    return name, token


def token_can_write(whoami, owner=OUT_OWNER):
    """HfApi.whoami() -> (True/False/None, reason): can this token create and
    write dataset repos under user `owner`? True only when clearly writable
    (role write/admin, or fine-grained repo.write on the user), False only
    when clearly not (role read, another user's token, or a fine-grained
    token with no repo write permission anywhere), None = cannot tell (e.g.
    fine-grained write on single repos only): create_repo decides."""
    who = whoami or {}
    name = who.get("name")
    tok = (who.get("auth") or {}).get("accessToken") or {}
    role = tok.get("role")
    if name is not None and name != owner:
        return False, f"token belongs to {name!r}, not {owner!r}"
    if role in ("write", "admin"):
        return True, f"token role {role}"
    if role == "read":
        return False, "token role read"
    if role == "fineGrained":
        fg = tok.get("fineGrained") or {}
        scoped = fg.get("scoped") or []
        for s in scoped:
            ent = s.get("entity") or {}
            if (ent.get("type") == "user" and ent.get("name") == owner
                    and "repo.write" in (s.get("permissions") or [])):
                return True, "fine-grained token with repo.write on " + owner
        perms = [p for s in scoped for p in s.get("permissions") or []]
        perms += list(fg.get("global") or [])
        writes = sorted({p for p in perms
                         if p.startswith("repo") and "write" in p})
        if not writes:
            return False, "fine-grained token without any repo write permission"
        return None, ("fine-grained token with repo write only on some "
                      f"entities ({', '.join(writes)})")
    return None, f"unknown token role {role!r} (user {name!r})"


# ------------------------------------------------------------- build helpers
def worker_cmd(i, n, batch, decode_threads, python="python", out=VOL_OUT,
               dl_root=REMOTE_DL):
    """cache_latents.py command of build worker i of n (run from the repo)."""
    return [python, "scripts/cache_latents.py", "--out", out,
            "--num-workers", str(n), "--worker-index", str(i),
            "--batch", str(batch), "--decode-threads", str(decode_threads),
            "--download-dir", f"{dl_root}/w{i}"]


def worker_env(base):
    """Subprocess env of a build worker: one intra-op CPU thread per decode
    thread. The VAE runs on the GPU, so torch's CPU threads only serve
    decode/normalize/stack; with OMP's default (all cores) every one of the
    --decode-threads would spawn its own core-wide pool and oversubscribe
    the cores the cost model sized as 1 core per decode thread."""
    return {**base, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1"}


def finalize_cmd(python="python", out=VOL_OUT, dl_root=REMOTE_DL):
    return [python, "scripts/cache_latents.py", "--out", out,
            "--write-manifest", "--download-dir", f"{dl_root}/manifest"]


def is_shard_line(line):
    return bool(SHARD_LINE_RE.match(line))


def stream_worker(cmd, on_shard, cwd=REMOTE_REPO, env=None, tail_n=40,
                  out=None):
    """Run `cmd`, echo its merged stdout/stderr line by line to `out`, and
    call on_shard() after every per-shard progress line (the shard file is
    complete on disk then). -> (rc, last tail_n lines, n shard lines)."""
    out = sys.stdout if out is None else out
    tail, n = deque(maxlen=tail_n), 0
    with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True,
                          bufsize=1) as p:
        for line in p.stdout:
            out.write(line)
            out.flush()
            tail.append(line.rstrip("\n"))
            if is_shard_line(line):
                n += 1
                on_shard()
        rc = p.wait()
    return rc, list(tail), n


def commit_volume(vol, log=print):
    """vol.commit(); a failure is logged, not raised (background commits and
    the commit at container shutdown still persist the files)."""
    try:
        vol.commit()
        return True
    except Exception as e:  # noqa: BLE001 - report, keep encoding
        log(f"WARNING: volume commit failed: {type(e).__name__}: {e}")
        return False


def summarize_workers(workers):
    """-> (all ok, one line per worker)."""
    lines, ok = [], True
    for w in sorted(workers, key=lambda w: w.get("worker", -1)):
        good = w.get("rc") == 0 and not w.get("error")
        ok &= good
        lines.append(
            f"w{w.get('worker')}: {'ok' if good else 'FAILED'} rc={w.get('rc')}"
            f" {w.get('device_name', '-')} shards={w.get('shards_written', '-')}"
            f" {w.get('seconds', 0) / 60:.1f} min"
            + (f" final_commit={w['final_commit']}" if "final_commit" in w
               else "")
            + (f" error: {_err_line(w['error'])}" if w.get("error") else ""))
    return ok, lines


def should_upload(relpath, ignore=UPLOAD_IGNORE):
    """Same rule as upload_large_folder(ignore_patterns=ignore)."""
    p = str(relpath).replace("\\", "/")
    return not any(fnmatch.fnmatchcase(p, pat) for pat in ignore)


def local_digests(path, chunk=1 << 20):
    """-> (sha256 hex, git blob sha1 hex) of a file, streamed. The git blob id
    is what the Hub reports for a file stored without LFS."""
    path = Path(path)
    sha, git = hashlib.sha256(), hashlib.sha1()
    git.update(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            sha.update(b)
            git.update(b)
    return sha.hexdigest(), git.hexdigest()


def expected_upload(out_dir):
    """{repo path: {size, sha256[, git_sha1]}} of what the repo must contain:
    metadata.parquet and latents/manifest.json (hashed here) and every shard
    the manifest lists (sha256 from its "digests", not re-read)."""
    out_dir = Path(out_dir)
    mpath = out_dir / "latents" / "manifest.json"
    manifest = json.loads(mpath.read_text())
    digests = manifest.get("digests") or {}
    exp = {}
    for p in (PARQUET, "latents/manifest.json"):
        sha, git = local_digests(out_dir / p)
        exp[p] = {"size": (out_dir / p).stat().st_size, "sha256": sha,
                  "git_sha1": git}
    for s in manifest["shards"]:
        exp[f"latents/{s}"] = {"size": (out_dir / "latents" / s).stat().st_size,
                               "sha256": digests.get(s)}
    return exp


def remote_entries(files):
    """list_repo_tree items -> {path: {size, sha256 (LFS oid or None),
    blob_id}}; folders (no blob_id) are skipped."""
    out = {}
    for f in files:
        if getattr(f, "blob_id", None) is None:
            continue
        lfs = getattr(f, "lfs", None)
        out[f.path] = {"size": f.size, "blob_id": f.blob_id,
                       "sha256": getattr(lfs, "sha256", None) if lfs else None}
    return out


def verify_upload(expected, remote):
    """-> list of problems: missing files, size mismatches, content
    mismatches (remote LFS sha256 vs expected sha256; for a non-LFS remote
    file its git blob id vs the local one; a file with neither is
    unverifiable = a problem). Extra remote files are fine."""
    problems = []
    for p, e in sorted(expected.items()):
        r = remote.get(p)
        if r is None:
            problems.append(f"missing {p}")
        elif r["size"] != e["size"]:
            problems.append(f"size {p}: remote {r['size']} != local "
                            f"{e['size']}")
        elif r.get("sha256"):
            if r["sha256"] != e.get("sha256"):
                problems.append(f"sha256 {p}: remote {r['sha256']} != "
                                f"expected {e.get('sha256')}")
        elif e.get("git_sha1"):
            if r.get("blob_id") != e["git_sha1"]:
                problems.append(f"git blob {p}: remote {r.get('blob_id')} != "
                                f"local {e['git_sha1']}")
        else:
            problems.append(f"unverifiable {p}: no LFS sha256 on the remote")
    return problems


def render_build_log(params, prep, workers, fin):
    ok, lines = summarize_workers(workers)
    out = [f"# VAE latent cache build ({time.strftime('%Y-%m-%d %H:%M')})", "",
           "params: " + json.dumps(params), "",
           f"prep: HF user {prep.get('whoami')!r}, can write "
           f"{prep.get('can_write')} ({prep.get('write_reason')})", "",
           "## workers", ""] + [f"- {ln}" for ln in lines]
    for w in sorted(workers, key=lambda w: w.get("worker", -1)):
        out += ["", f"### w{w.get('worker')} tail", "", "```"]
        out += list(w.get("tail") or []) + (
            [w["error"]] if w.get("error") else []) + ["```"]
    out += ["", "## finalize", ""]
    if fin is None:
        out.append("not run (a worker failed)" if not ok else "not run")
    else:
        out.append(f"ok={fin.get('ok')} repo={fin.get('repo')} "
                   f"files={fin.get('n_files')} "
                   f"GB={(fin.get('bytes') or 0) / 1e9:.2f}")
        if fin.get("error"):
            out += ["", "```", fin["error"], "```"]
        out += [f"- {p}" for p in (fin.get("problems") or [])[:50]]
        out += ["", "```"] + list(fin.get("log") or []) + ["```"]
    return "\n".join(out)


# ------------------------------------------------------------- bench helpers
def is_oom(e):
    return (type(e).__name__ == "OutOfMemoryError"
            or "out of memory" in str(e).lower())


def run_ladder(encode, imgs, batches, want_chw, cuda=None,
               min_images=MIN_IMAGES, min_seconds=MIN_SECONDS, log=print,
               intra_threads=1):
    """Throughput of `encode` over `imgs` (list of (3, H, W) CPU float32)
    per batch size: 1 warmup batch (shape + finiteness checked), then a
    timed loop of >= max(min_images, 3 batches) images and >= min_seconds,
    cycling over imgs. Each batch is stacked on the CPU and the latents come
    back as bf16 CPU tensors (as in cache_latents.encode_samples). `cuda` =
    torch.cuda (peak memory + synchronize) or None. Stops at the first
    failure; an OOM is recorded as error "OOM". Runs with `intra_threads`
    torch CPU threads (restored afterwards; None = unchanged): 1 matches the
    build workers' OMP_NUM_THREADS=1 (worker_env), which also covers the CPU
    torch.stack timed here."""
    import torch

    if intra_threads is None:
        return _run_ladder(torch, encode, imgs, batches, want_chw, cuda,
                           min_images, min_seconds, log)
    prev = torch.get_num_threads()
    torch.set_num_threads(intra_threads)
    try:
        return _run_ladder(torch, encode, imgs, batches, want_chw, cuda,
                           min_images, min_seconds, log)
    finally:
        torch.set_num_threads(prev)


def _run_ladder(torch, encode, imgs, batches, want_chw, cuda, min_images,
                min_seconds, log):
    rows, n = [], len(imgs)
    for b in batches:
        k = 0

        def nxt():
            nonlocal k
            x = torch.stack([imgs[(k * b + j) % n] for j in range(b)])
            k += 1
            return x
        try:
            if cuda is not None:
                cuda.empty_cache()
                cuda.reset_peak_memory_stats()
            lat = encode(nxt()).to(torch.bfloat16).cpu()  # warmup
            want = (b, *want_chw)
            if tuple(lat.shape) != want:
                raise ValueError(f"output {tuple(lat.shape)} != {want}")
            if not torch.isfinite(lat).all():
                raise ValueError("non-finite latents")
            if cuda is not None:
                cuda.synchronize()
            n_img, t0 = 0, time.perf_counter()
            while True:
                encode(nxt()).to(torch.bfloat16).cpu()
                n_img += b
                dt = time.perf_counter() - t0
                if n_img >= max(min_images, 3 * b) and dt >= min_seconds:
                    break
            if cuda is not None:
                cuda.synchronize()
            dt = time.perf_counter() - t0
            row = {"batch": b, "img_s": n_img / dt, "images": n_img,
                   "seconds": dt}
            if cuda is not None:
                row["peak_gib"] = cuda.max_memory_allocated() / 2**30
        except Exception as e:  # noqa: BLE001 - an error row, never a crash
            err = "OOM" if is_oom(e) else f"{type(e).__name__}: {e}"
            rows.append({"batch": b, "error": err})
            log(f"batch {b}: {err} -> ladder stops")
            break
        rows.append(row)
        log(f"batch {b}: {row['img_s']:.2f} img/s ({n_img} imgs, {dt:.1f}s"
            + (f", peak {row['peak_gib']:.1f} GiB" if "peak_gib" in row
               else "") + ")")
    return rows


def import_cache_latents(repo=REMOTE_REPO):
    sys.path.insert(0, f"{repo}/scripts")
    import cache_latents
    return cache_latents


def bench_gpu_impl(gpu_requested, n_decode):
    """Runs in the bench container. Never raises: failures end up in
    res["error"] (with whatever was measured before)."""
    res = {"gpu_requested": gpu_requested}
    t_start = time.time()
    try:
        from concurrent.futures import ThreadPoolExecutor

        import diffusers
        import torch

        cl = import_cache_latents()
        from cache_dataset import VaeEncoder

        res.update(device_name=torch.cuda.get_device_name(0),
                   torch=torch.__version__, torch_cuda=torch.version.cuda,
                   diffusers=diffusers.__version__)
        print(f"{gpu_requested}: {res['device_name']} torch {torch.__version__}"
              f" (cuda {torch.version.cuda})", flush=True)
        w, h = BENCH_BUCKET
        samples = [s for s in cl.read_tar(Path(VOL_BENCH) / BENCH_TAR)
                   if (s["w"], s["h"]) == (w, h)][:n_decode]
        if len(samples) < DECODE_TIMING_N:
            raise RuntimeError(f"only {len(samples)} {w}x{h} samples in "
                               f"{BENCH_TAR}")
        # one intra-op thread: the build runs every decode thread with
        # OMP_NUM_THREADS=1 (worker_env), so this is the per-core rate
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            t0 = time.perf_counter()
            for s in samples[:DECODE_TIMING_N]:
                cl.decode(s)
            res["decode_img_s_per_thread"] = (DECODE_TIMING_N
                                              / (time.perf_counter() - t0))
        finally:
            torch.set_num_threads(n_threads)
        # bench setup only (not timed): default threads
        with ThreadPoolExecutor(DECODE_POOL) as pool:
            imgs = [d[0] for d in pool.map(cl.decode, samples)]
        res["n_decoded"] = len(imgs)
        t0 = time.perf_counter()
        vae = VaeEncoder(VAE_REPO, "cuda", torch.bfloat16)
        res["vae_load_s"] = time.perf_counter() - t0
        # timed with 1 CPU thread, like production (run_ladder intra_threads)
        res["ladder"] = run_ladder(
            lambda x: cl.vae_encode_batch(vae, x), imgs, LADDER,
            (LAT_CHANNELS, h // VAE_SCALE, w // VAE_SCALE), cuda=torch.cuda,
            intra_threads=1)
        res.update(best_of(res["ladder"]))
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-4000:]
        print(res["error"], flush=True)
    res["seconds"] = time.time() - t_start
    return res


def prep_impl(bench_files):
    """Bench inputs (if bench_files) + VAE weights onto the volume, and the
    token's identity / write access (best effort, never fails on it)."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    name, token = setup_hf_token()
    out = {"token_env": name}
    try:
        who = HfApi(token=token).whoami()
        out["whoami"] = who.get("name")
        out["can_write"], out["write_reason"] = token_can_write(who)
    except Exception as e:  # noqa: BLE001
        out.update(whoami=None, can_write=None,
                   write_reason=f"whoami failed: {type(e).__name__}: {e}")
    print(f"HF token ({name}): user {out['whoami']!r}, can write "
          f"{out['can_write']} ({out['write_reason']})", flush=True)
    if bench_files:
        for f in (PARQUET, BENCH_TAR):
            if (Path(VOL_BENCH) / f).exists():
                out[f] = "present"
                continue
            t0 = time.time()
            hf_hub_download(DATASET_REPO, f, repo_type="dataset",
                            local_dir=VOL_BENCH, token=token)
            out[f] = f"downloaded in {time.time() - t0:.0f}s"
            print(f"{f}: {out[f]}", flush=True)
    t0 = time.time()
    out["vae_dir"] = snapshot_download(VAE_REPO, allow_patterns=VAE_PATTERNS,
                                       token=token)
    out["vae_s"] = time.time() - t0
    print(f"VAE {VAE_REPO} at {out['vae_dir']} ({out['vae_s']:.0f}s)",
          flush=True)
    return out


def gpu_name_smi():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def build_worker_impl(i, n, batch, decode_threads, vol):
    """One cache_latents.py worker; the volume is committed after every
    finished shard (explicitly, on top of Modal's background commits, so a
    killed container loses at most its in-flight shard) and at the end."""
    setup_hf_token()
    env = worker_env(os.environ)
    res = {"worker": i, "device_name": gpu_name_smi()}
    cmd = worker_cmd(i, n, batch, decode_threads, python=sys.executable)
    print(f"w{i}/{n} on {res['device_name']}: $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        rc, tail, shards = stream_worker(cmd, lambda: commit_volume(vol),
                                         env=env)
        res.update(rc=rc, tail=tail, shards_written=shards)
    except Exception:  # noqa: BLE001
        res.update(rc=None, error=traceback.format_exc()[-4000:])
    res["final_commit"] = commit_volume(vol)
    res["seconds"] = time.time() - t0
    return res


class WriteAccessError(Exception):
    pass


class NotPrivateError(Exception):
    pass


UPLOAD_STATE = ".cache/huggingface"   # upload_large_folder state, under out
CONTENT_HELP = (
    "remote content check FAILED for {n} file(s) in " + OUT_REPO + " (see "
    "problems). Delete the stale file(s) on the Hub AND re-run with "
    "`--finalize-only --reset-upload-cache` (drops the upload_large_folder "
    f"state {VOL_OUT}/{UPLOAD_STATE}, so every file is re-hashed and "
    "re-sent).")


def reset_upload_state(out_dir, vol, log=print):
    """Remove <out_dir>/.cache/huggingface (upload_large_folder's record of
    what was already uploaded) and commit the volume. -> True if removed."""
    d = Path(out_dir) / UPLOAD_STATE
    existed = d.exists()
    shutil.rmtree(d, ignore_errors=True)
    if d.exists():
        raise RuntimeError(f"could not remove {d}")
    commit_volume(vol, log)
    log(f"--reset-upload-cache: {'removed' if existed else 'no'} {d}")
    return existed


def finalize_impl(vol, api=None, out_dir=VOL_OUT, runner=subprocess.run,
                  python=sys.executable, cwd=REMOTE_REPO,
                  reset_upload_cache=False):
    """HF write check + repo creation + privacy check, --write-manifest,
    (--reset-upload-cache: drop the upload state), upload, remote content
    check. `api` (an HfApi; default: one with the
    secret's token) and `runner` (subprocess.run signature) are injectable
    for tests. Never raises: -> {ok, repo, error, problems, n_files, bytes,
    revision, log}."""
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    res = {"ok": False, "repo": OUT_REPO, "error": None, "problems": [],
           "log": log}
    try:
        try:
            vol.reload()
        except Exception as e:  # noqa: BLE001 - the mount is fresh anyway
            say(f"WARNING: volume reload failed: {type(e).__name__}: {e}")
        if api is None:
            from huggingface_hub import HfApi

            _, token = setup_hf_token()
            api = HfApi(token=token)
        # 1. write access FIRST (before reading ~49 GB for the manifest)
        try:
            who = api.whoami()
        except Exception as e:  # noqa: BLE001
            raise WriteAccessError(f"whoami failed: {type(e).__name__}: {e}")
        can, why = token_can_write(who)
        say(f"HF user {who.get('name')!r}: can write {can} ({why})")
        if can is False:
            raise WriteAccessError(why)
        if api.repo_exists(OUT_REPO, repo_type="dataset"):
            try:
                api.auth_check(OUT_REPO, repo_type="dataset", write=True)
            except Exception as e:  # noqa: BLE001
                raise WriteAccessError(
                    f"auth_check(write=True) failed: {type(e).__name__}: {e}")
        # create_repo is the real write test
        try:
            api.create_repo(OUT_REPO, repo_type="dataset", private=True,
                            exist_ok=True)
        except Exception as e:  # noqa: BLE001
            raise WriteAccessError(
                f"create_repo failed: {type(e).__name__}: {e}")
        # exist_ok keeps an existing repo's visibility: never upload to a
        # public one
        private = api.repo_info(OUT_REPO, repo_type="dataset").private
        if private is not True:
            raise NotPrivateError(
                f"{OUT_REPO} exists but is not private (private={private!r});"
                " nothing was uploaded. Make it private on the Hub (Settings"
                " -> visibility) or delete it, then re-run --finalize-only.")
        say(f"repo {OUT_REPO} (dataset, private) ready")
        # 2. verify all shards + digests, metadata.parquet, manifest.json
        cmd = finalize_cmd(python=python, out=str(out_dir))
        say("$ " + " ".join(cmd))
        rc = runner(cmd, cwd=cwd).returncode
        if rc != 0:
            raise RuntimeError(f"cache_latents.py --write-manifest exit {rc}")
        if not commit_volume(vol, say):
            say("continuing: the manifest is on the container's mount")
        # 3. upload (resumable; its state lives in /vol/krea2/.cache)
        if reset_upload_cache:
            reset_upload_state(out_dir, vol, say)
        t0 = time.time()
        api.upload_large_folder(OUT_REPO, str(out_dir), repo_type="dataset",
                                private=True, ignore_patterns=UPLOAD_IGNORE,
                                num_workers=UPLOAD_WORKERS)
        say(f"upload_large_folder done in {(time.time() - t0) / 60:.1f} min")
        commit_volume(vol, say)
        # 4. every expected file present with the expected content, read at
        #    one commit
        expected = expected_upload(out_dir)
        sha = api.repo_info(OUT_REPO, repo_type="dataset").sha
        remote = remote_entries(api.list_repo_tree(
            OUT_REPO, repo_type="dataset", recursive=True, expand=True,
            revision=sha))
        res["revision"] = sha
        res["problems"] = verify_upload(expected, remote)
        res["n_files"] = len(expected)
        res["bytes"] = sum(e["size"] for e in expected.values())
        say(f"remote check at {sha}: {len(expected)} files, "
            f"{res['bytes'] / 1e9:.2f} GB, {len(res['problems'])} problems")
        res["ok"] = not res["problems"]
        if res["problems"]:
            res["error"] = CONTENT_HELP.format(n=len(res["problems"]))
            say("FATAL: " + res["error"])
            for p in res["problems"][:20]:
                say("  " + p)
    except WriteAccessError as e:
        res["error"] = WRITE_HELP.format(reason=e)
        say("FATAL: " + res["error"])
    except NotPrivateError as e:
        res["error"] = str(e)
        say("FATAL: " + res["error"])
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-4000:]
        say(res["error"])
    return res


# ------------------------------------------------------------------ Modal app
if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(TORCH_PIN, extra_index_url=TORCH_INDEX)
        .pip_install(*PIP_PACKAGES)
        .env(CONTAINER_ENV)
        .add_local_dir(LOCAL_REPO / "krea2_jax", f"{REMOTE_REPO}/krea2_jax",
                       ignore=["**/__pycache__"])
        .add_local_dir(LOCAL_REPO / "scripts", f"{REMOTE_REPO}/scripts",
                       ignore=["**/__pycache__"])
    )
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True,
                                 version=VOLUME_VERSION)
    hf_secret = modal.Secret.from_name("huggingface")
    app = modal.App("krea2-cache-latents", image=image)
    mounts = {"volumes": {VOL: vol}, "secrets": [hf_secret]}

    @app.function(cpu=2.0, memory=8192, timeout=30 * 60, **mounts)
    def prep(bench_files: bool = True) -> dict:
        out = prep_impl(bench_files)
        vol.commit()   # the bench/build containers start from this state
        return out

    # gpu is always overridden per call with .with_options(gpu=...)
    @app.function(gpu="L4", cpu=float(BENCH_CPU), memory=MEM_MB,
                  timeout=20 * 60, **mounts)
    def bench_gpu(gpu_requested: str, n_decode: int) -> dict:
        return bench_gpu_impl(gpu_requested, n_decode)

    # gpu, cpu and timeout are always overridden per build with .with_options
    @app.function(gpu="L4", cpu=8.0, memory=MEM_MB,
                  timeout=DEFAULT_WORKER_TIMEOUT_H * 3600, **mounts)
    def build_worker(i: int, n: int, batch: int, decode_threads: int) -> dict:
        return build_worker_impl(i, n, batch, decode_threads, vol)

    @app.function(cpu=4.0, memory=16384, timeout=2 * 3600, **mounts)
    def finalize(reset_upload_cache: bool = False) -> dict:
        return finalize_impl(vol, reset_upload_cache=reset_upload_cache)

    def _print_prep(p):
        print(f"prep: HF user {p.get('whoami')!r} (token from "
              f"{p.get('token_env')}), can write {p.get('can_write')} "
              f"({p.get('write_reason')})")

    def bench_local(gpus, n_decode, overhead_min):
        gl = parse_gpus(gpus)
        if n_decode < DECODE_TIMING_N:
            raise SystemExit(f"--n-decode must be >= {DECODE_TIMING_N}")
        p = prep.remote(True)
        _print_prep(p)
        calls = []
        for g in gl:
            try:
                fn = bench_gpu.with_options(gpu=g, cpu=float(BENCH_CPU),
                                            memory=MEM_MB)
                calls.append((g, fn.spawn(g, n_decode)))
            except Exception as e:  # noqa: BLE001
                calls.append((g, e))
        print(f"spawned {len(calls)} bench containers: {', '.join(gl)}")
        deadline = time.time() + BENCH_WAIT_S
        results = []
        for g, c in calls:
            if isinstance(c, Exception):
                results.append({"gpu_requested": g, "error":
                                f"spawn failed: {type(c).__name__}: {c}"})
                continue
            try:
                r = c.get(timeout=max(1.0, deadline - time.time()))
            except Exception as e:  # noqa: BLE001 - unavailable / timed out
                r = {"gpu_requested": g,
                     "error": f"no result: {type(e).__name__}: {e}"}
                try:
                    c.cancel()
                except Exception:  # noqa: BLE001
                    pass
            print(f"{g}: " + (f"ERROR {_err_line(r['error'])}"
                              if r.get("error") else
                              f"{r.get('device_name')} "
                              f"best {r.get('best_img_s', 0):.2f} img/s"))
            results.append(r)
        rows = [cost_row(r, overhead_min=overhead_min) for r in results]
        choice = select_gpu(rows)
        md = render_bench_report(results, rows, choice, prep=p,
                                 overhead_min=overhead_min)
        print("\n" + "=" * 72 + "\n" + md)
        d = write_bench_results(results, rows, choice, md, prep=p)
        print(f"\nwrote {d}/report.md and bench.json")

    def _write_build_log(params, p, workers, fin):
        md = render_build_log(params, p, workers, fin)
        LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
        path = LOCAL_RESULTS / f"cache_build_{time.strftime('%Y%m%d-%H%M%S')}.md"
        path.write_text(md + "\n")
        print(md + f"\n\nwrote {path}")
        return path

    def _check_finalize(fin, path):
        if not fin.get("ok"):
            reason = (_err_line(fin["error"]) if fin.get("error")
                      else "; ".join(fin.get("problems") or [])[:500])
            raise SystemExit(f"finalize FAILED: {reason} (log: {path})")

    def build_local(gpu, containers, batch, cpu, decode_threads,
                    worker_timeout_h, reset_upload_cache):
        gpu = gpu.strip().upper()
        if not gpu:
            raise SystemExit("--build needs --gpu (see the bench report)")
        gpu_key(gpu)
        if containers < 1 or batch < 1 or cpu < 1 or decode_threads < 0:
            raise SystemExit("--containers, --batch, --cpu must be >= 1")
        if not 0 < worker_timeout_h <= MAX_WORKER_TIMEOUT_H:
            raise SystemExit(f"--worker-timeout-h must be in (0, "
                             f"{MAX_WORKER_TIMEOUT_H}]")
        threads = decode_threads or cpu
        params = {"gpu": gpu, "containers": containers, "batch": batch,
                  "cpu": cpu, "decode_threads": threads,
                  "worker_timeout_h": worker_timeout_h,
                  "reset_upload_cache": reset_upload_cache}
        p = prep.remote(False)   # VAE once, before N workers would race on it
        _print_prep(p)
        if p.get("can_write") is False:
            raise SystemExit("aborting before any GPU starts: "
                             + WRITE_HELP.format(reason=p["write_reason"]))
        fn = build_worker.with_options(
            gpu=gpu, cpu=float(cpu), memory=MEM_MB,
            timeout=int(worker_timeout_h * 3600))
        calls = [fn.spawn(i, containers, batch, threads)
                 for i in range(containers)]
        print(f"spawned {containers} workers on {gpu}: {params}")
        workers = []
        for i, c in enumerate(calls):
            try:
                w = c.get()
            except Exception as e:  # noqa: BLE001
                w = {"worker": i, "rc": None,
                     "error": f"{type(e).__name__}: {e}"}
            print(summarize_workers([w])[1][0])
            workers.append(w)
        ok, _ = summarize_workers(workers)
        fin = (finalize.remote(reset_upload_cache=reset_upload_cache)
               if ok else None)
        path = _write_build_log(params, p, workers, fin)
        if not ok:
            raise SystemExit(f"build FAILED: a worker failed, finalize not "
                             f"run; fix and re-run --build (done shards are "
                             f"skipped) (log: {path})")
        _check_finalize(fin, path)

    @app.local_entrypoint()
    def main(bench: bool = False, gpus: str = DEFAULT_GPUS,
             n_decode: int = 128,
             overhead_min: float = DEFAULT_OVERHEAD_MIN,
             build: bool = False, finalize_only: bool = False,
             gpu: str = "", containers: int = DEFAULT_CONTAINERS,
             batch: int = 8, cpu: int = 8, decode_threads: int = 0,
             worker_timeout_h: float = DEFAULT_WORKER_TIMEOUT_H,
             reset_upload_cache: bool = False):
        modes = [m for m, on in (("--bench", bench), ("--build", build),
                                 ("--finalize-only", finalize_only)) if on]
        if len(modes) != 1:
            raise SystemExit("choose exactly one of --bench, --build, "
                             f"--finalize-only (got {modes or 'none'})")
        if bench and reset_upload_cache:
            raise SystemExit("--reset-upload-cache goes with --build or "
                             "--finalize-only")
        if bench:
            bench_local(gpus, n_decode, overhead_min)
        elif build:
            build_local(gpu, containers, batch, cpu, decode_threads,
                        worker_timeout_h, reset_upload_cache)
        else:
            fin = finalize.remote(reset_upload_cache=reset_upload_cache)
            path = _write_build_log(
                {"finalize_only": True,
                 "reset_upload_cache": reset_upload_cache}, {}, [], fin)
            _check_finalize(fin, path)
