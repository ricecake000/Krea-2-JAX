"""Modal app: benchmark Modal GPU types for the VAE latent cache, then build
the `krea2-latents-v1` cache (scripts/cache_latents.py) on the chosen type and
upload it to a private HF dataset repo the training node can `hf download`
(v2), or, for a named cache (v3 on), build it from an image package on Google
Drive and upload it to Cloudflare R2 (the node kit's DATA_SOURCE=r2).

    modal run scripts/modal_cache_latents.py --bench                  # all GPU types x impls
    modal run scripts/modal_cache_latents.py --bench --gpus B200 --impls ref,2d,2d-compile
    modal run --detach scripts/modal_cache_latents.py --build --gpu B200 \\
        --containers 4 --batch 16 --cpu 12 --worker-timeout-h 4 --vae-impl 2d
    modal run scripts/modal_cache_latents.py --finalize-only          # manifest + upload again
    modal run scripts/modal_cache_latents.py --finalize-only --reset-upload-cache
    modal run scripts/modal_cache_latents.py --profile [--gpu B200]   # mode 4

Mode 1 (--bench): `prep` (CPU) puts metadata.parquet + data/train-0000.tar of
ricecake/danbooru-curated-v2 into /vol/bench and the Qwen-Image VAE into
HF_HOME=/vol/hf, and reports whether the HF token can write. Then one
`bench_gpu` container per --gpus type runs concurrently: decode --n-decode
832x1216 samples (the most common bucket, 32% of the data; every bucket is
~1 MP so the per-image cost is near-constant) with cache_latents.decode, time
single-threaded decode, then per VAE encode path of --impls
(scripts/vae_fast.py: ref = diffusers, 2d = single-frame 2D encoder,
2d-compile = 2d under torch.compile, 2d-compile-cl = 2d-compile with module
and input in channels_last; the default --impls leaves 2d-compile-cl out,
pass it explicitly): a correctness gate on 2 real images
against ref (gate_check: with TF32 off, fp32 posterior mean, std and seeded
normalized sample max rel error < 1e-4; in bf16, mean and std error vs the
fp32 ref <= 1.5 x the ref's own bf16 error + 1e-3; else the impl is skipped;
the first bf16 call of 2d-compile(-cl) is its compile time), then a batch
ladder 4/8/16/32 of the impl
on the CPU float32 tensors exactly like production (stack + H2D + encode +
bf16 D2H, 1 CPU thread), each >= max(64 images, 3 batches) and >= 10 s after
one warmup call, stopping at the first OOM. A GPU type that is unavailable
or fails becomes an error row; the others still report. Locally: cost per
image for every (GPU, impl) (GPU + the CPU cores needed to keep it fed + 32
GiB of memory, billed by use and modelled as 32 GiB: no function reserves
memory=; Modal prices of PRICES_DATE), the selection (lowest $/img; anything
within 5% of it is a tie, broken by higher img/s) and the exact build
command (incl. --vae-impl). torch.compile artifacts live in /vol/inductor,
so the build workers reuse the bench's compile.
Output: /home/mj/krea2-ops/mi350x/results/modal/cache_bench_<ts>/{report.md,bench.json}.
Rough cost of a full bench: ~10 min per GPU type with the three default
impls (+ one more compile with 2d-compile-cl).

Mode 2 (--build): `prep` (VAE only; aborts before any GPU starts when the
token is clearly read-only), then --containers `build_worker` containers on
--gpu (Modal timeout --worker-timeout-h), worker i running `cache_latents.py
--num-workers N --worker-index i --vae-impl X` into /vol/krea2 (shards i,
i+N, ...), then
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

Mode 3, named caches (--cache-name NAME: Drive in, R2 out; v3 on). The HF
image repo above is v2's and the HF private quota is full, so a new dataset
is never built into /vol/krea2 nor uploaded to the HF latents repo:
    --src-drive P   image package on Drive (metadata.parquet + data/train-
                    XXXX.tar, the v2 HF image repo's layout): a path under
                    gdrive:danbooru_pipeline (remote `gdrive` of the Modal
                    secret `rclone-gdrive`, RCLONE_CONF_B64; recreate it as in
                    danbooru-aesthetic-pred/modal_embed.py) or remote:path
    --dest-r2 X     bucket-relative prefix in the `krea2-r2` bucket
                    (krea2-fullft), e.g. latents_v3/
--build: `stage` (CPU, secret rclone-gdrive only) lists the package on
Drive (path, size, Drive's own md5) = its input generation, refuses a cache
dir already started from another generation (pick a new --cache-name or
delete /vol/krea2-NAME explicitly), rclone-syncs it into /vol/src/NAME
(--checksum: resumable; local tars the package no longer has are deleted),
re-lists it (it must not have changed meanwhile), checks every tar the
parquet names is in the listing and on disk, and records the generation in
/vol/src/NAME/stage.json and (first use) /vol/krea2-NAME/
.input_generation.json; workers and finalize_r2 refuse when the two differ.
Then finalize_r2's pre-check (generation, the prefix, and a write probe
object .write-probe-<random> written + deleted under it, so read-only
credentials fail here); a failure of either stops before any GPU starts.
Then the cost estimate (build_estimate at a measured rate of the same
gpu/impl/batch, MEASURED_RATES, and the staged row count), prep (VAE only;
no HF write check), the workers
(cache_latents.py --tar-dir /vol/src/NAME --out /vol/krea2-NAME), and
`finalize_r2` (CPU, secret krea2-r2 only, never HF): --write-manifest
against the staged parquet, then rclone copy --checksum of metadata.parquet +
the manifest's shards to r2:krea2-fullft/X/ and latents/manifest.json LAST
(a manifest on R2 = a complete cache; rclone's own transfer check is the
upload check). A prefix with a manifest is published and immutable: one
with another manifest (or foreign / different-size objects) is refused; the
same manifest only gets its missing objects (--ignore-existing, nothing is
replaced, metadata.parquet included). A prefix without one is resumed.
--finalize-only --cache-name NAME --dest-r2 X redoes finalize_r2;
--stage-only --cache-name NAME --src-drive P only stages (CPU, cents).
--src-drive / --dest-r2 without --cache-name, and --reset-upload-cache with
it, are refused.

v3 recipe (package gdrive:danbooru_pipeline/selection_v3/package, 196,099
images / 200 tars; no bench needed again: the v2 build's B200 setup with
the channels_last impl of the 11/02 profile, parity-checked there):
    modal run scripts/modal_cache_latents.py --stage-only --cache-name v3 \
        --src-drive selection_v3/package           # optional: CPU only, early
    modal run --detach scripts/modal_cache_latents.py --build --cache-name v3 \
        --src-drive selection_v3/package --dest-r2 latents_v3/ --gpu B200 \
        --containers 1 --batch 4 --cpu 8 --worker-timeout-h 4 \
        --vae-impl 2d-compile-cl
    modal run scripts/modal_cache_latents.py --finalize-only --cache-name v3 \
        --dest-r2 latents_v3/                      # after a laptop disconnect
Estimate (build_estimate, prices of PRICES_DATE, user 11/02: one container,
wall time does not matter): 2d-compile-cl at the B200 profile's run_ladder
rate 110.5 img/s (PROFILE_CL_RATE; no image decode, so optimistic: the v2
build ran at 41.0 img/s where run_ladder measured 59.0 for 2d-compile, and 8
decode threads may not feed 110 img/s): 1 x B200 = $3.74 for the GPU
container (0.49 GPU-h + 3 min startup at $6.88/h), ~33 min wall. Fallback
--vae-impl 2d-compile (the v2 build's impl, v2 rate 41.0 img/s; also the
cl cost if it only reaches that rate): 1 x B200 = $9.50 (1.33 GPU-h), ~83
min wall (2 x B200 = $9.84 / ~43 min). Plus the CPU stage + finalize_r2
containers (cents to well under $1) and R2 storage (~100 GB, ~$1.5/month).
--worker-timeout-h 4 is a cap, not a charge: ~2.9x the v2 rate's ~1.4 h
wall (the cost model says 3 h at the v2 rate, 2 h at the profile rate).
The training node downloads the cache with DATA_SOURCE=r2
DATA_R2_PREFIX=latents_v3 (krea2-ops/node).

Mode 4 (--profile): where the encoder's GPU time goes (scripts/vae_profile.py)
on one --gpu container (default B200, 8 cores, no memory reservation; hard
Modal timeout PROFILE_TIMEOUT_S = 20 min, so the cost cap printed before the
spawn is 20 min x that container's $/h with memory modelled as 32 GiB used:
B200 $2.29 at PRICES_DATE prices, plus
cents for `prep`). `prep` (VAE only) first, then `profile_gpu`: seeded
random images at the bench bucket (1216x832), batch 4, bf16: FLOPs per
stage, eager per-stage CUDA-event timing, parity (check_impls) of
2d-compile / 2d-compile-cl (channels_last) / 2d, torch.profiler of the
2d-compile production step (kernel categories, top kernels, GPU idle,
gzip chrome trace), the SDPA backends at the mid attention's shape, the
variants' run_ladder at batch 4 (sections bounded by PROFILE_BUDGET_S = 10
min), then, only with PROFILE_MAT_RESERVE_S of the timeout left, the same
for torch.compile mode="max-autotune-no-cudagraphs" (2d-compile-mat). The
report so far is saved to /vol/profile/<ts>/ after every section and
fetched from there if the container times out. Refused with any other mode or with
--cache-name / --src-drive / --dest-r2 / --reset-upload-cache. Output:
/home/mj/krea2-ops/mi350x/results/modal/vae_profile_<ts>/{report.md,
profile.json,trace.json.gz} (also kept in /vol/profile/<ts>/).

Volume `krea2-latents` (Volumes v2, created on first use):
    /vol/hf       HF_HOME (VAE weights downloaded once)
    /vol/bench    metadata.parquet, data/train-0000.tar (bench input)
    /vol/krea2    latents/train-XXXX.safetensors, then metadata.parquet +
                  latents/manifest.json (the v2 cache; uploaded as the repo
                  root)
    /vol/profile/TS     --profile results
    /vol/src/NAME       staged image package of a named cache + stage.json
    /vol/krea2-NAME     a named cache (same layout as /vol/krea2) +
                        .input_generation.json (never uploaded)
Modal secrets: `huggingface` (the HF token under HF_TOKEN,
HUGGING_FACE_HUB_TOKEN or HUGGINGFACE_TOKEN, exported as HF_TOKEN; bench,
prep, workers, v2 finalize), `rclone-gdrive` (stage), `krea2-r2`
(finalize_r2: R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY, passed to rclone only through its env). The image
carries rclone RCLONE_VER (zip sha256 pinned, as the node kit).

Importable without the Modal SDK (the app is only defined when `modal`
imports); pricing, cost model, selection, report rendering, worker command,
streaming/commit loop, upload checks, staging and R2 finalize are plain
functions (tests/test_modal_safety.py).
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
import tempfile
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
BENCH_TIMEOUT_S = 45 * 60   # one bench container: the 3 default impls
BENCH_WAIT_S = 75 * 60   # local wait for all bench containers (incl. queueing)
# VAE encode paths (scripts/vae_fast.py IMPLS; repeated here so this module
# imports without torch)
VAE_IMPLS = ("ref", "2d", "2d-compile", "2d-compile-cl")
# --bench default: 2d-compile-cl (one more compile pair) only on request
BENCH_DEFAULT_IMPLS = ("ref", "2d", "2d-compile")
DEFAULT_IMPLS = ",".join(BENCH_DEFAULT_IMPLS)
# correctness gate of an impl vs ref on the real VAE (gate_check):
#   fp32: posterior mean, std and seeded normalized sample max rel < CHECK_TOL
#   production dtype (bf16): impl mean/std error vs ref fp32 <=
#   BF16_SLACK * (ref bf16 vs ref fp32 error) + BF16_ABS
CHECK_TOL = 1e-4
BF16_SLACK = 1.5
BF16_ABS = 1e-3
CHECK_SEED = 1234
CHECK_IMAGES = 2
# --profile (scripts/vae_profile.py): one container, bench bucket, batch 4
PROFILE_GPU = "B200"
PROFILE_TIMEOUT_S = 20 * 60   # hard Modal timeout = the cost cap
PROFILE_BUDGET_S = 10 * 60    # GPU sections after the VAE load
PROFILE_BATCH = 4             # the v3 build's --batch
# the max-autotune variant (two compiles: fp32 parity + bf16) starts only
# with this much of PROFILE_TIMEOUT_S left (the report so far is on the
# volume after every section)
PROFILE_MAT_RESERVE_S = 8 * 60
VOL_PROFILE = f"{VOL}/profile"
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
# no Modal function sets memory= (user rule: cpu= only): memory is billed by
# actual use, without a reservation; the cost model assumes MEM_GIB in use
MEM_GIB = 32

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
VOL_INDUCTOR = f"{VOL}/inductor"
# torch.compile artifacts on the volume, so build workers reuse the bench's
# compiled kernels (keys include the GPU arch / torch version). Concurrency:
# several containers may write the same cache entry; inductor and triton
# write entries atomically (tmp file + rename) with deterministic content,
# so last-write-wins is harmless, and their file locks are only
# container-local (each container sees its own view until a commit).
CONTAINER_ENV = {
    "HF_HOME": VOL_HF,
    "HF_XET_CACHE": "/tmp/hf_xet",   # keep xet chunk caches off the volume
    "PYTHONUNBUFFERED": "1",
    "TORCHINDUCTOR_CACHE_DIR": VOL_INDUCTOR,
    "TRITON_CACHE_DIR": f"{VOL_INDUCTOR}/triton",
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
# --- named caches (--cache-name NAME, v3 on): Drive image package in, R2 out.
# Each name has its own dirs, so the v2 cache (/vol/krea2, HF) is never read
# or written by a named build.
VOL_SRC = f"{VOL}/src"    # staged image packages: /vol/src/<name>
SHARED_DIRS = (VOL_HF, VOL_BENCH, VOL_OUT, VOL_INDUCTOR)
CACHE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
# remote `gdrive` = the rclone config in the Modal secret `rclone-gdrive`
# (RCLONE_CONF_B64, the same secret danbooru-aesthetic-pred/modal_embed.py uses)
DRIVE_ROOT = "gdrive:danbooru_pipeline"
STAGE_INCLUDE = ["/metadata.parquet", "/data/train-*.tar"]
R2_SEG = r"[A-Za-z0-9][A-Za-z0-9._-]*"
R2_PREFIX_RE = re.compile(rf"^{R2_SEG}(/{R2_SEG})*$")
R2_ENV = ("R2_ACCOUNT_ID", "R2_BUCKET", "R2_ACCESS_KEY_ID",
          "R2_SECRET_ACCESS_KEY")   # Modal secret `krea2-r2` (krea2_jax/r2.py)
MANIFEST = "latents/manifest.json"
STAGE_JSON = "stage.json"   # /vol/src/NAME: the staged input generation
GEN_JSON = ".input_generation.json"   # /vol/krea2-NAME: the input it is built from
PROBE_PREFIX = ".write-probe-"   # R2 pre-check's write probe objects
# pinned rclone (the zip + sha256 the node kit's dl.sh pins)
RCLONE_VER = "v1.75.1"
RCLONE_ZIP_URL = (f"https://downloads.rclone.org/{RCLONE_VER}/rclone-"
                  f"{RCLONE_VER}-linux-amd64.zip")
RCLONE_ZIP_SHA256 = ("982b5aa772841168f8e380f139e9e787"
                     "b2a105403e32b94da8676a0e1c0a13ab")
RCLONE_BIN = "/usr/local/bin/rclone"
RCLONE_CONF = "/tmp/rclone.conf"    # Drive config, 0600, deleted after use
RCLONE_COMMON = ["--retries", "10", "--low-level-retries", "20",
                 "--stats", "60s", "--stats-one-line", "-v"]
R2_UPLOAD_FLAGS = ["--checksum", "--transfers", "16", "--checkers", "16",
                   "--s3-chunk-size", "64M", "--s3-upload-concurrency", "4"]
STAGE_TIMEOUT_H = 4
FINALIZE_R2_TIMEOUT_H = 8   # ~2x v2: hash ~100 GB + upload ~100 GB
# measured v2 build (2026-10-13, cache_build_b200_r2.log): 2 x B200,
# 2d-compile, batch 4, --cpu 8: 96,848 images in 17.8 + 21.6 container-min
# (after VAE load / compile) -> img/s per container
V2_BUILD = {"gpu": "B200", "impl": "2d-compile", "batch": 4, "cores": 8,
            "img_s": 96_848 / ((17.8 + 21.6) * 60)}
# measured B200 profile (2026-11-02, /home/mj/krea2-ops/mi350x/results/modal/
# vae_profile_20261102-102843/report.md, section e): run_ladder at batch 4,
# 1216x832 seeded inputs, CPU stack + H2D + encode + bf16 D2H on an 8-core
# container, WITHOUT image decode (a real build also decodes webp on CPU, so
# its rate may be lower; there run_ladder 2d-compile = 59.0 vs the v2 build's
# 41.0 img/s, and 8 decode threads may not keep up with 110 img/s)
PROFILE_CL_RATE = {"gpu": "B200", "impl": "2d-compile-cl", "batch": 4,
                   "cores": 8, "img_s": 110.5,
                   "label": "B200 profile run_ladder 2026-11-02, no image "
                            "decode: the real build decodes webp on CPU and "
                            "may be slower"}
# per-container rates render_estimate knows, by (gpu, impl, batch)
MEASURED_RATES = (dict(V2_BUILD, label="v2 build"), PROFILE_CL_RATE)
RCLONE_INSTALL = (
    "python -c \"import sys, urllib.request as u; open('/tmp/rclone.zip', "
    "'wb').write(u.urlopen(u.Request(sys.argv[1], headers={'User-Agent': "
    "'krea2-modal/1.0'})).read())\" " + RCLONE_ZIP_URL
    + f" && echo '{RCLONE_ZIP_SHA256}  /tmp/rclone.zip' | sha256sum -c -"
    " && python -m zipfile -e /tmp/rclone.zip /tmp/rclone_x"
    f" && install -m 755 /tmp/rclone_x/rclone-{RCLONE_VER}-linux-amd64/rclone"
    f" {RCLONE_BIN} && rm -rf /tmp/rclone.zip /tmp/rclone_x"
    f" && {RCLONE_BIN} version")

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


def parse_impls(s):
    """'ref, 2d' -> ['ref', '2d'] (validated against VAE_IMPLS, de-duplicated)."""
    out = []
    for i in str(s).split(","):
        i = i.strip()
        if not i:
            continue
        if i not in VAE_IMPLS:
            raise SystemExit(f"unknown VAE impl {i!r} (known: "
                             f"{', '.join(VAE_IMPLS)})")
        if i not in out:
            out.append(i)
    if not out:
        raise SystemExit(f"--impls is empty: {s!r}")
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


def flatten_results(results):
    """bench_gpu results (one per GPU, {"impls": {impl: {...}}}) -> one
    entry per (gpu, impl) with the GPU-level fields (device, decode rate)
    and the impl's check / ladder / best_*. The entry's "error" is the GPU's
    error, else the impl's, else a failed correctness check (such impls are
    never costed). A result without "impls" is one entry: its own "impl",
    else "-" when it failed (the container died before the impls ran), else
    "ref" (a single-impl result)."""
    out = []
    for res in results:
        base = {k: v for k, v in res.items() if k != "impls"}
        impls = res.get("impls")
        if not impls:
            out.append({**base, "impl": res.get("impl") or (
                "-" if res.get("error") else "ref")})
            continue
        for impl, r in impls.items():
            e = {**base, **r, "impl": impl}
            check = r.get("check") or {}
            err = res.get("error") or r.get("error") or check.get("error")
            if not err and check.get("ok") is False:
                err = ("correctness FAILED: "
                       + "; ".join(check.get("fail_reasons") or ["?"]))
            e["error"] = err
            out.append(e)
    return out


def cost_row(res, n_images=N_IMAGES, overhead_min=DEFAULT_OVERHEAD_MIN,
             containers=CONTAINER_COUNTS, mem_gib=MEM_GIB):
    """One (gpu, impl) bench entry (flatten_results) -> cost row, or None
    when it has no usable measurement (error incl. a failed correctness
    check, no successful batch, no decode rate).

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
    check = res.get("check") or {}
    return {
        "gpu": res["gpu_requested"], "impl": res.get("impl", "ref"),
        "device_name": res.get("device_name"),
        "batch": res.get("best_batch"), "img_s": img_s,
        "fp32_err": check.get("fp32_mean_max_rel"),
        "bf16_err": check.get("bf16_mean_max_rel"),
        "compile_s": check.get("first_call_s")
        if res.get("impl") in ("2d-compile", "2d-compile-cl") else None,
        "peak_gib": res.get("best_peak_gib"), "decode": dec, "cores": cores,
        "usd_h": hourly, "usd_per_1k": hourly / (img_s * 3600) * 1000,
        "hours_1": hours, "overhead_h": over,
        "proj": {k: {"usd": hourly * (hours + k * over),
                     "wall_h": hours / k + over} for k in containers},
    }


def select_gpu(rows, tie_frac=TIE_FRAC):
    """Lowest $/img wins; every row within tie_frac of the lowest is a tie,
    broken by the higher img/s. Rows are (gpu, impl); None rows (failed GPUs
    or impls, incl. a failed correctness check) are ignored."""
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
            f"--worker-timeout-h {t} --vae-impl {row.get('impl', 'ref')}")


def _ladder_str(ladder):
    return " ".join(f"{r['batch']}:{r['error']}" if "error" in r
                    else f"{r['batch']}:{r['img_s']:.2f}"
                    for r in ladder or [])


def _err_line(err):
    lines = [ln for ln in str(err).strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else "?"


def _fmt(x, spec, none="-"):
    return none if x is None else format(x, spec)


def render_bench_report(results, rows, choice, prep=None,
                        overhead_min=DEFAULT_OVERHEAD_MIN,
                        containers=CONTAINER_COUNTS, n_images=N_IMAGES):
    """Markdown report over (gpu, impl) rows: cost table, failed GPU types /
    impls, the choice and the exact build command. `results` are the raw
    bench_gpu results, `rows` the cost_row of each flatten_results entry."""
    entries = flatten_results(results)
    out = [f"# VAE latent cache: Modal GPU bench ({time.strftime('%Y-%m-%d %H:%M')})",
           "",
           f"Bucket {BENCH_BUCKET[0]}x{BENCH_BUCKET[1]}, {n_images} images, "
           f"prices of {PRICES_DATE} (cpu ${CPU_CORE_USD_H}/core-h, memory "
           f"{MEM_GIB} GiB at ${MEM_GIB_USD_H}/GiB-h: billed by use, no "
           f"reservation, modelled as {MEM_GIB} GiB), {overhead_min:g} min "
           "startup + VAE load per container. cores = max(4, ceil(1.5 * "
           "img/s / decode img/s/thread) + 1). fp32/bf16 err = max |impl - "
           "ref fp32| / max |ref fp32| of the posterior mean on "
           f"{CHECK_IMAGES} real images. Gate: fp32 mean, std and seeded "
           f"normalized sample < {CHECK_TOL:g}, bf16 mean and std <= "
           f"{BF16_SLACK:g} x the ref's own bf16 error + {BF16_ABS:g}. "
           "compile s = first bf16 call of 2d-compile(-cl).", ""]
    if prep:
        out += [f"HF token: user {prep.get('whoami')!r}, can write "
                f"{prep.get('can_write')} ({prep.get('write_reason')})", ""]
    head = ["gpu", "impl", "device", "batch", "img/s", "peak GiB",
            "decode img/s/thr", "cores", "$/h", "$/1k img"]
    for k in containers:
        head += [f"{k} ctr $", f"{k} ctr wall h"]
    head += ["fp32 err", "bf16 err", "compile s", "ladder (batch:img/s)"]
    out += ["| " + " | ".join(head) + " |",
            "|" + "---|" * len(head)]
    ok = [r for r in rows if r is not None]
    keys = {(r["gpu"], r["impl"]) for r in ok}
    ordered = sorted(ok, key=lambda r: r["usd_per_1k"])
    ladders = {(e["gpu_requested"], e["impl"]): e.get("ladder")
               for e in entries}
    for r in ordered:
        cells = [r["gpu"], r["impl"], str(r["device_name"]), str(r["batch"]),
                 f"{r['img_s']:.2f}", _fmt(r["peak_gib"], ".1f"),
                 f"{r['decode']:.2f}", str(r["cores"]), f"{r['usd_h']:.2f}",
                 f"{r['usd_per_1k']:.4f}"]
        for k in containers:
            p = r["proj"][k]
            cells += [f"{p['usd']:.2f}", f"{p['wall_h']:.2f}"]
        cells += [_fmt(r.get("fp32_err"), ".1e"), _fmt(r.get("bf16_err"), ".1e"),
                  _fmt(r.get("compile_s"), ".0f"),
                  _ladder_str(ladders.get((r["gpu"], r["impl"])))]
        out.append("| " + " | ".join(cells) + " |")
    failed = [e for e in entries
              if (e["gpu_requested"], e["impl"]) not in keys]
    if failed:
        out += ["", "Not usable:"]
        for e in failed:
            why = (_err_line(e["error"]) if e.get("error")
                   else "no successful batch / decode rate")
            out.append(f"- {e['gpu_requested']} {e['impl']} "
                       f"({e.get('device_name', '-')}): {why}"
                       + (f" [ladder {_ladder_str(e.get('ladder'))}]"
                          if e.get("ladder") else ""))
    out.append("")
    if choice is None:
        out.append("**No (GPU, impl) produced a usable measurement.**")
    else:
        lo = min(r["usd_per_1k"] for r in ok)
        ties = [f"{r['gpu']}/{r['impl']}" for r in ordered
                if r["usd_per_1k"] <= lo * (1 + TIE_FRAC)]
        out += [f"**Chosen: {choice['gpu']} / {choice['impl']}** "
                f"({choice['device_name']}): "
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
               dl_root=REMOTE_DL, vae_impl="ref", tar_dir=None):
    """cache_latents.py command of build worker i of n (run from the repo).
    tar_dir: a staged package (metadata.parquet + data/*.tar) instead of
    the HF dataset repo."""
    return [python, "scripts/cache_latents.py", "--out", out,
            "--num-workers", str(n), "--worker-index", str(i),
            "--batch", str(batch), "--decode-threads", str(decode_threads),
            "--download-dir", f"{dl_root}/w{i}", "--vae-impl", vae_impl,
            ] + (["--tar-dir", str(tar_dir)] if tar_dir else [])


def worker_env(base):
    """Subprocess env of a build worker: one intra-op CPU thread per decode
    thread. The VAE runs on the GPU, so torch's CPU threads only serve
    decode/normalize/stack; with OMP's default (all cores) every one of the
    --decode-threads would spawn its own core-wide pool and oversubscribe
    the cores the cost model sized as 1 core per decode thread."""
    return {**base, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1"}


def finalize_cmd(python="python", out=VOL_OUT, dl_root=REMOTE_DL,
                 tar_dir=None):
    """--write-manifest; tar_dir: verify against that package's parquet (a
    named cache), else the HF dataset repo's."""
    return [python, "scripts/cache_latents.py", "--out", str(out),
            "--write-manifest", "--download-dir", f"{dl_root}/manifest",
            ] + (["--tar-dir", str(tar_dir)] if tar_dir else [])


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
            t_w = time.perf_counter()
            lat = encode(nxt()).to(torch.bfloat16).cpu()  # warmup
            want = (b, *want_chw)
            if tuple(lat.shape) != want:
                raise ValueError(f"output {tuple(lat.shape)} != {want}")
            if not torch.isfinite(lat).all():
                raise ValueError("non-finite latents")
            if cuda is not None:
                cuda.synchronize()
            warmup_s = time.perf_counter() - t_w
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
                   "seconds": dt, "warmup_s": warmup_s}
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


def max_rel(a, b):
    """max |a - b| / max |b| (as float32)."""
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max()).item()


def to_plain(x):
    """Plain JSON types only (str/int/float/bool/None/list/dict): Modal
    returns are unpickled on the laptop, where torch is not installed (e.g.
    torch.__version__ is a torch.TorchVersion str subclass). Tensors / numpy
    scalars become numbers or lists, anything else its str()."""
    if x is None or type(x) in (bool, int, float, str):
        return x
    if isinstance(x, dict):
        return {str(k): to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_plain(v) for v in x]
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, str):
        return "".join([x])   # exact str, not the subclass
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x)
    if hasattr(x, "tolist"):   # torch / numpy tensors and scalars
        return to_plain(x.tolist())
    return str(x)


def gate_check(c, tol=CHECK_TOL, slack=BF16_SLACK, abs_tol=BF16_ABS):
    """Correctness verdict of one impl's check dict (check_impls) -> (ok,
    reasons). fp32 mean, std and seeded normalized sample max rel < tol;
    production-dtype mean and std error (vs ref fp32) <= slack * the ref's own
    production-dtype error + abs_tol. A missing measurement fails."""
    reasons = []
    if c.get("error"):
        return False, [f"error: {_err_line(c['error'])}"]
    for k in ("fp32_mean_max_rel", "fp32_std_max_rel",
              "fp32_sample_max_rel"):
        v = c.get(k)
        if v is None or not v < tol:
            reasons.append(f"{k} {v if v is None else format(v, '.2e')} "
                           f">= {tol:g}")
    for q in ("mean", "std"):
        v, ref = c.get(f"bf16_{q}_max_rel"), c.get(f"ref_bf16_{q}_max_rel")
        if v is None or ref is None:
            reasons.append(f"bf16_{q}_max_rel not measured")
            continue
        lim = slack * ref + abs_tol
        if not v <= lim:
            reasons.append(f"bf16_{q}_max_rel {v:.2e} > {slack:g} * ref "
                           f"{ref:.2e} + {abs_tol:g} = {lim:.2e}")
    return not reasons, reasons


def check_impls(vae, impls, x, tol=CHECK_TOL, cuda=None, log=print,
                factory=None):
    """Correctness of each impl against the diffusers reference on images x
    (N, 3, H, W), gated by gate_check:
      fp32 (VAE cast to fp32, TF32 off so cuDNN/cuBLAS really compute fp32):
        posterior mean/std max rel vs ref, and the normalized sample with the
        same torch seed (same noise draw);
      production dtype (the VAE's own, bf16): impl mean/std vs the fp32 ref,
        compared with the ref's own bf16 error (ref_bf16_*).
    -> (checks {impl: {fp32_*_max_rel, bf16_*_max_rel, ref_bf16_*_max_rel,
    first_call_s, ok, fail_reasons[, error]}}, encoders {impl: FastEncoder
    at the production dtype} for the impls that passed). first_call_s is the
    first production-dtype call (2d-compile(-cl): its compile time). factory
    {name: fn(vae) -> encoder} builds the named impls (variants of the
    profile, scripts/vae_profile.py); the others are vae_fast.make_encoder."""
    import torch
    import vae_fast

    def sync():
        if cuda is not None:
            cuda.synchronize()

    def seeded(enc):
        torch.manual_seed(CHECK_SEED)
        return enc(x)

    def make(impl):
        if impl in (factory or {}):
            return factory[impl](vae)
        return vae_fast.make_encoder(vae, impl)

    checks = {i: {} for i in impls}
    dtype0 = vae.dtype
    tf32 = (torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        vae.to_dtype(torch.float32)
        ref32 = vae_fast.make_encoder(vae, "ref")
        m_ref, s_ref = ref32.dist(x)
        z_ref = seeded(ref32)
        for impl in impls:
            c = checks[impl]
            if impl == "ref":
                c.update(fp32_mean_max_rel=0.0, fp32_std_max_rel=0.0,
                         fp32_sample_max_rel=0.0)
                continue
            try:
                enc = make(impl)
                m, s = enc.dist(x)
                c.update(fp32_mean_max_rel=max_rel(m, m_ref),
                         fp32_std_max_rel=max_rel(s, s_ref),
                         fp32_sample_max_rel=max_rel(seeded(enc), z_ref))
                del enc
            except Exception:  # noqa: BLE001
                c["error"] = traceback.format_exc()[-3000:]
    finally:
        torch.backends.cudnn.allow_tf32, \
            torch.backends.cuda.matmul.allow_tf32 = tf32
        vae.to_dtype(dtype0)
    # production dtype: the ref's own error is the yardstick
    m16, s16 = vae_fast.make_encoder(vae, "ref").dist(x)
    ref_err = {"ref_bf16_mean_max_rel": max_rel(m16, m_ref),
               "ref_bf16_std_max_rel": max_rel(s16, s_ref)}
    encoders = {}
    for impl in impls:
        c = checks[impl]
        c.update(ref_err)
        if "error" not in c:
            try:
                enc = make(impl)
                sync()
                t0 = time.perf_counter()
                m, s = enc.dist(x)
                sync()
                c["first_call_s"] = time.perf_counter() - t0
                c["bf16_mean_max_rel"] = max_rel(m, m_ref)
                c["bf16_std_max_rel"] = max_rel(s, s_ref)
            except Exception:  # noqa: BLE001
                c["error"] = traceback.format_exc()[-3000:]
        c["ok"], c["fail_reasons"] = gate_check(c, tol)
        nan = float("nan")
        log(f"check {impl}: fp32 mean/std/sample "
            f"{c.get('fp32_mean_max_rel', nan):.2e}/"
            f"{c.get('fp32_std_max_rel', nan):.2e}/"
            f"{c.get('fp32_sample_max_rel', nan):.2e}, {dtype0} mean/std "
            f"{c.get('bf16_mean_max_rel', nan):.2e}/"
            f"{c.get('bf16_std_max_rel', nan):.2e} (ref "
            f"{ref_err['ref_bf16_mean_max_rel']:.2e}/"
            f"{ref_err['ref_bf16_std_max_rel']:.2e}), first call "
            f"{c.get('first_call_s', nan):.1f}s -> "
            + ("ok" if c["ok"] else "FAILED: " + "; ".join(c["fail_reasons"])))
        if c["ok"]:
            encoders[impl] = enc
    return checks, encoders


def bench_gpu_impl(gpu_requested, n_decode, impls=BENCH_DEFAULT_IMPLS, *,
                   vae=None,
                   tar=None, bucket=BENCH_BUCKET, device="cuda",
                   ladder=LADDER, min_images=MIN_IMAGES,
                   min_seconds=MIN_SECONDS, decode_timing_n=DECODE_TIMING_N,
                   log=print):
    """Runs in the bench container. Never raises: failures end up in
    res["error"] / res["impls"][impl]["error"] (with whatever was measured
    before). Returns plain JSON types only (to_plain). `vae` (a
    cache_dataset.VaeEncoder-like object), `tar`, `bucket`, `device` and the
    ladder sizes are injectable for CPU tests."""
    res = {"gpu_requested": gpu_requested}
    t_start = time.time()
    try:
        from concurrent.futures import ThreadPoolExecutor

        import torch

        cl = import_cache_latents()
        cuda = torch.cuda if device == "cuda" else None
        res.update(device_name=(torch.cuda.get_device_name(0) if cuda
                                else str(device)),
                   torch=torch.__version__, torch_cuda=torch.version.cuda)
        try:
            import diffusers
            res["diffusers"] = diffusers.__version__
        except ImportError:
            res["diffusers"] = None
        log(f"{gpu_requested}: {res['device_name']} torch {torch.__version__}"
            f" (cuda {torch.version.cuda})")
        w, h = bucket
        tar = Path(VOL_BENCH) / BENCH_TAR if tar is None else Path(tar)
        samples = [s for s in cl.read_tar(tar)
                   if (s["w"], s["h"]) == (w, h)][:n_decode]
        if len(samples) < max(decode_timing_n, CHECK_IMAGES):
            raise RuntimeError(f"only {len(samples)} {w}x{h} samples in {tar}")
        # one intra-op thread: the build runs every decode thread with
        # OMP_NUM_THREADS=1 (worker_env), so this is the per-core rate
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            t0 = time.perf_counter()
            for s in samples[:decode_timing_n]:
                cl.decode(s)
            res["decode_img_s_per_thread"] = (decode_timing_n
                                              / (time.perf_counter() - t0))
        finally:
            torch.set_num_threads(n_threads)
        log(f"decode: {res['decode_img_s_per_thread']:.2f} img/s per thread "
            f"({decode_timing_n} {w}x{h} samples, 1 intra-op thread)")
        # bench setup only (not timed): default threads
        with ThreadPoolExecutor(DECODE_POOL) as pool:
            imgs = [d[0] for d in pool.map(cl.decode, samples)]
        res["n_decoded"] = len(imgs)
        if vae is None:
            from cache_dataset import VaeEncoder

            t0 = time.perf_counter()
            vae = VaeEncoder(VAE_REPO, "cuda", torch.bfloat16)
            res["vae_load_s"] = time.perf_counter() - t0
        res["vae_dtype"] = vae.dtype
        # correctness on real images first; failed impls are skipped
        checks, encoders = check_impls(
            vae, list(impls), torch.stack(imgs[:CHECK_IMAGES]), cuda=cuda,
            log=log)
        res["impls"] = {i: {"check": checks[i]} for i in impls}
        for impl in impls:
            r = res["impls"][impl]
            if impl not in encoders:
                continue
            # production setting: the 2d impls turn cudnn.benchmark on
            # (vae_fast), the reference path never does
            torch.backends.cudnn.benchmark = impl != "ref"
            log(f"--- ladder {impl}")
            try:
                # timed with 1 CPU thread, like production (intra_threads)
                r["ladder"] = run_ladder(
                    encoders.pop(impl), imgs, ladder,
                    (LAT_CHANNELS, h // VAE_SCALE, w // VAE_SCALE),
                    cuda=cuda, min_images=min_images,
                    min_seconds=min_seconds, log=log, intra_threads=1)
                r.update(best_of(r["ladder"]))
            except Exception:  # noqa: BLE001
                r["error"] = traceback.format_exc()[-4000:]
                log(r["error"])
            if cuda is not None:
                cuda.empty_cache()
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-4000:]
        log(res["error"])
    res["seconds"] = time.time() - t_start
    return to_plain(res)


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
    return to_plain(out)


def gpu_name_smi():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def build_worker_impl(i, n, batch, decode_threads, vol, vae_impl="ref",
                      cache_name=""):
    """One cache_latents.py worker; the volume is committed after every
    finished shard (explicitly, on top of Modal's background commits, so a
    killed container loses at most its in-flight shard) and at the end.
    cache_name: a named cache (/vol/krea2-NAME from the staged /vol/src/NAME)
    instead of the v2 one."""
    setup_hf_token()
    env = worker_env(os.environ)
    res = {"worker": i, "device_name": gpu_name_smi()}
    job = resolve_job(cache_name, mode="worker")
    if job["name"]:
        try:   # never add shards of another input to a started cache
            staged_generation(job)
        except SystemExit as e:
            res.update(rc=None, error=str(e), seconds=0)
            return to_plain(res)
    cmd = worker_cmd(i, n, batch, decode_threads, python=sys.executable,
                     out=job["out"], vae_impl=vae_impl, tar_dir=job["src"])
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
    return to_plain(res)


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
    return to_plain(res)


# ------------------------------------------- named caches: Drive in, R2 out
def resolve_mode(bench=False, build=False, finalize_only=False,
                 stage_only=False, profile=False, reset_upload_cache=False,
                 cache_name="", src_drive="", dest_r2=""):
    """CLI mode flags -> bench | build | finalize | stage | profile
    (SystemExit on misuse, before any Modal call). --profile takes only
    --gpu: no cache, Drive, R2 or upload flags."""
    modes = [m for m, on in (("--bench", bench), ("--build", build),
                             ("--finalize-only", finalize_only),
                             ("--stage-only", stage_only),
                             ("--profile", profile)) if on]
    if len(modes) != 1:
        raise SystemExit("choose exactly one of --bench, --build, "
                         "--finalize-only, --stage-only, --profile (got "
                         f"{modes or 'none'})")
    if profile:
        extra = [f for f, v in (("--cache-name", cache_name),
                                ("--src-drive", src_drive),
                                ("--dest-r2", dest_r2),
                                ("--reset-upload-cache", reset_upload_cache))
                 if v]
        if extra:
            raise SystemExit(f"--profile takes no {', '.join(extra)}")
        return "profile"
    if (bench or stage_only) and reset_upload_cache:
        raise SystemExit("--reset-upload-cache goes with --build or "
                         "--finalize-only")
    return ("bench" if bench else "build" if build else
            "stage" if stage_only else "finalize")


def profile_cost_cap(gpu, cores=BENCH_CPU, timeout_s=PROFILE_TIMEOUT_S):
    """-> (usd cap, line): the profile container's hard timeout x its $/h
    (GPU + cores + memory)."""
    h = hourly_usd(gpu, cores)
    cap = h * timeout_s / 3600
    return cap, (f"cost cap: {timeout_s / 60:g} min timeout x ${h:.2f}/h "
                 f"({gpu_key(gpu)} ${GPU_PRICE_USD_H[gpu_key(gpu)]:.2f} + "
                 f"{cores} cores + {MEM_GIB} GiB memory modelled (billed by "
                 f"use, no reservation), prices of {PRICES_DATE}) "
                 f"= ${cap:.2f}, plus cents for the CPU prep container")


def write_profile_results(r, root):
    """profile_gpu result {json, report_md, trace_gz} -> files in root
    (report.md, profile.json, trace.json.gz when present) -> root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.md").write_text((r.get("report_md") or "") + "\n")
    (root / "profile.json").write_text(r.get("json") or "{}")
    if r.get("trace_gz"):
        (root / "trace.json.gz").write_bytes(r["trace_gz"])
    return root


def profile_gpu_impl(gpu_requested, ts, out_root=VOL_PROFILE,
                     budget_s=PROFILE_BUDGET_S, commit=None, log=print):
    """Runs in the profile container: VAE (bf16, from the volume's HF cache)
    -> vae_profile.profile_impl at the bench bucket, batch PROFILE_BATCH ->
    {json (plain, serialized here so a torch type fails in the container,
    not on the laptop), report_md, trace_gz (bytes), vol_dir}; the same
    files go to out_root/<ts>/ on the volume, also after every section
    (checkpoint, then commit()), so a container killed by the timeout
    leaves the report so far; the max-autotune variant starts only with
    PROFILE_MAT_RESERVE_S of PROFILE_TIMEOUT_S left. Never raises."""
    t_start = time.monotonic()
    d = Path(out_root) / ts
    res = {"gpu_requested": gpu_requested}
    md = None

    def pack(r, md):
        tp = d / "trace.json.gz"
        return {"json": json.dumps(to_plain(r), indent=1), "report_md": md,
                "trace_gz": tp.read_bytes() if tp.exists() else None,
                "vol_dir": str(d)}

    def checkpoint(r):
        md = vae_profile.render_markdown(
            r, title=f"VAE encoder profile, {gpu_requested} (partial: "
            f"{', '.join(r.get('sections') or {})})")
        write_profile_results(pack({**res, **r}, md), d)
        if commit:
            commit()

    try:
        import torch

        import_cache_latents()   # /repo/scripts on sys.path
        import vae_profile
        from cache_dataset import VaeEncoder

        d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        vae = VaeEncoder(VAE_REPO, "cuda", torch.bfloat16)
        res["vae_load_s"] = time.perf_counter() - t0
        w, h = BENCH_BUCKET
        res.update(vae_profile.profile_impl(
            vae, "cuda", h=h, w=w, batch=PROFILE_BATCH, budget_s=budget_s,
            min_images=MIN_IMAGES, min_seconds=MIN_SECONDS,
            trace_path=str(d / "trace.json.gz"), check_impls=check_impls,
            run_ladder=run_ladder, checkpoint=checkpoint,
            mat_start_by=t_start + PROFILE_TIMEOUT_S - PROFILE_MAT_RESERVE_S,
            log=log))
        md = vae_profile.render_markdown(
            res, title=f"VAE encoder profile, {gpu_requested}")
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-4000:]
        log(res["error"])
    if md is None:
        md = (f"# VAE encoder profile, {gpu_requested}: FAILED\n\n```\n"
              f"{res.get('error')}\n```")
    out = pack(res, md)
    try:
        write_profile_results(out, d)
    except Exception as e:  # noqa: BLE001 - the return still carries it
        log(f"volume copy failed: {type(e).__name__}: {e}")
    return out


def resolve_job(cache_name="", src_drive="", dest_r2="", mode="build"):
    """CLI flags -> {name, out, src, src_remote, dest} (validated; SystemExit
    on misuse). mode: bench | build | finalize | stage | worker (paths only).

    name "" = the v2 cache: /vol/krea2 from the HF image repo, uploaded to the
    HF latents repo; --src-drive / --dest-r2 are refused there (they would
    mix another dataset into v2's dir or repo). NAME = /vol/krea2-NAME, input
    staged into /vol/src/NAME: needs --dest-r2 (the HF target repo is v2's)
    for build / finalize, and --src-drive for build / stage."""
    name = (cache_name or "").strip()
    src_drive, dest_r2 = (src_drive or "").strip(), (dest_r2 or "").strip()
    if mode == "bench":
        if name or src_drive or dest_r2:
            raise SystemExit("--bench takes no --cache-name / --src-drive / "
                             "--dest-r2")
        return {"name": "", "out": VOL_OUT, "src": None, "src_remote": None,
                "dest": None}
    if not name:
        if src_drive or dest_r2:
            raise SystemExit(
                "--src-drive / --dest-r2 need --cache-name NAME: the default "
                f"cache {VOL_OUT} holds the v2 cache (HF in, HF out) and must "
                "never be mixed with another dataset")
        if mode == "stage":
            raise SystemExit("--stage-only needs --cache-name and --src-drive")
        return {"name": "", "out": VOL_OUT, "src": None, "src_remote": None,
                "dest": None}
    if not CACHE_NAME_RE.match(name):
        raise SystemExit(f"--cache-name {name!r}: want {CACHE_NAME_RE.pattern}")
    out, src = f"{VOL_OUT}-{name}", f"{VOL_SRC}/{name}"
    for p in (out, src):
        if any(p == d or p.startswith(d + "/") for d in SHARED_DIRS):
            raise SystemExit(f"--cache-name {name!r} maps to {p}, inside a "
                             "shared / v2 dir")
    if mode in ("build", "stage") and not src_drive:
        raise SystemExit(f"--cache-name {name} --{mode} needs --src-drive "
                         "(the image package to stage into " + src + ")")
    if mode in ("build", "finalize") and not dest_r2:
        raise SystemExit(f"--cache-name {name} needs --dest-r2 PREFIX: the HF "
                         f"upload target {OUT_REPO} is the v2 cache's")
    return {"name": name, "out": out, "src": src,
            "src_remote": resolve_drive_src(src_drive) if src_drive else None,
            "dest": parse_r2_prefix(dest_r2) if dest_r2 else None}


def resolve_drive_src(spec):
    """'selection_v3/package' -> 'gdrive:danbooru_pipeline/selection_v3/
    package'; a 'remote:path' is kept. No '..' / empty segments."""
    s = str(spec).strip().rstrip("/")
    if ":" not in s:
        s = f"{DRIVE_ROOT}/{s.lstrip('/')}"
    remote, _, path = s.partition(":")
    segs = path.split("/")
    if (not remote or not path or ".." in segs or "" in segs
            or not re.fullmatch(r"[A-Za-z0-9_-]+", remote)):
        raise SystemExit(f"--src-drive {spec!r}: want a path under "
                         f"{DRIVE_ROOT} or remote:path")
    return s


def parse_r2_prefix(p):
    """'latents_v3/' -> 'latents_v3' (bucket-relative, no '..')."""
    s = str(p).strip().strip("/")
    if not R2_PREFIX_RE.match(s):
        raise SystemExit(f"--dest-r2 {p!r}: want a bucket-relative prefix "
                         "like latents_v3/")
    return s


def build_estimate(n_images, containers, gpu=V2_BUILD["gpu"],
                   cores=V2_BUILD["cores"], img_s=V2_BUILD["img_s"],
                   overhead_min=DEFAULT_OVERHEAD_MIN):
    """cost_row's projection (hourly_usd, every container pays the startup
    overhead) at a measured per-container rate (default: the v2 build) ->
    {usd, wall_h, gpu_h, usd_h, worker_timeout_h}."""
    hourly = hourly_usd(gpu, cores)
    hours = n_images / img_s / 3600
    over = overhead_min / 60
    return {"usd": hourly * (hours + containers * over),
            "wall_h": hours / containers + over, "gpu_h": hours,
            "usd_h": hourly,
            "worker_timeout_h": worker_timeout_h(
                {"hours_1": hours, "overhead_h": over}, containers)}


def render_estimate(n_images, gpu, cpu, containers, vae_impl, batch):
    """One line: build_estimate at the MEASURED_RATES entry matching the
    build's (gpu, impl, batch), labelled with its source, else a note."""
    key = (gpu_key(gpu), vae_impl, batch)
    v = next((r for r in MEASURED_RATES
              if (r["gpu"], r["impl"], r["batch"]) == key), None)
    if v is None:
        known = "; ".join(f"{r['gpu']}/{r['impl']}/batch {r['batch']} at "
                          f"{r['img_s']:.1f} img/s ({r['label']})"
                          for r in MEASURED_RATES)
        return (f"estimate: no measured rate for {gpu}/{vae_impl}/batch "
                f"{batch} (measured per container: {known})")
    e = build_estimate(n_images, containers, gpu=gpu, cores=cpu,
                       img_s=v["img_s"])
    return (f"estimate: {n_images} images on {containers} x {gpu} at "
            f"{v['img_s']:.1f} img/s each ({v['label']}): ${e['usd']:.2f} GPU "
            f"containers (${e['usd_h']:.2f}/h each, {e['gpu_h']:.2f} h + "
            f"{containers} x {DEFAULT_OVERHEAD_MIN:g} min startup; memory "
            f"billed by use, modelled as {MEM_GIB} GiB), wall "
            f"{e['wall_h'] * 60:.0f} min; cost model's worker timeout "
            f"{e['worker_timeout_h']} h")


def run_rclone(args, env, runner=subprocess.run, rclone=RCLONE_BIN,
               capture=False):
    """rclone ARGS with env (credentials only there, never in argv / logs).
    capture: stdout/stderr as bytes in the result, else streamed."""
    print("$ rclone " + " ".join(map(str, args)), flush=True)
    return runner([rclone, *map(str, args)], env=env, capture_output=capture)


def _tail(b, n=600):
    return (b or b"").decode("utf-8", "replace")[-n:].strip()


def r2_rclone_env(env):
    """-> (subprocess env with the env-only rclone remote `r2` (as the node
    kit's dl.sh r2()), bucket). Other RCLONE_* / R2_* / HF token vars are
    dropped."""
    missing = [k for k in R2_ENV if not env.get(k)]
    if missing:
        raise RuntimeError(f"Modal secret krea2-r2 lacks {missing}")
    base = {k: v for k, v in env.items()
            if not k.startswith(("RCLONE_", "R2_")) and k not in TOKEN_ENV_NAMES}
    return {**base, "RCLONE_CONFIG": "/dev/null",
            "RCLONE_CONFIG_R2_TYPE": "s3",
            "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
            "RCLONE_CONFIG_R2_ENDPOINT":
                f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID": env["R2_ACCESS_KEY_ID"],
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": env["R2_SECRET_ACCESS_KEY"],
            "RCLONE_CONFIG_R2_NO_CHECK_BUCKET": "true"}, env["R2_BUCKET"]


def r2_list(url, env, runner=subprocess.run, rclone=RCLONE_BIN):
    """{relpath: size} of every object under url ({} for an absent prefix)."""
    p = run_rclone(["lsjson", "-R", "--files-only", "--fast-list", url], env,
                   runner, rclone, capture=True)
    if p.returncode != 0:
        if "directory not found" in _tail(p.stderr, 4000).lower():
            return {}
        raise RuntimeError(f"rclone lsjson {url} rc={p.returncode}: "
                           f"{_tail(p.stderr)}")
    return {e["Path"]: int(e["Size"]) for e in json.loads(p.stdout or b"[]")}


def upload_list(out_dir):
    """{relpath: size} of the data files a named cache publishes:
    metadata.parquet + every shard the manifest lists (the manifest itself
    is uploaded last, separately)."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / MANIFEST).read_text())
    files = {PARQUET: (out_dir / PARQUET).stat().st_size}
    for s in manifest["shards"]:
        files[f"latents/{s}"] = (out_dir / "latents" / s).stat().st_size
    return files


def r2_conflicts(remote, local, local_manifest, remote_manifest):
    """Why uploading `local` ({relpath: size}, incl. MANIFEST) into a prefix
    holding `remote` is unsafe -> problems ([] = go). A prefix without
    latents/manifest.json is new or an unfinished upload (rclone copy
    --checksum resumes it). One with a manifest is published and immutable:
    only a re-run of the same upload passes (manifest bytes equal, every
    remote object one of ours with the same size), and that re-run only adds
    missing objects (finalize_r2_impl: --ignore-existing). Leftover write
    probes (.write-probe-*) are ignored."""
    if MANIFEST not in remote:
        return []
    probs = []
    if local_manifest is None or remote_manifest != local_manifest:
        probs.append(f"{MANIFEST} on R2 differs from the local one (another "
                     "build of this cache, or another dataset)")
    for p, n in sorted(remote.items()):
        if p.rsplit("/", 1)[-1].startswith(PROBE_PREFIX):
            continue
        if p not in local:
            probs.append(f"remote object {p} is not part of this cache")
        elif local[p] != n:
            probs.append(f"size {p}: remote {n} != local {local[p]}")
    return probs


def input_generation(files):
    """Identity of a package listing {path: {size, md5}} (Drive's own sizes
    and md5s): sha256 of its canonical JSON."""
    blob = json.dumps(sorted((p, f["size"], f["md5"]) for p, f in files.items()),
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def generation_refusal(out_dir, gen, have):
    return (f"{out_dir} was built from input generation {have} but the "
            f"package now is {gen}: its shards could mix two inputs. Pick a "
            f"new --cache-name, or delete the cache explicitly: modal volume "
            f"rm -r {VOLUME_NAME} {str(out_dir)[len(VOL) + 1:]}")


def staged_generation(job):
    """Input generation of a named cache: /vol/src/NAME/stage.json (written
    by stage) must equal the one the cache dir was started with
    (/vol/krea2-NAME/GEN_JSON) -> generation, else SystemExit."""
    st = _read_json(Path(job["src"]) / STAGE_JSON)
    mk = _read_json(Path(job["out"]) / GEN_JSON)
    if not st or not mk:
        raise SystemExit(f"{job['src']}/{STAGE_JSON} or {job['out']}/"
                         f"{GEN_JSON} missing: stage the package first "
                         "(--build or --stage-only)")
    if st.get("generation") != mk.get("generation"):
        raise SystemExit(generation_refusal(job["out"], st.get("generation"),
                                            mk.get("generation")))
    return st["generation"]


def stage_listing(src_remote, conf, env, runner=subprocess.run,
                  rclone=RCLONE_BIN):
    """{path: {size, md5}} of the package files on Drive (STAGE_INCLUDE;
    md5 = Drive's own hash, nothing is downloaded or hashed here)."""
    args = ["--config", conf, "lsjson", "-R", "--files-only", "--hash",
            "--hash-type", "md5"]
    for pat in STAGE_INCLUDE:
        args += ["--include", pat]
    p = run_rclone(args + [src_remote], env, runner, rclone, capture=True)
    if p.returncode != 0:
        raise RuntimeError(f"rclone lsjson {src_remote} rc={p.returncode}: "
                           f"{_tail(p.stderr)}")
    return {e["Path"]: {"size": int(e["Size"]),
                        "md5": (e.get("Hashes") or {}).get("md5")}
            for e in json.loads(p.stdout or b"[]")}


def check_staged(src_dir, remote_files):
    """Staged package -> {n_rows, n_tars, bytes, missing}: every tar the
    parquet names must be in the Drive listing `remote_files` and a file
    under src_dir."""
    import pyarrow.parquet as pq

    src_dir = Path(src_dir)
    t = pq.read_table(str(src_dir / PARQUET), columns=["id", "tar"])
    tars = sorted(set(t.column("tar").to_pylist()))
    missing = [x for x in tars
               if x not in remote_files or not (src_dir / x).is_file()]
    return {"n_rows": t.num_rows, "n_tars": len(tars), "missing": missing,
            "bytes": sum((src_dir / x).stat().st_size for x in tars
                         if x not in missing)}


def write_rclone_conf(env, path=RCLONE_CONF):
    """RCLONE_CONF_B64 (secret `rclone-gdrive`) -> 0600 file (never printed)."""
    import base64

    b64 = env.get("RCLONE_CONF_B64")
    if not b64:
        raise RuntimeError("Modal secret rclone-gdrive lacks RCLONE_CONF_B64")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, base64.b64decode(b64))
    finally:
        os.close(fd)
    return path


def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def stage_impl(vol, cache_name, src_drive, runner=subprocess.run, env=None,
               rclone=RCLONE_BIN, conf_path=RCLONE_CONF):
    """Drive package -> /vol/src/<name>:
      1. list it on Drive (path, size, Drive's md5) -> input generation; a
         cache dir already started from another generation is refused here,
         before anything changes;
      2. `rclone sync --checksum` (resumable; local tars the package no
         longer has are deleted; stage.json is outside the filter and kept);
      3. list again: the package must not have changed meanwhile;
      4. must-fail completeness check (check_staged: every tar the parquet
         names is in the listing and on disk);
      5. stage.json = {src_remote, generation, files}, and the cache dir's
         GEN_JSON = the generation if it has none yet (it is then bound to
         this input; a cache dir with shard output but no GEN_JSON is never
         adopted). stage.json is deleted (and committed) before the sync and
         only written back after 3-4 pass, so a failed restage leaves no
         stage.json and the workers / finalize_r2 refuse.
    Never raises: -> {ok, error, src, src_remote, generation, n_rows, n_tars,
    bytes, missing, log}."""
    env = os.environ if env is None else env
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    res = {"ok": False, "error": None, "log": log}
    try:
        job = resolve_job(cache_name, src_drive, "", "stage")
        src, out = Path(job["src"]), Path(job["out"])
        res.update(src=str(src), src_remote=job["src_remote"])
        try:
            vol.reload()
        except Exception as e:  # noqa: BLE001
            say(f"WARNING: volume reload failed: {type(e).__name__}: {e}")
        conf = write_rclone_conf(env, conf_path)
        renv = {k: v for k, v in env.items() if not k.startswith("RCLONE_")}
        files = stage_listing(job["src_remote"], conf, renv, runner, rclone)
        if PARQUET not in files:
            raise RuntimeError(f"no {PARQUET} in {job['src_remote']}")
        gen = input_generation(files)
        res["generation"] = gen
        say(f"package {job['src_remote']}: {len(files)} files, input "
            f"generation {gen[:16]}")
        mk = _read_json(out / GEN_JSON)
        if mk and mk.get("generation") != gen:
            raise SystemExit(generation_refusal(out, gen, mk.get("generation")))
        lat = out / "latents"
        if mk is None and lat.is_dir() and any(lat.iterdir()):
            raise SystemExit(
                f"{out} already holds shard output but no {GEN_JSON}: its "
                "input is unknown, it is never adopted. Pick a new "
                f"--cache-name, or delete the cache explicitly: modal volume "
                f"rm -r {VOLUME_NAME} {str(out)[len(VOL) + 1:]}")
        # the sync may mutate the inputs: no stage.json (workers / finalize
        # refuse) until the new state is validated below
        (src / STAGE_JSON).unlink(missing_ok=True)
        commit_volume(vol, say)
        src.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        args = ["--config", conf, "sync", job["src_remote"], str(src)]
        for pat in STAGE_INCLUDE:
            args += ["--include", pat]
        rc = run_rclone(args + ["--checksum", "--transfers", "8",
                                "--checkers", "16", *RCLONE_COMMON],
                        renv, runner, rclone).returncode
        say(f"rclone sync {job['src_remote']} -> {src}: rc {rc} in "
            f"{(time.time() - t0) / 60:.1f} min")
        commit_volume(vol, say)
        if rc != 0:
            raise RuntimeError(f"rclone sync exit {rc}")
        if stage_listing(job["src_remote"], conf, renv, runner,
                         rclone) != files:
            raise RuntimeError(f"{job['src_remote']} changed while it was "
                               "staged: re-run once it is final")
        st = check_staged(src, files)
        res.update(st)
        say(f"staged: {st['n_rows']} rows, {st['n_tars']} tars "
            f"({st['bytes'] / 1e9:.1f} GB), {len(st['missing'])} missing")
        if st["missing"]:
            raise RuntimeError(f"{len(st['missing'])} tars named in the "
                               f"parquet are not in the package, e.g. "
                               f"{st['missing'][:5]}")
        _write_json(src / STAGE_JSON, {"src_remote": job["src_remote"],
                                       "generation": gen, "files": files})
        if mk is None:
            out.mkdir(parents=True, exist_ok=True)
            _write_json(out / GEN_JSON, {"src_remote": job["src_remote"],
                                         "generation": gen})
        commit_volume(vol, say)
        res["ok"] = True
    except (Exception, SystemExit) as e:  # noqa: BLE001 - reported, never raised
        res["error"] = (str(e) if isinstance(e, (SystemExit, RuntimeError))
                        else traceback.format_exc()[-4000:])
        say("FATAL: " + res["error"])
    finally:
        Path(conf_path).unlink(missing_ok=True)
    return to_plain(res)


class R2ConflictError(Exception):
    pass


def r2_write_probe(url, env, runner=subprocess.run, rclone=RCLONE_BIN):
    """Write + delete a tiny <url>/.write-probe-<random> object: the
    credentials must be able to publish there. RuntimeError if not."""
    import secrets

    probe = f"{url}/{PROBE_PREFIX}{secrets.token_hex(8)}"
    fd, tmp = tempfile.mkstemp(prefix="r2_probe_")
    with os.fdopen(fd, "wb") as f:
        f.write(b"krea2 write probe\n")
    try:
        rc = run_rclone(["copyto", tmp, probe], env, runner, rclone).returncode
        if rc != 0:
            raise RuntimeError(f"R2 write probe failed (rclone copyto rc {rc}):"
                               f" the krea2-r2 credentials cannot write {url} "
                               "(read-only token?)")
        rc = run_rclone(["deletefile", probe], env, runner, rclone).returncode
        if rc != 0:
            raise RuntimeError(f"R2 write probe {probe} written but not "
                               f"deleted (rc {rc}): delete it by hand")
    finally:
        Path(tmp).unlink(missing_ok=True)


def finalize_r2_impl(vol, cache_name, dest_r2, runner=subprocess.run,
                     python=sys.executable, cwd=REMOTE_REPO, env=None,
                     rclone=RCLONE_BIN, precheck=False):
    """A named cache's finalize: input generation check (staged_generation),
    --write-manifest against the staged package's parquet, then
    metadata.parquet + the manifest's shards to r2:<bucket>/<prefix>/ (rclone
    copy --checksum; rclone's own transfer check is the only upload check)
    and latents/manifest.json last, so a manifest on R2 means a complete
    cache. A prefix with a manifest is published and immutable: a foreign
    one is refused before the copy (r2_conflicts; a cheap pre-check refuses
    it already before hashing when there is no local manifest), the same one
    only gets its missing objects (--ignore-existing: nothing, incl.
    metadata.parquet, is ever replaced). precheck: only the generation
    check, that pre-check and a write probe (r2_write_probe), run by --build
    before any GPU starts. Never touches HF. Never raises: -> {ok, repo,
    error, problems, n_files, bytes, log}."""
    env = os.environ if env is None else env
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    res = {"ok": False, "repo": None, "error": None, "problems": [],
           "log": log}
    try:
        job = resolve_job(cache_name, "", dest_r2, "finalize")
        out = Path(job["out"])
        mpath = out / MANIFEST
        try:
            vol.reload()
        except Exception as e:  # noqa: BLE001
            say(f"WARNING: volume reload failed: {type(e).__name__}: {e}")
        gen = staged_generation(job)
        say(f"input generation {gen[:16]} (stage.json == cache dir)")
        renv, bucket = r2_rclone_env(env)
        url = f"r2:{bucket}/{job['dest']}"
        res["repo"] = url

        def conflicts():
            remote = r2_list(url, renv, runner, rclone)
            rman = None
            if MANIFEST in remote:
                p = run_rclone(["cat", f"{url}/{MANIFEST}"], renv, runner,
                               rclone, capture=True)
                if p.returncode != 0:
                    raise RuntimeError(f"rclone cat {MANIFEST} rc="
                                       f"{p.returncode}: {_tail(p.stderr)}")
                rman = p.stdout
            if not mpath.exists():
                return r2_conflicts(remote, {}, None, rman), remote
            local = {**upload_list(out), MANIFEST: mpath.stat().st_size}
            return (r2_conflicts(remote, local, mpath.read_bytes(), rman),
                    remote)

        # 1. cheap: refuse a finished foreign prefix before hashing ~100 GB
        res["problems"], _ = conflicts()
        if res["problems"]:
            raise R2ConflictError(url)
        if precheck:
            r2_write_probe(url, renv, runner, rclone)
            say(f"R2 pre-check ok: {url} is new, unfinished or this cache; "
                "write probe ok")
            res["ok"] = True
            return to_plain(res)
        # 2. verify all shards + digests against the staged parquet
        cmd = finalize_cmd(python=python, out=out, tar_dir=job["src"])
        say("$ " + " ".join(cmd))
        rc = runner(cmd, cwd=cwd).returncode
        if rc != 0:
            raise RuntimeError(f"cache_latents.py --write-manifest exit {rc}")
        if not commit_volume(vol, say):
            say("continuing: the manifest is on the container's mount")
        # 3. the final local state against the prefix
        res["problems"], remote = conflicts()
        if res["problems"]:
            raise R2ConflictError(url)
        published = MANIFEST in remote
        files = upload_list(out)
        fd, lst = tempfile.mkstemp(prefix="r2_upload_", suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write("".join(f"{p}\n" for p in sorted(files)))
        t0 = time.time()
        rc = run_rclone(["copy", str(out), url, "--files-from-raw", lst,
                         *R2_UPLOAD_FLAGS,
                         *(["--ignore-existing"] if published else []),
                         *RCLONE_COMMON], renv, runner, rclone).returncode
        if rc != 0:
            raise RuntimeError(f"rclone copy -> {url} exit {rc} (re-run "
                               "--finalize-only: done files are skipped)")
        if published:
            say(f"{url} is published (same manifest): only missing objects "
                "were added, nothing replaced")
        else:
            rc = run_rclone(["copyto", str(mpath), f"{url}/{MANIFEST}",
                             "--checksum", *RCLONE_COMMON], renv, runner,
                            rclone).returncode
            if rc != 0:
                raise RuntimeError(f"rclone copyto {MANIFEST} exit {rc}")
        res["n_files"] = len(files) + 1
        res["bytes"] = sum(files.values()) + mpath.stat().st_size
        say(f"uploaded {res['n_files']} files, {res['bytes'] / 1e9:.2f} GB "
            f"to {url} in {(time.time() - t0) / 60:.1f} min")
        res["ok"] = True
    except R2ConflictError as e:
        res["error"] = (f"refusing to upload into {e}: it holds a finished "
                        "cache that is not this one (see problems); pick "
                        "another --dest-r2 or remove that prefix by hand")
        say("FATAL: " + res["error"])
        for p in res["problems"][:20]:
            say("  " + p)
    except (Exception, SystemExit) as e:  # noqa: BLE001 - reported, never raised
        res["error"] = (str(e) if isinstance(e, (SystemExit, RuntimeError))
                        else traceback.format_exc()[-4000:])
        say("FATAL: " + res["error"])
    return to_plain(res)


# ------------------------------------------------------------------ Modal app
if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        # torch.compile: triton builds its launcher stubs with a C compiler
        # at runtime (inductor's CPU codegen needs one too)
        .apt_install("build-essential")
        .pip_install(TORCH_PIN, extra_index_url=TORCH_INDEX)
        .pip_install(*PIP_PACKAGES)
        # pinned rclone (Drive staging, R2 upload): zip sha256 checked
        .run_commands(RCLONE_INSTALL)
        .env(CONTAINER_ENV)
        .add_local_dir(LOCAL_REPO / "krea2_jax", f"{REMOTE_REPO}/krea2_jax",
                       ignore=["**/__pycache__"])
        .add_local_dir(LOCAL_REPO / "scripts", f"{REMOTE_REPO}/scripts",
                       ignore=["**/__pycache__"])
    )
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True,
                                 version=VOLUME_VERSION)
    hf_secret = modal.Secret.from_name("huggingface")
    rclone_secret = modal.Secret.from_name("rclone-gdrive")
    r2_secret = modal.Secret.from_name("krea2-r2")
    app = modal.App("krea2-cache-latents", image=image)
    mounts = {"volumes": {VOL: vol}, "secrets": [hf_secret]}

    @app.function(cpu=2.0, timeout=30 * 60, **mounts)
    def prep(bench_files: bool = True) -> dict:
        out = prep_impl(bench_files)
        vol.commit()   # the bench/build containers start from this state
        return out

    # gpu is always overridden per call with .with_options(gpu=...)
    @app.function(gpu="L4", cpu=float(BENCH_CPU),
                  timeout=BENCH_TIMEOUT_S, **mounts)
    def bench_gpu(gpu_requested: str, n_decode: int, impls: list) -> dict:
        res = bench_gpu_impl(gpu_requested, n_decode, impls)
        commit_volume(vol)   # the compile cache (/vol/inductor) for workers
        return res

    # gpu is set per call with .with_options; the timeout is the cost cap
    @app.function(gpu=PROFILE_GPU, cpu=float(BENCH_CPU),
                  timeout=PROFILE_TIMEOUT_S, **mounts)
    def profile_gpu(gpu_requested: str, ts: str) -> dict:
        return profile_gpu_impl(gpu_requested, ts,
                                commit=lambda: commit_volume(vol))

    # gpu, cpu and timeout are always overridden per build with .with_options
    @app.function(gpu="L4", cpu=8.0,
                  timeout=DEFAULT_WORKER_TIMEOUT_H * 3600, **mounts)
    def build_worker(i: int, n: int, batch: int, decode_threads: int,
                     vae_impl: str = "ref", cache_name: str = "") -> dict:
        return build_worker_impl(i, n, batch, decode_threads, vol, vae_impl,
                                 cache_name)

    @app.function(cpu=4.0, timeout=2 * 3600, **mounts)
    def finalize(reset_upload_cache: bool = False) -> dict:
        return finalize_impl(vol, reset_upload_cache=reset_upload_cache)

    # named caches: only the secret each step needs (no HF token), no
    # cpu/memory reservations (rclone + hashing; Modal bills max(req, use))
    @app.function(timeout=STAGE_TIMEOUT_H * 3600, volumes={VOL: vol},
                  secrets=[rclone_secret])
    def stage(cache_name: str, src_drive: str) -> dict:
        return stage_impl(vol, cache_name, src_drive)

    @app.function(timeout=FINALIZE_R2_TIMEOUT_H * 3600, volumes={VOL: vol},
                  secrets=[r2_secret])
    def finalize_r2(cache_name: str, dest_r2: str,
                    precheck: bool = False) -> dict:
        return finalize_r2_impl(vol, cache_name, dest_r2, precheck=precheck)

    def _print_prep(p):
        print(f"prep: HF user {p.get('whoami')!r} (token from "
              f"{p.get('token_env')}), can write {p.get('can_write')} "
              f"({p.get('write_reason')})")

    def bench_local(gpus, n_decode, overhead_min, impls):
        gl = parse_gpus(gpus)
        il = parse_impls(impls)
        if n_decode < DECODE_TIMING_N:
            raise SystemExit(f"--n-decode must be >= {DECODE_TIMING_N}")
        p = prep.remote(True)
        _print_prep(p)
        calls = []
        for g in gl:
            try:
                fn = bench_gpu.with_options(gpu=g, cpu=float(BENCH_CPU))
                calls.append((g, fn.spawn(g, n_decode, il)))
            except Exception as e:  # noqa: BLE001
                calls.append((g, e))
        print(f"spawned {len(calls)} bench containers: {', '.join(gl)} "
              f"(impls {', '.join(il)})")
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
            for e in flatten_results([r]):
                print(f"{g} {e['impl']}: " + (
                    f"ERROR {_err_line(e['error'])}" if e.get("error") else
                    f"{e.get('device_name')} best "
                    f"{e.get('best_img_s') or 0:.2f} img/s"))
            results.append(r)
        rows = [cost_row(e, overhead_min=overhead_min)
                for e in flatten_results(results)]
        choice = select_gpu(rows)
        md = render_bench_report(results, rows, choice, prep=p,
                                 overhead_min=overhead_min)
        print("\n" + "=" * 72 + "\n" + md)
        d = write_bench_results(results, rows, choice, md, prep=p)
        print(f"\nwrote {d}/report.md and bench.json")

    def profile_local(gpu):
        gpu = (gpu or PROFILE_GPU).strip().upper()
        cap, line = profile_cost_cap(gpu)
        print(f"--profile on {gpu}: {line}")
        p = prep.remote(False)   # VAE onto the volume (no bench files)
        _print_prep(p)
        ts = time.strftime("%Y%m%d-%H%M%S")
        d = LOCAL_RESULTS / f"vae_profile_{ts}"
        fn = profile_gpu.with_options(gpu=gpu, cpu=float(BENCH_CPU),
                                      timeout=PROFILE_TIMEOUT_S)
        try:
            r = fn.remote(gpu, ts)
        except Exception as e:  # noqa: BLE001 - e.g. the hard timeout
            print(f"profile_gpu failed: {type(e).__name__}: {e}; fetching "
                  f"the latest checkpoint (saved after every section) from "
                  f"the volume")
            r = {}
            for k, f in (("report_md", "report.md"), ("json", "profile.json"),
                         ("trace_gz", "trace.json.gz")):
                try:
                    b = b"".join(vol.read_file(
                        f"{VOL_PROFILE[len(VOL) + 1:]}/{ts}/{f}"))
                    r[k] = b if k == "trace_gz" else b.decode()
                except Exception as e2:  # noqa: BLE001
                    print(f"  {f}: {type(e2).__name__}: {e2}")
            if not r:
                raise SystemExit("profile FAILED and no checkpoint on the "
                                 "volume") from e
        write_profile_results(r, d)
        print("\n" + "=" * 72 + "\n" + (r.get("report_md") or ""))
        print(f"\nwrote {d}/ (report.md, profile.json"
              + (", trace.json.gz" if r.get("trace_gz") else "") + ")")

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

    def _stage_local(job):
        """Drive -> /vol/src/NAME on a CPU container; must pass before any
        GPU starts. -> stage result."""
        print(f"staging {job['src_remote']} -> {job['src']} (CPU container)")
        st = stage.remote(job["name"], job["src_remote"])
        if not st.get("ok"):
            raise SystemExit("staging FAILED (no GPU started): "
                             + _err_line(st.get("error")))
        print(f"staged {st['n_rows']} rows / {st['n_tars']} tars "
              f"({st['bytes'] / 1e9:.1f} GB) in {st['src']}")
        return st

    def build_local(gpu, containers, batch, cpu, decode_threads,
                    worker_timeout_h, reset_upload_cache, vae_impl, job):
        gpu = gpu.strip().upper()
        if not gpu:
            raise SystemExit("--build needs --gpu (see the bench report)")
        gpu_key(gpu)
        if containers < 1 or batch < 1 or cpu < 1 or decode_threads < 0:
            raise SystemExit("--containers, --batch, --cpu must be >= 1")
        if not 0 < worker_timeout_h <= MAX_WORKER_TIMEOUT_H:
            raise SystemExit(f"--worker-timeout-h must be in (0, "
                             f"{MAX_WORKER_TIMEOUT_H}]")
        parse_impls(vae_impl)
        threads = decode_threads or cpu
        named = bool(job["name"])
        params = {"gpu": gpu, "vae_impl": vae_impl, "containers": containers,
                  "batch": batch,
                  "cpu": cpu, "decode_threads": threads,
                  "worker_timeout_h": worker_timeout_h,
                  "reset_upload_cache": reset_upload_cache}
        n_images = N_IMAGES
        if named:
            params.update(cache_name=job["name"], out=job["out"],
                          src=job["src"], src_drive=job["src_remote"],
                          dest_r2=job["dest"])
            n_images = _stage_local(job)["n_rows"]
            pc = finalize_r2.remote(job["name"], job["dest"], True)
            if not pc.get("ok"):
                raise SystemExit("R2 pre-check FAILED (no GPU started): "
                                 + _err_line(pc.get("error")))
        print(render_estimate(n_images, gpu, cpu, containers, vae_impl,
                              batch))
        p = prep.remote(False)   # VAE once, before N workers would race on it
        _print_prep(p)
        if not named and p.get("can_write") is False:
            raise SystemExit("aborting before any GPU starts: "
                             + WRITE_HELP.format(reason=p["write_reason"]))
        fn = build_worker.with_options(
            gpu=gpu, cpu=float(cpu), timeout=int(worker_timeout_h * 3600))
        calls = [fn.spawn(i, containers, batch, threads, vae_impl,
                          job["name"])
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
        fin = None
        if ok:
            fin = (finalize_r2.remote(job["name"], job["dest"]) if named
                   else finalize.remote(reset_upload_cache=reset_upload_cache))
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
             stage_only: bool = False, profile: bool = False,
             gpu: str = "", containers: int = DEFAULT_CONTAINERS,
             batch: int = 8, cpu: int = 8, decode_threads: int = 0,
             worker_timeout_h: float = DEFAULT_WORKER_TIMEOUT_H,
             reset_upload_cache: bool = False,
             impls: str = DEFAULT_IMPLS, vae_impl: str = "ref",
             cache_name: str = "", src_drive: str = "", dest_r2: str = ""):
        mode = resolve_mode(bench, build, finalize_only, stage_only,
                            profile, reset_upload_cache, cache_name,
                            src_drive, dest_r2)
        if mode == "profile":
            profile_local(gpu)
            return
        job = resolve_job(cache_name, src_drive, dest_r2, mode)
        if job["name"] and reset_upload_cache:
            raise SystemExit("--reset-upload-cache is for the HF upload of "
                             "the v2 cache (a named cache goes to R2)")
        if bench:
            bench_local(gpus, n_decode, overhead_min, impls)
        elif build:
            build_local(gpu, containers, batch, cpu, decode_threads,
                        worker_timeout_h, reset_upload_cache, vae_impl, job)
        elif stage_only:
            _stage_local(job)
        elif job["name"]:
            fin = finalize_r2.remote(job["name"], job["dest"])
            path = _write_build_log(
                {"finalize_only": True, "cache_name": job["name"],
                 "dest_r2": job["dest"]}, {}, [], fin)
            _check_finalize(fin, path)
        else:
            fin = finalize.remote(reset_upload_cache=reset_upload_cache)
            path = _write_build_log(
                {"finalize_only": True,
                 "reset_upload_cache": reset_upload_cache}, {}, [], fin)
            _check_finalize(fin, path)
