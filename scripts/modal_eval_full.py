"""Modal app: side-by-side quality eval of a full fine-tune (EMA export on R2)
against the base Krea 2 RAW checkpoint: same prompt, seed and sampler
settings for both, images back on the laptop with an HTML contact sheet.

    modal run scripts/modal_eval_full.py --only-prompts       # prompt table only (CPU)
    modal run scripts/modal_eval_full.py --limit 2            # cheap smoke run
    modal run scripts/modal_eval_full.py                      # 40 dataset + 10 OOD prompts
    modal run scripts/modal_eval_full.py --steps 28 --cfg 4.5 --seed 0 --gpu H100

    python scripts/modal_eval_full.py --only-prompts --parquet metadata.parquet
                                       # same table from a local parquet, no Modal

Options (local entrypoint):
  --n-dataset N   in-distribution prompts from the dataset's metadata.parquet
                  (default 40)
  --n-ood N       out-of-distribution prompts at 1024x1024 (default 10, max
                  len(OOD_PROMPTS))
  --steps/--cfg   sampler steps and CFG scale (default 28 / 4.5, as
                  scripts/sample.py; mu from the resolution, negative "")
  --seed S        prompt-selection seed; prompt i samples with seed S + i
                  (identical for BASE and FT)
  --only-prompts  run `prepare` for the parquet only, print the table, exit
  --limit N       first N prompts of the table (0 = all)
  --gpu G         Modal GPU of `generate`: H100 (default, 80 GB), H200, B200
                  or B300
  --out DIR       default /home/mj/krea2-ops/eval/run1_<UTC timestamp>

Stages:
  1. `prepare` (CPU): fills the Volume `krea2-eval-cache` idempotently (every
     artifact skipped when already complete, the volume committed after each):
       /vol/base/raw.safetensors     krea/Krea-2-Raw (gated; HF token)
       /vol/qwen3-vl-4b/             Qwen/Qwen3-VL-4B-Instruct (*.json,
                                     *.safetensors, merges.txt, vocab.json)
       /vol/qwen-image/vae/          Qwen/Qwen-Image VAE
       /vol/data/metadata.parquet    ricecake/danbooru-curated-v2-latents
                                     (private; only the parquet)
       /vol/ft/run1/model.safetensors  R2 krea2-fullft/run1/model.safetensors
                                     (boto3 multipart; reused only while the
                                     sidecar model.safetensors.r2.json == the
                                     object's key, ETag, size, LastModified)
     then builds the prompt table from the parquet (build_prompts).
  2. `generate` (GPU, jax[cuda13]==0.11.0 as scripts/modal_bench_b300.py;
     the same stack runs on Hopper): the text tower through the training
     code path (qwen3_text.from_hf_config + load_qwen3_text_params bf16,
     CaptionTemplate(text_rows=256), text_states -> bf16 states + mask,
     exactly what train_full feeds the DiT; the unconditional branch encodes
     "" the same way) for every prompt, then the tower is freed. The DiTs
     run one at a time so one 25.6 GB model fits an 80 GB H100: load BASE
     (bf16), sample all prompts (sorted by resolution) with
     sampling.sample_latents(attn cudnn), free it (device buffers deleted),
     load FT, sample the same prompts with the same seeds (the compiled steps
     are reused: params are arguments, not constants). Latents (float32
     .npy) go to /vol/runs/<run_id>/, not images.
  3. `decode` (CPU, torch + diffusers AutoencoderKLQwenImage, the math of
     scripts/sample.py::decode_latents), .map over chunks of
     DECODE_PROMPTS_PER_CALL prompts: PNG bytes of base, ft and the captioned
     pair.
  4. local: <out>/NN_base.png, NN_ft.png, NN_pair.png, prompts.json
     ({"weights": the evaluated checkpoints (HF revision; R2 key, ETag,
     size, LastModified), "settings", "prompts": [id, source, caption,
     width, height, seed, seconds, ...]}), generate.json (the GPU log) and
     index.html (prompt | base | ft table, relative links; the weights in
     its header).

Prompt set (build_prompts, deterministic for --seed): --n-dataset rows drawn
with np.random.default_rng(seed) among the TOP_BUCKETS most common
(bucket_w, bucket_h) buckets (rows with all three captions non-empty), at
their bucket resolution; the first 3/4 use the long caption + ", drawn by
{artist}" (when the artist is known), the rest alternate medium / short
without the artist. Captions are formatted like BucketDataset.caption (the
field names and the artist suffix are imported from krea2_jax.data). Then
--n-ood hard-coded prompts at 1024x1024 (general ability of the base model).

Secrets: Modal secret `huggingface` (HF_TOKEN, HUGGING_FACE_HUB_TOKEN or
HUGGINGFACE_TOKEN) and `krea2-r2` (R2_ACCOUNT_ID, R2_BUCKET,
R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY); values are never printed.

Importable without the Modal SDK (the app is only defined when `modal`
imports); prompt selection, caption formatting, the generate core
(generate_impl), decode (decode_items), pair images and the output writer
are plain functions (tests/test_modal_eval_full.py).
"""

import argparse
import gc
import html
import io
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    import modal
except ImportError:  # importable (pyflakes, tests) without the Modal SDK
    modal = None

LOCAL_REPO = Path(__file__).resolve().parent.parent
if str(LOCAL_REPO) not in sys.path:
    sys.path.insert(0, str(LOCAL_REPO))
REMOTE_REPO = "/repo"
LOCAL_OUT_ROOT = Path("/home/mj/krea2-ops/eval")
RUN_NAME = "run1"

# --- sources ---
BASE_REPO, BASE_FILE = "krea/Krea-2-Raw", "raw.safetensors"
TEXT_REPO = "Qwen/Qwen3-VL-4B-Instruct"
TEXT_PATTERNS = ["*.json", "*.safetensors", "merges.txt", "vocab.json"]
VAE_REPO = "Qwen/Qwen-Image"
VAE_PATTERNS = ["vae/*", "model_index.json"]
DATA_REPO, PARQUET = "ricecake/danbooru-curated-v2-latents", "metadata.parquet"
R2_KEY = "krea2-fullft/run1/model.safetensors"
R2_ENV = ("R2_ACCOUNT_ID", "R2_BUCKET", "R2_ACCESS_KEY_ID",
          "R2_SECRET_ACCESS_KEY")
TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")

# --- volume layout ---
VOLUME_NAME = "krea2-eval-cache"
VOL = "/vol"
VOL_BASE = f"{VOL}/base"
VOL_TEXT = f"{VOL}/qwen3-vl-4b"
VOL_VAE = f"{VOL}/qwen-image"
VOL_DATA = f"{VOL}/data"
VOL_FT = f"{VOL}/ft/{RUN_NAME}"
VOL_RUNS = f"{VOL}/runs"
VOL_JAX_CACHE = f"{VOL}/jax_cache"   # persistent compile cache across runs
MARKER = ".eval_done.json"           # written after a complete HF download

# --- prompts ---
TOP_BUCKETS = 3
LONG_FRAC = 0.75          # dataset prompts with the long caption (+ artist)
OOD_SIZE = (1024, 1024)   # (width, height)
OOD_PROMPTS = (
    "a photo of a dog on a beach",
    "a city street at night, photo",
    "a bowl of ramen, food photo",
    "portrait photo of an old man",
    "a mountain landscape",
    "an oil painting of a ship in a storm",
    "a red sports car",
    'a poster with the text "HELLO WORLD"',
    "a watercolor painting of a cat",
    "an isometric 3D render of a small house",
)
MODELS = ("base", "ft")

# --- sampling (scripts/sample.py defaults; train_full's text settings) ---
DEFAULT_STEPS = 28
DEFAULT_CFG = 4.5
NEGATIVE_PROMPT = ""
TEXT_ROWS = 256           # = the run's --text-rows (train_full default)
ATTN = "cudnn"            # verified on B300 (scripts/modal_bench_b300.py); Hopper
#                           also has cuDNN flash attention (not run here yet)
# Pad the text to TEXT_ROWS instead of compact_text: one compile per
# resolution (compaction rounds the text length up to 64, which would add a
# compile per (resolution, caption length) pair); outputs are unchanged
# (padded text rows are masked keys, see sampling.compact_text).
COMPACT_TEXT = False

# --- Modal resources / cost ---
DEFAULT_GPU = "H100"
GPU_CHOICES = ("H100", "H200", "B200", "B300")
GENERATE_TIMEOUT_S = 60 * 60
GENERATE_MARGIN_S = 150   # tail kept free for results + volume commit
# run_models' allowances until measured: one 25.6 GB DiT load, and the first
# sample of a resolution incl. its compile
LOAD_ALLOWANCE_S = 300
COLD_ALLOWANCE_S = 240
GENERATE_MEM_GIB = 64     # host RAM: one 25.6 GB checkpoint staged at a time
GENERATE_CPU = 8
PREPARE_TIMEOUT_S = 2 * 3600
DECODE_TIMEOUT_S = 30 * 60
DECODE_PROMPTS_PER_CALL = 4   # 8 images (base + ft) per decode container
DECODE_CPU = 8
# Modal pricing (modal.com/pricing, as in scripts/modal_cache_latents.py,
# fetched 2026-10-13)
GPU_PRICE_USD_H = {"B300": 7.10, "B200": 6.25, "H200": 4.54, "H100": 3.95}
CPU_CORE_USD_H = 0.0472
MEM_GIB_USD_H = 0.00799
# rough sampling cost: seconds per 1 MP image at 28 steps with CFG (2 DiT
# forwards per step, ~6.6 PFLOP) at ~45-50% of the dense bf16 peak; load +
# compile overheads on top
EST_S_PER_MP_IMAGE = {"H100": 14.0, "H200": 13.0, "B200": 9.0, "B300": 8.0}
EST_LOAD_S = 6 * 60       # container start + text tower + two 25.6 GB DiTs
EST_COMPILE_S = 90        # per distinct resolution (shared by both models)

# --- images ---
# Same pins as scripts/modal_bench_b300.py (jax 0.11.0, CUDA 13), plus the
# tokenizer the run used (tokenizers 0.23.3 in run1's run_env.txt).
JAX_VERSION = "0.11.0"
GPU_PIP = [
    f"jax[cuda13]=={JAX_VERSION}",
    f"jaxlib=={JAX_VERSION}",
    "safetensors==0.8.0",
    "ml_dtypes==0.6.0",
    "numpy==2.5.4",
    "tokenizers==0.23.3",
]
GPU_ENV = {
    "JAX_PLATFORMS": "cuda,cpu",   # load_krea2_params stages on the CPU device
    "PYTHONUNBUFFERED": "1",
    "JAX_COMPILATION_CACHE_DIR": VOL_JAX_CACHE,
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.92",
}
# CPU image (prepare + decode): the torch/diffusers pins of
# scripts/modal_cache_latents.py (verified there; the cu132 wheel also runs
# on CPU), boto3 for R2, ml_dtypes for krea2_jax.data.
TORCH_PIN = "torch==2.13.0+cu132"
TORCH_INDEX = "https://download.pytorch.org/whl/cu132"
CPU_PIP = [
    "diffusers==0.40.0", "accelerate==1.15.0", "safetensors==0.8.0",
    "pillow==12.3.0", "pyarrow==26.0.0", "numpy==2.5.1",
    "huggingface_hub[hf_xet]==1.24.0", "hf_xet==1.5.2",
    "boto3==1.43.112", "ml_dtypes==0.6.0",
]
CPU_ENV = {
    "HF_HOME": "/tmp/hf",            # hub caches off the volume
    "HF_XET_CACHE": "/tmp/hf_xet",
    "PYTHONUNBUFFERED": "1",
    "TORCHDYNAMO_DISABLE": "1",
}
R2_TRANSFER = dict(max_concurrency=32, multipart_threshold=64 * 2**20,
                   multipart_chunksize=64 * 2**20)


# --------------------------------------------------------------- prompts
def caption_text(texts, artist, variant, use_artist):
    """One caption exactly as BucketDataset.caption formats it: texts =
    (long, medium, short) strings (None -> ""), variant 0/1/2, plus
    ARTIST_SUFFIX.format(artist) when use_artist and the artist is known."""
    from krea2_jax.data import ARTIST_SUFFIX

    text = texts[int(variant)] or ""
    if use_artist and artist:
        text += ARTIST_SUFFIX.format(artist)
    return text


def read_parquet_rows(path):
    """metadata.parquet -> dict of columns (id int64 array, bucket_w,
    bucket_h, long/medium/short/artist_name lists with null -> ""), rows in
    id order (the parquet's own order is not part of the selection)."""
    import numpy as np
    import pyarrow.parquet as pq

    from krea2_jax.data import CAPTION_FIELDS

    cols = ("id", "bucket_w", "bucket_h") + CAPTION_FIELDS + ("artist_name",)
    table = pq.read_table(path, columns=list(cols))
    ids = table.column("id").to_numpy().astype(np.int64)
    order = np.argsort(ids, kind="stable")
    out = {"id": ids[order]}
    for c in ("bucket_w", "bucket_h"):
        out[c] = table.column(c).to_numpy().astype(np.int64)[order]
    for c in CAPTION_FIELDS + ("artist_name",):
        col = table.column(c).to_pylist()
        out[c] = [col[j] or "" for j in order.tolist()]
    return out


def top_buckets(bucket_w, bucket_h, k=TOP_BUCKETS):
    """The k most common (w, h) buckets, ties by (w, h) (as train_full's
    log line orders them) -> list of ((w, h), count)."""
    import collections

    counts = collections.Counter(zip(map(int, bucket_w), map(int, bucket_h)))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def select_dataset_prompts(rows, n, seed, k=TOP_BUCKETS, long_frac=LONG_FRAC):
    """n dataset prompts from read_parquet_rows output (see the module
    docstring). Returns prompt dicts without id/index/seed."""
    import numpy as np

    from krea2_jax.data import CAPTION_FIELDS

    if n <= 0:
        return []
    top = {wh for wh, _ in top_buckets(rows["bucket_w"], rows["bucket_h"], k)}
    cands = [j for j in range(len(rows["id"]))
             if (int(rows["bucket_w"][j]), int(rows["bucket_h"][j])) in top
             and all(rows[f][j] for f in CAPTION_FIELDS)]
    if len(cands) < n:
        raise ValueError(f"only {len(cands)} rows with all captions in the "
                         f"top {k} buckets, need {n}")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(cands), size=n, replace=False)
    n_long = int(round(n * long_frac))
    out = []
    for r, c in enumerate(picks.tolist()):
        j = cands[c]
        if r < n_long:
            variant, use_artist = 0, True
        else:
            variant, use_artist = 1 + (r - n_long) % 2, False
        artist = rows["artist_name"][j]
        texts = tuple(rows[f][j] for f in CAPTION_FIELDS)
        out.append({
            "source": "dataset",
            "row_id": int(rows["id"][j]),
            "variant": CAPTION_FIELDS[variant],
            "artist": artist if use_artist else "",
            "caption": caption_text(texts, artist, variant, use_artist),
            "width": int(rows["bucket_w"][j]),
            "height": int(rows["bucket_h"][j]),
        })
    return out


def ood_prompts(n):
    if not 0 <= n <= len(OOD_PROMPTS):
        raise ValueError(f"--n-ood must be in 0..{len(OOD_PROMPTS)}, got {n}")
    w, h = OOD_SIZE
    return [{"source": "ood", "row_id": None, "variant": None, "artist": "",
             "caption": c, "width": w, "height": h} for c in OOD_PROMPTS[:n]]


def finalize_prompts(prompts, seed):
    """Number the table: index i, id = zero-padded i (>= 2 digits), seed =
    seed + i."""
    width = max(2, len(str(max(len(prompts) - 1, 0))))
    out = []
    for i, p in enumerate(prompts):
        if p["width"] % 16 or p["height"] % 16:
            raise ValueError(f"prompt {i}: {p['width']}x{p['height']} is not "
                             "a multiple of 16")
        out.append({"id": f"{i:0{width}d}", "index": i, **p,
                    "seed": int(seed) + i})
    return out


def build_prompts(parquet, n_dataset, n_ood, seed):
    """The full prompt table (dataset prompts first, then OOD)."""
    ds = (select_dataset_prompts(read_parquet_rows(parquet), n_dataset, seed)
          if n_dataset > 0 else [])
    return finalize_prompts(ds + ood_prompts(n_ood), seed)


def apply_limit(prompts, limit):
    """First `limit` prompts (0 / None = all)."""
    if limit is None or limit <= 0:
        return list(prompts)
    return list(prompts)[:limit]


def prompt_table(prompts, width=100):
    """Plain-text prompt table."""
    lines = [f"{'id':>3} {'source':<7} {'variant':<7} {'res':>9} {'seed':>5}  "
             "caption"]
    for p in prompts:
        cap = p["caption"].replace("\n", " ")
        if len(cap) > width:
            cap = cap[:width - 3] + "..."
        lines.append(f"{p['id']:>3} {p['source']:<7} {p['variant'] or '-':<7} "
                     f"{p['width']:>4}x{p['height']:<4} {p['seed']:>5}  {cap}")
    return "\n".join(lines)


def generation_order(prompts):
    """Prompts sorted by resolution (h, w), table order within one, so every
    shape compiles once and runs back to back."""
    return sorted(prompts, key=lambda p: (p["height"], p["width"], p["index"]))


def check_gpu(gpu):
    """--gpu -> canonical upper-case name, or ValueError."""
    g = str(gpu).upper()
    if g not in GPU_CHOICES:
        raise ValueError(f"--gpu must be one of {GPU_CHOICES}, got {gpu!r}")
    return g


def estimate(prompts, steps, cfg, gpu=DEFAULT_GPU):
    """Rough generate-stage time/cost: {"minutes", "usd", "images"}."""
    gpu = check_gpu(gpu)
    per_step = (EST_S_PER_MP_IMAGE[gpu] / DEFAULT_STEPS
                * (1.0 if cfg > 0 else 0.5))
    sample_s = sum(len(MODELS) * steps * per_step
                   * p["width"] * p["height"] / 2**20 for p in prompts)
    shapes = {(p["width"], p["height"]) for p in prompts}
    total_s = EST_LOAD_S + EST_COMPILE_S * len(shapes) + sample_s
    usd_h = (GPU_PRICE_USD_H[gpu] + GENERATE_CPU * CPU_CORE_USD_H + GENERATE_MEM_GIB * MEM_GIB_USD_H)
    return {"minutes": total_s / 60, "usd": total_s / 3600 * usd_h,
            "images": len(MODELS) * len(prompts)}


def chunks(items, k):
    return [items[i:i + k] for i in range(0, len(items), k)]


# --------------------------------------------------------------- prepare
def pick_hf_token(env):
    """-> (env var name, token) of the first non-empty HF token variable, or
    (None, None). The token value is never printed."""
    for name in TOKEN_ENV_NAMES:
        if env.get(name):
            return name, env[name]
    return None, None


def marker_ok(dest):
    """True if dest holds a complete download: MARKER lists every file with
    its size, and all of them are present with that size."""
    dest = Path(dest)
    try:
        files = json.loads((dest / MARKER).read_text())["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not files:
        return False
    return all((dest / rel).is_file() and (dest / rel).stat().st_size == size
               for rel, size in files.items())


def write_marker(dest, repo_id, patterns, revision=None):
    """Record every downloaded file (relative path -> size) under dest
    (hub bookkeeping in .cache/ and the marker itself excluded) and the
    repo revision (commit sha) it came from."""
    dest = Path(dest)
    files = {str(p.relative_to(dest)): p.stat().st_size
             for p in sorted(dest.rglob("*"))
             if p.is_file() and p.name != MARKER
             and ".cache" not in p.relative_to(dest).parts}
    if not files:
        raise RuntimeError(f"{repo_id}: nothing downloaded into {dest} "
                           f"(patterns {patterns})")
    (dest / MARKER).write_text(json.dumps(
        {"repo": repo_id, "revision": revision, "patterns": patterns,
         "files": files}, indent=1))
    return files


def hf_fetch(repo_id, dest, patterns, token, repo_type=None, log=print):
    """snapshot_download(allow_patterns) of the repo's current commit into
    dest unless marker_ok(dest)."""
    if marker_ok(dest):
        log(f"{repo_id}: complete in {dest}, skipped")
        return "skipped"
    from huggingface_hub import HfApi, snapshot_download

    t0 = time.time()
    Path(dest).mkdir(parents=True, exist_ok=True)
    sha = HfApi(token=token).repo_info(repo_id, repo_type=repo_type).sha
    snapshot_download(repo_id, repo_type=repo_type, revision=sha,
                      local_dir=str(dest), allow_patterns=list(patterns),
                      token=token)
    files = write_marker(dest, repo_id, patterns, revision=sha)
    gib = sum(files.values()) / 2**30
    log(f"{repo_id}@{sha}: {len(files)} files, {gib:.2f} GiB in "
        f"{time.time() - t0:.0f} s -> {dest}")
    return "downloaded"


def r2_client(env):
    """boto3 S3 client for Cloudflare R2 from the `krea2-r2` secret env."""
    missing = [k for k in R2_ENV if not env.get(k)]
    if missing:
        raise RuntimeError(f"Modal secret krea2-r2 lacks {missing}")
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      max_pool_connections=64,
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"),
    )


def r2_marker_path(dest):
    """Sidecar with the R2 object identity of a complete download."""
    dest = Path(dest)
    return dest.with_name(dest.name + ".r2.json")


def r2_identity(head, key):
    """head_object response -> {"key", "etag", "size", "last_modified"}
    (plain JSON types)."""
    lm = head.get("LastModified")
    return {"key": key, "etag": str(head.get("ETag", "")).strip('"'),
            "size": int(head["ContentLength"]),
            "last_modified": (lm.isoformat() if hasattr(lm, "isoformat")
                              else None if lm is None else str(lm))}


def read_r2_marker(dest):
    """The identity recorded next to dest, or None."""
    try:
        return json.loads(r2_marker_path(dest).read_text())
    except (OSError, ValueError):
        return None


def r2_fetch(client, bucket, key, dest, log=print, transfer_config=None):
    """Download s3://bucket/key to dest (multipart, via dest.part) unless dest
    is a complete download of the CURRENT object: its sidecar marker
    (r2_marker_path, written only after a complete, size-checked download)
    must equal the object's head_object identity (key, ETag, size,
    LastModified) and dest must have that size. A replaced object (even of
    the same size) is downloaded again. s3transfer pins every ranged part
    GET to the ETag of its own HeadObject (IfMatch); head_object is asked
    again after the download, and an identity that changed meanwhile is an
    error (no file, no sidecar). -> (status, identity)."""
    dest = Path(dest)
    head = client.head_object(Bucket=bucket, Key=key)
    ident = r2_identity(head, key)
    size = ident["size"]
    if (dest.is_file() and dest.stat().st_size == size
            and read_r2_marker(dest) == ident):
        log(f"r2:{key}: complete ({size / 2**30:.2f} GiB, etag "
            f"{ident['etag']}), skipped")
        return "skipped", ident
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = r2_marker_path(dest)
    marker.unlink(missing_ok=True)   # dest is not trusted until re-marked
    part = dest.with_name(dest.name + ".part")
    t0 = time.time()
    if transfer_config is None:
        from boto3.s3.transfer import TransferConfig

        transfer_config = TransferConfig(**R2_TRANSFER)
    try:
        client.download_file(bucket, key, str(part), Config=transfer_config)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    after = r2_identity(client.head_object(Bucket=bucket, Key=key), key)
    if after != ident:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"r2:{key}: the object changed during the download "
                           f"(before {ident}, after {after}); run prepare "
                           "again")
    got = part.stat().st_size
    if got != size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"r2:{key}: downloaded {got} bytes, object has "
                           f"{size}")
    os.replace(part, dest)
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text(json.dumps(ident, indent=1))
    os.replace(tmp, marker)
    dt = time.time() - t0
    log(f"r2:{key}: {size / 2**30:.2f} GiB (etag {ident['etag']}) in "
        f"{dt:.0f} s ({size / 2**20 / max(dt, 1e-6):.0f} MiB/s) -> {dest}")
    return "downloaded", ident


def weights_identity(base_dir=VOL_BASE, ft_path=f"{VOL_FT}/model.safetensors"):
    """Which weights an eval uses: {"base": {repo, revision, file, size},
    "ft": the R2 identity of the downloaded object}. Raises if the FT file
    has no identity marker matching its size (prepare did not complete it)."""
    ft = read_r2_marker(ft_path)
    if (ft is None or not Path(ft_path).is_file()
            or Path(ft_path).stat().st_size != ft.get("size")):
        raise RuntimeError(f"{ft_path}: no matching R2 identity marker "
                           f"({r2_marker_path(ft_path).name}); run prepare")
    try:
        m = json.loads((Path(base_dir) / MARKER).read_text())
    except (OSError, ValueError):
        m = {}
    base = {"repo": m.get("repo", BASE_REPO), "revision": m.get("revision"),
            "file": BASE_FILE, "size": m.get("files", {}).get(BASE_FILE)}
    return {"base": base, "ft": ft}


def weights_lines(weights):
    """Human-readable identity lines (index.html info, local log)."""
    b, f = weights["base"], weights["ft"]
    return [f"base: {b['repo']}/{b['file']} @ {b['revision']} "
            f"({b['size']} bytes)",
            f"ft: r2 {f['key']} etag {f['etag']}, {f['size']} bytes, "
            f"last modified {f['last_modified']}"]


def prepare_impl(parquet_only, n_dataset, n_ood, seed, commit=lambda: None,
                 env=None, log=print):
    """Fill the volume (everything, or only the parquet), then build the
    prompt table. -> {"artifacts": {name: status}, "errors": {name: msg},
    "prompts": [...] (None when the parquet is missing)}."""
    env = os.environ if env is None else env
    token_name, token = pick_hf_token(env)
    log(f"HF token from {token_name or 'nowhere (anonymous)'}")
    tasks = [("parquet", lambda: hf_fetch(DATA_REPO, VOL_DATA, [PARQUET],
                                          token, repo_type="dataset", log=log))]
    if not parquet_only:
        tasks += [
            ("text_encoder", lambda: hf_fetch(TEXT_REPO, VOL_TEXT,
                                              TEXT_PATTERNS, token, log=log)),
            ("vae", lambda: hf_fetch(VAE_REPO, VOL_VAE, VAE_PATTERNS, token,
                                     log=log)),
            ("base", lambda: hf_fetch(BASE_REPO, VOL_BASE, [BASE_FILE], token,
                                      log=log)),
            ("ft", lambda: r2_fetch(r2_client(env), env["R2_BUCKET"], R2_KEY,
                                    f"{VOL_FT}/model.safetensors",
                                    log=log)[0]),
        ]
    out = {"artifacts": {}, "errors": {}, "prompts": None}
    for name, fn in tasks:
        try:
            out["artifacts"][name] = fn()
        except Exception as e:  # noqa: BLE001 - report, try the others
            traceback.print_exc()
            out["artifacts"][name] = "FAILED"
            out["errors"][name] = f"{type(e).__name__}: {e}"
        commit()
    parquet = Path(VOL_DATA) / PARQUET
    if parquet.is_file():
        out["prompts"] = build_prompts(parquet, n_dataset, n_ood, seed)
    return out


# --------------------------------------------------------------- generate
def check_text_fit(txt_cfg, template, cfg):
    """train_full's --text-encoder checks: [] when the tower fits the DiT."""
    problems = []
    if txt_cfg.prefix_len != template.prefix_len:
        problems.append(f"prefix_len {txt_cfg.prefix_len} != the tokenizer's "
                        f"template prefix {template.prefix_len}")
    if txt_cfg.hidden_size != cfg.txtdim:
        problems.append(f"hidden_size {txt_cfg.hidden_size} != DiT txtdim "
                        f"{cfg.txtdim}")
    if len(txt_cfg.select_layers) != cfg.txtlayers:
        problems.append(f"{len(txt_cfg.select_layers)} select_layers != DiT "
                        f"txtlayers {cfg.txtlayers}")
    return problems


def free_tree(tree):
    """Delete the device buffers of every jax.Array leaf now (not when the
    last Python reference goes away)."""
    import jax

    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array) and not leaf.is_deleted():
            leaf.delete()


def _device_gib(jax):
    """'X.XX GiB' in use on device 0, or 'n/a' (no memory_stats, e.g. CPU)."""
    try:
        stats = jax.devices()[0].memory_stats() or {}
        return f"{stats['bytes_in_use'] / 2**30:.2f} GiB"
    except Exception:  # noqa: BLE001 - informational only
        return "n/a"


def run_models(order, models, load, sample, free, deadline=None,
               clock=time.time, load_allowance=LOAD_ALLOWANCE_S,
               cold_allowance=COLD_ALLOWANCE_S, log=print):
    """Run the models one after another over `order` within `deadline`.

    load(name, path) -> params; sample(params, name, prompt) -> dict of extra
    row fields; free(params). Durations are measured with `clock`.

    Time estimates (no sample may end after `deadline`):
      cold  first sample of a resolution (includes its compile): the largest
            measured cold time, else `cold_allowance`
      warm  a resolution compiled before (by any model: the models share the
            compiled steps): the largest measured warm time OF THAT
            RESOLUTION, else its measured cold time
      load  the previous model's measured load time, else `load_allowance`
    A model is loaded only if some prompt is left for it and the time left
    covers its load plus its first sample; otherwise it is skipped with a
    note. A sample starts only if it fits AND the later models can still
    load and pair it, with every prompt done so far, each at its
    resolution's warm time (an unmeasured resolution: the largest warm time
    measured so far, 0 before any): a later model only samples the prompts
    every earlier model finished, so the outputs stay paired.
    -> (rows, skipped "<id>_<model>", notes, {name: load seconds})."""
    rows, skipped, notes, load_s = [], [], [], {}
    cold, warm, last_load = {}, {}, None   # (w, h) -> seconds
    allowed = {p["id"] for p in order}

    def left():
        return math.inf if deadline is None else deadline - clock()

    def shape_of(p):
        return (p["width"], p["height"])

    def est(shape):
        """Time of the next sample of this resolution."""
        if shape not in cold:
            return max(cold.values(), default=cold_allowance)
        return warm.get(shape, cold[shape])

    def pair_cost(shape):
        """A later model's (warm) sample of this resolution."""
        return warm.get(shape, max(warm.values(), default=0.0))

    for k, (name, path) in enumerate(models):
        todo = [p for p in order if p["id"] in allowed]
        if not todo:
            notes.append(f"{name}: not loaded (no prompt left to sample)")
            continue
        need = ((last_load if last_load is not None else load_allowance)
                + min(est(shape_of(p)) for p in todo))
        if need > left():
            notes.append(f"{name}: not loaded ({max(left(), 0):.0f} s left < "
                         f"{need:.0f} s for its load + one sample)")
            skipped += [f"{p['id']}_{name}" for p in todo]
            allowed = set()
            continue
        t0 = clock()
        params = load(name, path)
        last_load = load_s[name] = clock() - t0
        later = len(models) - k - 1
        done, done_shapes = set(), []
        for p in todo:
            shape = shape_of(p)
            reserve = later * (last_load + sum(
                pair_cost(s) for s in done_shapes + [shape]))
            if est(shape) + reserve > left():
                skipped.append(f"{p['id']}_{name}")
                continue
            t0 = clock()
            extra = sample(params, name, p)
            dt = clock() - t0
            first = shape not in cold
            if first:
                cold[shape] = dt
            else:
                warm[shape] = max(warm.get(shape, 0.0), dt)
            rows.append({"id": p["id"], "model": name, "seconds": dt,
                         "first_of_shape": first, **extra})
            done.add(p["id"])
            done_shapes.append(shape)
        allowed &= done
        free(params)
        del params
    if skipped:
        notes.append(f"deadline: {len(skipped)} (prompt, model) samples "
                     f"skipped ({skipped})")
    return rows, skipped, notes, load_s


def generate_impl(prompts, models, te_dir, out_dir, steps=DEFAULT_STEPS,
                  guidance=DEFAULT_CFG, *, attn=ATTN, text_rows=TEXT_ROWS,
                  text_overrides=None, model_cfg=None, compact=COMPACT_TEXT,
                  deadline=None, weights=None, log=print):
    """Sample every prompt with every model; latents -> out_dir/<id>_<model>.npy
    (float32, (1, C, H/8, W/8)).

    models: ((name, safetensors path), ...), run one after another: load
    (bf16), sample every prompt, free (device buffers deleted) before the
    next model loads, so only one DiT is resident. Text: the train_full
    path (from_hf_config(te_dir,
    **text_overrides), CaptionTemplate(text_rows), load_qwen3_text_params
    bf16, text_states) -> bf16 (1, text_rows, txtlayers, txtdim) + bool
    mask, as sample.py feeds sample_latents; uncond = NEGATIVE_PROMPT
    through the same path. The text tower is freed before the DiTs load.
    deadline: time.time() by which sampling must end (the caller keeps its
    own tail margin for writing results); scheduling in run_models.
    weights: identity of the evaluated checkpoints (weights_identity),
    copied into the result. Returns a plain-JSON dict: rows (id, model,
    seconds, first_of_shape, finite, latent_std, shape), skipped
    "<id>_<model>", notes, timings, text settings, caption token counts,
    weights, env.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from krea2_jax.checkpoint import load_krea2_params, make_mesh
    from krea2_jax.config import KREA2_CONFIG
    from krea2_jax.qwen3_text import (
        from_hf_config, load_qwen3_text_params, load_tokenizer, text_states,
    )
    from krea2_jax.sampling import sample_latents
    from krea2_jax.text_template import PREFIX, CaptionTemplate

    t_start = time.time()
    cfg = model_cfg or KREA2_CONFIG
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}

    # --- text tower: exactly train_full's config/template path ---
    txt_cfg = from_hf_config(te_dir, **(text_overrides or {}))
    tokenizer = load_tokenizer(te_dir)
    template = CaptionTemplate(tokenizer, text_rows=text_rows)
    problems = check_text_fit(txt_cfg, template, cfg)
    if problems:
        raise RuntimeError(f"text encoder {te_dir} does not fit the model: "
                           + "; ".join(problems))
    t0 = time.time()
    txt_params = load_qwen3_text_params(te_dir, jnp.bfloat16, cfg=txt_cfg)
    timings["text_load_s"] = time.time() - t0
    encode = jax.jit(text_states, static_argnums=1)

    def enc(caption):
        ids, txt_len = template.encode_captions([caption])
        states, mask = encode(txt_params, txt_cfg, ids, txt_len)
        return states.astype(jnp.bfloat16), mask

    t0 = time.time()
    use_cfg = guidance > 0
    un = enc(NEGATIVE_PROMPT) if use_cfg else (None, None)
    conds, cap_tokens = {}, {}
    for p in prompts:
        conds[p["id"]] = enc(p["caption"])
        cap_tokens[p["id"]] = (len(tokenizer.encode(PREFIX + p["caption"]).ids)
                               - template.prefix_len)
    jax.block_until_ready([c[0] for c in conds.values()])
    timings["encode_s"] = time.time() - t0
    free_tree(txt_params)
    txt_params = None   # the tower is not needed again
    gc.collect()
    states_shape = tuple(next(iter(conds.values()))[0].shape) if conds else None
    log(f"text: {len(conds)} captions + uncond {NEGATIVE_PROMPT!r} -> states "
        f"{states_shape} bf16 in {timings['encode_s']:.1f} s (prefix "
        f"{template.prefix_len}, text_rows {text_rows}, select_layers "
        f"{list(txt_cfg.select_layers)}, position_mode "
        f"{txt_cfg.position_mode})")
    long_caps = [k for k, n in cap_tokens.items()
                 if n > template.max_caption_tokens]
    if long_caps:
        log(f"captions truncated to {template.max_caption_tokens} tokens (as "
            f"in training): {long_caps}")

    # --- DiTs (bf16), one at a time (fits an 80 GB H100), mesh as sample.py ---
    mesh = make_mesh()

    def load(name, path):
        params = load_krea2_params(str(path), mesh=mesh, dtype=jnp.bfloat16)
        log(f"loaded {name} ({path}); device memory in use "
            f"{_device_gib(jax)}")
        return params

    def sample(params, name, p):
        txt, mask = conds[p["id"]]
        t0 = time.time()
        lat = sample_latents(
            params, cfg, txt, mask,
            untxt=un[0], untxt_mask=un[1],
            height=p["height"], width=p["width"],
            steps=steps, guidance=guidance, mu=None, seed=p["seed"],
            attn_impl=attn, attn_mesh=mesh if mesh.size > 1 else None,
            compact=compact,
        )
        arr = np.asarray(jax.device_get(lat).astype(jnp.float32))
        np.save(out_dir / f"{p['id']}_{name}.npy", arr)
        finite = bool(np.isfinite(arr).all())
        log(f"[{p['id']}] {name} {p['width']}x{p['height']} seed "
            f"{p['seed']}: {time.time() - t0:.1f} s"
            f"{'' if finite else ' NON-FINITE'}")
        return {"finite": finite,
                "latent_std": float(arr.std()) if finite else None,
                "shape": list(arr.shape)}

    def free(params):
        free_tree(params)
        gc.collect()
        log(f"freed a DiT; device memory in use {_device_gib(jax)}")

    rows, skipped, notes, load_s = run_models(
        generation_order(prompts), models, load, sample, free,
        deadline=deadline, log=log)
    for name, sec in load_s.items():
        timings[f"load_{name}_s"] = sec
    for n in notes:
        log(f"NOTE: {n}")
    timings["total_s"] = time.time() - t_start
    return {
        "rows": rows, "skipped": skipped, "notes": notes, "timings": timings,
        "text": {"prefix_len": template.prefix_len, "text_rows": text_rows,
                 "max_caption_tokens": template.max_caption_tokens,
                 "select_layers": list(txt_cfg.select_layers),
                 "position_mode": txt_cfg.position_mode,
                 "states_shape": list(states_shape) if states_shape else None},
        "caption_tokens": cap_tokens,
        "weights": weights,
        "env": {"jax": jax.__version__, "backend": jax.default_backend(),
                "device": jax.devices()[0].device_kind, "attn": attn,
                "compact_text": compact, "steps": steps, "cfg": guidance},
    }


# --------------------------------------------------------------- decode
def load_vae(vae_dir):
    import torch
    from diffusers import AutoencoderKLQwenImage

    return (AutoencoderKLQwenImage.from_pretrained(vae_dir, subfolder="vae")
            .to("cpu").eval().requires_grad_(False)).to(torch.float32)


def decode_latent(vae, latent):
    """(1, C, h, w) float32 -> (H, W, 3) uint8; the math of
    scripts/sample.py::decode_latents (un-normalize with the VAE's
    latents_mean/std, single-frame decode, [-1, 1] -> [0, 255])."""
    import numpy as np
    import torch

    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    z = torch.from_numpy(np.asarray(latent)).float().unsqueeze(2)
    with torch.no_grad():
        z = z * std + mean
        img = vae.decode(z).sample[:, :, 0]
    img = (img.clamp(-1, 1) * 0.5 + 0.5) * 255.0
    return img.permute(0, 2, 3, 1).byte().cpu().numpy()[0]


def png_bytes(img):
    from PIL import Image

    buf = io.BytesIO()
    (img if isinstance(img, Image.Image) else Image.fromarray(img)).save(
        buf, format="PNG")
    return buf.getvalue()


def _font(size):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without FreeType sizing
        return ImageFont.load_default()


def make_pair(base, ft, label, labels=MODELS, strip=None):
    """base | ft side by side under a text strip (label, then each half's
    name). base/ft: (H, W, 3) uint8 arrays or PIL images (either may be None
    -> a gray placeholder of the other's size). -> PIL image."""
    import numpy as np
    from PIL import Image, ImageDraw

    imgs = [None if x is None else
            (x if isinstance(x, Image.Image) else Image.fromarray(np.asarray(x)))
            for x in (base, ft)]
    ref = next((i for i in imgs if i is not None), None)
    if ref is None:
        raise ValueError("make_pair needs at least one image")
    imgs = [i if i is not None else Image.new("RGB", ref.size, (96, 96, 96))
            for i in imgs]
    w, h = ref.size
    strip = strip or max(24, h // 24)
    out = Image.new("RGB", (2 * w, h + strip), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    font = _font(max(10, int(strip * 0.6)))
    for k, (img, name) in enumerate(zip(imgs, labels)):
        out.paste(img.convert("RGB"), (k * w, strip))
        text = f"{label}  {name}" if k == 0 else name
        draw.text((k * w + 8, strip // 6), text, fill=(0, 0, 0), font=font)
    return out


def decode_items(items, latent_dir, vae, log=print):
    """items: prompt dicts. For each, decode <id>_<model>.npy of every model
    that exists in latent_dir -> [{"id", "png": {model: bytes, "pair":
    bytes}, "missing": [models], "seconds"}]."""
    import numpy as np

    out = []
    for p in items:
        t0 = time.time()
        imgs, missing, pngs = {}, [], {}
        for name in MODELS:
            f = Path(latent_dir) / f"{p['id']}_{name}.npy"
            if not f.is_file():
                missing.append(name)
                continue
            imgs[name] = decode_latent(vae, np.load(f))
            pngs[name] = png_bytes(imgs[name])
        if imgs:
            label = f"{p['id']}  {p['width']}x{p['height']}  seed {p['seed']}"
            pngs["pair"] = png_bytes(make_pair(imgs.get("base"),
                                               imgs.get("ft"), label))
        dt = time.time() - t0
        log(f"[{p['id']}] decoded {sorted(imgs)} in {dt:.1f} s"
            + (f" (missing {missing})" if missing else ""))
        out.append({"id": p["id"], "png": pngs, "missing": missing,
                    "seconds": dt})
    return out


# --------------------------------------------------------------- outputs
def merge_timings(prompts, gen_rows):
    """prompts + per-model seconds from generate's rows -> new dicts with
    "seconds": {model: s}."""
    secs = {}
    for r in gen_rows:
        secs.setdefault(r["id"], {})[r["model"]] = round(r["seconds"], 2)
    return [{**p, "seconds": secs.get(p["id"], {})} for p in prompts]


def render_index(prompts, title, info_lines=()):
    """index.html: one table row per prompt (id + caption + settings | base
    | ft), images by relative link."""
    esc = html.escape
    rows = []
    for p in prompts:
        cells = []
        for name in MODELS:
            f = f"{p['id']}_{name}.png"
            sec = p.get("seconds", {}).get(name)
            cap = f"{name}" + (f" ({sec:.1f} s)" if sec is not None else "")
            if p.get("files", {}).get(name, True):
                cells.append(f'<td><a href="{f}"><img src="{f}" loading="lazy">'
                             f"</a><div class=cap>{esc(cap)}</div></td>")
            else:
                cells.append(f"<td class=missing>{esc(name)}: not "
                             "generated</td>")
        meta = (f"{p['source']}"
                + (f" row {p['row_id']} ({p['variant']})"
                   if p.get("row_id") is not None else "")
                + f" | {p['width']}x{p['height']} | seed {p['seed']}")
        rows.append(
            f"<tr><td class=prompt><b>{esc(p['id'])}</b> "
            f'<a href="{esc(p["id"])}_pair.png">pair</a>'
            f"<div class=meta>{esc(meta)}</div>"
            f"<p>{esc(p['caption'])}</p></td>{''.join(cells)}</tr>")
    info = "".join(f"<li>{esc(x)}</li>" for x in info_lines)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
body {{ font: 14px/1.4 system-ui, sans-serif; margin: 16px; background: #fff;
       color: #111; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 6px; vertical-align: top; }}
td.prompt {{ width: 28%; }}
td img {{ max-width: 100%; max-height: 640px; display: block; }}
.meta, .cap {{ color: #555; font-size: 12px; }}
td.missing {{ color: #a00; }}
</style></head><body>
<h1>{esc(title)}</h1>
<ul>{info}</ul>
<table><tr><th>prompt</th><th>base</th><th>ft</th></tr>
{chr(10).join(rows)}
</table></body></html>
"""


def write_outputs(out_dir, prompts, decoded, gen=None, title="eval",
                  info_lines=()):
    """Write <id>_{base,ft,pair}.png, prompts.json ({"weights", "settings",
    "prompts"}; weights/settings from gen), generate.json (when gen) and
    index.html into out_dir. decoded: decode_items output (any order).
    Returns the list of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {d["id"]: d for d in decoded}
    prompts = merge_timings(prompts, gen["rows"] if gen else [])
    written = []
    for p in prompts:
        d = by_id.get(p["id"], {"png": {}})
        p["files"] = {}
        for name in MODELS + ("pair",):
            data = d["png"].get(name)
            p["files"][name] = data is not None
            if data is None:
                continue
            path = out_dir / f"{p['id']}_{name}.png"
            path.write_bytes(data)
            written.append(path)
    (out_dir / "prompts.json").write_text(json.dumps({
        "weights": gen.get("weights") if gen else None,
        "settings": gen.get("env") if gen else None,
        "prompts": prompts}, indent=1))
    written.append(out_dir / "prompts.json")
    if gen is not None:
        (out_dir / "generate.json").write_text(json.dumps(gen, indent=1))
        written.append(out_dir / "generate.json")
    (out_dir / "index.html").write_text(render_index(prompts, title,
                                                     info_lines))
    written.append(out_dir / "index.html")
    return written


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def print_estimate(prompts, steps, cfg, gpu):
    e = estimate(prompts, steps, cfg, gpu)
    print(f"estimate ({gpu}, rough): {e['images']} images, ~{e['minutes']:.0f} "
          f"min GPU incl. load/compile, ~${e['usd']:.2f} (generate timeout "
          f"{GENERATE_TIMEOUT_S // 60} min); decode on CPU containers")
    return e


# ------------------------------------------------------------------ Modal app
if modal is not None:
    cpu_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(TORCH_PIN, extra_index_url=TORCH_INDEX)
        .pip_install(*CPU_PIP)
        .env(CPU_ENV)
        .add_local_dir(LOCAL_REPO / "krea2_jax", f"{REMOTE_REPO}/krea2_jax",
                       ignore=["**/__pycache__"])
    )
    gpu_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*GPU_PIP)
        .env(GPU_ENV)
        .add_local_dir(LOCAL_REPO / "krea2_jax", f"{REMOTE_REPO}/krea2_jax",
                       ignore=["**/__pycache__"])
    )
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    hf_secret = modal.Secret.from_name("huggingface")
    r2_secret = modal.Secret.from_name("krea2-r2")
    app = modal.App("krea2-eval-full")

    def _remote_path():
        if REMOTE_REPO not in sys.path:
            sys.path.insert(0, REMOTE_REPO)

    @app.function(image=cpu_image, cpu=4.0, memory=16 * 1024,
                  timeout=PREPARE_TIMEOUT_S, volumes={VOL: vol},
                  secrets=[hf_secret, r2_secret])
    def prepare(parquet_only: bool, n_dataset: int, n_ood: int,
                seed: int) -> dict:
        _remote_path()
        return prepare_impl(parquet_only, n_dataset, n_ood, seed,
                            commit=vol.commit)

    # gpu is overridden per call with .with_options(gpu=--gpu)
    @app.function(image=gpu_image, gpu=DEFAULT_GPU, cpu=float(GENERATE_CPU),
                  memory=GENERATE_MEM_GIB * 1024, timeout=GENERATE_TIMEOUT_S,
                  volumes={VOL: vol})
    def generate(prompts: list, steps: int, cfg: float, run_id: str) -> dict:
        _remote_path()
        deadline = time.time() + GENERATE_TIMEOUT_S - GENERATE_MARGIN_S
        weights = weights_identity()   # refuses an FT file without identity
        for line in weights_lines(weights):
            print(line)
        try:
            return generate_impl(
                prompts, (("base", f"{VOL_BASE}/{BASE_FILE}"),
                          ("ft", f"{VOL_FT}/model.safetensors")),
                VOL_TEXT, f"{VOL_RUNS}/{run_id}", steps, cfg,
                deadline=deadline, weights=weights)
        finally:
            vol.commit()   # latents + the persistent compile cache

    @app.function(image=cpu_image, cpu=float(DECODE_CPU),
                  timeout=DECODE_TIMEOUT_S, volumes={VOL: vol})
    def decode(items: list, run_id: str) -> list:
        _remote_path()
        import torch

        torch.set_num_threads(DECODE_CPU)
        vol.reload()
        return decode_items(items, f"{VOL_RUNS}/{run_id}", load_vae(VOL_VAE))

    @app.local_entrypoint()
    def main(n_dataset: int = 40, n_ood: int = len(OOD_PROMPTS),
             steps: int = DEFAULT_STEPS, cfg: float = DEFAULT_CFG,
             seed: int = 0, only_prompts: bool = False, limit: int = 0,
             gpu: str = DEFAULT_GPU, out: str = ""):
        if n_dataset < 0 or steps <= 0 or cfg < 0 or limit < 0:
            raise SystemExit("--n-dataset/--limit must be >= 0, --steps > 0, "
                             "--cfg >= 0")
        ood_prompts(n_ood)   # validates --n-ood
        try:
            gpu = check_gpu(gpu)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        stamp = utc_stamp()
        print(f"prepare: {'parquet only' if only_prompts else 'all artifacts'}"
              f" -> volume {VOLUME_NAME}")
        prep = prepare.remote(only_prompts, n_dataset, n_ood, seed)
        for name, status in prep["artifacts"].items():
            print(f"  {name}: {status}"
                  + (f" ({prep['errors'][name]})" if name in prep["errors"]
                     else ""))
        if prep["prompts"] is None:
            raise SystemExit("no metadata.parquet on the volume: see the "
                             "prepare errors above")
        prompts = apply_limit(prep["prompts"], limit)
        print("\n" + prompt_table(prompts) + "\n")
        if only_prompts:
            return
        if prep["errors"]:
            raise SystemExit(f"prepare failed: {sorted(prep['errors'])}")
        if not prompts:
            raise SystemExit("no prompts")
        print_estimate(prompts, steps, cfg, gpu)
        run_id = f"{RUN_NAME}_{stamp}"
        out_dir = Path(out) if out else LOCAL_OUT_ROOT / run_id
        t0 = time.time()
        gen = generate.with_options(gpu=gpu).remote(prompts, steps, cfg,
                                                    run_id)
        print(f"generate: {len(gen['rows'])} images in "
              f"{(time.time() - t0) / 60:.1f} min (container: "
              + ", ".join(f"{k} {v:.0f}s" for k, v in gen["timings"].items())
              + ")")
        for n in gen["notes"]:
            print(f"  NOTE: {n}")
        bad = [f"{r['id']}_{r['model']}" for r in gen["rows"] if not r["finite"]]
        if bad:
            print(f"  WARNING: non-finite latents: {bad}")
        done = {r["id"] for r in gen["rows"]}
        todo = [p for p in prompts if p["id"] in done]
        t0 = time.time()
        decoded = []
        for res in decode.map(chunks(todo, DECODE_PROMPTS_PER_CALL),
                              kwargs={"run_id": run_id}):
            decoded.extend(res)
        print(f"decode: {len(decoded)} prompts in "
              f"{(time.time() - t0) / 60:.1f} min")
        env = gen["env"]
        for line in weights_lines(gen["weights"]):
            print(f"  {line}")
        info = [f"run {run_id}"] + weights_lines(gen["weights"]) + [
            f"steps {steps}, cfg {cfg}, negative {NEGATIVE_PROMPT!r}, mu from "
            f"resolution, attn {env['attn']}, compact text "
            f"{env['compact_text']}",
            f"jax {env['jax']} on {env['device']}; text_rows "
            f"{gen['text']['text_rows']}, select_layers "
            f"{gen['text']['select_layers']}",
        ] + [f"NOTE: {n}" for n in gen["notes"]]
        paths = write_outputs(out_dir, prompts, decoded, gen,
                              title=f"Krea 2 full FT {RUN_NAME} vs base",
                              info_lines=info)
        print(f"\nwrote {len(paths)} files to {out_dir}")
        print(f"open {out_dir / 'index.html'}")


# ------------------------------------------------------------------ local CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] + " (run with `modal run`; plain "
                    "python only prints the prompt table)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Modal: modal run scripts/modal_eval_full.py [--n-dataset 40 "
               "--n-ood 10 --steps 28 --cfg 4.5 --seed 0 --only-prompts "
               "--limit N --gpu H100 --out DIR]")
    ap.add_argument("--n-dataset", type=int, default=40)
    ap.add_argument("--n-ood", type=int, default=len(OOD_PROMPTS))
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--cfg", type=float, default=DEFAULT_CFG)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-prompts", action="store_true",
                    help="print the prompt table (needs --parquet) and exit")
    ap.add_argument("--parquet", default=None,
                    help="local metadata.parquet of the dataset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", default=DEFAULT_GPU, type=str.upper,
                    choices=GPU_CHOICES,
                    help="GPU of the generate stage (cost estimate)")
    ap.add_argument("--out", default="")
    return ap


def local_main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if not args.only_prompts:
        raise SystemExit("this script runs on Modal: modal run "
                         "scripts/modal_eval_full.py ... (locally only "
                         "--only-prompts --parquet PATH)")
    if not args.parquet:
        raise SystemExit("--only-prompts locally needs --parquet PATH")
    prompts = apply_limit(build_prompts(args.parquet, args.n_dataset,
                                        args.n_ood, args.seed), args.limit)
    print(prompt_table(prompts))
    print_estimate(prompts, args.steps, args.cfg, args.gpu)
    return prompts


if __name__ == "__main__":
    local_main()
