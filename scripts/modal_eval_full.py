"""Modal app: side-by-side quality eval of Krea 2 checkpoints (the HF base
krea/Krea-2-Raw and full fine-tune exports on R2): same prompt, seed and
sampler settings for every model; every (model, prompt) image is generated
ONCE, cached on the Modal Volume and reused by every later comparison; pair
dirs (A vs B) with an HTML contact sheet are assembled on the laptop.

    modal run scripts/modal_eval_full.py --only-prompts       # prompt table only (CPU)
    modal run scripts/modal_eval_full.py --limit 2            # cheap smoke run
    modal run scripts/modal_eval_full.py                      # 40 dataset + 10 OOD prompts,
                                                              # hf-base vs run1 EMA
    modal run scripts/modal_eval_full.py --steps 28 --cfg 4.5 --seed 0 --gpu H100
    modal run scripts/modal_eval_full.py \
        --models base=hf-base,run1raw=krea2-fullft/run1/exports/model_step1513_params.safetensors,run2=krea2-fullft/run2/model.safetensors \
        --pairs run1raw:run2,base:run2     # 3 models, 2 pair dirs; cached entries reused
    modal run scripts/modal_eval_full.py --models ... --pairs ... --cache-only
                                       # what is cached / would be generated (CPU only)
    modal run scripts/modal_eval_full.py \
        --base-key krea2-fullft/run1/model.safetensors \
        --ft-key krea2-fullft/run1/exports/model_step1513_params.safetensors \
        --labels run1-ema,run1-raw            # legacy form: 2 models, 1 pair

    python scripts/modal_eval_full.py --only-prompts --parquet metadata.parquet \
        [--models ... --pairs ...]     # same table from a local parquet, no Modal

Options (local entrypoint):
  --n-dataset N   in-distribution prompts from the dataset's metadata.parquet
                  (default 40)
  --n-ood N       out-of-distribution (anime-style) prompts at 1024x1024
                  (default 10, max len(OOD_PROMPTS))
  --steps/--cfg   sampler steps and CFG scale (default 28 / 4.5, as
                  scripts/sample.py; mu from the resolution, negative "")
  --seed S        prompt-selection seed; prompt i samples with seed S + i
                  (identical for every model)
  --only-prompts  run `prepare` for the parquet only, print the table, exit
  --limit N       first N prompts of the table (0 = all)
  --gpu G         Modal GPU of `generate`: H100 (default, 80 GB), H200, B200
                  or B300
  --models L=KEY,...  the models, by label (letters, digits, . _ -): KEY is
                  an R2 object key (e.g. krea2-fullft/run2/model.safetensors)
                  or the literal `hf-base` (krea/Krea-2-Raw from HF). Labels
                  and keys must be unique.
  --pairs A:B,... pair dirs to assemble, by label: A is written as the "base"
                  side (<id>_base.png), B as the "ft" side (<id>_ft.png).
                  Optional with exactly 2 models (then the first : second).
                  Models in no pair are still generated (pre-cached).
  --base-key/--ft-key/--labels  the legacy form (no --models): models
                  {labels[0]: --base-key or hf-base, labels[1]: --ft-key}
                  (defaults base=hf-base, ft=krea2-fullft/run1/model.safetensors,
                  run1's EMA export), one pair labels[0]:labels[1]
  --out DIR       one pair: that pair's dir; several: their parent dir.
                  Default /home/mj/krea2-ops/eval/<A>_vs_<B>_<UTC timestamp>
  --no-cache      regenerate (and re-decode) every entry of this run,
                  overwriting the cached ones
  --cache-only    prepare's cache listing + the estimate, then exit: nothing
                  is downloaded, generated or decoded (no GPU)

Generated-image cache (Volume `krea2-eval-cache`):
  /vol/gen/<SAMPLER_VERSION>/<model_id>/<entry_hash>.npz   final latents
        (float32 (1, C, H/8, W/8), key "latent")
  /vol/gen/<SAMPLER_VERSION>/<model_id>/<entry_hash>.json  sidecar
        {"hash", "key" (the full key below), "meta" (seconds, finite,
        latent_std, device, jax, created)}; written after the .npz (both
        atomically), so an entry is valid iff its sidecar parses, its key
        hashes to the file name and the .npz exists
  /vol/gen/<SAMPLER_VERSION>/<model_id>/<entry_hash>.png   decoded image
        (written by `decode`; removed whenever the .npz is rewritten)
  local mirror /home/mj/krea2-ops/eval/cache/<same relative path>.png/.json
        (PNG + sidecar copied back by `decode`; a pair dir copies from here)
        and .../cache/<SAMPLER_VERSION>/pairs/<hash>.png (pair images)
  model_id = readable slug + hash of the model identity; entry_hash = the
  first 32 hex of sha256(canonical JSON of the key: sort_keys, compact
  separators, ASCII). The key is {"sampler": {sampler_version, jax, dtype,
  steps, cfg, negative, cfg_batch, attn, compact_text, text_rows,
  text_encoder (repo + revision of the volume's Qwen3-VL), prompt: {caption,
  width, height, seed, mu (resolved from the resolution)}}, "model": the
  checkpoint identity (R2: key, ETag, size, LastModified - the
  _r2_cached_identity fields; HF base: repo, resolved revision (commit),
  file, size from the volume's download marker)}. cfg 0 normalizes negative
  to null and cfg_batch to false. The GPU type is NOT in the key (recorded
  in the sidecar meta; see the determinism caveat).

  SAMPLER_VERSION policy: bump it (a new string) whenever the sampling
  numerics or code path change (sampler, model forward, attention kernels,
  text path, dtypes, pins); old entries then stay under their own directory
  and are never mixed with new ones.

  FAIRNESS RULE (enforced by check_pairable, in `decode` before composing a
  pair image and on the laptop before writing a pair dir): two entries may
  only be paired when their whole "sampler" key part is identical (sampler
  version, steps, cfg, negative, cfg_batch, attn, compact_text, text_rows,
  text encoder, and per prompt caption / size / seed / mu); only the model
  identity may differ. Anything else is refused with UnfairPairError naming
  the differing fields.

  Determinism caveat: cuDNN / XLA kernels are not guaranteed bitwise
  deterministic across containers, GPU types or autotuning results, so a
  regenerated entry may differ from the cached one in the last bits (the
  cache keeps the first one; a pair always compares two entries sampled
  with the same settings). The GPU smoke run checks this on H100.

Stages:
  1. `prepare` (CPU): fills the Volume `krea2-eval-cache` idempotently (every
     artifact skipped when already complete, the volume committed after each):
       /vol/base/raw.safetensors     krea/Krea-2-Raw (gated; HF token; only
                                     when a model is hf-base)
       /vol/qwen3-vl-4b/             Qwen/Qwen3-VL-4B-Instruct (*.json,
                                     *.safetensors, merges.txt, vocab.json)
       /vol/qwen-image/vae/          Qwen/Qwen-Image VAE
       /vol/data/metadata.parquet    ricecake/danbooru-curated-v2-latents
                                     (private; only the parquet)
       /vol/ft/<rest>                R2 objects under krea2-fullft/<rest>,
                                     any other key at /vol/r2/<key>
                                     (r2_cache_path; boto3 multipart; reused
                                     only while the sidecar <file>.r2.json ==
                                     the object's key, ETag, size,
                                     LastModified)
     builds the prompt table from the parquet (build_prompts), then the
     cache plan: each model's identity (R2 head_object; HF base from its
     marker), the key + cache status of every (model, prompt), and downloads
     only the checkpoints of models with a missing entry.
  2. `generate` (GPU, jax[cuda13]==0.11.0 as scripts/modal_bench_b300.py;
     the same stack runs on Hopper), only when some entry is missing: checks
     that every checkpoint on the volume still has the planned identity, the
     text tower through the training code path (qwen3_text.from_hf_config +
     load_qwen3_text_params bf16, CaptionTemplate(text_rows=256),
     text_states -> bf16 states + mask, exactly what train_full feeds the
     DiT; the unconditional branch encodes "" the same way), then the DiTs
     one at a time (one 25.6 GB model fits an 80 GB H100), each loaded at
     most once and only for its missing prompts (sorted by resolution) with
     sampling.sample_latents(attn cudnn, cfg_batch=True: cond + uncond in
     one batch-2 forward), freed before the next (compiled steps reused:
     params are arguments). Entries are written to /vol/gen/ as they finish
     (volume committed after each model).
  3. `decode` (CPU, torch + diffusers AutoencoderKLQwenImage, the math of
     scripts/sample.py::decode_latents), .map over chunks of whole prompts
     with at most DECODE_IMAGES_PER_CALL VAE decodes each: the entry PNGs
     (decoded once, cached on the volume) the laptop lacks, and the
     captioned pair images (after check_pairable).
  4. local, per pair A:B: <dir>/NN_base.png (A), NN_ft.png (B), NN_pair.png,
     prompts.json ({"weights": {"base", "ft"} checkpoint identities,
     "settings": the sampler key part, "labels": {"base": A, "ft": B},
     "pair", "entries" (the cache paths), "prompts": [id, source, caption,
     width, height, seed, mu, seconds, ...]}), generate.json (provenance and
     this run's GPU log) and index.html (prompt | A | B table, relative
     links). The layout is that of the earlier runs, so
     krea2-ops/eval/tools/make_blind.py works unchanged.

Prompt set (build_prompts, deterministic for --seed): --n-dataset rows drawn
with np.random.default_rng(seed) among the TOP_BUCKETS most common
(bucket_w, bucket_h) buckets (rows with all three captions non-empty), at
their bucket resolution; the first 3/4 use the long caption + ", drawn by
{artist}" (when the artist is known), the rest alternate medium / short
without the artist. Captions are formatted like BucketDataset.caption (the
field names and the artist suffix are imported from krea2_jax.data). Then
--n-ood hard-coded anime-style prompts at 1024x1024 (hands, multi-character
scenes, backgrounds, text, costume detail, perspective, action, lighting,
chibi style: general ability of an anime model; photoreal preservation is
not a goal).

Secrets: Modal secret `huggingface` (HF_TOKEN, HUGGING_FACE_HUB_TOKEN or
HUGGINGFACE_TOKEN) and `krea2-r2` (R2_ACCOUNT_ID, R2_BUCKET,
R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY); values are never printed.

Importable without the Modal SDK (the app is only defined when `modal`
imports); the laptop side (local entrypoint: CLI parsing, planning, the
estimate, the local mirror, pair dirs) is stdlib-only, as `modal run`'s own
interpreter has no numpy / PIL. Prompt selection, caption formatting, cache
keys, the fairness rule, the planner, the generate core (generate_impl,
generate_jobs_impl), decode (decode_tasks), pair images and the pair-dir
writer are plain functions (tests/test_modal_eval_full.py,
tests/test_eval_cache.py).
"""

import argparse
import gc
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
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
REMOTE_REPO = "/repo"
# the repo locally; /repo (krea2_jax attached by the images) in a container,
# where this file is mounted on its own
for _repo in (str(LOCAL_REPO), REMOTE_REPO):
    if (_repo not in sys.path
            and os.path.isfile(os.path.join(_repo, "krea2_jax", "r2.py"))):
        sys.path.insert(0, _repo)

from krea2_jax.r2 import (  # noqa: E402 - needs the sys.path above
    head_or_none, r2_client, r2_fetch, r2_identity, r2_marker_path,
    read_r2_marker,
)

LOCAL_OUT_ROOT = Path("/home/mj/krea2-ops/eval")
RUN_NAME = "run1"

# --- sources ---
BASE_REPO, BASE_FILE = "krea/Krea-2-Raw", "raw.safetensors"
TEXT_REPO = "Qwen/Qwen3-VL-4B-Instruct"
TEXT_PATTERNS = ["*.json", "*.safetensors", "merges.txt", "vocab.json"]
VAE_REPO = "Qwen/Qwen-Image"
VAE_PATTERNS = ["vae/*", "model_index.json"]
DATA_REPO, PARQUET = "ricecake/danbooru-curated-v2-latents", "metadata.parquet"
R2_PREFIX = "krea2-fullft"
R2_KEY = f"{R2_PREFIX}/{RUN_NAME}/model.safetensors"   # default --ft-key
TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")

# --- volume layout ---
VOLUME_NAME = "krea2-eval-cache"
VOL = "/vol"
VOL_BASE = f"{VOL}/base"
VOL_TEXT = f"{VOL}/qwen3-vl-4b"
VOL_VAE = f"{VOL}/qwen-image"
VOL_DATA = f"{VOL}/data"
VOL_FT_ROOT = f"{VOL}/ft"            # R2 objects under R2_PREFIX/ (r2_cache_path)
VOL_FT = f"{VOL_FT_ROOT}/{RUN_NAME}"
VOL_R2 = f"{VOL}/r2"                 # any other R2 object, by its key
VOL_RUNS = f"{VOL}/runs"           # latents of the pre-cache runs (unused now)
VOL_JAX_CACHE = f"{VOL}/jax_cache"   # persistent compile cache across runs
MARKER = ".eval_done.json"           # written after a complete HF download

# --- prompts ---
TOP_BUCKETS = 3
LONG_FRAC = 0.75          # dataset prompts with the long caption (+ artist)
OOD_SIZE = (1024, 1024)   # (width, height)
OOD_PROMPTS = (
    "anime illustration of a girl holding a bouquet of flowers with both "
    "hands, close-up on her hands and fingers, soft daylight",
    "three anime characters sitting around a table playing cards, laughing, "
    "one of them pointing at another, warm indoor lighting",
    "a detailed anime background of a crowded shopping street at dusk, many "
    "shop signs, power lines, people walking, no main character",
    'manga-style illustration of a boy shouting, with a speech bubble that '
    'says "WAIT FOR ME!"',
    'anime girl standing in front of a cafe with a sign that reads "OPEN", '
    'smiling, spring afternoon',
    "anime knight in ornate silver armor holding a long spear, full body, "
    "intricate costume details, castle courtyard",
    "extreme low-angle view looking up at an anime girl jumping over the "
    "camera, blue sky, dynamic perspective",
    "two anime characters in a sword fight mid-motion, sparks flying, "
    "dramatic action poses, rooftop at night",
    "anime boy reading a book by a window at night, lit only by a desk lamp, "
    "strong rim light and deep shadows",
    "chibi anime cat girl cooking pancakes in a cozy kitchen, cute style, "
    "pastel colors",
)
MODELS = ("base", "ft")   # the two sides of a pair dir (file names)

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
# cond + uncond in one batch-2 forward per step (sampling.sample_latents
# cfg_batch; valid since COMPACT_TEXT=False pads both to TEXT_ROWS)
CFG_BATCH = True
DTYPE = "bfloat16"        # DiT params / sampler latents

# --- generated-image cache (see the module docstring) ---
# Bump whenever the sampling numerics / code path change (sampler, forward,
# attention kernels, text path, dtypes, pins): entries live under it.
SAMPLER_VERSION = "v2-cfgbatch"
HF_BASE = "hf-base"       # --models KEY of the HF base checkpoint
VOL_GEN = f"{VOL}/gen"
LOCAL_CACHE_ROOT = LOCAL_OUT_ROOT / "cache"
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

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
# VAE decodes per decode container (chunks are whole prompts: every model's
# entry of a prompt + its pair images go to the same container)
DECODE_IMAGES_PER_CALL = 8
DECODE_TASKS_PER_CALL = 24    # PNGs returned per container (~1.5 MB each)
DECODE_CPU = 8
# Modal pricing (modal.com/pricing, as in scripts/modal_cache_latents.py,
# fetched 2026-10-13)
GPU_PRICE_USD_H = {"B300": 7.10, "B200": 6.25, "H200": 4.54, "H100": 3.95}
CPU_CORE_USD_H = 0.0472
MEM_GIB_USD_H = 0.00799
# rough sampling cost: seconds per 1 MP image at 28 steps with CFG.
# H100: measured 11.6 s (run1_20261013-194505, two-call CFG, 100 images);
# the other GPUs scaled from it by their earlier spec-based ratios
# (unmeasured). Load: ~40 s per 25.6 GB DiT measured, text tower ~10 s.
EST_S_PER_MP_IMAGE = {"H100": 12.0, "H200": 11.0, "B200": 7.7, "B300": 6.9}
EST_START_S = 120         # container start + image + text tower
EST_MODEL_LOAD_S = 60     # per DiT that has something to sample
EST_COMPILE_S = 30        # per distinct resolution (shared by the models)

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


def resolved_mu(width, height, cfg=None):
    """The timestep-shift mu sample_latents derives from the resolution
    (cfg.mu_for_seq_len of the image token count; default KREA2_CONFIG)."""
    from krea2_jax.config import KREA2_CONFIG

    cfg = cfg or KREA2_CONFIG
    align = cfg.vae_compression * cfg.patch
    return float(cfg.mu_for_seq_len((int(height) // align)
                                    * (int(width) // align)))


def finalize_prompts(prompts, seed):
    """Number the table: index i, id = zero-padded i (>= 2 digits), seed =
    seed + i, mu = resolved_mu(width, height) (passed to the sampler and
    part of the cache key)."""
    width = max(2, len(str(max(len(prompts) - 1, 0))))
    out = []
    for i, p in enumerate(prompts):
        if p["width"] % 16 or p["height"] % 16:
            raise ValueError(f"prompt {i}: {p['width']}x{p['height']} is not "
                             "a multiple of 16")
        out.append({"id": f"{i:0{width}d}", "index": i, **p,
                    "seed": int(seed) + i,
                    "mu": resolved_mu(p["width"], p["height"])})
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


def parse_labels(text):
    """--labels 'base,ft' -> ("base", "ft"): the display names of the two
    sides (images, index.html, prompts.json), or ValueError. Files keep the
    side names (NN_base.png, NN_ft.png)."""
    labels = tuple(x.strip() for x in str(text).split(","))
    if len(labels) != 2 or not all(labels) or labels[0] == labels[1]:
        raise ValueError(f"--labels must be two different non-empty "
                         f"comma-separated names (base,ft), got {text!r}")
    return labels


def check_sides(ft_key, base_key, labels):
    """Validate --ft-key / --base-key (r2_cache_path) and --labels -> the
    labels tuple, or ValueError."""
    r2_cache_path(ft_key)
    if base_key:
        r2_cache_path(base_key)
    return parse_labels(labels)


def check_label(label):
    """A model label (dir names, --pairs): letters, digits, '.', '_', '-'."""
    if not LABEL_RE.match(str(label)):
        raise ValueError(f"bad model label {label!r}: use letters, digits, "
                         "'.', '_' or '-' (starting with a letter or digit)")
    return label


def parse_models(text):
    """--models 'label=KEY,label=KEY' -> [(label, KEY)]; KEY is an R2 object
    key (validated by r2_cache_path) or HF_BASE. Labels and keys unique."""
    out = []
    for item in str(text).split(","):
        label, sep, key = (x.strip() for x in item.strip().partition("="))
        if not sep or not label or not key:
            raise ValueError(f"--models item {item.strip()!r}: need "
                             "label=KEY (KEY an R2 key or hf-base)")
        check_label(label)
        if key != HF_BASE:
            r2_cache_path(key)
        out.append((label, key))
    labels = [m[0] for m in out]
    keys = [m[1] for m in out]
    for what, xs in (("label", labels), ("KEY", keys)):
        dup = sorted({x for x in xs if xs.count(x) > 1})
        if dup:
            raise ValueError(f"--models: duplicate {what} {dup} (each "
                             "checkpoint once, under one label)")
    return out


def parse_pairs(text, labels):
    """--pairs 'a:b,c:d' -> [(a, b)] over the known labels; a is the "base"
    side of the pair dir, b the "ft" side."""
    out = []
    for item in str(text).split(","):
        a, sep, b = (x.strip() for x in item.strip().partition(":"))
        if not sep or not a or not b:
            raise ValueError(f"--pairs item {item.strip()!r}: need A:B "
                             "(two --models labels)")
        for x in (a, b):
            if x not in labels:
                raise ValueError(f"--pairs {a}:{b}: unknown label {x!r} "
                                 f"(--models has {list(labels)})")
        if a == b:
            raise ValueError(f"--pairs {a}:{b}: a model against itself")
        if (a, b) in out:
            raise ValueError(f"--pairs: {a}:{b} given twice")
        out.append((a, b))
    return out


def resolve_models(models="", pairs="", base_key="", ft_key=R2_KEY,
                   labels=",".join(MODELS)):
    """The CLI -> ([(label, KEY)], [(a, b)]). --models / --pairs, or the
    legacy --base-key / --ft-key / --labels (2 models, 1 pair); mixing them
    is refused. ValueError on any invalid part."""
    if models:
        if base_key or ft_key != R2_KEY or labels != ",".join(MODELS):
            raise ValueError("--models/--pairs replace --base-key/--ft-key/"
                             "--labels; do not mix them")
        ms = parse_models(models)
        names = [m[0] for m in ms]
        if pairs:
            ps = parse_pairs(pairs, names)
        elif len(ms) == 2:
            ps = [(names[0], names[1])]
        else:
            raise ValueError(f"--pairs is required with {len(ms)} --models "
                             "(optional with exactly 2)")
        return ms, ps
    if pairs:
        raise ValueError("--pairs needs --models")
    la, lb = check_sides(ft_key, base_key, labels)
    ms = parse_models(f"{check_label(la)}={base_key or HF_BASE},"
                      f"{check_label(lb)}={ft_key}")
    return ms, [(la, lb)]


def model_path(key):
    """Volume path of a --models KEY's checkpoint."""
    return (Path(VOL_BASE, BASE_FILE) if key == HF_BASE
            else r2_cache_path(key))


# --------------------------------------------------------------- cache keys
class UnfairPairError(ValueError):
    """Two cache entries whose sampler settings differ were to be paired."""


def canonical_json(obj):
    """The JSON text cache keys are hashed over (sorted keys, compact
    separators, ASCII, no NaN)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def short_hash(obj, n=32):
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:n]


def sampler_settings(steps, cfg, text_encoder=None, *, attn=None,
                     compact=None, text_rows=None, cfg_batch=None,
                     negative=None):
    """The run-wide sampler part of every cache key (plain JSON); None
    options = the module constants (ATTN, COMPACT_TEXT, TEXT_ROWS,
    CFG_BATCH, NEGATIVE_PROMPT). cfg 0 normalizes the CFG-only fields
    (negative -> None, cfg_batch -> False). text_encoder: {"repo",
    "revision"} of the volume's text tower (filled by prepare)."""
    attn = ATTN if attn is None else attn
    compact = COMPACT_TEXT if compact is None else compact
    text_rows = TEXT_ROWS if text_rows is None else text_rows
    cfg_batch = CFG_BATCH if cfg_batch is None else cfg_batch
    negative = NEGATIVE_PROMPT if negative is None else negative
    use_cfg = float(cfg) > 0
    return {"sampler_version": SAMPLER_VERSION, "jax": JAX_VERSION,
            "dtype": DTYPE, "steps": int(steps), "cfg": float(cfg),
            "negative": negative if use_cfg else None,
            "cfg_batch": bool(cfg_batch) and use_cfg, "attn": str(attn),
            "compact_text": bool(compact), "text_rows": int(text_rows),
            "text_encoder": text_encoder}


def prompt_part(p):
    """The per-prompt part of a cache key."""
    mu = p.get("mu")
    return {"caption": p["caption"], "width": int(p["width"]),
            "height": int(p["height"]), "seed": int(p["seed"]),
            "mu": float(mu) if mu is not None
            else resolved_mu(p["width"], p["height"])}


def entry_key(settings, prompt, identity):
    """{"sampler": settings + {"prompt": prompt_part}, "model": identity}:
    everything a generated latent depends on. Only "model" may differ
    between the two sides of a pair (check_pairable)."""
    if not identity:
        raise ValueError("entry_key needs the model identity")
    return {"sampler": {**settings, "prompt": prompt_part(prompt)},
            "model": dict(identity)}


def entry_hash(key):
    return short_hash(key, 32)


def model_id(identity):
    """Directory name of a model's entries: a readable slug + 12 hex of the
    identity hash (a replaced R2 object gets a new directory)."""
    if "key" in identity:
        slug = str(identity["key"])
        if slug.startswith(R2_PREFIX + "/"):
            slug = slug[len(R2_PREFIX) + 1:]
        if slug.endswith(".safetensors"):
            slug = slug[:-len(".safetensors")]
    else:
        slug = "hf-" + str(identity.get("repo", "model")).split("/")[-1]
    slug = re.sub(r"[^a-z0-9._-]+", "_", slug.lower()).strip("._-")[:60]
    return f"{slug or 'model'}-{short_hash(identity, 12)}"


def entry_rel(key):
    """<SAMPLER_VERSION>/<model_id>/<entry_hash>: an entry's path (without
    suffix) under a cache root (VOL_GEN, LOCAL_CACHE_ROOT)."""
    ver = str(key["sampler"]["sampler_version"])
    if not LABEL_RE.match(ver):
        raise ValueError(f"bad sampler_version {ver!r}")
    return f"{ver}/{model_id(key['model'])}/{entry_hash(key)}"


def entry_file(root, rel, suffix):
    return Path(root, rel + suffix)


def read_sidecar(root, rel):
    """The sidecar of the entry `rel` under root when valid (parses, holds a
    key that maps to `rel`), else None."""
    try:
        doc = json.loads(entry_file(root, rel, ".json").read_text())
        if entry_rel(doc["key"]) != rel:
            return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return doc


def entry_status(root, key):
    """{"latent": the entry's latents are cached under root (valid sidecar
    with this exact key + the .npz), "png": its decoded PNG too}."""
    rel = entry_rel(key)
    doc = read_sidecar(root, rel)
    ok = (doc is not None and doc["key"] == key
          and entry_file(root, rel, ".npz").is_file())
    return {"latent": ok, "png": ok and entry_file(root, rel, ".png").is_file()}


def _atomic_write(path, data):
    """Write bytes / text via a temp file + os.replace (no torn files on the
    committed volume or the local mirror)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if isinstance(data, str):
        tmp.write_text(data)
    else:
        tmp.write_bytes(data)
    os.replace(tmp, path)


def write_entry(root, key, latent, meta):
    """Store a generated latent as a cache entry: the old sidecar / PNG are
    removed first (the entry is invalid meanwhile), then the .npz, then the
    sidecar {"hash", "key", "meta"}, each atomically. -> rel."""
    import numpy as np

    rel = entry_rel(key)
    for suffix in (".json", ".png"):
        entry_file(root, rel, suffix).unlink(missing_ok=True)
    buf = io.BytesIO()
    np.savez(buf, latent=np.asarray(latent, dtype=np.float32))
    _atomic_write(entry_file(root, rel, ".npz"), buf.getvalue())
    _atomic_write(entry_file(root, rel, ".json"), json.dumps(
        {"hash": entry_hash(key), "key": key, "meta": meta}, indent=1))
    return rel


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def check_pairable(key_a, key_b, where="pair"):
    """THE FAIRNESS RULE: two cache entries may be paired only when their
    "sampler" key parts (sampler version, jax, dtype, steps, cfg, negative,
    cfg_batch, attn, compact_text, text_rows, text encoder, and the prompt's
    caption / size / seed / mu) are identical; only "model" may differ.
    Raises UnfairPairError naming every differing field."""
    try:
        fa, fb = _flat(key_a["sampler"]), _flat(key_b["sampler"])
    except (KeyError, TypeError, AttributeError) as e:
        raise UnfairPairError(f"{where}: not a cache key ({e!r})") from e
    missing = object()
    diff = sorted(k for k in fa.keys() | fb.keys()
                  if fa.get(k, missing) != fb.get(k, missing))
    if diff:
        detail = "; ".join(f"{k}: {fa.get(k, '<absent>')!r} vs "
                           f"{fb.get(k, '<absent>')!r}" for k in diff)
        raise UnfairPairError(
            f"{where}: refusing to pair entries sampled with different "
            f"settings ({detail}); only the model may differ - regenerate "
            "with the same settings")


# --------------------------------------------------------------- planning
def plan_generation(models_info, prompt_ids, use_cache=True):
    """prepare's per-model cache status -> {label: [prompt ids to sample]}
    for the models with something to sample (in --models order); a model
    whose entries are all cached is absent (not loaded). An entry is to be
    sampled when it is unknown (None: identity not resolved), not cached,
    or always with use_cache False."""
    out = {}
    for m in models_info:
        todo = [pid for pid in prompt_ids
                if not use_cache or m["entries"].get(pid) is None
                or not m["entries"][pid]["latent"]]
        if todo:
            out[m["label"]] = todo
    return out


def generate_jobs(models_info, todo):
    """The `generate` work list: one job per model with something to sample
    (label, KEY, volume path, planned identity, prompt ids)."""
    return [{"label": m["label"], "key": m["key"], "path": m["path"],
             "identity": m["identity"], "todo": list(todo[m["label"]])}
            for m in models_info if todo.get(m["label"])]


def cache_report(models_info, prompt_ids, todo, local_ok=None):
    """Human-readable cache listing, one line per model."""
    lines = []
    for m in models_info:
        ents = [m["entries"].get(pid) for pid in prompt_ids]
        vol = sum(1 for e in ents if e and e["latent"])
        png = sum(1 for e in ents if e and e["png"])
        loc = sum(1 for pid in prompt_ids
                  if local_ok and local_ok.get((m["label"], pid)))
        gen = len(todo.get(m["label"], []))
        ident = (f"model_id {m['model_id']}" if m.get("identity")
                 else "identity unknown (not on the volume yet)")
        lines.append(
            f"{m['label']} ({m['key']}; {ident}): {vol}/{len(ents)} latents "
            f"cached, {png} PNGs on the volume, {loc} in the local mirror; "
            + (f"{gen} to generate" if gen else "fully cached, not loaded"))
    return lines


def estimate(prompts, steps, cfg, gpu=DEFAULT_GPU, todo=None):
    """Rough generate-stage time/cost of sampling `todo` ({label: [prompt
    ids]}; default: every prompt with each of the two MODELS). Cached
    entries cost nothing; nothing to sample -> 0 (no GPU container).
    -> {"minutes", "usd", "images", "models"}."""
    gpu = check_gpu(gpu)
    by_id = {p["id"]: p for p in prompts}
    if todo is None:
        todo = {m: [p["id"] for p in prompts] for m in MODELS}
    todo = {m: ids for m, ids in todo.items() if ids}
    items = [by_id[i] for ids in todo.values() for i in ids]
    if not items:
        return {"minutes": 0.0, "usd": 0.0, "images": 0, "models": 0}
    per_step = (EST_S_PER_MP_IMAGE[gpu] / DEFAULT_STEPS
                * (1.0 if cfg > 0 else 0.5))
    sample_s = sum(steps * per_step * p["width"] * p["height"] / 2**20
                   for p in items)
    shapes = {(p["width"], p["height"]) for p in items}
    total_s = (EST_START_S + EST_MODEL_LOAD_S * len(todo)
               + EST_COMPILE_S * len(shapes) + sample_s)
    usd_h = (GPU_PRICE_USD_H[gpu] + GENERATE_CPU * CPU_CORE_USD_H
             + GENERATE_MEM_GIB * MEM_GIB_USD_H)
    return {"minutes": total_s / 60, "usd": total_s / 3600 * usd_h,
            "images": len(items), "models": len(todo)}


def chunks(items, k):
    return [items[i:i + k] for i in range(0, len(items), k)]


def pack_groups(groups, max_decodes=DECODE_IMAGES_PER_CALL,
                max_tasks=DECODE_TASKS_PER_CALL):
    """groups: [(tasks, n_decodes)] (one per prompt) -> task lists, each a
    run of whole groups with at most max_decodes VAE decodes and max_tasks
    tasks (bounds the returned PNG bytes), a single larger group alone."""
    out, cur, n = [], [], 0
    for tasks, k in groups:
        if not tasks:
            continue
        if cur and (n + k > max_decodes or len(cur) + len(tasks) > max_tasks):
            out.append(cur)
            cur, n = [], 0
        cur = cur + list(tasks)
        n += k
    if cur:
        out.append(cur)
    return out


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


def r2_cache_path(key):
    """Volume path caching the R2 object `key`: keys under R2_PREFIX/ map to
    VOL_FT_ROOT/<rest> (the default --ft-key stays at
    /vol/ft/run1/model.safetensors), any other key to VOL_R2/<key>. Raises
    ValueError for a key that is not a plain relative object path."""
    key = str(key)
    parts = key.split("/")
    if (not key or key.startswith("/")
            or any(p in ("", ".", "..") for p in parts)):
        raise ValueError(f"bad R2 key {key!r}: need a relative object key "
                         "(no empty, '.' or '..' components)")
    if parts[0] == R2_PREFIX and len(parts) > 1:
        return Path(VOL_FT_ROOT, *parts[1:])
    return Path(VOL_R2, *parts)


def _r2_cached_identity(path):
    """The R2 identity marker of a cached object, checked against the file's
    size; raises if prepare did not complete it."""
    ident = read_r2_marker(path)
    if (ident is None or not Path(path).is_file()
            or Path(path).stat().st_size != ident.get("size")):
        raise RuntimeError(f"{path}: no matching R2 identity marker "
                           f"({r2_marker_path(path).name}); run prepare")
    return ident


def hf_base_identity(base_dir=None, strict=True):
    """{repo, revision, file, size} of the HF base on the volume (base_dir,
    default VOL_BASE), from its download marker. strict: None unless the
    download is complete (marker_ok); otherwise whatever the marker holds
    (legacy reporting)."""
    base_dir = VOL_BASE if base_dir is None else base_dir
    if strict and not marker_ok(base_dir):
        return None
    try:
        m = json.loads((Path(base_dir) / MARKER).read_text())
    except (OSError, ValueError):
        m = {}
    return {"repo": m.get("repo", BASE_REPO), "revision": m.get("revision"),
            "file": BASE_FILE, "size": m.get("files", {}).get(BASE_FILE)}


def text_encoder_identity(te_dir=None):
    """{repo, revision} of the volume's text tower (te_dir, default
    VOL_TEXT; part of every cache key), or None when it is not completely
    downloaded."""
    te_dir = VOL_TEXT if te_dir is None else te_dir
    if not marker_ok(te_dir):
        return None
    m = json.loads((Path(te_dir) / MARKER).read_text())
    return {"repo": m.get("repo", TEXT_REPO), "revision": m.get("revision")}


def identity_at(path, key):
    """The identity of the checkpoint for --models KEY as it is at `path` on
    the volume: HF_BASE -> hf_base_identity of its directory, an R2 key ->
    _r2_cached_identity (raises when prepare did not complete it)."""
    if key == HF_BASE:
        ident = hf_base_identity(Path(path).parent)
        if ident is None:
            raise RuntimeError(f"{path}: the HF base is not completely "
                               "downloaded; run prepare")
        return ident
    ident = _r2_cached_identity(path)
    if ident.get("key") != key:
        raise RuntimeError(f"{path}: identity marker is for "
                           f"{ident.get('key')!r}, not {key!r}")
    return ident


def weights_identity(base_dir=VOL_BASE, ft_path=f"{VOL_FT}/model.safetensors",
                     base_path=None):
    """Which weights an eval uses: {"base": {repo, revision, file, size}
    (HF), or the R2 identity of base_path when the base side comes from R2
    (--base-key), "ft": the R2 identity of the downloaded object}. Raises if
    an R2 file has no identity marker matching its size (prepare did not
    complete it)."""
    ft = _r2_cached_identity(ft_path)
    if base_path is not None:
        return {"base": _r2_cached_identity(base_path), "ft": ft}
    return {"base": hf_base_identity(base_dir, strict=False), "ft": ft}


def weights_lines(weights, labels=MODELS):
    """Human-readable identity lines (index.html info, local log), one per
    side, named by its label (plus the side when the label differs)."""
    out = []
    for side, label in zip(MODELS, labels):
        w = weights[side]
        name = side if label == side else f"{label} ({side})"
        if not w:
            out.append(f"{name}: identity unknown")
        elif "key" in w:
            out.append(f"{name}: r2 {w['key']} etag {w['etag']}, {w['size']} "
                       f"bytes, last modified {w['last_modified']}")
        else:
            out.append(f"{name}: {w['repo']}/{w['file']} @ {w['revision']} "
                       f"({w['size']} bytes)")
    return out


def prepare_impl(parquet_only, n_dataset, n_ood, seed, commit=lambda: None,
                 env=None, log=print, ft_key=R2_KEY, base_key=None,
                 models=None, settings=None, limit=0, cache_only=False,
                 use_cache=True, gen_root=None):
    """Fill the volume, build the prompt table and (with `settings`) the
    cache plan.

    models: [(label, KEY)] (default: the legacy pair base = base_key or
    HF_BASE, ft = ft_key). parquet_only: only the parquet (and the table).
    Without settings every checkpoint is downloaded (no planning). With
    settings (sampler_settings without text_encoder): the text tower + VAE
    are fetched, then per model its identity (R2: head_object; HF base: its
    marker, downloaded first when absent), the cache key and status of every
    (model, prompt) under gen_root, and the checkpoint is downloaded only
    when some entry is missing (or use_cache is False). cache_only: nothing
    is downloaded (identities that need a download stay unknown).
    -> {"artifacts": {name: status}, "errors": {name: msg}, "prompts": [...]
    (None when the parquet is missing; first `limit` rows), "settings" (with
    text_encoder), "models": [{label, key, path, identity, model_id,
    entries: {prompt id: {rel, hash, key, latent, png} or None}}]}."""
    env = os.environ if env is None else env
    gen_root = VOL_GEN if gen_root is None else gen_root
    if models is None:
        models = [("base", base_key or HF_BASE), ("ft", ft_key)]
    token_name, token = pick_hf_token(env)
    log(f"HF token from {token_name or 'nowhere (anonymous)'}")
    plan = settings is not None and not parquet_only
    fetch = not parquet_only and not cache_only

    def r2_task(key):
        return lambda: r2_fetch(r2_client(env), env["R2_BUCKET"], key,
                                r2_cache_path(key), log=log)[0]

    def hf_base_task():
        return hf_fetch(BASE_REPO, VOL_BASE, [BASE_FILE], token, log=log)

    tasks = [("parquet", lambda: hf_fetch(DATA_REPO, VOL_DATA, [PARQUET],
                                          token, repo_type="dataset", log=log))]
    if fetch:
        tasks += [
            ("text_encoder", lambda: hf_fetch(TEXT_REPO, VOL_TEXT,
                                              TEXT_PATTERNS, token, log=log)),
            ("vae", lambda: hf_fetch(VAE_REPO, VOL_VAE, VAE_PATTERNS, token,
                                     log=log)),
        ]
        if not plan:
            tasks += [(label, hf_base_task if key == HF_BASE
                       else r2_task(key)) for label, key in models]
    out = {"artifacts": {}, "errors": {}, "prompts": None,
           "settings": None, "models": []}

    def attempt(name, fn):
        try:
            res = fn()
            out["artifacts"][name] = res if isinstance(res, str) else "ok"
            return res
        except Exception as e:  # noqa: BLE001 - report, try the others
            traceback.print_exc()
            out["artifacts"][name] = "FAILED"
            out["errors"][name] = f"{type(e).__name__}: {e}"
            return None
        finally:
            commit()

    for name, fn in tasks:
        attempt(name, fn)
    parquet = Path(VOL_DATA) / PARQUET
    if parquet.is_file():
        out["prompts"] = apply_limit(
            build_prompts(parquet, n_dataset, n_ood, seed), limit)
    if not plan or out["prompts"] is None:
        return out

    settings = {**settings, "text_encoder": text_encoder_identity()}
    out["settings"] = settings
    client = []

    def r2_ident(key):
        if not client:
            client.append(r2_client(env))
        head = head_or_none(client[0], env["R2_BUCKET"], key)
        if head is None:
            raise RuntimeError(f"r2:{key}: no such object")
        return r2_identity(head, key)

    for label, key in models:
        if key == HF_BASE:
            ident = hf_base_identity()
            if ident is None and fetch:
                attempt(label, hf_base_task)
                ident = hf_base_identity()
        else:
            ident = attempt(f"{label} (r2 head)", lambda: r2_ident(key))
            out["artifacts"].pop(f"{label} (r2 head)", None)
        entries = {}
        for p in out["prompts"]:
            if ident is None or settings["text_encoder"] is None:
                entries[p["id"]] = None
                continue
            k = entry_key(settings, p, ident)
            entries[p["id"]] = {"rel": entry_rel(k), "hash": entry_hash(k),
                                "key": k, **entry_status(gen_root, k)}
        missing = (not use_cache
                   or any(e is None or not e["latent"]
                          for e in entries.values()))
        if label in out["artifacts"]:
            pass   # HF base downloaded (or failed) above
        elif not missing:
            out["artifacts"][label] = "not needed (all entries cached)"
        elif key == HF_BASE and ident is not None:
            out["artifacts"][label] = "complete on the volume"
        elif not fetch:
            out["artifacts"][label] = "not fetched (--cache-only)"
        elif ident is not None:
            def get(key=key, ident=ident):
                status, got = r2_fetch(r2_client(env), env["R2_BUCKET"], key,
                                       r2_cache_path(key), log=log)
                if got != ident:
                    raise RuntimeError(f"r2:{key}: object changed while "
                                       f"planning ({ident} -> {got}); run "
                                       "again")
                return status
            attempt(label, get)
        out["models"].append({
            "label": label, "key": key, "path": str(model_path(key)),
            "identity": ident, "model_id": model_id(ident) if ident else None,
            "entries": entries})
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

    models: ((name, path), ...) - each model samples every prompt of order -
    or ((name, path, prompt ids), ...) - each only its own prompts (the ones
    without a cache entry). load(name, path) -> params; sample(params, name,
    prompt) -> dict of extra row fields; free(params). Durations are
    measured with `clock`.

    Time estimates (no sample may end after `deadline`):
      cold  first sample of a resolution (includes its compile): the largest
            measured cold time, else `cold_allowance`
      warm  a resolution compiled before (by any model: the models share the
            compiled steps): the largest measured warm time OF THAT
            RESOLUTION, else its measured cold time
      load  the previous model's measured load time, else `load_allowance`
    A model is loaded only if some prompt is left for it and the time left
    covers its load plus its first sample; otherwise it is skipped with a
    note. A sample starts only if it fits AND every later model can still
    load and pair it, with every prompt it shares with this model done so
    far, each at its resolution's warm time (an unmeasured resolution: the
    largest warm time measured so far, 0 before any): a prompt an earlier
    model had to sample but did not is dropped by the later models, so the
    outputs of one run stay paired (a later run fills the rest from the
    cache).
    -> (rows, skipped "<id>_<model>", notes, {name: load seconds})."""
    rows, skipped, notes, load_s = [], [], [], {}
    cold, warm, last_load = {}, {}, None   # (w, h) -> seconds
    every = [p["id"] for p in order]
    models = [(m[0], m[1], set(m[2]) if len(m) > 2 else set(every))
              for m in models]
    dropped = set()

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

    for k, (name, path, ids) in enumerate(models):
        todo = [p for p in order if p["id"] in ids and p["id"] not in dropped]
        if not todo:
            notes.append(f"{name}: not loaded (no prompt left to sample)")
            continue
        need = ((last_load if last_load is not None else load_allowance)
                + min(est(shape_of(p)) for p in todo))
        if need > left():
            notes.append(f"{name}: not loaded ({max(left(), 0):.0f} s left < "
                         f"{need:.0f} s for its load + one sample)")
            skipped += [f"{p['id']}_{name}" for p in todo]
            dropped |= {p["id"] for p in todo}
            continue
        t0 = clock()
        params = load(name, path)
        last_load = load_s[name] = clock() - t0
        later = [m[2] for m in models[k + 1:]]
        done, done_list = set(), []
        for p in todo:
            shape = shape_of(p)
            reserve = 0.0
            for ids_j in later:
                if not ids_j - dropped:
                    continue   # nothing left for that model: never loaded
                reserve += last_load + sum(
                    pair_cost(s) for pid, s in done_list + [(p["id"], shape)]
                    if pid in ids_j)
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
            done_list.append((p["id"], shape))
        dropped |= {p["id"] for p in todo} - done
        free(params)
        del params
    if skipped:
        notes.append(f"deadline: {len(skipped)} (prompt, model) samples "
                     f"skipped ({skipped})")
    return rows, skipped, notes, load_s


def generate_impl(prompts, models, te_dir, out_dir=None, steps=DEFAULT_STEPS,
                  guidance=DEFAULT_CFG, *, attn=ATTN, text_rows=TEXT_ROWS,
                  text_overrides=None, model_cfg=None, compact=COMPACT_TEXT,
                  cfg_batch=CFG_BATCH, negative=NEGATIVE_PROMPT,
                  deadline=None, weights=None, save=None,
                  after_model=lambda: None, log=print):
    """Sample the prompts with the models; each latent (float32, (1, C, H/8,
    W/8)) goes to save(name, prompt, array, meta) (default: np.save to
    out_dir/<id>_<model>.npy).

    models: ((name, safetensors path), ...) - every model samples every
    prompt - or ((name, path, prompt ids), ...) - only those; run one after
    another: load (bf16), sample its prompts, free (device buffers deleted)
    and after_model() before the next model loads, so only one DiT is
    resident. The sampler gets each prompt's seed and mu (p["mu"], None =
    from the resolution) and cfg_batch (cond + uncond in one batch-2
    forward when guidance > 0). Text: the train_full
    path (from_hf_config(te_dir,
    **text_overrides), CaptionTemplate(text_rows), load_qwen3_text_params
    bf16, text_states) -> bf16 (1, text_rows, txtlayers, txtdim) + bool
    mask, as sample.py feeds sample_latents; uncond = `negative` through
    the same path; only the captions of prompts some model samples are
    encoded. The text tower is freed before the DiTs load.
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
    if save is None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def save(name, p, arr, meta):
            np.save(out_dir / f"{p['id']}_{name}.npy", arr)
    wanted = set()
    for m in models:
        wanted |= set(m[2]) if len(m) > 2 else {p["id"] for p in prompts}
    prompts = [p for p in prompts if p["id"] in wanted]
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
    un = enc(negative) if use_cfg else (None, None)
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
    log(f"text: {len(conds)} captions + uncond {negative!r} -> states "
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
            steps=steps, guidance=guidance, mu=p.get("mu"), seed=p["seed"],
            attn_impl=attn, attn_mesh=mesh if mesh.size > 1 else None,
            compact=compact, cfg_batch=cfg_batch,
        )
        arr = np.asarray(jax.device_get(lat).astype(jnp.float32))
        dt = time.time() - t0
        finite = bool(np.isfinite(arr).all())
        meta = {"seconds": round(dt, 3), "finite": finite,
                "latent_std": float(arr.std()) if finite else None,
                "shape": list(arr.shape), "jax": jax.__version__,
                "device": jax.devices()[0].device_kind,
                "created": datetime.now(timezone.utc).isoformat(
                    timespec="seconds")}
        save(name, p, arr, meta)
        log(f"[{p['id']}] {name} {p['width']}x{p['height']} seed "
            f"{p['seed']}: {dt:.1f} s{'' if finite else ' NON-FINITE'}")
        return {"finite": finite, "latent_std": meta["latent_std"],
                "shape": meta["shape"]}

    def free(params):
        free_tree(params)
        gc.collect()
        log(f"freed a DiT; device memory in use {_device_gib(jax)}")
        after_model()

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
                "compact_text": compact, "steps": steps, "cfg": guidance,
                "cfg_batch": bool(cfg_batch) and use_cfg,
                "negative": negative, "sampler_version": SAMPLER_VERSION},
    }


def generate_jobs_impl(prompts, jobs, settings, te_dir, gen_root, *,
                       use_cache=True, deadline=None,
                       commit=lambda: None, log=print, **impl_kw):
    """The `generate` stage over prepare's plan: refuses a checkpoint whose
    identity on the volume is not the planned one (or another text tower
    than settings["text_encoder"]), recomputes every entry key from the
    planned settings, skips entries that are cached meanwhile (use_cache),
    then generate_impl with the sampler settings of the key and each latent
    written as a cache entry under gen_root (committed after each model).
    jobs: generate_jobs output. -> generate_impl's result + "written":
    {label: {prompt id: rel}}, "already_cached": ["<id>_<label>"]."""
    te = text_encoder_identity(te_dir)
    if te != settings.get("text_encoder"):
        raise RuntimeError(f"text encoder on the volume {te} != the planned "
                           f"{settings.get('text_encoder')}; run again")
    by_id = {p["id"]: p for p in prompts}
    models, keys, already = [], {}, []
    for job in jobs:
        ident = identity_at(job["path"], job["key"])
        if ident != job["identity"]:
            raise RuntimeError(
                f"{job['label']}: the checkpoint on the volume ({ident}) is "
                f"not the planned one ({job['identity']}); run again")
        ids = []
        for pid in job["todo"]:
            k = entry_key(settings, by_id[pid], ident)
            if use_cache and entry_status(gen_root, k)["latent"]:
                already.append(f"{pid}_{job['label']}")
                continue
            keys[(job["label"], pid)] = k
            ids.append(pid)
        if ids:
            models.append((job["label"], job["path"], ids))
    for name, path, ids in models:
        log(f"{name}: {len(ids)} entries to sample ({path})")
    if already:
        log(f"cached meanwhile, not sampled again: {already}")
    written = {}

    def save(name, p, arr, meta):
        rel = write_entry(gen_root, keys[(name, p["id"])], arr, meta)
        written.setdefault(name, {})[p["id"]] = rel

    if not models:
        return {"rows": [], "skipped": [], "notes": ["nothing to sample"],
                "timings": {}, "written": {}, "already_cached": already,
                "weights": {j["label"]: j["identity"] for j in jobs}}
    res = generate_impl(
        prompts, models, te_dir, None, settings["steps"], settings["cfg"],
        attn=settings["attn"], text_rows=settings["text_rows"],
        compact=settings["compact_text"], cfg_batch=settings["cfg_batch"],
        negative=(settings["negative"] if settings["negative"] is not None
                  else NEGATIVE_PROMPT),
        deadline=deadline, save=save, after_model=commit,
        weights={j["label"]: j["identity"] for j in jobs}, log=log,
        **impl_kw)
    res["written"] = written
    res["already_cached"] = already
    return res


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


def decode_tasks(tasks, root, vae_loader, log=print):
    """The `decode` stage over cache entries under root (VOL_GEN).

    tasks: {"kind": "entry", "rel", "key"} -> the entry's PNG bytes + its
    sidecar (for the local mirror), or {"kind": "pair", "name", "a": {rel,
    key}, "b": {rel, key}, "label", "labels": [A, B]} -> the captioned pair
    image A | B, after check_pairable on the two sidecar keys. Every entry
    must be valid with exactly the requested key. An entry's PNG is decoded
    from its latents once (vae_loader() called lazily, at most once) and
    cached next to them (<rel>.png); later calls read it.
    -> [{"kind", "rel" | "name", "png": bytes, "sidecar" (entries),
    "decoded": bool (entries)}] in task order."""
    import numpy as np
    from PIL import Image

    vae = []
    cache = {}

    def ensure(e):
        rel = e["rel"]
        if rel in cache:
            return cache[rel]
        doc = read_sidecar(root, rel)
        npz = entry_file(root, rel, ".npz")
        if doc is None or doc["key"] != e["key"] or not npz.is_file():
            raise RuntimeError(f"{rel}: no valid cache entry with the "
                               "requested key")
        png = entry_file(root, rel, ".png")
        decoded = not png.is_file()
        t0 = time.time()
        if decoded:
            if not vae:
                vae.append(vae_loader())
            with np.load(npz) as z:
                lat = z["latent"]
            data = png_bytes(decode_latent(vae[0], lat))
            _atomic_write(png, data)
        else:
            data = png.read_bytes()
        log(f"{rel}: {'decoded' if decoded else 'cached PNG'} in "
            f"{time.time() - t0:.1f} s")
        cache[rel] = (doc, data, decoded)
        return cache[rel]

    out = []
    for t in tasks:
        if t["kind"] == "entry":
            doc, data, decoded = ensure(t)
            out.append({"kind": "entry", "rel": t["rel"], "png": data,
                        "sidecar": doc, "decoded": decoded})
        elif t["kind"] == "pair":
            da, pa, _ = ensure(t["a"])
            db, pb, _ = ensure(t["b"])
            check_pairable(da["key"], db["key"], where=t["name"])
            img = make_pair(Image.open(io.BytesIO(pa)),
                            Image.open(io.BytesIO(pb)), t["label"],
                            labels=tuple(t["labels"]))
            out.append({"kind": "pair", "name": t["name"],
                        "rel": t.get("rel"), "png": png_bytes(img)})
        else:
            raise ValueError(f"unknown decode task kind {t.get('kind')!r}")
    return out


# --------------------------------------------------------------- outputs
def merge_timings(prompts, gen_rows):
    """prompts + per-model seconds from generate's rows -> new dicts with
    "seconds": {model: s}."""
    secs = {}
    for r in gen_rows:
        if r.get("seconds") is not None:
            secs.setdefault(r["id"], {})[r["model"]] = round(r["seconds"], 2)
    return [{**p, "seconds": secs.get(p["id"], {})} for p in prompts]


def render_index(prompts, title, info_lines=(), labels=MODELS):
    """index.html: one table row per prompt (id + caption + settings | base
    | ft, headed and captioned by `labels`), images by relative link."""
    esc = html.escape
    rows = []
    for p in prompts:
        cells = []
        for name, label in zip(MODELS, labels):
            f = f"{p['id']}_{name}.png"
            sec = p.get("seconds", {}).get(name)
            cap = f"{label}" + (f" ({sec:.1f} s)" if sec is not None else "")
            if p.get("files", {}).get(name, True):
                cells.append(f'<td><a href="{f}"><img src="{f}" loading="lazy">'
                             f"</a><div class=cap>{esc(cap)}</div></td>")
            else:
                cells.append(f"<td class=missing>{esc(label)}: not "
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
    head = "".join(f"<th>{esc(x)}</th>" for x in labels)
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
<table><tr><th>prompt</th>{head}</tr>
{chr(10).join(rows)}
</table></body></html>
"""


def write_outputs(out_dir, prompts, decoded, gen=None, title="eval",
                  info_lines=(), labels=MODELS, extra=None):
    """Write <id>_{base,ft,pair}.png, prompts.json ({"weights", "settings",
    "labels": {side: display name}, **extra, "prompts"}; weights/settings
    from gen), generate.json (when gen) and index.html into out_dir.
    decoded: [{"id", "png": {"base" | "ft" | "pair": PNG bytes or a Path to
    copy}}] (any order). Returns the list of written paths."""
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
            if isinstance(data, (bytes, bytearray)):
                path.write_bytes(data)
            else:
                shutil.copyfile(data, path)
            written.append(path)
    (out_dir / "prompts.json").write_text(json.dumps({
        "weights": gen.get("weights") if gen else None,
        "settings": gen.get("env") if gen else None,
        "labels": dict(zip(MODELS, labels)),
        **(extra or {}),
        "prompts": prompts}, indent=1))
    written.append(out_dir / "prompts.json")
    if gen is not None:
        (out_dir / "generate.json").write_text(json.dumps(gen, indent=1))
        written.append(out_dir / "generate.json")
    (out_dir / "index.html").write_text(render_index(prompts, title,
                                                     info_lines, labels))
    written.append(out_dir / "index.html")
    return written


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def print_estimate(prompts, steps, cfg, gpu, todo=None, log=print):
    e = estimate(prompts, steps, cfg, gpu, todo)
    if not e["images"]:
        log("estimate: every entry is cached, no GPU container (~$0.00)")
        return e
    log(f"estimate ({gpu}, rough): {e['images']} images from {e['models']} "
          f"model(s), ~{e['minutes']:.0f} min GPU incl. load/compile, "
          f"~${e['usd']:.2f} (generate timeout {GENERATE_TIMEOUT_S // 60} "
          "min; cached entries cost nothing); decode on CPU containers")
    return e


# --------------------------------------------------------------- local mirror
def local_entry_ok(local_root, entry):
    """The local mirror holds this entry's PNG with a sidecar of exactly its
    key."""
    if not entry:
        return False
    doc = read_sidecar(local_root, entry["rel"])
    return (doc is not None and doc["key"] == entry["key"]
            and entry_file(local_root, entry["rel"], ".png").is_file())


def store_local_entry(local_root, rel, png, sidecar):
    """Mirror an entry's PNG + sidecar (sidecar last: it validates the
    PNG)."""
    entry_file(local_root, rel, ".json").unlink(missing_ok=True)
    _atomic_write(entry_file(local_root, rel, ".png"), png)
    _atomic_write(entry_file(local_root, rel, ".json"),
                  json.dumps(sidecar, indent=1))


def pair_label(p):
    """The caption strip of a prompt's pair image."""
    return f"{p['id']}  {p['width']}x{p['height']}  seed {p['seed']}"


def pair_image_rel(entry_a, entry_b, label, labels):
    """Local mirror path (no suffix) of a pair image: it depends on both
    entries, the caption strip and the labels."""
    return (f"{SAMPLER_VERSION}/pairs/"
            + short_hash([entry_a["rel"], entry_b["rel"], label,
                          list(labels)], 32))


def pair_dir_name(a, b, stamp):
    return f"{a}_vs_{b}_{stamp}"


def assemble_pair_dir(out_dir, prompts, pair, models_info, settings,
                      local_root=LOCAL_CACHE_ROOT, gen=None, run_id="",
                      title=None):
    """Write the pair dir A:B (A = "base" side, B = "ft" side) from the local
    mirror in the layout of the earlier runs (<id>_base.png, <id>_ft.png,
    <id>_pair.png, prompts.json with "labels", generate.json, index.html),
    so make_blind.py works unchanged. Each side's PNG is used only when the
    mirror's sidecar has exactly the planned key; when both sides are
    present, check_pairable(key A, key B) must pass (UnfairPairError
    otherwise, before anything is written). A prompt missing on a side is
    shown as "not generated". -> written paths."""
    a, b = pair
    info = {m["label"]: m for m in models_info}
    ma, mb = info[a], info[b]
    decoded, rows, entries, devices = [], [], {}, set()
    for p in prompts:
        pid = p["id"]
        files, keys = {}, {}
        for side, m in (("base", ma), ("ft", mb)):
            e = m["entries"].get(pid)
            if not local_entry_ok(local_root, e):
                continue
            doc = read_sidecar(local_root, e["rel"])
            files[side] = entry_file(local_root, e["rel"], ".png")
            keys[side] = doc["key"]
            meta = doc.get("meta") or {}
            rows.append({"id": pid, "model": side,
                         "seconds": meta.get("seconds")})
            if meta.get("device"):
                devices.add(meta["device"])
            entries.setdefault(pid, {})[side] = e["rel"]
        if len(keys) == 2:
            check_pairable(keys["base"], keys["ft"],
                           where=f"pair {a}:{b}, prompt {pid}")
            pimg = entry_file(local_root, pair_image_rel(
                ma["entries"][pid], mb["entries"][pid], pair_label(p),
                (a, b)), ".png")
            if pimg.is_file():
                files["pair"] = pimg
        decoded.append({"id": pid, "png": files})
    weights = {"base": ma["identity"], "ft": mb["identity"]}
    n_both = sum(1 for d in decoded if "base" in d["png"] and "ft" in d["png"])
    gen_doc = {"weights": weights, "env": settings, "rows": rows,
               "pair": [a, b], "run_id": run_id,
               "devices": sorted(devices), "generate": gen}
    info_lines = [f"run {run_id}" if run_id else "run (cache)"] + \
        weights_lines(weights, (a, b)) + [
        f"steps {settings['steps']}, cfg {settings['cfg']}, negative "
        f"{settings['negative']!r}, mu from resolution, attn "
        f"{settings['attn']}, compact text {settings['compact_text']}, "
        f"cfg_batch {settings['cfg_batch']}, text_rows "
        f"{settings['text_rows']}",
        f"sampler {settings['sampler_version']}, jax {settings['jax']} "
        f"{settings['dtype']}; text encoder "
        + (f"{settings['text_encoder']['repo']} @ "
           f"{settings['text_encoder']['revision']}"
           if settings.get("text_encoder") else "unknown"),
        f"{n_both}/{len(prompts)} prompts with both sides; entries sampled "
        f"on {sorted(devices) or ['?']} (from the cache {LOCAL_CACHE_ROOT.name}"
        f"/{SAMPLER_VERSION})",
    ] + [f"NOTE: {n}" for n in (gen or {}).get("notes", [])]
    return write_outputs(
        out_dir, prompts, decoded, gen_doc,
        title=title or f"Krea 2: {b} vs {a}", info_lines=info_lines,
        labels=(a, b), extra={"pair": [a, b], "entries": entries})


# ------------------------------------------------------------------ orchestration
def decode_plan(prompts, pairs, models_info, local_root, use_cache=True,
                regenerated=()):
    """The decode work of a run: per prompt, an "entry" task for every entry
    a pair needs whose PNG the local mirror lacks (or that was regenerated
    this run, or every one with use_cache False), and a "pair" task for
    every pair image the mirror lacks; packed into chunks of whole prompts
    (pack_groups: VAE decodes = referenced entries without a volume PNG).
    Entries without latents (not generated) are left out."""
    info = {m["label"]: m for m in models_info}
    regenerated = set(regenerated)
    groups = []
    for p in prompts:
        pid = p["id"]
        tasks, refs = [], {}
        fetched = set()
        for a, b in pairs:
            ea, eb = info[a]["entries"].get(pid), info[b]["entries"].get(pid)
            for label, e in ((a, ea), (b, eb)):
                if not e or not e["latent"] or e["rel"] in fetched:
                    continue
                if (use_cache and (label, pid) not in regenerated
                        and local_entry_ok(local_root, e)):
                    continue
                tasks.append({"kind": "entry", "rel": e["rel"],
                              "key": e["key"]})
                fetched.add(e["rel"])
                refs[e["rel"]] = e
            if not (ea and eb and ea["latent"] and eb["latent"]):
                continue
            label = pair_label(p)
            prel = pair_image_rel(ea, eb, label, (a, b))
            if (use_cache and (a, pid) not in regenerated
                    and (b, pid) not in regenerated
                    and entry_file(local_root, prel, ".png").is_file()):
                continue
            tasks.append({"kind": "pair", "name": f"pair {a}:{b}, prompt {pid}",
                          "rel": prel, "label": label, "labels": [a, b],
                          "a": {"rel": ea["rel"], "key": ea["key"]},
                          "b": {"rel": eb["rel"], "key": eb["key"]}})
            refs[ea["rel"]], refs[eb["rel"]] = ea, eb
        n_dec = sum(1 for e in refs.values() if not e["png"])
        groups.append((tasks, n_dec))
    return pack_groups(groups)


def run_eval(*, prepare, generate, decode_map, models, pairs, n_dataset=40,
             n_ood=len(OOD_PROMPTS), steps=DEFAULT_STEPS, cfg=DEFAULT_CFG,
             seed=0, only_prompts=False, limit=0, gpu=DEFAULT_GPU, out="",
             use_cache=True, cache_only=False, local_root=LOCAL_CACHE_ROOT,
             out_root=LOCAL_OUT_ROOT, stamp=None, log=print):
    """The local entrypoint's flow (stdlib only; the stages are callables:
    prepare(**kw) -> prepare_impl's dict, generate(prompts, jobs, settings,
    use_cache) -> generate_jobs_impl's dict, decode_map(chunks) -> iterable
    of decode_tasks results). prepare + plan -> (unless --cache-only)
    generate the missing entries -> decode what the local mirror lacks ->
    one pair dir per pair. -> {"prompts", "todo", "gen", "dirs"}."""
    stamp = stamp or utc_stamp()
    for label, key in models:
        src = f"hf {BASE_REPO}" if key == HF_BASE else f"r2 {key}"
        log(f"{label}: {src} -> {model_path(key)}")
    log("pairs: " + ", ".join(f"{a} (base side) vs {b} (ft side)"
                              for a, b in pairs))
    settings = sampler_settings(steps, cfg)
    log(f"prepare: {'parquet only' if only_prompts else 'cache plan only' if cache_only else 'all artifacts'}"
        f" -> volume {VOLUME_NAME}")
    prep = prepare(parquet_only=only_prompts, n_dataset=n_dataset,
                   n_ood=n_ood, seed=seed, models=[list(m) for m in models],
                   settings=None if only_prompts else settings, limit=limit,
                   cache_only=cache_only, use_cache=use_cache)
    for name, status in prep["artifacts"].items():
        log(f"  {name}: {status}")
    for name, err in prep["errors"].items():
        log(f"  ERROR {name}: {err}")
    if prep["prompts"] is None:
        raise SystemExit("no metadata.parquet on the volume: see the "
                         "prepare errors above")
    prompts = prep["prompts"]
    log("\n" + prompt_table(prompts) + "\n")
    result = {"prompts": prompts, "todo": None, "gen": None, "dirs": []}
    if only_prompts:
        return result
    if not prompts:
        raise SystemExit("no prompts")
    if prep["errors"] and not cache_only:
        raise SystemExit(f"prepare failed: {sorted(prep['errors'])}")
    settings, info = prep["settings"], prep["models"]
    ids = [p["id"] for p in prompts]
    todo = plan_generation(info, ids, use_cache)
    result["todo"] = todo
    local_ok = {(m["label"], pid): use_cache and local_entry_ok(
        local_root, m["entries"].get(pid)) for m in info for pid in ids}
    log(f"cache ({VOLUME_NAME}:{VOL_GEN}/{SAMPLER_VERSION}, local "
        f"{local_root}/{SAMPLER_VERSION}"
        + ("" if use_cache else "; --no-cache: everything regenerated")
        + "):")
    for line in cache_report(info, ids, todo, local_ok):
        log(f"  {line}")
    print_estimate(prompts, steps, cfg, gpu, todo, log=log)
    if cache_only:
        return result
    unknown = [m["label"] for m in info if not m["identity"]]
    if unknown or settings.get("text_encoder") is None:
        raise SystemExit(f"identity unknown after prepare: {unknown or 'text encoder'}")
    regenerated = set()
    if todo:
        t0 = time.time()
        gen = generate(prompts, generate_jobs(info, todo), settings,
                       use_cache)
        log(f"generate: {len(gen['rows'])} images in "
            f"{(time.time() - t0) / 60:.1f} min (container: "
            + ", ".join(f"{k} {v:.0f}s" for k, v in gen["timings"].items())
            + ")")
        for n in gen["notes"]:
            log(f"  NOTE: {n}")
        bad = [f"{r['id']}_{r['model']}" for r in gen["rows"]
               if not r.get("finite", True)]
        if bad:
            log(f"  WARNING: non-finite latents: {bad}")
        by_label = {m["label"]: m for m in info}
        for label, done in gen["written"].items():
            for pid, rel in done.items():
                e = by_label[label]["entries"][pid]
                if e["rel"] != rel:
                    raise SystemExit(f"generate wrote {rel} for {label} "
                                     f"{pid}, planned {e['rel']}")
                e["latent"], e["png"] = True, False
                regenerated.add((label, pid))
        for tag in gen.get("already_cached", []):
            pid, _, label = tag.partition("_")
            by_label[label]["entries"][pid]["latent"] = True
        result["gen"] = gen
    chunks_ = decode_plan(prompts, pairs, info, local_root, use_cache,
                          regenerated)
    if chunks_:
        t0 = time.time()
        n_dec = n_png = 0
        for res in decode_map(chunks_):
            for r in res:
                if r["kind"] == "entry":
                    store_local_entry(local_root, r["rel"], r["png"],
                                      r["sidecar"])
                    n_dec += bool(r["decoded"])
                    n_png += 1
                else:
                    _atomic_write(entry_file(local_root, r["rel"], ".png"),
                                  r["png"])
        log(f"decode: {len(chunks_)} container(s), {n_png} entry PNGs "
            f"({n_dec} decoded now), pair images, in "
            f"{(time.time() - t0) / 60:.1f} min")
    gen_summary = None if result["gen"] is None else {
        k: result["gen"].get(k) for k in ("notes", "timings", "skipped",
                                          "env", "text", "written",
                                          "already_cached")}
    for a, b in pairs:
        if out:
            d = (Path(out) if len(pairs) == 1
                 else Path(out) / pair_dir_name(a, b, stamp))
        else:
            d = Path(out_root) / pair_dir_name(a, b, stamp)
        paths = assemble_pair_dir(d, prompts, (a, b), info, settings,
                                  local_root, gen=gen_summary,
                                  run_id=f"eval_{stamp}")
        log(f"\n{a} vs {b}: wrote {len(paths)} files to {d}")
        log(f"open {d / 'index.html'}")
        result["dirs"].append(d)
    return result


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
    def prepare(parquet_only: bool, n_dataset: int, n_ood: int, seed: int,
                models: list, settings=None, limit: int = 0,
                cache_only: bool = False, use_cache: bool = True) -> dict:
        _remote_path()
        return prepare_impl(parquet_only, n_dataset, n_ood, seed,
                            commit=vol.commit,
                            models=[tuple(m) for m in models],
                            settings=settings, limit=limit,
                            cache_only=cache_only, use_cache=use_cache)

    # gpu is overridden per call with .with_options(gpu=--gpu)
    @app.function(image=gpu_image, gpu=DEFAULT_GPU, cpu=float(GENERATE_CPU),
                  memory=GENERATE_MEM_GIB * 1024, timeout=GENERATE_TIMEOUT_S,
                  volumes={VOL: vol})
    def generate(prompts: list, jobs: list, settings: dict,
                 use_cache: bool = True) -> dict:
        _remote_path()
        deadline = time.time() + GENERATE_TIMEOUT_S - GENERATE_MARGIN_S
        for j in jobs:
            print(f"{j['label']}: {j['key']} -> {j['path']}, "
                  f"{len(j['todo'])} entries planned")
        try:
            return generate_jobs_impl(prompts, jobs, settings, VOL_TEXT,
                                      VOL_GEN, use_cache=use_cache,
                                      deadline=deadline, commit=vol.commit)
        finally:
            vol.commit()   # entries + the persistent compile cache

    @app.function(image=cpu_image, cpu=float(DECODE_CPU),
                  timeout=DECODE_TIMEOUT_S, volumes={VOL: vol})
    def decode(tasks: list) -> list:
        _remote_path()
        import torch

        torch.set_num_threads(DECODE_CPU)
        vol.reload()
        try:
            return decode_tasks(tasks, VOL_GEN, lambda: load_vae(VOL_VAE))
        finally:
            vol.commit()   # decoded PNGs

    @app.local_entrypoint()
    def main(n_dataset: int = 40, n_ood: int = len(OOD_PROMPTS),
             steps: int = DEFAULT_STEPS, cfg: float = DEFAULT_CFG,
             seed: int = 0, only_prompts: bool = False, limit: int = 0,
             gpu: str = DEFAULT_GPU, out: str = "", ft_key: str = R2_KEY,
             base_key: str = "", labels: str = ",".join(MODELS),
             models: str = "", pairs: str = "", cache: bool = True,
             cache_only: bool = False):
        if n_dataset < 0 or steps <= 0 or cfg < 0 or limit < 0:
            raise SystemExit("--n-dataset/--limit must be >= 0, --steps > 0, "
                             "--cfg >= 0")
        try:
            ood_prompts(n_ood)   # validates --n-ood
            gpu = check_gpu(gpu)
            ms, ps = resolve_models(models, pairs, base_key, ft_key, labels)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        run_eval(
            prepare=lambda **kw: prepare.remote(**kw),
            generate=lambda *a: generate.with_options(gpu=gpu).remote(*a),
            decode_map=lambda chunks_: decode.map(chunks_),
            models=ms, pairs=ps, n_dataset=n_dataset, n_ood=n_ood,
            steps=steps, cfg=cfg, seed=seed, only_prompts=only_prompts,
            limit=limit, gpu=gpu, out=out, use_cache=cache,
            cache_only=cache_only)


# ------------------------------------------------------------------ local CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] + " (run with `modal run`; plain "
                    "python only prints the prompt table)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Modal: modal run scripts/modal_eval_full.py [--n-dataset 40 "
               "--n-ood 10 --steps 28 --cfg 4.5 --seed 0 --only-prompts "
               "--limit N --gpu H100 --out DIR --models L=KEY,... --pairs "
               "A:B,... --no-cache --cache-only | --ft-key KEY --base-key "
               "KEY --labels base,ft]")
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
    ap.add_argument("--ft-key", default=R2_KEY,
                    help="R2 object of the ft side (legacy form)")
    ap.add_argument("--base-key", default="",
                    help="R2 object of the base side (legacy form; default: "
                         "the HF base)")
    ap.add_argument("--labels", default=",".join(MODELS),
                    help="labels of the base and ft sides (legacy form)")
    ap.add_argument("--models", default="",
                    help="label=KEY,... (KEY an R2 key or hf-base)")
    ap.add_argument("--pairs", default="",
                    help="A:B,... by label (A = base side, B = ft side)")
    return ap


def local_main(argv=None):
    """Local (no Modal) --only-prompts: the prompt table from a local
    parquet, the CLI validation and the estimate with nothing cached."""
    args = build_arg_parser().parse_args(argv)
    if not args.only_prompts:
        raise SystemExit("this script runs on Modal: modal run "
                         "scripts/modal_eval_full.py ... (locally only "
                         "--only-prompts --parquet PATH)")
    if not args.parquet:
        raise SystemExit("--only-prompts locally needs --parquet PATH")
    try:
        ms, ps = resolve_models(args.models, args.pairs, args.base_key,
                                args.ft_key, args.labels)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    prompts = apply_limit(build_prompts(args.parquet, args.n_dataset,
                                        args.n_ood, args.seed), args.limit)
    print(prompt_table(prompts))
    print("models: " + ", ".join(f"{label}={key}" for label, key in ms)
          + "; pairs: " + ", ".join(f"{a}:{b}" for a, b in ps))
    print_estimate(prompts, args.steps, args.cfg, args.gpu,
                   {label: [p["id"] for p in prompts] for label, _ in ms})
    return prompts


if __name__ == "__main__":
    local_main()
