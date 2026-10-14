"""Modal app: score every image of the stage-3 dataset for "inset picture on
a flat canvas" (white/flat margins around a rectangular picture, pillar- or
letterbox bars, a frame line + mat) and build a human review page to pick
images for removal.

    modal run scripts/modal_score_borders.py --shards 2       # smoke run (2 tars)
    modal run scripts/modal_score_borders.py                  # all ~100 shards
    modal run scripts/modal_score_borders.py --top-n 1500 --score-min 0.05 \\
        --thumb-frac 0.03 --control-n 150 --seed 0 --out DIR

    python scripts/modal_score_borders.py --tar-dir DIR --parquet P --out OUT
                       # the same pipeline locally on tars already on disk (no Modal)

Metric (border_metrics, pure numpy on the full-resolution RGB image):
  1. background candidate: the outer RING_PX ring. bg = the per-channel
     median of the whole ring, or of one side strip (top/bottom/left/right)
     when that matches more ring pixels (pillar/letterbox bars cover only
     part of the ring). ring_uniform = fraction of ring pixels whose
     max-abs-channel difference to bg is <= TOL; bg luminance, near-white /
     near-black recorded.
  2. bg mask = pixels within TOL of bg.
  3. margins: consecutive full rows from the top/bottom whose bg fraction is
     >= ROW_BG_FRAC, then columns from the left/right measured inside those
     rows, then rows again inside the columns (2 passes). Fractions of H/W;
     content bbox = the rest; margin_area_frac = 1 - bbox area / image area;
     n_sides = sides with margin >= MIN_SIDE_FRAC.
  4. fill = non-bg fraction inside the bbox.
  5. per-side straightness = max over the first EDGE_PX rows/cols inside the
     bbox of their non-bg fraction; edge_straightness = mean over the sides
     with margin > 0 (rectangle ~1, silhouette low).
  6. pillar_or_letter = margins (>= MIN_SIDE_FRAC) on exactly two opposite
     sides, both with straightness >= PILLAR_STRAIGHT.
     score = margin_area_frac * edge_straightness * min(1, fill / FILL_FULL)
     when ring_uniform >= RING_UNIFORM_MIN or pillar_or_letter, else 0
     (bars on two sides cover only part of the ring, so the ring gate alone
     would zero every pillarbox).
  7. frame line + mat: when all four margins are thin (0 < m < MIN_SIDE_FRAC,
     `thin_frame`), the image is measured again inside the frame (+2 px);
     if that scores higher (a mat around an inset), its margins (added to
     the frame's), fill and straightness are used (`peeled`, `frame_px`).
     A thin frame line around a full-bleed picture keeps its tiny score
     (margin_area_frac ~ the line) and is only flagged `thin_frame`.

Stages (Modal; the local entrypoint is stdlib-only, like the other Modal
scripts, so everything that needs numpy/pyarrow/PIL runs in containers):
  1. `prepare` (CPU): reads metadata.parquet (the local copy LOCAL_PARQUET is
     sent when it exists, else it is downloaded from the private dataset
     repo with the HF token) -> per id: tar, tags of interest, caption flags
     (long caption mentions border|frame|margin|letterbox|pillarbox|inset /
     watermark), caption snippet; the ids that always get a thumbnail
     (sample watermark, border-tagged without a border caption, a seeded
     random control sample of `white background`).
  2. `score_shard` (CPU, Modal default resources) .starmap over the tars:
     hf_hub_download of one tar (container disk), border_metrics of every
     image (ids from the member names; nothing else read), a JPEG thumbnail
     (long side THUMB_SIDE, q THUMB_QUALITY) for candidates = score >=
     --score-min OR pillar_or_letter OR the shard's top --thumb-frac by score
     OR the always-thumb ids. A failed shard is reported, the rest written.
  3. local: <out>/thumbs/<id>.jpg, <out>/scores.parquet (all rows, metrics +
     flags; built by the `finalize` container unless pyarrow is importable
     locally), <out>/review/index.html (self-contained, inline JSON, thumbs
     by relative path), <out>/summary.json and a printed summary.

Review page tabs: (A) top --top-n by score (score > 0 only), (B) sample watermark (all),
(C) border-tagged but the long caption does not mention a border, (D)
--control-n random `white background` images (false-positive check). Click
a thumbnail = toggle "remove" (red outline; kept in localStorage); "export"
downloads datafilter_remove.json (id + tab + score per selected image).

Secrets: Modal secret `huggingface` (HF_TOKEN, HUGGING_FACE_HUB_TOKEN or
HUGGINGFACE_TOKEN); the value is never printed.

Importable without the Modal SDK; the metric, the shard scorer, flags, the
plan, the table, the summary and the HTML writer are plain functions
(tests/test_score_borders.py).
"""

import argparse
import html
import importlib.util
import io
import json
import math
import os
import random
import re
import tarfile
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

try:
    import modal
except ImportError:  # importable (pyflakes, tests) without the Modal SDK
    modal = None

LOCAL_REPO = Path(__file__).resolve().parent.parent
LOCAL_PARQUET = Path("/home/mj/krea2-ops/mi350x/data/metadata.parquet")
LOCAL_OUT = Path("/home/mj/krea2-ops/eval/datafilter")

DATASET_REPO = "ricecake/danbooru-curated-v2"
PARQUET = "metadata.parquet"
TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
REMOTE_DL = "/tmp/dl"     # tar downloads: container-local disk
IMAGE_EXTS = {"webp", "png", "jpg", "jpeg"}

# --- metric ---
# TOL: max-abs-channel distance to the canvas colour still counted as canvas.
# A flat canvas after lossy WebP carries ~+-2..4 of block noise and up to
# ~+-10 of ringing next to the content edge; paper-like canvases add a few
# more. 18 keeps those margin rows >= 98% canvas (tests: sigma-4 Gaussian
# noise + WebP q80 round trip), while a picture's near-white highlights
# (>= ~20 away, e.g. 235 on a 255 canvas) still count as content.
TOL = 18
RING_PX = 2
ROW_BG_FRAC = 0.98        # a margin row/col is >= 98% canvas
MIN_SIDE_FRAC = 0.015     # a "real" margin side: >= 1.5% of that dimension
EDGE_PX = 4               # straightness: best of the first 4 rows/cols inside
#                           the bbox (WebP chroma ringing spills 2-3 px past a
#                           hard edge, making the first 1-2 lines partial)
FILL_FULL = 0.85          # fill at which the fill factor saturates
RING_UNIFORM_MIN = 0.9
PILLAR_STRAIGHT = 0.9
PEEL_PAD = 2              # px skipped inside a thin frame (antialiased edge)
PEEL_MIN_PX = 32          # no peel when the inside is smaller than this
NEAR_WHITE_MIN = 235      # bg_white: every channel >= this
NEAR_BLACK_MAX = 25       # bg_black: every channel <= this
SIDES = ("top", "bottom", "left", "right")
ROUND = 4

# --- candidates / review ---
SCORE_MIN = 0.05          # see README: a 1.3% border on all four sides,
#                           a 5% band on one side, or a rectangle in a
#                           large canvas
THUMB_FRAC = 0.03         # plus each shard's own top 3% by score
TOP_N = 1500
CONTROL_N = 150
THUMB_SIDE = 512
THUMB_QUALITY = 80
SHARD_THREADS = 2
SHARD_TIMEOUT_S = 20 * 60
# Resources. score_shard runs on Modal's default resources (no cpu/memory
# reservation): one synthetic 950-image shard (832x1216 WebP, 293 MB) peaked
# at 91 MiB RSS with 2 threads (69 MiB with 1); 2 threads use spare CPU when
# the host has it. prepare / finalize hold the whole metadata (load_meta of
# the real parquet: 716 MiB peak; the joined rows -> parquet: < 700 MiB), so
# they reserve 1 GiB.
META_MEM_MIB = 1024
# rough per-shard container time for the estimate: start + ~200 MB download
# + ~950 images at ~23 ms (WebP decode 18 ms + metric 5 ms) on 1-2 cores
EST_S_PER_SHARD = 60
EST_CORES = 2
TAGS_OF_INTEREST = (
    "border", "white border", "outside border", "pillarboxed", "page number",
    "sample watermark", "picture frame", "framed", "inset", "collage",
    "white background",
)
BORDER_TAGS = ("border", "white border", "outside border", "pillarboxed")
CAPTION_BORDER_RE = re.compile(
    r"\b(border|frame|margin|letterbox|pillarbox|inset)", re.IGNORECASE)
CAPTION_WATERMARK_RE = re.compile(r"watermark", re.IGNORECASE)
SNIPPET_CHARS = 240
TABS = (
    ("A", "top by score"),
    ("B", "sample watermark"),
    ("C", "border tag, no border caption"),
    ("D", "white background control"),
)

# Modal pricing (modal.com/pricing, fetched 2026-10-13, as in
# scripts/modal_cache_latents.py): per physical core and per GiB hour
CPU_CORE_USD_H = 0.0472
MEM_GIB_USD_H = 0.00799

PIP_PACKAGES = [
    "pillow==12.3.0", "numpy==2.5.1", "pyarrow==26.0.0",
    "huggingface_hub[hf_xet]==1.24.0", "hf_xet==1.5.2",
]
CONTAINER_ENV = {
    "HF_HOME": "/tmp/hf",
    "HF_XET_CACHE": "/tmp/hf_xet",
    "PYTHONUNBUFFERED": "1",
}


# ------------------------------------------------------------------ metric
def _r(x):
    return round(float(x), ROUND)


def _bg_mask(rgb, bg, tol=TOL):
    """bool [H, W]: max-abs-channel difference to bg <= tol (uint8
    comparisons against the clipped [bg - tol, bg + tol] box)."""
    import numpy as np

    lo = np.clip(np.asarray(bg, np.int16) - tol, 0, 255).astype(np.uint8)
    hi = np.clip(np.asarray(bg, np.int16) + tol, 0, 255).astype(np.uint8)
    m = (rgb[..., 0] >= lo[0]) & (rgb[..., 0] <= hi[0])
    for c in (1, 2):
        ch = rgb[..., c]
        m &= ch >= lo[c]
        m &= ch <= hi[c]
    return m


def _pick_bg(rgb, tol=TOL, k=RING_PX):
    """-> (bg uint8[3], ring_uniform, source). Candidates: the median of the
    whole ring and of each side strip; the one matching most ring pixels
    wins (the whole ring on ties)."""
    import numpy as np

    h, w = rgb.shape[:2]
    k = max(1, min(k, h // 2, w // 2))
    ring = np.concatenate([
        rgb[:k].reshape(-1, 3), rgb[h - k:].reshape(-1, 3),
        rgb[k:h - k, :k].reshape(-1, 3), rgb[k:h - k, w - k:].reshape(-1, 3)])
    strips = {"ring": ring, "top": rgb[:k].reshape(-1, 3),
              "bottom": rgb[h - k:].reshape(-1, 3),
              "left": rgb[:, :k].reshape(-1, 3),
              "right": rgb[:, w - k:].reshape(-1, 3)}
    best = None
    ring16 = ring.astype(np.int16)
    for name, px in strips.items():
        cand = np.round(np.median(px, axis=0)).astype(np.int16)
        frac = float((np.abs(ring16 - cand).max(axis=1) <= tol).mean())
        if best is None or frac > best[1] + 1e-12:
            best = (cand.astype(np.uint8), frac, name)
    return best


def _lead(frac, thr=ROW_BG_FRAC):
    """Number of leading entries of frac that are >= thr."""
    import numpy as np

    bad = np.flatnonzero(frac < thr)
    return int(bad[0]) if bad.size else int(len(frac))


def _frac_bg(mask, axis):
    import numpy as np

    n = mask.shape[axis]
    return np.count_nonzero(mask, axis=axis) / max(n, 1)


def _measure(rgb, tol=TOL):
    """Core measurement of one image -> dict with px margins (t, b, l, r),
    fill, per-side straightness of the bbox edges (all four sides), bg
    colour / ring uniformity. blank=True when no content bbox exists."""
    import numpy as np

    h, w = rgb.shape[:2]
    bg, ring_uniform, src = _pick_bg(rgb, tol)
    mask = _bg_mask(rgb, bg, tol)
    t = b = l = r = 0
    blank = False
    for _ in range(2):
        rows = _frac_bg(mask[:, l:w - r], axis=1)
        t = _lead(rows)
        if t >= h:
            blank = True
            break
        b = _lead(rows[::-1])
        cols = _frac_bg(mask[t:h - b, :], axis=0)
        l = _lead(cols)
        if l >= w:
            blank = True
            break
        r = _lead(cols[::-1])
    out = {"h": h, "w": w, "bg": [int(x) for x in bg],
           "ring_uniform": ring_uniform, "bg_src": src}
    if blank:
        out.update(px=(h, 0, 0, 0), fill=0.0, blank=True,
                   straight={s: 0.0 for s in SIDES})
        return out
    box = mask[t:h - b, l:w - r]          # True = canvas
    bh, bw = box.shape
    fill = 1.0 - float(np.count_nonzero(box)) / box.size
    e = min(EDGE_PX, bh, bw)
    straight = {
        "top": max(1.0 - float(box[i].mean()) for i in range(e)),
        "bottom": max(1.0 - float(box[bh - 1 - i].mean()) for i in range(e)),
        "left": max(1.0 - float(box[:, i].mean()) for i in range(e)),
        "right": max(1.0 - float(box[:, bw - 1 - i].mean()) for i in range(e)),
    }
    out.update(px=(t, b, l, r), fill=fill, blank=False, straight=straight)
    return out


def _finish(h, w, px, fill, straight, ring_uniform, blank=False):
    """Derived numbers + score from px margins (t, b, l, r) of an h x w
    image, the bbox fill and per-side straightness."""
    t, b, l, r = px
    frac = {"top": t / h, "bottom": b / h, "left": l / w, "right": r / w}
    if blank:
        maf = 1.0
    else:
        maf = 1.0 - (h - t - b) * (w - l - r) / float(h * w)
    with_margin = [s for s in SIDES if frac[s] > 0]
    real = {s for s in SIDES if frac[s] >= MIN_SIDE_FRAC}
    es = (sum(straight[s] for s in with_margin) / len(with_margin)
          if with_margin and not blank else 0.0)
    pillar = (not blank and real in ({"top", "bottom"}, {"left", "right"})
              and min(straight[s] for s in real) >= PILLAR_STRAIGHT)
    gate = ring_uniform >= RING_UNIFORM_MIN or pillar
    score = maf * es * min(1.0, fill / FILL_FULL) if gate and not blank else 0.0
    out = {f"margin_{s}": _r(frac[s]) for s in SIDES}
    out.update({
        "margin_area_frac": _r(maf), "n_sides": len(real), "fill": _r(fill),
        **{f"straight_{s}": (_r(straight[s]) if frac[s] > 0 else -1.0)
           for s in SIDES},
        "edge_straightness": _r(es), "pillar_or_letter": bool(pillar),
        "score": _r(score),
    })
    return out


def border_metrics(rgb, tol=TOL):
    """rgb: uint8 [H, W, 3] (full resolution, no downscale) -> dict of
    rounded floats / ints / bools (see the module docstring)."""
    import numpy as np

    rgb = np.ascontiguousarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 [H, W, 3], got {rgb.dtype} "
                         f"{rgb.shape}")
    h, w = rgb.shape[:2]
    m = _measure(rgb, tol)
    t, b, l, r = m["px"]
    thin = (not m["blank"] and min(m["px"]) > 0
            and t < MIN_SIDE_FRAC * h and b < MIN_SIDE_FRAC * h
            and l < MIN_SIDE_FRAC * w and r < MIN_SIDE_FRAC * w)
    res = _finish(h, w, m["px"], m["fill"], m["straight"], m["ring_uniform"],
                  m["blank"])
    bg, ru, peeled, frame_px = m["bg"], m["ring_uniform"], False, 0
    if thin:
        y0, y1 = t + PEEL_PAD, h - b - PEEL_PAD
        x0, x1 = l + PEEL_PAD, w - r - PEEL_PAD
        if y1 - y0 >= PEEL_MIN_PX and x1 - x0 >= PEEL_MIN_PX:
            inner = _measure(rgb[y0:y1, x0:x1], tol)
            if not inner["blank"]:
                it, ib, il, ir = inner["px"]
                px = (y0 + it, (h - y1) + ib, x0 + il, (w - x1) + ir)
                cand = _finish(h, w, px, inner["fill"], inner["straight"],
                               inner["ring_uniform"])
                if cand["score"] > res["score"]:
                    res, peeled = cand, True
                    bg, ru = inner["bg"], inner["ring_uniform"]
                    frame_px = max(m["px"])
    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    res.update({
        "h": h, "w": w, "bg_r": bg[0], "bg_g": bg[1], "bg_b": bg[2],
        "bg_lum": _r(lum), "bg_white": bool(min(bg) >= NEAR_WHITE_MIN),
        "bg_black": bool(max(bg) <= NEAR_BLACK_MAX),
        "ring_uniform": _r(ru), "bg_src": m["bg_src"] if not peeled else "peel",
        "thin_frame": bool(thin), "peeled": bool(peeled),
        "frame_px": int(frame_px), "blank": bool(m["blank"]),
    })
    return res


# ------------------------------------------------------------------ shard
def decode_rgb(data):
    """Encoded image bytes -> uint8 [H, W, 3] (sRGB; alpha dropped)."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def thumb_jpeg(rgb, side=THUMB_SIDE, quality=THUMB_QUALITY):
    from PIL import Image

    im = Image.fromarray(rgb)
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def iter_tar_images(tar_path):
    """Yield (id, encoded bytes) of every image member (id = member stem;
    json/txt members skipped, nothing else read)."""
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, _, ext = Path(m.name).name.partition(".")
            if ext.lower() not in IMAGE_EXTS:
                continue
            yield int(stem), tf.extractfile(m).read()


def pick_thumbs(rows, always=(), score_min=SCORE_MIN, thumb_frac=THUMB_FRAC):
    """Ids of a shard that get a thumbnail: score >= score_min, pillar_or_letter,
    the shard's top ceil(thumb_frac * n) by score (score > 0 only), and the
    always ids present in the shard."""
    ok = [r for r in rows if "error" not in r]
    keep = {r["id"] for r in ok
            if r["score"] >= score_min or r["pillar_or_letter"]}
    k = math.ceil(thumb_frac * len(ok)) if thumb_frac > 0 else 0
    ranked = sorted((r for r in ok if r["score"] > 0),
                    key=lambda r: (-r["score"], r["id"]))
    keep.update(r["id"] for r in ranked[:k])
    present = {r["id"] for r in ok}
    keep.update(i for i in always if i in present)
    return keep


def score_shard_impl(tar_path, always=(), score_min=SCORE_MIN,
                     thumb_frac=THUMB_FRAC, threads=SHARD_THREADS, log=print):
    """Score every image of one tar -> {"rows": [{id, **metrics} or {id,
    error}], "thumbs": {id: jpeg bytes}, "seconds", "n"}. Two sequential reads
    of the tar: all metrics, then thumbnails of the candidates only."""
    t0 = time.time()
    always = {int(i) for i in always}

    def one(item):
        sid, data = item
        try:
            return {"id": sid, **border_metrics(decode_rgb(data))}
        except Exception as e:   # a broken image must not lose the shard
            return {"id": sid, "error": f"{type(e).__name__}: {e}"[:300]}

    threads = max(1, threads)
    rows, pending = [], deque()
    with ThreadPoolExecutor(threads) as pool:
        # bounded look-ahead: only a few encoded images in memory at a time
        # (Executor.map would read the whole ~200 MB tar up front)
        for item in iter_tar_images(tar_path):
            pending.append(pool.submit(one, item))
            if len(pending) >= 4 * threads:
                rows.append(pending.popleft().result())
        while pending:
            rows.append(pending.popleft().result())
    t1 = time.time()
    want = pick_thumbs(rows, always, score_min, thumb_frac)
    thumbs = {}
    for sid, data in iter_tar_images(tar_path):
        if sid in want:
            try:
                thumbs[sid] = thumb_jpeg(decode_rgb(data))
            except Exception as e:
                log(f"{Path(tar_path).name}: thumb {sid} failed: {e}")
    t2 = time.time()
    n_err = sum("error" in r for r in rows)
    log(f"{Path(tar_path).name}: {len(rows)} imgs, {n_err} errors, "
        f"{len(thumbs)} thumbs; metrics {t1 - t0:.1f}s, thumbs {t2 - t1:.1f}s")
    return {"rows": rows, "thumbs": thumbs, "n": len(rows),
            "seconds": {"metrics": round(t1 - t0, 2),
                        "thumbs": round(t2 - t1, 2)}}


# ------------------------------------------------------------------ metadata
def split_tags(s):
    return {t.strip() for t in (s or "").split(",") if t.strip()}


def flags_of(tags, long_caption):
    """Per-image flags: exact tag matches of TAGS_OF_INTEREST + caption
    mentions (long caption)."""
    ts = split_tags(tags)
    out = {"tag_" + t.replace(" ", "_"): t in ts for t in TAGS_OF_INTEREST}
    cap = long_caption or ""
    out["caption_mentions_border"] = bool(CAPTION_BORDER_RE.search(cap))
    out["caption_mentions_watermark"] = bool(CAPTION_WATERMARK_RE.search(cap))
    return out


def tags_hit(flags):
    return [t for t in TAGS_OF_INTEREST if flags["tag_" + t.replace(" ", "_")]]


def is_border_uncaptioned(flags):
    return (any(flags["tag_" + t.replace(" ", "_")] for t in BORDER_TAGS)
            and not flags["caption_mentions_border"])


def load_meta(parquet_path):
    """metadata.parquet -> {id: {tar, bucket_w, bucket_h, flags..., cap}}
    (plain Python values)."""
    import pyarrow.parquet as pq

    cols = ["id", "long", "tags", "bucket_w", "bucket_h", "tar"]
    t = pq.read_table(parquet_path, columns=cols).to_pydict()
    meta = {}
    for i, sid in enumerate(t["id"]):
        long_cap = t["long"][i] or ""
        meta[int(sid)] = {
            "tar": t["tar"][i], "bucket_w": int(t["bucket_w"][i]),
            "bucket_h": int(t["bucket_h"][i]),
            **flags_of(t["tags"][i], long_cap),
            "cap": long_cap[:SNIPPET_CHARS],
        }
    return meta


def plan_job(meta, seed=0, control_n=CONTROL_N):
    """-> {"shards": {tar: n ids}, "always": {tar: [ids]}, "control": [ids],
    "watermark": [ids], "uncaptioned": [ids]} (sorted ids)."""
    watermark = sorted(i for i, m in meta.items() if m["tag_sample_watermark"])
    unc = sorted(i for i, m in meta.items() if is_border_uncaptioned(m))
    white = sorted(i for i, m in meta.items() if m["tag_white_background"])
    control = sorted(random.Random(seed).sample(white, min(control_n,
                                                           len(white))))
    always = {}
    for i in sorted(set(watermark) | set(unc) | set(control)):
        always.setdefault(meta[i]["tar"], []).append(i)
    shards = Counter(m["tar"] for m in meta.values())
    return {"shards": dict(sorted(shards.items())), "always": always,
            "control": control, "watermark": watermark, "uncaptioned": unc}


# ------------------------------------------------------------------ outputs
def join_rows(rows, meta):
    """Score rows + metadata flags -> list of flat dicts sorted by score
    (desc), then id. Rows without metadata keep their metrics (tar None)."""
    out = []
    for r in rows:
        m = meta.get(r["id"], {})
        row = dict(r)
        row.update({k: v for k, v in m.items() if k != "cap"})
        row.setdefault("tar", None)
        row["border_uncaptioned"] = bool(m) and is_border_uncaptioned(m)
        out.append(row)
    out.sort(key=lambda r: (-(r.get("score") or 0.0), r["id"]))
    return out


def table_bytes(joined):
    """Joined rows -> scores.parquet bytes (pyarrow; columns = the union of
    the row keys, missing values null)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys = []
    for r in joined:
        for k in r:
            if k not in keys:
                keys.append(k)
    t = pa.Table.from_pydict({k: [r.get(k) for r in joined] for k in keys})
    buf = io.BytesIO()
    pq.write_table(t, buf, compression="zstd")
    return buf.getvalue()


def quantiles(xs, qs=(0.5, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 1.0)):
    xs = sorted(xs)
    if not xs:
        return {}
    return {f"p{q * 100:g}": xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)]
            for q in qs}


def build_tabs(joined, plan, top_n=TOP_N):
    ok = [r for r in joined if "error" not in r and r["score"] > 0]
    return {
        "A": [r["id"] for r in ok[:top_n]],
        "B": list(plan["watermark"]),
        "C": list(plan["uncaptioned"]),
        "D": list(plan["control"]),
    }


def summarize(joined, tabs, have_thumbs, plan, seconds, failed=(),
              score_min=SCORE_MIN):
    ok = [r for r in joined if "error" not in r]
    scores = [r["score"] for r in ok]
    top = set(tabs["A"])
    overlap = {}
    for t in BORDER_TAGS + ("picture frame", "framed", "inset"):
        k = "tag_" + t.replace(" ", "_")
        tagged = {r["id"] for r in ok if r.get(k)}
        overlap[t] = {"tagged": len(tagged), "in_top": len(tagged & top)}
    anyb = {r["id"] for r in ok
            if any(r.get("tag_" + t.replace(" ", "_")) for t in BORDER_TAGS)}
    overlap["any border tag"] = {"tagged": len(anyb), "in_top": len(anyb & top)}
    expected = sum(plan["shards"].values()) if plan else None
    return {
        "images": len(joined), "errors": len(joined) - len(ok),
        "expected": expected, "failed_shards": list(failed),
        "score_quantiles": quantiles(scores),
        "score_gt0": sum(s > 0 for s in scores),
        "score_ge_min": sum(s >= score_min for s in scores),
        "pillar_or_letter": sum(bool(r.get("pillar_or_letter")) for r in ok),
        "thin_frame": sum(bool(r.get("thin_frame")) for r in ok),
        "peeled": sum(bool(r.get("peeled")) for r in ok),
        "tabs": {k: len(v) for k, v in tabs.items()},
        "missing_thumbs": {k: sum(i not in have_thumbs for i in v)
                           for k, v in tabs.items()},
        "top_overlap": overlap, "seconds": seconds,
    }


def print_summary(s, log=print):
    log(f"images scored: {s['images']} (expected {s['expected']}), "
        f"errors {s['errors']}, failed shards {len(s['failed_shards'])}"
        + (f" {s['failed_shards']}" if s["failed_shards"] else ""))
    log("score quantiles: " + ", ".join(f"{k} {v:.4f}" for k, v in
                                        s["score_quantiles"].items()))
    log(f"score > 0: {s['score_gt0']}, >= score-min: {s['score_ge_min']}, "
        f"pillar_or_letter: {s['pillar_or_letter']}, thin_frame: "
        f"{s['thin_frame']}, peeled: {s['peeled']}")
    log("tabs: " + ", ".join(f"{k} {n} (no thumb {s['missing_thumbs'][k]})"
                             for k, n in s["tabs"].items()))
    log("top-N overlap with tags: " + ", ".join(
        f"{t} {v['in_top']}/{v['tagged']}" for t, v in s["top_overlap"].items()))
    log("runtime: " + ", ".join(f"{k} {v:.1f}s" for k, v in s["seconds"].items()))


def review_items(joined, tabs, meta, have_thumbs):
    """Compact per-id records for every id in a tab."""
    by_id = {r["id"]: r for r in joined}
    items = {}
    for ids in tabs.values():
        for i in ids:
            if i in items:
                continue
            r = by_id.get(i, {})
            m = meta.get(i, {})
            items[str(i)] = {
                "s": r.get("score"), "err": r.get("error"),
                "m": [r.get(f"margin_{s}") for s in SIDES],
                "st": r.get("edge_straightness"), "f": r.get("fill"),
                "pl": bool(r.get("pillar_or_letter")),
                "tg": tags_hit(m) if m else [],
                "cb": bool(m.get("caption_mentions_border")),
                "cap": m.get("cap", ""), "th": i in have_thumbs,
            }
    return items


def render_review(tabs, items, title="datafilter review", info=""):
    """Self-contained review page (inline JSON, thumbs at ../thumbs/<id>.jpg)."""
    data = json.dumps({"tabs": tabs, "items": items,
                       "names": dict(TABS)}, separators=(",", ":"))
    data = data.replace("</", "<\\/")
    return (REVIEW_HTML.replace("__TITLE__", html.escape(title))
            .replace("__INFO__", html.escape(info))
            .replace("__DATA__", data))


REVIEW_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fafafa;--fg:#1b1b1b;--muted:#666;--card:#fff;--line:#ddd;
--accent:#2563eb;--remove:#dc2626}
@media (prefers-color-scheme: dark){:root{--bg:#141414;--fg:#e8e8e8;
--muted:#9a9a9a;--card:#1f1f1f;--line:#333;--accent:#60a5fa;--remove:#f87171}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);
border-bottom:1px solid var(--line);padding:10px 16px;display:flex;
flex-wrap:wrap;gap:8px;align-items:center}
h1{font-size:16px;margin:0 12px 0 0}
.tab{border:1px solid var(--line);background:var(--card);color:var(--fg);
padding:5px 10px;border-radius:6px;cursor:pointer}
.tab.on{border-color:var(--accent);color:var(--accent);font-weight:600}
.sp{flex:1}
#count{color:var(--muted)}
button.act{border:1px solid var(--line);background:var(--card);color:var(--fg);
padding:5px 10px;border-radius:6px;cursor:pointer}
#info{color:var(--muted);font-size:12px;padding:6px 16px 0}
main{padding:12px 16px;display:grid;gap:10px;
grid-template-columns:repeat(auto-fill,minmax(180px,1fr))}
.c{background:var(--card);border:3px solid transparent;border-radius:8px;
padding:4px;cursor:pointer;outline:1px solid var(--line)}
.c.rm{border-color:var(--remove);outline-color:var(--remove)}
.c img{width:100%;height:220px;object-fit:contain;display:block;
background:repeating-conic-gradient(#8882 0 25%,transparent 0 50%) 0 0/16px 16px}
.c .ph{height:220px;display:flex;align-items:center;justify-content:center;
color:var(--muted)}
.c .t{font-size:12px;margin-top:4px;word-break:break-word}
.c .t b{font-variant-numeric:tabular-nums}
.c .g{color:var(--muted);font-size:11px}
</style></head><body>
<header><h1>__TITLE__</h1><span id="tabs"></span><span class="sp"></span>
<span id="count"></span>
<button class="act" id="clear">clear tab</button>
<button class="act" id="export">export</button></header>
<div id="info">__INFO__</div>
<main id="grid"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
var D=JSON.parse(document.getElementById('data').textContent);
var KEY='datafilter_remove_v1', sel={}, cur='A';
try{sel=JSON.parse(localStorage.getItem(KEY)||'{}')||{}}catch(e){sel={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(sel))}catch(e){}}
function fmt(x){return x==null?'-':(+x).toFixed(3)}
function pct(x){return x==null?'-':Math.round(x*1000)/10}
function count(){var n=Object.keys(sel).length;
document.getElementById('count').textContent=n+' marked for removal'}
function tabs(){var el=document.getElementById('tabs');el.innerHTML='';
Object.keys(D.tabs).forEach(function(k){var b=document.createElement('button');
b.className='tab'+(k===cur?' on':'');
b.textContent=k+': '+(D.names[k]||'')+' ('+D.tabs[k].length+')';
b.onclick=function(){cur=k;tabs();grid()};el.appendChild(b)})}
function grid(){var g=document.getElementById('grid');g.innerHTML='';
var f=document.createDocumentFragment();
D.tabs[cur].forEach(function(id){var it=D.items[String(id)]||{};
var c=document.createElement('div');c.className='c'+(sel[id]?' rm':'');
var m=it.m||[];
c.title=(it.cap||'')+'\\n\\ntags: '+((it.tg||[]).join(', ')||'-')+
'\\nmargins t/b/l/r %: '+m.map(pct).join(' / ')+
'\\nstraightness '+fmt(it.st)+', fill '+fmt(it.f)+
(it.pl?'\\npillar/letterbox':'')+(it.cb?'\\ncaption mentions border':'')+
(it.err?'\\nerror: '+it.err:'');
if(it.th){var im=document.createElement('img');im.loading='lazy';
im.src='../thumbs/'+id+'.jpg';im.alt=String(id);c.appendChild(im)}
else{var p=document.createElement('div');p.className='ph';
p.textContent='no thumbnail';c.appendChild(p)}
var t=document.createElement('div');t.className='t';
t.innerHTML='<b>'+id+'</b> score <b>'+fmt(it.s)+'</b>'+(it.pl?' PL':'');
var gg=document.createElement('div');gg.className='g';
gg.textContent='m '+m.map(pct).join('/')+(it.tg&&it.tg.length?' | '+it.tg.join(', '):'');
t.appendChild(gg);c.appendChild(t);
c.onclick=function(){if(sel[id]){delete sel[id];c.classList.remove('rm')}
else{sel[id]={tab:cur,score:it.s};c.classList.add('rm')}save();count()};
f.appendChild(c)});g.appendChild(f);count()}
document.getElementById('clear').onclick=function(){
D.tabs[cur].forEach(function(id){delete sel[id]});save();grid()};
document.getElementById('export').onclick=function(){
var out={exported:new Date().toISOString(),items:Object.keys(sel).map(function(id){
return {id:+id,tab:sel[id].tab,score:sel[id].score}})};
var a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));
a.download='datafilter_remove.json';document.body.appendChild(a);a.click();
setTimeout(function(){URL.revokeObjectURL(a.href);a.remove()},1000)};
tabs();grid();
})();
</script></body></html>
"""


def write_outputs(out_dir, rows, thumbs, meta, plan, seconds, failed=(),
                  top_n=TOP_N, score_min=SCORE_MIN, table=None, info=""):
    """Write thumbs/<id>.jpg, scores.parquet, review/index.html and
    summary.json under out_dir. `table`: scores.parquet bytes made elsewhere
    (the `finalize` container), else built here with pyarrow. -> summary."""
    out_dir = Path(out_dir)
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    (out_dir / "review").mkdir(parents=True, exist_ok=True)
    for sid, jpg in thumbs.items():
        (out_dir / "thumbs" / f"{int(sid)}.jpg").write_bytes(jpg)
    have = {int(p.stem) for p in (out_dir / "thumbs").glob("*.jpg")
            if p.stem.isdigit()}
    joined = join_rows(rows, meta)
    if table is None:
        table = table_bytes(joined)
    (out_dir / "scores.parquet").write_bytes(table)
    tabs = build_tabs(joined, plan, top_n)
    items = review_items(joined, tabs, meta, have)
    s = summarize(joined, tabs, have, plan, seconds, failed, score_min)
    info = info or (f"{s['images']} images, score quantiles "
                    + ", ".join(f"{k} {v:.3f}" for k, v in
                                s["score_quantiles"].items()))
    (out_dir / "review" / "index.html").write_text(
        render_review(tabs, items, info=info), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1))
    return s


# ------------------------------------------------------------------ HF / env
def pick_hf_token(env):
    """-> (env var name, token) of the first non-empty HF token variable, or
    (None, None). The token value is never printed."""
    for name in TOKEN_ENV_NAMES:
        if env.get(name):
            return name, env[name]
    return None, None


def fetch_dataset_file(filename, dest_dir, token):
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATASET_REPO, filename, repo_type="dataset",
                                local_dir=str(dest_dir), token=token))


def peak_rss_mib():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def estimate_cost(n_shards, s_per_shard, cores=1.0, mem_gib=1.0):
    """Rough Modal cost: container-seconds x (cores + memory) prices."""
    h = n_shards * s_per_shard / 3600
    return h * (cores * CPU_CORE_USD_H + mem_gib * MEM_GIB_USD_H)


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ------------------------------------------------------------------ Modal app
if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*PIP_PACKAGES)
        .env(CONTAINER_ENV)
    )
    hf_secret = modal.Secret.from_name("huggingface")
    app = modal.App("krea2-score-borders", image=image)

    @app.function(secrets=[hf_secret], timeout=30 * 60,
                  memory=META_MEM_MIB)
    def prepare(parquet: bytes = b"", seed: int = 0,
                control_n: int = CONTROL_N) -> dict:
        """metadata.parquet (sent bytes, else downloaded) -> meta + plan."""
        if parquet:
            path = Path("/tmp") / PARQUET
            path.write_bytes(parquet)
            src = "local copy"
        else:
            name, token = pick_hf_token(os.environ)
            print(f"HF token from {name or 'nowhere (anonymous)'}")
            path = fetch_dataset_file(PARQUET, "/tmp/meta", token)
            src = f"hf {DATASET_REPO}"
        meta = load_meta(path)
        return {"meta": meta, "plan": plan_job(meta, seed, control_n),
                "source": src}

    @app.function(secrets=[hf_secret], timeout=SHARD_TIMEOUT_S, retries=2)
    def score_shard(tar: str, always: list, score_min: float,
                    thumb_frac: float) -> dict:
        name, token = pick_hf_token(os.environ)
        t0 = time.time()
        path = fetch_dataset_file(tar, REMOTE_DL, token)
        dl = time.time() - t0
        try:
            res = score_shard_impl(path, always, score_min, thumb_frac)
        finally:
            path.unlink(missing_ok=True)
        res["tar"] = tar
        res["seconds"]["download"] = round(dl, 2)
        res["peak_rss_mib"] = round(peak_rss_mib(), 1)
        return res

    @app.function(timeout=30 * 60, memory=META_MEM_MIB)
    def finalize(joined: list) -> bytes:
        return table_bytes(joined)

    @app.local_entrypoint()
    def main(shards: int = 0, top_n: int = TOP_N,
             score_min: float = SCORE_MIN, thumb_frac: float = THUMB_FRAC,
             control_n: int = CONTROL_N, seed: int = 0, out: str = "",
             parquet: str = str(LOCAL_PARQUET)):
        if shards < 0 or top_n < 0 or control_n < 0 or not 0 <= thumb_frac <= 1:
            raise SystemExit("--shards/--top-n/--control-n must be >= 0, "
                             "--thumb-frac in [0, 1]")
        out_dir = Path(out) if out else LOCAL_OUT
        t_start = time.time()
        pq_bytes = Path(parquet).read_bytes() if Path(parquet).is_file() else b""
        print(f"prepare: metadata from "
              f"{parquet if pq_bytes else 'HF (no local copy)'}")
        prep = prepare.remote(pq_bytes, seed, control_n)
        meta = {int(k): v for k, v in prep["meta"].items()}
        plan = prep["plan"]
        tars = list(plan["shards"])
        if shards:
            tars = tars[:shards]
            plan = dict(plan, shards={t: plan["shards"][t] for t in tars})
        print(f"{len(meta)} images in {len(plan['shards'])} shards "
              f"(running {len(tars)}); always-thumb: {len(plan['watermark'])} "
              f"watermark, {len(plan['uncaptioned'])} border-uncaptioned, "
              f"{len(plan['control'])} control")
        usd = estimate_cost(len(tars), EST_S_PER_SHARD, EST_CORES, 0.25)
        print(f"estimate (rough): {len(tars)} containers x ~{EST_S_PER_SHARD}"
              f" s, ~${usd:.2f}")
        t_map = time.time()
        rows, thumbs, failed, per = [], {}, [], []
        args = [(t, plan["always"].get(t, []), score_min, thumb_frac)
                for t in tars]
        for tar, res in zip(tars, score_shard.starmap(
                args, return_exceptions=True, order_outputs=True)):
            if isinstance(res, BaseException):
                print(f"  {tar}: FAILED {type(res).__name__}: {res}")
                failed.append(tar)
                continue
            rows.extend(res["rows"])
            thumbs.update(res["thumbs"])
            per.append(res)
            print(f"  {tar}: {res['n']} imgs, {len(res['thumbs'])} thumbs, "
                  + ", ".join(f"{k} {v:.0f}s" for k, v in
                              res["seconds"].items())
                  + f", peak RSS {res['peak_rss_mib']:.0f} MiB")
        t_fin = time.time()
        if shards:   # control/watermark tabs only for the shards that ran
            done = set(tars)
            for k in ("control", "watermark", "uncaptioned"):
                plan[k] = [i for i in plan[k] if meta[i]["tar"] in done]
        table = None
        if importlib.util.find_spec("pyarrow") is None:
            # Modal's own CLI env has no pyarrow: build it in a container
            table = finalize.remote(join_rows(rows, meta))
        seconds = {"total": time.time() - t_start, "map": t_fin - t_map}
        if per:
            seconds["shard_mean"] = sum(
                sum(r["seconds"].values()) for r in per) / len(per)
            seconds["shard_max"] = max(sum(r["seconds"].values()) for r in per)
        s = write_outputs(out_dir, rows, thumbs, meta, plan, seconds, failed,
                          top_n, score_min, table)
        print_summary(s)
        print(f"wrote {out_dir}/scores.parquet, {len(thumbs)} thumbs, "
              f"{out_dir}/review/index.html")
        if failed:
            raise SystemExit(f"{len(failed)} shards failed: {failed}")


# ------------------------------------------------------------------ local CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] + " (local mode: tars on disk)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Modal: modal run scripts/modal_score_borders.py [--shards N "
               "--top-n 1500 --score-min 0.05 --thumb-frac 0.03 --control-n "
               "150 --seed 0 --out DIR --parquet PATH]")
    ap.add_argument("--tar-dir", required=True,
                    help="directory with the dataset tars (by their parquet "
                         "`tar` path or basename)")
    ap.add_argument("--parquet", default=str(LOCAL_PARQUET))
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--score-min", type=float, default=SCORE_MIN)
    ap.add_argument("--thumb-frac", type=float, default=THUMB_FRAC)
    ap.add_argument("--control-n", type=int, default=CONTROL_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=SHARD_THREADS)
    return ap


def local_main(argv=None):
    """Same pipeline as the Modal run on tars already on disk (only the
    tars present are scored)."""
    a = build_arg_parser().parse_args(argv)
    t0 = time.time()
    meta = load_meta(a.parquet)
    plan = plan_job(meta, a.seed, a.control_n)
    tars = []
    for tar in plan["shards"]:
        for cand in (Path(a.tar_dir) / tar, Path(a.tar_dir) / Path(tar).name):
            if cand.is_file():
                tars.append((tar, cand))
                break
    if a.shards:
        tars = tars[:a.shards]
    if not tars:
        raise SystemExit(f"no tar of the parquet found under {a.tar_dir}")
    rows, thumbs = [], {}
    for tar, path in tars:
        res = score_shard_impl(path, plan["always"].get(tar, []),
                               a.score_min, a.thumb_frac, a.threads)
        rows.extend(res["rows"])
        thumbs.update(res["thumbs"])
    done = {t for t, _ in tars}
    plan = dict(plan, shards={t: plan["shards"][t] for t in done})
    for k in ("control", "watermark", "uncaptioned"):
        plan[k] = [i for i in plan[k] if meta[i]["tar"] in done]
    s = write_outputs(a.out, rows, thumbs, meta, plan,
                      {"total": time.time() - t0}, (), a.top_n, a.score_min)
    print_summary(s)
    return s


if __name__ == "__main__":
    local_main()
