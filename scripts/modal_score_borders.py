"""Modal app: score every image of the stage-3 dataset for "inset picture on
a flat canvas" (white/flat margins around a rectangular picture, pillar- or
letterbox bars, a frame line + mat) and build a human review page to pick
images for removal.

    modal run scripts/modal_score_borders.py --shards 2       # smoke run (2 tars)
    modal run scripts/modal_score_borders.py                  # all ~100 shards
    modal run scripts/modal_score_borders.py --top-n 1500 --control-n 150 \\
        --seed 0 --out DIR --revision SHA     # default: the default branch head

    python scripts/modal_score_borders.py --tar-dir DIR --parquet P --out OUT
                       # the same pipeline locally on tars already on disk (no Modal)
    python scripts/modal_score_borders.py --rerender DIR [--top-n N]
                       # rebuild only DIR/review/index.html from a finished
                       # run (scores.parquet, summary.json, thumbs; no tars)

Metric (border_metrics, pure numpy on the full-resolution RGB image):
  1. canvas colour candidates: the per-channel median of the outer RING_PX
     ring and of each side strip (top/bottom/left/right); only identical
     colours are merged, and a colour covering < SIDE_MATCH_MIN (49%) of
     every side strip is skipped (provably cannot produce a margin). Every
     remaining candidate goes through steps 2-6; the one with
     the highest score wins (the whole-ring median on ties; `bg_src`,
     `n_bg_candidates`). ring_uniform = fraction of ring pixels whose
     max-abs-channel difference to the colour is <= TOL; bg luminance,
     near-white / near-black recorded.
  2. bg mask = pixels within TOL of the colour.
  3. margins: consecutive full rows from the top/bottom whose bg fraction is
     >= ROW_BG_FRAC, then columns from the left/right measured inside those
     rows, then rows again inside the columns (2 passes). Fractions of H/W;
     content bbox = the rest; margin_area_frac = 1 - bbox area / image area;
     n_sides = sides with margin >= MIN_SIDE_FRAC (score / pillar only).
  4. fill = non-bg fraction inside the bbox.
  5. per-side straightness = max over the first EDGE_PX rows/cols inside the
     bbox of their non-bg fraction; edge_straightness = GEOMETRIC mean over
     the sides with margin > 0 (rectangle ~1, silhouette low; one straight
     cut of a bust cannot carry three round sides).
  6. pillar_or_letter = margins (>= MIN_SIDE_FRAC) on exactly two opposite
     sides, both with straightness >= PILLAR_STRAIGHT.
     score = margin_area_frac * edge_straightness * min(1, fill / FILL_FULL)
     when ring_uniform >= RING_UNIFORM_MIN or pillar_or_letter, else 0
     (bars on two sides cover only part of the ring, so the ring gate alone
     would zero every pillarbox).
  7. frame line + mat: for EACH outer candidate whose four margins are thin
     (0 < m < MIN_SIDE_FRAC, `thin_frame`), the image is measured again
     inside the frame (+2 px; best candidate of the crop); when that scores
     higher (a mat around an inset), its margins (added to the frame's),
     fill and straightness replace the candidate's own (`peeled`,
     `frame_px`). The final winner is chosen over the peeled / unpeeled
     results of all candidates.
     A thin frame line around a full-bleed picture keeps its tiny score
     (margin_area_frac ~ the line) and is only flagged `thin_frame`.
  8. side_pattern ("4", "3", "2-opp" = top+bottom or left+right, "2-adj",
     "1", "0"): the sides whose margin is >= SIDE_MIN_PX (3) px AND whose
     straightness is >= SIDE_STRAIGHT (0.25). A border is 4, 3 or 2
     opposite sides (BORDER_PATTERNS); see SIDE_MIN_PX for the thresholds.

Stages (Modal; the local entrypoint is stdlib-only, like the other Modal
scripts, so everything that needs numpy/pyarrow/PIL runs in containers):
  1. `prepare` (CPU): resolves the dataset repo's commit sha ONCE (--revision,
     default the default branch); every download of the run is pinned to it
     and it is recorded in summary.json, the scores.parquet schema metadata
     and the review page. metadata.parquet at that commit: the local copy
     LOCAL_PARQUET is used only when byte-identical to the commit's file
     (LFS sha256 / git blob id), else it is downloaded -> per id: tar, tags of interest, caption flags
     (long caption mentions border|frame|margin|letterbox|pillarbox|inset /
     watermark), caption snippet; the expected ids of every tar; the ids
     that always get a thumbnail (sample watermark, border-tagged without a
     border caption, a seeded random control sample of `white background`).
  2. pass 1, `score_shard` (CPU, Modal default resources) .starmap over the
     tars: hf_hub_download of one tar (container disk), border_metrics of
     every image, JPEG thumbnails (long side THUMB_SIDE, q THUMB_QUALITY) of
     the always-thumb ids of that tar. Members are WebDataset keys
     `<id>.<ext>` (id = basename up to the first dot, digits only; ext in
     MEMBER_EXTS): any other member is an error. Every shard result is
     checked against the tar's expected ids from the parquet (missing,
     extra and duplicate ids); a failed or mismatching shard fails the run
     (exit != 0, nothing written).
  3. global selection: top --top-n by score (score > 0, border side
     pattern) + every
     pillar_or_letter + the always-thumb ids; pass 2, `thumb_shard` over the
     tars that hold selected ids without a thumbnail, returning exactly
     those. A selected id still without a thumbnail fails the run.
  4. local: <out>/thumbs/<id>.jpg, <out>/scores.parquet (all rows, metrics +
     flags; built by the `finalize` container unless pyarrow is importable
     locally), <out>/review/index.html (self-contained, inline JSON, thumbs
     by relative path), <out>/summary.json and a printed summary.

Review page tabs: (A) top --top-n by score among rows with score > 0 and a
border side pattern (4 / 3 / 2 opposite sides), (B) sample watermark
(all), (C) border-tagged but the long caption does not mention a border,
(D) --control-n random `white background` images (false-positive check),
(E) pillar/letterbox. Every card shows its side pattern as a badge. A click
cycles keep (no outline) -> caption ("캡션 보완", amber) -> crop ("자르기",
blue) -> remove ("제거", red) -> keep; the header shows a legend with the
per-state counts. States live in localStorage under
datafilter_review_v2:<review id> (hash of the tabs + generation time +
revision) as {id: state}; foreign ids and malformed entries are dropped.
"export" downloads datafilter_actions.json {review_id, generated,
dataset_revision, items: [{id, action, tabs, score, side_pattern}]} (non-
keep images only).

--rerender DIR rebuilds only the review page of a finished run (see
`rerender`): side patterns are recomputed under the current rule from the
rounded margin / straightness columns (exact integer px and coverage
counts recovered with h / w), the other tabs keep their membership, and by
default tab A holds every border-pattern row within the run's score-rank
cutoff (summary.json `a_rule`; a legacy run: its unfiltered top-N, all
thumbnailed); a tab id without a thumbnail fails it. summary.json and the
page record the tab A rule (`a_rule`: name, version, thresholds, top_n,
rank_cutoff).

Secrets: Modal secret `huggingface` (HF_TOKEN, HUGGING_FACE_HUB_TOKEN or
HUGGINGFACE_TOKEN); the value is never printed.

Importable without the Modal SDK; the metric, the shard scorer, validation,
selection, flags, the plan, the table, the summary and the HTML writer are
plain functions (tests/test_score_borders.py).
"""

import argparse
import hashlib
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
MEMBER_EXTS = IMAGE_EXTS | {"json", "txt"}   # anything else is an error

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
# A side-strip candidate must cover >= 49% of one full side strip. Safe: any
# margin needs pass 1 of _measure to find one (later passes only refine
# pass 1's box), and pass 1's first test is a FULL outer row (top/bottom)
# or, when those fail, a FULL outer column; >= 98% canvas there covers
# >= 0.98 / RING_PX = 49% of that side's strip. A pruned candidate would
# have scored 0.
SIDE_MATCH_MIN = 0.98 / RING_PX
PEEL_PAD = 2              # px skipped inside a thin frame (antialiased edge)
PEEL_MIN_PX = 32          # no peel when the inside is smaller than this
NEAR_WHITE_MIN = 235      # bg_white: every channel >= this
NEAR_BLACK_MAX = 25       # bg_black: every channel <= this
SIDES = ("top", "bottom", "left", "right")
ROUND = 4

# --- selection / review ---
TOP_N = 1500
CONTROL_N = 150
THUMB_SIDE = 512
THUMB_QUALITY = 80
SHARD_THREADS = 2
SHARD_TIMEOUT_S = 20 * 60
# Resources. score_shard / thumb_shard run on Modal's default resources (no
# cpu/memory reservation): one synthetic 950-image shard (832x1216 WebP,
# 293 MB) peaked at ~90 MiB RSS with 2 threads; 2 threads use spare CPU when
# the host has it. prepare / finalize hold the whole metadata (load_meta of
# the real parquet: 716 MiB peak; the joined rows -> parquet: < 700 MiB), so
# they reserve 1 GiB.
META_MEM_MIB = 1024
# rough container time for the estimate: start + ~200 MB download + ~950
# images at ~25 ms (WebP decode 18 ms + metric) on 1-2 cores; pass 2
# re-downloads the tars holding selected ids without a thumbnail
EST_S_PER_SHARD = 60
EST_S_PER_THUMB_SHARD = 30
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
# side_pattern: which sides carry a band. A border = solid bands on 4 sides,
# 3 sides or 2 OPPOSITE sides; 2 adjacent sides, 1 side or none are not.
# A side counts when its margin is >= SIDE_MIN_PX pixels AND its straightness
# (non-canvas coverage of the first content lines on that side) is >=
# SIDE_STRAIGHT. Independent of MIN_SIDE_FRAC / n_sides (score only):
#  * px, not a fraction: a thin but real band (12 px of 1024, 1.2%) counts.
#    3 px: a true k px band on a WebP q80 canvas measures k-1 px (ringing
#    eats ~1 px; 3 px -> 1-2), plain full-bleed WebPs give 0 px, and the
#    real run's 1-12 px straight margins are flat (no noise spike at 1-2 px)
#  * 0.25: a backdrop panel behind a character covers only part of the
#    content box side (~0.7 when the head / legs stick out; 0.26-0.49 when
#    it is faint or occluded), a rounded-corner inset ~0.4; a round
#    silhouette tip gives ~0.1. A silhouette may reach 0.25 on one side (a
#    flat cut), but the border patterns need 3+ sides or an opposite pair
SIDE_MIN_PX = 3
SIDE_STRAIGHT = 0.25
SIDE_PATTERN_LABELS = {"4": "4면", "3": "3면", "2-opp": "2면(마주봄)",
                       "2-adj": "2면(이웃)", "1": "1면", "0": "0면"}
BORDER_PATTERNS = ("4", "3", "2-opp")
# The tab A selection rule, recorded in summary.json and on the page (the
# re-render derives its default cutoff from it).
A_RULE = {"name": "side-pattern", "version": 2, "patterns": list(BORDER_PATTERNS),
          "side_min_px": SIDE_MIN_PX, "side_straight": SIDE_STRAIGHT}
TABS = (
    ("A", "top by score, border sides (4 / 3 / 2 opposite)"),
    ("B", "sample watermark"),
    ("C", "border tag, no border caption"),
    ("D", "white background control"),
    ("E", "pillar / letterbox"),
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


class ShardError(RuntimeError):
    """A shard whose members or ids do not match the metadata."""


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


def _bg_candidates(rgb, tol=TOL, k=RING_PX):
    """-> [(source, colour uint8[3], ring_uniform)]: the medians of the whole
    ring and of each side strip; only identical colours are merged (two
    medians within tol can still give different masks, and only one of them
    may be the canvas). The whole ring always comes first; a side-strip
    median is skipped only when it covers < SIDE_MATCH_MIN of every side
    strip, which provably cannot give a margin (see SIDE_MATCH_MIN)."""
    import numpy as np

    h, w = rgb.shape[:2]
    k = max(1, min(k, h // 2, w // 2))
    sides = {"top": rgb[:k].reshape(-1, 3),
             "bottom": rgb[h - k:].reshape(-1, 3),
             "left": rgb[:, :k].reshape(-1, 3),
             "right": rgb[:, w - k:].reshape(-1, 3)}
    ring = np.concatenate([
        sides["top"], sides["bottom"],
        rgb[k:h - k, :k].reshape(-1, 3), rgb[k:h - k, w - k:].reshape(-1, 3)])
    ring16 = ring.astype(np.int16)
    sides16 = {s: px.astype(np.int16) for s, px in sides.items()}

    def cover(px16, cand):
        return float((np.abs(px16 - cand).max(axis=1) <= tol).mean())

    out = []
    for name, px in (("ring", ring), *sides.items()):
        cand = np.round(np.median(px, axis=0)).astype(np.int16)
        if any(np.array_equal(cand, c.astype(np.int16)) for _, c, _ in out):
            continue
        if name != "ring" and max(cover(sides16[s], cand)
                                  for s in SIDES) < SIDE_MATCH_MIN:
            continue
        out.append((name, cand.astype(np.uint8), cover(ring16, cand)))
    return out


def _lead(frac, thr=ROW_BG_FRAC):
    """Number of leading entries of frac that are >= thr."""
    import numpy as np

    bad = np.flatnonzero(frac < thr)
    return int(bad[0]) if bad.size else int(len(frac))


def _frac_bg(mask, axis):
    import numpy as np

    n = mask.shape[axis]
    return np.count_nonzero(mask, axis=axis) / max(n, 1)


def _measure(rgb, bg, tol=TOL):
    """Margin geometry of one image for one canvas colour -> dict with px
    margins (t, b, l, r), fill, per-side straightness of the bbox edges (all
    four sides). blank=True when no content bbox exists."""
    import numpy as np

    h, w = rgb.shape[:2]
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
    if blank:
        return {"px": (h, 0, 0, 0), "fill": 0.0, "blank": True,
                "straight": {s: 0.0 for s in SIDES}}
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
    return {"px": (t, b, l, r), "fill": fill, "blank": False,
            "straight": straight}


def _per_side(v, what):
    """Mapping keyed by SIDES or a (top, bottom, left, right) sequence ->
    list in SIDES order."""
    if hasattr(v, "keys"):
        return [v.get(s) for s in SIDES]
    v = list(v)
    if len(v) != 4:
        raise ValueError(f"want 4 {what} (t, b, l, r), got {v!r}")
    return v


def side_pattern(px, straight):
    """Pixel margins + per-side straightness -> "4" | "3" | "2-opp" |
    "2-adj" | "1" | "0": the sides with margin >= SIDE_MIN_PX px AND
    straightness >= SIDE_STRAIGHT (None / -1 = no margin). Both arguments:
    a mapping keyed by SIDES or a (top, bottom, left, right) sequence."""
    real = {s for s, p, st in zip(SIDES, _per_side(px, "margins"),
                                  _per_side(straight, "straightness"))
            if p is not None and st is not None and p >= SIDE_MIN_PX
            and st >= SIDE_STRAIGHT}
    if len(real) == 2:
        return ("2-opp" if real in ({"top", "bottom"}, {"left", "right"})
                else "2-adj")
    return str(len(real))


def _exact(v, n):
    """A count / n fraction stored rounded to ROUND digits -> the exact
    fraction (exact for n < 10**ROUND / 2)."""
    return round(v * n) / n if n else v


def row_px_straight(r):
    """Exact (px margins, straightness) of a score row from its rounded
    columns: px = round(margin * h|w); the straightness of top / bottom is a
    count over the content box width w - l - r, of left / right over its
    height h - t - b."""
    h, w = r["h"], r["w"]
    t, b, l, rr = (round((r[f"margin_{s}"] or 0) * (h if s in ("top", "bottom")
                                                    else w)) for s in SIDES)
    px = {"top": t, "bottom": b, "left": l, "right": rr}
    length = {"top": w - l - rr, "bottom": w - l - rr,
              "left": h - t - b, "right": h - t - b}
    st = {}
    for s in SIDES:
        v = r.get(f"straight_{s}")
        st[s] = None if v is None or v < 0 else _exact(v, length[s])
    return px, st


def row_side_pattern(r):
    """side_pattern of a score row: the stored column when present, else
    from the rounded margin / straightness columns (an old scores.parquet)
    via row_px_straight, i.e. what _finish computes unrounded."""
    sp = r.get("side_pattern")
    if sp:
        return sp
    return side_pattern(*row_px_straight(r))


def is_border_row(r):
    """Tab A eligibility: score > 0 and a border side pattern."""
    return (r.get("score") or 0) > 0 and row_side_pattern(r) in BORDER_PATTERNS


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
    if with_margin and not blank:   # geometric mean over the margin sides
        es = math.prod(max(0.0, straight[s]) for s in with_margin) ** (
            1.0 / len(with_margin))
    else:
        es = 0.0
    pillar = (not blank and real in ({"top", "bottom"}, {"left", "right"})
              and min(straight[s] for s in real) >= PILLAR_STRAIGHT)
    gate = ring_uniform >= RING_UNIFORM_MIN or pillar
    score = maf * es * min(1.0, fill / FILL_FULL) if gate and not blank else 0.0
    out = {f"margin_{s}": _r(frac[s]) for s in SIDES}
    out.update({
        "margin_area_frac": _r(maf), "n_sides": len(real),
        "side_pattern": side_pattern(
            px, [None if blank or frac[s] == 0 else straight[s]
                 for s in SIDES]),
        "fill": _r(fill),
        **{f"straight_{s}": (_r(straight[s]) if frac[s] > 0 else -1.0)
           for s in SIDES},
        "edge_straightness": _r(es), "pillar_or_letter": bool(pillar),
        "score": _r(score),
    })
    return out


def _outer(rgb, name, bg, ru, tol=TOL):
    """One canvas candidate through the full geometry -> (finished dict,
    measurement + {bg, ring_uniform, bg_src})."""
    h, w = rgb.shape[:2]
    m = _measure(rgb, bg, tol)
    m.update(bg=[int(x) for x in bg], ring_uniform=ru, bg_src=name)
    return _finish(h, w, m["px"], m["fill"], m["straight"], ru,
                   m["blank"]), m


def _best(rgb, tol=TOL):
    """Every canvas candidate through the full geometry (no peel) ->
    (finished dict, measurement, n candidates) of the best score (the first
    candidate, the whole-ring median, on ties). Used inside a frame."""
    cands = _bg_candidates(rgb, tol)
    best = None
    for name, bg, ru in cands:
        res, m = _outer(rgb, name, bg, ru, tol)
        if best is None or res["score"] > best[0]["score"]:
            best = (res, m)
    return best[0], best[1], len(cands)


def is_thin(px, blank, h, w):
    """All four margins present and < MIN_SIDE_FRAC: a frame line."""
    t, b, l, r = px
    return (not blank and min(px) > 0
            and t < MIN_SIDE_FRAC * h and b < MIN_SIDE_FRAC * h
            and l < MIN_SIDE_FRAC * w and r < MIN_SIDE_FRAC * w)


def _with_peel(rgb, res, m, tol=TOL):
    """One outer candidate's result, peeled when its margins are a thin
    frame and the inside (best candidate of the crop) scores higher ->
    (finished dict, info)."""
    h, w = rgb.shape[:2]
    thin = is_thin(m["px"], m["blank"], h, w)
    info = {"bg": m["bg"], "ring_uniform": m["ring_uniform"],
            "bg_src": m["bg_src"], "thin": thin, "peeled": False,
            "frame_px": 0, "blank": m["blank"]}
    if not thin:
        return res, info
    t, b, l, r = m["px"]
    y0, y1 = t + PEEL_PAD, h - b - PEEL_PAD
    x0, x1 = l + PEEL_PAD, w - r - PEEL_PAD
    if y1 - y0 < PEEL_MIN_PX or x1 - x0 < PEEL_MIN_PX:
        return res, info
    _, inner, _ = _best(rgb[y0:y1, x0:x1], tol)
    if inner["blank"]:
        return res, info
    it, ib, il, ir = inner["px"]
    px = (y0 + it, (h - y1) + ib, x0 + il, (w - x1) + ir)
    cand = _finish(h, w, px, inner["fill"], inner["straight"],
                   inner["ring_uniform"])
    if cand["score"] > res["score"]:
        info.update(bg=inner["bg"], ring_uniform=inner["ring_uniform"],
                    bg_src="peel:" + inner["bg_src"], peeled=True,
                    frame_px=max(m["px"]))
        return cand, info
    return res, info


def border_metrics(rgb, tol=TOL):
    """rgb: uint8 [H, W, 3] (full resolution, no downscale) -> dict of
    rounded floats / ints / bools (see the module docstring). Every outer
    canvas candidate is scored with and (when its margins are a thin frame)
    without its frame; the best final score wins (the first candidate on
    ties)."""
    import numpy as np

    rgb = np.ascontiguousarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 [H, W, 3], got {rgb.dtype} "
                         f"{rgb.shape}")
    h, w = rgb.shape[:2]
    cands = _bg_candidates(rgb, tol)
    best = None
    for name, bg, ru in cands:
        res, info = _with_peel(rgb, *_outer(rgb, name, bg, ru, tol), tol)
        if best is None or res["score"] > best[0]["score"]:
            best = (res, info)
    res, info = best
    bg = info["bg"]
    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    res.update({
        "h": h, "w": w, "bg_r": bg[0], "bg_g": bg[1], "bg_b": bg[2],
        "bg_lum": _r(lum), "bg_white": bool(min(bg) >= NEAR_WHITE_MIN),
        "bg_black": bool(max(bg) <= NEAR_BLACK_MAX),
        "ring_uniform": _r(info["ring_uniform"]), "bg_src": info["bg_src"],
        "n_bg_candidates": len(cands), "thin_frame": bool(info["thin"]),
        "peeled": bool(info["peeled"]), "frame_px": int(info["frame_px"]),
        "blank": bool(info["blank"]),
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


def parse_member(name):
    """WebDataset member name -> (id, ext): key = basename up to the first
    dot, digits only; ext (lower case) in MEMBER_EXTS. Anything else raises
    ShardError."""
    base = Path(name).name
    stem, dot, ext = base.partition(".")
    ext = ext.lower()
    if not dot or not stem.isdigit() or ext not in MEMBER_EXTS:
        raise ShardError(f"unexpected tar member {name!r} (want <id>."
                         f"{{{','.join(sorted(MEMBER_EXTS))}}})")
    return int(stem), ext


def iter_tar_images(tar_path):
    """Yield (id, encoded bytes) of every image member; json/txt members are
    skipped unread, directories ignored, any other member raises."""
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if m.isdir():
                continue
            if not m.isfile():
                raise ShardError(f"unexpected tar member {m.name!r} (not a "
                                 f"regular file)")
            sid, ext = parse_member(m.name)
            if ext in IMAGE_EXTS:
                yield sid, tf.extractfile(m).read()


def validate_shard(ids, expected):
    """Image ids read from a tar vs the ids the parquet lists for it ->
    {"missing", "extra", "duplicate"} (sorted lists; only non-empty keys;
    {} = ok)."""
    counts = Counter(ids)
    expected = set(expected)
    out = {"missing": sorted(expected - set(counts)),
           "extra": sorted(set(counts) - expected),
           "duplicate": sorted(i for i, n in counts.items() if n > 1)}
    return {k: v for k, v in out.items() if v}


def problem_text(problems, n=5):
    return "; ".join(f"{len(v)} {k} ids {v[:n]}" for k, v in problems.items())


def score_shard_impl(tar_path, thumb_ids=(), threads=SHARD_THREADS,
                     log=print):
    """Score every image of one tar -> {"rows": [{id, **metrics} or {id,
    error}], "thumbs": {id: jpeg bytes} for the thumb_ids found, "seconds",
    "n"}. A malformed member raises ShardError."""
    t0 = time.time()
    thumb_ids = {int(i) for i in thumb_ids}

    def one(item):
        sid, data = item
        thumb = None
        try:
            rgb = decode_rgb(data)
            row = {"id": sid, **border_metrics(rgb)}
            if sid in thumb_ids:
                thumb = thumb_jpeg(rgb)
            return row, thumb
        except Exception as e:   # a broken image is an error row, not a lost shard
            return {"id": sid, "error": f"{type(e).__name__}: {e}"[:300]}, None

    threads = max(1, threads)
    rows, thumbs, pending = [], {}, deque()

    def take():
        row, thumb = pending.popleft().result()
        rows.append(row)
        if thumb is not None:
            thumbs[row["id"]] = thumb

    with ThreadPoolExecutor(threads) as pool:
        # bounded look-ahead: only a few encoded images in memory at a time
        # (Executor.map would read the whole ~200 MB tar up front)
        for item in iter_tar_images(tar_path):
            pending.append(pool.submit(one, item))
            if len(pending) >= 4 * threads:
                take()
        while pending:
            take()
    dt = time.time() - t0
    n_err = sum("error" in r for r in rows)
    log(f"{Path(tar_path).name}: {len(rows)} imgs, {n_err} errors, "
        f"{len(thumbs)} thumbs in {dt:.1f}s")
    return {"rows": rows, "thumbs": thumbs, "n": len(rows),
            "seconds": {"score": round(dt, 2)}}


def thumb_shard_impl(tar_path, ids, log=print):
    """Pass 2: thumbnails of exactly `ids` (those present in the tar)."""
    t0 = time.time()
    ids = {int(i) for i in ids}
    thumbs = {}
    for sid, data in iter_tar_images(tar_path):
        if sid in ids:
            thumbs[sid] = thumb_jpeg(decode_rgb(data))
    dt = time.time() - t0
    log(f"{Path(tar_path).name}: {len(thumbs)}/{len(ids)} thumbs in {dt:.1f}s")
    return {"thumbs": thumbs, "seconds": {"thumbs": round(dt, 2)}}


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
    (plain Python values). Duplicate ids raise."""
    import pyarrow.parquet as pq

    cols = ["id", "long", "tags", "bucket_w", "bucket_h", "tar"]
    t = pq.read_table(parquet_path, columns=cols).to_pydict()
    meta = {}
    for i, sid in enumerate(t["id"]):
        sid = int(sid)
        if sid in meta:
            raise ValueError(f"{parquet_path}: duplicate id {sid}")
        long_cap = t["long"][i] or ""
        meta[sid] = {
            "tar": t["tar"][i], "bucket_w": int(t["bucket_w"][i]),
            "bucket_h": int(t["bucket_h"][i]),
            **flags_of(t["tags"][i], long_cap),
            "cap": long_cap[:SNIPPET_CHARS],
        }
    return meta


def plan_job(meta, seed=0, control_n=CONTROL_N):
    """-> {"shards": {tar: n ids}, "ids": {tar: [ids]}, "always": {tar:
    [ids]}, "control", "watermark", "uncaptioned": [ids]} (sorted ids)."""
    watermark = sorted(i for i, m in meta.items() if m["tag_sample_watermark"])
    unc = sorted(i for i, m in meta.items() if is_border_uncaptioned(m))
    white = sorted(i for i, m in meta.items() if m["tag_white_background"])
    control = sorted(random.Random(seed).sample(white, min(control_n,
                                                           len(white))))
    always = {}
    for i in sorted(set(watermark) | set(unc) | set(control)):
        always.setdefault(meta[i]["tar"], []).append(i)
    ids = {}
    for i in sorted(meta):
        ids.setdefault(meta[i]["tar"], []).append(i)
    ids = dict(sorted(ids.items()))
    return {"shards": {t: len(v) for t, v in ids.items()}, "ids": ids,
            "always": always, "control": control, "watermark": watermark,
            "uncaptioned": unc}


def restrict_plan(plan, meta, tars):
    """The plan limited to `tars` (--shards N): review lists keep only ids
    of those tars."""
    done = set(tars)
    p = dict(plan)
    p["shards"] = {t: n for t, n in plan["shards"].items() if t in done}
    p["ids"] = {t: v for t, v in plan["ids"].items() if t in done}
    p["always"] = {t: v for t, v in plan["always"].items() if t in done}
    for k in ("control", "watermark", "uncaptioned"):
        p[k] = [i for i in plan[k] if meta[i]["tar"] in done]
    return p


def check_shard_result(tar, res, plan, log=print):
    """Validate one pass-1 result against the plan -> error text or None."""
    problems = validate_shard([r["id"] for r in res["rows"]],
                              plan["ids"].get(tar, []))
    if problems:
        msg = f"{tar}: ids do not match metadata.parquet: " + problem_text(
            problems)
        log(f"  FAILED {msg}")
        return msg
    return None


# ------------------------------------------------------------------ selection
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


def build_tabs(joined, plan, top_n=TOP_N):
    """joined: sorted by join_rows. A = the first top_n rows with score > 0
    and a border side pattern (4 / 3 / 2 opposite sides)."""
    ok = [r for r in joined if "error" not in r]
    eligible = [r for r in ok if is_border_row(r)]
    return {
        "A": [r["id"] for r in eligible[:top_n]],
        "B": list(plan["watermark"]),
        "C": list(plan["uncaptioned"]),
        "D": list(plan["control"]),
        "E": [r["id"] for r in ok if r.get("pillar_or_letter")],
    }


def a_rule_record(joined, a_ids, top_n):
    """The rule that produced tab A + its score-rank cutoff: every eligible
    row among the first `rank_cutoff` rows with score > 0 (joined order) is
    in A (A was not cut short by top_n beyond that rank)."""
    positives = [r["id"] for r in joined if "error" not in r
                 and (r.get("score") or 0) > 0]
    if len(a_ids) < top_n or not a_ids:
        cutoff = len(positives) if len(a_ids) < top_n else 0
    else:
        cutoff = positives.index(a_ids[-1]) + 1
    return {**A_RULE, "top_n": top_n, "rank_cutoff": cutoff}


def select_thumbs(tabs):
    """Global thumbnail selection = every id on the review page (top-N by
    score, pillar/letterbox, the always-thumb lists)."""
    return {i for ids in tabs.values() for i in ids}


def thumb_jobs(selected, have, meta):
    """Selected ids without a thumbnail, grouped by tar -> {tar: [ids]}."""
    jobs = {}
    for i in sorted(set(selected) - set(have)):
        jobs.setdefault(meta[i]["tar"], []).append(i)
    return dict(sorted(jobs.items()))


def check_thumbs(selected, have):
    """Raise SystemExit when a selected id has no thumbnail."""
    missing = sorted(set(selected) - set(have))
    if missing:
        raise SystemExit(f"{len(missing)} selected ids have no thumbnail "
                         f"after pass 2: {missing[:20]}")


# ------------------------------------------------------------------ outputs
def table_bytes(joined, revision=None):
    """Joined rows -> scores.parquet bytes (pyarrow; columns = the union of
    the row keys, missing values null; schema metadata dataset_repo /
    dataset_revision)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys = []
    for r in joined:
        for k in r:
            if k not in keys:
                keys.append(k)
    t = pa.Table.from_pydict({k: [r.get(k) for r in joined] for k in keys})
    t = t.replace_schema_metadata({"dataset_repo": DATASET_REPO,
                                   "dataset_revision": str(revision)})
    buf = io.BytesIO()
    pq.write_table(t, buf, compression="zstd")
    return buf.getvalue()


def quantiles(xs, qs=(0.5, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 1.0)):
    xs = sorted(xs)
    if not xs:
        return {}
    return {f"p{q * 100:g}": xs[min(len(xs) - 1,
                                    int(math.ceil(q * len(xs))) - 1)]
            for q in qs}


def pattern_counts(rows):
    """{side pattern: n} in SIDE_PATTERN_LABELS order (all keys)."""
    c = Counter(row_side_pattern(r) for r in rows)
    return {k: c.get(k, 0) for k in SIDE_PATTERN_LABELS}


def summarize(joined, tabs, have_thumbs, plan, seconds, revision=None):
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
    overlap["any border tag"] = {"tagged": len(anyb),
                                 "in_top": len(anyb & top)}
    return {
        "dataset_repo": DATASET_REPO, "dataset_revision": revision,
        "images": len(joined), "errors": len(joined) - len(ok),
        "error_ids": sorted(r["id"] for r in joined if "error" in r)[:50],
        "expected": sum(plan["shards"].values()),
        "shards": len(plan["shards"]),
        "score_quantiles": quantiles(scores),
        "score_gt0": sum(s > 0 for s in scores),
        "side_patterns_score_gt0": pattern_counts(
            r for r in ok if r["score"] > 0),
        "side_patterns_A": pattern_counts(
            r for r in ok if r["id"] in top),
        "pillar_or_letter": sum(bool(r.get("pillar_or_letter")) for r in ok),
        "thin_frame": sum(bool(r.get("thin_frame")) for r in ok),
        "peeled": sum(bool(r.get("peeled")) for r in ok),
        "bg_src": dict(Counter(r.get("bg_src") for r in ok)),
        "tabs": {k: len(v) for k, v in tabs.items()},
        "thumbs_selected": len(select_thumbs(tabs)),
        "missing_thumbs": {k: sum(i not in have_thumbs for i in v)
                           for k, v in tabs.items()},
        "top_overlap": overlap, "seconds": seconds,
    }


def _fmt_counts(c):
    return ", ".join(f"{k} {v}" for k, v in c.items())


def print_summary(s, log=print):
    log(f"dataset {s['dataset_repo']}@{s['dataset_revision']}")
    log(f"images scored: {s['images']} (expected {s['expected']}, "
        f"{s['shards']} shards), errors {s['errors']}"
        + (f" {s['error_ids'][:10]}" if s["errors"] else ""))
    log("score quantiles: " + ", ".join(f"{k} {v:.4f}" for k, v in
                                        s["score_quantiles"].items()))
    log(f"score > 0: {s['score_gt0']}, pillar_or_letter: "
        f"{s['pillar_or_letter']}, thin_frame: {s['thin_frame']}, peeled: "
        f"{s['peeled']}, canvas colour from: {s['bg_src']}")
    log("side patterns (score > 0): " + _fmt_counts(s["side_patterns_score_gt0"])
        + "; tab A: " + _fmt_counts(s["side_patterns_A"]))
    log("tabs: " + ", ".join(f"{k} {n} (no thumb {s['missing_thumbs'][k]})"
                             for k, n in s["tabs"].items())
        + f"; {s['thumbs_selected']} thumbnails selected")
    log("top-N overlap with tags: " + ", ".join(
        f"{t} {v['in_top']}/{v['tagged']}" for t, v in s["top_overlap"].items()))
    log("runtime: " + ", ".join(f"{k} {v:.1f}s" for k, v in
                                s["seconds"].items()))


def review_items(joined, tabs, meta, have_thumbs):
    """Compact per-id records for every id in a tab."""
    by_id = {r["id"]: r for r in joined}
    items = {}
    for ids in tabs.values():
        for i in ids:
            if str(i) in items:
                continue
            r = by_id.get(i, {})
            m = meta.get(i, {})
            items[str(i)] = {
                "s": r.get("score"), "err": r.get("error"),
                "m": [r.get(f"margin_{s}") for s in SIDES],
                "st": r.get("edge_straightness"), "f": r.get("fill"),
                "pl": bool(r.get("pillar_or_letter")),
                "sp": (row_side_pattern(r) if r and "error" not in r
                       else None),
                "tg": tags_hit(m) if m else [],
                "cb": bool(m.get("caption_mentions_border")),
                "cap": m.get("cap", ""), "th": i in have_thumbs,
            }
    return items


def review_id(tabs, generated, revision=None):
    """Identity of one review page: hash of its tab id lists, generation
    time and dataset revision (scopes the localStorage selection and the
    export)."""
    blob = json.dumps({"tabs": tabs, "generated": generated,
                       "revision": revision},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def render_review(tabs, items, generated, title="datafilter review", info="",
                  revision=None, a_rule=None):
    """Self-contained review page (inline JSON, thumbs at ../thumbs/<id>.jpg)."""
    data = json.dumps({"tabs": tabs, "items": items, "names": dict(TABS),
                       "patterns": SIDE_PATTERN_LABELS,
                       "border_patterns": list(BORDER_PATTERNS),
                       "review_id": review_id(tabs, generated, revision),
                       "generated": generated, "dataset_repo": DATASET_REPO,
                       "revision": revision, "a_rule": a_rule},
                      separators=(",", ":"))
    data = data.replace("</", "<\\/")
    return (REVIEW_HTML.replace("__TITLE__", html.escape(title))
            .replace("__INFO__", html.escape(info))
            .replace("__LOGIC__", REVIEW_LOGIC_JS)
            .replace("__DATA__", data))


# Pure state logic (no DOM; tests/test_score_borders.py runs it in node).
# A selection is {id: state} with state in STATES minus "keep" (keep = absent).
REVIEW_LOGIC_JS = """
var STORE_PREFIX='datafilter_review_v2:';
var STATES=['keep','caption','crop','remove'];
var STATE_LABELS={keep:'유지',caption:'캡션 보완',crop:'자르기',remove:'제거'};
function own(o,k){return Object.prototype.hasOwnProperty.call(o,k)}
function storeKey(D){return STORE_PREFIX+D.review_id}
function marked(s){return typeof s==='string'&&STATES.indexOf(s)>0}
function nextState(s){var i=STATES.indexOf(s);if(i<0)i=0;
return STATES[(i+1)%STATES.length]}
function loadSel(raw,D){var sel,out={};
try{sel=JSON.parse(raw||'{}')}catch(e){return out}
if(!sel||typeof sel!=='object'||Array.isArray(sel))return out;
Object.keys(sel).forEach(function(id){
if(own(D.items,id)&&marked(sel[id]))out[id]=sel[id]});return out}
function setState(sel,id,s){id=String(id);
if(marked(s))sel[id]=s;else delete sel[id];return sel}
function cycle(sel,id){var s=nextState(own(sel,String(id))?sel[String(id)]:'keep');
setState(sel,id,s);return s}
function stateCounts(sel,D){var c={};STATES.forEach(function(s){c[s]=0});
Object.keys(D.items).forEach(function(id){
c[own(sel,id)&&marked(sel[id])?sel[id]:'keep']++});return c}
function tabsOf(id,D){return Object.keys(D.tabs).filter(function(k){
return D.tabs[k].indexOf(+id)>=0})}
function exportObj(sel,D){return {review_id:D.review_id,
generated:D.generated,dataset_revision:D.revision,
items:Object.keys(sel).filter(function(id){
return own(D.items,id)&&marked(sel[id])}).sort(function(a,b){return a-b})
.map(function(id){var it=D.items[id];return {id:+id,action:sel[id],
tabs:tabsOf(id,D),score:it.s==null?null:it.s,
side_pattern:it.sp==null?null:it.sp}})}}
"""

REVIEW_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fafafa;--fg:#1b1b1b;--muted:#666;--card:#fff;--line:#ddd;
--accent:#2563eb;--caption:#d97706;--crop:#2563eb;--remove:#dc2626;
--badge:#e8eefc;--badge-off:#eee}
@media (prefers-color-scheme: dark){:root{--bg:#141414;--fg:#e8e8e8;
--muted:#9a9a9a;--card:#1f1f1f;--line:#333;--accent:#60a5fa;--caption:#fbbf24;
--crop:#60a5fa;--remove:#f87171;--badge:#1e2a44;--badge-off:#2a2a2a}}
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
#legend{display:flex;flex-wrap:wrap;gap:10px;color:var(--muted);font-size:13px}
#legend span{display:inline-flex;align-items:center;gap:4px;
font-variant-numeric:tabular-nums}
#legend i{display:inline-block;width:12px;height:12px;border-radius:3px;
border:2px solid var(--line)}
#legend i.caption{border-color:var(--caption)}
#legend i.crop{border-color:var(--crop)}
#legend i.remove{border-color:var(--remove)}
button.act{border:1px solid var(--line);background:var(--card);color:var(--fg);
padding:5px 10px;border-radius:6px;cursor:pointer}
#info{color:var(--muted);font-size:12px;padding:6px 16px 0}
main{padding:12px 16px;display:grid;gap:10px;
grid-template-columns:repeat(auto-fill,minmax(180px,1fr))}
.c{background:var(--card);border:3px solid transparent;border-radius:8px;
padding:4px;cursor:pointer;outline:1px solid var(--line);position:relative}
.c.st-caption{border-color:var(--caption);outline-color:var(--caption)}
.c.st-crop{border-color:var(--crop);outline-color:var(--crop)}
.c.st-remove{border-color:var(--remove);outline-color:var(--remove)}
.c .stl{position:absolute;top:8px;left:8px;padding:1px 6px;border-radius:4px;
font-size:12px;font-weight:600;color:#fff;display:none}
.c.st-caption .stl{display:block;background:var(--caption)}
.c.st-crop .stl{display:block;background:var(--crop)}
.c.st-remove .stl{display:block;background:var(--remove)}
.c img{width:100%;height:220px;object-fit:contain;display:block;
background:repeating-conic-gradient(#8882 0 25%,transparent 0 50%) 0 0/16px 16px}
.c .ph{height:220px;display:flex;align-items:center;justify-content:center;
color:var(--muted)}
.c .t{font-size:12px;margin-top:4px;word-break:break-word}
.c .t b{font-variant-numeric:tabular-nums}
.c .g{color:var(--muted);font-size:11px}
.pt{display:inline-block;margin-left:4px;padding:0 5px;border-radius:4px;
font-size:11px;background:var(--badge);color:var(--accent)}
.pt.off{background:var(--badge-off);color:var(--muted)}
</style></head><body>
<header><h1>__TITLE__</h1><span id="tabs"></span><span class="sp"></span>
<span id="legend"></span>
<button class="act" id="clear">clear tab</button>
<button class="act" id="export">export</button></header>
<div id="info">__INFO__ · 클릭: 유지 → 캡션 보완 → 자르기 → 제거 → 유지</div>
<main id="grid"></main>
<script id="data" type="application/json">__DATA__</script>
<script id="logic">__LOGIC__</script>
<script>
(function(){
var D=JSON.parse(document.getElementById('data').textContent);
var KEY=storeKey(D), sel={}, cur='A';
try{sel=loadSel(localStorage.getItem(KEY),D)}catch(e){sel={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(sel))}catch(e){}}
function fmt(x){return x==null?'-':(+x).toFixed(3)}
function pct(x){return x==null?'-':Math.round(x*1000)/10}
function legend(){var el=document.getElementById('legend'),c=stateCounts(sel,D);
el.innerHTML='';STATES.forEach(function(s){var sp=document.createElement('span');
var i=document.createElement('i');i.className=s;sp.appendChild(i);
sp.appendChild(document.createTextNode(STATE_LABELS[s]+' '+c[s]));
el.appendChild(sp)})}
function tabs(){var el=document.getElementById('tabs');el.innerHTML='';
Object.keys(D.tabs).forEach(function(k){var b=document.createElement('button');
b.className='tab'+(k===cur?' on':'');
b.textContent=k+': '+(D.names[k]||'')+' ('+D.tabs[k].length+')';
b.onclick=function(){cur=k;tabs();grid()};el.appendChild(b)})}
function paint(c,id){var s=own(sel,String(id))?sel[String(id)]:'keep';
c.className='c'+(s!=='keep'?' st-'+s:'');
c.querySelector('.stl').textContent=STATE_LABELS[s]}
function grid(){var g=document.getElementById('grid');g.innerHTML='';
var f=document.createDocumentFragment();
D.tabs[cur].forEach(function(id){var it=D.items[String(id)]||{};
var c=document.createElement('div');
var m=it.m||[], pn=it.sp==null?'-':(D.patterns[it.sp]||it.sp);
c.title=(it.cap||'')+'\\n\\ntags: '+((it.tg||[]).join(', ')||'-')+
'\\nmargins t/b/l/r %: '+m.map(pct).join(' / ')+' ('+pn+')'+
'\\nstraightness '+fmt(it.st)+', fill '+fmt(it.f)+
(it.pl?'\\npillar/letterbox':'')+(it.cb?'\\ncaption mentions border':'')+
(it.err?'\\nerror: '+it.err:'');
var stl=document.createElement('div');stl.className='stl';c.appendChild(stl);
if(it.th){var im=document.createElement('img');im.loading='lazy';
im.src='../thumbs/'+id+'.jpg';im.alt=String(id);c.appendChild(im)}
else{var p=document.createElement('div');p.className='ph';
p.textContent='no thumbnail';c.appendChild(p)}
var t=document.createElement('div');t.className='t';
t.innerHTML='<b>'+id+'</b> score <b>'+fmt(it.s)+'</b>'+(it.pl?' PL':'');
var pt=document.createElement('span');
pt.className='pt'+(D.border_patterns.indexOf(it.sp)<0?' off':'');
pt.textContent=pn;t.appendChild(pt);
var gg=document.createElement('div');gg.className='g';
gg.textContent='m '+m.map(pct).join('/')+(it.tg&&it.tg.length?' | '+it.tg.join(', '):'');
t.appendChild(gg);c.appendChild(t);paint(c,id);
c.onclick=function(){cycle(sel,id);paint(c,id);save();legend()};
f.appendChild(c)});g.appendChild(f);legend()}
document.getElementById('clear').onclick=function(){
if(!confirm('tab '+cur+': reset every image to keep?'))return;
D.tabs[cur].forEach(function(id){setState(sel,id,'keep')});save();grid()};
document.getElementById('export').onclick=function(){
var out=exportObj(sel,D);
var a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));
a.download='datafilter_actions.json';document.body.appendChild(a);a.click();
setTimeout(function(){URL.revokeObjectURL(a.href);a.remove()},1000)};
tabs();grid();
})();
</script></body></html>
"""


def write_outputs(out_dir, rows, thumbs, meta, plan, seconds, top_n=TOP_N,
                  table=None, info="", generated=None, revision=None):
    """Write thumbs/<id>.jpg, scores.parquet, review/index.html and
    summary.json under out_dir. `thumbs` must cover select_thumbs(tabs)
    (check_thumbs). `table`: scores.parquet bytes made elsewhere (the
    `finalize` container), else built here with pyarrow. -> summary."""
    out_dir = Path(out_dir)
    joined = join_rows(rows, meta)
    tabs = build_tabs(joined, plan, top_n)
    check_thumbs(select_thumbs(tabs), thumbs)
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    (out_dir / "review").mkdir(parents=True, exist_ok=True)
    for sid, jpg in thumbs.items():
        (out_dir / "thumbs" / f"{int(sid)}.jpg").write_bytes(jpg)
    have = {int(i) for i in thumbs}
    if table is None:
        table = table_bytes(joined, revision)
    (out_dir / "scores.parquet").write_bytes(table)
    items = review_items(joined, tabs, meta, have)
    s = summarize(joined, tabs, have, plan, seconds, revision)
    s["a_rule"] = a_rule_record(joined, tabs["A"], top_n)
    generated = generated or datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    info = info or (f"{DATASET_REPO}@{revision}: {s['images']} images, "
                    f"generated {generated}, score quantiles "
                    + ", ".join(f"{k} {v:.3f}" for k, v in
                                s["score_quantiles"].items()))
    (out_dir / "review" / "index.html").write_text(
        render_review(tabs, items, generated, info=info, revision=revision,
                      a_rule=s["a_rule"]), encoding="utf-8")
    s["review_id"] = review_id(tabs, generated, revision)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1))
    return s


# ------------------------------------------------------------------ re-render
DATA_RE = re.compile(
    r'<script id="data" type="application/json">(.*?)</script>', re.S)


def page_data(path):
    """Inline JSON of a review page; None when the file does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    m = DATA_RE.search(p.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"{p}: no inline review data")
    return json.loads(m.group(1))


def _ids_text(ids, n=50):
    ids = sorted(ids)
    return f"{ids[:n]}" + (f" ... (+{len(ids) - n})" if len(ids) > n else "")


def load_score_rows(path):
    """scores.parquet -> (rows sorted like join_rows, schema metadata).
    A null `error` is dropped (error rows keep theirs)."""
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    rows = t.to_pylist()
    for r in rows:
        if r.get("error") is None:
            r.pop("error", None)
    ids = Counter(r["id"] for r in rows)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        raise SystemExit(f"{path}: {len(dup)} duplicate ids {_ids_text(dup)}")
    rows.sort(key=lambda r: (-(r.get("score") or 0.0), r["id"]))
    return rows, md


def rerender(out_dir, top_n=None, seed=0, control_n=CONTROL_N, generated=None,
             log=print):
    """Rebuild <out_dir>/review/index.html from a finished run's outputs
    (no Modal, no tars): scores.parquet (side_pattern from the margins when
    the column is missing), summary.json (dataset revision, the run's tab A
    size), thumbs/ and the previous page(s) (caption snippets, the control
    list D). B / C / E come from the parquet flags; D from the previous page
    (else re-drawn with seed / control_n); every list on a previous page
    must equal its re-derived one. Side patterns are recomputed under the
    current rule. Tab A = the top top_n rows with score > 0 and a border
    side pattern. Default top_n: every eligible row within the run's
    score-rank cutoff, from the run's recorded `a_rule` in summary.json (a
    legacy run without one: its unfiltered top-N, all thumbnailed). The
    membership is decided from scores / patterns / cutoff alone; then any
    tab id without thumbs/<id>.jpg fails (count + ids). The rule used is
    recorded on the page (`a_rule`). A v1 page is backed up to review/index.v1.html first
    (never overwritten). Writes only review/index.html (+ that backup)."""
    out_dir = Path(out_dir)
    rdir = out_dir / "review"
    summary = json.loads((out_dir / "summary.json").read_text())
    rows, md = load_score_rows(out_dir / "scores.parquet")
    revision = summary.get("dataset_revision")
    if md.get("dataset_revision") not in (None, str(revision)):
        raise SystemExit(f"scores.parquet revision {md['dataset_revision']} "
                         f"!= summary.json {revision}")
    ok = [r for r in rows if "error" not in r]
    # side patterns always from the columns under the CURRENT rule (a
    # stored column may come from another rule version)
    stale = 0
    for r in rows:
        sp = r.pop("side_pattern", None)
        if sp is not None and "error" not in r and sp != row_side_pattern(r):
            stale += 1
    if stale:
        log(f"{stale} stored side_pattern values differ from the current "
            f"rule {A_RULE}; recomputed")
    # previous pages (the backup first); a v1 page is backed up before it
    # is replaced (below, after every check)
    cur, v1 = rdir / "index.html", rdir / "index.v1.html"
    cur_data = page_data(cur)
    backup = cur_data is not None and "patterns" not in cur_data
    if backup and v1.exists() and v1.read_bytes() != cur.read_bytes():
        raise SystemExit(f"{cur} is a v1 page and {v1} exists with other "
                         "content; move one of them away")
    pages = [(p, d) for p, d in ((v1, page_data(v1)), (cur, cur_data))
             if d is not None]
    # tab lists B / C / D / E
    plan = {
        "watermark": sorted(r["id"] for r in rows if r.get("tag_sample_watermark")),
        "uncaptioned": sorted(r["id"] for r in rows if r.get("border_uncaptioned")),
    }
    if pages:
        plan["control"] = list(pages[0][1]["tabs"]["D"])
        d_src = f"{pages[0][0].name} (tab D)"
    else:
        white = sorted(r["id"] for r in rows if r.get("tag_white_background"))
        plan["control"] = sorted(random.Random(seed).sample(
            white, min(control_n, len(white))))
        d_src = f"re-drawn (seed {seed}, n {control_n})"
    by_id = {r["id"]: r for r in rows}
    unknown = [i for i in plan["control"] if i not in by_id]
    if unknown:
        raise SystemExit(f"{len(unknown)} control ids not in scores.parquet: "
                         f"{_ids_text(unknown)}")
    # the run's own tab A rule -> the score-rank cutoff above which its
    # selection was thumbnailed
    positives = [r["id"] for r in ok if r["score"] > 0]
    run_rule = summary.get("a_rule")
    if run_rule is None:      # legacy run: unfiltered top-N by score
        run_rule = {"name": "unfiltered-top-n", "version": 1,
                    "top_n": int(summary["tabs"]["A"]),
                    "rank_cutoff": int(summary["tabs"]["A"])}
    cutoff = int(run_rule["rank_cutoff"])
    old_a = positives[:cutoff]
    full = build_tabs(rows, plan, top_n=0)
    for p, d in pages:
        want = {"B": full["B"], "C": full["C"], "D": plan["control"],
                "E": full["E"]}
        if "patterns" not in d and run_rule["version"] == 1:
            want["A"] = old_a            # v1 page: unfiltered top-N by score
        diff = [k for k, v in want.items() if list(d["tabs"].get(k, [])) != v]
        if diff or d.get("revision") != revision:
            raise SystemExit(f"{p}: tabs {diff or ''} / revision "
                             f"{d.get('revision')} disagree with scores.parquet"
                             f" + summary.json ({revision})")
    for k in ("B", "C", "D", "E"):
        n = len(full[k]) if k != "D" else len(plan["control"])
        if summary["tabs"].get(k) != n:
            raise SystemExit(f"tab {k}: {n} ids, summary.json says "
                             f"{summary['tabs'].get(k)}")
    have = {int(p.stem) for p in (out_dir / "thumbs").glob("*.jpg")
            if p.stem.isdigit()}
    # intended tab A membership, independent of which thumbnails exist
    old_set = set(old_a)
    eligible = [r["id"] for r in ok if is_border_row(r)]
    exact_n = sum(i in old_set for i in eligible)   # a prefix of eligible
    if top_n is None:
        top_n = exact_n
        log(f"default --top-n {top_n}: the eligible rows (score > 0, side "
            f"pattern {'/'.join(BORDER_PATTERNS)}, >= {SIDE_MIN_PX} px and "
            f"straightness >= {SIDE_STRAIGHT}) within the run's score-rank "
            f"cutoff {cutoff} ({run_rule['name']} v{run_rule['version']}; "
            f"{len(eligible)} eligible in total)")
    if top_n < 0:
        raise SystemExit("--top-n must be >= 0")
    tabs = build_tabs(rows, plan, top_n)
    a_rule = a_rule_record(rows, tabs["A"], top_n)
    if top_n == exact_n:   # A = every eligible row within the run's cutoff
        a_rule["rank_cutoff"] = max(a_rule["rank_cutoff"], cutoff)
    a_rule["source"] = "re-render"
    a_rule["run_rule"] = run_rule
    missing = {k: [i for i in v if i not in have] for k, v in tabs.items()}
    n_miss = sum(map(len, missing.values()))
    if n_miss:
        raise SystemExit(f"{n_miss} tab ids have no thumbnail in "
                         f"{out_dir / 'thumbs'}, nothing written: " + "; ".join(
                             f"{k} {len(v)} {_ids_text(v)}"
                             for k, v in missing.items() if v))
    caps = {}
    for _, d in pages:
        for i, it in d.get("items", {}).items():
            caps.setdefault(int(i), it.get("cap") or "")
    sel = select_thumbs(tabs)
    meta = {}
    for i in sel:
        r = by_id[i]
        if r.get("tar") is None:
            continue
        meta[i] = {k: v for k, v in r.items()
                   if k.startswith(("tag_", "caption_mentions_"))}
        meta[i]["cap"] = caps.get(i, "")
    no_cap = sorted(i for i in sel if i not in caps)
    items = review_items(rows, tabs, meta, have)
    generated = generated or datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    run_gen = pages[0][1].get("generated") if pages else None
    info = (f"{DATASET_REPO}@{revision}: {len(rows)} images; re-rendered "
            f"{generated} from scores.parquet (run {run_gen or 'unknown'}); "
            f"tab A = top {top_n} with score > 0 and side pattern "
            f"{' / '.join(SIDE_PATTERN_LABELS[p] for p in BORDER_PATTERNS)}")
    page = render_review(tabs, items, generated, info=info, revision=revision,
                         a_rule=a_rule)
    rdir.mkdir(parents=True, exist_ok=True)
    if backup and not v1.exists():
        v1.write_bytes(cur.read_bytes())
        log(f"backed up the v1 page to {v1}")
    cur.write_text(page, encoding="utf-8")
    res = {
        "review_id": review_id(tabs, generated, revision),
        "generated": generated, "dataset_revision": revision,
        "top_n": top_n, "exact_default_top_n": exact_n,
        "eligible": len(eligible), "rank_cutoff": cutoff,
        "run_rule": run_rule, "a_rule": a_rule,
        "tabs": {k: len(v) for k, v in tabs.items()},
        "side_patterns_A": pattern_counts(by_id[i] for i in tabs["A"]),
        "side_patterns_old_A": pattern_counts(by_id[i] for i in old_a),
        "thumbs_used": len(sel), "no_caption_snippet": len(no_cap),
        "control_from": d_src, "pages_read": [str(p) for p, _ in pages],
    }
    log(f"control list D from {d_src}; caption snippets from "
        f"{[p.name for p, _ in pages] or 'nowhere'} ({len(no_cap)} ids "
        "without one)")
    log("tabs: " + ", ".join(f"{k} {n}" for k, n in res["tabs"].items())
        + f"; {len(sel)} thumbnails")
    log("side patterns, tab A: " + _fmt_counts(res["side_patterns_A"])
        + f"; the run's top {cutoff} by score: "
        + _fmt_counts(res["side_patterns_old_A"]))
    log(f"wrote {cur} (review id {res['review_id']})")
    return res


# ------------------------------------------------------------------ HF / env
def pick_hf_token(env):
    """-> (env var name, token) of the first non-empty HF token variable, or
    (None, None). The token value is never printed."""
    for name in TOKEN_ENV_NAMES:
        if env.get(name):
            return name, env[name]
    return None, None


def fetch_dataset_file(filename, dest_dir, token, revision):
    """One file of the dataset repo at a pinned commit (a real file in
    dest_dir, deletable after use)."""
    from huggingface_hub import hf_hub_download

    if not revision:
        raise ValueError("fetch_dataset_file needs a pinned revision")
    return Path(hf_hub_download(DATASET_REPO, filename, repo_type="dataset",
                                revision=revision, local_dir=str(dest_dir),
                                token=token))


def resolve_revision(api, revision=None):
    """The dataset repo's commit sha for `revision` (branch, tag or sha;
    None = the default branch), resolved once per run."""
    sha = api.dataset_info(DATASET_REPO, revision=revision or None).sha
    if not sha:
        raise RuntimeError(f"{DATASET_REPO}: no commit sha for revision "
                           f"{revision!r}")
    return sha


def file_matches(data, info):
    """True if `data` is the repo file described by `info` (huggingface_hub
    RepoFile: LFS sha256 when stored in LFS, else the git blob sha1)."""
    if len(data) != info.size:
        return False
    if info.lfs is not None:
        return hashlib.sha256(data).hexdigest() == info.lfs.sha256
    blob = b"blob %d\0" % len(data) + data
    return hashlib.sha1(blob).hexdigest() == info.blob_id


def prepare_impl(parquet=b"", seed=0, control_n=CONTROL_N, revision="",
                 token=None, api=None, download=fetch_dataset_file,
                 tmp="/tmp", log=print):
    """Resolve the dataset commit once, then metadata.parquet AT that commit:
    the sent local copy when it is byte-identical to the commit's file, else
    a download pinned to it -> {meta, plan, revision, source,
    parquet_sha256}."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    sha = resolve_revision(api, revision)
    infos = [i for i in api.get_paths_info(DATASET_REPO, [PARQUET],
                                           revision=sha, repo_type="dataset")
             if getattr(i, "path", None) == PARQUET]
    if not infos:
        raise RuntimeError(f"{DATASET_REPO}@{sha} has no {PARQUET}")
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    if parquet and file_matches(parquet, infos[0]):
        path = tmp / PARQUET
        path.write_bytes(parquet)
        src = "local copy (identical to the commit's file)"
    else:
        path = download(PARQUET, tmp / "meta", token, sha)
        src = "downloaded" + (" (the local copy differs from the commit's "
                              "file)" if parquet else "")
    log(f"{DATASET_REPO}@{sha}: {PARQUET} {src}")
    meta = load_meta(path)
    return {"meta": meta, "plan": plan_job(meta, seed, control_n),
            "revision": sha, "source": src,
            "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_on_tar(tar, revision, token, fn, download=fetch_dataset_file,
               dl_dir=REMOTE_DL):
    """Download one tar at the pinned revision, run fn(path), delete it."""
    t0 = time.time()
    path = Path(download(tar, dl_dir, token, revision))
    dl = time.time() - t0
    try:
        res = fn(path)
    finally:
        path.unlink(missing_ok=True)
    res["tar"] = tar
    res["revision"] = revision
    res["seconds"]["download"] = round(dl, 2)
    res["peak_rss_mib"] = round(peak_rss_mib(), 1)
    return res


def peak_rss_mib():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def estimate_cost(container_s, cores=EST_CORES, mem_gib=0.25):
    """Rough Modal cost of container-seconds x (cores + memory) prices."""
    return container_s / 3600 * (cores * CPU_CORE_USD_H
                                 + mem_gib * MEM_GIB_USD_H)


# ------------------------------------------------------------------ Modal app
if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*PIP_PACKAGES)
        .env(CONTAINER_ENV)
    )
    hf_secret = modal.Secret.from_name("huggingface")
    app = modal.App("krea2-score-borders", image=image)

    def _token():
        return pick_hf_token(os.environ)[1]

    @app.function(secrets=[hf_secret], timeout=30 * 60,
                  memory=META_MEM_MIB)
    def prepare(parquet: bytes = b"", seed: int = 0,
                control_n: int = CONTROL_N, revision: str = "") -> dict:
        """Pin the dataset commit; metadata.parquet at it -> meta + plan."""
        name, token = pick_hf_token(os.environ)
        print(f"HF token from {name or 'nowhere (anonymous)'}")
        return prepare_impl(parquet, seed, control_n, revision, token)

    @app.function(secrets=[hf_secret], timeout=SHARD_TIMEOUT_S, retries=2)
    def score_shard(tar: str, thumb_ids: list, revision: str) -> dict:
        return run_on_tar(tar, revision, _token(),
                          lambda p: score_shard_impl(p, thumb_ids))

    @app.function(secrets=[hf_secret], timeout=SHARD_TIMEOUT_S, retries=2)
    def thumb_shard(tar: str, ids: list, revision: str) -> dict:
        return run_on_tar(tar, revision, _token(),
                          lambda p: thumb_shard_impl(p, ids))

    @app.function(timeout=30 * 60, memory=META_MEM_MIB)
    def finalize(joined: list, revision: str) -> bytes:
        return table_bytes(joined, revision)

    def _secs(res):
        return ", ".join(f"{k} {v:.0f}s" for k, v in res["seconds"].items())

    @app.local_entrypoint()
    def main(shards: int = 0, top_n: int = TOP_N, control_n: int = CONTROL_N,
             seed: int = 0, out: str = "", parquet: str = str(LOCAL_PARQUET),
             revision: str = ""):
        if shards < 0 or top_n < 0 or control_n < 0:
            raise SystemExit("--shards/--top-n/--control-n must be >= 0")
        out_dir = Path(out) if out else LOCAL_OUT
        t_start = time.time()
        pq_bytes = (Path(parquet).read_bytes() if Path(parquet).is_file()
                    else b"")
        print(f"prepare: local copy {parquet if pq_bytes else '(none)'}, "
              f"revision {revision or 'default branch'}")
        prep = prepare.remote(pq_bytes, seed, control_n, revision)
        rev = prep["revision"]   # every download below is pinned to it
        print(f"dataset {DATASET_REPO}@{rev}; {PARQUET}: {prep['source']}")
        meta = {int(k): v for k, v in prep["meta"].items()}
        plan = prep["plan"]
        tars = list(plan["shards"])[:shards] if shards else list(plan["shards"])
        plan = restrict_plan(plan, meta, tars)
        print(f"{len(meta)} images in {len(prep['plan']['shards'])} shards "
              f"(running {len(tars)}); always-thumb: {len(plan['watermark'])} "
              f"watermark, {len(plan['uncaptioned'])} border-uncaptioned, "
              f"{len(plan['control'])} control")
        usd = estimate_cost(len(tars) * (EST_S_PER_SHARD
                                         + EST_S_PER_THUMB_SHARD))
        print(f"estimate (rough): {len(tars)} containers x ~{EST_S_PER_SHARD}"
              f" s + pass 2, ~${usd:.2f}")
        # pass 1: metrics of every image + thumbnails of the always ids
        t1 = time.time()
        rows, thumbs, failed, per = [], {}, [], []
        args = [(t, plan["always"].get(t, []), rev) for t in tars]
        for tar, res in zip(tars, score_shard.starmap(
                args, return_exceptions=True, order_outputs=True)):
            if isinstance(res, BaseException):
                msg = f"{tar}: {type(res).__name__}: {res}"
                print(f"  FAILED {msg}")
                failed.append(msg)
                continue
            err = check_shard_result(tar, res, plan)
            if err is None and res.get("revision") != rev:
                err = f"{tar}: scored at {res.get('revision')}, not {rev}"
            if err:
                failed.append(err)
                continue
            rows.extend(res["rows"])
            thumbs.update(res["thumbs"])
            per.append(res)
            print(f"  {tar}: {res['n']} imgs, {len(res['thumbs'])} thumbs, "
                  f"{_secs(res)}, peak RSS {res['peak_rss_mib']:.0f} MiB")
        if failed:
            raise SystemExit(f"{len(failed)} shards failed, nothing "
                             "written:\n  " + "\n  ".join(failed))
        # pass 2: the global selection's missing thumbnails
        t2 = time.time()
        tabs = build_tabs(join_rows(rows, meta), plan, top_n)
        jobs = thumb_jobs(select_thumbs(tabs), thumbs, meta)
        print(f"pass 2: {sum(map(len, jobs.values()))} thumbnails from "
              f"{len(jobs)} shards")
        for tar, res in zip(jobs, thumb_shard.starmap(
                [(t, ids, rev) for t, ids in jobs.items()],
                return_exceptions=True, order_outputs=True)):
            if isinstance(res, BaseException):
                failed.append(f"{tar}: {type(res).__name__}: {res}")
                continue
            if res.get("revision") != rev:
                failed.append(f"{tar}: thumbnails at {res.get('revision')}, "
                              f"not {rev}")
                continue
            thumbs.update(res["thumbs"])
            print(f"  {tar}: {len(res['thumbs'])}/{len(jobs[tar])} thumbs, "
                  f"{_secs(res)}")
        if failed:
            raise SystemExit(f"{len(failed)} thumbnail shards failed, "
                             "nothing written:\n  " + "\n  ".join(failed))
        check_thumbs(select_thumbs(tabs), thumbs)
        t3 = time.time()
        table = None
        if importlib.util.find_spec("pyarrow") is None:
            # Modal's own CLI env has no pyarrow: build it in a container
            table = finalize.remote(join_rows(rows, meta), rev)
        seconds = {"total": time.time() - t_start, "pass1": t2 - t1,
                   "pass2": t3 - t2}
        if per:
            seconds["shard_mean"] = sum(
                sum(r["seconds"].values()) for r in per) / len(per)
            seconds["shard_max"] = max(sum(r["seconds"].values()) for r in per)
        s = write_outputs(out_dir, rows, thumbs, meta, plan, seconds, top_n,
                          table, revision=rev)
        print_summary(s)
        print(f"wrote {out_dir}/scores.parquet, {len(thumbs)} thumbs, "
              f"{out_dir}/review/index.html (review id {s['review_id']})")


# ------------------------------------------------------------------ local CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] + " (local mode: tars on disk)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Modal: modal run scripts/modal_score_borders.py [--shards N "
               "--top-n 1500 --control-n 150 --seed 0 --out DIR --parquet "
               "PATH --revision SHA]; re-render only the review page of a "
               "finished run: --rerender DIR [--top-n N]")
    ap.add_argument("--tar-dir",
                    help="directory with the dataset tars (by their parquet "
                         "`tar` path or basename)")
    ap.add_argument("--parquet", default=str(LOCAL_PARQUET))
    ap.add_argument("--out")
    ap.add_argument("--rerender", metavar="DIR",
                    help="only rebuild DIR/review/index.html from a finished "
                         "run's DIR/{scores.parquet,summary.json,thumbs} "
                         "(no tars; the v1 page is kept as index.v1.html)")
    ap.add_argument("--shards", type=int, default=0,
                    help="score only the first N tars of the parquet")
    ap.add_argument("--top-n", type=int, default=None,
                    help=f"tab A size (default {TOP_N}; with --rerender: the "
                         "border-pattern rows inside the run's own top-N, "
                         "which all have a thumbnail)")
    ap.add_argument("--control-n", type=int, default=CONTROL_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=SHARD_THREADS)
    ap.add_argument("--revision", default=None,
                    help="dataset commit the local tars + parquet come from "
                         "(recorded only; default: unknown)")
    return ap


def local_main(argv=None):
    """Same pipeline as the Modal run on tars already on disk. Every tar of
    the parquet (or the first --shards N) must be present."""
    ap = build_arg_parser()
    a = ap.parse_args(argv)
    if a.top_n is not None and a.top_n < 0:
        ap.error("--top-n must be >= 0")
    if a.rerender:
        if a.tar_dir or a.out:
            ap.error("--rerender takes no --tar-dir / --out")
        return rerender(a.rerender, a.top_n, a.seed, a.control_n)
    if not a.tar_dir or not a.out:
        ap.error("--tar-dir and --out are required (or --rerender DIR)")
    if a.top_n is None:
        a.top_n = TOP_N
    t0 = time.time()
    meta = load_meta(a.parquet)
    plan = plan_job(meta, a.seed, a.control_n)
    names = list(plan["shards"])[:a.shards] if a.shards else list(plan["shards"])
    tars, failed = {}, []
    for tar in names:
        for cand in (Path(a.tar_dir) / tar, Path(a.tar_dir) / Path(tar).name):
            if cand.is_file():
                tars[tar] = cand
                break
        else:
            failed.append(f"{tar}: not found under {a.tar_dir}")
    plan = restrict_plan(plan, meta, names)
    rows, thumbs = [], {}
    for tar, path in tars.items():
        try:
            res = score_shard_impl(path, plan["always"].get(tar, []),
                                   a.threads)
        except ShardError as e:
            failed.append(f"{tar}: {e}")
            continue
        err = check_shard_result(tar, res, plan)
        if err:
            failed.append(err)
            continue
        rows.extend(res["rows"])
        thumbs.update(res["thumbs"])
    if failed:
        raise SystemExit(f"{len(failed)} shards failed, nothing written:\n  "
                         + "\n  ".join(failed))
    tabs = build_tabs(join_rows(rows, meta), plan, a.top_n)
    for tar, ids in thumb_jobs(select_thumbs(tabs), thumbs, meta).items():
        thumbs.update(thumb_shard_impl(tars[tar], ids)["thumbs"])
    check_thumbs(select_thumbs(tabs), thumbs)
    s = write_outputs(a.out, rows, thumbs, meta, plan,
                      {"total": time.time() - t0}, a.top_n,
                      revision=a.revision)
    print_summary(s)
    return s


if __name__ == "__main__":
    local_main()
