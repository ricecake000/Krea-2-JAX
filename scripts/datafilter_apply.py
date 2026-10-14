"""Apply the border review (datafilter_actions.json of the review page built
by scripts/modal_score_borders.py) to the stage-3 dataset as a PATCH
DIRECTORY: caption fixes, crops re-made from the ORIGINAL images, removals.
Local CPU only (no Modal, no HF, no network); the latent patch / new dataset
version is built from the patch directory by scripts/modal_patch_latents.py.

    python scripts/datafilter_apply.py --actions ~/다운로드/datafilter_actions.json \\
        --out /home/mj/krea2-ops/eval/datafilter/patch [--originals DIR] \\
        [--metadata P] [--scores P] [--tokenizer tokenizer.json] [--workers 4]

Inputs: the actions json {review_id, generated, dataset_revision, items:
[{id, action in caption|crop|remove, tabs, score, side_pattern}]},
metadata.parquet of the dataset (captions, tags, artist, buckets), the
review run's scores.parquet (border metrics of the DATASET image; its
schema metadata carries the dataset revision, which must equal the
actions'), the originals (`<originals>/<id>.webp`, scripts/
fetch_originals.py) and the Qwen3-VL tokenizer of the local HF cache.
Every acted id must be a dataset id; duplicate ids, unknown actions or a
revision mismatch fail the run before anything is written.

Actions:
  * crop: the original is decoded (EXIF orientation applied, a non-sRGB
    ICC profile converted to sRGB, then alpha composited on white) and
    measured with
    modal_score_borders.border_metrics (downscaled with a box filter to a
    long side <= --measure-max-side for the measurement; the integer margins
    are recovered with row_px_straight and mapped back to original pixels
    conservatively (the cut ends after the whole span of the first content
    line, so no canvas survives the downscale; exact at full resolution),
    then every side WITH a margin is inset by --inset px to drop the
    antialiased edge). The crop box goes to the bucket of the nearest aspect
    ratio (log space) among the dataset's buckets (all (bucket_w, bucket_h)
    pairs of metadata.parquet) and is cover-resized + center-cropped by the
    repo's own scripts/cache_dataset.load_image (Lanczos, the path
    cache_latents.decode uses for an off-bucket image), then saved as WebP
    q95 (sRGB) to <out>/images/<id>.webp. Border phrases are stripped from
    the captions (see strip_border).
      status crop_too_small: the crop box would need upscaling by more than
        --max-upscale (default 1.0 = none) -> no image; the row keeps the
        OLD bucket and captions (the planned strip is listed in notes).
      status crop_unsure: the original's measurement disagrees with the
        dataset image's (original score 0 or < --unsure-ratio x the
        scores.parquet score, or the box keeps >= 98% of the original) ->
        the image IS produced, flagged in the report.
      status error: original missing / undecodable / blank -> no image, old
        bucket and captions.
  * caption: a border phrase built from the dataset image's measurement
    (scores.parquet) and the tags (border_phrases) is appended at the end
    of each caption that does not already mention a border; `outside
    border` tag -> "character extending outside the border" (unless already
    mentioned).
  * remove: recorded only (captions unchanged).

Token counts (long_tok, medium_tok, short_tok) are recounted for every row
exactly like the dataset's own columns: the caption plus ", drawn by
{artist_name}" (when the artist is known), tokenized as PREFIX + text and
sliced at the prefix (krea2_jax.text_template.CaptionTemplate, the
trainer's path). The budget is CaptionTemplate.max_caption_tokens (251 for
--text-rows 256); a caption that ends up over budget AND longer than
before -> status over_budget (the trainer truncates from the end, i.e. it
would drop the appended phrase). Captions already over budget in the
dataset and not lengthened are only noted.

      status caption_empty: stripping would leave a caption empty (or
        only filler words) -> no image, old bucket and captions.

Status precedence (one per row): error > caption_empty > crop_too_small >
crop_unsure > over_budget > ok; notes holds everything ("; "-joined). The
latent patch builder applies only ok and crop_unsure rows; every other
status is reported as unapplied.

Output (<out>, built in <out>.tmp-<pid> under an exclusive flock on
<out>.lock; an existing <out> needs --overwrite and is renamed aside, then
deleted only once the new one is in place, restored if that rename fails):
    images/<id>.webp   final bucket-sized WebP q95, only for crop rows with
                       status ok / crop_unsure / over_budget (image present
                       <=> the file exists)
    patch.parquet      one row per acted id: id, action, bucket_w, bucket_h
                       (new for crops with an image, old otherwise), long,
                       medium, short (new captions), long_old, medium_old,
                       short_old, long_tok, medium_tok, short_tok
                       (recounted), crop_box_orig ([x0, y0, x1, y1] in
                       original px, null unless crop + measured), orig_w,
                       orig_h (null unless crop + decoded), status, notes
    patch.json         dataset_revision (+ source_dataset_revision, same
                       value), source repo, review id, counts,
                       parameters, tokenizer digest, tool versions
    report.html        self-contained: per acted id before/after thumbnails
                       (crops: original with the crop box + final image;
                       others: the review thumbnail), caption diffs
                       (removed red, added green), status badges
Exit code: 0, or 2 when any row has status error (patch still written), 1 on
invalid input (nothing written).
"""

import argparse
import base64
import colorsys
import difflib
import hashlib
import html
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import modal_score_borders as msb  # noqa: E402
from krea2_jax.text_template import (  # noqa: E402
    DEFAULT_TEXT_ROWS, PREFIX, CaptionTemplate)

FORMAT = "krea2-datafilter-patch-v1"
ACTIONS = ("caption", "crop", "remove")
STATUS_ORDER = ("error", "caption_empty", "crop_too_small", "crop_unsure",
                "over_budget", "ok")
# words that alone do not make a caption (a stripped caption left with only
# these -> caption_empty)
FILLER_WORDS = frozenset("a an the and or with of in on at by to as".split())
CAPTION_FIELDS = ("long", "medium", "short")
ARTIST_SUFFIX = ", drawn by {}"     # krea2_jax/data.py ARTIST_SUFFIX
MEASURE_MAX_SIDE = 2048
CROP_INSET_PX = 1
UNSURE_RATIO = 0.5
NOTHING_TO_CROP = 0.98              # box keeps >= 98% of the original area
CENTER_LOSS_NOTE = 0.10             # note when the bucket crop drops > 10%
WEBP_QUALITY = 95
REPORT_THUMB = 320
REPORT_JPEG_Q = 78
MAX_IMAGE_PIXELS = 400_000_000
LOCAL_METADATA = Path("/home/mj/krea2-ops/mi350x/data/metadata.parquet")
LOCAL_REVIEW = Path("/home/mj/krea2-ops/eval/datafilter")
LOCAL_ORIGINALS = Path("/home/mj/.cache/krea2-orig/img")
HF_TOKENIZER_GLOB = ("models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*/"
                     "tokenizer.json")
PATCH_COLUMNS = (
    "id", "action", "bucket_w", "bucket_h", "long", "medium", "short",
    "long_old", "medium_old", "short_old", "long_tok", "medium_tok",
    "short_tok", "crop_box_orig", "orig_w", "orig_h", "status", "notes")


class InputError(ValueError):
    """Invalid actions / metadata / scores: nothing is written."""


# ------------------------------------------------------------------ inputs
def load_actions(path):
    """-> (doc, [items]) with ids as int; unknown actions, malformed or
    duplicate ids raise InputError."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else None
    if not isinstance(items, list):
        raise InputError(f"{path}: no items list")
    out, seen, bad = [], set(), []
    for it in items:
        try:
            sid = int(it["id"])
            act = it["action"]
        except (KeyError, TypeError, ValueError):
            bad.append(repr(it)[:80])
            continue
        if act not in ACTIONS:
            bad.append(f"{sid}: action {act!r}")
            continue
        if sid in seen:
            bad.append(f"{sid}: duplicate")
            continue
        seen.add(sid)
        out.append({**it, "id": sid})
    if bad:
        raise InputError(f"{path}: {len(bad)} bad items: {bad[:10]}")
    return doc, out


def load_metadata(path, ids):
    """-> ({id: row} for `ids`, sorted [(bucket_w, bucket_h)] of the whole
    dataset). An id missing from the parquet raises InputError."""
    import pyarrow.parquet as pq

    cols = ["id", "bucket_w", "bucket_h", "tags", "artist_name",
            *CAPTION_FIELDS, *(f + "_tok" for f in CAPTION_FIELDS)]
    t = pq.read_table(path, columns=cols).to_pydict()
    buckets = sorted({(int(w), int(h))
                      for w, h in zip(t["bucket_w"], t["bucket_h"])})
    want = set(ids)
    rows = {}
    for i, sid in enumerate(t["id"]):
        if int(sid) in want:
            rows[int(sid)] = {c: t[c][i] for c in cols}
    missing = sorted(want - set(rows))
    if missing:
        raise InputError(f"{path}: {len(missing)} acted ids not in the "
                         f"dataset: {missing[:10]}")
    return rows, buckets


def load_scores(path, ids):
    """-> ({id: score row} for `ids`, dataset_revision or None)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    md = pf.schema_arrow.metadata or {}
    rev = md.get(b"dataset_revision")
    t = pf.read().to_pydict()
    want = set(ids)
    rows = {}
    for i, sid in enumerate(t["id"]):
        if int(sid) in want:
            rows[int(sid)] = {k: v[i] for k, v in t.items()}
    missing = sorted(want - set(rows))
    if missing:
        raise InputError(f"{path}: {len(missing)} acted ids without a "
                         f"score row: {missing[:10]}")
    return rows, (rev.decode() if rev else None)


def find_tokenizer(arg=None):
    """--tokenizer (tokenizer.json or a directory holding it), else the
    local HF cache snapshot of Qwen3-VL-4B-Instruct."""
    if arg:
        p = Path(arg)
        return p / "tokenizer.json" if p.is_dir() else p
    hub = Path(os.environ.get("HF_HUB_CACHE") or
               Path(os.environ.get("HF_HOME", Path.home() / ".cache" /
                                   "huggingface")) / "hub")
    found = sorted(hub.glob(HF_TOKENIZER_GLOB))
    if not found:
        raise InputError(f"no Qwen3-VL tokenizer.json under {hub} "
                         "(pass --tokenizer)")
    return found[-1]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ buckets
def choose_bucket(w, h, buckets):
    """Bucket (bw, bh) whose aspect ratio is nearest to w / h in log space
    (ties: the first in sorted order)."""
    a = math.log(w / h)
    return min(sorted(buckets), key=lambda b: abs(math.log(b[0] / b[1]) - a))


def cover_scale(w, h, bw, bh):
    """Scale of the cover resize of a w x h picture to bw x bh."""
    return max(bw / w, bh / h)


def center_loss(w, h, bw, bh):
    """Fraction of the (cover-resized) picture the center crop drops."""
    s = cover_scale(w, h, bw, bh)
    return 1.0 - (bw * bh) / (w * s * h * s)


def resize_to_bucket(img, bw, bh):
    """PIL RGB image -> PIL RGB bw x bh through the dataset pipeline's own
    cover-resize + center-crop (scripts/cache_dataset.load_image, Lanczos;
    a lossless PNG round trip hands it the picture, and its [-1, 1] float
    output maps back to uint8 exactly)."""
    import numpy as np
    from PIL import Image

    from cache_dataset import load_image

    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=1)
    buf.seek(0)
    t = load_image(buf, bw, bh)[0]                       # (3, bh, bw)
    arr = ((t.permute(1, 2, 0).numpy().astype(np.float64) + 1.0) * 127.5)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ images
def open_original(path):
    """-> (PIL RGB image, [notes]): EXIF orientation applied first, then a
    non-sRGB ICC profile converted to sRGB (alpha kept through the
    transform), then alpha composited on white."""
    from PIL import Image, ImageCms, ImageOps

    notes = []
    with Image.open(path) as im:
        im.load()
        icc = im.info.get("icc_profile")
        orient = im.getexif().get(0x0112, 1)
        im = ImageOps.exif_transpose(im)
        if orient not in (None, 1):
            notes.append(f"EXIF orientation {orient} applied")
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (
            im.mode == "P" and "transparency" in im.info)
        mode = "RGBA" if has_alpha else "RGB"
        img = im.convert(mode)
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            desc = (ImageCms.getProfileDescription(src) or "").strip()
            if "srgb" not in desc.lower():
                img = ImageCms.profileToProfile(
                    img, src, ImageCms.createProfile("sRGB"), outputMode=mode)
                notes.append(f"ICC {desc!r} converted to sRGB")
        except Exception as e:  # noqa: BLE001 - a broken profile: keep pixels
            notes.append(f"ICC profile unreadable ({type(e).__name__}), "
                         "pixels used as sRGB")
    if has_alpha:
        if img.getextrema()[3][0] < 255:
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
            notes.append("alpha composited on white")
        img = img.convert("RGB")
    return img, notes


def measure_box(img, max_side=MEASURE_MAX_SIDE, inset=CROP_INSET_PX):
    """border_metrics of the original (box-downscaled to a long side <=
    max_side) -> (crop box (x0, y0, x1, y1) in original px or None when
    blank / empty, metrics dict, measured (w, h))."""
    import numpy as np
    from PIL import Image

    W, H = img.size
    meas = img
    if max(W, H) > max_side:
        s = max_side / max(W, H)
        meas = img.resize((max(1, round(W * s)), max(1, round(H * s))),
                          Image.BOX)
    mw, mh = meas.size
    m = msb.border_metrics(np.asarray(meas))
    if m["blank"]:
        return None, m, (mw, mh)
    px, _ = msb.row_px_straight(m)
    sx, sy = W / mw, H / mh

    def cut(p, scale):
        """Original px to drop for a p-line measured margin: the first
        content line (the one after the margin) may hold the edge anywhere
        in its scale-px span, so the cut ends after that span (exactly p at
        full resolution), plus the inset. No margin -> nothing."""
        return math.ceil((p + 1) * scale) - 1 + inset if p else 0

    t, b, l, r = (px[s] for s in msb.SIDES)
    y0, y1 = cut(t, sy), H - cut(b, sy)
    x0, x1 = cut(l, sx), W - cut(r, sx)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None, m, (mw, mh)
    return (x0, y0, x1, y1), m, (mw, mh)


def jpeg_data_uri(img, side=REPORT_THUMB, box=None):
    """Small JPEG data URI of a PIL image; `box` (original px) drawn red."""
    from PIL import ImageDraw

    im = img.copy()
    im.thumbnail((side, side))
    if box is not None:
        sx, sy = im.size[0] / img.size[0], im.size[1] / img.size[1]
        x0, y0, x1, y1 = box
        ImageDraw.Draw(im).rectangle(
            (x0 * sx, y0 * sy, max(x0 * sx, x1 * sx - 1),
             max(y0 * sy, y1 * sy - 1)), outline=(230, 20, 20), width=2)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=REPORT_JPEG_Q)
    return "data:image/jpeg;base64," + base64.b64encode(
        buf.getvalue()).decode()


def save_webp(img, path, quality=WEBP_QUALITY):
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    img.save(tmp, "WEBP", quality=quality)
    os.replace(tmp, path)


def crop_one(sid, orig_path, ds_score, buckets, images_dir, *,
             max_side=MEASURE_MAX_SIDE, inset=CROP_INSET_PX,
             unsure_ratio=UNSURE_RATIO, max_upscale=1.0, thumbs=True):
    """Measure + crop + bucket one original. -> dict(status, notes, box,
    orig_w, orig_h, bucket (bw, bh) or None, image written?, thumbs)."""
    from PIL import Image

    res = {"status": "ok", "notes": [], "box": None, "orig_w": None,
           "orig_h": None, "bucket": None, "image": False,
           "thumb_before": None, "thumb_after": None}
    if not Path(orig_path).is_file():
        res.update(status="error")
        res["notes"].append(f"original missing: {orig_path}")
        return res
    try:
        img, notes = open_original(orig_path)
    except Exception as e:  # noqa: BLE001 - one bad file is a row error
        res.update(status="error")
        res["notes"].append(f"original undecodable: {type(e).__name__}: {e}")
        return res
    res["notes"] += notes
    W, H = img.size
    res.update(orig_w=W, orig_h=H)
    box, m, (mw, mh) = measure_box(img, max_side, inset)
    res["notes"].append(
        f"original {W}x{H} measured at {mw}x{mh}: score {m['score']} "
        f"(dataset {ds_score}), pattern {m['side_pattern']}, bg "
        f"{(m['bg_r'], m['bg_g'], m['bg_b'])}")
    if box is None:
        res.update(status="error")
        res["notes"].append("no content box in the original (blank)")
        if thumbs:
            res["thumb_before"] = jpeg_data_uri(img)
        return res
    res["box"] = list(box)
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    unsure = []
    if m["score"] <= 0:
        unsure.append("original scores 0")
    elif ds_score and m["score"] < unsure_ratio * ds_score:
        unsure.append(f"original score {m['score']} < {unsure_ratio} x "
                      f"dataset {ds_score}")
    if cw * ch >= NOTHING_TO_CROP * W * H:
        unsure.append(f"box keeps {cw * ch / (W * H):.1%} of the original")
    bw, bh = choose_bucket(cw, ch, buckets)
    scale = cover_scale(cw, ch, bw, bh)
    res["notes"].append(f"crop {cw}x{ch} -> bucket {bw}x{bh} "
                        f"(scale {scale:.3f})")
    loss = center_loss(cw, ch, bw, bh)
    if loss > CENTER_LOSS_NOTE:
        res["notes"].append(f"bucket center crop drops {loss:.1%} of the "
                            "cropped picture")
    if thumbs:
        res["thumb_before"] = jpeg_data_uri(img, box=box)
    if scale > max_upscale:
        res.update(status="crop_too_small")
        res["notes"].append(f"crop {cw}x{ch} smaller than bucket {bw}x{bh} "
                            f"(needs {scale:.2f}x upscaling)")
        return res
    if unsure:
        res.update(status="crop_unsure")
        res["notes"] += unsure
    out = resize_to_bucket(img.crop(box), bw, bh)
    assert out.size == (bw, bh), (out.size, bw, bh)
    save_webp(out, Path(images_dir) / f"{sid}.webp")
    res.update(bucket=(bw, bh), image=True)
    if thumbs:
        with Image.open(Path(images_dir) / f"{sid}.webp") as saved:
            res["thumb_after"] = jpeg_data_uri(saved)
    return res


# ------------------------------------------------------------------ phrases
COLOURS_GREY = ("light grey", "grey", "dark grey")


def colour_name(rgb):
    """Simple colour name of an sRGB triple: white / black (the score's
    near-white / near-black thresholds), greys, else a hue name with a
    pale / dark modifier."""
    r, g, b = (int(x) for x in rgb)
    if min(r, g, b) >= msb.NEAR_WHITE_MIN:
        return "white"
    if max(r, g, b) <= msb.NEAR_BLACK_MAX:
        return "black"
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    if max(r, g, b) - min(r, g, b) < 24 or s < 0.15:
        return ("light grey" if l > 0.75 else "dark grey" if l < 0.3
                else "grey")
    deg = h * 360
    if deg < 15 or deg >= 345:
        name = "red"
    elif deg < 45:
        name = "brown" if l < 0.35 else "orange"
    elif deg < 70:
        name = "yellow"
    elif deg < 165:
        name = "green"
    elif deg < 200:
        name = "cyan"
    elif deg < 255:
        name = "blue"
    elif deg < 290:
        name = "purple"
    else:
        name = "pink"
    if l > 0.8 and name != "brown":
        return "pale " + name
    if l < 0.25 and name != "brown":
        return "dark " + name
    return name


def border_phrases(score_row, tags):
    """-> ([phrases], info) for a caption fix: the border phrase (colour from
    the dataset image's canvas measurement when the measurement is a border,
    else the `white border` tag; bars for 2 opposite sides, or a pillar /
    letterbox flag whose side pattern is not 4 / 3, or the `pillarboxed`
    tag) and "character extending outside the border" for the `outside
    border` tag."""
    ts = msb.split_tags(tags)
    px, st = msb.row_px_straight(score_row)
    sp = msb.row_side_pattern(score_row)
    pillar = bool(score_row.get("pillar_or_letter"))
    measured = sp in msb.BORDER_PATTERNS or pillar
    bg = (score_row["bg_r"], score_row["bg_g"], score_row["bg_b"])
    if measured:
        colour = colour_name(bg)
    elif "white border" in ts:
        colour = "white"
    else:
        colour = None
    if sp == "2-opp":
        # the two sides side_pattern counted (margin >= SIDE_MIN_PX px and
        # straightness >= SIDE_STRAIGHT), not the larger margin pair
        q = {x for x in msb.SIDES if px[x] is not None and st[x] is not None
             and px[x] >= msb.SIDE_MIN_PX and st[x] >= msb.SIDE_STRAIGHT}
        kind = ("bars on the left and right" if q == {"left", "right"}
                else "bars on the top and bottom")
    elif pillar and sp not in ("4", "3"):
        # pillar_or_letter: margins >= MIN_SIDE_FRAC on exactly one
        # opposite pair
        lr = min(score_row["margin_left"] or 0, score_row["margin_right"] or 0)
        kind = ("bars on the left and right" if lr >= msb.MIN_SIDE_FRAC
                else "bars on the top and bottom")
    elif not measured and "pillarboxed" in ts:
        kind = "bars on the left and right"
    else:
        kind = "border"
    if colour:
        phrase = f"{colour} {kind}"
    else:
        phrase = "border around the image" if kind == "border" else kind
    out = [phrase]
    if "outside border" in ts:
        out.append("character extending outside the border")
    return out, {"side_pattern": sp, "measured": measured, "colour": colour}


# Border vocabulary for stripping (phrase level). "frame" counts only in the
# border sense; "inset" alone (chibi / reference insets = sub-pictures) and
# "margin" as a location ("signature in the right margin") never count.
STRIP_PATTERNS = (
    ("border", re.compile(r"\bborder(?:s|ed)?\b(?!\s+coll)", re.I)),
    ("letterbox", re.compile(r"\bletterbox(?:ed|ing|es)?\b", re.I)),
    ("pillarbox", re.compile(r"\bpillarbox(?:ed|ing|es)?\b", re.I)),
    ("margin", re.compile(
        r"\b(?:white|black|grey|gray|blank|empty|plain|wide|thin|thick|"
        r"solid)\s+margins?\b", re.I)),
    ("frame", re.compile(
        r"\b(?:white|black|grey|gray|thin|thick|plain|solid|simple)\s+"
        r"frame\b(?!\s*glasses)", re.I)),
    ("framed-bg", re.compile(
        r"\bframed\s+(?:[\w-]+\s+){0,2}?(?:background|composition)\b", re.I)),
    ("bars", re.compile(
        r"\b(?:side)?bars\b(?:\s+\w+){0,3}?\s+(?:at\s+|on\s+)?(?:the\s+)?"
        r"(?:top and bottom|left and right|sides)\b|\bsidebars\b", re.I)),
)
# A phrase (or a "with" / "framed by" ... tail) is removed only when EVERY
# word of it is border vocabulary or a modifier / colour / position /
# filler word (BORDER_WORDS): "thin white border framing the image",
# "black letterbox bars at top and bottom", "character extending outside the
# border". A phrase that hits the vocabulary but carries other content
# ("walking a border collie", "vanity mirror bordered with bulbs", "white
# photo frame border placed on a corkboard surface") is kept and listed.
BORDER_WORDS = frozenset("""
border borders bordered frame frames framed framing margin margins
letterbox letterboxed letterboxing pillarbox pillarboxed pillarboxing
bar bars sidebar sidebars line lines outline mat matte inset edge edges
a an the with by in on at around along of and to within inside surrounding
surrounded enclosed enclosing flanked all four both sides side top bottom
left right upper lower image picture illustration artwork art drawing canvas
composition format effect style cinematic widescreen aspect ratio like
outside extending extends extend breaking popping out beyond overlapping
over parts part character characters scene
thin thick narrow wide broad simple plain solid subtle faint soft clean
sharp crisp rough decorative ornate ornamental elegant rounded rectangular
square outer inner double dashed dotted striped patterned geometric floral
lace scalloped sticker vignette textured uneven minimal minimalist slight
white black grey gray red orange yellow green cyan teal blue navy purple
violet lavender lilac pink magenta brown beige cream ivory gold golden
silver off light dark pale pastel bright deep muted warm cool neutral mint
peach coral maroon crimson turquoise aqua sky olive tan khaki rose salmon
charcoal colored coloured multicolored rainbow gradient vertical horizontal
""".split())
# adjective drops: "pink framed background" -> "pink background"
ADJ_DROP = re.compile(r"\b(?:framed|bordered)\s+(?=(?:[\w-]+\s+){0,2}?"
                      r"(?:background|composition)\b)", re.I)
CONNECTOR = re.compile(
    r"\s+(with|framed by|framed with|framed in|framed within|surrounded by|"
    r"flanked by|bordered by|bordered with|enclosed by|enclosed in|inside|"
    r"within)\s+", re.I)
PROTECTED = re.compile(r"\btext reading\b", re.I)
# quoted text: double / curly quotes (an unclosed opening quote runs to
# the end) and single-quote pairs. A straight or curly apostrophe between
# two letters (girl's, girl\u2019s, it's) is an apostrophe, never a quote;
# everything matches across newlines. When the quote characters are not
# all accounted for (unbalanced / ambiguous), quote_info protects the whole
# stretch from the first to the last quote-like character as a backstop.
_L = r"[^\W\d_]"                    # a letter
_APOS = rf"(?<={_L})['\u2019](?={_L})"
APOSTROPHE = re.compile(_APOS)
QUOTE_CHARS = "\"'\u2018\u2019\u201c\u201d"
QUOTE_RES = (
    re.compile(r'"[^"]*(?:"|$)', re.S),
    re.compile("\u201c[^\u201d]*(?:\u201d|$)", re.S),
    # the content alternatives are disjoint (the other kind of apostrophe is
    # already in the negated class), so a missing closing quote cannot
    # backtrack exponentially
    re.compile(rf"\u2018(?:[^\u2019]|(?<={_L})\u2019(?={_L}))*(?:\u2019|$)",
               re.S),
    re.compile(rf"(?<![\w'])'(?=\S)(?:[^']|(?<={_L})'(?={_L}))*?(?<=\S)'"
               r"(?!\w)", re.S),
)
# "already captioned" check: canvas-border vocabulary only (no frame /
# margin mentions: "white frame on the wall" is not a border caption)
CANVAS_MENTION = re.compile(
    r"(?<![\w-])border(?:s|ed)?\b(?!\s+coll)|\bletterbox\w*|"
    r"\bpillarbox\w*|" + dict(STRIP_PATTERNS)["bars"].pattern, re.I)
SUSPECT = re.compile(r"(?<![\w-])(?:frames?|framed|inset|margins?)\b", re.I)
OUTSIDE_MENTION = re.compile(
    r"\boutside\s+(?:of\s+)?(?:the\s+)?(?:border|frame)\b|"
    r"\bbreaking\s+(?:out\s+of\s+)?(?:the\s+)?(?:border|frame)\b|"
    r"\bout\s+of\s+(?:the\s+)?frame\b", re.I)


def border_hits(text):
    """Names of the STRIP_PATTERNS matching `text`."""
    return [n for n, rx in STRIP_PATTERNS if rx.search(text)]


def _merge(spans):
    merged = []
    for a, b in sorted(spans):
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def quote_info(text):
    """-> (sorted merged [(start, end)] protected spans, ambiguous). The
    quotes are ambiguous when an opening quote is never closed, two
    matches overlap without nesting, or a quote-like character (not an
    apostrophe between letters) lies outside every match; then the stretch
    from the first to the last quote-like character (apostrophes included)
    is protected too."""
    raw = [(m.start(), m.end()) for rx in QUOTE_RES
           for m in rx.finditer(text) if m.end() > m.start()]
    ambiguous = False
    for a, b in raw:
        closer = {'"': '"', "\u201c": "\u201d", "\u2018": "\u2019",
                  "'": "'"}[text[a]]
        if b - a < 2 or text[b - 1] != closer:
            ambiguous = True                    # unclosed: ran to the end
    for i, (a, b) in enumerate(raw):
        for c, d in raw[i + 1:]:
            if a < d and c < b and not (a <= c and d <= b or
                                        c <= a and b <= d):
                ambiguous = True
    apos = {m.start() for m in APOSTROPHE.finditer(text)}
    spans = _merge(raw)
    qpos = [i for i, ch in enumerate(text) if ch in QUOTE_CHARS]
    if any(i not in apos and not any(a <= i < b for a, b in spans)
           for i in qpos):
        ambiguous = True
    if ambiguous and qpos:
        spans = _merge(spans + [(qpos[0], qpos[-1] + 1)])
    return spans, ambiguous


def quote_spans(text):
    """Sorted, merged [(start, end)] of the protected (quoted) parts."""
    return quote_info(text)[0]


def unquoted(text):
    """text with every quoted part blanked (same length)."""
    for a, b in quote_spans(text):
        text = text[:a] + " " * (b - a) + text[b:]
    return text


def split_segments(caption):
    """Comma split outside quotes -> [(segment, has_quote)]; the segments
    joined with "," give the caption back."""
    spans = quote_spans(caption)
    cuts = [i for i, ch in enumerate(caption) if ch == ","
            and not any(a <= i < b for a, b in spans)]
    out, start = [], 0
    for c in cuts + [len(caption)]:
        out.append((caption[start:c],
                    any(a < c and b > start for a, b in spans)))
        start = c + 1
    return out


def canvas_mentions(caption):
    """Canvas-border mentions outside quotes."""
    return [m.group(0) for m in CANVAS_MENTION.finditer(
        unquoted(caption or ""))]


def has_border_mention(caption):
    return bool(canvas_mentions(caption))


def is_pure_border(text):
    """Every word of text is in BORDER_WORDS (and there is one)."""
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and all(w in BORDER_WORDS for w in words)


def _strip_unit(text):
    """One phrase / sentence -> (new text or None = drop, [removed parts],
    kept note or None); the unchanged object when nothing is removed. A
    trailing sentence period stays with a kept head."""
    core = text.rstrip()
    punct = len(core) - len(core.rstrip(".!?"))
    if not punct:
        return _strip_core(text)
    new, removed, note = _strip_core(core[:-punct])
    if not removed:
        return text, removed, note
    return (None if new is None else new + text[len(core) - punct:],
            removed, note)


def _strip_core(text):
    if not border_hits(text):
        if SUSPECT.search(text):
            return text, [], text.strip()
        return text, [], None
    if PROTECTED.search(text):
        return text, [], text.strip() + " (quoted text)"
    # 1. adjective drop: framed / bordered before background / composition
    removed = [m.group(0).strip() for m in ADJ_DROP.finditer(text)]
    dropped = ADJ_DROP.sub("", text)
    if not border_hits(dropped):
        return dropped, removed, None
    # 2. a pure border phrase goes as a whole
    if is_pure_border(dropped):
        return None, [text.strip()], None
    # 3. HEAD <connector> TAIL: keep the head (and the other "and" parts of a
    #    "with" tail) when the head is real content without border words and
    #    every removed part is a pure border phrase
    for m in CONNECTOR.finditer(dropped):
        head, conn, tail = dropped[:m.start()], m.group(1), dropped[m.end():]
        if (not head.strip() or border_hits(head) or is_pure_border(head)
                or not border_hits(tail)):
            continue
        if conn.lower() == "with":
            parts = re.split(r"\s+and\s+", tail)
            gone = [x for x in parts if border_hits(x)]
            keep = [x for x in parts if not border_hits(x)]
            if keep and all(is_pure_border(x) for x in gone):
                return (head + m.group(0) + " and ".join(keep),
                        removed + [" and ".join(gone)], None)
        if is_pure_border(tail):
            return head, removed + [dropped[m.start():].strip()], None
    return text, [], text.strip() + " (border word with other content)"


def strip_border(caption):
    """Remove border phrases from a comma-separated caption (sentences
    inside a phrase are handled one by one). -> (new caption, [removed
    phrases / parts], [kept phrases that mention frame / inset / margin in
    another sense, for review]). Every other byte is kept as is."""
    caption = caption or ""
    parsed = split_segments(caption)
    ambiguous = quote_info(caption)[1]
    segs = [seg for seg, _ in parsed]
    out, removed, suspects = [], [], []
    for seg, quoted in parsed:
        if quoted:      # never modify a phrase holding quoted text
            if border_hits(seg) or SUSPECT.search(seg):
                suspects.append(seg.strip() + (
                    " (inside unbalanced quotes)" if ambiguous
                    else " (quoted text)"))
            out.append(seg)
            continue
        units = re.split(r"(?<=\.)(\s+)", seg)  # sentence, sep, sentence...
        new_units, changed = [], False
        for k in range(0, len(units), 2):
            new, gone, sus = _strip_unit(units[k])
            removed += gone
            if sus:
                suspects.append(sus)
            if new is not units[k]:
                changed = True
            if new is not None:
                if new_units:
                    new_units.append(units[k - 1])
                new_units.append(new)
        if not changed:
            out.append(seg)
            continue
        if not new_units or not "".join(new_units).strip():
            continue
        text = "".join(new_units)
        if seg[:1].isspace() and not text[:1].isspace():
            text = " " + text
        out.append(text)
    new = ",".join(out)
    if out and out[0] is not segs[0]:
        new = new.lstrip()
    return new, removed, suspects


def add_phrases(caption, phrases):
    """Append each phrase not already covered at the end of the caption.
    -> (new caption, [added], [skipped with reason])."""
    caption = caption or ""
    added, skipped = [], []
    for p in phrases:
        outside = p.startswith("character extending outside")
        if outside and OUTSIDE_MENTION.search(unquoted(caption)):
            skipped.append(f"{p!r}: already mentioned")
            continue
        if not outside and has_border_mention(caption):
            skipped.append(f"{p!r}: caption already mentions a border "
                           f"({', '.join(canvas_mentions(caption))})")
            continue
        base = caption.rstrip().rstrip(",").rstrip()
        caption = f"{base}, {p}" if base else p
        added.append(p)
    return caption, added, skipped


# ------------------------------------------------------------------ tokens
def token_counts(template, captions, artists):
    """Caption tokens as the dataset's *_tok columns count them: caption +
    ", drawn by {artist}" (when known), PREFIX + text sliced at the prefix
    (CaptionTemplate's layout, untruncated)."""
    texts = [PREFIX + c + (ARTIST_SUFFIX.format(a) if a else "")
             for c, a in zip(captions, artists)]
    encs = template.tokenizer.encode_batch(texts)
    return [len(e.ids) - template.prefix_len for e in encs]


# ------------------------------------------------------------------ build
def caption_is_empty(caption):
    words = re.findall(r"[^\W_]+", (caption or "").lower())
    return not words or all(w in FILLER_WORDS for w in words)


def worst(statuses):
    return min(statuses, key=STATUS_ORDER.index)


def build_rows(items, meta, scores, buckets, template, originals, images_dir,
               *, workers=4, max_side=MEASURE_MAX_SIDE, inset=CROP_INSET_PX,
               unsure_ratio=UNSURE_RATIO, max_upscale=1.0, thumbs_dir=None,
               log=print):
    """-> list of row dicts (PATCH_COLUMNS + report extras: added, removed,
    suspects, thumb_before, thumb_after, statuses)."""
    crops = [it["id"] for it in items if it["action"] == "crop"]
    crop_res = {}
    if crops:
        log(f"crop: {len(crops)} originals, {workers} workers")

        def job(sid):
            return sid, crop_one(
                sid, Path(originals) / f"{sid}.webp", scores[sid]["score"],
                buckets, images_dir, max_side=max_side, inset=inset,
                unsure_ratio=unsure_ratio, max_upscale=max_upscale)

        with ThreadPoolExecutor(max(1, workers)) as pool:
            for n, (sid, res) in enumerate(pool.map(job, crops), 1):
                crop_res[sid] = res
                if n % 25 == 0 or n == len(crops):
                    log(f"  {n}/{len(crops)} cropped")
    rows = []
    for it in items:
        sid, act = it["id"], it["action"]
        m, s = meta[sid], scores[sid]
        old = {f: m[f] or "" for f in CAPTION_FIELDS}
        new = dict(old)
        statuses, notes = ["ok"], []
        added, removed, suspects = {}, {}, {}
        row = {"id": sid, "action": act, "bucket_w": int(m["bucket_w"]),
               "bucket_h": int(m["bucket_h"]), "crop_box_orig": None,
               "orig_w": None, "orig_h": None, "thumb_before": None,
               "thumb_after": None}
        if act == "caption":
            phrases, info = border_phrases(s, m["tags"])
            notes.append(f"phrases {phrases} (pattern {info['side_pattern']}"
                         f", colour {info['colour']})")
            for f in CAPTION_FIELDS:
                new[f], added[f], skipped = add_phrases(old[f], phrases)
                notes += [f"{f}: {x}" for x in skipped]
            if new == old:
                notes.append("all captions already mention it: unchanged")
        elif act == "crop":
            r = crop_res[sid]
            statuses.append(r["status"])
            notes += r["notes"]
            row.update(crop_box_orig=r["box"], orig_w=r["orig_w"],
                       orig_h=r["orig_h"], thumb_before=r["thumb_before"],
                       thumb_after=r["thumb_after"])
            planned = {f: strip_border(old[f]) for f in CAPTION_FIELDS}
            for f, (_, gone, sus) in planned.items():
                if sus:
                    suspects[f] = sus
            empty = [f for f, (cap, _, _) in planned.items()
                     if cap != old[f] and caption_is_empty(cap)]
            if r["image"] and empty:
                statuses.append("caption_empty")
                notes.append(f"stripping empties {empty}: not applied, "
                             "image dropped")
                (Path(images_dir) / f"{sid}.webp").unlink(missing_ok=True)
                row["thumb_after"] = None
            if r["image"] and not empty:
                row.update(bucket_w=r["bucket"][0], bucket_h=r["bucket"][1])
                for f, (cap, gone, _) in planned.items():
                    new[f], removed[f] = cap, gone
            else:
                for f, (_, gone, _) in planned.items():
                    if gone:
                        notes.append(f"{f}: would strip {gone} (not applied:"
                                     " no image)")
            for f, sus in suspects.items():
                notes.append(f"{f}: kept {sus}")
        if row["thumb_before"] is None and thumbs_dir:
            tp = Path(thumbs_dir) / f"{sid}.jpg"
            if tp.is_file():
                from PIL import Image
                with Image.open(tp) as im:
                    row["thumb_before"] = jpeg_data_uri(im)
        row.update({f: new[f] for f in CAPTION_FIELDS})
        row.update({f + "_old": old[f] for f in CAPTION_FIELDS})
        row.update(added=added, removed=removed, suspects=suspects,
                   statuses=statuses, notes=notes)
        rows.append(row)
    # recount tokens (one batch)
    caps, artists = [], []
    for row in rows:
        a = meta[row["id"]]["artist_name"] or ""
        for f in CAPTION_FIELDS:
            caps += [row[f], row[f + "_old"]]
            artists += [a, a]
    counts = iter(token_counts(template, caps, artists))
    budget = template.max_caption_tokens
    for row in rows:
        for f in CAPTION_FIELDS:
            n_new, n_old = next(counts), next(counts)
            row[f + "_tok"] = n_new
            if n_new > budget and n_new > n_old:
                row["statuses"].append("over_budget")
                row["notes"].append(f"{f}: {n_old} -> {n_new} tokens > "
                                    f"budget {budget}")
            elif n_new > budget:
                row["notes"].append(f"{f}: {n_new} tokens > budget {budget} "
                                    f"(already {n_old} in the dataset)")
        row["status"] = worst(row["statuses"])
        row["notes"] = "; ".join(row["notes"])
    return rows


def patch_table(rows):
    import pyarrow as pa

    schema = pa.schema([
        ("id", pa.int64()), ("action", pa.string()),
        ("bucket_w", pa.int64()), ("bucket_h", pa.int64()),
        *((f, pa.string()) for f in CAPTION_FIELDS),
        *((f + "_old", pa.string()) for f in CAPTION_FIELDS),
        *((f + "_tok", pa.int64()) for f in CAPTION_FIELDS),
        ("crop_box_orig", pa.list_(pa.int64())),
        ("orig_w", pa.int64()), ("orig_h", pa.int64()),
        ("status", pa.string()), ("notes", pa.string()),
    ])
    assert tuple(schema.names) == PATCH_COLUMNS
    return pa.Table.from_pylist(
        [{c: r[c] for c in PATCH_COLUMNS} for r in rows], schema=schema)


def tool_versions():
    out = {"python": platform.python_version()}
    for mod in ("PIL", "numpy", "pyarrow", "tokenizers", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    try:
        out["repo_commit"] = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        out["repo_commit"] = None
    return out


# ------------------------------------------------------------------ report
def diff_html(old, new):
    """Word-level diff of two captions: removed red (struck), added green."""
    a = re.findall(r"\s+|[^\s,]+|,", old or "")
    b = re.findall(r"\s+|[^\s,]+|,", new or "")
    out = []
    for op, i0, i1, j0, j1 in difflib.SequenceMatcher(
            None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(html.escape("".join(a[i0:i1])))
            continue
        if i1 > i0:
            out.append(f'<del>{html.escape("".join(a[i0:i1]))}</del>')
        if j1 > j0:
            out.append(f'<ins>{html.escape("".join(b[j0:j1]))}</ins>')
    return "".join(out)


REPORT_CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e1dc;
--del:#c62828;--delbg:#fde7e7;--ins:#1b7a33;--insbg:#e3f5e7;
--ok:#2e7d32;--warn:#b26a00;--bad:#c62828;--info:#1565c0}
@media (prefers-color-scheme:dark){:root{--bg:#181816;--fg:#e8e6e1;
--mut:#9a988f;--card:#22221f;--line:#3a3934;--del:#ff8a80;--delbg:#3d1f1f;
--ins:#8fd19e;--insbg:#1d3323}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,sans-serif}
h1{font-size:20px;margin:0 0 4px}.mut{color:var(--mut)}
.bar{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}
.bar button{border:1px solid var(--line);background:var(--card);color:var(--fg);
border-radius:14px;padding:3px 10px;cursor:pointer}
.bar button.on{border-color:var(--fg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px;margin:10px 0;display:flex;gap:14px;flex-wrap:wrap}
.imgs{display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap}
.imgs figure{margin:0}.imgs img{max-width:320px;max-height:320px;display:block;
border:1px solid var(--line)}
figcaption{font-size:12px;color:var(--mut)}
.txt{flex:1;min-width:260px}
.badge{display:inline-block;border-radius:10px;padding:1px 8px;font-size:12px;
color:#fff;margin-right:4px}
.s-ok{background:var(--ok)}.s-over_budget,.s-crop_unsure{background:var(--warn)}
.s-crop_too_small,.s-error{background:var(--bad)}
.a-caption{background:#b8860b}.a-crop{background:var(--info)}
.a-remove{background:#6d4c41}
.cap{margin:6px 0;padding:6px;border-left:3px solid var(--line);
word-break:break-word}.cap b{font-size:12px;color:var(--mut)}
del{color:var(--del);background:var(--delbg)}
ins{color:var(--ins);background:var(--insbg);text-decoration:none}
.notes{font-size:12px;color:var(--mut);white-space:pre-wrap}
table.c{border-collapse:collapse}table.c td{padding:2px 10px 2px 0}
"""

REPORT_JS = """
document.querySelectorAll('.bar button').forEach(function(b){
  b.addEventListener('click', function(){
    document.querySelectorAll('.bar button').forEach(function(x){
      x.classList.toggle('on', x === b);});
    var k = b.dataset.k;
    document.querySelectorAll('.card').forEach(function(c){
      c.style.display = (k === 'all' || c.dataset.status === k ||
                         c.dataset.action === k) ? '' : 'none';});
  });
});
"""


def render_report(rows, info):
    by = Counter((r["action"], r["status"]) for r in rows)
    keys = ["all", *ACTIONS, *[s for s in STATUS_ORDER
                               if any(r["status"] == s for r in rows)]]
    counts = "".join(
        f"<tr><td>{html.escape(a)}</td><td>{html.escape(s)}</td>"
        f"<td>{n}</td></tr>" for (a, s), n in sorted(by.items()))
    cards = []
    order = sorted(rows, key=lambda r: (STATUS_ORDER.index(r["status"]),
                                        ACTIONS.index(r["action"]), r["id"]))
    for r in order:
        figs = []
        if r["thumb_before"]:
            lab = ("original + crop box" if r["action"] == "crop"
                   else "dataset image")
            figs.append(f'<figure><img src="{r["thumb_before"]}" alt="">'
                        f"<figcaption>{lab}</figcaption></figure>")
        if r["thumb_after"]:
            figs.append(f'<figure><img src="{r["thumb_after"]}" alt="">'
                        f"<figcaption>final {r['bucket_w']}x{r['bucket_h']}"
                        "</figcaption></figure>")
        caps = "".join(
            f'<div class="cap"><b>{f} ({r[f + "_tok"]} tok)</b><br>'
            f'{diff_html(r[f + "_old"], r[f])}</div>' for f in CAPTION_FIELDS)
        notes = html.escape(r["notes"].replace("; ", "\n"))
        cards.append(
            f'<div class="card" data-status="{r["status"]}" '
            f'data-action="{r["action"]}"><div class="imgs">{"".join(figs)}'
            f'</div><div class="txt"><div><span class="badge a-{r["action"]}"'
            f'>{r["action"]}</span><span class="badge s-{r["status"]}">'
            f'{r["status"]}</span> <b>{r["id"]}</b> <span class="mut">'
            f'bucket {r["bucket_w"]}x{r["bucket_h"]}</span></div>{caps}'
            f'<div class="notes">{notes}</div></div></div>')
    buttons = "".join(f'<button data-k="{k}"{" class=on" if k == "all" else ""}'
                      f">{k}</button>" for k in keys)
    head = html.escape(json.dumps(info, ensure_ascii=False))
    return ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            '<meta name=viewport content="width=device-width,initial-scale=1">'
            f"<title>Datafilter patch</title><style>{REPORT_CSS}</style>"
            "</head><body><h1>Datafilter patch</h1>"
            f'<div class="mut">{head}</div>'
            f'<table class="c">{counts}</table><div class="bar">{buttons}'
            f'</div>{"".join(cards)}<script>{REPORT_JS}</script>'
            "</body></html>")


# ------------------------------------------------------------------ main
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--actions", required=True,
                   help="datafilter_actions.json exported by the review page")
    p.add_argument("--out", required=True, help="patch directory to write")
    p.add_argument("--metadata", default=str(LOCAL_METADATA))
    p.add_argument("--scores", default=str(LOCAL_REVIEW / "scores.parquet"))
    p.add_argument("--thumbs", default=str(LOCAL_REVIEW / "thumbs"),
                   help="review thumbnails for the report (optional)")
    p.add_argument("--originals", default=str(LOCAL_ORIGINALS),
                   help="directory of <id>.webp originals (fetch_originals)")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer.json (default: the local HF cache's "
                        "Qwen3-VL-4B-Instruct)")
    p.add_argument("--text-rows", type=int, default=DEFAULT_TEXT_ROWS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--measure-max-side", type=int, default=MEASURE_MAX_SIDE)
    p.add_argument("--inset", type=int, default=CROP_INSET_PX)
    p.add_argument("--unsure-ratio", type=float, default=UNSURE_RATIO)
    p.add_argument("--max-upscale", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing --out directory")
    return p


def lock_path(out):
    return out.with_name(out.name + ".lock")


def acquire_lock(out):
    """Exclusive non-blocking flock on <out>.lock (kept on disk) -> the open
    file (closing it releases the lock); InputError when another run holds
    it."""
    import fcntl

    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path(out), "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise InputError(f"{lock_path(out)} is held by another run")
    return f


def swap_in(tmp, out):
    """Move the finished tmp dir to out. An existing out is first renamed
    aside and deleted only once the new one is in place; a failed rename
    puts it back."""
    old = None
    if out.exists():
        old = out.with_name(f"{out.name}.old-{os.getpid()}")
        if old.exists():
            shutil.rmtree(old)
        os.replace(out, old)
    try:
        os.replace(tmp, out)
    except BaseException:
        if old is not None:
            os.replace(old, out)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def main(argv=None, log=print):
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    try:
        lock = acquire_lock(out)
    except InputError as e:
        log(f"error: {e}")
        return 1
    try:
        return _run(args, out, log)
    finally:
        lock.close()


def _run(args, out, log):
    from PIL import Image
    from tokenizers import Tokenizer

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        if out.exists() and not args.overwrite:
            raise InputError(f"{out} exists (pass --overwrite)")
        doc, items = load_actions(args.actions)
        ids = [it["id"] for it in items]
        meta, buckets = load_metadata(args.metadata, ids)
        scores, rev = load_scores(args.scores, ids)
        arev = doc.get("dataset_revision")
        if arev and rev and arev != rev:
            raise InputError(f"actions dataset_revision {arev} != "
                             f"scores.parquet {rev}")
        tok_path = find_tokenizer(args.tokenizer)
        if not tok_path.is_file():
            raise InputError(f"{tok_path} not found")
        template = CaptionTemplate(Tokenizer.from_file(str(tok_path)),
                                   text_rows=args.text_rows)
    except InputError as e:
        log(f"error: {e}")
        return 1
    log(f"{len(items)} acted ids ({dict(Counter(i['action'] for i in items))})"
        f", {len(buckets)} buckets, revision {rev}")
    tmp = out.with_name(f"{out.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "images").mkdir(parents=True)
    try:
        rows = build_rows(
            items, meta, scores, buckets, template, args.originals,
            tmp / "images", workers=args.workers,
            max_side=args.measure_max_side, inset=args.inset,
            unsure_ratio=args.unsure_ratio, max_upscale=args.max_upscale,
            thumbs_dir=args.thumbs, log=log)
        import pyarrow.parquet as pq

        pq.write_table(patch_table(rows), tmp / "patch.parquet")
        by_status = Counter(r["status"] for r in rows)
        info = {
            "format": FORMAT,
            "source_dataset_repo": msb.DATASET_REPO,
            "dataset_revision": rev,
            "source_dataset_revision": rev,
            "review_id": doc.get("review_id"),
            "actions_generated": doc.get("generated"),
            "actions_sha256": file_sha256(args.actions),
            "created": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "counts": {
                "rows": len(rows),
                "action": dict(Counter(r["action"] for r in rows)),
                "status": dict(by_status),
                "action_status": {f"{a}/{s}": n for (a, s), n in sorted(
                    Counter((r["action"], r["status"]) for r in rows)
                    .items())},
                "images": len(list((tmp / "images").glob("*.webp"))),
            },
            "params": {
                "measure_max_side": args.measure_max_side,
                "inset_px": args.inset, "unsure_ratio": args.unsure_ratio,
                "max_upscale": args.max_upscale,
                "webp_quality": WEBP_QUALITY,
                "resize": "cache_dataset.load_image (Lanczos cover resize + "
                          "center crop)",
                "buckets": [list(b) for b in buckets],
                "text_rows": args.text_rows,
                "caption_budget": template.max_caption_tokens,
                "tokens_include_artist_suffix": True,
            },
            "tokenizer": {"path": str(tok_path),
                          "sha256": file_sha256(tok_path)},
            "tool_versions": tool_versions(),
        }
        (tmp / "patch.json").write_text(json.dumps(info, indent=2) + "\n")
        (tmp / "report.html").write_text(
            render_report(rows, {k: info[k] for k in (
                "dataset_revision", "review_id", "created")}),
            encoding="utf-8")
        swap_in(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log(f"wrote {out}: {len(rows)} rows, status {dict(by_status)}, "
        f"{info['counts']['images']} images; report {out / 'report.html'}")
    return 2 if by_status.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
