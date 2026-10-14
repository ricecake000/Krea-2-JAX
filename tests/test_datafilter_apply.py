"""CPU tests of scripts/datafilter_apply.py (no network, no Modal, no HF):

  * bucket choice (nearest aspect in log space), the reused dataset resize
    (cache_dataset.load_image round trip == a direct PIL Lanczos cover
    resize + center crop, pixel exact)
  * crop geometry on synthetic originals: a downscaled measurement (> 2048
    px) maps back inside the true picture within a few px; sides without a
    margin are not inset; statuses ok / crop_too_small / crop_unsure /
    error (missing original); alpha composited on white
  * caption phrases: colour names, border / bars / tag-only / outside
    border phrases, append without duplicates; strip of real caption
    phrases (whole pure-border phrases, "with" / "framed by" tails,
    adjective drops, kept phrases with other content, quoted text,
    sentences inside a phrase); every generated phrase strips back
  * token budget: counts equal the CaptionTemplate path (caption + artist
    suffix), over_budget only when an edit lengthens past the budget
  * end to end through main(): patch.parquet schema, images only for crops
    with an image, patch.json counts, self-contained report.html, exit
    codes, --overwrite, revision mismatch / unknown ids rejected with
    nothing written

    python tests/test_datafilter_apply.py   # or: pytest tests/test_datafilter_apply.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tokenizers import Tokenizer

import datafilter_apply as da
import modal_score_borders as msb
from krea2_jax.text_template import PREFIX, CaptionTemplate
from test_qwen3_text import make_tiny_tokenizer
from test_score_borders import canvas, photo

REV = "05a9a13d4159363a5f5a5258ff3ae904506267bb"
BUCKETS = [(640, 1536), (640, 1600), (704, 1408), (704, 1472), (768, 1280),
           (768, 1344), (832, 1216), (896, 1152), (960, 1088), (1024, 1024),
           (1088, 960), (1152, 896), (1216, 832), (1280, 768), (1344, 768),
           (1408, 704), (1472, 704), (1536, 640), (1600, 640)]


# ------------------------------------------------------------------ helpers
def inset_original(h, w, box, color=(255, 255, 255), seed=1):
    """uint8 [h, w, 3]: flat canvas (light noise) with a photo in box
    (x0, y0, x1, y1)."""
    c = canvas(color, noise=2.0, h=h, w=w, seed=seed)
    x0, y0, x1, y1 = box
    c[y0:y1, x0:x1] = photo(y1 - y0, x1 - x0, seed)
    return c


def save_original(path, arr, lossless=True):
    Image.fromarray(arr).save(path, "WEBP", lossless=lossless, quality=95)


def score_row(h, w, px=(0, 0, 0, 0), straight=1.0, bg=(255, 255, 255),
              score=None, fill=0.95):
    """A scores.parquet row through modal_score_borders._finish."""
    st = (dict(straight) if isinstance(straight, dict)
          else {s: straight for s in msb.SIDES})
    r = msb._finish(h, w, px, fill, st, 1.0)
    r.update(h=h, w=w, bg_r=bg[0], bg_g=bg[1], bg_b=bg[2])
    r.pop("side_pattern")       # like the stored parquet (recomputed)
    if score is not None:
        r["score"] = score
    return r


def tokenizer(root):
    return make_tiny_tokenizer(Path(root) / "tok", extra_corpus=[
        "white border, black bars on the left and right, character "
        "extending outside the border, drawn by"] * 5)


# ------------------------------------------------------------------ buckets
def test_choose_bucket():
    assert da.choose_bucket(1000, 1000, BUCKETS) == (1024, 1024)
    assert da.choose_bucket(832, 1216, BUCKETS) == (832, 1216)
    assert da.choose_bucket(3000, 1000, BUCKETS) == (1600, 640)   # beyond
    assert da.choose_bucket(1000, 3000, BUCKETS) == (640, 1600)
    # nearest in log space, for every aspect between two buckets
    for a in np.linspace(0.35, 2.8, 97):
        w, h = 1000 * a, 1000
        got = da.choose_bucket(w, h, BUCKETS)
        best = min(abs(math.log(bw / bh) - math.log(a)) for bw, bh in BUCKETS)
        assert abs(abs(math.log(got[0] / got[1]) - math.log(a)) - best) < 1e-12
    assert da.cover_scale(1200, 900, 1152, 896) < 1.0
    assert da.cover_scale(1000, 800, 1152, 896) > 1.0


def test_resize_is_dataset_pipeline():
    """resize_to_bucket goes through cache_dataset.load_image and maps its
    [-1, 1] floats back to the exact uint8 of a direct PIL Lanczos cover
    resize + center crop."""
    img = Image.fromarray(photo(901, 1203, 3))
    for bw, bh in ((1152, 896), (832, 1216), (1024, 1024)):
        got = np.asarray(da.resize_to_bucket(img, bw, bh))
        s = max(bw / 1203, bh / 901)
        ref = img.resize((max(bw, round(1203 * s)), max(bh, round(901 * s))),
                         Image.LANCZOS)
        left, top = (ref.size[0] - bw) // 2, (ref.size[1] - bh) // 2
        ref = np.asarray(ref.crop((left, top, left + bw, top + bh)))
        assert got.shape == (bh, bw, 3)
        assert np.array_equal(got, ref), (bw, bh, np.abs(
            got.astype(int) - ref).max())


# ------------------------------------------------------------------ crops
def test_measure_box_downscaled():
    """Originals measured at 2048 px: the mapped box lies inside the true
    picture by >= the 1 px inset and loses at most 4 px per side (any
    phase of the picture edge against the downscale grid)."""
    for k, (h, w) in enumerate(((2100, 2600), (2600, 2311), (2049, 3001))):
        for truth in ((310, 205, w - 310, h - 203),
                      (311 + k, 206, w - 297, h - 199 - k),
                      (157, 401 + k, w - 158 - k, h - 402)):
            img = Image.fromarray(inset_original(h, w, truth, seed=k))
            box, m, (mw, mh) = da.measure_box(img)
            assert max(mw, mh) == 2048 and m["side_pattern"] == "4", m
            x0, y0, x1, y1 = box
            tx0, ty0, tx1, ty1 = truth
            assert tx0 + 1 <= x0 <= tx0 + 4 and ty0 + 1 <= y0 <= ty0 + 4, (
                box, truth)
            assert tx1 - 4 <= x1 <= tx1 - 1 and ty1 - 4 <= y1 <= ty1 - 1, (
                box, truth)
    # full resolution: exact + 1 px inset
    img = Image.fromarray(inset_original(1100, 1400, (100, 60, 1300, 1040)))
    box, _, size = da.measure_box(img)
    assert size == (1400, 1100) and box == (101, 61, 1299, 1039), box


def test_measure_box_no_inset_without_margin():
    """Letterbox (bars top and bottom only): left / right stay at 0 / W."""
    a = canvas((0, 0, 0), noise=1.0, h=1000, w=1400)
    a[150:850] = photo(700, 1400, 5)
    box, m, _ = da.measure_box(Image.fromarray(a))
    assert box == (0, 151, 1400, 849), (box, m)


def test_crop_one_statuses():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "img").mkdir()
        # ok: 1200x900 picture -> 1152x896, no upscaling
        save_original(d / "1.webp", inset_original(1100, 1400,
                                                  (100, 100, 1300, 1000)))
        r = da.crop_one(1, d / "1.webp", 0.3, BUCKETS, d / "img")
        assert r["status"] == "ok" and r["image"], r
        assert r["box"] == [101, 101, 1299, 999] and r["bucket"] == (1152, 896)
        assert (r["orig_w"], r["orig_h"]) == (1400, 1100)
        with Image.open(d / "img" / "1.webp") as im:
            assert im.size == (1152, 896) and im.format == "WEBP"
            out = np.asarray(im.convert("RGB")).astype(int)
        # the canvas is gone: no near-white rows / columns at the edges
        assert out[:2].mean() < 230 and out[-2:].mean() < 230
        assert out[:, :2].mean() < 230 and out[:, -2:].mean() < 230
        assert r["thumb_before"].startswith("data:image/jpeg;base64,")
        assert r["thumb_after"].startswith("data:image/jpeg;base64,")
        # crop_too_small: a 700x500 picture would need upscaling
        save_original(d / "2.webp", inset_original(800, 1000,
                                                  (150, 150, 850, 650)))
        r = da.crop_one(2, d / "2.webp", 0.3, BUCKETS, d / "img")
        assert r["status"] == "crop_too_small" and not r["image"], r
        assert r["box"] is not None and not (d / "img" / "2.webp").exists()
        r = da.crop_one(2, d / "2.webp", 0.3, BUCKETS, d / "img",
                        max_upscale=2.0)
        assert r["status"] == "ok" and r["image"], r
        # crop_unsure: a full-bleed original (score 0) still gets an image
        save_original(d / "3.webp", photo(1100, 1400, 9))
        r = da.crop_one(3, d / "3.webp", 0.3, BUCKETS, d / "img")
        assert r["status"] == "crop_unsure" and r["image"], r
        assert r["box"] == [0, 0, 1400, 1100]
        assert any("original scores 0" in n for n in r["notes"])
        # original much weaker than the dataset score -> unsure
        save_original(d / "4.webp", inset_original(1100, 1400,
                                                  (20, 20, 1380, 1080)))
        r = da.crop_one(4, d / "4.webp", 0.4, BUCKETS, d / "img")
        assert r["status"] == "crop_unsure", r
        # error: missing original
        r = da.crop_one(5, d / "nope.webp", 0.3, BUCKETS, d / "img")
        assert r["status"] == "error" and not r["image"] and r["box"] is None


def test_alpha_composited_on_white():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a = np.zeros((1100, 1400, 4), np.uint8)       # transparent black
        a[100:1000, 100:1300, :3] = photo(900, 1200, 2)
        a[100:1000, 100:1300, 3] = 255
        Image.fromarray(a, "RGBA").save(d / "7.webp", "WEBP", lossless=True)
        img, notes = da.open_original(d / "7.webp")
        assert img.mode == "RGB" and "alpha composited on white" in notes
        assert np.asarray(img)[0, 0].tolist() == [255, 255, 255]
        box, m, _ = da.measure_box(img)
        assert box == (101, 101, 1299, 999) and m["bg_white"], (box, m)


# ------------------------------------------------------------------ phrases
def test_colour_name():
    assert da.colour_name((255, 255, 255)) == "white"
    assert da.colour_name((240, 238, 236)) == "white"
    assert da.colour_name((10, 12, 9)) == "black"
    assert da.colour_name((128, 128, 128)) == "grey"
    assert da.colour_name((215, 215, 218)) == "light grey"
    assert da.colour_name((230, 30, 30)) == "red"
    assert da.colour_name((250, 190, 210)) == "pale pink"
    assert da.colour_name((40, 60, 160)) == "blue"
    assert da.colour_name((20, 30, 90)) == "dark blue"
    assert da.colour_name((110, 70, 30)) == "brown"


def test_border_phrases():
    four = score_row(1216, 832, (60, 60, 40, 40))
    assert da.border_phrases(four, "1girl, solo")[0] == ["white border"]
    pillar = score_row(1024, 1024, (0, 0, 120, 120), bg=(5, 5, 5))
    assert da.border_phrases(pillar, "")[0] == [
        "black bars on the left and right"]
    letter = score_row(1024, 1024, (100, 100, 0, 0), bg=(250, 200, 215))
    assert da.border_phrases(letter, "")[0] == [
        "pale pink bars on the top and bottom"]
    # pillar flag (two >= 1.5% sides) but thin bands on all 4 -> a border
    thin4 = score_row(1024, 1024, (5, 5, 120, 120))
    assert thin4["pillar_or_letter"]
    assert da.border_phrases(thin4, "")[0] == ["white border"]
    flat = score_row(1216, 832)                     # no measured border
    assert da.border_phrases(flat, "border, white border")[0] == [
        "white border"]
    assert da.border_phrases(flat, "border")[0] == ["border around the image"]
    assert da.border_phrases(flat, "pillarboxed")[0] == [
        "bars on the left and right"]
    assert da.border_phrases(four, "border, outside border")[0] == [
        "white border", "character extending outside the border"]


def test_add_phrases():
    cap = "anime illustration of a girl, smiling, white background"
    new, added, skipped = da.add_phrases(cap, ["white border"])
    assert new == cap + ", white border" and added == ["white border"]
    assert da.add_phrases(cap + ", ", ["white border"])[0] == (
        cap + ", white border")
    # already mentions a border: not duplicated (per caption)
    has = "a girl, thin white border framing the image, soft shading"
    new, added, skipped = da.add_phrases(
        has, ["white border", "character extending outside the border"])
    assert new == has + ", character extending outside the border"
    assert added == ["character extending outside the border"]
    assert skipped and "already mentions a border" in skipped[0]
    out = "a girl, hair breaking out of the frame"
    assert da.add_phrases(out, ["character extending outside the border"])[
        0] == out


STRIP_CASES = [
    # whole pure-border phrases
    ("a girl, white border, soft shading", "a girl, soft shading"),
    ("a girl, soft shading, white border around the image",
     "a girl, soft shading"),
    ("a girl, thin white border framing the image, soft shading",
     "a girl, soft shading"),
    ("white border, a girl", "a girl"),
    ("a girl, letterboxed with black bars top and bottom", "a girl"),
    ("a girl, black letterbox bars at top and bottom, rain", "a girl, rain"),
    ("a girl, pillarboxed", "a girl"),
    ("a girl, framed by a thin white border", "a girl"),
    # tails and adjectives
    ("a girl, plain grey background with white border",
     "a girl, plain grey background"),
    ("a girl, cyan background with white star motifs and pillarbox border",
     "a girl, cyan background with white star motifs"),
    ("a girl, evening sky background framed by black letterbox bars",
     "a girl, evening sky background"),
    ("a girl, pink framed background", "a girl, pink background"),
    ("a girl, simple bordered background", "a girl, simple background"),
    ("a girl, pale yellow rectangular framed background with white border",
     "a girl, pale yellow rectangular background"),
    # sentences inside a comma phrase
    ("a girl, and a red skirt. solid white background with black pillarbox"
     " borders. sharp style", "a girl, and a red skirt. solid white "
     "background. sharp style"),
    # possessive / contraction apostrophes are not quotes
    ("a girl, the girl's portrait with white border, it's raining",
     "a girl, the girl's portrait, it's raining"),
    ("a girl, red skirt. white letterbox border", "a girl, red skirt."),
]
KEEP_CASES = [
    "a girl walking a border collie dog, park",
    "a girl, vanity mirror bordered with glowing round marquee bulbs",
    "a girl, chibi inset in the top right, reference inset",
    "a girl, red-framed glasses, picture frames on the wall",
    "a girl, artist signature in the right margin",
    "a girl, text reading 'WHITE BORDER' in the center",
    "a girl, white photo frame border placed on a corkboard surface",
    # quoted text spanning commas is never touched
    'a girl, text reading "HELLO, white border, WORLD", soft shading',
    "a girl, sign reading 'OPEN, white border, CLOSED' in the top left",
    "a girl, \u201cwhite border, thin frame\u201d logo",
    "a girl, wearing a dress with a white-bordered collar",
]


def test_strip_border():
    for cap, want in STRIP_CASES:
        got, removed, _ = da.strip_border(cap)
        assert got == want, (cap, got, want)
        assert removed, cap
    for cap in KEEP_CASES:
        got, removed, kept = da.strip_border(cap)
        assert got == cap and not removed, (cap, got, removed)
    _, _, kept = da.strip_border(KEEP_CASES[1])
    assert any("other content" in k for k in kept)
    # a phrase holding a quote is kept whole, listed when it has border
    # words outside the quotes
    cap = "a girl, white border with the logo 'KREA'"
    got, removed, kept = da.strip_border(cap)
    assert got == cap and not removed and any("quoted" in k for k in kept)
    assert da.strip_border("") == ("", [], [])
    # untouched captions are byte-identical (odd spacing kept)
    odd = "a girl ,  smiling,x ,y"
    assert da.strip_border(odd)[0] == odd


def test_added_phrases_strip_back():
    cap = "anime illustration of a girl, smiling, soft shading"
    rows = [score_row(1216, 832, (60, 60, 40, 40)),
            score_row(1024, 1024, (0, 0, 120, 120), bg=(5, 5, 5)),
            score_row(1024, 1024, (100, 100, 0, 0), bg=(250, 200, 215)),
            score_row(1216, 832)]
    for r in rows:
        for tags in ("", "border", "white border", "pillarboxed",
                     "border, outside border"):
            phrases, _ = da.border_phrases(r, tags)
            added = da.add_phrases(cap, phrases)[0]
            assert added != cap
            assert da.strip_border(added)[0] == cap, (phrases, added)


# ------------------------------------------------------------------ tokens
def test_token_counts_match_template():
    with tempfile.TemporaryDirectory() as d:
        tmpl = CaptionTemplate(tokenizer(d), text_rows=40)
        caps = ["a girl, white border", "", "smiling"]
        arts = ["someone", "", "x"]
        got = da.token_counts(tmpl, caps, arts)
        for c, a, n in zip(caps, arts, got):
            text = c + (", drawn by " + a if a else "")
            ids, txt_len = tmpl.encode_captions([text])
            full = len(tmpl.tokenizer.encode(PREFIX + text).ids)
            assert n == full - tmpl.prefix_len
            if n <= tmpl.max_caption_tokens:
                assert txt_len[0] == n + len(tmpl.suffix_ids)


# ------------------------------------------------------------------ e2e
CAP = "anime illustration of a girl, smiling, soft shading"


def make_inputs(root, n_extra=0):
    """Fake metadata.parquet (every bucket present), scores.parquet (with
    the revision), originals, thumbs, actions json -> paths."""
    root = Path(root)
    rows = []
    for i, (bw, bh) in enumerate(BUCKETS):
        rows.append({"id": 1000 + i, "bucket_w": bw, "bucket_h": bh})
    acted = {
        1: ("crop", (1216, 832), CAP + ", white border", "", "artist_a"),
        2: ("crop", (1216, 832), CAP + ", white border framing the image",
            "", ""),
        3: ("crop", (832, 1216), CAP, "", ""),                   # missing
        4: ("caption", (832, 1216), CAP, "border, white border", "artist_b"),
        5: ("caption", (1344, 768), CAP + ", black bars on the left and "
            "right", "pillarboxed", ""),
        6: ("remove", (1024, 1024), CAP, "sample watermark", ""),
        7: ("caption", (832, 1216), CAP, "border, outside border", ""),
    }
    for sid, (act, (bw, bh), cap, tags, artist) in acted.items():
        rows.append({"id": sid, "bucket_w": bw, "bucket_h": bh,
                     "tags": tags, "artist_name": artist,
                     "long": cap, "medium": cap, "short": "a girl"})
    for r in rows:
        r.setdefault("tags", "1girl")
        r.setdefault("artist_name", "")
        for f in ("long", "medium", "short"):
            r.setdefault(f, "a girl")
        for f in ("long", "medium", "short"):
            r[f + "_tok"] = 7
        r["tags_tok"] = 3
        r["tar"] = "data/train-0000.tar"
    pq.write_table(pa.Table.from_pylist(rows), root / "metadata.parquet")
    srows = []
    for r in rows:
        h, w = r["bucket_h"], r["bucket_w"]
        if r["id"] in (1, 2, 4):
            s = score_row(h, w, (60, 60, 40, 40))
        elif r["id"] == 5:
            s = score_row(h, w, (0, 0, 100, 100), bg=(3, 3, 3))
        else:
            s = score_row(h, w)
        s["id"] = r["id"]
        srows.append(s)
    t = pa.Table.from_pylist(srows)
    t = t.replace_schema_metadata({"dataset_repo": msb.DATASET_REPO,
                                   "dataset_revision": REV})
    pq.write_table(t, root / "scores.parquet")
    orig = root / "orig"
    orig.mkdir()
    save_original(orig / "1.webp", inset_original(1100, 1400,
                                                 (100, 100, 1300, 1000)))
    save_original(orig / "2.webp", inset_original(800, 1000,
                                                 (150, 150, 850, 650)))
    thumbs = root / "thumbs"
    thumbs.mkdir()
    Image.fromarray(photo(300, 200, 4)).save(thumbs / "4.jpg", "JPEG")
    actions = {"review_id": "rid1", "generated": "2026-10-14T00:00:00Z",
               "dataset_revision": REV,
               "items": [{"id": sid, "action": a[0], "tabs": ["A"],
                          "score": 0.2, "side_pattern": "4"}
                         for sid, a in acted.items()]}
    (root / "actions.json").write_text(json.dumps(actions))
    tokenizer(root)
    return {"--actions": root / "actions.json",
            "--metadata": root / "metadata.parquet",
            "--scores": root / "scores.parquet", "--originals": orig,
            "--thumbs": thumbs, "--tokenizer": root / "tok" / "tokenizer.json"}


def argv(paths, out, *extra):
    a = []
    for k, v in paths.items():
        a += [k, str(v)]
    return a + ["--out", str(out), "--workers", "2", *extra]


def test_main_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        paths = make_inputs(d)
        out = d / "patch"
        rc = da.main(argv(paths, out, "--text-rows", "200"), log=lambda *a: 0)
        assert rc == 2                                 # id 3: missing original
        t = pq.read_table(out / "patch.parquet")
        assert tuple(t.schema.names) == da.PATCH_COLUMNS
        assert t.schema.field("crop_box_orig").type == pa.list_(pa.int64())
        rows = {r["id"]: r for r in t.to_pylist()}
        assert list(rows) == [1, 2, 3, 4, 5, 6, 7]
        # 1: crop ok -> new bucket, border phrase stripped, image
        r = rows[1]
        assert r["status"] == "ok" and r["action"] == "crop", r
        assert (r["bucket_w"], r["bucket_h"]) == (1152, 896)
        assert r["crop_box_orig"] == [101, 101, 1299, 999]
        assert (r["orig_w"], r["orig_h"]) == (1400, 1100)
        assert r["long"] == CAP and r["long_old"] == CAP + ", white border"
        assert r["short"] == "a girl"
        # 2: too small -> old bucket + old captions, no image, box recorded
        r = rows[2]
        assert r["status"] == "crop_too_small", r
        assert (r["bucket_w"], r["bucket_h"]) == (1216, 832)
        assert r["long"] == r["long_old"] and r["crop_box_orig"] is not None
        assert "would strip" in r["notes"]
        # 3: missing original
        r = rows[3]
        assert r["status"] == "error" and r["crop_box_orig"] is None
        assert r["orig_w"] is None and "original missing" in r["notes"]
        # 4: caption fix appended to long / medium / short
        r = rows[4]
        assert r["status"] == "ok" and r["long"] == CAP + ", white border"
        assert r["short"] == "a girl, white border"
        assert r["crop_box_orig"] is None and r["orig_w"] is None
        assert (r["bucket_w"], r["bucket_h"]) == (832, 1216)
        # 5: already mentions bars -> unchanged
        r = rows[5]
        assert r["long"] == r["long_old"] and r["short"] == "a girl, black "\
            "bars on the left and right"
        # 6: remove recorded, captions unchanged
        r = rows[6]
        assert r["action"] == "remove" and r["long"] == r["long_old"]
        # 7: tag-only border + outside border
        assert rows[7]["long"] == (CAP + ", border around the image, "
                                   "character extending outside the border")
        # tokens recounted with the artist suffix (the --tokenizer file)
        tok = Tokenizer.from_file(str(paths["--tokenizer"]))
        plen = len(tok.encode(PREFIX).ids)
        for sid in (1, 4, 6):
            artist = {1: "artist_a", 4: "artist_b", 6: ""}[sid]
            for f in ("long", "medium", "short"):
                text = rows[sid][f] + (", drawn by " + artist if artist
                                       else "")
                assert rows[sid][f + "_tok"] == len(
                    tok.encode(PREFIX + text).ids) - plen
        # images: exactly the crops with an image
        imgs = sorted(p.name for p in (out / "images").iterdir())
        assert imgs == ["1.webp"]
        info = json.loads((out / "patch.json").read_text())
        assert info["source_dataset_revision"] == REV
        assert info["dataset_revision"] == REV       # the builder's key
        assert info["review_id"] == "rid1"
        assert info["counts"]["rows"] == 7 and info["counts"]["images"] == 1
        assert info["counts"]["status"]["error"] == 1
        assert info["counts"]["action"] == {"crop": 3, "caption": 3,
                                            "remove": 1}
        assert info["params"]["caption_budget"] == 195
        assert len(info["tokenizer"]["sha256"]) == 64
        page = (out / "report.html").read_text()
        assert "http://" not in page and "https://" not in page
        assert page.count('class="card"') == 7
        assert "<del>" in page and "<ins>" in page
        assert page.count("data:image/jpeg;base64,") >= 4   # 1 x2, 2, 4
        for s in ("crop_too_small", "error", "ok"):
            assert f"s-{s}" in page
        assert not list(d.glob("patch.tmp-*"))
        # an existing --out is refused without --overwrite (nothing touched)
        before = (out / "patch.parquet").read_bytes()
        assert da.main(argv(paths, out), log=lambda *a: 0) == 1
        assert (out / "patch.parquet").read_bytes() == before
        # --overwrite: id 3's original appears -> no error
        save_original(paths["--originals"] / "3.webp",
                      inset_original(1500, 1100, (60, 80, 1040, 1420)))
        assert da.main(argv(paths, out, "--overwrite"), log=lambda *a: 0) == 0
        rows = {r["id"]: r for r in pq.read_table(
            out / "patch.parquet").to_pylist()}
        assert rows[3]["status"] == "ok" and rows[3]["bucket_w"] == 896
        assert sorted(p.name for p in (out / "images").iterdir()) == [
            "1.webp", "3.webp"]


def test_over_budget():
    """An appended phrase that pushes a caption past the budget ->
    over_budget; a caption over budget that only shrinks is not."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        paths = make_inputs(d)
        tok = Tokenizer.from_file(str(paths["--tokenizer"]))
        tmpl = CaptionTemplate(tok, text_rows=40)
        n4, n4b = da.token_counts(tmpl, [CAP, CAP + ", white border"],
                                  ["artist_b"] * 2)
        n1, n1b = da.token_counts(tmpl, [CAP + ", white border", CAP],
                                  ["artist_a"] * 2)
        assert n4 < n4b and n1b < n1
        # budget = id 4's long caption before the phrase (fits), after: not
        text_rows = n4 + len(tmpl.suffix_ids)
        out = d / "patch"
        da.main(argv(paths, out, "--text-rows", str(text_rows)),
                log=lambda *a: 0)
        rows = {r["id"]: r for r in pq.read_table(
            out / "patch.parquet").to_pylist()}
        assert rows[4]["status"] == "over_budget", rows[4]
        assert rows[4]["long_tok"] == n4b
        assert f"long: {n4} -> {n4b} tokens > budget {n4}" in rows[4]["notes"]
        # id 1: long over budget before (n1 > n4b > n4) and stripped
        assert n1 > n4 and rows[1]["long_tok"] == n1b
        assert rows[1]["status"] == "ok", rows[1]


def test_rejects_bad_inputs():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        paths = make_inputs(d)
        out = d / "patch"
        doc = json.loads(paths["--actions"].read_text())
        bad = dict(doc, dataset_revision="0" * 40)
        (d / "a1.json").write_text(json.dumps(bad))
        assert da.main(argv({**paths, "--actions": d / "a1.json"}, out),
                       log=lambda *a: 0) == 1
        bad = dict(doc, items=doc["items"] + [{"id": 999999,
                                               "action": "remove"}])
        (d / "a2.json").write_text(json.dumps(bad))
        assert da.main(argv({**paths, "--actions": d / "a2.json"}, out),
                       log=lambda *a: 0) == 1
        bad = dict(doc, items=doc["items"] + [doc["items"][0]])
        (d / "a3.json").write_text(json.dumps(bad))
        assert da.main(argv({**paths, "--actions": d / "a3.json"}, out),
                       log=lambda *a: 0) == 1
        bad = dict(doc, items=[{"id": 1, "action": "keep"}])
        (d / "a4.json").write_text(json.dumps(bad))
        assert da.main(argv({**paths, "--actions": d / "a4.json"}, out),
                       log=lambda *a: 0) == 1
        assert not out.exists() and not list(d.glob("patch.tmp-*"))


def test_quote_aware_split():
    cap = 'a, text reading "x, white border, y", b\'s, c'
    segs = da.split_segments(cap)
    assert ",".join(x for x, _ in segs) == cap
    assert [q for _, q in segs] == [False, True, False, False]
    assert da.quote_spans("the girl's hat, it's fine") == []
    assert da.unquoted("say 'white border' now") == "say " + " " * 14 + " now"
    # a quoted border word is not a mention: the phrase is still added
    cap = "a girl, text reading 'WHITE BORDER'"
    assert da.add_phrases(cap, ["white border"])[0] == cap + ", white border"


def test_quote_round2_repros():
    # a curly apostrophe inside a word does not close a curly quote
    cap = ("a girl, sign reading \u2018girl\u2019s hat, white border, "
           "OPEN\u2019, white border")
    new, removed, kept = da.strip_border(cap)
    assert new == ("a girl, sign reading \u2018girl\u2019s hat, white "
                   "border, OPEN\u2019"), new
    assert removed == ["white border"]
    assert any("quoted text" in k for k in kept)
    assert not da.quote_info(cap)[1]
    # a single-quote pair spanning a newline
    cap = ("a girl, text reading 'HELLO,\nwhite border, WORLD', soft "
           "shading, white border")
    new, removed, _ = da.strip_border(cap)
    assert new == ("a girl, text reading 'HELLO,\nwhite border, WORLD', "
                   "soft shading"), new
    # straight apostrophes between letters inside a straight quote
    cap = "a girl, text reading 'it's a white border, ok', white border"
    assert da.strip_border(cap)[0] == ("a girl, text reading 'it's a white "
                                       "border, ok'")


def test_quote_backstop_unbalanced():
    """Unbalanced / ambiguous quote characters: nothing between the first
    and the last quote-like character is stripped; it is listed instead."""
    cap = ("a girl, the girls' hats, white border, rock 'n roll, smiling, "
           "white border")
    spans, ambiguous = da.quote_info(cap)
    assert ambiguous
    new, removed, kept = da.strip_border(cap)
    # the white border between the two stray quotes stays; the last one
    # (after the last quote-like character) goes
    assert new == "a girl, the girls' hats, white border, rock 'n roll, " \
        "smiling", new
    assert removed == ["white border"]
    assert any("unbalanced" in k and "white border" in k for k in kept)
    # an unclosed double quote protects to the end
    cap = 'a girl, text reading "OPEN, white border, smile, white border'
    new, removed, kept = da.strip_border(cap)
    assert new == cap and not removed and da.quote_info(cap)[1]
    assert any("unbalanced" in k for k in kept)
    # crossing quote pairs are ambiguous too
    cap = "a, 'x \"y' z\", white border, w"
    assert da.quote_info(cap)[1]


def test_mention_is_canvas_vocabulary_only():
    for cap in ("a girl, white frame on the wall",
                "a girl, thin frame glasses, wide margins of a notebook",
                "a girl, simple framed background"):
        new, added, _ = da.add_phrases(cap, ["white border"])
        assert added == ["white border"] and new == cap + ", white border"
    for cap in ("a girl, white border", "a girl, letterboxed",
                "a girl, black bars on the left and right",
                "a girl, pillarbox format"):
        assert da.add_phrases(cap, ["white border"])[0] == cap, cap


def test_bar_direction_from_qualifying_sides():
    # left / right qualify (straight); top / bottom margins are larger but
    # not straight (< SIDE_STRAIGHT) -> 2-opp on left / right
    st = {"top": 0.1, "bottom": 0.1, "left": 1.0, "right": 1.0}
    r = score_row(1024, 1024, (200, 200, 60, 60), straight=st)
    assert msb.row_side_pattern(r) == "2-opp"
    assert da.border_phrases(r, "")[0] == ["white bars on the left and right"]
    st = {"top": 1.0, "bottom": 1.0, "left": 0.1, "right": 0.1}
    r = score_row(1024, 1024, (40, 40, 200, 200), straight=st)
    assert msb.row_side_pattern(r) == "2-opp"
    assert da.border_phrases(r, "")[0] == ["white bars on the top and bottom"]


def test_icc_before_alpha_composite():
    """ICC -> sRGB runs on the RGBA image, then the composite on white: a
    transparent pixel ends up pure white whatever the transform does."""
    from PIL import ImageCms

    calls = []

    def fake_p2p(im, src, dst, outputMode=None, **kw):
        calls.append((im.mode, outputMode))
        a = np.asarray(im).copy()
        a[..., :3] = 255 - a[..., :3]          # a visible "transform"
        return Image.fromarray(a, im.mode)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a = np.zeros((64, 80, 4), np.uint8)
        a[16:48, 20:60] = (10, 200, 30, 255)
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        Image.fromarray(a, "RGBA").save(d / "x.webp", "WEBP", lossless=True,
                                        icc_profile=icc)
        old = (ImageCms.profileToProfile, ImageCms.getProfileDescription)
        ImageCms.profileToProfile = fake_p2p
        ImageCms.getProfileDescription = lambda prof: "Display P3"
        try:
            img, notes = da.open_original(d / "x.webp")
        finally:
            ImageCms.profileToProfile, ImageCms.getProfileDescription = old
        assert calls == [("RGBA", "RGBA")], calls
        px = np.asarray(img)
        assert px[0, 0].tolist() == [255, 255, 255]            # composited after
        assert px[30, 40].tolist() == [245, 55, 225]           # transformed
        assert notes == ["ICC 'Display P3' converted to sRGB",
                         "alpha composited on white"]


def test_exif_orientation_applied():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a = np.full((60, 100, 3), 128, np.uint8)
        a[:, :10] = (255, 0, 0)                 # red strip on the left
        im = Image.fromarray(a)
        ex = im.getexif()
        ex[0x0112] = 6                          # rotate 90 CW on display
        im.save(d / "o.webp", "WEBP", lossless=True, exif=ex.tobytes())
        img, notes = da.open_original(d / "o.webp")
        assert img.size == (60, 100), img.size
        assert np.asarray(img)[:10, 30].tolist()[0] == [255, 0, 0]  # now top
        assert "EXIF orientation 6 applied" in notes


def test_caption_empty_not_applied():
    """A crop whose stripped caption would be empty keeps its old captions
    and bucket, its image is dropped, status caption_empty."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "img").mkdir()
        tmpl = CaptionTemplate(tokenizer(d))
        meta = {9: {"bucket_w": 832, "bucket_h": 1216, "tags": "",
                    "artist_name": "", "long": "white border",
                    "medium": "a girl, white border", "short": "the border"}}
        scores = {9: score_row(1216, 832, (60, 60, 40, 40))}

        def fake_crop(sid, orig, ds, buckets, images_dir, **kw):
            (Path(images_dir) / f"{sid}.webp").write_bytes(b"x")
            return {"status": "ok", "notes": [], "box": [1, 1, 900, 900],
                    "orig_w": 1000, "orig_h": 1000, "bucket": (1024, 1024),
                    "image": True, "thumb_before": None, "thumb_after": "u"}

        old = da.crop_one
        da.crop_one = fake_crop
        try:
            rows = da.build_rows([{"id": 9, "action": "crop"}], meta, scores,
                                 BUCKETS, tmpl, d, d / "img", workers=1,
                                 log=lambda *a: 0)
        finally:
            da.crop_one = old
        r = rows[0]
        assert r["status"] == "caption_empty", r
        assert (r["long"], r["medium"], r["short"]) == (
            "white border", "a girl, white border", "the border")
        assert (r["bucket_w"], r["bucket_h"]) == (832, 1216)
        assert not (d / "img" / "9.webp").exists() and r["thumb_after"] is None
        assert da.caption_is_empty("") and da.caption_is_empty(" the, a ")
        assert not da.caption_is_empty("a girl")


def test_output_swap_keeps_old_and_lock():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        paths = make_inputs(d)
        out = d / "patch"
        assert da.main(argv(paths, out), log=lambda *a: 0) == 2
        before = (out / "patch.parquet").read_bytes()
        # the final rename fails: the previous patch stays in place
        real = da.os.replace

        def failing(src, dst):
            if Path(src).name.startswith("patch.tmp-"):
                raise OSError("simulated rename failure")
            return real(src, dst)

        da.os.replace = failing
        try:
            try:
                da.main(argv(paths, out, "--overwrite"), log=lambda *a: 0)
                raise AssertionError("expected the rename failure")
            except OSError:
                pass
        finally:
            da.os.replace = real
        assert (out / "patch.parquet").read_bytes() == before
        assert sorted(x.name for x in d.glob("patch*")) == [
            "patch", "patch.lock"]
        # a concurrent run (lock held) is refused, nothing touched
        import fcntl

        with open(d / "patch.lock", "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert da.main(argv(paths, out, "--overwrite"),
                           log=lambda *a: 0) == 1
        assert (out / "patch.parquet").read_bytes() == before
        # lock released: --overwrite replaces it
        assert da.main(argv(paths, out, "--overwrite"), log=lambda *a: 0) == 2
        assert sorted(x.name for x in d.glob("patch*")) == [
            "patch", "patch.lock"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
