"""CPU tests of scripts/modal_score_borders.py without Modal or network:

  * border_metrics on synthetic images, each with mild canvas noise and a
    WebP round trip (PIL, q80) to emulate the dataset's compression:
    rectangular photo-like insets on white (several sizes / aspect) score
    high with straightness ~1; character-like blob silhouettes on white
    (floating, cut off by the bottom edge) score near 0; full-bleed -> 0;
    black pillarbox / letterbox -> pillar_or_letter (despite a ring that is
    only part bars); thin dark frame line around a full image -> tiny score,
    thin_frame only; frame line + white mat + inset -> peeled, high score;
    light-grey / coloured canvas ~ white; busy edges -> 0; blank canvas
  * review round 1: canvas colour chosen by score over all candidates (a
    letterbox whose picture has flat side edges); geometric-mean
    straightness (a bust with one flat cut stays below a 3% inset, an inset
    with one weaker side stays high); tar member parsing and per-shard id
    validation (missing / extra / duplicate -> the run fails, nothing
    written); global two-pass thumbnail selection (exactly the review
    page's ids, a missing one fails); localStorage key scoped to the review
    id, foreign ids dropped on load / export (the page's logic run in node)
  * tag / caption flags, the always-thumb plan (watermark,
    border-uncaptioned, seeded white-background control)
  * dry run: a fake tar of synthetic WebPs (+ json/txt members) through the
    per-shard scorer, pass 2, then the join + scores.parquet + review page
    writer (files exist, inline JSON parses, tabs, thumbs, no external
    URLs), and the local CLI (--tar-dir) end to end

    python tests/test_score_borders.py      # or: pytest tests/test_score_borders.py
"""

import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
from PIL import Image, ImageDraw

import modal_score_borders as msb

H, W = 1216, 832   # the most common bucket (portrait)


# ------------------------------------------------------------------ images
def photo(h, w, seed):
    """Smooth colourful texture + noise, values kept away from white/black."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        a, b, p = r.uniform(1, 6, 3)
        img[..., c] = 120 + 70 * np.sin(a * xx * 6 + b * yy * 5 + p)
    img += r.normal(0, 12, img.shape)
    return np.clip(img, 25, 215).astype(np.uint8)


def canvas(color=(255, 255, 255), noise=4.0, h=H, w=W, seed=0):
    c = np.empty((h, w, 3), np.float32)
    c[:] = color
    c += np.random.default_rng(seed).normal(0, noise, c.shape)
    return np.clip(c, 0, 255).astype(np.uint8)


def webp_bytes(a, q=80):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "WEBP", quality=q)
    return b.getvalue()


def webp(a, q=80):
    return msb.decode_rgb(webp_bytes(a, q))


def inset(mt_frac, ml_frac, color=(255, 255, 255), h=H, w=W, seed=1):
    c = canvas(color, h=h, w=w)
    mt, ml = int(h * mt_frac), int(w * ml_frac)
    c[mt:h - mt, ml:w - ml] = photo(h - 2 * mt, w - 2 * ml, seed)
    return webp(c)


def silhouette(seed, touch_bottom):
    """Character-like blob on white: hair + head + torso + arms + legs as
    ellipses (round outlines, no straight edges)."""
    r = np.random.default_rng(seed)
    im = Image.fromarray(canvas(seed=seed))
    d = ImageDraw.Draw(im)
    skin = tuple(int(x) for x in r.integers(150, 215, 3))
    cloth = tuple(int(x) for x in r.integers(30, 180, 3))
    hair = tuple(int(x) for x in r.integers(20, 120, 3))
    cx = W // 2 + int(r.integers(-60, 60))
    top = int(r.integers(80, 200))
    d.ellipse([cx - 170, top, cx + 170, top + 230], fill=hair)
    d.ellipse([cx - 120, top + 60, cx + 120, top + 300], fill=skin)
    bottom = H + 200 if touch_bottom else H - int(r.integers(60, 160))
    d.ellipse([cx - 210, top + 280, cx + 210, top + 820], fill=cloth)
    d.ellipse([cx - 330, top + 330, cx - 180, top + 700], fill=skin)   # arms
    d.ellipse([cx + 180, top + 330, cx + 320, top + 640], fill=skin)
    d.ellipse([cx - 170, top + 700, cx - 20, bottom], fill=cloth)     # legs
    d.ellipse([cx + 20, top + 700, cx + 170, bottom], fill=cloth)
    return webp(np.asarray(im))


# ------------------------------------------------------------------ metric
def test_rect_inset_beats_silhouette():
    insets = {
        "3%": inset(0.03, 0.03), "8%": inset(0.08, 0.08),
        "15%x10%": inset(0.15, 0.10), "25%x5%": inset(0.25, 0.05),
        "landscape 6%": inset(0.06, 0.06, h=832, w=1216),
    }
    for name, img in insets.items():
        m = msb.border_metrics(img)
        assert m["ring_uniform"] >= 0.99, (name, m)
        assert m["edge_straightness"] >= 0.95, (name, m)
        assert m["n_sides"] == 4 and m["fill"] >= 0.95, (name, m)
        assert m["bg_white"] and not m["bg_black"], (name, m)
        assert m["score"] >= 0.1, (name, m)
        assert not m["pillar_or_letter"] and not m["thin_frame"], (name, m)
    # margins measured within a few px of the truth
    m = msb.border_metrics(insets["15%x10%"])
    assert abs(m["margin_top"] - 0.15) < 4 / H and abs(m["margin_left"] - 0.10) < 4 / W, m
    # score ~ margin area for a full rectangle
    assert abs(m["score"] - m["margin_area_frac"]) < 0.02, m
    sil = {}
    for seed in range(4):
        sil[f"float{seed}"] = silhouette(seed, touch_bottom=False)
        sil[f"cut{seed}"] = silhouette(seed, touch_bottom=True)
    worst = 0.0
    for name, img in sil.items():
        m = msb.border_metrics(img)
        # silhouettes keep ~0.1 straightness (the tip of a round outline
        # within EDGE_PX lines) -> ~0.02-0.035 with a 30-40% margin area
        assert m["score"] < 0.04, (name, m)
        assert m["edge_straightness"] < 0.2, (name, m)
        worst = max(worst, m["score"])
    best_inset_min = min(msb.border_metrics(i)["score"] for i in insets.values())
    assert best_inset_min > 2.5 * worst, (best_inset_min, worst)
    # a thin (1.6%) white border all round still outranks every silhouette
    assert msb.border_metrics(inset(0.016, 0.016))["score"] > 1.2 * worst


def test_full_bleed_and_busy_edges_zero():
    m = msb.border_metrics(webp(photo(H, W, 5)))
    assert m["score"] == 0 and m["margin_area_frac"] < 0.01, m
    assert m["ring_uniform"] < 0.5, m
    # busy edges: noise ring around a white middle
    busy = np.random.default_rng(3).integers(0, 256, (H, W, 3), dtype=np.uint8)
    busy[150:H - 150, 150:W - 150] = 255
    m = msb.border_metrics(webp(busy))
    assert m["ring_uniform"] < 0.9 and m["score"] == 0, m
    # a white corner region does not make the ring uniform
    half = photo(H, W, 8)
    half[:300, :300] = 255
    m = msb.border_metrics(webp(half))
    assert m["score"] == 0, m


def test_pillarbox_and_letterbox_flagged():
    pb = canvas((0, 0, 0), 2.0, h=832, w=1216)
    pb[:, 304:1216 - 304] = photo(832, 608, 3)
    m = msb.border_metrics(webp(pb))
    assert m["pillar_or_letter"] and m["bg_black"], m
    assert m["ring_uniform"] < 0.9, m           # bars are only part of the ring
    assert m["n_sides"] == 2 and abs(m["margin_left"] - 0.25) < 0.01, m
    assert m["score"] > 0.4, m
    lb = canvas((0, 0, 0), 2.0)
    lb[100:H - 100] = photo(H - 200, W, 4)      # portrait letterbox: ring mostly content
    m = msb.border_metrics(webp(lb))
    assert m["pillar_or_letter"] and m["bg_black"], m
    assert m["margin_left"] == 0 and m["margin_right"] == 0, m
    assert m["score"] > 0.12, m


def test_letterbox_with_flat_picture_sides():
    """Picture side edges flat (sky colour) and covering more of the ring
    than the bars: the whole-ring median is the picture colour (scores 0);
    the bar colour from a side strip must win through the geometry."""
    a = canvas((0, 0, 0), 2.0)
    p = photo(H - 120, W, 4)
    p[:, :60] = (120, 170, 220)
    p[:, W - 60:] = (120, 170, 220)
    a[60:H - 60] = p
    m = msb.border_metrics(webp(a))
    assert m["n_bg_candidates"] == 2 and m["bg_src"] in ("top", "bottom"), m
    assert m["bg_black"] and m["pillar_or_letter"], m
    assert m["ring_uniform"] < 0.5, m      # the bars are the ring's minority
    assert abs(m["margin_top"] - 60 / H) < 3 / H and m["score"] > 0.08, m


def bust(cut, seed=0):
    """White-background bust: round head/hair/shoulders, one flat
    horizontal cut `cut` px above the bottom (white below)."""
    im = Image.fromarray(canvas(seed=seed))
    d = ImageDraw.Draw(im)
    cx = W // 2
    d.ellipse([cx - 330, 620, cx + 330, 1500], fill=(70, 90, 160))
    d.ellipse([cx - 150, 260, cx + 150, 680], fill=(225, 190, 170))
    d.ellipse([cx - 190, 200, cx + 190, 470], fill=(60, 40, 30))
    a = np.asarray(im).copy()
    a[H - cut:] = canvas(seed=seed + 1)[H - cut:]
    return webp(a)


def test_geometric_straightness():
    ref = msb.border_metrics(inset(0.03, 0.03))["score"]
    for cut in (150, 250, 350):
        m = msb.border_metrics(bust(cut))
        st = [m[f"straight_{s}"] for s in msb.SIDES if m[f"straight_{s}"] >= 0]
        assert m["straight_bottom"] > 0.95 and len(st) == 4, m
        assert m["score"] < 0.6 * ref, (cut, m, ref)
        # the arithmetic mean of the first version ranked it ~ the inset
        arith = (m["margin_area_frac"] * sum(st) / len(st)
                 * min(1, m["fill"] / msb.FILL_FULL))
        assert arith > 0.75 * ref, (cut, arith, ref)
    # an inset with one weaker side (~0.6: white patch along its top edge)
    c = canvas()
    mt, ml = int(H * 0.08), int(W * 0.08)
    p = photo(H - 2 * mt, W - 2 * ml, 1)
    p[:6, :int(p.shape[1] * 0.4)] = 255
    c[mt:H - mt, ml:W - ml] = p
    m = msb.border_metrics(webp(c))
    assert 0.5 < m["straight_top"] < 0.7 and m["edge_straightness"] > 0.85, m
    assert m["score"] > 0.8 * msb.border_metrics(inset(0.08, 0.08))["score"], m


def test_thin_frame_and_frame_with_mat():
    fr = photo(H, W, 6)
    fr[:6] = fr[-6:] = 20
    fr[:, :6] = fr[:, -6:] = 20
    m = msb.border_metrics(webp(fr))
    # documented behaviour: a frame line around a full-bleed picture is only
    # flagged; its score is the line's area (< the candidate threshold)
    assert m["thin_frame"] and not m["peeled"] and m["n_sides"] == 0, m
    assert m["score"] < 0.03, m
    fm = canvas()
    fm[:6] = fm[-6:] = 20
    fm[:, :6] = fm[:, -6:] = 20
    fm[90:H - 90, 70:W - 70] = photo(H - 180, W - 140, 7)
    m = msb.border_metrics(webp(fm))
    assert m["thin_frame"] and m["peeled"] and m["frame_px"] >= 6, m
    assert m["bg_white"] and m["n_sides"] == 4, m
    assert abs(m["margin_top"] - 90 / H) < 4 / H, m
    assert m["score"] > 0.25, m


def test_grey_and_colored_canvas_like_white():
    white = msb.border_metrics(inset(0.08, 0.08))
    for color in ((200, 200, 200), (240, 190, 200), (40, 60, 110)):
        m = msb.border_metrics(inset(0.08, 0.08, color))
        assert not m["bg_white"], (color, m)
        assert abs(m["score"] - white["score"]) < 0.02, (color, m, white)
        assert tuple(abs(a - b) <= 6 for a, b in
                     zip((m["bg_r"], m["bg_g"], m["bg_b"]), color)) == (True,) * 3, m


def test_blank_and_input_checks():
    m = msb.border_metrics(webp(canvas()))
    assert m["blank"] and m["score"] == 0 and m["margin_area_frac"] == 1.0, m
    for bad in (np.zeros((8, 8), np.uint8), np.zeros((8, 8, 3), np.float32),
                np.zeros((8, 8, 4), np.uint8)):
        try:
            msb.border_metrics(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad.dtype} {bad.shape}")
    # tiny images do not crash
    for shape in ((1, 1, 3), (3, 5, 3), (40, 7, 3)):
        msb.border_metrics(np.full(shape, 128, np.uint8))
    m = msb.border_metrics(np.zeros((64, 64, 3), np.uint8) + 255)
    assert m["blank"], m
    keys = set(msb.border_metrics(inset(0.05, 0.05)))
    for k in ("score", "margin_top", "margin_bottom", "margin_left",
              "margin_right", "margin_area_frac", "n_sides", "fill",
              "edge_straightness", "ring_uniform", "bg_lum", "bg_white",
              "bg_black", "pillar_or_letter", "thin_frame", "peeled"):
        assert k in keys, k


# ------------------------------------------------------------------ selection
def test_thumb_selection():
    joined = [{"id": i, "score": sc, "pillar_or_letter": pl}
              for i, sc, pl in [(1, 0.5, False), (2, 0.04, False),
                                (3, 0.0, True), (4, 0.01, False),
                                (5, 0.0, False)]]
    joined.append({"id": 6, "error": "broken"})
    plan = {"watermark": [5], "uncaptioned": [], "control": [7]}
    tabs = msb.build_tabs(joined, plan, top_n=2)
    assert tabs == {"A": [1, 2], "B": [5], "C": [], "D": [7], "E": [3]}
    sel = msb.select_thumbs(tabs)
    assert sel == {1, 2, 3, 5, 7}       # 4 (rank 3) and 6 (error) not selected
    meta = {i: {"tar": f"t{i % 2}"} for i in range(1, 8)}
    assert msb.thumb_jobs(sel, {5: b"x"}, meta) == {"t0": [2], "t1": [1, 3, 7]}
    msb.check_thumbs(sel, {i: b"" for i in sel})
    try:
        msb.check_thumbs(sel, {1: b"", 2: b""})
    except SystemExit as e:
        assert "3 selected ids have no thumbnail" in str(e) and "[3, 5, 7]" in str(e)
    else:
        raise AssertionError("missing thumbnails accepted")


def _tar(path, members):
    with tarfile.open(path, "w") as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def test_member_parsing_and_shard_validation():
    assert msb.parse_member("data/123.webp") == (123, "webp")
    assert msb.parse_member("123.JSON") == (123, "json")
    for bad in ("abc.webp", "123.bmp", "123", "12a.webp", "123.webp.bak",
                ".webp", "123.foo.webp"):
        try:
            msb.parse_member(bad)
        except msb.ShardError:
            pass
        else:
            raise AssertionError(f"accepted member {bad!r}")
    assert msb.validate_shard([1, 2, 3], {1, 2, 3}) == {}
    assert msb.validate_shard([1, 2, 2, 4], {1, 2, 3}) == {
        "missing": [3], "extra": [4], "duplicate": [2]}
    img = webp_bytes(np.full((64, 48, 3), 128, np.uint8))
    with tempfile.TemporaryDirectory() as d:
        ok = Path(d) / "ok.tar"
        _tar(ok, [("7.webp", img), ("7.json", b"{}"), ("7.txt", b"c"),
                  ("8.webp", img), ("8.json", b"{}")])
        res = msb.score_shard_impl(ok, [8], log=lambda *_: None)
        assert [r["id"] for r in res["rows"]] == [7, 8] and set(res["thumbs"]) == {8}
        for name in ("x7.webp", "9.bmp", "9.webp.orig"):
            bad = Path(d) / "bad.tar"
            _tar(bad, [("7.webp", img), (name, img)])
            try:
                msb.score_shard_impl(bad, log=lambda *_: None)
            except msb.ShardError as e:
                assert name in str(e)
            else:
                raise AssertionError(f"tar member {name!r} accepted")
        # duplicate image of one id in a tar -> duplicate in validation
        dup = Path(d) / "dup.tar"
        _tar(dup, [("7.webp", img), ("7.png", img)])
        res = msb.score_shard_impl(dup, log=lambda *_: None)
        plan = {"ids": {"dup": [7, 8]}}
        err = msb.check_shard_result("dup", res, plan, log=lambda *_: None)
        assert "1 missing ids [8]" in err and "1 duplicate ids [7]" in err, err


def test_flags_and_plan():
    f = msb.flags_of("1girl, white border, simple background",
                     "a girl, Framed by a thin margin")
    assert f["tag_white_border"] and not f["tag_border"], f
    assert f["caption_mentions_border"] and not f["caption_mentions_watermark"]
    f = msb.flags_of("borderline, sample watermark", "sunset over a lake")
    assert not f["tag_border"] and f["tag_sample_watermark"], f
    assert not f["caption_mentions_border"], f
    assert msb.flags_of("", None)["tag_inset"] is False
    assert msb.flags_of("x", "WATERMARK in the corner")["caption_mentions_watermark"]
    meta = {}
    for i in range(400):
        tags = ["1girl"]
        cap = "a girl"
        if i % 50 == 0:
            tags.append("sample watermark")
        if i % 40 == 1:
            tags.append("border")
        if i % 80 == 1:
            cap = "a girl inside a white border"
        if i % 2 == 0:
            tags.append("white background")
        meta[1000 + i] = {"tar": f"data/train-{i % 4:04d}.tar", "bucket_w": 832,
                          "bucket_h": 1216, **msb.flags_of(", ".join(tags), cap),
                          "cap": cap}
    p = msb.plan_job(meta, seed=0, control_n=20)
    assert p["watermark"] == [1000 + i for i in range(0, 400, 50)]
    assert p["uncaptioned"] == [1000 + i for i in range(1, 400, 40) if i % 80 != 1]
    assert len(p["control"]) == 20 and all(meta[i]["tag_white_background"]
                                           for i in p["control"])
    assert p["control"] == msb.plan_job(meta, seed=0, control_n=20)["control"]
    assert p["control"] != msb.plan_job(meta, seed=1, control_n=20)["control"]
    assert p["shards"] == {f"data/train-{k:04d}.tar": 100 for k in range(4)}
    allw = set(p["watermark"]) | set(p["uncaptioned"]) | set(p["control"])
    assert {i for v in p["always"].values() for i in v} == allw
    for tar, ids in p["always"].items():
        assert all(meta[i]["tar"] == tar for i in ids)


# ------------------------------------------------------------------ dry run
def _fake_dataset(root):
    """Two tars of synthetic WebPs + a metadata.parquet like the real one."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    (root / "data").mkdir(parents=True)
    kinds = []
    for k in range(6):
        kinds += [("inset", lambda s: inset(0.05 + 0.02 * (s % 3), 0.06, seed=s)),
                  ("sil", lambda s: silhouette(s, s % 2 == 0)),
                  ("photo", lambda s: webp(photo(H, W, s)))]
    rows = {c: [] for c in ("id", "long", "medium", "short", "tags",
                            "bucket_w", "bucket_h", "tar")}
    truth = {}
    for t in range(2):
        tar_name = f"data/train-{t:04d}.tar"
        with tarfile.open(root / tar_name, "w") as tf:
            for j, (kind, make) in enumerate(kinds):
                sid = 5000 + 100 * t + j
                img = make(sid)
                tags = {"inset": "1girl, white border",
                        "sil": "1girl, white background",
                        "photo": "1girl, scenery"}[kind]
                if j == 2:
                    tags += ", sample watermark"
                members = [(f"{sid}.webp", webp_bytes(img, 90)),
                           (f"{sid}.json", json.dumps({"id": sid}).encode()),
                           (f"{sid}.txt", b"caption")]
                for name, data in members:
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
                truth[sid] = kind
                rows["id"].append(sid)
                rows["long"].append(f"image {sid} </script> \"quoted\"")
                rows["medium"].append("m")
                rows["short"].append("s")
                rows["tags"].append(tags)
                rows["bucket_w"].append(W)
                rows["bucket_h"].append(H)
                rows["tar"].append(tar_name)
    pq.write_table(pa.table(rows), root / "metadata.parquet")
    return root, truth


def _check_outputs(out, truth, s):
    import pyarrow.parquet as pq

    out = Path(out)
    t = pq.read_table(out / "scores.parquet").to_pylist()
    assert len(t) == len(truth) == s["images"], (len(t), len(truth), s)
    by = {r["id"]: r for r in t}
    for sid, kind in truth.items():
        r = by[sid]
        if kind == "inset":
            assert r["score"] >= 0.1 and r["tag_white_border"], r
            assert r["border_uncaptioned"], r     # caption has no border word
        else:
            assert r["score"] < 0.04, r
    scores = [r["score"] for r in t]
    assert scores == sorted(scores, reverse=True)
    page = (out / "review" / "index.html").read_text()
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                  page, re.S)
    data = json.loads(m.group(1))
    assert set(data["tabs"]) == {"A", "B", "C", "D", "E"}
    assert data["tabs"]["E"] == []
    assert re.fullmatch(r"[0-9a-f]{16}", data["review_id"]) and data["generated"]
    assert data["review_id"] == s["review_id"]
    insets = sorted(i for i, k in truth.items() if k == "inset")
    top = data["tabs"]["A"]
    assert set(top[:min(len(insets), len(top))]) <= set(insets), top
    assert sorted(data["tabs"]["B"]) == sorted(
        i for i in truth if i % 100 == 2), data["tabs"]["B"]
    assert sorted(data["tabs"]["C"]) == insets
    assert data["tabs"]["D"] and all(truth[i] == "sil" for i in data["tabs"]["D"])
    thumbs = {int(p.stem) for p in (out / "thumbs").glob("*.jpg")}
    # exactly the review page's ids (two-pass global selection)
    assert thumbs == {i for v in data["tabs"].values() for i in v}, thumbs
    assert set(insets) <= thumbs
    for i in thumbs:
        with Image.open(out / "thumbs" / f"{i}.jpg") as im:
            assert max(im.size) == msb.THUMB_SIDE and im.format == "JPEG"
    for i in data["tabs"]["A"]:
        assert data["items"][str(i)]["th"] == (i in thumbs)
    assert "</script> \"quoted\"" not in m.group(1)  # escaped inside the JSON
    assert data["items"][str(insets[0])]["cap"] == f"image {insets[0]} </script> \"quoted\""
    assert not re.search(r'(src|href)\s*=\s*["\']?https?:', page), "external URL"
    assert "@import" not in page and "prefers-color-scheme: dark" in page
    assert "localStorage" in page and "datafilter_remove.json" in page
    js = json.loads((out / "summary.json").read_text())
    assert set(js["missing_thumbs"].values()) == {0}, js["missing_thumbs"]


def test_dry_run_shard_and_outputs():
    with tempfile.TemporaryDirectory() as d:
        root, truth = _fake_dataset(Path(d) / "ds")
        meta = msb.load_meta(root / "metadata.parquet")
        plan = msb.plan_job(meta, seed=0, control_n=3)
        assert plan["ids"] == {t: sorted(i for i in truth if meta[i]["tar"] == t)
                               for t in plan["shards"]}
        rows, thumbs, logs = [], {}, []
        for tar in plan["shards"]:          # pass 1
            res = msb.score_shard_impl(root / tar, plan["always"].get(tar, []),
                                       threads=2, log=logs.append)
            assert res["n"] == 18 and not any("error" in r for r in res["rows"])
            assert msb.check_shard_result(tar, res, plan) is None
            assert set(res["thumbs"]) == set(plan["always"].get(tar, []))
            rows.extend(res["rows"])
            thumbs.update(res["thumbs"])
        assert len(logs) == 2 and "18 imgs" in logs[0]
        tabs = msb.build_tabs(msb.join_rows(rows, meta), plan, top_n=16)
        sel = msb.select_thumbs(tabs)
        try:                                 # pass 1 alone is not enough
            msb.write_outputs(Path(d) / "early", rows, thumbs, meta, plan, {},
                              top_n=16)
        except SystemExit as e:
            assert "no thumbnail" in str(e)
            assert not (Path(d) / "early").exists()
        else:
            raise AssertionError("write_outputs accepted missing thumbnails")
        jobs = msb.thumb_jobs(sel, thumbs, meta)
        assert jobs and set(jobs) <= set(plan["shards"])
        for tar, ids in jobs.items():         # pass 2: exactly those ids
            got = msb.thumb_shard_impl(root / tar, ids, log=logs.append)
            assert set(got["thumbs"]) == set(ids)
            thumbs.update(got["thumbs"])
        assert set(thumbs) == sel
        out = Path(d) / "out"
        s = msb.write_outputs(out, rows, thumbs, meta, plan, {"total": 1.0},
                              top_n=16)
        _check_outputs(out, truth, s)
        lines = []
        msb.print_summary(s, log=lines.append)
        assert any(l.startswith("score quantiles") for l in lines)
        assert s["top_overlap"]["white border"] == {"tagged": 12, "in_top": 12}
        # finalize-made table bytes are used as is
        tb = msb.table_bytes(msb.join_rows(rows, meta))
        out2 = Path(d) / "out2"
        msb.write_outputs(out2, rows, thumbs, meta, plan, {}, top_n=16,
                          table=tb)
        assert (out2 / "scores.parquet").read_bytes() == tb


def test_review_identity_and_storage_logic():
    tabs = {"A": [1, 2], "B": [3], "C": [], "D": [], "E": []}
    gen = "2026-10-14T00:00:00+00:00"
    rid = msb.review_id(tabs, gen)
    assert rid == msb.review_id(tabs, gen)
    assert rid != msb.review_id(tabs, "2026-10-14T00:00:01+00:00")
    assert rid != msb.review_id(dict(tabs, A=[1, 4]), gen)
    items = {str(i): {"s": 0.1 * i} for i in (1, 2, 3)}
    page = msb.render_review(tabs, items, gen)
    data = json.loads(re.search(
        r'<script id="data" type="application/json">(.*?)</script>', page,
        re.S).group(1))
    assert data["review_id"] == rid and data["generated"] == gen
    assert "storeKey(D)" in page and "loadSel(localStorage.getItem(KEY),D)" in page
    node = shutil.which("node")
    if node is None:
        print("node not found: storage logic not executed")
        return
    logic = re.search(r'<script id="logic">(.*?)</script>', page, re.S).group(1)
    js = logic + """
var D=%s;
function eq(a,b,m){if(JSON.stringify(a)!==JSON.stringify(b)){
  throw new Error(m+': '+JSON.stringify(a)+' != '+JSON.stringify(b))}}
eq(storeKey(D),'datafilter_remove_v1:'+D.review_id,'key');
var raw=JSON.stringify({'1':{tab:'A',score:0.1},'99':{tab:'A',score:1},
  '3':{tab:'B',score:null},'2':'junk'});
var sel=loadSel(raw,D);
eq(Object.keys(sel).sort(),['1','3'],'foreign / malformed ids dropped');
eq(loadSel('not json',D),{},'bad json');
eq(loadSel('[1,2]',D),{},'array');
eq(loadSel(null,D),{},'empty');
sel['77']={tab:'A',score:2};
var ex=exportObj(sel,D,'NOW');
eq(ex.review_id,D.review_id,'export id');eq(ex.generated,D.generated,'gen');
eq(ex.items.map(function(x){return x.id}).sort(),[1,3],'export ids');
console.log('ok');
""" % json.dumps(data)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.js"
        f.write_text(js)
        p = subprocess.run([node, str(f)], capture_output=True, text=True,
                           timeout=60)
    assert p.returncode == 0 and p.stdout.strip() == "ok", p.stderr


def test_local_cli_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        root, truth = _fake_dataset(Path(d) / "ds")
        out = Path(d) / "cli"
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "modal_score_borders.py"),
             "--tar-dir", str(root), "--parquet", str(root / "metadata.parquet"),
             "--out", str(out), "--top-n", "10", "--control-n", "3"],
            capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, p.stderr[-3000:]
        assert "score quantiles" in p.stdout and "tabs: A 10" in p.stdout, p.stdout
        s = json.loads((out / "summary.json").read_text())
        _check_outputs(out, truth, s)
        # --shards 1: only the first tar, tabs restricted to it
        out1 = Path(d) / "cli1"
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "modal_score_borders.py"),
             "--tar-dir", str(root / "data"), "--parquet",
             str(root / "metadata.parquet"), "--out", str(out1), "--shards", "1"],
            capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, p.stderr[-3000:]
        s1 = json.loads((out1 / "summary.json").read_text())
        assert s1["images"] == 18 and s1["tabs"]["B"] == 1, s1
        # the parquet lists an id the tar does not hold -> shard fails,
        # exit != 0, nothing written
        import pyarrow as pa
        import pyarrow.parquet as pq

        t = pq.read_table(root / "metadata.parquet")
        extra = pa.table({c: [t.column(c)[0].as_py()] for c in t.column_names})
        extra = extra.set_column(0, "id", pa.array([999999], pa.int64()))
        pq.write_table(pa.concat_tables([t, extra]), root / "bad.parquet")
        outb = Path(d) / "clibad"
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "modal_score_borders.py"),
             "--tar-dir", str(root), "--parquet", str(root / "bad.parquet"),
             "--out", str(outb)],
            capture_output=True, text=True, timeout=600)
        assert p.returncode != 0, p.stdout
        assert ("data/train-0000.tar: ids do not match metadata.parquet: "
                "1 missing ids [999999]") in p.stderr, p.stderr[-2000:]
        assert "nothing written" in p.stderr and not outb.exists()
        # a malformed member -> the shard fails the same way
        with tarfile.open(root / "data" / "train-0001.tar", "a") as tf:
            ti = tarfile.TarInfo("notes.md")
            ti.size = 1
            tf.addfile(ti, io.BytesIO(b"x"))
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "modal_score_borders.py"),
             "--tar-dir", str(root), "--parquet",
             str(root / "metadata.parquet"), "--out", str(outb)],
            capture_output=True, text=True, timeout=600)
        assert p.returncode != 0 and "unexpected tar member 'notes.md'" in \
            p.stderr, p.stderr[-2000:]
        assert not outb.exists()
    h = subprocess.run([sys.executable, str(REPO / "scripts" /
                                            "modal_score_borders.py"), "--help"],
                       capture_output=True, text=True, timeout=60)
    assert h.returncode == 0 and "modal run" in h.stdout


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
