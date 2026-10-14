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
  * thumbnail candidate selection, tag / caption flags, the always-thumb
    plan (watermark, border-uncaptioned, seeded white-background control)
  * dry run: a fake tar of synthetic WebPs (+ json/txt members) through the
    per-shard scorer, then the join + scores.parquet + review page writer
    (files exist, inline JSON parses, tabs, thumbs, no external URLs), and
    the local CLI (--tar-dir) end to end

    python tests/test_score_borders.py      # or: pytest tests/test_score_borders.py
"""

import io
import json
import re
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
    # thin (1.5%) white border all round still clears the candidate threshold
    assert msb.border_metrics(inset(0.016, 0.016))["score"] >= msb.SCORE_MIN


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
def test_pick_thumbs():
    rows = [{"id": i, "score": s, "pillar_or_letter": pl}
            for i, s, pl in [(1, 0.5, False), (2, 0.04, False), (3, 0.0, True),
                             (4, 0.01, False), (5, 0.0, False)]]
    rows.append({"id": 6, "error": "broken"})
    assert msb.pick_thumbs(rows, (), 0.05, 0.0) == {1, 3}
    # top-frac: ceil(0.2 * 5 ok rows) = 1 -> the best positive score (1)
    assert msb.pick_thumbs(rows, (), 0.6, 0.2) == {1, 3}
    assert msb.pick_thumbs(rows, (), 0.6, 0.4) == {1, 2, 3}
    # zero scores never fill the top-frac; always ids only when present
    assert msb.pick_thumbs(rows, (5, 99), 0.6, 1.0) == {1, 2, 3, 4, 5}


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
    assert set(data["tabs"]) == {"A", "B", "C", "D"}
    insets = sorted(i for i, k in truth.items() if k == "inset")
    top = data["tabs"]["A"]
    assert set(top[:min(len(insets), len(top))]) <= set(insets), top
    assert sorted(data["tabs"]["B"]) == sorted(
        i for i in truth if i % 100 == 2), data["tabs"]["B"]
    assert sorted(data["tabs"]["C"]) == insets
    assert data["tabs"]["D"] and all(truth[i] == "sil" for i in data["tabs"]["D"])
    thumbs = {int(p.stem) for p in (out / "thumbs").glob("*.jpg")}
    for tab in "BCD":   # always-thumb lists
        assert set(data["tabs"][tab]) <= thumbs, tab
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
    assert js["missing_thumbs"]["B"] == js["missing_thumbs"]["C"] == 0
    assert js["missing_thumbs"]["D"] == 0


def test_dry_run_shard_and_outputs():
    with tempfile.TemporaryDirectory() as d:
        root, truth = _fake_dataset(Path(d) / "ds")
        meta = msb.load_meta(root / "metadata.parquet")
        plan = msb.plan_job(meta, seed=0, control_n=3)
        rows, thumbs, logs = [], {}, []
        for tar in plan["shards"]:
            res = msb.score_shard_impl(root / tar, plan["always"].get(tar, []),
                                       msb.SCORE_MIN, 0.0, threads=2,
                                       log=logs.append)
            assert res["n"] == 18 and not any("error" in r for r in res["rows"])
            assert set(res["seconds"]) == {"metrics", "thumbs"}
            rows.extend(res["rows"])
            thumbs.update(res["thumbs"])
        assert len(logs) == 2 and "18 imgs" in logs[0]
        # thumbs: candidates (insets) + always ids, nothing else
        want = ({i for i, k in truth.items() if k == "inset"}
                | {i for v in plan["always"].values() for i in v})
        assert set(thumbs) == want, (sorted(thumbs), sorted(want))
        out = Path(d) / "out"
        s = msb.write_outputs(out, rows, thumbs, meta, plan, {"total": 1.0},
                              top_n=10)
        _check_outputs(out, truth, s)
        lines = []
        msb.print_summary(s, log=lines.append)
        assert any(l.startswith("score quantiles") for l in lines)
        assert s["top_overlap"]["white border"] == {"tagged": 12, "in_top": 10}
        # finalize-made table bytes are used as is
        tb = msb.table_bytes(msb.join_rows(rows, meta))
        out2 = Path(d) / "out2"
        msb.write_outputs(out2, rows, thumbs, meta, plan, {}, table=tb)
        assert (out2 / "scores.parquet").read_bytes() == tb


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
    h = subprocess.run([sys.executable, str(REPO / "scripts" /
                                            "modal_score_borders.py"), "--help"],
                       capture_output=True, text=True, timeout=60)
    assert h.returncode == 0 and "modal run" in h.stdout


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
