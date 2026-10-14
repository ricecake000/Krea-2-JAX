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
  * review round 2: canvas candidates within TOL of each other are all
    scored (Codex's grey letterbox repro); round 3: the thin-frame peel runs
    for every outer candidate (Codex's frame repro); the side-strip pruning
    never changes the final post-peel score (brute force over every median
    with per-candidate peel, random bar and framed fuzz);
    the dataset commit is resolved once and pinned for metadata.parquet and
    both tar passes (fake HfApi / download), recorded in summary.json,
    the scores.parquet schema metadata and the review page / export
  * review v2: side_pattern (4 / 3 / 2-opp / 2-adj / 1 / 0, stored on new
    rows, recovered exactly from an old row's rounded margins + h / w); tab A
    = score > 0 AND a border pattern (4 / 3 / 2-opp); the 4-state click
    cycle (keep -> caption -> crop -> remove), v2 localStorage key, foreign /
    malformed entries dropped, datafilter_actions.json export (node);
    --rerender of a v1 run (synthetic): default tab A exact by construction,
    only review/index.html written (v1 page backed up), a tab id without a
    thumbnail / lists that disagree with scores.parquet fail; the real run's
    outputs re-rendered from a temp copy (read-only, skipped if absent)
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


def test_close_candidates_all_scored():
    """Codex repro: 1216x832, grey-240 letterbox bars, grey-211 picture with
    2 px grey-225 side edges, WebP q80. The whole-ring median (~226) is
    within TOL of 240 and makes the picture canvas (blank); the bar colour
    must still be scored and win."""
    a = np.full((1216, 832, 3), 240, np.uint8)
    a[60:1216 - 60] = 211
    a[60:1216 - 60, :2] = 225
    a[60:1216 - 60, 830:] = 225
    img = webp(a, 80)
    cands = msb._bg_candidates(img)
    ring = [c for n, c, _ in cands if n == "ring"][0]
    assert abs(int(ring[0]) - 240) <= msb.TOL and int(ring[0]) != 240, cands
    assert any(int(c[0]) == 240 for n, c, _ in cands), cands
    m = msb.border_metrics(img)
    assert m["bg_r"] == 240 and m["bg_src"] == "top", m
    assert m["pillar_or_letter"] and abs(m["score"] - 0.1007) < 0.005, m
    assert not m["blank"], m


def _brute_candidates(rgb):
    """EVERY ring / side-strip median (no pruning, no merging) in
    border_metrics' order: [(colour uint8[3], ring_uniform)]."""
    h, w = rgb.shape[:2]
    k = max(1, min(msb.RING_PX, h // 2, w // 2))
    sides = [rgb[:k].reshape(-1, 3), rgb[h - k:].reshape(-1, 3),
             rgb[:, :k].reshape(-1, 3), rgb[:, w - k:].reshape(-1, 3)]
    ring = np.concatenate([sides[0], sides[1],
                           rgb[k:h - k, :k].reshape(-1, 3),
                           rgb[k:h - k, w - k:].reshape(-1, 3)]).astype(np.int16)
    out = []
    for px in [ring] + sides:
        cand = np.round(np.median(px, axis=0)).astype(np.int16)
        ru = float((np.abs(ring - cand).max(axis=1) <= msb.TOL).mean())
        out.append((cand.astype(np.uint8), ru))
    return out


def _brute_measure(rgb):
    """Best (score, measurement) over all brute candidates, no peel (first
    wins ties)."""
    h, w = rgb.shape[:2]
    best = None
    for cand, ru in _brute_candidates(rgb):
        mm = msb._measure(rgb, cand)
        mm["ring_uniform"] = ru
        sc = msb._finish(h, w, mm["px"], mm["fill"], mm["straight"], ru,
                         mm["blank"])["score"]
        if best is None or sc > best[0]:
            best = (sc, mm)
    return best


def _brute_final_score(rgb):
    """border_metrics' final score by brute force: every candidate, each
    peeled (inside = brute best of the crop) when its margins are thin."""
    h, w = rgb.shape[:2]
    best = 0.0
    for cand, ru in _brute_candidates(rgb):
        mm = msb._measure(rgb, cand)
        sc = msb._finish(h, w, mm["px"], mm["fill"], mm["straight"], ru,
                         mm["blank"])["score"]
        if msb.is_thin(mm["px"], mm["blank"], h, w):
            t, b, l, r = mm["px"]
            y0, y1 = t + msb.PEEL_PAD, h - b - msb.PEEL_PAD
            x0, x1 = l + msb.PEEL_PAD, w - r - msb.PEEL_PAD
            if y1 - y0 >= msb.PEEL_MIN_PX and x1 - x0 >= msb.PEEL_MIN_PX:
                _, inner = _brute_measure(rgb[y0:y1, x0:x1])
                if not inner["blank"]:
                    it, ib, il, ir = inner["px"]
                    px = (y0 + it, (h - y1) + ib, x0 + il, (w - x1) + ir)
                    sc = max(sc, msb._finish(
                        h, w, px, inner["fill"], inner["straight"],
                        inner["ring_uniform"])["score"])
        best = max(best, sc)
    return best


def _framed_fuzz(rng):
    """1-3 px frame line, a mat and an inset in close greys (thin frames
    on 150-260 px images), sometimes one frame side in another grey."""
    h, w = int(rng.integers(150, 260)), int(rng.integers(150, 260))
    g = lambda: int(rng.integers(0, 256))   # noqa: E731
    a = np.full((h, w, 3), g(), np.uint8)
    f, fc = int(rng.integers(1, 4)), g()
    a[-f:], a[:, :f], a[:, -f:] = fc, fc, fc
    a[:f, f:w - f] = (int(np.clip(fc + rng.integers(-30, 31), 0, 255))
                      if rng.random() < 0.4 else fc)
    m = int(rng.integers(f + 3, min(h, w) // 3))
    a[m:h - m, m:w - m] = g()
    if rng.random() < 0.5:
        n = m + int(rng.integers(2, 12))
        a[n:h - n, n:w - n] = g()
    return a


def test_candidate_pruning_never_changes_the_final():
    imgs = [inset(0.05, 0.05), inset(0.1, 0.02, (200, 200, 200)),
            silhouette(1, False), silhouette(2, True), webp(photo(H, W, 3))]
    lb = canvas((0, 0, 0), 2.0)
    lb[60:H - 60] = photo(H - 120, W, 4)
    lb[60:H - 60, :60] = (120, 170, 220)
    lb[60:H - 60, W - 60:] = (120, 170, 220)
    imgs += [webp(lb), _codex_frame()]
    fm = canvas()
    fm[:6] = fm[-6:] = 20
    fm[:, :6] = fm[:, -6:] = 20
    fm[90:H - 90, 70:W - 70] = photo(H - 180, W - 140, 7)
    imgs.append(webp(fm))
    rng = np.random.default_rng(7)
    for _ in range(60):     # small random bar / box layouts with close greys
        h, w = int(rng.integers(40, 120)), int(rng.integers(40, 120))
        a = np.full((h, w, 3), int(rng.integers(150, 256)), np.uint8)
        for _ in range(int(rng.integers(1, 5))):
            y0, x0 = int(rng.integers(0, h)), int(rng.integers(0, w))
            y1 = int(rng.integers(y0 + 1, h + 1))
            x1 = int(rng.integers(x0 + 1, w + 1))
            a[y0:y1, x0:x1] = int(rng.integers(150, 256))
        if rng.random() < 0.5:
            a = a + rng.integers(0, 4, a.shape).astype(np.uint8)
        imgs.append(a)
    n_peeled = 0
    for _ in range(40):
        a = _framed_fuzz(rng)
        imgs.append(a if rng.random() < 0.5 else webp(a, 80))
    for i, img in enumerate(imgs):
        got = msb.border_metrics(img)
        want = _brute_final_score(img)
        assert got["score"] == want, (i, got, want)
        n_peeled += got["peeled"]
    assert n_peeled >= 10, n_peeled    # the peel path is really exercised


def _codex_frame():
    """Codex round-3 repro: 1216x832 grey-50 mat, 6 px frame grey-20 on
    three sides and grey-35 on top, grey-70 picture [80:-80] with a grey-25
    interior [88:-88], WebP q80."""
    a = np.full((1216, 832, 3), 50, np.uint8)
    a[-6:] = 20
    a[:, :6] = 20
    a[:, -6:] = 20
    a[:6, 6:-6] = 35
    a[80:-80, 80:-80] = 70
    a[88:-88, 88:-88] = 25
    return webp(a, 80)


def test_peel_for_every_candidate():
    img = _codex_frame()
    h, w = img.shape[:2]
    outer = {}
    for name, bg, ru in msb._bg_candidates(img):
        res, mm = msb._outer(img, name, bg, ru)
        outer[name] = (res["score"], msb.is_thin(mm["px"], mm["blank"], h, w))
    # the un-thin top candidate beats the thin ring candidate before peeling
    assert outer["top"][0] > outer["ring"][0] and outer["ring"][1], outer
    assert not outer["top"][1], outer
    m = msb.border_metrics(img)
    assert m["peeled"] and m["thin_frame"] and m["frame_px"] == 6, m
    assert abs(m["score"] - 0.2986) < 0.005 and m["bg_r"] == 50, m


class FakeApi:
    """HfApi stand-in: dataset_info -> sha; get_paths_info -> RepoFile."""

    def __init__(self, sha, data, lfs=True):
        from types import SimpleNamespace as NS

        self.sha, self.calls = sha, []
        import hashlib

        if lfs:
            self.info = NS(path="metadata.parquet", size=len(data), blob_id="x",
                           lfs=NS(sha256=hashlib.sha256(data).hexdigest()))
        else:
            blob = b"blob %d\0" % len(data) + data
            self.info = NS(path="metadata.parquet", size=len(data), lfs=None,
                           blob_id=hashlib.sha1(blob).hexdigest())

    def dataset_info(self, repo, revision=None):
        from types import SimpleNamespace as NS

        self.calls.append(("dataset_info", repo, revision))
        return NS(sha=self.sha)

    def get_paths_info(self, repo, paths, revision=None, repo_type=None):
        self.calls.append(("get_paths_info", repo, tuple(paths), revision,
                           repo_type))
        return [self.info]


def _tiny_parquet(path, ids, tar="data/train-0000.tar"):
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(ids)
    pq.write_table(pa.table({
        "id": ids, "long": ["a girl"] * n, "tags": ["1girl, border"] * n,
        "bucket_w": [W] * n, "bucket_h": [H] * n, "tar": [tar] * n}), path)
    return Path(path).read_bytes()


def test_revision_pinned_everywhere():
    sha = "0123456789abcdef0123456789abcdef01234567"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        good = _tiny_parquet(d / "good.parquet", [1, 2, 3])
        other = _tiny_parquet(d / "other.parquet", [1, 2])
        downloads = []

        def download(filename, dest, token, revision):
            downloads.append((filename, revision))
            Path(dest).mkdir(parents=True, exist_ok=True)
            out = Path(dest) / filename
            out.write_bytes(good)
            return out

        for lfs in (True, False):
            # local copy identical to the commit's file: used, no download
            api = FakeApi(sha, good, lfs)
            downloads.clear()
            r = msb.prepare_impl(good, 0, 1, "", "tok", api=api,
                                 download=download, tmp=d / f"a{lfs}",
                                 log=lambda *_: None)
            assert r["revision"] == sha and downloads == [], r["source"]
            assert ("dataset_info", msb.DATASET_REPO, None) in api.calls
            assert api.calls[1][3] == sha and sorted(r["meta"]) == [1, 2, 3]
            # local copy differs -> downloaded at the pinned sha
            r = msb.prepare_impl(other, 0, 1, "v1.0", "tok", api=api,
                                 download=download, tmp=d / f"b{lfs}",
                                 log=lambda *_: None)
            assert downloads == [("metadata.parquet", sha)], downloads
            assert ("dataset_info", msb.DATASET_REPO, "v1.0") in api.calls
            assert "differs" in r["source"] and sorted(r["meta"]) == [1, 2, 3]
        # no local copy -> downloaded at the pinned sha
        downloads.clear()
        r = msb.prepare_impl(b"", 0, 1, "", "tok", api=FakeApi(sha, good),
                             download=download, tmp=d / "c",
                             log=lambda *_: None)
        assert downloads == [("metadata.parquet", sha)]
        assert r["parquet_sha256"] == __import__("hashlib").sha256(good).hexdigest()
        # tar passes: the download gets the pinned revision; file removed
        img = webp_bytes(np.full((64, 48, 3), 128, np.uint8))
        _tar(d / "src.tar", [("7.webp", img), ("7.json", b"{}")])
        seen = []

        def dl_tar(tar, dest, token, revision):
            seen.append((tar, revision))
            Path(dest).mkdir(parents=True, exist_ok=True)
            out = Path(dest) / Path(tar).name
            shutil.copy(d / "src.tar", out)
            return out

        res = msb.run_on_tar("data/train-0000.tar", sha, "tok",
                             lambda p: msb.score_shard_impl(p, [7], log=lambda *_: None),
                             download=dl_tar, dl_dir=d / "dl")
        res2 = msb.run_on_tar("data/train-0000.tar", sha, "tok",
                              lambda p: msb.thumb_shard_impl(p, [7], log=lambda *_: None),
                              download=dl_tar, dl_dir=d / "dl")
        assert seen == [("data/train-0000.tar", sha)] * 2
        assert res["revision"] == res2["revision"] == sha
        assert set(res2["thumbs"]) == {7} and not (d / "dl" / "train-0000.tar").exists()
        try:
            msb.fetch_dataset_file("x", d, None, "")
        except ValueError:
            pass
        else:
            raise AssertionError("unpinned download allowed")
        # recorded in summary.json, scores.parquet metadata, review page
        import pyarrow.parquet as pq

        rows = [{"id": 7, **msb.border_metrics(np.full((64, 48, 3), 128, np.uint8))}]
        meta = {7: {"tar": "t", "bucket_w": 48, "bucket_h": 64,
                    **msb.flags_of("", "x"), "cap": "x"}}
        plan = {"shards": {"t": 1}, "ids": {"t": [7]}, "always": {},
                "watermark": [], "uncaptioned": [], "control": []}
        s = msb.write_outputs(d / "out", rows, {}, meta, plan, {},
                              revision=sha)
        assert s["dataset_revision"] == sha
        assert json.loads((d / "out" / "summary.json").read_text())[
            "dataset_revision"] == sha
        md = pq.read_schema(d / "out" / "scores.parquet").metadata
        assert md[b"dataset_revision"] == sha.encode()
        assert md[b"dataset_repo"] == msb.DATASET_REPO.encode()
        page = (d / "out" / "review" / "index.html").read_text()
        data = json.loads(re.search(
            r'<script id="data" type="application/json">(.*?)</script>',
            page, re.S).group(1))
        assert data["revision"] == sha and sha in page
        assert data["review_id"] == s["review_id"] != msb.review_id(
            data["tabs"], data["generated"], "other")


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
PATTERN_SIDES = {"4": 4, "3": 3, "2-opp": 2, "2-adj": 2, "1": 1, "0": 0}
# px margins (t, b, l, r) per pattern; "0" = a 2 px band (below SIDE_MIN_PX)
PX = {"4": (100, 100, 80, 80), "3": (100, 0, 60, 60), "2-opp": (0, 0, 120, 120),
      "2-adj": (90, 0, 70, 0), "1": (0, 150, 0, 0), "0": (2, 0, 0, 0)}
PATS = list(PX)


def _cols(pattern, h=H, w=W, straight=1.0):
    """Score-row columns of a pattern: rounded margin fractions, h / w,
    straight_* (-1 where there is no margin)."""
    out = {"h": h, "w": w}
    for s, p in zip(msb.SIDES, PX[pattern]):
        n = h if s in ("top", "bottom") else w
        out[f"margin_{s}"] = round(p / n, msb.ROUND)
        out[f"straight_{s}"] = straight if p else -1.0
    return out


def _like_4484389():
    """1024x1024 white canvas, picture with margins top 36 / bottom 12 /
    left 132 / right 152 px (bottom 1.17% < MIN_SIDE_FRAC)."""
    c = canvas(h=1024, w=1024)
    c[36:1024 - 12, 132:1024 - 152] = photo(1024 - 48, 1024 - 284, 3)
    return webp(c)


def _like_8248423():
    """896 wide x 1152 high white canvas; a yellow rectangle (~70% of the
    content height) behind a character whose round head sticks out above
    (24 px top margin) and whose legs run off the bottom edge."""
    w, h = 896, 1152
    im = Image.fromarray(canvas(h=h, w=w))
    d = ImageDraw.Draw(im)
    d.rectangle([155, 230, w - 134, 1020], fill=(240, 200, 40))
    d.ellipse([360, 24, 560, 300], fill=(60, 40, 30))            # head
    d.ellipse([330, 250, 590, 900], fill=(70, 90, 160))          # body
    d.ellipse([360, 850, 450, h + 150], fill=(70, 90, 160))      # legs
    d.ellipse([470, 850, 560, h + 150], fill=(70, 90, 160))
    return webp(np.asarray(im))


def test_side_pattern():
    sp = msb.side_pattern
    one = (1.0,) * 4
    for want, px in PX.items():
        assert sp(px, one) == want, (want, px)
        assert sp(dict(zip(msb.SIDES, px)), dict(zip(msb.SIDES, one))) == want
        row = _cols(want)
        assert msb.row_side_pattern(row) == want, (want, row)
    assert sp((10, 10, 0, 0), one) == "2-opp"                 # letterbox
    assert sp((0, 10, 0, 10), one) == "2-adj"
    assert sp((0, 10, 10, 10), one) == "3"
    m = msb.SIDE_MIN_PX
    s = msb.SIDE_STRAIGHT
    assert (m, s) == (3, 0.5)
    assert sp((m,) * 4, (s,) * 4) == "4"                      # >= counts
    assert sp((m - 1,) * 4, one) == "0"                       # a 2 px band
    assert sp((10,) * 4, (s - 1e-9,) * 4) == "0"              # not straight
    assert sp((10,) * 4, (1.0, -1.0, None, 1.0)) == "2-adj"   # -1 / None
    assert sp((10, 10, None, 0), one) == "2-opp"
    for bad in (((1, 2), one), ((1, 2, 3, 4), (1.0,))):
        try:
            sp(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    assert set(msb.SIDE_PATTERN_LABELS) == {"4", "3", "2-opp", "2-adj", "1", "0"}
    assert msb.BORDER_PATTERNS == ("4", "3", "2-opp")
    # the user's two cases as score rows (old parquet: no side_pattern)
    r4484389 = {"h": 1024, "w": 1024, "margin_top": 0.0352,
                "margin_bottom": 0.0117, "margin_left": 0.1289,
                "margin_right": 0.1484, "straight_top": 1.0,
                "straight_bottom": 1.0, "straight_left": 0.999,
                "straight_right": 0.999}
    assert msb.row_px_straight(r4484389)[0] == {
        "top": 36, "bottom": 12, "left": 132, "right": 152}
    assert msb.row_side_pattern(r4484389) == "4"
    r8248423 = {"h": 1152, "w": 896, "margin_top": 0.0208,
                "margin_bottom": 0.0, "margin_left": 0.173,
                "margin_right": 0.1484, "straight_top": 0.099,
                "straight_bottom": -1.0, "straight_left": 0.718,
                "straight_right": 0.719}
    assert msb.row_side_pattern(r8248423) == "2-opp"
    # exact straightness from the rounded column: count / content length
    px, st = msb.row_px_straight(r8248423)
    assert st["left"] == round(0.718 * (1152 - 24)) / (1152 - 24), st
    assert st["bottom"] is None and px["bottom"] == 0
    # a stored column wins (new parquets)
    assert msb.row_side_pattern(dict(r4484389, side_pattern="1")) == "1"


def test_side_pattern_images():
    pb = canvas((0, 0, 0), 2.0, h=832, w=1216)
    pb[:, 304:1216 - 304] = photo(832, 608, 3)
    cases = {"4": [inset(0.08, 0.08), _like_4484389()],
             "2-opp": [webp(pb), _like_8248423()],
             "0": [webp(photo(H, W, 5)), silhouette(1, False),
                   silhouette(2, False)]}
    c = canvas()
    c[int(H * 0.1):, int(W * 0.1):] = photo(H - int(H * 0.1), W - int(W * 0.1), 2)
    cases["2-adj"] = [webp(c)]
    c = canvas()
    c[int(H * 0.1):, int(W * 0.05):W - int(W * 0.05)] = photo(
        H - int(H * 0.1), W - 2 * int(W * 0.05), 2)
    # a 2 px straight band at the bottom does not count (WebP: ~1 px)
    c2 = canvas()
    c2[100:H - 2, 80:W - 80] = photo(H - 102, W - 160, 4)
    # a ~10 px band below a ROUND bottom (silhouette tip) does not count
    im = Image.fromarray(canvas())
    d = ImageDraw.Draw(im)
    d.rectangle([130, 100, W - 150, H - 300], fill=(90, 120, 160))
    d.ellipse([200, H - 500, W - 220, H - 10], fill=(90, 120, 160))
    cases["3"] = [webp(c), webp(c2), webp(np.asarray(im))]
    for want, imgs in cases.items():
        for k, img in enumerate(imgs):
            m = msb.border_metrics(img)
            assert m["side_pattern"] == want, (want, k, m)
            old = {kk: v for kk, v in m.items() if kk != "side_pattern"}
            assert msb.row_side_pattern(old) == want, (want, k, old)
    m = msb.border_metrics(_like_4484389())
    assert m["n_sides"] == 3 and 0 < m["margin_bottom"] < msb.MIN_SIDE_FRAC, m
    m = msb.border_metrics(_like_8248423())
    assert 0.6 < m["straight_left"] < 0.8 and m["straight_top"] < 0.3, m
    m = msb.border_metrics(webp(np.asarray(im)))
    assert 5 <= round(m["margin_bottom"] * H) <= 12, m
    assert m["straight_bottom"] < 0.3, m
    m = msb.border_metrics(webp(c2))
    assert round(m["margin_bottom"] * H) < msb.SIDE_MIN_PX, m


def test_tab_a_border_patterns_only():
    joined = []
    for i, (pat, sc) in enumerate([("2-adj", 0.9), ("4", 0.8), ("1", 0.7),
                                   ("2-opp", 0.6), ("0", 0.5), ("3", 0.4),
                                   ("4", 0.3), ("4", 0.0)], start=1):
        joined.append({"id": i, "score": sc, **_cols(pat)})
    joined.append({"id": 20, "error": "broken"})
    plan = {"watermark": [], "uncaptioned": [], "control": []}
    assert msb.build_tabs(joined, plan, top_n=3)["A"] == [2, 4, 6]
    assert msb.build_tabs(joined, plan, top_n=10)["A"] == [2, 4, 6, 7]
    # a stored side_pattern column wins over the margins
    joined[0]["side_pattern"] = "4"
    assert msb.build_tabs(joined, plan, top_n=2)["A"] == [1, 2]
    s = msb.summarize(joined[1:], msb.build_tabs(joined[1:], plan, 10), set(),
                      {"shards": {}}, {})
    assert s["side_patterns_A"] == {"4": 2, "3": 1, "2-opp": 1, "2-adj": 0,
                                    "1": 0, "0": 0}, s["side_patterns_A"]
    assert s["side_patterns_score_gt0"]["1"] == 1


def test_thumb_selection():
    joined = [{"id": i, "score": sc, "pillar_or_letter": pl, **_cols(pat)}
              for i, sc, pl, pat in [(1, 0.5, False, "4"),
                                     (2, 0.04, False, "3"),
                                     (3, 0.0, True, "2-opp"),
                                     (4, 0.01, False, "4"),
                                     (5, 0.0, False, "0"),
                                     (8, 0.3, False, "2-adj")]]
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
    assert "localStorage" in page and "datafilter_actions.json" in page
    assert "datafilter_review_v2:" in page and "datafilter_remove" not in page
    for i in data["tabs"]["A"]:      # tab A: border side patterns only
        assert data["items"][str(i)]["sp"] in msb.BORDER_PATTERNS
    for r in t:                      # stored column == the margin-derived one
        assert r["side_pattern"] == msb.row_side_pattern(
            {k: v for k, v in r.items() if k != "side_pattern"}), r
    js = json.loads((out / "summary.json").read_text())
    assert set(js["missing_thumbs"].values()) == {0}, js["missing_thumbs"]
    # the tab A rule is recorded in summary.json and on the page
    assert js["a_rule"] == data["a_rule"], (js["a_rule"], data["a_rule"])
    assert js["a_rule"]["name"] == "side-pattern" and js["a_rule"]["version"] == 2
    pos = [r["id"] for r in t if r["score"] > 0]
    assert set(data["tabs"]["A"]) == {
        i for i in pos[:js["a_rule"]["rank_cutoff"]]
        if msb.row_side_pattern(by[i]) in msb.BORDER_PATTERNS}


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
        # every tab A inset is also border-uncaptioned (an always-thumb id);
        # drop those pass-1 thumbnails so pass 2 has to fetch them
        for i in tabs["A"]:
            thumbs.pop(i, None)
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


def _run_node(js):
    node = shutil.which("node")
    if node is None:
        print("node not found: page logic not executed")
        return None
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.js"
        f.write_text(js)
        p = subprocess.run([node, str(f)], capture_output=True, text=True,
                           timeout=60)
    assert p.returncode == 0 and p.stdout.strip() == "ok", p.stderr
    return p


def test_review_identity_and_storage_logic():
    tabs = {"A": [1, 2], "B": [3, 1], "C": [], "D": [], "E": [2]}
    gen = "2026-10-14T00:00:00+00:00"
    rid = msb.review_id(tabs, gen)
    assert rid == msb.review_id(tabs, gen)
    assert rid != msb.review_id(tabs, "2026-10-14T00:00:01+00:00")
    assert rid != msb.review_id(dict(tabs, A=[1, 4]), gen)
    items = {"1": {"s": 0.1, "sp": "4"}, "2": {"s": 0.2, "sp": "2-opp"},
             "3": {"s": None, "sp": None}}
    page = msb.render_review(tabs, items, gen, revision="abc")
    data = json.loads(re.search(
        r'<script id="data" type="application/json">(.*?)</script>', page,
        re.S).group(1))
    assert data["review_id"] == msb.review_id(tabs, gen, "abc") != rid
    assert data["generated"] == gen and data["revision"] == "abc"
    assert data["patterns"] == msb.SIDE_PATTERN_LABELS
    assert data["border_patterns"] == ["4", "3", "2-opp"]
    assert "storeKey(D)" in page and "loadSel(localStorage.getItem(KEY),D)" in page
    assert "datafilter_actions.json" in page and "datafilter_remove" not in page
    for label in ("캡션 보완", "자르기", "제거", "id=\"legend\""):
        assert label in page, label
    logic = re.search(r'<script id="logic">(.*?)</script>', page, re.S).group(1)
    js = logic + """
var D=%s;
function eq(a,b,m){if(JSON.stringify(a)!==JSON.stringify(b)){
  throw new Error(m+': '+JSON.stringify(a)+' != '+JSON.stringify(b))}}
eq(storeKey(D),'datafilter_review_v2:'+D.review_id,'key');
eq(STATES,['keep','caption','crop','remove'],'states');
// 4-state cycle keep -> caption -> crop -> remove -> keep
var sel={}, seen=[];
for(var i=0;i<5;i++){seen.push(cycle(sel,1))}
eq(seen,['caption','crop','remove','keep','caption'],'cycle');
eq(sel,{'1':'caption'},'cycle state');
cycle(sel,1);cycle(sel,1);cycle(sel,1);eq(sel,{},'keep = absent');
eq(nextState(undefined),'caption','unknown -> caption');
eq(nextState('bogus'),'caption','bogus -> caption');
setState(sel,2,'crop');setState(sel,3,'remove');setState(sel,1,'caption');
eq(stateCounts(sel,D),{keep:0,caption:1,crop:1,remove:1},'counts');
setState(sel,3,'keep');setState(sel,1,'bogus');
eq(sel,{'2':'crop'},'setState keep / bogus deletes');
eq(stateCounts(sel,D),{keep:2,caption:0,crop:1,remove:0},'counts 2');
// load: foreign ids, keep, unknown states, the v1 object format dropped
var raw=JSON.stringify({'1':'caption','99':'remove','2':'keep','3':'crop',
  '__proto__':'remove','x':'crop'});
eq(loadSel(raw,D),{'1':'caption','3':'crop'},'foreign / keep dropped');
eq(loadSel(JSON.stringify({'1':{tab:'A',score:0.1},'2':'bogus','3':5}),D),{},
   'malformed');
eq(loadSel('not json',D),{},'bad json');
eq(loadSel('[1,2]',D),{},'array');
eq(loadSel('null',D),{},'null');
eq(loadSel(null,D),{},'empty');
// export: only non-keep ids of this page, every tab of the id
var s2={'2':'remove','1':'caption','3':'crop','77':'remove'};
var ex=exportObj(s2,D);
eq(Object.keys(ex),['review_id','generated','dataset_revision','items'],'keys');
eq(ex.review_id,D.review_id,'export id');eq(ex.generated,D.generated,'gen');
eq(ex.dataset_revision,'abc','export revision');
eq(ex.items,[
  {id:1,action:'caption',tabs:['A','B'],score:0.1,side_pattern:'4'},
  {id:2,action:'remove',tabs:['A','E'],score:0.2,side_pattern:'2-opp'},
  {id:3,action:'crop',tabs:['B'],score:null,side_pattern:null}],'items');
eq(exportObj({},D).items,[],'empty export');
console.log('ok');
""" % json.dumps(data)
    _run_node(js)


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


# ------------------------------------------------------------------ re-render
REV = "05a9a13d4159363a5f5a5258ff3ae904506267bb"


def _old_run(root, old_n=12, control_n=4):
    """A finished v1 run: scores.parquet WITHOUT side_pattern (rounded
    margins, h / w), summary.json, thumbs of every v1 tab id and a v1 page
    (its inline data; no `patterns`). -> (root, info)."""
    root = Path(root)
    (root / "thumbs").mkdir(parents=True)
    (root / "review").mkdir()
    rows = []
    for i in range(1, 41):
        pat = PATS[i % 6]
        row = {"id": i, **_cols(pat), "n_sides": PATTERN_SIDES[pat],
               "score": round(0.5 - 0.01 * i, 4) if i <= 30 else 0.0,
               "pillar_or_letter": pat == "2-opp" and i % 2 == 0,
               "h": H, "w": W, "tar": "data/t.tar", **msb.flags_of("", "x")}
        row.update(tag_sample_watermark=i % 7 == 0,
                   tag_white_background=i % 3 == 0,
                   border_uncaptioned=i % 5 == 0)
        rows.append(row)
    rows.append({"id": 50, "error": "broken", "tar": "data/t.tar"})
    rows.sort(key=lambda r: (-(r.get("score") or 0.0), r["id"]))
    ok = [r for r in rows if "error" not in r]
    white = sorted(r["id"] for r in rows if r.get("tag_white_background"))
    tabs = {
        "A": [r["id"] for r in ok if r["score"] > 0][:old_n],
        "B": sorted(r["id"] for r in rows if r.get("tag_sample_watermark")),
        "C": sorted(r["id"] for r in rows if r.get("border_uncaptioned")),
        "D": sorted(__import__("random").Random(0).sample(white, control_n)),
        "E": [r["id"] for r in ok if r["pillar_or_letter"]],
    }
    (root / "scores.parquet").write_bytes(msb.table_bytes(rows, REV))
    (root / "summary.json").write_text(json.dumps({
        "dataset_revision": REV, "tabs": {k: len(v) for k, v in tabs.items()}}))
    ids = {i for v in tabs.values() for i in v}
    for i in ids:
        (root / "thumbs" / f"{i}.jpg").write_bytes(b"jpg%d" % i)
    data = {"tabs": tabs, "items": {str(i): {"cap": f"cap {i}"} for i in ids},
            "names": {}, "review_id": "0" * 16, "dataset_repo": msb.DATASET_REPO,
            "generated": "2026-10-13T00:00:00+00:00", "revision": REV}
    (root / "review" / "index.html").write_text(
        '<!doctype html><script id="data" type="application/json">'
        + json.dumps(data) + "</script><script>var STORE='datafilter_remove_v1:'"
        "</script>", encoding="utf-8")
    pat = {r["id"]: PATS[r["id"] % 6] for r in ok}
    return root, {"tabs": tabs, "pat": pat, "rows": rows}


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def _page(root, name="index.html"):
    return msb.page_data(Path(root) / "review" / name)


def _expect_fail(fn, *needles):
    try:
        fn()
    except SystemExit as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return str(e)
    raise AssertionError("did not fail")


def test_rerender_default_exact():
    with tempfile.TemporaryDirectory() as d:
        root, info = _old_run(Path(d) / "run")
        before = _snapshot(root)
        lines = []
        res = msb.rerender(root, generated="2026-10-14T00:00:00+00:00",
                           log=lines.append)
        old = info["tabs"]
        want_a = [i for i in old["A"] if info["pat"][i] in msb.BORDER_PATTERNS]
        assert len(want_a) < len(old["A"]) and res["top_n"] == len(want_a)
        assert any(l.startswith(f"default --top-n {len(want_a)}") for l in lines)
        data = _page(root)
        assert data["tabs"]["A"] == want_a, (data["tabs"]["A"], want_a)
        for k in "BCDE":
            assert data["tabs"][k] == old[k], k
        assert data["patterns"] == msb.SIDE_PATTERN_LABELS
        assert data["review_id"] == res["review_id"] == msb.review_id(
            data["tabs"], "2026-10-14T00:00:00+00:00", REV)
        assert data["revision"] == REV
        for k, ids in data["tabs"].items():
            for i in ids:
                it = data["items"][str(i)]
                assert it["sp"] == info["pat"][i] and it["th"], (k, i, it)
                assert it["cap"] == f"cap {i}"
        assert res["side_patterns_A"] == {
            p: sum(info["pat"][i] == p for i in want_a)
            for p in msb.SIDE_PATTERN_LABELS}
        assert res["tabs"] == {k: len(v) for k, v in data["tabs"].items()}
        # only review/index.html changed; the v1 page is backed up as is
        after = _snapshot(root)
        assert after["review/index.v1.html"] == before["review/index.html"]
        assert {k for k in after if after[k] != before.get(k)} == {
            "review/index.html", "review/index.v1.html"}
        # a second re-render reads the backup (D, captions), keeps it
        v2 = after["review/index.html"]
        res2 = msb.rerender(root, log=lambda *_: None)
        assert _page(root)["tabs"] == data["tabs"]
        assert res2["control_from"].startswith("index.v1.html")
        assert _snapshot(root)["review/index.v1.html"] == before["review/index.html"]
        assert _snapshot(root)["review/index.html"] != v2   # new generation time


def test_rerender_missing_thumbs_fail():
    with tempfile.TemporaryDirectory() as d:
        root, info = _old_run(Path(d) / "run")
        before = _snapshot(root)
        eligible = [r["id"] for r in info["rows"] if "error" not in r
                    and r["score"] > 0
                    and info["pat"][r["id"]] in msb.BORDER_PATTERNS]
        n_def = sum(i in info["tabs"]["A"] for i in eligible)
        # a larger tab A reaches eligible rows below the run's cutoff that
        # were never thumbnailed -> count + ids, nothing written
        beyond = [i for i in eligible[n_def:n_def + 3]
                  if not (root / "thumbs" / f"{i}.jpg").exists()]
        assert beyond
        _expect_fail(lambda: msb.rerender(root, top_n=n_def + 3,
                                          log=lambda *_: None),
                     f"{len(beyond)} tab ids have no thumbnail",
                     f"A {len(beyond)} {sorted(beyond)}")
        assert _snapshot(root) == before            # no backup, no page
        # an eligible id inside the old top-N lost its thumbnail -> the
        # default fails rather than dropping it (also the LAST one: the
        # membership must not depend on which thumbnails exist)
        for gone in (eligible[1], eligible[n_def - 1]):
            (root / "thumbs" / f"{gone}.jpg").unlink()
            _expect_fail(lambda: msb.rerender(root, log=lambda *_: None),
                         f"A 1 [{gone}]")
            (root / "thumbs" / f"{gone}.jpg").write_bytes(b"jpg%d" % gone)
            assert _snapshot(root) == before
        # an id of another tab without a thumbnail
        b0 = info["tabs"]["B"][0]
        (root / "thumbs" / f"{b0}.jpg").unlink()
        _expect_fail(lambda: msb.rerender(root, log=lambda *_: None),
                     f"B 1 [{b0}]")
        (root / "thumbs" / f"{b0}.jpg").write_bytes(b"x")
        assert not (root / "review" / "index.v1.html").exists()
        # a previous page whose lists disagree with scores.parquet
        page = (root / "review" / "index.html").read_text()
        data = msb.page_data(root / "review" / "index.html")
        bad = dict(data, tabs=dict(data["tabs"], C=data["tabs"]["C"][1:]))
        (root / "review" / "index.html").write_text(page.replace(
            json.dumps(data), json.dumps(bad)))
        _expect_fail(lambda: msb.rerender(root, log=lambda *_: None),
                     "tabs ['C']", "disagree")
        (root / "review" / "index.html").write_text(page)
        # summary.json from another run (tab D size)
        s = json.loads((root / "summary.json").read_text())
        (root / "summary.json").write_text(json.dumps(
            dict(s, tabs=dict(s["tabs"], D=99))))
        _expect_fail(lambda: msb.rerender(root, log=lambda *_: None),
                     "tab D", "summary.json says 99")
        (root / "summary.json").write_text(json.dumps(s))
        # no previous page: D re-drawn (seed 0) == the run's, no captions
        (root / "review" / "index.html").unlink()
        res = msb.rerender(root, control_n=4, log=lambda *_: None)
        assert res["control_from"].startswith("re-drawn")
        data = _page(root)
        assert data["tabs"]["D"] == info["tabs"]["D"]
        assert res["no_caption_snippet"] == len(
            {i for v in data["tabs"].values() for i in v})
        assert not (root / "review" / "index.v1.html").exists()


def test_rerender_recorded_v2_rule():
    """A v2 run records its tab A rule; the re-render's default cutoff is
    that rule's score-rank cutoff, not the A size read as an unfiltered
    top-N (Codex: ranked 1 = 2-adj, 2 = 4, 3 = 4, top_n 2 -> A [2, 3])."""
    rows = [{"id": i, "score": sc, **_cols(pat), "tar": "data/t.tar",
             "pillar_or_letter": False, **msb.flags_of("", "x"),
             "border_uncaptioned": False}
            for i, sc, pat in [(1, 0.5, "2-adj"), (2, 0.4, "4"), (3, 0.3, "4"),
                               (4, 0.2, "3"), (5, 0.0, "4")]]
    plan = {"watermark": [], "uncaptioned": [], "control": []}
    tabs = msb.build_tabs(rows, plan, top_n=2)
    assert tabs["A"] == [2, 3]
    rule = msb.a_rule_record(rows, tabs["A"], 2)
    assert rule["rank_cutoff"] == 3 and rule["version"] == 2
    assert rule["side_min_px"] == msb.SIDE_MIN_PX
    assert rule["side_straight"] == msb.SIDE_STRAIGHT
    assert msb.a_rule_record(rows, [2, 3, 4], 10)["rank_cutoff"] == 4
    assert msb.a_rule_record(rows, [], 0)["rank_cutoff"] == 0
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "thumbs").mkdir()
        for i in (2, 3):
            (root / "thumbs" / f"{i}.jpg").write_bytes(b"x")
        (root / "scores.parquet").write_bytes(msb.table_bytes(rows, REV))
        (root / "summary.json").write_text(json.dumps({
            "dataset_revision": REV, "a_rule": rule,
            "tabs": {k: len(v) for k, v in tabs.items()}}))
        res = msb.rerender(root, control_n=0, log=lambda *_: None)
        data = _page(root)
        assert data["tabs"]["A"] == [2, 3] and res["rank_cutoff"] == 3
        assert data["a_rule"]["rank_cutoff"] == 3
        assert data["a_rule"]["run_rule"] == rule
        # the same summary read as a legacy unfiltered top-2 would give [2]
        legacy = dict(json.loads((root / "summary.json").read_text()))
        legacy.pop("a_rule")
        (root / "summary.json").write_text(json.dumps(legacy))
        (root / "review" / "index.html").unlink()
        res = msb.rerender(root, control_n=0, log=lambda *_: None)
        assert _page(root)["tabs"]["A"] == [2] and res["rank_cutoff"] == 2


def test_rerender_cli():
    script = str(REPO / "scripts" / "modal_score_borders.py")
    with tempfile.TemporaryDirectory() as d:
        root, info = _old_run(Path(d) / "run")
        p = subprocess.run([sys.executable, script, "--rerender", str(root)],
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stderr[-2000:]
        assert "default --top-n" in p.stdout and "review id" in p.stdout
        assert (root / "review" / "index.v1.html").exists()
        p = subprocess.run([sys.executable, script, "--rerender", str(root),
                            "--top-n", "30"],
                           capture_output=True, text=True, timeout=120)
        assert p.returncode != 0 and "have no thumbnail" in p.stderr, p.stderr
        for extra in (["--out", str(root)], ["--top-n", "-1"]):
            p = subprocess.run([sys.executable, script, "--rerender",
                                str(root)] + extra,
                               capture_output=True, text=True, timeout=120)
            assert p.returncode == 2, (extra, p.stderr)
        p = subprocess.run([sys.executable, script, "--out", str(root)],
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 2 and "--tar-dir and --out" in p.stderr


def test_rerender_real_run_copy():
    """The finished full run (read-only): a copy of its parquet / summary /
    page(s) + its thumbs (symlinked) re-renders exactly; the originals are
    not touched."""
    src = msb.LOCAL_OUT
    if not (src / "scores.parquet").is_file():
        print(f"{src} not found: real-run re-render not tested")
        return
    import hashlib

    def digest():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [src / "scores.parquet", src / "summary.json",
                          *sorted((src / "review").glob("*.html"))]}

    before = digest()
    n_thumbs = len(list((src / "thumbs").glob("*.jpg")))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "review").mkdir()
        shutil.copy(src / "scores.parquet", d)
        shutil.copy(src / "summary.json", d)
        for p in (src / "review").glob("index*.html"):
            shutil.copy(p, d / "review" / p.name)
        (d / "thumbs").symlink_to(src / "thumbs")
        res = msb.rerender(d, log=lambda *_: None)
        data = _page(d)
        old = _page(d, "index.v1.html")
        s = json.loads((src / "summary.json").read_text())
        assert res["top_n"] == res["exact_default_top_n"] == len(data["tabs"]["A"])
        assert set(data["tabs"]["A"]) <= set(old["tabs"]["A"])
        assert all(data["items"][str(i)]["sp"] in msb.BORDER_PATTERNS
                   for i in data["tabs"]["A"])
        rows = {r["id"]: r for r in msb.load_score_rows(d / "scores.parquet")[0]}
        dropped = [i for i in old["tabs"]["A"] if i not in set(data["tabs"]["A"])]
        assert dropped and all(msb.row_side_pattern(rows[i]) not in
                               msb.BORDER_PATTERNS for i in dropped)
        assert data["tabs"]["A"] == [i for i in old["tabs"]["A"]
                                     if i not in set(dropped)]
        for k in "BCDE":
            assert data["tabs"][k] == old["tabs"][k] and len(old["tabs"][k]) == s["tabs"][k]
    assert digest() == before
    assert len(list((src / "thumbs").glob("*.jpg"))) == n_thumbs


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
