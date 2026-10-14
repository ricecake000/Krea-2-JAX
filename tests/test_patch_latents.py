"""Tests for scripts/modal_patch_latents.py (data-filter patch -> new
krea2-latents-v1 version) and the patched-version support in
krea2_jax/data.py::BucketDataset. CPU only, offline (stub VAE, fake Hub,
fake Modal volume).

    python tests/test_patch_latents.py      # or: pytest tests/test_patch_latents.py
"""

import atexit
import faulthandler
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import types
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

import modal_patch_latents as mpl  # noqa: E402
from krea2_jax.data import BucketDataset  # noqa: E402
from test_data import bucket_rows, make_fake_latent_cache  # noqa: E402

BF16 = ml_dtypes.bfloat16
MAIN_BASE = "92b5ee5"   # main before patched-version support
# BucketDataset.fingerprint() of the golden fixture (bucket_rows(), fixed
# manifest digests, FAKE_TEMPLATE) computed with main@92b5ee5's data.py
GOLDEN = {
    (16, (0.5, 0.3, 0.2), 0.5):
        "80e73388a5172ea72c8231870081fa7e17dad59b1671ba1174ed790489fc4269",
    (4096, (0.6, 0.3, 0.1), 0.25):
        "1768c9e5b57fc2d83fc94387f076010623f172c34b70ebc1ae8edd63ea3bef11",
}
FAKE_TEMPLATE = SimpleNamespace(text_rows=32, prefix_ids=[11, 12, 13],
                                suffix_ids=[14, 15], pad_id=0)
VAE = "stub"            # cache_latents.STUB_VAE_ID: the stub encoder's id
BASE_REV = "c0" * 20
_TMP = []


def tmpdir():
    d = Path(tempfile.mkdtemp(prefix="patch_latents_"))
    _TMP.append(d)
    return d


@atexit.register
def _cleanup():
    while _TMP:
        shutil.rmtree(_TMP.pop(), ignore_errors=True)


def quiet(fn, *a, **k):
    with redirect_stdout(StringIO()):
        return fn(*a, **k)


def expect_exit(fn, *needles):
    try:
        quiet(fn)
    except SystemExit as e:
        msg = str(e)
        for n in needles:
            assert n in msg, (n, msg)
        return msg
    raise AssertionError(f"no SystemExit (wanted {needles})")


def tree_digest(d):
    """{relpath: sha256} of every file under d (symlinks followed)."""
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(d).rglob("*")) if p.is_file()}


def ds_of(d, **kw):
    kw.setdefault("img_tokens", 16)
    return BucketDataset(d, FAKE_TEMPLATE, patch=2, **kw)


# ----------------------------------------------------------------- fixtures
def make_base(root):
    """bucket_rows() cache with 16-channel latents of VAE "stub" (two shards
    train-0000 / train-0001)."""
    make_fake_latent_cache(root / "base", bucket_rows(), lat_channels=16,
                           vae=VAE)
    return root / "base"


def base_rows(base_dir):
    t = pq.read_table(base_dir / "metadata.parquet").to_pylist()
    return {int(r["id"]): r for r in t}


def write_image(path, w, h, seed):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, "WEBP", quality=95)


def make_patch(root, base_dir, specs, name="patch", review_id="r1"):
    """specs: [{id, action, status, bucket (w, h) for crops, image (bool),
    caption suffix}] -> patch dir with patch.parquet / patch.json /
    images/."""
    import pyarrow as pa

    rows_b = base_rows(base_dir)
    d = root / name
    (d / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for s in specs:
        b = rows_b[s["id"]]
        w, h = s.get("bucket", (b["bucket_w"], b["bucket_h"]))
        suffix = s.get("suffix", "")
        row = {"id": s["id"], "action": s["action"], "status": s["status"],
               "bucket_w": w, "bucket_h": h, "notes": ""}
        for c in ("long", "medium", "short"):
            row[c + "_old"] = b[c]
            row[c] = b[c] + suffix
            row[c + "_tok"] = len(row[c].split())
        row["crop_box_orig"] = [1, 2, 300, 400] if s["action"] == "crop" \
            else None
        row["orig_w"], row["orig_h"] = 512, 512
        rows.append(row)
        if s.get("image", s["action"] == "crop"):
            write_image(d / "images" / f"{s['id']}.webp", w, h, s["id"])
    cols = list(rows[0])
    types_ = {"crop_box_orig": pa.list_(pa.int64())}
    pq.write_table(pa.table({c: pa.array([r[c] for r in rows],
                                         type=types_.get(c))
                             for c in cols}), d / "patch.parquet")
    (d / "patch.json").write_text(json.dumps({
        "review_id": review_id, "dataset_revision": "05a9a13d",
        "counts": {}}))
    (d / "report.html").write_text("<html>not uploaded</html>")
    return d


SPECS = [
    {"id": 13, "action": "crop", "status": "ok", "bucket": (48, 48),
     "suffix": ""},
    {"id": 450, "action": "crop", "status": "crop_unsure", "bucket": (64, 32)},
    {"id": 600, "action": "caption", "status": "ok",
     "suffix": ", white border"},
    {"id": 77, "action": "remove", "status": "ok"},
    {"id": 999, "action": "crop", "status": "crop_too_small",
     "bucket": (48, 48), "image": False},
    {"id": 5, "action": "caption", "status": "over_budget",
     "suffix": ", very long"},
]


def encode_stub(patch_dir):
    return quiet(mpl.cli, ["encode", "--patch-dir", str(patch_dir),
                           "--vae-stub", "--batch", "2"])


def build(base, patch, out, **kw):
    kw.setdefault("unapplied", "keep")
    return quiet(mpl.build_version, base, patch, out, BASE_REV, **kw)


def stub_latent(path, w, h):
    import torch

    import cache_latents as cl
    from cache_dataset import load_image
    lat = cl.StubEncoder()(load_image(str(path), w, h))[0]
    return lat.to(torch.bfloat16).float().numpy()


def old_data_module():
    """krea2_jax/data.py as of main@MAIN_BASE (None without git/commit)."""
    try:
        src = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{MAIN_BASE}:krea2_jax/data.py"],
            capture_output=True, text=True, check=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return None
    mod = types.ModuleType("old_krea2_data")
    exec(compile(src, f"{MAIN_BASE}:krea2_jax/data.py", "exec"), mod.__dict__)
    return mod


# -------------------------------------------------------------------- tests
def test_unpatched_fingerprint_unchanged():
    """An unpatched cache (every existing one, incl. run1's) loads and
    fingerprints exactly as before: golden values from main's code, and the
    old module side by side when git has it."""
    root = tmpdir()
    make_fake_latent_cache(root / "g", bucket_rows())
    mpath = root / "g" / "latents" / "manifest.json"
    m = json.loads(mpath.read_text())
    m["digests"] = {n: c * 64 for n, c in zip(m["shards"], "ab")}
    mpath.write_text(json.dumps(m))
    for (it, probs, ap), want in GOLDEN.items():
        ds = BucketDataset(root / "g", FAKE_TEMPLATE, img_tokens=it, patch=2,
                           caption_probs=probs, artist_prob=ap,
                           verify_digests=False)
        assert ds.fingerprint() == want, (it, ds.fingerprint())
        assert ds.superseded == {}
    # real digests, full verification, old vs new module
    make_fake_latent_cache(root / "r", bucket_rows(), lat_channels=16)
    old = old_data_module()
    new = ds_of(root / "r")
    if old is None:
        print("  (git history unavailable: old-module comparison skipped)")
    else:
        o = old.BucketDataset(root / "r", FAKE_TEMPLATE, img_tokens=16,
                              patch=2)
        assert o.fingerprint() == new.fingerprint()
        assert np.array_equal(o.shard_of, new.shard_of)
        assert np.array_equal(o.hw, new.hw)
        assert o.verified_shards == new.verified_shards == 2
        for j in range(new.n):
            assert np.array_equal(o.read_latent(j).view(np.uint16),
                                  new.read_latent(j).view(np.uint16))
    print("  unpatched: fingerprint == main@92b5ee5 (golden + old module) OK")


def test_build_and_load():
    root = tmpdir()
    base = make_base(root)
    before = tree_digest(base)
    fp_base = ds_of(base).fingerprint()
    patch = make_patch(root, base, SPECS)
    res = encode_stub(patch)
    assert res["n"] == 2
    patch_before = tree_digest(patch)
    rep = build(base, patch, root / "new", link_base_shards=True)
    assert tree_digest(base) == before, "base files changed"
    assert tree_digest(patch) == patch_before, "patch dir changed"
    assert rep["counts"] == {"replaced": 2, "recaptioned": 1, "removed": 1,
                             "unapplied_kept": 2, "unapplied_removed": 0,
                             "samples_before": 13, "samples_after": 12}
    assert rep["loader_check"]["verified_shards"] == 3

    new = root / "new"
    man = json.loads((new / "latents" / "manifest.json").read_text())
    bman = json.loads((base / "latents" / "manifest.json").read_text())
    assert man["shards"] == bman["shards"] + ["patch-0000.safetensors"]
    for s in bman["shards"]:   # old shards: same digests, same bytes
        assert man["digests"][s] == bman["digests"][s]
        assert (hashlib.sha256((new / "latents" / s).read_bytes()).hexdigest()
                == bman["digests"][s])
    assert man["n"] == 13 + 2
    # 13 lives in train-0001, 450 and 77 in train-0000
    assert man["superseded"] == {"train-0000.safetensors": [77, 450],
                                 "train-0001.safetensors": [13]}
    entry = man["patches"][-1]
    assert entry["base_revision"] == BASE_REV
    assert entry["review_id"] == "r1"
    assert entry["shard"]["count"] == 2

    ds = ds_of(new)
    assert ds.verified_shards == 3
    assert ds.n == 12 and 77 not in ds.ids.tolist()
    assert ds.fingerprint() != fp_base
    assert ds.fingerprint() == ds_of(new, verify_digests=False).fingerprint()
    rows_b = base_rows(base)
    pos = {i: j for j, i in enumerate(ds.ids.tolist())}
    # crops: new bucket, stub latent of the new image, patch shard
    for i, (w, h) in ((13, (48, 48)), (450, (64, 32))):
        j = pos[i]
        assert ds.shards[ds.shard_of[j]] == "patch-0000.safetensors"
        assert ds.hw[j].tolist() == [h // 16, w // 16]
        want = stub_latent(patch / "images" / f"{i}.webp", w, h)
        assert np.array_equal(ds.read_latent(j).astype(np.float32), want)
        out = np.zeros((16, ds.feat), BF16)
        ds.read_latent_into(j, out)
    # caption fix: new captions, old latent / shard
    j = pos[600]
    assert ds.caption(j, 0, False) == rows_b[600]["long"] + ", white border"
    assert ds.shards[ds.shard_of[j]] == "train-0001.safetensors"
    # unapplied (kept): unchanged
    j = pos[5]
    assert ds.caption(j, 2, False) == rows_b[5]["short"]
    j = pos[999]
    assert ds.hw[j].tolist() == [64 // 16, 32 // 16]
    # untouched ids: same latent bytes as the base
    bds = ds_of(base)
    bpos = {i: j for j, i in enumerate(bds.ids.tolist())}
    for i in (907, 2, 1001, 600, 999):
        assert np.array_equal(ds.read_latent(pos[i]).view(np.uint16),
                              bds.read_latent(bpos[i]).view(np.uint16))
    # metadata: base order and columns kept, token counts patched
    t_new = pq.read_table(new / "metadata.parquet")
    t_old = pq.read_table(base / "metadata.parquet")
    assert t_new.column_names == t_old.column_names + ["latent_shard"]
    for f in t_old.schema:
        assert t_new.schema.field(f.name).type == f.type
    assert t_new.column("id").to_pylist() == [
        i for i in t_old.column("id").to_pylist() if i != 77]
    r = {x["id"]: x for x in t_new.to_pylist()}
    assert r[600]["long_tok"] == len(r[600]["long"].split())
    assert r[13]["bucket_w"] == 48 and r[13]["tar"] == rows_b[13]["tar"]
    # the base itself still loads with its old fingerprint
    assert ds_of(base).fingerprint() == fp_base
    print("  build: patched latents/captions/buckets, removed id dropped, "
          "unapplied kept, old shards byte-identical, fingerprint changed, "
          "3 shards verified OK")


def test_unapplied_modes_and_validation():
    root = tmpdir()
    base = make_base(root)
    patch = make_patch(root, base, SPECS)
    encode_stub(patch)
    expect_exit(lambda: mpl.build_version(base, patch, root / "f", BASE_REV),
                "2 unapplied rows", "999:crop/crop_too_small",
                "5:caption/over_budget", "--unapplied keep")
    build(base, patch, root / "rm", unapplied="remove",
          link_base_shards=True)
    ds = ds_of(root / "rm")
    assert ds.n == 10
    assert not {5, 77, 999} & set(ds.ids.tolist())
    man = json.loads((root / "rm" / "latents" / "manifest.json").read_text())
    assert man["superseded"] == {"train-0000.safetensors": [5, 77, 450],
                                 "train-0001.safetensors": [13, 999]}
    assert man["patches"][-1]["counts"]["unapplied_removed"] == 2
    expect_exit(lambda: build(base, patch, root / "rm"), "already holds")
    p_all = make_patch(root, base, [
        {"id": i, "action": "remove", "status": "ok"}
        for i in base_rows(base)], name="all")
    expect_exit(lambda: build(base, p_all, root / "empty"),
                "would remove every sample")
    assert not (root / "empty").exists(), "empty build wrote files"
    expect_exit(lambda: mpl.build_version(base, patch, root / "x", ""),
                "--base-revision is required")

    def variant(name, specs, edit=None):
        p = make_patch(root, base, specs, name=name)
        if edit:
            edit(p)
        return p

    # patch made against other captions
    p = variant("old", SPECS[2:3], lambda p: _edit_parquet(
        p, lambda r: r.update(long_old="something else")))
    expect_exit(lambda: build(base, p, root / "o1"),
                "600: long_old differs from the base long")
    p = variant("tok", SPECS[2:3], lambda p: _edit_parquet(
        p, lambda r: r.update(short_tok=-1, medium=" ")))
    expect_exit(lambda: build(base, p, root / "o1"),
                "600: short_tok -1 is not a token count",
                "600: new medium caption is empty")
    # id not in the base / caption changing the bucket / foreign bucket
    p = variant("bad", [{"id": 600, "action": "caption", "status": "ok",
                         "bucket": (48, 48)},
                        {"id": 13, "action": "crop", "status": "ok",
                         "bucket": (96, 32)}],
                lambda p: _edit_parquet(p, lambda r: r.update(id=12345)
                                        if r["id"] == 600 else None))
    expect_exit(lambda: build(base, p, root / "o2"),
                "12345: not in the base metadata")
    p = variant("bkt", [{"id": 600, "action": "caption", "status": "ok",
                         "bucket": (48, 48)},
                        {"id": 13, "action": "crop", "status": "ok",
                         "bucket": (96, 32)}])
    expect_exit(lambda: build(base, p, root / "o3"),
                "600: caption row changes the bucket",
                "13: bucket 96x32 is not a dataset bucket")
    # image of the wrong size / missing / stray
    p = variant("img", SPECS[:2])
    write_image(p / "images" / "13.webp", 64, 32, 1)
    write_image(p / "images" / "4242.webp", 64, 32, 1)
    (p / "images" / "450.webp").unlink()
    expect_exit(lambda: encode_stub(p), "13: images/13.webp is 64x32",
                "450: no images/450.webp", "4242.webp")
    # shard encoded from other images
    p = variant("stale", SPECS[:2])
    encode_stub(p)
    write_image(p / "images" / "13.webp", 48, 48, 777)
    expect_exit(lambda: build(base, p, root / "o4"),
                "images_sha256", "re-run the encode step")
    # no encoded shard yet
    p = variant("noenc", SPECS[:1])
    expect_exit(lambda: build(base, p, root / "o5"), "run the encode step")
    # caption-only patch: no shard, no superseded, captions changed
    p = variant("caps", SPECS[2:3])
    rep = build(base, p, root / "o6")
    assert rep["patch_shard"] is None
    man = json.loads((root / "o6" / "latents" / "manifest.json").read_text())
    assert "superseded" not in man and man["shards"] == [
        "train-0000.safetensors", "train-0001.safetensors"]
    _link(base, root / "o6")
    assert ds_of(root / "o6").fingerprint() != ds_of(base).fingerprint()
    print("  build: --unapplied fail/remove, stale captions, unknown id, "
          "bucket rules, image checks, stale shard, caption-only OK")


def _edit_parquet(p, fn):
    import pyarrow as pa
    t = pq.read_table(p / "patch.parquet")
    rows = t.to_pylist()
    for r in rows:
        fn(r)
    pq.write_table(pa.Table.from_pylist(rows, schema=t.schema),
                   p / "patch.parquet")


def _link(base, new):
    for s in json.loads((base / "latents" / "manifest.json").read_text()
                        )["shards"]:
        dst = new / "latents" / s
        if not dst.exists():
            dst.symlink_to((base / "latents" / s).resolve())


def test_loader_rejects_inconsistent_patch():
    root = tmpdir()
    base = make_base(root)
    patch = make_patch(root, base, SPECS)
    encode_stub(patch)
    build(base, patch, root / "new", link_base_shards=True)
    new = root / "new"
    mpath = new / "latents" / "manifest.json"
    good = mpath.read_text()

    def with_manifest(edit, *needles):
        m = json.loads(good)
        edit(m)
        mpath.write_text(json.dumps(m))
        try:
            ds_of(new, verify_digests=False)
        except ValueError as e:
            for n in needles:
                assert n in str(e), (n, str(e))
        else:
            raise AssertionError(f"accepted ({needles})")
        finally:
            mpath.write_text(good)

    # superseded id that is not in its shard
    with_manifest(lambda m: m["superseded"]["train-0000.safetensors"].append(
        907 + 100000), "superseded ids are not in their shard",
        "train-0000.safetensors:100907")
    with_manifest(lambda m: m["superseded"].update({"nope.safetensors": [1]}),
                  "superseded shards are not in the manifest's shards")
    with_manifest(lambda m: m["superseded"].update(
        {"train-0001.safetensors": "13"}), "is not a list of integer ids")
    # without "superseded" the old rules apply: duplicates / orphans refused
    with_manifest(lambda m: m.pop("superseded"),
                  "ids appear in more than one shard",
                  "latents have no parquet row")
    # latent_shard column must match the resolution
    import pyarrow as pa
    ppath = new / "metadata.parquet"
    t = pq.read_table(ppath)
    col = t.column("latent_shard").to_pylist()
    ids = t.column("id").to_pylist()
    col[ids.index(13)] = "train-0001.safetensors"
    bad = t.set_column(t.column_names.index("latent_shard"), "latent_shard",
                       pa.array(col, pa.string()))
    keep = ppath.read_bytes()
    pq.write_table(bad, ppath)
    try:
        ds_of(new, verify_digests=False)
    except ValueError as e:
        assert "latent_shard values differ" in str(e), str(e)
        assert "13:'train-0001.safetensors'!='patch-0000.safetensors'" in \
            str(e), str(e)
    else:
        raise AssertionError("wrong latent_shard accepted")
    ppath.write_bytes(keep)
    # modified patch shard bytes under the manifest: --verify-latents refuses
    shard = new / "latents" / "patch-0000.safetensors"
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
    with safe_open(str(shard), framework="numpy") as f:
        meta = f.metadata()
    tensors = load_file(str(shard))
    tensors["13"] = tensors["13"].copy()
    tensors["13"][0, 0, 0] += BF16(1.0)
    save_file(tensors, str(shard), metadata=meta)
    try:
        ds_of(new)
    except ValueError as e:
        assert "patch-0000.safetensors: expected" in str(e), str(e)
    else:
        raise AssertionError("modified patch shard accepted")
    print("  loader: bad superseded / missing superseded / wrong "
          "latent_shard / modified patch shard refused OK")


def test_chained_patch():
    """A second patch on top of a patched version: ids living in
    patch-0000 are superseded there, the new shard is patch-0001."""
    root = tmpdir()
    base = make_base(root)
    p1 = make_patch(root, base, SPECS[:4])
    encode_stub(p1)
    build(base, p1, root / "v1", link_base_shards=True)
    v1 = root / "v1"
    p2 = make_patch(root, v1, [
        {"id": 13, "action": "crop", "status": "ok", "bucket": (64, 32)},
        {"id": 450, "action": "remove", "status": "ok"},
        {"id": 907, "action": "caption", "status": "ok", "suffix": ", inset"},
    ], name="p2", review_id="r2")
    encode_stub(p2)
    rep = build(v1, p2, root / "v2", link_base_shards=True)
    assert rep["loader_check"]["verified_shards"] == 4
    man = json.loads((root / "v2" / "latents" / "manifest.json").read_text())
    assert man["shards"][-2:] == ["patch-0000.safetensors",
                                  "patch-0001.safetensors"]
    assert man["superseded"] == {"patch-0000.safetensors": [13, 450],
                                 "train-0000.safetensors": [77, 450],
                                 "train-0001.safetensors": [13]}
    assert [e["review_id"] for e in man["patches"]] == ["r1", "r2"]
    assert man["n"] == 13 + 2 + 1
    ds = ds_of(root / "v2")
    pos = {i: j for j, i in enumerate(ds.ids.tolist())}
    assert ds.n == 11 and 450 not in pos
    assert ds.shards[ds.shard_of[pos[13]]] == "patch-0001.safetensors"
    want = stub_latent(p2 / "images" / "13.webp", 64, 32)
    assert np.array_equal(ds.read_latent(pos[13]).astype(np.float32), want)
    assert ds.caption(pos[907], 1, False).endswith(", inset")
    print("  chained patch: patch-0001 over patch-0000, superseded across "
          "patch shards OK")


def test_cache_latents_refuses_patched():
    import cache_latents as cl
    root = tmpdir()
    base = make_base(root)
    patch = make_patch(root, base, SPECS[2:4])
    build(base, patch, root / "new")
    expect_exit(lambda: cl.write_manifest(root / "new",
                                          base / "metadata.parquet"),
                "belongs to a patched version")
    print("  cache_latents --write-manifest refuses a patched version OK")


# ------------------------------------------------------------ fake Hub
class FakeHub:
    """Minimal HfApi: commits {sha: {path: entry}}, branches, LFS for
    .parquet/.safetensors (sha256), git blob ids for the rest."""

    def __init__(self, base, private=True):
        self.private = private
        self.commits, self.refs, self.calls = {}, {}, []
        self.trees, self.before_commit = 0, None
        files = {}
        for p in sorted(base.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(base))] = self.entry(p)
        self.commits[BASE_REV] = files
        self.refs["main"] = BASE_REV

    @staticmethod
    def entry(p):
        d = mpl._local_digests(p)
        lfs = p.suffix in (".parquet", ".safetensors")
        return {"size": d["size"], "blob_id": d["git_sha1"],
                "sha256": d["sha256"] if lfs else None}

    def resolve(self, rev):
        return self.refs.get(rev, rev)

    def repo_info(self, repo, repo_type, revision):
        return SimpleNamespace(private=self.private,
                               sha=self.resolve(revision))

    def list_repo_tree(self, repo, repo_type, recursive, expand, revision):
        self.trees += 1
        out = []
        for path, e in self.commits[self.resolve(revision)].items():
            lfs = SimpleNamespace(sha256=e["sha256"]) if e["sha256"] else None
            out.append(SimpleNamespace(path=path, size=e["size"],
                                       blob_id=e["blob_id"], lfs=lfs))
        out.append(SimpleNamespace(path="latents", blob_id=None))
        return out

    def list_repo_refs(self, repo, repo_type):
        return SimpleNamespace(branches=[SimpleNamespace(
            name=k, target_commit=v) for k, v in self.refs.items()])

    def create_branch(self, repo, branch, revision, repo_type):
        self.calls.append(("create_branch", branch, revision))
        self.refs[branch] = self.resolve(revision)

    def add_commit(self, branch, files):
        sha = hashlib.sha1(json.dumps(sorted(files.items())).encode()
                           ).hexdigest()
        self.commits[sha] = files
        self.refs[branch] = sha
        return sha

    def create_commit(self, repo, ops, repo_type, revision, commit_message,
                      commit_description, parent_commit=None):
        if self.before_commit:
            self.before_commit()
        self.calls.append(("create_commit", revision,
                           sorted(p for p, _ in ops), parent_commit))
        if parent_commit is not None and parent_commit != self.resolve(
                revision):
            raise RuntimeError("412 parent commit is not the branch head")
        files = dict(self.commits[self.resolve(revision)])
        for path, local in ops:
            files[path] = self.entry(Path(local))
        self.add_commit(revision, files)


def test_upload_fake_hub():
    root = tmpdir()
    base = make_base(root)
    patch = make_patch(root, base, SPECS)
    encode_stub(patch)
    build(base, patch, root / "new")
    new = root / "new"
    written = ["latents/manifest.json", "latents/patch-0000.safetensors",
               "metadata.parquet"]

    def up(hub, branch="datafilter-r1", where=new, **kw):
        return quiet(mpl.upload_version, where, branch, api=hub,
                     make_op=lambda p, local: (p, local), **kw)

    hub = FakeHub(base)
    expect_exit(lambda: up(hub, "main"), "refusing to commit to main")
    res = up(hub)
    assert res["uploaded"] and res["branch"] == "datafilter-r1"
    assert hub.calls[0] == ("create_branch", "datafilter-r1", BASE_REV)
    assert hub.calls[1] == ("create_commit", "datafilter-r1", written,
                            BASE_REV)
    assert hub.trees == 2, f"{hub.trees} tree listings (base + new head)"
    assert hub.refs["main"] == BASE_REV, "main must not move"
    files = hub.commits[hub.refs["datafilter-r1"]]
    assert files["latents/train-0000.safetensors"] == \
        hub.commits[BASE_REV]["latents/train-0000.safetensors"]
    # idempotent re-run: base + branch head listed once each
    n, hub.trees = len(hub.calls), 0
    assert up(hub)["uploaded"] is False and len(hub.calls) == n
    assert hub.trees == 2, hub.trees
    # the Hub's base is not what was built on
    hub2 = FakeHub(base)
    hub2.commits[BASE_REV]["latents/train-0001.safetensors"]["sha256"] = \
        "0" * 64
    expect_exit(lambda: up(hub2), "is not the base this version was built",
                "latents/train-0001.safetensors")
    expect_exit(lambda: up(FakeHub(base, private=False)), "is not private")

    # (review 1) files modified after build: refused before any Hub call
    for rel in ("latents/patch-0000.safetensors", "metadata.parquet"):
        bad = Path(shutil.copytree(new, root / f"mod_{rel.replace('/', '_')}"))
        with open(bad / rel, "ab") as f:
            f.write(b"\0")
        hub3 = FakeHub(base)
        expect_exit(lambda: up(hub3, where=bad),
                    "local files differ from the manifest", rel)
        assert hub3.calls == [] and hub3.trees == 0

    # (review 2) an existing branch that does not fit: refused untouched
    hub4 = FakeHub(base)
    other = dict(hub4.commits[BASE_REV])
    other["latents/patch-0009.safetensors"] = {
        "size": 1, "blob_id": "x", "sha256": "1" * 64}
    head4 = hub4.add_commit("datafilter-r1", other)
    expect_exit(lambda: up(hub4), "exists with content that does not fit",
                "latents/patch-0009.safetensors")
    hub5 = FakeHub(base)
    other = dict(hub5.commits[BASE_REV])
    other["latents/train-0000.safetensors"] = {
        "size": 1, "blob_id": "x", "sha256": "2" * 64}
    head5 = hub5.add_commit("datafilter-r1", other)
    expect_exit(lambda: up(hub5), "exists with content that does not fit",
                "sha256 latents/train-0000.safetensors")
    for h, head in ((hub4, head4), (hub5, head5)):
        assert h.refs["datafilter-r1"] == head
        assert not [c for c in h.calls if c[0] == "create_commit"]
    # (review round 2) repro 1: an existing branch holding exactly this
    # version PLUS an unlisted shard is not an idempotent no-op
    hub8 = FakeHub(base)
    up(hub8)
    extra = {"size": 1, "blob_id": "x", "sha256": "3" * 64}
    head8 = hub8.add_commit("datafilter-r1", dict(
        hub8.commits[hub8.refs["datafilter-r1"]],
        **{"latents/patch-0009.safetensors": extra}))
    n8 = len(hub8.calls)
    expect_exit(lambda: up(hub8), "exists with content that does not fit",
                "1 shards not in the new manifest",
                "latents/patch-0009.safetensors")
    assert hub8.refs["datafilter-r1"] == head8 and len(hub8.calls) == n8
    # repro 2: a base revision with an unlisted shard (or a file at the
    # patch shard's path) is refused before any branch is created
    for stray in ("latents/patch-0009.safetensors",
                  "latents/patch-0000.safetensors"):
        hub9 = FakeHub(base)
        hub9.commits[BASE_REV][stray] = dict(extra)
        expect_exit(lambda: up(hub9), "is not the base this version was "
                    "built on", f"1 shards not in the new manifest ({stray})")
        assert hub9.calls == [] and "datafilter-r1" not in hub9.refs
    # a compatible existing branch (e.g. created, commit lost) gets the
    # commit with parent_commit = the checked head
    hub6 = FakeHub(base)
    hub6.refs["datafilter-r1"] = BASE_REV
    assert up(hub6)["uploaded"]
    assert hub6.calls == [("create_commit", "datafilter-r1", written,
                           BASE_REV)]
    # the branch moves between the check and the commit: refused by the Hub
    hub7 = FakeHub(base)

    def race():
        hub7.before_commit = None
        hub7.add_commit("datafilter-r1", dict(hub7.commits[BASE_REV],
                                              **{"x.txt": {
                                                  "size": 1, "blob_id": "y",
                                                  "sha256": None}}))
    hub7.before_commit = race
    try:
        up(hub7)
    except RuntimeError as e:
        assert "parent commit" in str(e)
    else:
        raise AssertionError("commit on a moved branch accepted")
    print("  upload: branch at base revision, 3 files committed with "
          "parent_commit, main untouched, verified, idempotent, 2 tree "
          "listings; modified local files / incompatible existing branch / "
          "moved branch / foreign base / public repo / main refused OK")


# -------------------------------------------------------- Modal (no Modal)
class FakeVol:
    def __init__(self):
        self.commits = self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


def test_modal_side_without_modal():
    import cache_latents as cl
    root = tmpdir()
    base = make_base(root)
    patch = make_patch(root, base, SPECS)
    files = mpl.patch_upload_list(patch)
    rels = [r for r, _ in files]
    assert rels == ["patch.parquet", "patch.json", "images/13.webp",
                    "images/450.webp"], rels   # report.html stays local
    digests, key = mpl.upload_digests(files)
    assert mpl.upload_digests(mpl.patch_upload_list(patch))[1] == key
    assert len(key) == 16
    # "upload": copy to the fake volume root
    vroot = root / "vol"
    for rel, p in files:
        (vroot / key / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, vroot / key / rel)
    vol = FakeVol()
    loads = []

    def loader(impl):
        loads.append(impl)
        return cl.StubEncoder(), "cpu"

    res = quiet(mpl.encode_remote_impl, key, digests, "2d", 2, 2, False, vol,
                root=vroot, loader=loader)
    assert res["n"] == 2 and not res["skipped"] and vol.commits == 1
    shard = vroot / key / mpl.PATCH_LATENTS
    assert res["sha256"] == mpl.file_sha256(shard)
    meta, keys = mpl.read_shard_header(shard)
    assert sorted(keys) == ["13", "450"] and meta["vae_impl"] == "2d"
    assert meta["source"] == "patch:r1"
    # same images + impl: skipped, no model load
    res2 = quiet(mpl.encode_remote_impl, key, digests, "2d", 2, 2, False,
                 vol, root=vroot, loader=loader)
    assert res2["skipped"] and res2["sha256"] == res["sha256"]
    assert loads == ["2d"]
    # a corrupted upload is refused before any encode
    (vroot / key / "images" / "13.webp").write_bytes(b"junk")
    try:
        quiet(mpl.encode_remote_impl, key, digests, "2d", 2, 2, True, vol,
              root=vroot, loader=loader)
    except RuntimeError as e:
        assert "uploaded patch files differ" in str(e)
    else:
        raise AssertionError("corrupted upload accepted")
    # a shard encoded with the real vae id is refused by a stub-vae base
    shutil.copyfile(shard, patch / mpl.PATCH_LATENTS)
    expect_exit(lambda: build(base, patch, root / "new"),
                "metadata vae 'Qwen/Qwen-Image' != 'stub'")
    print("  modal side: upload list/key, container encode + skip + "
          "corrupt-upload refusal (fake volume) OK")


def main():
    faulthandler.dump_traceback_later(300, exit=True)
    t0 = time.time()
    try:
        test_unpatched_fingerprint_unchanged()
        test_build_and_load()
        test_unapplied_modes_and_validation()
        test_loader_rejects_inconsistent_patch()
        test_chained_patch()
        test_cache_latents_refuses_patched()
        test_upload_fake_hub()
        test_modal_side_without_modal()
    finally:
        _cleanup()
    faulthandler.cancel_dump_traceback_later()
    print(f"ALL PATCH LATENTS TESTS PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
