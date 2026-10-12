"""Tests for scripts/cache_latents.py (krea2-latents-v1 writer), CPU only,
offline, with the script's StubEncoder in place of the VAE.

    python tests/test_cache_latents.py
"""

import faulthandler
import io
import json
import shutil
import sys
import tarfile
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import ml_dtypes  # noqa: E402,F401 - numpy-framework bf16 reads need it
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cache_latents as cl  # noqa: E402
from cache_dataset import VaeEncoder, load_image  # noqa: E402

BF16 = ml_dtypes.bfloat16

# (id, bucket_w, bucket_h, stored image size (w, h), tar). id 5531 is stored
# at the wrong size -> resized with a warning. ids are not sorted, rank order
# (parquet row order) differs from id order.
SAMPLES = [
    (907, 32, 64, (32, 64), "data/train-0000.tar"),
    (12, 64, 32, (64, 32), "data/train-0000.tar"),
    (5531, 32, 64, (48, 80), "data/train-0000.tar"),
    (77, 64, 32, (64, 32), "data/train-0001.tar"),
    (3000, 32, 64, (32, 64), "data/train-0001.tar"),
    (41, 32, 64, (32, 64), "data/train-0001.tar"),
]


def sample_json(sid, w, h):
    return {"long": f"long caption {sid}", "medium": f"medium {sid}",
            "short": f"short {sid}", "tags": "1girl, solo",
            "character_name": "c", "series_name": "s",
            "artist_name": "" if sid % 2 else f"artist{sid}",
            "rating": "general", "bucket_w": w, "bucket_h": h,
            "score": 0.5, "rank": 0, "id": sid}


def image_bytes(sid, size):
    rng = np.random.default_rng(sid)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="WEBP", lossless=True)
    return buf.getvalue()


def add_member(tf, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def make_fake_dataset(root, samples=SAMPLES):
    """Tars (webp, json, txt per sample, in that order) + metadata.parquet
    with every S1 column, rows in rank order. Returns {id: webp bytes}."""
    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    by_tar, blobs = {}, {}
    for sid, w, h, size, tar in samples:
        by_tar.setdefault(tar, []).append((sid, w, h, size))
    for tar, rows in by_tar.items():
        with tarfile.open(root / tar, "w") as tf:
            for sid, w, h, size in rows:
                blobs[sid] = image_bytes(sid, size)
                add_member(tf, f"{sid}.webp", blobs[sid])
                add_member(tf, f"{sid}.json",
                           json.dumps(sample_json(sid, w, h)).encode())
                add_member(tf, f"{sid}.txt", b"some caption")
    write_parquet(root / "metadata.parquet", samples)
    return blobs


def write_parquet(path, samples):
    cols = {k: [] for k in (
        "id", "long", "medium", "short", "tags", "character_name",
        "series_name", "artist_name", "rating", "bucket_w", "bucket_h",
        "score", "rank", "long_tok", "medium_tok", "short_tok", "tags_tok",
        "tar")}
    for rank, (sid, w, h, _, tar) in enumerate(samples):
        j = sample_json(sid, w, h)
        for k in ("long", "medium", "short", "tags", "character_name",
                  "series_name", "artist_name", "rating"):
            cols[k].append(j[k])
        cols["id"].append(sid)
        cols["bucket_w"].append(w)
        cols["bucket_h"].append(h)
        cols["score"].append(1.0 - rank / 10)
        cols["rank"].append(rank)
        for k in ("long_tok", "medium_tok", "short_tok", "tags_tok"):
            cols[k].append(5 + rank)
        cols["tar"].append(tar)
    schema = pa.schema([(k, pa.float32() if k == "score" else pa.int64()
                         if k in ("id", "bucket_w", "bucket_h", "rank")
                         or k.endswith("_tok") else pa.string())
                        for k in cols])
    pq.write_table(pa.table(cols, schema=schema), str(path))


def expected_latent(blob, w, h):
    img = load_image(io.BytesIO(blob), w, h)
    return cl.StubEncoder()(img)[0].to(torch.bfloat16).float().numpy()


def read_shard(path):
    with safe_open(str(path), framework="numpy") as f:
        return dict(f.metadata()), {k: f.get_tensor(k) for k in f.keys()}


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


def expect_exit(fn, *needles):
    try:
        quiet(fn)
    except SystemExit as e:
        msg = str(e)
        for n in needles:
            assert n in msg, (n, msg)
        return msg
    raise AssertionError(f"expected SystemExit mentioning {needles}")


# ---------------------------------------------------------------------------
def test_bf16_roundtrip(root):
    """torch-written bf16 reads back via the numpy framework as ml_dtypes."""
    x = torch.randn(16, 4, 8).to(torch.bfloat16)
    save_file({"1": x}, str(root / "rt.safetensors"), metadata={"a": "b"})
    with safe_open(str(root / "rt.safetensors"), framework="numpy") as f:
        y = f.get_tensor("1")
        assert f.get_slice("1").get_dtype() == "BF16"
    assert y.dtype == BF16 and y.shape == (16, 4, 8), (y.dtype, y.shape)
    assert np.array_equal(y.view(np.uint16),
                          x.view(torch.int16).numpy().view(np.uint16))
    print("  bf16 torch -> numpy round trip OK")


def test_parse_shards():
    assert cl.parse_shards("0-3") == [0, 1, 2, 3]
    assert cl.parse_shards("5,0-1, 3") == [0, 1, 3, 5]
    assert cl.shard_name("data/train-0042.tar") == "train-0042.safetensors"
    assert cl.shard_number("data/train-0042.tar") == 42
    print("  shard spec parsing OK")


def test_batched_vae(root):
    """vae_encode_batch == per-sample VaeEncoder.__call__ (same ops and
    normalization) on a fake diffusers-like VAE."""
    torch.manual_seed(0)
    conv = torch.nn.Conv3d(3, 32, (1, 8, 8), stride=(1, 8, 8))

    class FakeVae:
        def encode(self, x):
            assert x.dim() == 5 and x.shape[2] == 1, x.shape
            mean, logvar = conv(x).chunk(2, dim=1)
            return SimpleNamespace(latent_dist=SimpleNamespace(
                sample=lambda: mean))  # deterministic stand-in

    enc = VaeEncoder.__new__(VaeEncoder)
    enc.vae, enc.device, enc.dtype = FakeVae(), "cpu", torch.float32
    enc.mean = torch.linspace(-1, 1, 16).view(1, -1, 1, 1, 1)
    enc.std = torch.linspace(0.5, 2, 16).view(1, -1, 1, 1, 1)
    imgs = torch.rand(3, 3, 32, 64) * 2 - 1
    batched = cl.vae_encode_batch(enc, imgs)
    single = torch.stack([enc(imgs[i:i + 1]) for i in range(3)])
    assert batched.shape == (3, 16, 4, 8), batched.shape
    torch.testing.assert_close(batched, single, rtol=1e-5, atol=1e-5)
    print("  vae_encode_batch == stacked VaeEncoder.__call__ OK")


def test_process_shard(src, out_dir):
    blobs = make_fake_dataset(src)
    tar0, out0 = src / "data/train-0000.tar", out_dir / "train-0000.safetensors"
    enc = cl.StubEncoder()
    logs = []
    stats = cl.process_shard(tar0, out0, enc, 2, vae_id="stub",
                             source="data/train-0000.tar", expected_count=3,
                             decode_threads=2, log=logs.append)
    assert stats["skipped"] is False and stats["n"] == 3, stats
    assert stats["resized"] == 1 and len(logs) == 1 and "5531" in logs[0], logs
    # Batches never mix buckets and never exceed --batch.
    assert sorted(enc.calls) == [(1, 3, 32, 64), (2, 3, 64, 32)], enc.calls
    assert not list(out_dir.glob("*.tmp"))
    meta, tensors = read_shard(out0)
    assert meta == {"format": "krea2-latents-v1", "vae": "stub",
                    "source": "data/train-0000.tar", "count": "3"}, meta
    assert set(tensors) == {"907", "12", "5531"}, set(tensors)
    for sid, w, h, _, tar in SAMPLES[:3]:
        t = tensors[str(sid)]
        assert t.dtype == BF16 and t.shape == (16, h // 8, w // 8), (
            sid, t.dtype, t.shape)
        np.testing.assert_allclose(t.astype(np.float32),
                                   expected_latent(blobs[sid], w, h),
                                   rtol=1e-2, atol=1e-2)
    print("  process_shard: S1 keys/shapes/bf16/metadata, bucket batching, "
          "resize warning OK")

    # Resumable: a complete shard is skipped without reading the tar ...
    mtime = out0.stat().st_mtime_ns
    enc2 = cl.StubEncoder()
    st = cl.process_shard(Path("/nonexistent.tar"), out0, enc2, 2,
                          vae_id="stub", source="data/train-0000.tar",
                          expected_count=3)
    assert st["skipped"] and not enc2.calls and out0.stat().st_mtime_ns == mtime
    # ... or after counting the tar when no count is given.
    st = cl.process_shard(tar0, out0, enc2, 2, vae_id="stub",
                          source="data/train-0000.tar")
    assert st["skipped"] and not enc2.calls
    # A different VAE id re-encodes; a count mismatch is an error.
    st = cl.process_shard(tar0, out0, enc2, 8, vae_id="other",
                          source="data/train-0000.tar", log=lambda m: None)
    assert not st["skipped"] and len(enc2.calls) == 2, enc2.calls
    try:
        cl.process_shard(tar0, out_dir / "x.safetensors", enc2, 2,
                         vae_id="stub", source="data/train-0000.tar",
                         expected_count=4)
        raise AssertionError("count mismatch not detected")
    except ValueError as e:
        assert "expects 4" in str(e), e
    # A truncated shard is not "done" and gets rewritten.
    cl.process_shard(tar0, out0, enc2, 2, vae_id="stub",
                     source="data/train-0000.tar", log=lambda m: None)
    data = out0.read_bytes()
    ref_meta, ref = read_shard(out0)
    out0.write_bytes(data[:len(data) // 2])
    assert not cl.shard_done(out0, "stub", "data/train-0000.tar", 3)
    st = cl.process_shard(tar0, out0, cl.StubEncoder(), 2, vae_id="stub",
                          source="data/train-0000.tar", expected_count=3,
                          log=lambda m: None)
    assert not st["skipped"]
    meta, tensors = read_shard(out0)  # (header key order is not stable)
    assert meta == ref_meta and set(tensors) == set(ref)
    assert all(np.array_equal(tensors[k].view(np.uint16), ref[k].view(np.uint16))
               for k in ref)
    print("  process_shard: resume skip, vae change, count mismatch, "
          "truncated-file rewrite OK")


def test_main_and_manifest(src, out):
    args = ["--tar-dir", str(src), "--out", str(out), "--vae-stub",
            "--device", "cpu", "--num-workers", "2", "--batch", "2",
            "--decode-threads", "2"]
    _, log0 = quiet(cl.main, args + ["--worker-index", "0"])
    _, log1 = quiet(cl.main, args + ["--worker-index", "1"])
    assert "1 to encode (3 images)" in log0 and "train-0000" in log0, log0
    assert "1 to encode (3 images)" in log1 and "train-0001" in log1, log1
    lat = out / "latents"
    assert sorted(p.name for p in lat.iterdir()) == [
        "train-0000.safetensors", "train-0001.safetensors"]
    assert (src / "data/train-0000.tar").exists()  # local tars never deleted
    _, log = quiet(cl.main, args + ["--worker-index", "0"])
    assert "0 to encode" in log, log  # resumable through main()

    quiet(cl.main, ["--tar-dir", str(src), "--out", str(out),
                    "--write-manifest"])
    manifest = json.loads((lat / "manifest.json").read_text())
    assert manifest == {"format": "krea2-latents-v1", "vae": "stub",
                        "n": len(SAMPLES), "shards": [
                            "train-0000.safetensors",
                            "train-0001.safetensors"]}, manifest
    assert ((out / "metadata.parquet").read_bytes()
            == (src / "metadata.parquet").read_bytes())
    ids = set()
    for name in manifest["shards"]:
        _, tensors = read_shard(lat / name)
        ids |= set(tensors)
    assert ids == {str(s[0]) for s in SAMPLES}
    print("  main(): 2 workers striped, resume, --write-manifest OK")


def test_manifest_errors(src, out, tmp):
    """--write-manifest refuses an inconsistent cache."""
    def run(parquet_dir, lat_dir=out):
        return lambda: cl.main(["--tar-dir", str(parquet_dir), "--out",
                                str(lat_dir), "--write-manifest"])

    # parquet id with no latent
    extra = tmp / "extra"
    extra.mkdir()
    write_parquet(extra / "metadata.parquet",
                  SAMPLES + [(999, 64, 32, (64, 32), "data/train-0001.tar")])
    expect_exit(run(extra), "parquet id without latent: 1", "999")
    # latent with no parquet row
    fewer = tmp / "fewer"
    fewer.mkdir()
    write_parquet(fewer / "metadata.parquet", SAMPLES[:-1])
    expect_exit(run(fewer), "latent without parquet row: 1", "41")
    # wrong bucket in the parquet -> shape mismatch
    swapped = tmp / "swapped"
    swapped.mkdir()
    write_parquet(swapped / "metadata.parquet",
                  [(907, 64, 32, (64, 32), "data/train-0000.tar")]
                  + SAMPLES[1:])
    expect_exit(run(swapped), "wrong shape: 1", "907")
    # missing shard file, then a duplicated latent in a stray copy
    broken = tmp / "broken"
    shutil.copytree(out, broken)
    (broken / "latents/manifest.json").unlink()
    (broken / "latents/train-0001.safetensors").unlink()
    expect_exit(run(src, broken), "missing/unreadable shard: 1",
                "parquet id without latent: 3")
    shutil.copy(out / "latents/train-0000.safetensors",
                broken / "latents/train-0001.safetensors")
    expect_exit(run(src, broken), "duplicate latent: 3", "wrong source")
    assert not (broken / "latents/manifest.json").exists()
    print("  --write-manifest: missing id, extra latent, wrong shape, "
          "missing shard, duplicates all rejected OK")


def test_repo_download(src, tmp):
    """--repo path: tars fetched into the download dir and deleted after
    their shard is written (unless --keep-tars); --shards selects."""
    fetched = []

    def fake_fetch(repo, filename, download_dir):
        assert repo == "fake/repo", repo
        dst = Path(download_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / filename, dst)
        fetched.append((filename, dst))
        return dst

    orig = cl.fetch_file
    cl.fetch_file = fake_fetch
    try:
        out = tmp / "remote"
        quiet(cl.main, ["--repo", "fake/repo", "--out", str(out),
                        "--vae-stub", "--device", "cpu", "--shards", "1"])
        assert [f for f, _ in fetched] == ["metadata.parquet",
                                           "data/train-0001.tar"], fetched
        assert sorted(p.name for p in (out / "latents").iterdir()) == [
            "train-0001.safetensors"]
        assert not fetched[1][1].exists()  # tar deleted
        assert not (out / "_download").exists() or not any(
            (out / "_download").rglob("*.tar"))
        fetched.clear()
        keep = tmp / "keep"
        quiet(cl.main, ["--repo", "fake/repo", "--out", str(keep),
                        "--vae-stub", "--device", "cpu", "--keep-tars",
                        "--download-dir", str(tmp / "dl")])
        assert [f for f, _ in fetched] == [
            "metadata.parquet", "data/train-0000.tar", "data/train-0001.tar"]
        assert all(p.exists() for _, p in fetched)
        quiet(cl.main, ["--repo", "fake/repo", "--out", str(keep),
                        "--write-manifest", "--download-dir",
                        str(tmp / "dl")])
        assert json.loads((keep / "latents/manifest.json").read_text())[
            "n"] == len(SAMPLES)
    finally:
        cl.fetch_file = orig
    expect_exit(lambda: cl.main(["--tar-dir", str(src), "--out",
                                 str(tmp / "x"), "--shards", "7"]),
                "not in the parquet")
    print("  --repo: download, delete-after-write, --keep-tars, --shards OK")


def main():
    faulthandler.dump_traceback_later(120, exit=True)  # never hang CI
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, out = tmp / "src", tmp / "out"
        test_bf16_roundtrip(tmp)
        test_parse_shards()
        test_batched_vae(tmp)
        test_process_shard(src, tmp / "direct")
        test_main_and_manifest(src, out)
        test_manifest_errors(src, out, tmp)
        test_repo_download(src, tmp)
    faulthandler.cancel_dump_traceback_later()
    print(f"ALL CACHE_LATENTS TESTS PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
