"""scripts/cache_latents.py (krea2-latents-v1 writer) on a fake dataset, StubEncoder in place of the VAE."""

import hashlib
import io
import json
import shutil
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from safetensors import safe_open

torch = pytest.importorskip("torch")
import cache_latents as cl  # noqa: E402
from cache_dataset import load_image  # noqa: E402
from helpers import BF16, file_sha256, make_tiny_tokenizer  # noqa: E402

# (id, bucket_w, bucket_h, stored (w, h), tar); 5531 is stored at the wrong size
SAMPLES = [(907, 32, 64, (32, 64), "data/train-0000.tar"), (12, 64, 32, (64, 32), "data/train-0000.tar"),
           (5531, 32, 64, (48, 80), "data/train-0000.tar"), (77, 64, 32, (64, 32), "data/train-0001.tar"),
           (3000, 32, 64, (32, 64), "data/train-0001.tar"), (41, 32, 64, (32, 64), "data/train-0001.tar")]
TEXT = ("long", "medium", "short", "tags", "character_name", "series_name", "artist_name", "rating")
TAR0 = "data/train-0000.tar"
GOOD = hashlib.sha256(b"x").hexdigest()


def write_parquet(path, samples):
    t = pa.Table.from_pylist([{"id": sid, **{c: f"{c} {sid}" for c in TEXT}, "bucket_w": w, "bucket_h": h,
                               "score": 1.0 - r / 10, "rank": r, **{f"{c}_tok": 5 + r for c in TEXT[:4]},
                               "tar": tar} for r, (sid, w, h, _, tar) in enumerate(samples)])
    pq.write_table(t.set_column(t.column_names.index("score"), "score", t["score"].cast(pa.float32())), str(path))


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """Tars of (webp, json, txt) per sample + metadata.parquet. -> (root, {id: webp bytes})."""
    root, blobs = tmp_path_factory.mktemp("src"), {}
    (root / "data").mkdir()
    for tar in sorted({s[4] for s in SAMPLES}):
        with tarfile.open(root / tar, "w") as tf:
            for sid, w, h, size, _ in (s for s in SAMPLES if s[4] == tar):
                arr = np.random.default_rng(sid).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
                Image.fromarray(arr).save(buf := io.BytesIO(), format="WEBP", lossless=True)
                blobs[sid] = buf.getvalue()
                for name, data in ((f"{sid}.webp", blobs[sid]), (f"{sid}.txt", b"caption"),
                                   (f"{sid}.json", json.dumps({"id": sid, "bucket_w": w, "bucket_h": h}).encode())):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
    write_parquet(root / "metadata.parquet", SAMPLES)
    return root, blobs


@pytest.fixture(scope="module")
def cache(src, tmp_path_factory):
    out = tmp_path_factory.mktemp("cache")
    cl.main(["--tar-dir", str(src[0]), "--out", str(out), "--vae-stub", "--device", "cpu",
             "--batch", "2", "--decode-threads", "2"])
    cl.main(["--tar-dir", str(src[0]), "--out", str(out), "--write-manifest"])
    return out


def shard0(src, out, enc=None, batch=2, vae_id="stub", tar=None, **kw):
    return cl.process_shard(tar or src[0] / TAR0, out, enc or cl.StubEncoder(), batch, vae_id=vae_id,
                            source=TAR0, decode_threads=2, log=lambda m: None, **kw)


def read_shard(path):
    with safe_open(str(path), framework="numpy") as f:
        return dict(f.metadata()), {k: f.get_tensor(k) for k in f.keys()}


def test_process_shard_format_and_bucket_batching(src, tmp_path):
    enc, out = cl.StubEncoder(), tmp_path / "train-0000.safetensors"
    stats = shard0(src, out, enc, expected_count=3)
    assert stats["n"] == 3 and stats["resized"] == 1 and not stats["skipped"]
    assert sorted(enc.calls) == [(1, 3, 32, 64), (2, 3, 64, 32)]
    assert [p.name for p in tmp_path.iterdir()] == [out.name]
    meta, tensors = read_shard(out)
    assert meta == {"format": "krea2-latents-v1", "vae": "stub", "source": TAR0, "count": "3"}
    assert set(tensors) == {"907", "12", "5531"}
    for sid, w, h, _, _ in SAMPLES[:3]:
        t = tensors[str(sid)]
        assert t.dtype == BF16 and t.shape == (16, h // 8, w // 8)
        want = cl.StubEncoder()(load_image(io.BytesIO(src[1][sid]), w, h))[0].to(torch.bfloat16).float()
        np.testing.assert_allclose(t.astype(np.float32), want.numpy(), rtol=1e-2, atol=1e-2)


def test_process_shard_resume(src, tmp_path):
    """Complete shard skipped (no tar read); other vae re-encodes; count mismatch; truncation rewritten."""
    out = tmp_path / "train-0000.safetensors"
    shard0(src, out, expected_count=3)
    ref, mtime, enc = read_shard(out), out.stat().st_mtime_ns, cl.StubEncoder()
    assert shard0(src, out, enc, tar=tmp_path / "nonexistent.tar", expected_count=3)["skipped"]
    assert shard0(src, out, enc)["skipped"] and not enc.calls and out.stat().st_mtime_ns == mtime
    assert not shard0(src, out, enc, batch=8, vae_id="other")["skipped"] and len(enc.calls) == 2
    with pytest.raises(ValueError, match="expects 4"):
        shard0(src, tmp_path / "x.safetensors", expected_count=4)
    shard0(src, out, expected_count=3)
    out.write_bytes(out.read_bytes()[: out.stat().st_size // 2])
    assert not shard0(src, out, expected_count=3)["skipped"]
    meta, tensors = read_shard(out)
    assert meta == ref[0] and set(tensors) == set(ref[1])
    assert all(np.array_equal(tensors[k].view(np.uint16), v.view(np.uint16)) for k, v in ref[1].items())


def test_main_writes_manifest_with_digests(src, cache):
    lat = cache / "latents"
    manifest = json.loads((lat / "manifest.json").read_text())
    digests = manifest.pop("digests")
    assert manifest == {"format": "krea2-latents-v1", "vae": "stub", "n": len(SAMPLES),
                        "shards": ["train-0000.safetensors", "train-0001.safetensors"]}
    assert digests == {n: file_sha256(lat / n) for n in manifest["shards"]}
    assert (cache / "metadata.parquet").read_bytes() == (src[0] / "metadata.parquet").read_bytes()
    assert set().union(*(read_shard(lat / n)[1] for n in manifest["shards"])) == {str(s[0]) for s in SAMPLES}


@pytest.mark.parametrize("case,samples,msg", [
    ("extra row", SAMPLES + [(999, 64, 32, (64, 32), "data/train-0001.tar")], "parquet id without latent: 1"),
    ("missing row", SAMPLES[:-1], "latent without parquet row: 1"),
    ("wrong shape", [(907, 64, 32, (64, 32), TAR0)] + SAMPLES[1:], "wrong shape: 1"),
    ("missing shard", SAMPLES, "missing/unreadable shard: 1")])
def test_write_manifest_refuses_inconsistent_cache(cache, tmp_path, case, samples, msg):
    out = tmp_path / "out"
    shutil.copytree(cache, out)
    (out / "latents" / "manifest.json").unlink()
    if case == "missing shard":
        (out / "latents" / "train-0001.safetensors").unlink()
    write_parquet(tmp_path / "metadata.parquet", samples)
    with pytest.raises(SystemExit, match=msg):
        cl.main(["--tar-dir", str(tmp_path), "--out", str(out), "--write-manifest"])
    assert not (out / "latents" / "manifest.json").exists()


def test_digests_verified_on_rerun(src, cache, tmp_path):
    """A changed shard is refused (manifest kept) unless --no-verify-digests; BucketDataset follows."""
    from safetensors.numpy import save_file
    from krea2_jax.data import BucketDataset
    from krea2_jax.text_template import CaptionTemplate
    d, tmpl = tmp_path / "cache", CaptionTemplate(make_tiny_tokenizer(tmp_path / "tok"), text_rows=16)
    shutil.copytree(cache, d)
    mpath, shard = d / "latents" / "manifest.json", d / "latents" / "train-0000.safetensors"
    man0 = json.loads(mpath.read_text())
    cl.main(["--tar-dir", str(src[0]), "--out", str(d), "--write-manifest"])
    assert json.loads(mpath.read_text()) == man0
    fp0 = BucketDataset(d, tmpl).fingerprint()
    meta, tensors = read_shard(shard)
    tensors["907"] = tensors["907"] + BF16(1.0)
    save_file(tensors, str(shard), metadata=meta)
    with pytest.raises(SystemExit, match="digest verification FAILED: 1 shards differ"):
        cl.main(["--tar-dir", str(src[0]), "--out", str(d), "--write-manifest"])
    assert json.loads(mpath.read_text()) == man0
    with pytest.raises(ValueError, match="train-0000.safetensors"):
        BucketDataset(d, tmpl)
    assert BucketDataset(d, tmpl, verify_digests=False).fingerprint() == fp0
    cl.main(["--tar-dir", str(src[0]), "--out", str(d), "--write-manifest", "--no-verify-digests"])
    new = json.loads(mpath.read_text())["digests"]
    assert new == {**man0["digests"], "train-0000.safetensors": file_sha256(shard)} != man0["digests"]
    assert BucketDataset(d, tmpl).fingerprint() != fp0


@pytest.mark.parametrize("bad", [None, "", GOOD[:-1], GOOD.upper(), GOOD + "0", "g" * 64, 123])
def test_check_digests_rejects_malformed(bad):
    assert cl.check_digests({"a": GOOD}, {"a": GOOD, "b": GOOD}) == (1, ["b"], [])
    for old, new in (({"a": bad}, {"a": GOOD}), ({"a": GOOD}, {"a": bad})):
        with pytest.raises(SystemExit, match="1 malformed digests"):
            cl.check_digests(old, new)
