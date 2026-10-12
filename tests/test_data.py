"""Tests for krea2_jax/data.py (streaming cache loaders), CPU only (JAX is
imported only to compare the host patchify with krea2_jax.model.patchify).

Also provides `make_fake_cache`, a stand-in for scripts/cache_dataset.py
output, `make_fake_latent_cache` (scripts/cache_latents.py's
krea2-latents-v1 layout) and `make_tiny_tokenizer` (re-exported from
tests/test_qwen3_text.py), which other tests import (keep their signatures
stable).
"""

import faulthandler
import gc
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import ml_dtypes
import numpy as np
from safetensors.numpy import save_file

BF16 = ml_dtypes.bfloat16


def make_fake_cache(cache_dir, n, lat_shape=(4, 8, 8), txt_lens=None,
                    txt_layers=3, txt_dim=64, seed=0):
    """Write a cache in scripts/cache_dataset.py's on-disk format.

    One `{i:06d}.safetensors` per sample with bf16 tensors "latent"
    (lat_shape) and "txt" (txt_lens[i], txt_layers, txt_dim), plus meta.json
    (width/height = latent h, w * 8; samples [{file, txt_len, image}]).
    txt_lens defaults to a varied deterministic list in [2, 12]. Returns the
    meta dict.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if txt_lens is None:
        txt_lens = [2 + (7 * i + 3) % 11 for i in range(n)]
    txt_lens = [int(l) for l in txt_lens]
    assert len(txt_lens) == n, (len(txt_lens), n)
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        name = f"{i:06d}.safetensors"
        image = f"img_{i:04d}.png"
        latent = rng.standard_normal(lat_shape).astype(BF16)
        txt = rng.standard_normal(
            (txt_lens[i], txt_layers, txt_dim)).astype(BF16)
        save_file({"latent": latent, "txt": txt}, str(cache_dir / name),
                  metadata={"caption": f"caption {i}", "image": image})
        samples.append({"file": name, "txt_len": txt_lens[i], "image": image})
    height, width = lat_shape[-2] * 8, lat_shape[-1] * 8
    meta = {
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "latent_channels": lat_shape[0],
        "txt_dim": txt_dim,
        "txt_layers": txt_layers,
        "max_length": 512,
        "compute_dtype": "bf16",
        "caption_prefix": "",
        "text_encoder": "fake",
        "vae": "fake",
        "samples": samples,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


LATENT_SHARD_META = "krea2-latents-v1"
# the tiny Qwen-style BPE tokenizer (special tokens <|endoftext|>
# <|im_start|> <|im_end|>) shared with tests/test_qwen3_text.py's tiny text
# encoder: make_tiny_tokenizer(out_dir) writes out_dir/tokenizer.json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen3_text import make_tiny_tokenizer  # noqa: E402,F401


def make_fake_latent_cache(data_dir, rows, lat_channels=4, seed=0,
                           vae="fake-vae"):
    """Write a krea2-latents-v1 cache (scripts/cache_latents.py's layout).

    rows: list of dicts {id, bucket_w, bucket_h, long, medium, short,
    artist_name, tar} (only `id` is required; other S1 columns get
    defaults). metadata.parquet holds all S1 columns with rows in a "rank"
    order that differs from id order; latents/<tar stem>.safetensors hold
    bf16 (lat_channels, bucket_h // 8, bucket_w // 8) tensors keyed by
    str(id) (values from default_rng([seed, id])), plus manifest.json.
    Returns {id: latent}.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    data_dir = Path(data_dir)
    (data_dir / "latents").mkdir(parents=True, exist_ok=True)
    full = []
    for r in rows:
        i = int(r["id"])
        d = {
            "id": i, "long": f"long caption of {i}",
            "medium": f"medium caption {i}", "short": f"short {i}",
            "tags": f"tag{i}, 1girl", "character_name": f"char{i % 5}",
            "series_name": f"series{i % 3}", "artist_name": f"artist{i % 7}",
            "rating": "general", "bucket_w": 32, "bucket_h": 32,
            "score": float(i % 10) / 10.0, "tar": "data/train-0000.tar",
        }
        d.update(r)
        full.append(d)
    n = len(full)
    perm = np.random.default_rng([seed, n]).permutation(n)
    by_id = np.argsort([d["id"] for d in full], kind="stable")
    if n > 1 and np.array_equal(perm, by_id):
        perm = perm[::-1]
    ordered = [full[j] for j in perm]
    for rank, d in enumerate(ordered):
        d["rank"] = rank
        for f in ("long", "medium", "short", "tags"):
            d[f + "_tok"] = len(d[f].split())
    cols = ["id", "long", "medium", "short", "tags", "character_name",
            "series_name", "artist_name", "rating", "bucket_w", "bucket_h",
            "score", "rank", "long_tok", "medium_tok", "short_tok",
            "tags_tok", "tar"]
    types = {"id": pa.int64(), "bucket_w": pa.int64(), "bucket_h": pa.int64(),
             "score": pa.float32(), "rank": pa.int64(),
             "long_tok": pa.int64(), "medium_tok": pa.int64(),
             "short_tok": pa.int64(), "tags_tok": pa.int64()}
    table = pa.table({c: pa.array([d[c] for d in ordered],
                                  type=types.get(c, pa.string()))
                      for c in cols})
    pq.write_table(table, str(data_dir / "metadata.parquet"))

    latents, shards = {}, {}
    for d in ordered:
        i = int(d["id"])
        latents[i] = np.random.default_rng([seed, i]).standard_normal(
            (lat_channels, int(d["bucket_h"]) // 8, int(d["bucket_w"]) // 8)
        ).astype(BF16)
        shards.setdefault(d["tar"], {})[str(i)] = latents[i]
    names = []
    for tar, tensors in sorted(shards.items()):
        name = Path(tar).stem + ".safetensors"
        names.append(name)
        save_file(tensors, str(data_dir / "latents" / name), metadata={
            "format": LATENT_SHARD_META, "vae": vae, "source": tar,
            "count": str(len(tensors))})
    (data_dir / "latents" / "manifest.json").write_text(json.dumps({
        "format": LATENT_SHARD_META, "vae": vae, "n": n,
        "shards": sorted(names)}))
    return latents


def loader_threads():
    return [t for t in threading.enumerate()
            if t.name.startswith("BatchLoader")]


def wait_no_threads(timeout=10.0):
    t0 = time.time()
    while loader_threads() and time.time() - t0 < timeout:
        time.sleep(0.02)
    return not loader_threads()


def take(loader, k):
    out = []
    for b in loader:
        out.append(tuple(a.copy() for a in b))
        if len(out) == k:
            break
    return out


def same_batch(a, b):
    return all(x.dtype == y.dtype and x.shape == y.shape
               and np.array_equal(x.view(np.uint8), y.view(np.uint8))
               for x, y in zip(a, b))


def test_dataset(root):
    from krea2_jax.data import CachedDataset
    lens = [5, 2, 9, 1, 7, 3, 12, 4, 6, 8, 10, 11, 2]
    meta = make_fake_cache(root / "c", 13, txt_lens=lens)
    ds = CachedDataset(root / "c")
    assert ds.n == len(ds) == 13
    assert ds.txt_len == 12, ds.txt_len
    assert ds.lat_hw == (8, 8) and ds.resolution == "64x64", (
        ds.lat_hw, ds.resolution)
    assert ds.meta["latent_channels"] == 4

    from safetensors import safe_open
    for i in range(13):
        lat, txt, mask = ds[i]
        assert lat.shape == (4, 8, 8) and lat.dtype == BF16, (lat.shape,
                                                              lat.dtype)
        assert txt.shape == (12, 3, 64) and txt.dtype == BF16
        assert mask.shape == (12,) and mask.dtype == np.bool_
        assert mask.sum() == lens[i] and mask[:lens[i]].all()
        assert not np.any(txt[lens[i]:].astype(np.float32)), "pad not zero"
        with safe_open(str(root / "c" / meta["samples"][i]["file"]),
                       framework="numpy") as f:
            assert np.array_equal(f.get_tensor("latent"), lat)
            assert np.array_equal(f.get_tensor("txt"), txt[:lens[i]])

    # int txt_len: truncation and padding
    ds5 = CachedDataset(root / "c", txt_len=5)
    lat, txt, mask = ds5[6]  # 12 tokens -> first 5
    assert txt.shape == (5, 3, 64) and mask.all()
    assert np.array_equal(txt, ds[6][1][:5])
    lat, txt, mask = ds5[3]  # 1 token -> padded
    assert mask.tolist() == [True, False, False, False, False]
    ds20 = CachedDataset(root / "c", txt_len="20")
    assert ds20.txt_len == 20 and ds20[0][1].shape == (20, 3, 64)

    # auto falls back to file headers when meta lacks txt_len
    m = json.loads((root / "c" / "meta.json").read_text())
    for s in m["samples"]:
        del s["txt_len"]
    (root / "c2").mkdir()
    for s in m["samples"]:
        (root / "c2" / s["file"]).write_bytes(
            (root / "c" / s["file"]).read_bytes())
    (root / "c2" / "meta.json").write_text(json.dumps(m))
    assert CachedDataset(root / "c2").txt_len == 12
    print("  dataset: shapes/dtypes/padding/txt_len OK")

    # fingerprint: stable across instances, sensitive to sample order, source
    # image names and preprocessing identity (meta.json only)
    fp = ds.fingerprint()
    assert isinstance(fp, str) and len(fp) == 64, fp
    int(fp, 16)
    assert CachedDataset(root / "c").fingerprint() == fp
    assert CachedDataset(root / "c", txt_len=5).fingerprint() == fp

    def variant(name, edit):
        m = json.loads((root / "c" / "meta.json").read_text())
        edit(m)
        (root / name).mkdir()
        (root / name / "meta.json").write_text(json.dumps(m))
        return CachedDataset(root / name).fingerprint()

    def swap(m):
        m["samples"][2], m["samples"][5] = m["samples"][5], m["samples"][2]

    def wider(m):
        m["width"] = m["width"] * 2

    def new_image(m):
        m["samples"][3]["image"] = "other.png"

    fp_swap, fp_wide = variant("fp_swap", swap), variant("fp_wide", wider)
    assert variant("fp_img", new_image) != fp, \
        "changing a sample's source image did not change the fingerprint"
    assert fp_swap != fp, "swapping two samples did not change the fingerprint"
    assert fp_wide != fp, "changing width did not change the fingerprint"
    assert variant("fp_same", lambda m: None) == fp  # re-serialized meta
    print("  dataset: fingerprint stable / order-, image- and width-sensitive OK")
    return ds


def test_order(ds):
    from krea2_jax.data import BatchLoader
    n, bs, seed = 13, 4, 7
    spe = n // bs  # 3 steps/epoch, 1 sample dropped per epoch

    # Reference: train_lora.py's rule with per-epoch seeding.
    def ref_indices(steps):
        out, epoch = [], 0
        perm, pos = np.random.default_rng([seed, 0]).permutation(n), 0
        for _ in range(steps):
            if pos + bs > n:
                epoch += 1
                perm = np.random.default_rng([seed, epoch]).permutation(n)
                pos = 0
            out.append(perm[pos:pos + bs])
            pos += bs
        return out

    steps = 11
    ref = ref_indices(steps)
    with BatchLoader(ds, bs, seed, prefetch=3, workers=4) as ld:
        assert ld.steps_per_epoch == spe
        for s in range(steps):
            assert np.array_equal(ld.indices(s), ref[s]), s
        batches = take(ld, steps)
    for s, (lat, txt, mask) in enumerate(batches):
        assert lat.shape == (bs, 4, 8, 8) and lat.dtype == BF16
        assert txt.shape == (bs, 12, 3, 64) and txt.dtype == BF16
        assert mask.shape == (bs, 12) and mask.dtype == np.bool_
        for j, i in enumerate(ref[s]):
            exp = ds[int(i)]
            assert same_batch((lat[j], txt[j], mask[j]), exp), (s, j, i)

    # Epoch rollover: each epoch's batches are distinct indices covering
    # n - n % bs samples, the tail is dropped, the next epoch reshuffles.
    for e in range(steps // spe):
        idx = np.concatenate(ref[e * spe:(e + 1) * spe])
        assert len(set(idx.tolist())) == spe * bs
        perm = np.random.default_rng([seed, e]).permutation(n)
        assert np.array_equal(idx, perm[:spe * bs])
    assert not np.array_equal(ref[0], ref[spe])

    # Determinism across loaders; different seed differs.
    with BatchLoader(ds, bs, seed, prefetch=1, workers=1) as ld2:
        again = take(ld2, steps)
    assert all(same_batch(a, b) for a, b in zip(batches, again))
    with BatchLoader(ds, bs, seed + 1) as ld3:
        other = take(ld3, spe)
    assert not all(same_batch(a, b) for a, b in zip(batches, other))
    print("  order: shuffle/tail-drop/rollover/determinism OK")

    # Resume: start_step=k yields fresh run's batches k+1, k+2, ...
    for k in [1, 2, 3, 4, 5, 6, 8]:
        with BatchLoader(ds, bs, seed, start_step=k, prefetch=2) as ldk:
            got = take(ldk, 3)
        for t, b in enumerate(got):
            assert same_batch(b, batches[k + t]), (k, t)
    print("  resume: start_step fast-forward OK")

    try:
        BatchLoader(ds, 14, seed)
        raise AssertionError("batch_size > n should raise")
    except ValueError as e:
        assert "exceeds" in str(e)
    BatchLoader(ds, 13, seed).close()  # == n is fine (one batch/epoch)
    print("  batch_size > n raises OK")


class SlowDataset:
    """Wraps a CachedDataset; read_into sleeps and tracks concurrency."""

    def __init__(self, ds, delay=0.05, fail_on=None):
        self.ds, self.delay, self.fail_on = ds, delay, fail_on
        self.n, self.txt_len = ds.n, ds.txt_len
        self.active = self.max_active = self.calls = 0
        self.lock = threading.Lock()

    def sample_spec(self):
        return self.ds.sample_spec()

    def read_into(self, i, *out):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if self.fail_on is not None and i == self.fail_on:
                raise OSError(f"simulated read failure for sample {i}")
            self.ds.read_into(i, *out)
        finally:
            with self.lock:
                self.active -= 1


def test_threads(ds, root):
    from krea2_jax.data import BatchLoader

    # Concurrency + early break + close.
    slow = SlowDataset(ds)
    ld = BatchLoader(slow, 4, 0, prefetch=4, workers=4)
    t0 = time.time()
    got = take(ld, 1)
    assert len(got) == 1
    assert slow.max_active > 1, slow.max_active
    assert loader_threads(), "workers should be alive before close()"
    ld.close()
    assert not loader_threads(), [t.name for t in loader_threads()]
    assert slow.calls < 4 * 4 + 4, slow.calls  # bounded prefetch
    assert time.time() - t0 < 10
    ld.close()  # idempotent
    try:
        iter(ld).__next__()
        raise AssertionError("iterating a closed loader should raise")
    except RuntimeError:
        pass
    print(f"  threads: concurrent loads (max {slow.max_active} active), "
          "early break + close() joins workers OK")

    # close() from another thread while the consumer is iterating.
    slow2 = SlowDataset(ds, delay=0.1)
    ld = BatchLoader(slow2, 4, 0, prefetch=2, workers=2)
    timer = threading.Timer(0.3, ld.close)
    timer.start()
    t0 = time.time()
    count = sum(1 for _ in ld)
    timer.join()
    assert time.time() - t0 < 10 and count < 10, count
    assert wait_no_threads()
    print("  threads: close() from another thread ends iteration OK")

    # Worker exception propagates to the iterating thread.
    bad = SlowDataset(ds, delay=0.01, fail_on=int(
        np.random.default_rng([0, 0]).permutation(13)[5]))
    ld = BatchLoader(bad, 4, 0, prefetch=2, workers=2)
    try:
        take(ld, 10)
        raise AssertionError("exception was swallowed")
    except OSError as e:
        assert "simulated" in str(e)
    ld.close()
    assert not loader_threads()

    # ... including a real corrupt file.
    make_fake_cache(root / "bad", 6, txt_lens=[3] * 6)
    (root / "bad" / "000004.safetensors").write_bytes(b"\x00garbage" * 10)
    from krea2_jax.data import CachedDataset
    with BatchLoader(CachedDataset(root / "bad"), 2, 0) as ld:
        try:
            take(ld, 6)
            raise AssertionError("corrupt file did not raise")
        except AssertionError:
            raise
        except Exception as e:
            print(f"  threads: worker errors propagate OK "
                  f"({type(e).__name__})")
    assert not loader_threads()

    # Dropping the loader without close() (GC) still stops the workers.
    ld = BatchLoader(SlowDataset(ds, delay=0.02), 4, 0, prefetch=3)
    take(ld, 2)
    del ld
    gc.collect()
    assert wait_no_threads(), [t.name for t in loader_threads()]
    print("  threads: GC shuts workers down OK")


# --------------------------------------------------------------------------
# BucketDataset / BucketBatchLoader (krea2-latents-v1 + online captions)
# --------------------------------------------------------------------------

BUCKET_IDS = [907, 13, 450, 2, 77, 1001, 38, 600, 5, 999, 120, 64, 333]
BUCKETS = [(64, 32), (32, 64), (48, 48), (80, 48)]  # (w, h) -> 8, 8, 9, 15 tok
TEXT_ROWS = 32
IMG_TOKENS = 16


def bucket_rows(ids=BUCKET_IDS):
    rows = []
    for k, i in enumerate(ids):
        w, h = BUCKETS[k % len(BUCKETS)]
        rows.append({
            "id": i, "bucket_w": w, "bucket_h": h,
            "long": f"a long caption for image {i} with many words",
            "medium": f"medium caption {i}", "short": f"short {i}",
            "artist_name": "" if i in (2, 77) else f"artist {i % 4}",
            "tar": f"data/train-{k % 2:04d}.tar"})
    return rows


def edit_parquet(src, dst, edit):
    """Copy cache dir src -> dst with metadata.parquet edited by
    edit(DataFrame) -> DataFrame."""
    import shutil
    import pandas as pd
    shutil.copytree(src, dst)
    df = pd.read_parquet(Path(src) / "metadata.parquet")
    edit(df).to_parquet(Path(dst) / "metadata.parquet", index=False)
    return dst


def test_bucket_dataset(root, template):
    import pyarrow.parquet as pq
    from krea2_jax.data import BucketDataset
    rows = bucket_rows()
    lats = make_fake_latent_cache(root / "b", rows)
    rank_ids = pq.read_table(root / "b" / "metadata.parquet").column(
        "id").to_pylist()
    assert rank_ids != sorted(rank_ids), "fake parquet should not be id-sorted"

    ds = BucketDataset(root / "b", template, img_tokens=IMG_TOKENS, patch=2)
    assert ds.n == len(ds) == len(rows)
    assert ds.ids.tolist() == sorted(BUCKET_IDS), ds.ids
    assert ds.ids.dtype == np.int64
    assert ds.lat_channels == 4 and ds.feat == 16 and ds.img_tokens == 16
    assert ds.vae == "fake-vae" and ds.template is template
    by_id = {r["id"]: r for r in rows}
    assert ds.hw.shape == (ds.n, 2) and ds.hw.dtype == np.int32
    for j, i in enumerate(ds.ids.tolist()):
        r = by_id[i]
        assert ds.hw[j].tolist() == [r["bucket_h"] // 16, r["bucket_w"] // 16]
        assert ds.caption(j, 0, False) == r["long"]
        assert ds.caption(j, 1, False) == r["medium"]
        assert ds.caption(j, 2, False) == r["short"]
        if r["artist_name"]:
            assert ds.caption(j, 2, True) == (
                f"{r['short']}, drawn by {r['artist_name']}")
        else:  # empty artist never gets the suffix
            assert ds.caption(j, 0, True) == r["long"]
            assert ds.caption(j, 1, np.True_) == r["medium"]
    exp_buckets = {}
    for r in rows:
        key = (r["bucket_w"], r["bucket_h"])
        exp_buckets[key] = exp_buckets.get(key, 0) + 1
    assert ds.buckets() == exp_buckets, ds.buckets()
    print("  bucket dataset: id order, hw, buckets, caption assembly OK")

    # host patchify == krea2_jax.model.patchify; padding rows zeroed
    import jax.numpy as jnp
    from krea2_jax.model import patchify
    seen = set()
    for j, i in enumerate(ds.ids.tolist()):
        lat = lats[i]
        assert np.array_equal(ds.read_latent(j).view(np.uint16),
                              lat.view(np.uint16))
        out = np.full((IMG_TOKENS, ds.feat), 7.0, dtype=BF16)  # dirty buffer
        ds.read_latent_into(j, out)
        ref = np.asarray(patchify(jnp.asarray(lat[None]), 2))[0]
        n_img = int(ds.hw[j, 0] * ds.hw[j, 1])
        assert ref.shape == (n_img, ds.feat) and ref.dtype == BF16
        assert np.array_equal(out[:n_img].view(np.uint16), ref.view(np.uint16))
        assert not np.any(out[n_img:].astype(np.float32)), "padding not zero"
        seen.add(tuple(lat.shape))
    assert (4, 6, 10) in seen and (4, 4, 8) in seen  # non-square buckets
    print("  bucket dataset: host patchify == model.patchify (non-square), "
          "zero padding OK")
    return ds


def test_bucket_validation(root, template):
    from krea2_jax.data import BucketDataset
    import pandas as pd
    base = root / "val"
    make_fake_latent_cache(base, bucket_rows(BUCKET_IDS[:6]))
    BucketDataset(base, template, img_tokens=IMG_TOKENS)  # valid

    def expect_error(path, *needles, **kw):
        try:
            BucketDataset(path, template, **{"img_tokens": IMG_TOKENS, **kw})
        except ValueError as e:
            msg = str(e)
            for needle in needles:
                assert needle in msg, (needle, msg)
            return msg
        raise AssertionError(f"{path}: expected ValueError {needles}")

    def add_row(df):
        extra = df.iloc[[0]].copy()
        extra["id"] = 123456
        return pd.concat([df, extra], ignore_index=True)

    expect_error(edit_parquet(base, root / "v_missing", add_row),
                 "1 parquet ids have no latent (123456)")
    expect_error(edit_parquet(base, root / "v_extra",
                              lambda df: df[df["id"] != 450]),
                 "1 latents have no parquet row (450)")
    msg = expect_error(
        edit_parquet(base, root / "v_both",
                     lambda df: add_row(df[df["id"] != 450])),
        "1 parquet ids have no latent (123456)",
        "1 latents have no parquet row (450)")
    assert msg.count("\n") == 2, msg  # every problem listed

    def wrong_bucket(df):
        df.loc[df["id"] == 907, "bucket_w"] = 48
        return df
    expect_error(edit_parquet(base, root / "v_shape", wrong_bucket),
                 "1 latents do not match their bucket", "907:(4, 4, 8)!=(4, 4, 6)")

    def dup_row(df):
        return pd.concat([df, df.iloc[[1]]], ignore_index=True)
    expect_error(edit_parquet(base, root / "v_dup", dup_row), "duplicate ids")

    # latent grid not divisible by patch: 40x40 image -> 5x5 latent
    rows = bucket_rows(BUCKET_IDS[:3])
    rows[1].update(bucket_w=40, bucket_h=40)
    make_fake_latent_cache(root / "v_patch", rows)
    expect_error(root / "v_patch", "1 latent grids are not divisible by patch 2")
    # too many tokens for img_tokens (80x48 bucket = 15 tokens)
    expect_error(base, "samples exceed img_tokens 12", img_tokens=12)

    import shutil
    shutil.copytree(base, root / "v_fmt")
    m = json.loads((root / "v_fmt/latents/manifest.json").read_text())
    m["format"] = "krea2-latents-v0"
    (root / "v_fmt/latents/manifest.json").write_text(json.dumps(m))
    expect_error(root / "v_fmt", "krea2-latents-v0")
    m["format"], m["n"] = "krea2-latents-v1", 7
    (root / "v_fmt/latents/manifest.json").write_text(json.dumps(m))
    expect_error(root / "v_fmt", "manifest n=7 but the shards hold 6")
    m["n"], m["shards"] = 6, m["shards"] + ["train-0009.safetensors"]
    (root / "v_fmt/latents/manifest.json").write_text(json.dumps(m))
    expect_error(root / "v_fmt", "1 manifest shards missing")
    for bad in [dict(caption_probs=(0.5, 0.5)), dict(caption_probs=(1, 1, 1)),
                dict(artist_prob=1.5)]:
        expect_error(base, "", **bad)
    print("  bucket validation: missing/extra/duplicate/shape/patch/"
          "img_tokens/manifest errors OK")


def test_epoch_choices(root, template):
    from krea2_jax.data import BucketDataset
    n = 5000
    rows = [{"id": 3 * k + 1, "tar": f"data/train-{k % 3:04d}.tar"}
            for k in range(n)]
    make_fake_latent_cache(root / "ec", rows)
    for probs, ap in [((0.5, 0.3, 0.2), 0.5), ((0.1, 0.6, 0.3), 0.85)]:
        ds = BucketDataset(root / "ec", template, img_tokens=IMG_TOKENS,
                           caption_probs=probs, artist_prob=ap)
        v, a = ds.epoch_choices(3, 0)
        assert v.shape == (n,) and v.dtype == np.int8
        assert a.shape == (n,) and a.dtype == np.bool_
        assert not v.flags.writeable and not a.flags.writeable
        freq = np.bincount(v, minlength=3) / n
        assert np.all(np.abs(freq - np.array(probs)) < 0.03), (freq, probs)
        assert abs(a.mean() - ap) < 0.03, (a.mean(), ap)
        assert ds.epoch_choices(3, 0)[0] is v  # memoized
        v1, a1 = ds.epoch_choices(3, 1)
        assert not np.array_equal(v1, v) and not np.array_equal(a1, a)
        v0, a0 = ds.epoch_choices(3, 0)  # recomputed, identical
        assert v0 is not v and np.array_equal(v0, v) and np.array_equal(a0, a)
        assert not np.array_equal(ds.epoch_choices(4, 0)[0], v)
        ds2 = BucketDataset(root / "ec", template, img_tokens=IMG_TOKENS,
                            caption_probs=probs, artist_prob=ap)
        assert all(np.array_equal(x, y)
                   for x, y in zip(ds2.epoch_choices(3, 0), (v, a)))
        r = np.random.default_rng([3, 0, 1])  # documented draw
        assert np.array_equal(v, r.choice(3, size=n, p=probs))
    print("  epoch_choices: deterministic, memoized, frequencies within "
          "0.03 (n=5000) OK")


def bucket_ref_indices(n, bs, seed, steps):
    spe = n // bs
    return [np.random.default_rng([seed, s // spe]).permutation(n)[
        (s % spe) * bs:(s % spe + 1) * bs] for s in range(steps)]


def test_bucket_loader(ds, root):
    from krea2_jax.data import BucketBatchLoader, BucketDataset
    n, bs, seed, steps = ds.n, 4, 7, 11
    spe = n // bs
    t = ds.template
    ref = bucket_ref_indices(n, bs, seed, steps)
    with BucketBatchLoader(ds, bs, seed, prefetch=3, workers=4) as ld:
        assert ld.steps_per_epoch == spe == 3
        for s in range(steps):
            assert np.array_equal(ld.indices(s), ref[s]), s
        batches = take(ld, steps)
    variants = set()
    for s, (img, hw, ids, txt_len) in enumerate(batches):
        assert img.shape == (bs, IMG_TOKENS, ds.feat) and img.dtype == BF16
        assert hw.shape == (bs, 2) and hw.dtype == np.int32
        assert ids.shape == (bs, t.total_len) and ids.dtype == np.int32
        assert txt_len.shape == (bs,) and txt_len.dtype == np.int32
        epoch = s // spe
        caps = ds.captions(ref[s], seed, epoch)
        exp_ids, exp_len = t.encode_captions(caps)
        assert np.array_equal(ids, exp_ids) and np.array_equal(txt_len,
                                                               exp_len)
        v, _ = ds.epoch_choices(seed, epoch)
        variants.update(v[ref[s]].tolist())
        for j, i in enumerate(ref[s]):
            row = np.empty((IMG_TOKENS, ds.feat), dtype=BF16)
            ds.read_latent_into(int(i), row)
            assert np.array_equal(img[j].view(np.uint16), row.view(np.uint16))
            assert hw[j].tolist() == ds.hw[i].tolist()
    assert len(variants) > 1, variants
    # tail drop / reshuffle per epoch
    for e in range(steps // spe):
        idx = np.concatenate(ref[e * spe:(e + 1) * spe])
        assert len(set(idx.tolist())) == spe * bs
        assert np.array_equal(
            idx, np.random.default_rng([seed, e]).permutation(n)[:spe * bs])
    assert not np.array_equal(ref[0], ref[spe])
    print("  bucket loader: shapes/dtypes, indices() == batches, latents/"
          "captions/ids per sample, tail drop OK")

    with BucketBatchLoader(ds, bs, seed, prefetch=1, workers=1) as ld2:
        again = take(ld2, steps)
    assert all(same_batch(a, b) for a, b in zip(batches, again))
    with BucketBatchLoader(ds, bs, seed + 1) as ld3:
        other = take(ld3, spe)
    assert not all(same_batch(a, b) for a, b in zip(batches, other))
    for k in [1, 2, 3, 4, 5, 6, 8]:
        with BucketBatchLoader(ds, bs, seed, start_step=k, prefetch=2) as ldk:
            got = take(ldk, 3)
        for s, b in enumerate(got):
            assert same_batch(b, batches[k + s]), (k, s)
    # resume in a "new process": fresh dataset instance, epoch 1 first
    fresh = BucketDataset(ds.data_dir, t, img_tokens=IMG_TOKENS)
    with BucketBatchLoader(fresh, bs, seed, start_step=5) as ldf:
        got = take(ldf, 3)
    assert all(same_batch(b, batches[5 + s]) for s, b in enumerate(got))
    try:
        BucketBatchLoader(ds, n + 1, seed)
        raise AssertionError("batch_size > n should raise")
    except ValueError as e:
        assert "exceeds" in str(e)
    print("  bucket loader: determinism, start_step fast-forward (bitwise "
          "incl. ids/txt_len), fresh-instance resume OK")


class SlowBucketDataset:
    """Delegates to a BucketDataset; read_latent_into sleeps."""

    def __init__(self, ds, delay=0.05):
        self.ds, self.delay, self.calls = ds, delay, 0

    def __getattr__(self, name):
        return getattr(self.ds, name)

    def read_latent_into(self, i, out_row):
        self.calls += 1
        time.sleep(self.delay)
        self.ds.read_latent_into(i, out_row)


def test_bucket_threads(ds, root):
    import shutil
    from krea2_jax.data import BucketBatchLoader, BucketDataset
    slow = SlowBucketDataset(ds)
    ld = BucketBatchLoader(slow, 4, 0, prefetch=4, workers=4)
    assert len(take(ld, 1)) == 1
    assert loader_threads(), "workers should be alive before close()"
    ld.close()  # with prefetched batches still in flight
    assert wait_no_threads(), [t.name for t in loader_threads()]
    assert slow.calls < 4 * 4 + 4, slow.calls

    ld = BucketBatchLoader(SlowBucketDataset(ds, 0.1), 4, 0, prefetch=2,
                           workers=2)
    timer = threading.Timer(0.3, ld.close)
    timer.start()
    t0 = time.time()
    count = sum(1 for _ in ld)
    timer.join()
    assert time.time() - t0 < 10 and count < 10, count
    assert wait_no_threads()

    # corrupt shard after construction -> worker error re-raised
    shutil.copytree(ds.data_dir, root / "bad_b")
    bad = BucketDataset(root / "bad_b", ds.template, img_tokens=IMG_TOKENS)
    (root / "bad_b/latents/train-0001.safetensors").write_bytes(
        b"\x00garbage" * 10)
    with BucketBatchLoader(bad, 2, 0) as ld:
        try:
            take(ld, 6)
            raise AssertionError("corrupt shard did not raise")
        except AssertionError:
            raise
        except Exception as e:
            err = type(e).__name__
    assert wait_no_threads()
    print(f"  bucket threads: close() during prefetch / from another thread "
          f"joins workers, worker errors propagate ({err}) OK")


def test_bucket_fingerprint(ds, root):
    from krea2_jax.data import BucketBatchLoader, BucketDataset
    from krea2_jax.text_template import CaptionTemplate
    t = ds.template
    kw = dict(img_tokens=IMG_TOKENS)
    fp = ds.fingerprint()
    assert isinstance(fp, str) and len(fp) == 64
    int(fp, 16)
    with BucketBatchLoader(ds, 4, 1) as ld:
        take(ld, 2)
    with BucketBatchLoader(BucketDataset(ds.data_dir, t, **kw), 3, 9) as ld:
        assert ld.dataset.fingerprint() == fp
    assert ds.fingerprint() == fp
    # same content, different parquet row order -> same samples, same fp
    same = edit_parquet(ds.data_dir, root / "bfp_same",
                        lambda df: df.iloc[::-1].reset_index(drop=True))
    assert BucketDataset(same, t, **kw).fingerprint() == fp

    def medium_edit(df):
        df.loc[df["id"] == 600, "medium"] = "medium caption 600!"
        return df

    def artist_edit(df):
        df.loc[df["id"] == 600, "artist_name"] = "someone else"
        return df

    differ = {
        "caption edit": BucketDataset(edit_parquet(
            ds.data_dir, root / "bfp_cap", medium_edit), t, **kw),
        "artist edit": BucketDataset(edit_parquet(
            ds.data_dir, root / "bfp_art", artist_edit), t, **kw),
        "artist_prob": BucketDataset(ds.data_dir, t, artist_prob=0.4, **kw),
        "caption_probs": BucketDataset(ds.data_dir, t,
                                       caption_probs=(0.4, 0.4, 0.2), **kw),
        "img_tokens": BucketDataset(ds.data_dir, t, img_tokens=32),
        "text_rows": BucketDataset(
            ds.data_dir, CaptionTemplate(t.tokenizer, text_rows=40), **kw),
    }
    rows = bucket_rows()
    a, b = 1, 2  # ids 13 (32x64) and 450 (48x48): swap their buckets
    for key in ("bucket_w", "bucket_h"):
        rows[a][key], rows[b][key] = rows[b][key], rows[a][key]
    make_fake_latent_cache(root / "bfp_bucket", rows)
    differ["swapped bucket"] = BucketDataset(root / "bfp_bucket", t, **kw)
    for what, other in differ.items():
        assert other.fingerprint() != fp, f"{what} did not change it"
    print("  bucket fingerprint: stable across instances/loaders/row order; "
          "changed by caption, artist, bucket, probs, img_tokens, "
          "text_rows OK")


def test_buckets(root):
    from krea2_jax.text_template import CaptionTemplate
    template = CaptionTemplate(make_tiny_tokenizer(root / "tok"),
                               text_rows=TEXT_ROWS)
    ds = test_bucket_dataset(root, template)
    test_bucket_validation(root, template)
    test_epoch_choices(root, template)
    test_bucket_loader(ds, root)
    test_bucket_threads(ds, root)
    test_bucket_fingerprint(ds, root)


def main():
    faulthandler.dump_traceback_later(240, exit=True)  # never hang CI
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = test_dataset(root)
        test_order(ds)
        test_threads(ds, root)
        test_buckets(root)
    assert not loader_threads()
    faulthandler.cancel_dump_traceback_later()
    print(f"ALL DATA TESTS PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
