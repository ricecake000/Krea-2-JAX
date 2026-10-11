"""Tests for krea2_jax/data.py (streaming cache loader), CPU only, no JAX.

Also provides `make_fake_cache`, a stand-in for scripts/cache_dataset.py
output that other tests import (keep its signature stable).
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


def main():
    faulthandler.dump_traceback_later(240, exit=True)  # never hang CI
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = test_dataset(root)
        test_order(ds)
        test_threads(ds, root)
    assert not loader_threads()
    faulthandler.cancel_dump_traceback_later()
    print(f"ALL DATA TESTS PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
