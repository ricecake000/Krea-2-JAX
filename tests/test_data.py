"""BucketDataset / BucketBatchLoader (krea2_jax/data.py) on a fake multi-bucket
krea2-latents-v1 cache, and CaptionTemplate.encode_captions."""

import json
import shutil
from itertools import islice

import numpy as np
import pandas as pd
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from helpers import (BF16, file_sha256, make_fake_latent_cache,
                     make_tiny_tokenizer, refresh_digests)
from krea2_jax.data import BucketBatchLoader, BucketDataset
from krea2_jax.text_template import PREFIX, CaptionTemplate

IDS = [907, 13, 450, 2, 77, 1001, 38, 600, 5, 999, 120, 64, 333]
BUCKETS = [(64, 32), (32, 64), (48, 48), (80, 48)]  # (w, h) -> 8, 8, 9, 15 tok
IT = 16


def bucket_rows(ids=IDS):
    return [{"id": i, "bucket_w": BUCKETS[k % 4][0], "bucket_h": BUCKETS[k % 4][1],
             "long": f"a long caption for image {i} with many words",
             "medium": f"medium caption {i}", "short": f"short {i}",
             "artist_name": "" if i in (2, 77) else f"artist {i % 4}",
             "tar": f"data/train-{k % 2:04d}.tar"} for k, i in enumerate(ids)]


def edit_parquet(src, dst, edit):
    shutil.copytree(src, dst)
    edit(pd.read_parquet(src / "metadata.parquet")).to_parquet(
        dst / "metadata.parquet", index=False)
    return dst


def take(ds, k, bs=4, seed=7, **kw):
    with BucketBatchLoader(ds, bs, seed, **kw) as ld:
        return list(islice(ld, k))


def same(a, b):
    return all(x.dtype == y.dtype and x.tobytes() == y.tobytes() for x, y in zip(a, b))


@pytest.fixture(scope="module")
def tmpl(tmp_path_factory):
    return CaptionTemplate(make_tiny_tokenizer(tmp_path_factory.mktemp("tok")), text_rows=32)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    d = tmp_path_factory.mktemp("bc") / "b"
    return d, make_fake_latent_cache(d, bucket_rows())


@pytest.fixture(scope="module")
def ds(cache, tmpl):
    return BucketDataset(cache[0], tmpl, img_tokens=IT, patch=2)


@pytest.fixture(scope="module")
def batches(ds):
    return take(ds, 11, prefetch=3, workers=4)


def test_ids_hw_captions(ds):
    by_id = {r["id"]: r for r in bucket_rows()}
    assert ds.ids.tolist() == sorted(IDS) and ds.feat == 16
    for j, i in enumerate(ds.ids.tolist()):
        r = by_id[i]
        assert ds.hw[j].tolist() == [r["bucket_h"] // 16, r["bucket_w"] // 16]
        assert [ds.caption(j, v, False) for v in range(3)] == [r["long"], r["medium"], r["short"]]
        want = f"{r['short']}, drawn by {r['artist_name']}" if r["artist_name"] else r["short"]
        assert ds.caption(j, 2, np.True_) == want
    assert ds.buckets() == {(64, 32): 4, (32, 64): 3, (48, 48): 3, (80, 48): 3}


def test_read_latent_into_matches_patchify(ds, cache):
    import jax.numpy as jnp
    from krea2_jax.model import patchify
    shapes = set()
    for j, i in enumerate(ds.ids.tolist()):
        lat = cache[1][i]
        out = np.full((IT, ds.feat), 7.0, dtype=BF16)
        ds.read_latent_into(j, out)
        ref = np.asarray(patchify(jnp.asarray(lat[None]), 2))[0]
        n = int(ds.hw[j, 0] * ds.hw[j, 1])
        assert ref.shape == (n, ds.feat)
        assert np.array_equal(out[:n].view(np.uint16), ref.view(np.uint16))
        assert not np.any(out[n:].astype(np.float32))
        shapes.add(lat.shape)
    assert {(4, 6, 10), (4, 4, 8)} <= shapes


def _add_row(df):
    return pd.concat([df, df.iloc[[0]].assign(id=123456)], ignore_index=True)


def _set(df, i, **kv):
    df.loc[df["id"] == i, list(kv)] = list(kv.values())
    return df


def _parquet(fn):
    return lambda d: fn(pd.read_parquet(d / "metadata.parquet")).to_parquet(
        d / "metadata.parquet", index=False)


def _manifest(fn):
    def edit(d):
        m = json.loads((d / "latents/manifest.json").read_text())
        fn(m)
        (d / "latents/manifest.json").write_text(json.dumps(m))
    return edit


def _digest(fn):
    return _manifest(lambda m: m["digests"].update(
        {"train-0000.safetensors": fn(m["digests"]["train-0000.safetensors"])}))


def _bad_patch(d):
    rows = bucket_rows(IDS[:6])
    rows[1].update(bucket_w=40, bucket_h=40)
    make_fake_latent_cache(d, rows)


VALIDATION = [
    ("no latent", _parquet(_add_row), {}, ["1 parquet ids have no latent (123456)"]),
    ("no row", _parquet(lambda df: df[df["id"] != 450]), {},
     ["1 latents have no parquet row (450)"]),
    ("bucket shape", _parquet(lambda df: _set(df, 907, bucket_w=48)), {},
     ["1 latents do not match their bucket", "907:(4, 4, 8)!=(4, 4, 6)"]),
    ("duplicate", _parquet(lambda df: pd.concat([df, df.iloc[[1]]])), {}, ["duplicate ids"]),
    ("patch", _bad_patch, {}, ["1 latent grids are not divisible by patch 2"]),
    ("img_tokens", lambda d: None, {"img_tokens": 12}, ["samples exceed img_tokens 12"]),
    ("format", _manifest(lambda m: m.update(format="krea2-latents-v0")), {},
     ["'krea2-latents-v0', expected 'krea2-latents-v1'"]),
    ("n", _manifest(lambda m: m.update(n=7)), {}, ["manifest n=7 but the shards hold 6"]),
    ("missing shard", _manifest(lambda m: m["shards"].append("train-0009.safetensors")), {},
     ["1 manifest shards missing on disk (train-0009.safetensors)",
      "1 manifest shards have no digest (train-0009.safetensors)"]),
    ("no digests", _manifest(lambda m: m.pop("digests")), {},
     ['no shard "digests"', "--write-manifest"]),
    ("null digest", _digest(lambda g: None), {"verify_digests": False},
     ["1 manifest digests are not 64-char lowercase hex", "train-0000.safetensors=None"]),
    ("upper digest", _digest(str.upper), {},
     ["1 manifest digests are not 64-char lowercase hex", "train-0000.safetensors='"]),
    ("short digest", _digest(lambda g: g[:-1]), {"verify_digests": False},
     ["1 manifest digests are not 64-char lowercase hex"]),
]


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("small") / "s"
    make_fake_latent_cache(d, bucket_rows(IDS[:6]))
    assert BucketDataset(d, None, img_tokens=IT).verified_shards == 2
    return d


@pytest.mark.parametrize("edit,kw,needles", [v[1:] for v in VALIDATION],
                         ids=[v[0] for v in VALIDATION])
def test_validation(small, tmpl, tmp_path, edit, kw, needles):
    d = shutil.copytree(small, tmp_path / "c")
    edit(d)
    with pytest.raises(ValueError) as e:
        BucketDataset(d, tmpl, **{"img_tokens": IT, **kw})
    for n in needles:
        assert n in str(e.value), (n, str(e.value))


def test_modified_shard_refused(ds, tmp_path):
    d = shutil.copytree(ds.data_dir, tmp_path / "m")
    shard = d / "latents" / ds.shards[0]
    with safe_open(str(shard), framework="numpy") as f:
        meta = f.metadata()
    t = load_file(str(shard))
    k = sorted(t)[0]
    t[k] = t[k].copy()
    t[k][0, 0, 0] += BF16(1.0)
    save_file(t, str(shard), metadata=meta)
    with pytest.raises(ValueError) as e:
        BucketDataset(d, ds.template, img_tokens=IT)
    msg = str(e.value)
    assert "latent digest verification FAILED" in msg and "1 of 2 shards differ" in msg
    assert f"{ds.shards[0]}: expected {ds.digests[ds.shards[0]]}, actual {file_sha256(shard)}" in msg
    assert ds.shards[1] not in msg
    stale = BucketDataset(d, ds.template, img_tokens=IT, verify_digests=False)
    assert stale.verified_shards == 0 and stale.fingerprint() == ds.fingerprint()
    refresh_digests(d)
    assert BucketDataset(d, ds.template, img_tokens=IT).fingerprint() != ds.fingerprint()


def test_loader_order_and_batches(ds, batches):
    n, bs, seed, spe = ds.n, 4, 7, 3
    t = ds.template
    ld = BucketBatchLoader(ds, bs, seed)
    for s, (img, hw, ids, txt_len) in enumerate(batches):
        e, k = divmod(s, spe)
        idx = np.random.default_rng([seed, e]).permutation(n)[k * bs:(k + 1) * bs]
        assert np.array_equal(ld.indices(s), idx)
        exp_ids, exp_len = t.encode_captions(ds.captions(idx, seed, e))
        assert np.array_equal(ids, exp_ids) and np.array_equal(txt_len, exp_len)
        assert np.array_equal(hw, ds.hw[idx])
        for j, i in enumerate(idx):
            row = np.empty((IT, ds.feat), dtype=BF16)
            ds.read_latent_into(int(i), row)
            assert np.array_equal(img[j].view(np.uint16), row.view(np.uint16))
    ld.close()
    with pytest.raises(ValueError, match="exceeds"):
        BucketBatchLoader(ds, n + 1, seed)


def test_loader_deterministic_and_resume(ds, batches, tmpl):
    again = take(ds, 11, prefetch=1, workers=1)
    assert len(again) == 11 and all(same(a, b) for a, b in zip(batches, again))
    assert not all(same(a, b) for a, b in zip(batches, take(ds, 3, seed=8)))
    fresh = BucketDataset(ds.data_dir, tmpl, img_tokens=IT)
    for d, k in ((ds, 1), (ds, 3), (ds, 5), (ds, 8), (fresh, 4)):
        got = take(d, 3, start_step=k)
        assert len(got) == 3 and all(same(b, batches[k + s]) for s, b in enumerate(got))


def test_fingerprint(ds, tmp_path):
    t, kw, fp = ds.template, dict(img_tokens=IT), ds.fingerprint()
    assert BucketDataset(ds.data_dir, t, **kw).fingerprint() == fp
    rev = edit_parquet(ds.data_dir, tmp_path / "rev", lambda df: df.iloc[::-1])
    assert BucketDataset(rev, t, **kw).fingerprint() == fp
    rows = bucket_rows()
    for key in ("bucket_w", "bucket_h"):
        rows[1][key], rows[2][key] = rows[2][key], rows[1][key]
    make_fake_latent_cache(tmp_path / "swap", rows)
    differ = {
        "caption": BucketDataset(edit_parquet(ds.data_dir, tmp_path / "cap", lambda df: _set(
            df, 600, medium="medium caption 600!")), t, **kw),
        "artist": BucketDataset(edit_parquet(ds.data_dir, tmp_path / "art", lambda df: _set(
            df, 600, artist_name="someone else")), t, **kw),
        "artist_prob": BucketDataset(ds.data_dir, t, artist_prob=0.4, **kw),
        "caption_probs": BucketDataset(ds.data_dir, t, caption_probs=(0.4, 0.4, 0.2), **kw),
        "img_tokens": BucketDataset(ds.data_dir, t, img_tokens=32),
        "text_rows": BucketDataset(ds.data_dir, CaptionTemplate(t.tokenizer, text_rows=40), **kw),
        "swapped bucket": BucketDataset(tmp_path / "swap", t, **kw),
    }
    for what, other in differ.items():
        assert other.fingerprint() != fp, what


def test_epoch_choices(ds, tmpl):
    probs, ap = (0.1, 0.6, 0.3), 0.85
    a = BucketDataset(ds.data_dir, tmpl, img_tokens=IT, caption_probs=probs, artist_prob=ap)
    b = BucketDataset(ds.data_dir, tmpl, img_tokens=IT, caption_probs=probs, artist_prob=ap)
    v, use = a.epoch_choices(3, 0)
    r = np.random.default_rng([3, 0, 1])
    assert np.array_equal(v, r.choice(3, size=a.n, p=probs))
    assert np.array_equal(use, r.random(a.n) < ap)
    a.epoch_choices(3, 1)
    assert all(np.array_equal(x, y) for x, y in zip(a.epoch_choices(3, 0), b.epoch_choices(3, 0)))
    assert not np.array_equal(a.epoch_choices(3, 1)[0], v)


def test_encode_captions_boundary(tmp_path):
    tok = make_tiny_tokenizer(tmp_path, extra_corpus=["a\n\nb"] * 20)
    merge = ["\na girl", "\n\na girl with long silver hair", "\n" + "a girl " * 20]
    normal = ["", "a girl", "two cats, drawn by someone", "a girl with long silver hair " * 10]
    for rows in (32, 8):
        t = CaptionTemplate(tok, text_rows=rows)
        pl, keep = t.prefix_len, t.prefix_len + t.max_caption_tokens
        ids, tl = t.encode_captions(merge + normal)
        for b, c in enumerate(merge + normal):
            full = tok.encode(PREFIX + c).ids
            assert (full[:pl] != t.prefix_ids) == (b < len(merge)), c
            row = full[:keep] + t.suffix_ids
            assert ids[b, :len(row)].tolist() == row and np.all(ids[b, len(row):] == t.pad_id)
            assert tl[b] == len(row) - pl
        old0 = t.prefix_ids + tok.encode(merge[0]).ids + t.suffix_ids
        assert ids[0, :len(old0)].tolist() != old0 and tl[0] == len(old0) - pl - 1
