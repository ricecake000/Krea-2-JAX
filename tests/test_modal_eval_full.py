"""CPU tests of scripts/modal_eval_full.py without Modal, network or real
weights:

  * caption formatting == BucketDataset.caption on a fake latent cache
    (every sample x variant x artist flag, incl. empty / null artists)
  * prompt selection: deterministic per seed, top-3 buckets only, rows with
    an empty caption excluded, 3/4 long + artist then medium/short, bucket
    resolution, ids / seeds, OOD prompts at 1024x1024; --limit; the local
    --only-prompts CLI and --help
  * idempotent downloads: HF marker check, R2 size check / skip (fake client)
  * generate_impl end to end on a tiny random DiT + tiny Qwen3-VL text
    encoder (attn cudnn = dot_product_attention's xla path on CPU):
    latent shapes, same seed for both models (identical checkpoints give
    identical latents), padded == compacted text, deadline skipping
  * decode (stub VAE), pair image, prompts.json / index.html writing

    python tests/test_modal_eval_full.py      # or: pytest tests/test_modal_eval_full.py
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

import modal_eval_full as mef

TINY = dict(
    features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
    layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3,
)
TEXT_OVERRIDES = {"select_layers": [0, 1, 2]}
TEXT_ROWS = 32


def write_parquet(path, rows):
    """metadata.parquet with the columns read_parquet_rows needs (strings may
    be None)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ("id", "bucket_w", "bucket_h", "long", "medium", "short",
            "artist_name")
    types = {"id": pa.int64(), "bucket_w": pa.int64(), "bucket_h": pa.int64()}
    pq.write_table(pa.table({c: pa.array([r[c] for r in rows],
                                         type=types.get(c, pa.string()))
                             for c in cols}), str(path))


def synthetic_rows(n=400, seed=1):
    """Rows over 5 buckets with counts 40% / 25% / 15% / 12% / 8%; ~10% with
    no artist, a few with an empty or null caption."""
    rng = np.random.default_rng(seed)
    buckets = [(832, 1216), (1216, 832), (1344, 768), (1024, 1024),
               (768, 1344)]
    probs = [0.40, 0.25, 0.15, 0.12, 0.08]
    rows = []
    for k in range(n):
        w, h = buckets[rng.choice(len(buckets), p=probs)]
        i = 1000 + 7 * k
        rows.append({
            "id": i, "bucket_w": w, "bucket_h": h,
            "long": f"a long caption of image {i}, detailed",
            "medium": f"medium caption {i}", "short": f"short {i}",
            "artist_name": (None if k % 20 == 3 else
                            "" if k % 10 == 5 else f"artist{k % 13}"),
        })
    rows[5]["long"] = ""
    rows[6]["medium"] = None
    rows[8]["short"] = ""
    return rows


def test_caption_matches_bucket_dataset(tmp_path):
    from test_data import make_fake_latent_cache
    from test_qwen3_text import make_tiny_tokenizer

    from krea2_jax.data import BucketDataset
    from krea2_jax.text_template import CaptionTemplate

    rows = []
    for k, i in enumerate([5, 3, 11, 8, 2, 7]):
        rows.append({"id": i, "bucket_w": 32 if k % 2 else 48,
                     "bucket_h": 32,
                     "artist_name": [f"art{i}", "", None][k % 3],
                     "long": f"long text {i}" if k != 4 else ""})
    make_fake_latent_cache(tmp_path / "cache", rows)
    tok = make_tiny_tokenizer(tmp_path / "tok")
    ds = BucketDataset(tmp_path / "cache", CaptionTemplate(tok, text_rows=32),
                       img_tokens=16)
    pr = mef.read_parquet_rows(tmp_path / "cache" / "metadata.parquet")
    assert pr["id"].tolist() == ds.ids.tolist()
    assert pr["bucket_w"].tolist() == ds.bucket_wh[:, 0].tolist()
    n_cmp = 0
    for i in range(ds.n):
        texts = (pr["long"][i], pr["medium"][i], pr["short"][i])
        for variant in range(3):
            for use_artist in (False, True):
                got = mef.caption_text(texts, pr["artist_name"][i], variant,
                                       use_artist)
                assert got == ds.caption(i, variant, use_artist), (i, variant)
                n_cmp += 1
    assert any(", drawn by art" in ds.caption(i, 0, True) for i in range(ds.n))
    print(f"    caption_text == BucketDataset.caption on {n_cmp} cases")


def test_prompt_selection(tmp_path):
    pq_path = tmp_path / "metadata.parquet"
    rows = synthetic_rows()
    write_parquet(pq_path, rows)
    a = mef.build_prompts(pq_path, 40, 10, seed=0)
    b = mef.build_prompts(pq_path, 40, 10, seed=0)
    c = mef.build_prompts(pq_path, 40, 10, seed=1)
    assert a == b, "selection is not deterministic"
    assert [p["row_id"] for p in a[:40]] != [p["row_id"] for p in c[:40]]
    assert len(a) == 50 and len({p["row_id"] for p in a[:40]}) == 40

    by_id = {r["id"]: r for r in rows}
    top = {(832, 1216), (1216, 832), (1344, 768)}
    assert [wh for wh, _ in mef.top_buckets(
        [r["bucket_w"] for r in rows], [r["bucket_h"] for r in rows])] == [
        (832, 1216), (1216, 832), (1344, 768)]
    for i, p in enumerate(a):
        assert p["index"] == i and p["id"] == f"{i:02d}" and p["seed"] == i
    for r, p in enumerate(a[:40]):
        src = by_id[p["row_id"]]
        assert p["source"] == "dataset"
        assert (p["width"], p["height"]) in top
        assert (p["width"], p["height"]) == (src["bucket_w"], src["bucket_h"])
        assert src["long"] and src["medium"] and src["short"], src
        if r < 30:
            assert p["variant"] == "long"
            want = src["long"] + (f", drawn by {src['artist_name']}"
                                  if src["artist_name"] else "")
            assert p["caption"] == want and p["artist"] == (
                src["artist_name"] or "")
        else:
            assert p["variant"] == ("medium", "short")[(r - 30) % 2]
            assert p["caption"] == src[p["variant"]] and p["artist"] == ""
    assert any(p["artist"] for p in a[:30])
    for p, text in zip(a[40:], mef.OOD_PROMPTS):
        assert p["source"] == "ood" and p["caption"] == text
        assert (p["width"], p["height"]) == (1024, 1024)
    # resolutions are sorted for generation; --limit keeps the table order
    order = mef.generation_order(a)
    assert [(p["height"], p["width"]) for p in order] == sorted(
        (p["height"], p["width"]) for p in a)
    assert mef.apply_limit(a, 3) == a[:3] and mef.apply_limit(a, 0) == a
    only_ood = mef.build_prompts(pq_path, 0, 2, seed=0)
    assert [p["source"] for p in only_ood] == ["ood", "ood"]
    for bad in (lambda: mef.ood_prompts(11),
                lambda: mef.build_prompts(pq_path, 10_000, 0, 0)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("no ValueError")
    table = mef.prompt_table(a)
    assert table.count("\n") == 50 and "1024x1024" in table
    print("    prompt selection deterministic, top-3 buckets, 30 long + 10 "
          "medium/short, 10 OOD")


def test_cli_and_estimate(tmp_path):
    pq_path = tmp_path / "metadata.parquet"
    write_parquet(pq_path, synthetic_rows())
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = mef.local_main(["--only-prompts", "--parquet", str(pq_path),
                              "--limit", "5", "--gpu", "h100"])
    assert len(got) == 5 and "estimate (H100" in buf.getvalue()
    for argv in (["--only-prompts"], ["--steps", "4"],
                 ["--only-prompts", "--parquet", str(pq_path), "--gpu",
                  "A100"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                mef.local_main(argv)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"{argv} did not exit")
    p = subprocess.run([sys.executable, str(REPO / "scripts" /
                                            "modal_eval_full.py"), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "--only-prompts" in p.stdout, p.stderr
    assert mef.check_gpu("b300") == "B300"
    prompts = [{"width": 1024, "height": 1024}] * 10
    e = mef.estimate(prompts, 28, 4.5, "H100")
    sample_s = 10 * 2 * 14.0
    want_s = mef.EST_LOAD_S + mef.EST_COMPILE_S + sample_s
    assert abs(e["minutes"] - want_s / 60) < 1e-9 and e["images"] == 20
    usd_h = 3.95 + mef.GENERATE_CPU * 0.0472 + mef.GENERATE_MEM_GIB * 0.00799
    assert abs(e["usd"] - want_s / 3600 * usd_h) < 1e-9
    assert mef.estimate(prompts, 28, 0, "H100")["minutes"] < e["minutes"]
    print("    local --only-prompts / --help / --gpu validation, estimate")


def test_downloads(tmp_path):
    d = tmp_path / "hf"
    assert not mef.marker_ok(d)
    (d / "vae").mkdir(parents=True)
    (d / "vae" / "w.bin").write_bytes(b"x" * 10)
    (d / ".cache" / "huggingface").mkdir(parents=True)
    (d / ".cache" / "huggingface" / "x.lock").write_bytes(b"")
    files = mef.write_marker(d, "repo", ["vae/*"])
    assert files == {"vae/w.bin": 10}, files
    assert mef.marker_ok(d)
    (d / "vae" / "w.bin").write_bytes(b"x" * 9)   # truncated
    assert not mef.marker_ok(d)
    assert mef.hf_fetch is not None

    payload = bytes(range(256)) * 40

    class FakeS3:
        def __init__(self, data, size=None):
            self.data, self.size, self.downloads = data, size, 0

        def head_object(self, Bucket, Key):
            assert (Bucket, Key) == ("bkt", mef.R2_KEY)
            return {"ContentLength": self.size or len(payload)}

        def download_file(self, bucket, key, filename, Config=None):
            self.downloads += 1
            assert filename.endswith(".part") and Config == "cfg"
            Path(filename).write_bytes(self.data)

    dest = tmp_path / "ft" / "model.safetensors"
    log = []
    s3 = FakeS3(payload)
    assert mef.r2_fetch(s3, "bkt", mef.R2_KEY, dest, log.append,
                        transfer_config="cfg") == "downloaded"
    assert dest.read_bytes() == payload and s3.downloads == 1
    assert mef.r2_fetch(s3, "bkt", mef.R2_KEY, dest, log.append,
                        transfer_config="cfg") == "skipped"
    assert s3.downloads == 1
    short = FakeS3(payload[:-1], size=len(payload) + 5)
    dest2 = tmp_path / "ft2" / "model.safetensors"
    try:
        mef.r2_fetch(short, "bkt", mef.R2_KEY, dest2, log.append,
                     transfer_config="cfg")
    except RuntimeError as e:
        assert "downloaded" in str(e)
    else:
        raise AssertionError("size mismatch not detected")
    assert not dest2.exists()
    try:
        mef.r2_client({"R2_BUCKET": "b"})
    except RuntimeError as e:
        assert "R2_ACCOUNT_ID" in str(e) and "b" not in str(e).split("[")[0]
    else:
        raise AssertionError("missing R2 env not detected")
    assert mef.pick_hf_token({"HUGGINGFACE_TOKEN": "t"}) == (
        "HUGGINGFACE_TOKEN", "t")
    print("    HF marker check, R2 size check + skip + mismatch")


# ---------------------------------------------------------------------------
# generate_impl on tiny random weights
# ---------------------------------------------------------------------------


def make_tiny_dit(path, seed):
    import jax.numpy as jnp

    from krea2_jax.checkpoint import abstract_krea2_params, save_krea2_params
    from krea2_jax.config import Krea2Config

    import jax

    abstract = abstract_krea2_params(Krea2Config(**TINY), dtype=jnp.float32)
    rng = np.random.default_rng(seed)
    params = jax.tree.map(
        lambda s: jnp.asarray(
            (0.02 * rng.standard_normal(s.shape)).astype(np.float32)),
        abstract)
    save_krea2_params(str(path), params, dtype=jnp.bfloat16)
    return path


def tiny_prompts():
    return mef.finalize_prompts([
        {"source": "dataset", "row_id": 1, "variant": "long", "artist": "a",
         "caption": "a long caption, drawn by a", "width": 48, "height": 32},
        {"source": "ood", "row_id": None, "variant": None, "artist": "",
         "caption": "a red sports car", "width": 32, "height": 32},
        {"source": "ood", "row_id": None, "variant": None, "artist": "",
         "caption": "a watercolor painting of a cat", "width": 32,
         "height": 48},
    ], seed=3)


def test_generate_tiny(tmp_path):
    from test_qwen3_text import make_tiny_text_encoder_dir

    from krea2_jax.config import Krea2Config

    te_dir, _ = make_tiny_text_encoder_dir(tmp_path / "te")
    a = make_tiny_dit(tmp_path / "a.safetensors", 0)
    b = make_tiny_dit(tmp_path / "b.safetensors", 1)
    prompts = tiny_prompts()
    kw = dict(steps=2, guidance=4.5, attn="cudnn", text_rows=TEXT_ROWS,
              text_overrides=TEXT_OVERRIDES, model_cfg=Krea2Config(**TINY),
              log=lambda *_: None)

    out = tmp_path / "lat"
    res = mef.generate_impl(prompts, (("base", a), ("ft", b)), te_dir, out,
                            **kw)
    json.dumps(res)   # plain JSON (Modal return)
    assert res["text"]["states_shape"] == [1, TEXT_ROWS, 3, 64], res["text"]
    assert len(res["rows"]) == 6 and not res["skipped"], res
    assert [r["model"] for r in res["rows"]] == ["base"] * 3 + ["ft"] * 3
    assert [r["id"] for r in res["rows"][:3]] == [
        p["id"] for p in mef.generation_order(prompts)]
    assert sum(r["first_of_shape"] for r in res["rows"]) == 3
    for p in prompts:
        for name in mef.MODELS:
            lat = np.load(out / f"{p['id']}_{name}.npy")
            assert lat.shape == (1, 4, p["height"] // 8, p["width"] // 8)
            assert lat.dtype == np.float32 and np.isfinite(lat).all()
        assert not np.array_equal(np.load(out / f"{p['id']}_base.npy"),
                                  np.load(out / f"{p['id']}_ft.npy"))

    # same checkpoint for both -> identical latents (same seed per prompt);
    # padded text == compacted text (exact up to summation order)
    out2 = tmp_path / "lat2"
    mef.generate_impl(prompts, (("base", a), ("ft", a)), te_dir, out2,
                      compact=True, **kw)
    for p in prompts:
        x = np.load(out2 / f"{p['id']}_base.npy")
        assert np.array_equal(x, np.load(out2 / f"{p['id']}_ft.npy"))
        y = np.load(out / f"{p['id']}_base.npy")
        err = float(np.abs(x - y).max() / np.abs(y).max())
        assert err < 5e-2, (p["id"], err)

    # a deadline in the past: nothing sampled, every sample reported skipped
    res3 = mef.generate_impl(prompts, (("base", a), ("ft", b)), te_dir,
                             tmp_path / "lat3", deadline=time.time() - 1, **kw)
    assert res3["rows"] == [] and len(res3["skipped"]) == 3, res3
    assert res3["notes"] and "skipped" in res3["notes"][0]
    print("    generate_impl: tiny DiT x2 sequential, shapes, same seed, "
          "padded == compact, deadline")


# ---------------------------------------------------------------------------
# decode + outputs
# ---------------------------------------------------------------------------


class StubVAE:
    """latents_mean/std + decode() returning a (b, 3, 1, 8h, 8w) sample."""

    def __init__(self, c):
        self.config = SimpleNamespace(latents_mean=[0.5] * c,
                                      latents_std=[2.0] * c)

    def decode(self, z):
        import torch

        x = z[:, :3].repeat_interleave(8, -1).repeat_interleave(8, -2)
        return SimpleNamespace(sample=torch.tanh(x))


def test_decode_and_outputs(tmp_path):
    from PIL import Image

    prompts = tiny_prompts()
    lat_dir = tmp_path / "lat"
    lat_dir.mkdir()
    rng = np.random.default_rng(0)
    for p, models in zip(prompts, (("base", "ft"), ("base",), ())):
        for m in models:
            np.save(lat_dir / f"{p['id']}_{m}.npy", rng.standard_normal(
                (1, 4, p["height"] // 8, p["width"] // 8)).astype(np.float32))
    dec = mef.decode_items(prompts, lat_dir, StubVAE(4), log=lambda *_: None)
    assert [d["missing"] for d in dec] == [[], ["ft"], ["base", "ft"]]
    assert sorted(dec[0]["png"]) == ["base", "ft", "pair"]
    assert sorted(dec[1]["png"]) == ["base", "pair"] and dec[2]["png"] == {}
    import io

    img = Image.open(io.BytesIO(dec[0]["png"]["base"]))
    assert img.size == (prompts[0]["width"], prompts[0]["height"])
    # decode math (sample.py): x * std + mean, tanh stub, [-1, 1] -> uint8
    z = np.load(lat_dir / f"{prompts[0]['id']}_base.npy")[0, :3] * 2.0 + 0.5
    want = ((np.clip(np.tanh(z), -1, 1) * 0.5 + 0.5) * 255.0).astype(np.uint8)
    got = np.asarray(img)[::8, ::8].transpose(2, 0, 1)
    assert np.abs(got.astype(int) - want.astype(int)).max() <= 1
    pair = Image.open(io.BytesIO(dec[0]["png"]["pair"]))
    assert pair.size[0] == 2 * prompts[0]["width"]
    assert pair.size[1] > prompts[0]["height"]
    one = mef.make_pair(np.zeros((32, 16, 3), np.uint8), None, "x")
    assert one.size[0] == 32

    gen = {"rows": [{"id": prompts[0]["id"], "model": "base",
                     "seconds": 1.234}, {"id": prompts[0]["id"],
                                         "model": "ft", "seconds": 2.0}]}
    prompts[0]["caption"] = 'a <b>"quoted"</b> & caption'
    out = tmp_path / "out"
    paths = mef.write_outputs(out, prompts, list(reversed(dec)), gen,
                              title="t", info_lines=["steps 2"])
    names = sorted(p.name for p in paths)
    i0, i1 = prompts[0]["id"], prompts[1]["id"]
    assert names == sorted([f"{i0}_base.png", f"{i0}_ft.png",
                            f"{i0}_pair.png", f"{i1}_base.png",
                            f"{i1}_pair.png", "prompts.json",
                            "generate.json", "index.html"]), names
    pj = json.loads((out / "prompts.json").read_text())
    assert pj[0]["seconds"] == {"base": 1.23, "ft": 2.0}
    assert pj[1]["files"] == {"base": True, "ft": False, "pair": True}
    for k in ("id", "source", "caption", "width", "height", "seed"):
        assert k in pj[0], k
    page = (out / "index.html").read_text()
    assert f'src="{i0}_base.png"' in page and f'src="{i0}_ft.png"' in page
    assert "&lt;b&gt;&quot;quoted&quot;&lt;/b&gt; &amp; caption" in page
    assert "ft: not generated" in page and "steps 2" in page
    assert "/home/" not in page and str(tmp_path) not in page
    print("    decode (stub VAE) math, pair image, prompts.json, index.html")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
    print(f"all {len(tests)} tests passed")
