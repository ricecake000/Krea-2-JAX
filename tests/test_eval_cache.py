"""CPU tests of the generated-image cache of scripts/modal_eval_full.py
(no Modal, network or real weights):

  * cache keys: canonical JSON (dict order / JSON round trip do not matter),
    every key part changes the hash, cfg 0 normalization, model_id /
    entry_rel layout, entry files (write / status / sidecar validation)
  * the fairness rule: check_pairable refuses entries whose sampler part
    differs (SAMPLER_VERSION, steps, cfg_batch, seed, ...), accepts a
    model-only difference; assemble_pair_dir refuses an unfair pair before
    writing anything
  * --models / --pairs parsing, the legacy --base-key/--ft-key/--labels
    mapping, mixing / duplicate / unknown-label errors, local CLI
  * the planner: which (model, prompt) entries are generated, fully cached
    models are not loaded, run_models with per-model prompt subsets
  * decode settings: VAE identity (revision, else file sizes) in the PNG
    name / mirror sidecar / pair-image key / fairness; another VAE ->
    re-decode, latents reused
  * pair-dir assembly in the exact make_blind.py layout (make_blind.py run
    on it when krea2-ops is present), incomplete prompts left out of
    "prompts" (make_blind.py on a partial dir), existing non-empty dirs and
    colliding dir names refused
  * end to end (run_eval with the real prepare_impl / generate_jobs_impl /
    decode_tasks on a tiny DiT + text tower, a temp "volume" and a stub
    VAE): run 1 samples 2 models; run 2 adds a third model and only that
    one is downloaded, loaded and sampled; run 3 is fully cached (no
    generate, no decode); --cache-only (never downloads the parquet); an
    existing --out refused before prepare; a new VAE revision re-decodes
    without generating; --no-cache regenerates

    python tests/test_eval_cache.py      # or: pytest tests/test_eval_cache.py
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import contextlib
import copy
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

import modal_eval_full as mef

MAKE_BLIND = Path("/home/mj/krea2-ops/eval/tools/make_blind.py")
TE = {"repo": mef.TEXT_REPO, "revision": "te-rev-1"}
R2_ID = {"key": "krea2-fullft/run2/model.safetensors", "etag": "e2",
         "size": 25_600, "last_modified": "2026-10-16T01:02:03+00:00"}
HF_ID = {"repo": mef.BASE_REPO, "revision": "c0ffee", "file": mef.BASE_FILE,
         "size": 25_000}
OOD_SPEC = [
    "anime illustration of a girl holding a bouquet of flowers with both hands, close-up on her hands and fingers, soft daylight",
    "three anime characters sitting around a table playing cards, laughing, one of them pointing at another, warm indoor lighting",
    "a detailed anime background of a crowded shopping street at dusk, many shop signs, power lines, people walking, no main character",
    "manga-style illustration of a boy shouting, with a speech bubble that says \"WAIT FOR ME!\"",
    "anime girl standing in front of a cafe with a sign that reads \"OPEN\", smiling, spring afternoon",
    "anime knight in ornate silver armor holding a long spear, full body, intricate costume details, castle courtyard",
    "extreme low-angle view looking up at an anime girl jumping over the camera, blue sky, dynamic perspective",
    "two anime characters in a sword fight mid-motion, sparks flying, dramatic action poses, rooftop at night",
    "anime boy reading a book by a window at night, lit only by a desk lamp, strong rim light and deep shadows",
    "chibi anime cat girl cooking pancakes in a cozy kitchen, cute style, pastel colors",
]


def quiet(*_):
    pass


def prompts_n(n, w=32, h=32):
    return mef.finalize_prompts(
        [{"source": "ood", "row_id": None, "variant": None, "artist": "",
          "caption": f"caption {i}", "width": w, "height": h}
         for i in range(n)], seed=5)


def raises(exc, fn, *texts):
    try:
        fn()
    except exc as e:
        for t in texts:
            assert t in str(e), (t, str(e))
        return e
    raise AssertionError(f"no {exc.__name__}")


def tiny_png(color):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


def test_ood_prompts_and_constants(tmp_path=None):
    assert list(mef.OOD_PROMPTS) == OOD_SPEC, mef.OOD_PROMPTS
    assert mef.OOD_SIZE == (1024, 1024)
    assert mef.SAMPLER_VERSION == "v2-cfgbatch" and mef.CFG_BATCH is True
    assert mef.COMPACT_TEXT is False and mef.TEXT_ROWS == 256
    p = mef.ood_prompts(10)
    assert [x["caption"] for x in p] == OOD_SPEC
    print("    10 anime OOD prompts at 1024x1024, SAMPLER_VERSION v2-cfgbatch")


def test_cache_key(tmp_path):
    settings = mef.sampler_settings(28, 4.5, TE)
    assert settings == {
        "sampler_version": "v2-cfgbatch", "jax": mef.JAX_VERSION,
        "dtype": "bfloat16", "steps": 28, "cfg": 4.5, "negative": "",
        "cfg_batch": True, "attn": "cudnn", "compact_text": False,
        "text_rows": 256, "text_encoder": TE}, settings
    p = prompts_n(2, 1024, 1024)[1]
    assert p["mu"] == mef.resolved_mu(1024, 1024) and p["seed"] == 6
    k = mef.entry_key(settings, p, R2_ID)
    h = mef.entry_hash(k)
    assert len(h) == 32 and int(h, 16) >= 0
    # same inputs (other dict order, JSON round trip, int-valued cfg) -> same
    rev = {kk: settings[kk] for kk in reversed(list(settings))}
    p_rev = {kk: p[kk] for kk in reversed(list(p))}
    id_rev = {kk: R2_ID[kk] for kk in reversed(list(R2_ID))}
    assert mef.entry_hash(mef.entry_key(rev, p_rev, id_rev)) == h
    assert mef.entry_hash(json.loads(json.dumps(k))) == h
    assert mef.canonical_json(k) == mef.canonical_json(json.loads(
        json.dumps(k, indent=3)))
    assert mef.entry_hash(mef.entry_key(
        mef.sampler_settings(28.0, 4.5, TE), p, R2_ID)) == h
    # extra prompt fields (index, source, ...) are not part of the key
    assert mef.entry_hash(mef.entry_key(settings, {**p, "id": "77",
                                                   "index": 77,
                                                   "source": "x"},
                                        R2_ID)) == h

    def with_sampler(**kw):
        return mef.entry_key({**settings, **kw}, p, R2_ID)

    def with_prompt(**kw):
        return mef.entry_key(settings, {**p, **kw}, R2_ID)

    def with_model(ident=R2_ID, **kw):
        return mef.entry_key(settings, p, {**ident, **kw})

    variants = {
        "sampler_version": with_sampler(sampler_version="v3"),
        "jax": with_sampler(jax="0.12.0"),
        "dtype": with_sampler(dtype="float32"),
        "steps": with_sampler(steps=27),
        "cfg": with_sampler(cfg=4.0),
        "negative": with_sampler(negative="blurry"),
        "cfg_batch": with_sampler(cfg_batch=False),
        "attn": with_sampler(attn="ref"),
        "compact_text": with_sampler(compact_text=True),
        "text_rows": with_sampler(text_rows=128),
        "text_encoder": with_sampler(text_encoder={**TE, "revision": "r2"}),
        "caption": with_prompt(caption=p["caption"] + " "),
        "width": with_prompt(width=1216),
        "height": with_prompt(height=832),
        "seed": with_prompt(seed=7),
        "mu": with_prompt(mu=p["mu"] + 1e-9),
        "r2 key": with_model(key="krea2-fullft/run3/model.safetensors"),
        "etag": with_model(etag="e3"),
        "size": with_model(size=25_601),
        "last_modified": with_model(
            last_modified="2026-10-16T01:02:04+00:00"),
        "hf model": with_model(HF_ID),
        "hf revision": with_model(HF_ID, revision="beef"),
        "hf size": with_model(HF_ID, size=1),
    }
    hashes = {name: mef.entry_hash(v) for name, v in variants.items()}
    for name, hv in hashes.items():
        assert hv != h, f"changing {name} kept the hash"
    assert len(set(hashes.values())) == len(hashes), hashes
    # the sampler options of sampler_settings each reach the key
    for kw in ({"attn": "ref"}, {"compact": True}, {"text_rows": 64},
               {"cfg_batch": False}, {"negative": "x"}):
        assert mef.sampler_settings(28, 4.5, TE, **kw) != settings, kw
    # cfg 0: negative / cfg_batch do not matter
    assert (mef.sampler_settings(28, 0, TE, cfg_batch=True, negative="x")
            == mef.sampler_settings(28, 0.0, TE, cfg_batch=False))
    nocfg = mef.sampler_settings(28, 0, TE)
    assert nocfg["negative"] is None and nocfg["cfg_batch"] is False
    # model ids: readable + identity hash; entries under SAMPLER_VERSION
    mid = mef.model_id(R2_ID)
    assert mid.startswith("run2_model-") and len(mid) == len("run2_model-") + 12
    assert mef.model_id({**R2_ID, "etag": "x"}) != mid
    assert mef.model_id(HF_ID).startswith("hf-krea-2-raw-")
    assert mef.model_id({"key": "other/a b/c.safetensors", "etag": "e",
                         "size": 1, "last_modified": None}).startswith(
        "other_a_b_c-")
    rel = mef.entry_rel(k)
    assert rel == f"v2-cfgbatch/{mid}/{h}", rel
    raises(ValueError, lambda: mef.entry_rel(with_sampler(
        sampler_version="../x")), "sampler_version")
    raises(ValueError, lambda: mef.entry_key(settings, p, None), "identity")

    # entry files
    root = tmp_path / "gen"
    assert mef.entry_status(root, k) == {"latent": False, "png": False}
    lat = np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3)
    assert mef.write_entry(root, k, lat, {"seconds": 1.0}) == rel
    assert mef.entry_status(root, k) == {"latent": True, "png": False}
    with np.load(root / (rel + ".npz")) as z:
        assert np.array_equal(z["latent"], lat)
    doc = json.loads((root / (rel + ".json")).read_text())
    assert doc == {"hash": h, "key": k, "meta": {"seconds": 1.0}}, doc
    D = mef.decode_settings({"repo": mef.VAE_REPO, "revision": "v1"})
    D2 = mef.decode_settings({"repo": mef.VAE_REPO, "revision": "v2"})
    assert mef.png_suffix(D) != mef.png_suffix(D2)
    (root / (rel + mef.png_suffix(D))).write_bytes(b"png")
    (root / (rel + mef.png_suffix(D2))).write_bytes(b"png2")
    assert mef.entry_status(root, k, D) == {"latent": True, "png": True}
    assert mef.entry_status(root, k)["png"] is False   # decode unknown
    D3 = mef.decode_settings({"repo": mef.VAE_REPO, "revision": "v3"})
    assert mef.entry_status(root, k, D3)["png"] is False
    assert not list(root.rglob("*.tmp"))
    # a rewrite drops the stale PNGs (of every decode)
    mef.write_entry(root, k, lat + 1, {"seconds": 2.0})
    assert mef.entry_status(root, k, D) == {"latent": True, "png": False}
    assert not list(root.rglob("*.png"))
    # a sidecar whose key does not map to its file name is not an entry
    other = with_sampler(steps=3)
    doc["key"] = other
    (root / (rel + ".json")).write_text(json.dumps(doc))
    assert mef.read_sidecar(root, rel) is None
    assert mef.entry_status(root, k)["latent"] is False
    (root / (rel + ".json")).write_text("{not json")
    assert mef.read_sidecar(root, rel) is None
    mef.write_entry(root, k, lat, {})
    (root / (rel + ".npz")).unlink()
    assert mef.entry_status(root, k)["latent"] is False
    print(f"    cache key: canonical, {len(variants)} key parts each change "
          "the hash, cfg-0 normalization, entry files")


def test_decode_identity(tmp_path):
    vdir = tmp_path / "vae"
    assert mef.vae_identity(vdir) is None and mef.decode_settings(None) is None
    (vdir / "vae").mkdir(parents=True)
    (vdir / "vae" / "w.safetensors").write_bytes(b"x" * 5)
    mef.write_marker(vdir, mef.VAE_REPO, mef.VAE_PATTERNS, revision="abc")
    v = mef.vae_identity(vdir)
    assert v == {"repo": mef.VAE_REPO, "revision": "abc"}, v
    d = mef.decode_settings(v)
    assert d == {"decode_version": mef.DECODE_VERSION, "vae": v,
                 "vae_dtype": "float32", "torch": mef.TORCH_PIN,
                 "diffusers": "diffusers==0.40.0"}, d
    assert mef.png_reusable(d)
    # no recorded revision: a content digest of the files, cached by
    # (path, size, mtime); a same-size replacement gets another digest
    mef.write_marker(vdir, mef.VAE_REPO, mef.VAE_PATTERNS)
    v2 = mef.vae_identity(vdir)
    assert v2["revision"] is None and len(v2["sha256"]) == 64, v2
    assert set(v2) == {"repo", "revision", "sha256"}, v2
    assert (vdir / mef.VAE_DIGEST_CACHE).is_file()
    assert "eval_vae_digest" not in json.dumps(json.loads(
        (vdir / mef.MARKER).read_text())["files"])   # outside the marker
    calls = []
    real_sha = mef.hashlib.sha256
    mef.hashlib.sha256 = lambda *a: calls.append(1) or real_sha(*a)
    try:
        assert mef.vae_identity(vdir) == v2 and not calls   # cached
        st = (vdir / "vae" / "w.safetensors").stat()
        (vdir / "vae" / "w.safetensors").write_bytes(b"y" * 5)   # same size
        os.utime(vdir / "vae" / "w.safetensors",
                 ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        v3 = mef.vae_identity(vdir)
        assert calls, "digest not recomputed after a change"
    finally:
        mef.hashlib.sha256 = real_sha
    assert v3["sha256"] != v2["sha256"], (v2, v3)
    d2, d3 = mef.decode_settings(v2), mef.decode_settings(v3)
    assert mef.png_reusable(d2) and mef.png_suffix(d2) != mef.png_suffix(d3)
    # same-size different-content VAE -> other tag -> decoded again
    from test_modal_eval_full import StubVAE

    root = tmp_path / "gen"
    p = prompts_n(1)[0]
    s = mef.sampler_settings(2, 4.5, TE)
    lat = np.zeros((1, 4, 4, 4), np.float32)
    ka = mef.entry_key(s, p, HF_ID)
    kb = mef.entry_key(s, p, R2_ID)
    ea = {"kind": "entry", "rel": mef.write_entry(root, ka, lat, {}),
          "key": ka}
    eb = {"kind": "entry", "rel": mef.write_entry(root, kb, lat, {}),
          "key": kb}
    loads = []

    def loader():
        loads.append(1)
        return StubVAE(4)

    r2 = mef.decode_tasks([ea], root, loader, d2, current=d2, log=quiet)
    r2b = mef.decode_tasks([ea], root, loader, d2, current=d2, log=quiet)
    r3 = mef.decode_tasks([ea], root, loader, d3, current=d3, log=quiet)
    assert r2[0]["decoded"] and not r2b[0]["decoded"] and r3[0]["decoded"]
    assert len(loads) == 2
    assert mef.entry_status(root, ka, d2)["png"]
    assert not mef.entry_status(root, kb, d3)["png"]
    # a pair of images decoded under the two VAEs is refused
    raises(mef.UnfairPairError, lambda: mef.check_pairable(
        ka, kb, "mixed", d2, d3), "different decode settings")
    raises(RuntimeError, lambda: mef.decode_tasks(
        [ea], root, loader, d2, current=d3, log=quiet), "decode settings")
    # neither revision nor digest: never reused (always decoded again,
    # nothing cached), but the run still works
    d0 = mef.decode_settings({"repo": mef.VAE_REPO, "revision": None,
                              "sha256": None})
    assert not mef.png_reusable(d0) and not mef.png_reusable(None)
    n_png = len(list(root.rglob("*.png")))
    for _ in range(2):
        r0 = mef.decode_tasks([ea, {"kind": "pair", "name": "p", "rel": "x",
                                    "a": ea, "b": eb, "label": "l",
                                    "labels": ["a", "b"]}], root, loader, d0,
                              current=d0, log=quiet)
        assert r0[0]["decoded"] and r0[1]["png"]
    assert len(list(root.rglob("*.png"))) == n_png
    (root / (ea["rel"] + mef.png_suffix(d0))).write_bytes(b"stale")
    assert not mef.entry_status(root, ka, d0)["png"]
    local = tmp_path / "local"
    mef.store_local_entry(local, ea["rel"], b"png", {"hash": "h", "key": ka},
                          d0)
    e = {"rel": ea["rel"], "key": ka}
    assert not mef.local_entry_ok(local, e, d0)          # planning: no reuse
    assert mef.local_entry_ok(local, e, d0, reuse=False)  # assembly: fresh
    print("    VAE identity (revision, else a cached content digest; same-size "
          "replacement re-decodes, mixed pair refused; unknown -> no reuse)")


def test_fairness(tmp_path):
    s = mef.sampler_settings(28, 4.5, TE)
    p = prompts_n(1)[0]
    ka = mef.entry_key(s, p, HF_ID)
    kb = mef.entry_key(s, p, R2_ID)
    mef.check_pairable(ka, kb)   # only the model differs: ok
    mef.check_pairable(ka, copy.deepcopy(ka))
    cases = {
        "sampler_version": mef.entry_key({**s, "sampler_version": "v1"}, p,
                                         R2_ID),
        "steps": mef.entry_key({**s, "steps": 20}, p, R2_ID),
        "cfg_batch": mef.entry_key({**s, "cfg_batch": False}, p, R2_ID),
        "cfg": mef.entry_key({**s, "cfg": 3.0}, p, R2_ID),
        "attn": mef.entry_key({**s, "attn": "ref"}, p, R2_ID),
        "text_rows": mef.entry_key({**s, "text_rows": 128}, p, R2_ID),
        "compact_text": mef.entry_key({**s, "compact_text": True}, p, R2_ID),
        "negative": mef.entry_key({**s, "negative": "bad"}, p, R2_ID),
        "text_encoder.revision": mef.entry_key(
            {**s, "text_encoder": {**TE, "revision": "x"}}, p, R2_ID),
        "prompt.seed": mef.entry_key(s, {**p, "seed": 99}, R2_ID),
        "prompt.caption": mef.entry_key(s, {**p, "caption": "other"}, R2_ID),
        "prompt.width": mef.entry_key(s, {**p, "width": 48}, R2_ID),
        "prompt.mu": mef.entry_key(s, {**p, "mu": 0.1}, R2_ID),
    }
    for field, kx in cases.items():
        e = raises(mef.UnfairPairError, lambda: mef.check_pairable(ka, kx,
                                                                  "t"),
                   field, "only the model may differ")
        assert str(e).startswith("t: "), e
    # several fields at once are all named; a missing field counts
    both = mef.entry_key({**s, "steps": 20, "cfg_batch": False}, p, R2_ID)
    raises(mef.UnfairPairError, lambda: mef.check_pairable(ka, both),
           "steps: 28 vs 20", "cfg_batch: True vs False")
    no_field = copy.deepcopy(kb)
    del no_field["sampler"]["cfg_batch"]
    raises(mef.UnfairPairError, lambda: mef.check_pairable(ka, no_field),
           "cfg_batch: True vs '<absent>'")
    raises(mef.UnfairPairError, lambda: mef.check_pairable(ka, {"x": 1}),
           "not a cache key")
    # both images must be decoded under the same decode settings
    D = mef.decode_settings({"repo": mef.VAE_REPO, "revision": "v1"})
    D2 = mef.decode_settings({"repo": mef.VAE_REPO, "revision": "v2"})
    mef.check_pairable(ka, kb, "t", D, copy.deepcopy(D))
    raises(mef.UnfairPairError, lambda: mef.check_pairable(ka, kb, "t", D,
                                                          D2),
           "different decode settings", "v2")
    assert issubclass(mef.UnfairPairError, ValueError)
    print(f"    fairness: model-only difference ok, {len(cases)} sampler "
          "differences refused")


def test_cli_parsing(tmp_path):
    k1 = "krea2-fullft/run1/exports/model_step1513_params.safetensors"
    k2 = "krea2-fullft/run2/model.safetensors"
    ms = mef.parse_models(f" base=hf-base , run1raw={k1},run2={k2}")
    assert ms == [("base", "hf-base"), ("run1raw", k1), ("run2", k2)], ms
    labels = [m[0] for m in ms]
    assert mef.parse_pairs("run1raw:run2, base:run2", labels) == [
        ("run1raw", "run2"), ("base", "run2")]
    for bad, msg in (("base", "label=KEY"), ("=hf-base", "label=KEY"),
                     ("a=", "label=KEY"), ("a=hf-base,", "label=KEY"),
                     ("a b=hf-base", "bad model label"),
                     ("a/b=hf-base", "bad model label"),
                     ("-a=hf-base", "bad model label"),
                     ("a:b=hf-base", "bad model label"),
                     ("a=hf-base,a=" + k2, "duplicate label"),
                     ("a=" + k2 + ",b=" + k2, "duplicate KEY"),
                     ("a=/abs/key", "bad R2 key"),
                     ("a=x/../y", "bad R2 key")):
        raises(ValueError, lambda: mef.parse_models(bad), msg)
    for bad, msg in (("run2", "need A:B"), ("run1raw:", "need A:B"),
                     ("x:run2", "unknown label 'x'"),
                     ("run2:run2", "against itself"),
                     ("base:run2,base:run2", "given twice"),
                     ("base:run2,", "need A:B")):
        raises(ValueError, lambda: mef.parse_pairs(bad, labels), msg)
    # resolve_models: --models / --pairs
    assert mef.resolve_models(f"a=hf-base,b={k2}") == (
        [("a", "hf-base"), ("b", k2)], [("a", "b")])
    assert mef.resolve_models(f"a=hf-base,b={k2}", "b:a")[1] == [("b", "a")]
    raises(ValueError, lambda: mef.resolve_models(
        f"a=hf-base,b={k2},c={k1}"), "--pairs is required with 3")
    raises(ValueError, lambda: mef.resolve_models("a=hf-base"),
           "--pairs is required with 1")
    for kw in ({"base_key": k1}, {"ft_key": k2}, {"labels": "x,y"}):
        raises(ValueError, lambda: mef.resolve_models(
            f"a=hf-base,b={k2}", **kw), "do not mix")
    raises(ValueError, lambda: mef.resolve_models("", "a:b"),
           "--pairs needs --models")
    # legacy form = 2 models, 1 pair
    assert mef.resolve_models() == ([("base", "hf-base"), ("ft", mef.R2_KEY)],
                                    [("base", "ft")])
    assert mef.resolve_models(base_key=mef.R2_KEY, ft_key=k1,
                              labels="run1-ema,run1-raw") == (
        [("run1-ema", mef.R2_KEY), ("run1-raw", k1)],
        [("run1-ema", "run1-raw")])
    raises(ValueError, lambda: mef.resolve_models(base_key=k1, ft_key=k1),
           "duplicate KEY")
    raises(ValueError, lambda: mef.resolve_models(labels="a b,c"),
           "bad model label")
    raises(ValueError, lambda: mef.resolve_models(labels="x,x"), "--labels")
    # the local CLI validates them too
    from test_modal_eval_full import synthetic_rows, write_parquet

    pq = tmp_path / "metadata.parquet"
    write_parquet(pq, synthetic_rows())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = mef.local_main(["--only-prompts", "--parquet", str(pq),
                              "--n-dataset", "2", "--models",
                              f"base=hf-base,run1raw={k1},run2={k2}",
                              "--pairs", "run1raw:run2,base:run2"])
    out = buf.getvalue()
    assert len(got) == 12 and "36 images from 3 model(s)" in out, out
    assert "pairs: run1raw:run2, base:run2" in out, out
    for argv in (["--models", "a=hf-base,b=hf-base"],
                 ["--models", f"a=hf-base,b={k2}", "--pairs", "a:c"],
                 ["--models", f"a=hf-base,b={k2}", "--ft-key", k1],
                 ["--pairs", "a:b"]):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mef.local_main(["--only-prompts", "--parquet", str(pq)]
                               + argv)
        except SystemExit as e:
            assert e.code, argv
        else:
            raise AssertionError(f"{argv} accepted")
    print("    --models/--pairs parsing + errors, legacy mapping, local CLI")


def fake_info(label, key, ident, prompts, status):
    """A prepare 'models' row; status: {prompt id: (latent, png)}."""
    s = mef.sampler_settings(28, 4.5, TE)
    entries = {}
    for p in prompts:
        k = mef.entry_key(s, p, ident)
        lat, png = status.get(p["id"], (False, False))
        entries[p["id"]] = {"rel": mef.entry_rel(k), "hash": mef.entry_hash(k),
                            "key": k, "latent": lat, "png": png}
    return {"label": label, "key": key, "path": f"/vol/x/{label}",
            "identity": ident, "model_id": mef.model_id(ident),
            "entries": entries}


def test_planner(tmp_path):
    ps = prompts_n(4)
    ids = [p["id"] for p in ps]
    full = {i: (True, True) for i in ids}
    info = [
        fake_info("base", "hf-base", HF_ID, ps, full),
        fake_info("run1", "k/run1", {**R2_ID, "key": "k/run1"}, ps,
                  {ids[0]: (True, False), ids[2]: (True, True)}),
        fake_info("run2", "k/run2", {**R2_ID, "key": "k/run2"}, ps, {}),
    ]
    info.append({**fake_info("new", "k/new", R2_ID, ps, {}),
                 "identity": None, "model_id": None,
                 "entries": {i: None for i in ids}})
    todo = mef.plan_generation(info, ids)
    assert todo == {"run1": [ids[1], ids[3]], "run2": ids, "new": ids}, todo
    assert "base" not in todo   # fully cached: not loaded
    assert mef.plan_generation(info, ids, use_cache=False) == {
        m["label"]: ids for m in info}
    assert mef.plan_generation(info[:1], ids) == {}
    jobs = mef.generate_jobs(info[:3], todo)
    assert [j["label"] for j in jobs] == ["run1", "run2"]
    assert jobs[0] == {"label": "run1", "key": "k/run1",
                       "path": "/vol/x/run1", "identity": info[1]["identity"],
                       "todo": [ids[1], ids[3]]}
    lines = mef.cache_report(info, ids, todo, {("base", ids[0]): True})
    assert "fully cached, not loaded" in lines[0], lines
    assert "4/4 latents cached, 4 PNGs on the volume, 1 in the local" \
        in lines[0], lines
    assert "2/4 latents cached" in lines[1] and "2 to generate" in lines[1]
    assert "identity unknown" in lines[3], lines
    e = mef.estimate(ps, 28, 4.5, "H100", todo)
    assert e["images"] == 10 and e["models"] == 3, e

    # run_models with per-model subsets: an empty subset is never loaded;
    # each model samples exactly its prompts
    loads, samples = [], []
    rows, skipped, notes, _ = mef.run_models(
        mef.generation_order(ps),
        (("run1", "p1", todo["run1"]), ("idle", "p0", []),
         ("run2", "p2", todo["run2"])),
        lambda n, path: loads.append(n) or n,
        lambda params, n, p: samples.append((n, p["id"])) or {},
        lambda params: None, log=quiet)
    assert loads == ["run1", "run2"], loads
    assert samples == [("run1", i) for i in todo["run1"]] + [
        ("run2", i) for i in ids], samples
    assert any("idle: not loaded (no prompt left" in n for n in notes)
    assert not skipped
    # a prompt an earlier model had to sample but skipped (deadline) is
    # dropped by the later models; prompts only the later model needs stay
    clock = [0.0]

    def load(n, path):
        clock[0] += 1
        return n

    def sample(params, n, p):
        clock[0] += 10
        return {}
    rows, skipped, notes, _ = mef.run_models(
        mef.generation_order(ps),
        (("a", "pa", ids[:3]), ("b", "pb", ids)), load, sample,
        lambda params: None, deadline=45, clock=lambda: clock[0],
        load_allowance=1, cold_allowance=10, log=quiet)
    got = [(r["model"], r["id"]) for r in rows]
    assert clock[0] <= 45, clock
    assert got == [("a", ids[0]), ("a", ids[1]), ("b", ids[0]),
                   ("b", ids[1])], got
    assert skipped == [f"{ids[2]}_a", f"{ids[3]}_b"], skipped   # 02 dropped
    print("    planner: cached entries skipped, fully cached models not "
          "loaded, per-model subsets, dropped-after-skip")


D_TEST = mef.decode_settings({"repo": mef.VAE_REPO, "revision": "vae-rev"})


def mirror_entry(local_root, info_row, pid, color, decode=D_TEST):
    e = info_row["entries"][pid]
    mef.store_local_entry(local_root, e["rel"], tiny_png(color),
                          {"hash": e["hash"], "key": e["key"],
                           "meta": {"seconds": 2.5, "device": "cpu"}},
                          decode)


def run_make_blind(d):
    if not MAKE_BLIND.is_file():
        print(f"    (make_blind.py not found at {MAKE_BLIND}: skipped)")
        return None
    r = subprocess.run([sys.executable, str(MAKE_BLIND), str(d), "7"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert (d / "blind" / "index.html").is_file()
    return r.stdout


def test_pair_dir_assembly(tmp_path):
    D = D_TEST
    ps = prompts_n(3)
    ids = [p["id"] for p in ps]
    local = tmp_path / "cache"
    a = fake_info("run1raw", "k/run1", {**R2_ID, "key": "k/run1"}, ps, {})
    b = fake_info("run2", "k/run2", R2_ID, ps, {})
    for pid in ids:
        mirror_entry(local, a, pid, (200, 0, 0))
        mirror_entry(local, b, pid, (0, 0, 200))
    pimg = mef.pair_image_rel(a["entries"][ids[0]], b["entries"][ids[0]],
                              mef.pair_label(ps[0]), ("run1raw", "run2"), D)
    assert pimg != mef.pair_image_rel(
        a["entries"][ids[0]], b["entries"][ids[0]], mef.pair_label(ps[0]),
        ("run1raw", "run2"), mef.decode_settings({"repo": "x",
                                                  "revision": "y"}))
    (local / (pimg + ".png")).parent.mkdir(parents=True, exist_ok=True)
    (local / (pimg + ".png")).write_bytes(tiny_png((9, 9, 9)))
    settings = mef.sampler_settings(28, 4.5, TE)
    d = tmp_path / "run1raw_vs_run2_x"
    mef.assemble_pair_dir(d, ps, ("run1raw", "run2"), [a, b], settings, D,
                          local, run_id="eval_x")
    names = sorted(x.name for x in d.iterdir())
    want = sorted([f"{i}_{s}.png" for i in ids for s in ("base", "ft")]
                  + [f"{ids[0]}_pair.png", "prompts.json", "generate.json",
                     "index.html"])
    assert names == want, names
    assert (d / f"{ids[1]}_base.png").read_bytes() == tiny_png((200, 0, 0))
    assert (d / f"{ids[1]}_ft.png").read_bytes() == tiny_png((0, 0, 200))
    doc = json.loads((d / "prompts.json").read_text())
    assert doc["labels"] == {"base": "run1raw", "ft": "run2"}
    assert doc["pair"] == ["run1raw", "run2"]
    assert doc["weights"] == {"base": a["identity"], "ft": b["identity"]}
    assert doc["settings"] == settings and doc["decode"] == D
    assert doc["incomplete"] == []
    assert doc["entries"][ids[0]] == {"base": a["entries"][ids[0]]["rel"],
                                      "ft": b["entries"][ids[0]]["rel"]}
    assert [p["id"] for p in doc["prompts"]] == ids
    for p in doc["prompts"]:
        assert {"id", "source", "caption"} <= set(p)
        assert p["seconds"] == {"base": 2.5, "ft": 2.5}
    page = (d / "index.html").read_text()
    assert "<th>run1raw</th><th>run2</th>" in page
    assert "Krea 2: run2 vs run1raw" in page and "cfg_batch True" in page
    assert "vae-rev" in page
    out = run_make_blind(d)
    if out is not None:
        assert "3 pairs" in out, out
    # an existing non-empty pair dir is never written over
    raises(ValueError, lambda: mef.assemble_pair_dir(
        d, ps, ("run1raw", "run2"), [a, b], settings, D, local),
        "already exists and is not empty")
    # a side missing in the mirror: the prompt is left out of "prompts"
    # (make_blind.py still works) and listed as incomplete
    (local / (b["entries"][ids[2]]["rel"] + mef.png_suffix(D))).unlink()
    d2 = tmp_path / "partial"
    d2.mkdir()   # an existing EMPTY dir is fine
    mef.assemble_pair_dir(d2, ps, ("run1raw", "run2"), [a, b], settings, D,
                          local)
    assert not (d2 / f"{ids[2]}_ft.png").exists()
    assert not (d2 / f"{ids[2]}_base.png").exists()
    doc2 = json.loads((d2 / "prompts.json").read_text())
    assert [p["id"] for p in doc2["prompts"]] == ids[:2]
    assert doc2["incomplete"] == [{"id": ids[2], "caption": ps[2]["caption"],
                                   "have": ["base"]}], doc2["incomplete"]
    assert ids[2] not in doc2["entries"]
    page2 = (d2 / "index.html").read_text()
    import html as _html
    assert _html.escape(f"incomplete, left out of this dir: {ids[2]} (has "
                        "['base'])") in page2, page2
    out = run_make_blind(d2)
    if out is not None:
        assert "2 pairs" in out, out
    # PNGs decoded under other decode settings (another VAE) are not used
    d4 = tmp_path / "othervae"
    D2 = mef.decode_settings({"repo": mef.VAE_REPO, "revision": "other"})
    mef.assemble_pair_dir(d4, ps, ("run1raw", "run2"), [a, b], settings, D2,
                          local)
    doc4 = json.loads((d4 / "prompts.json").read_text())
    assert doc4["prompts"] == [] and len(doc4["incomplete"]) == 3
    # FAIRNESS: b's entries sampled with other steps -> refused, nothing
    # written
    c = fake_info("run2", "k/run2", R2_ID, ps, {})
    s20 = mef.sampler_settings(20, 4.5, TE)
    for pid, p in zip(ids, ps):
        k = mef.entry_key(s20, p, R2_ID)
        c["entries"][pid] = {"rel": mef.entry_rel(k), "hash": mef.entry_hash(
            k), "key": k, "latent": True, "png": True}
        mirror_entry(local, c, pid, (0, 200, 0))
    d3 = tmp_path / "unfair"
    raises(mef.UnfairPairError, lambda: mef.assemble_pair_dir(
        d3, ps, ("run1raw", "run2"), [a, c], settings, D, local),
        "steps: 28 vs 20", f"prompt {ids[0]}")
    assert not d3.exists()
    # a mirror sidecar whose key is not the planned one: not used
    bad = copy.deepcopy(b)
    bad["entries"][ids[0]]["key"] = c["entries"][ids[0]]["key"]
    assert not mef.local_entry_ok(local, bad["entries"][ids[0]], D)
    assert mef.local_entry_ok(local, b["entries"][ids[0]], D)
    assert not mef.local_entry_ok(local, b["entries"][ids[0]], D2)
    # output dirs: computed up front, duplicates refused
    assert mef.pair_dirs([("a", "b"), ("c", "d")], "", tmp_path, "s") == [
        tmp_path / "a_vs_b_s", tmp_path / "c_vs_d_s"]
    assert mef.pair_dirs([("a", "b")], str(tmp_path / "o"), tmp_path,
                         "s") == [tmp_path / "o"]
    assert mef.pair_dirs([("a", "b"), ("c", "d")], str(tmp_path / "o"),
                         tmp_path, "s")[1] == tmp_path / "o" / "c_vs_d_s"
    raises(ValueError, lambda: mef.pair_dirs(
        [("x_vs_y", "z"), ("x", "y_vs_z")], "", tmp_path, "s"),
        "would both be written to")
    mef.check_dirs_free([tmp_path / "nope", d2.parent / "empty"])
    raises(ValueError, lambda: mef.check_dirs_free([d]), "not empty")
    print("    pair dir: exact make_blind layout, decode settings, existing "
          "dir refused, incomplete prompts left out (make_blind ok), other "
          "VAE not used, unfair pair refused before writing, dir names")


# ---------------------------------------------------------------------------
# end to end on tiny models (real prepare / generate / decode code paths)
# ---------------------------------------------------------------------------


class World:
    """A temp 'volume' + fake HF / R2 for prepare_impl, and the stage
    callables run_eval gets (generate_jobs_impl on a tiny DiT, decode_tasks
    with a stub VAE)."""

    def __init__(self, tmp):
        from test_modal_eval_full import (
            TEXT_OVERRIDES, TINY, StubVAE, make_tiny_dit, synthetic_rows,
            write_parquet,
        )
        from test_qwen3_text import make_tiny_text_encoder_dir

        from krea2_jax.config import Krea2Config

        self.tmp = Path(tmp)
        vol = self.tmp / "vol"
        self.patch = {
            "VOL_DATA": str(vol / "data"), "VOL_TEXT": str(vol / "te"),
            "VOL_BASE": str(vol / "base"), "VOL_FT_ROOT": str(vol / "ft"),
            "VOL_R2": str(vol / "r2"), "VOL_VAE": str(vol / "vae"),
            "OOD_SIZE": (32, 48),
        }
        self.gen_root = vol / "gen"
        self.local = self.tmp / "local-cache"
        self.out_root = self.tmp / "eval"
        Path(self.patch["VOL_DATA"]).mkdir(parents=True)
        write_parquet(Path(self.patch["VOL_DATA"]) / mef.PARQUET,
                      synthetic_rows(50))
        te, _ = make_tiny_text_encoder_dir(Path(self.patch["VOL_TEXT"]))
        mef.write_marker(te, mef.TEXT_REPO, mef.TEXT_PATTERNS,
                         revision="te-sha")
        self.set_vae("vae-sha-1")
        base = Path(self.patch["VOL_BASE"])
        base.mkdir(parents=True)
        make_tiny_dit(base / mef.BASE_FILE, 0)
        mef.write_marker(base, mef.BASE_REPO, [mef.BASE_FILE],
                         revision="base-sha")
        # R2 objects: key -> (tiny checkpoint, etag)
        self.objects = {}
        for i, key in enumerate(("krea2-fullft/run1/model.safetensors",
                                 "krea2-fullft/run2/model.safetensors"), 1):
            src = self.tmp / f"obj{i}.safetensors"
            make_tiny_dit(src, i)
            self.objects[key] = (src, f"etag{i}")
        self.downloads, self.generate_calls, self.decode_calls = [], [], []
        self.loads = []
        self.impl_kw = dict(text_overrides=TEXT_OVERRIDES,
                            model_cfg=Krea2Config(**TINY), log=quiet)
        self.stub_vae = StubVAE

    def set_vae(self, revision):
        vae = Path(self.patch["VOL_VAE"])
        (vae / "vae").mkdir(parents=True, exist_ok=True)
        (vae / "vae" / "config.json").write_text("{}")
        mef.write_marker(vae, mef.VAE_REPO, mef.VAE_PATTERNS,
                         revision=revision)

    def head(self, key):
        src, etag = self.objects[key]
        return {"ContentLength": src.stat().st_size, "ETag": f'"{etag}"',
                "LastModified": "2026-10-16T00:00:00+00:00"}

    @contextlib.contextmanager
    def patched(self):
        saved = {k: getattr(mef, k) for k in list(self.patch) + [
            "hf_fetch", "r2_client", "head_or_none", "r2_fetch"]}
        world = self

        def r2_fetch(client, bucket, key, dest, log=print):
            dest = Path(dest)
            ident = mef.r2_identity(world.head(key), key)
            if mef.read_r2_marker(dest) == ident and dest.is_file():
                return "skipped", ident
            world.downloads.append(key)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(world.objects[key][0], dest)
            mef.r2_marker_path(dest).write_text(json.dumps(ident))
            return "downloaded", ident

        try:
            for k, v in self.patch.items():
                setattr(mef, k, v)
            mef.hf_fetch = lambda *a, **k: "skipped"
            mef.r2_client = lambda env: "client"
            mef.head_or_none = lambda client, bucket, key: (
                world.head(key) if key in world.objects else None)
            mef.r2_fetch = r2_fetch
            yield
        finally:
            for k, v in saved.items():
                setattr(mef, k, v)

    def run(self, models, pairs, stamp, **kw):
        world = self

        def prepare(**pkw):
            return mef.prepare_impl(env={"R2_BUCKET": "bkt"}, log=quiet,
                                    gen_root=world.gen_root, **pkw)

        def generate(prompts, jobs, settings, use_cache):
            world.generate_calls.append([(j["label"], list(j["todo"]))
                                         for j in jobs])
            res = mef.generate_jobs_impl(
                prompts, jobs, settings, world.patch["VOL_TEXT"],
                world.gen_root, use_cache=use_cache, **world.impl_kw)
            world.loads.append(sorted(k[len("load_"):-2]
                                      for k in res["timings"]
                                      if k.startswith("load_")))
            return json.loads(json.dumps(res))   # Modal: plain JSON

        def decode_map(chunks_, decode):
            world.decode_calls.extend(chunks_)
            current = mef.decode_settings(mef.vae_identity())   # patched
            return [mef.decode_tasks(json.loads(json.dumps(c)),
                                     world.gen_root,
                                     lambda: world.stub_vae(4), decode,
                                     current=current, log=quiet)
                    for c in chunks_]
        world.prepare_calls = getattr(world, "prepare_calls", 0)

        def counted_prepare(**pkw):
            world.prepare_calls += 1
            return prepare(**pkw)

        ms, ps = mef.resolve_models(models, pairs)
        with self.patched():
            return mef.run_eval(
                prepare=counted_prepare, generate=generate,
                decode_map=decode_map,
                models=ms, pairs=ps, n_dataset=0, n_ood=2,
                local_root=self.local, out_root=self.out_root, stamp=stamp,
                log=quiet, **{"steps": 2, **kw})


def dir_bytes(d):
    return {x.name: x.read_bytes() for x in sorted(Path(d).iterdir())
            if x.suffix == ".png"}


def test_end_to_end_tiny(tmp_path):
    w = World(tmp_path)
    r1k = "krea2-fullft/run1/model.safetensors"
    r2k = "krea2-fullft/run2/model.safetensors"
    # run 1: base vs run1
    res1 = w.run(f"base=hf-base,run1={r1k}", "", "t1")
    ids = [p["id"] for p in res1["prompts"]]
    assert len(ids) == 2 and res1["prompts"][0]["caption"] == OOD_SPEC[0]
    assert w.generate_calls == [[("base", ids), ("run1", ids)]], \
        w.generate_calls
    assert w.loads == [["base", "run1"]] and w.downloads == [r1k]
    d1 = res1["dirs"][0]
    assert d1 == w.out_root / "base_vs_run1_t1"
    assert len(list(w.gen_root.rglob("*.npz"))) == 4
    assert len(list(w.gen_root.rglob("*.png"))) == 4   # decoded once
    b1 = dir_bytes(d1)
    assert sorted(b1) == sorted(f"{i}_{s}.png" for i in ids
                                for s in ("base", "ft", "pair"))
    out = run_make_blind(d1)
    if out is not None:
        assert "2 pairs" in out, out
    doc1 = json.loads((d1 / "prompts.json").read_text())
    assert doc1["labels"] == {"base": "base", "ft": "run1"}
    assert doc1["settings"]["cfg_batch"] is True
    assert doc1["settings"]["text_encoder"] == {"repo": mef.TEXT_REPO,
                                                "revision": "te-sha"}

    # run 2: + run2, pairs run1:run2 and base:run2 -> only run2 is
    # downloaded, loaded and sampled; base / run1 come from the cache
    w.generate_calls.clear(), w.loads.clear(), w.decode_calls.clear()
    res2 = w.run(f"base=hf-base,run1={r1k},run2={r2k}",
                 "run1:run2,base:run2", "t2")
    assert res2["todo"] == {"run2": ids}, res2["todo"]
    assert w.generate_calls == [[("run2", ids)]], w.generate_calls
    assert w.loads == [["run2"]] and w.downloads == [r1k, r2k], w.downloads
    entry_tasks = [t for c in w.decode_calls for t in c
                   if t["kind"] == "entry"]
    assert len(entry_tasks) == 2 and all(   # only run2's PNGs fetched
        mef.model_id(t["key"]["model"]).startswith("run2_model")
        for t in entry_tasks), entry_tasks
    dA, dB = res2["dirs"]
    assert dA.name == "run1_vs_run2_t2" and dB.name == "base_vs_run2_t2"
    bA, bB = dir_bytes(dA), dir_bytes(dB)
    for i in ids:   # cached images reused byte for byte
        assert bA[f"{i}_base.png"] == b1[f"{i}_ft.png"]
        assert bB[f"{i}_base.png"] == b1[f"{i}_base.png"]
        assert bA[f"{i}_ft.png"] == bB[f"{i}_ft.png"]
        assert bA[f"{i}_ft.png"] != bA[f"{i}_base.png"]
    assert json.loads((dB / "prompts.json").read_text())["labels"] == {
        "base": "base", "ft": "run2"}
    for d in (dA, dB):
        run_make_blind(d)

    # run 3: identical -> fully cached: no generate, no decode, same images
    w.generate_calls.clear(), w.decode_calls.clear()
    res3 = w.run(f"base=hf-base,run1={r1k},run2={r2k}",
                 "run1:run2,base:run2", "t3")
    assert res3["todo"] == {} and not w.generate_calls and not w.decode_calls
    assert w.downloads == [r1k, r2k]
    assert dir_bytes(res3["dirs"][0]) == bA and dir_bytes(res3["dirs"][1]) \
        == bB

    # --cache-only with another seed: nothing generated / written
    w.generate_calls.clear()
    res4 = w.run(f"base=hf-base,run2={r2k}", "", "t4", seed=1,
                 cache_only=True)
    assert res4["todo"] == {"base": ids, "run2": ids} and res4["dirs"] == []
    assert not w.generate_calls and not w.decode_calls
    assert not (w.out_root / "base_vs_run2_t4").exists()

    # other settings (steps) -> other entries; legacy-style 2 models
    res5 = w.run(f"base=hf-base,run1={r1k}", "", "t5", steps=3,
                 cache_only=True)
    assert res5["todo"] == {"base": ids, "run1": ids}

    # an existing non-empty --out is refused before any remote work
    n_prep = w.prepare_calls
    raises(SystemExit, lambda: w.run(f"base=hf-base,run1={r1k}", "", "t8",
                                     out=str(d1)),
           "already exists and is not empty")
    assert w.prepare_calls == n_prep
    # --cache-only never downloads the parquet: none on the volume -> stop
    pq = Path(w.patch["VOL_DATA"]) / mef.PARQUET
    pq.rename(pq.with_suffix(".bak"))
    try:
        raises(SystemExit, lambda: w.run(f"base=hf-base,run1={r1k}", "",
                                         "t9", cache_only=True),
               "--cache-only does not download it")
    finally:
        pq.with_suffix(".bak").rename(pq)

    # another VAE revision: latents reused (no generate), every needed
    # entry decoded again into its own PNG; the old PNGs stay
    n_vol_png = len(list(w.gen_root.rglob("*.png")))
    w.set_vae("vae-sha-2")
    w.generate_calls.clear(), w.decode_calls.clear()
    res7 = w.run(f"base=hf-base,run1={r1k}", "", "t7")
    assert res7["todo"] == {} and not w.generate_calls
    ent7 = [t for c in w.decode_calls for t in c if t["kind"] == "entry"]
    assert len(ent7) == 4, ent7
    assert len(list(w.gen_root.rglob("*.png"))) == n_vol_png + 4
    doc7 = json.loads((res7["dirs"][0] / "prompts.json").read_text())
    assert doc7["decode"]["vae"]["revision"] == "vae-sha-2"
    assert doc1["decode"]["vae"]["revision"] == "vae-sha-1"
    assert [p["id"] for p in doc7["prompts"]] == ids

    # VAE without revision and without a determinable digest: no PNG reuse,
    # every needed entry decoded again, the pair dir still complete
    w.set_vae(None)
    real_digest = mef.vae_content_digest
    mef.vae_content_digest = lambda vae_dir: None
    try:
        for stamp in ("t10", "t11"):
            w.generate_calls.clear(), w.decode_calls.clear()
            r = w.run(f"base=hf-base,run1={r1k}", "", stamp)
            assert not w.generate_calls
            assert sum(t["kind"] == "entry" for c in w.decode_calls
                       for t in c) == 4
            assert sum(t["kind"] == "pair" for c in w.decode_calls
                       for t in c) == 2
            assert sorted(dir_bytes(r["dirs"][0])) == sorted(b1)
    finally:
        mef.vae_content_digest = real_digest
    w.set_vae("vae-sha-2")

    # --no-cache: every entry regenerated and re-decoded
    w.generate_calls.clear(), w.decode_calls.clear()
    res6 = w.run(f"base=hf-base,run1={r1k}", "", "t6", use_cache=False)
    assert w.generate_calls == [[("base", ids), ("run1", ids)]]
    assert sum(t["kind"] == "entry" for c in w.decode_calls for t in c) == 4
    assert sorted(dir_bytes(res6["dirs"][0])) == sorted(b1)
    print("    end to end: run 1 (2 models), run 2 (+1 model: only it "
          "sampled, cached images reused), fully cached run 3, "
          "--cache-only, --no-cache")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
    print(f"all {len(tests)} tests passed")
