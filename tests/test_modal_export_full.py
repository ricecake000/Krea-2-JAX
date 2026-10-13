"""CPU tests of scripts/modal_export_full.py without Modal, network or real
weights (a fake S3 client serves a tiny real orbax checkpoint):

  * object selection: only the step's top-level files, <tree>/ and meta/
    (never opt_state/ or the other tree); step 5 does not match step 50; a
    missing tree / unsafe key is refused
  * checkpoint download: the selection restores with
    fullft_ckpt.restore_tree_on_host; a re-run downloads nothing; a replaced
    object (new ETag), a truncated local file and a missing marker entry are
    downloaded again; stale local files are deleted; an object replaced
    during the download or a short download is an error leaving no file;
    paginated listings; download_file ExtraArgs within the s3transfer 0.19.x
    allowlist
  * upload: skipped only for an object with the same size AND user metadata
    (checkpoint fingerprint + sha256); same-size other weights are
    overwritten without --force; ExtraArgs within the s3transfer 0.19.x
    upload allowlist; a size mismatch after the upload is an error
  * export reuse keyed by the checkpoint identity: another --prefix or a
    replaced checkpoint object re-exports (sidecar + drift JSON record it)
  * drift report on synthetic safetensors with known drift (per tensor,
    global, median / p90 / max, by block, by kind, missing / extra / shape
    mismatch / non-finite, bf16 input), tensor kinds of every official key
  * export_impl end to end: fake R2 -> download -> scripts/export_full.py
    (subprocess) -> upload -> drift (params vs EMA vs base), idempotent
    re-run, --force, --drift-only without R2
  * CLI parsing (--compare repeatable, validation) and --help

    python tests/test_modal_export_full.py   # or: pytest tests/test_modal_export_full.py
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

import modal_export_full as mxf

TINY = dict(
    features=128, tdim=32, txtdim=64, heads=4, kvheads=2, multiplier=4,
    layers=2, patch=2, channels=4, txtheads=4, txtkvheads=4, txtlayers=3,
)
# s3transfer 0.19.x constants.py ALLOWED_DOWNLOAD_ARGS (boto3 download_file
# rejects any other ExtraArgs key)
ALLOWED_DOWNLOAD_ARGS = {
    "ChecksumMode", "VersionId", "SSECustomerAlgorithm", "SSECustomerKey",
    "SSECustomerKeyMD5", "RequestPayer", "ExpectedBucketOwner",
}
# s3transfer 0.19.x manager.py TransferManager.ALLOWED_UPLOAD_ARGS
# (_ALLOWED_SHARED_ARGS + ChecksumType, MpuObjectSize + the full-object
# checksum args); boto3 upload_file rejects any other ExtraArgs key
ALLOWED_UPLOAD_ARGS = {
    "ACL", "CacheControl", "ChecksumAlgorithm", "ContentDisposition",
    "ContentEncoding", "ContentLanguage", "ContentType", "ExpectedBucketOwner",
    "Expires", "GrantFullControl", "GrantRead", "GrantReadACP",
    "GrantWriteACP", "Metadata", "ObjectLockLegalHoldStatus",
    "ObjectLockMode", "ObjectLockRetainUntilDate", "RequestPayer",
    "ServerSideEncryption", "StorageClass", "SSECustomerAlgorithm",
    "SSECustomerKey", "SSECustomerKeyMD5", "SSEKMSKeyId",
    "SSEKMSEncryptionContext", "Tagging", "WebsiteRedirectLocation",
    "ChecksumType", "MpuObjectSize", "ChecksumCRC32", "ChecksumCRC32C",
    "ChecksumCRC64NVME", "ChecksumMD5", "ChecksumSHA1", "ChecksumSHA256",
    "ChecksumSHA512", "ChecksumXXHASH3",
}
T1 = datetime(2026, 10, 14, 3, 0, tzinfo=timezone.utc)


class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """In-memory bucket "bkt": key -> bytes, ETag per key; list pages of
    `page` objects. Hooks: after_download(key) runs after each download
    (e.g. to replace the object meanwhile); short = keys downloaded one byte
    short; upload_size_delta corrupts the uploaded size."""

    def __init__(self, page=3):
        self.objects, self.etags, self.page = {}, {}, page
        self.metadata = {}
        self.downloads, self.uploads, self.lists = [], [], 0
        self.after_download = None
        self.short = set()
        self.upload_size_delta = 0

    def put(self, key, data, etag=None, metadata=None):
        self.objects[key] = bytes(data)
        self.etags[key] = etag or f"e{len(self.etags)}-{len(data)}"
        self.metadata[key] = dict(metadata or {})

    def put_dir(self, local, prefix):
        for p in sorted(Path(local).rglob("*")):
            if p.is_file():
                self.put(prefix + p.relative_to(local).as_posix(),
                         p.read_bytes())

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        assert Bucket == "bkt"
        self.lists += 1
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        chunk = keys[start:start + self.page]
        resp = {"Contents": [{"Key": k, "Size": len(self.objects[k]),
                              "ETag": f'"{self.etags[k]}"'} for k in chunk],
                "IsTruncated": start + self.page < len(keys)}
        if resp["IsTruncated"]:
            resp["NextContinuationToken"] = str(start + self.page)
        return resp

    def head_object(self, Bucket, Key):
        assert Bucket == "bkt"
        if Key not in self.objects:
            raise FakeClientError("404")
        return {"ContentLength": len(self.objects[Key]),
                "ETag": f'"{self.etags[Key]}"', "LastModified": T1,
                "Metadata": dict(self.metadata[Key])}

    def download_file(self, bucket, key, filename, ExtraArgs=None,
                      Config=None):
        bad = set(ExtraArgs or {}) - ALLOWED_DOWNLOAD_ARGS
        if bad:
            raise ValueError(f"Invalid extra_args key(s) {sorted(bad)}")
        assert bucket == "bkt" and Config == "cfg"
        assert filename.endswith(".part"), filename
        data = self.objects[key]
        Path(filename).write_bytes(data[:-1] if key in self.short else data)
        self.downloads.append(key)
        if self.after_download:
            self.after_download(key)

    def upload_file(self, filename, bucket, key, ExtraArgs=None, Config=None):
        bad = set(ExtraArgs or {}) - ALLOWED_UPLOAD_ARGS
        if bad:
            raise ValueError(f"Invalid extra_args key(s) {sorted(bad)}")
        assert bucket == "bkt" and Config == "cfg"
        data = Path(filename).read_bytes()
        if self.upload_size_delta:
            data = data[:len(data) + self.upload_size_delta]
        self.put(key, data, etag=f"u{len(self.uploads)}",
                 metadata=(ExtraArgs or {}).get("Metadata"))
        self.uploads.append(key)


def save_train_state(ckpt_dir, step, params, ema, opt_state=None):
    from krea2_jax import fullft_ckpt

    mgr = fullft_ckpt.make_manager(str(ckpt_dir))
    fullft_ckpt.save_state(
        mgr, step, {"params": params, "ema": ema,
                    "opt_state": opt_state if opt_state is not None
                    else {"m": np.zeros(3, np.float32)}, "step": step})
    fullft_ckpt.wait(mgr)
    mgr.close()


def tiny_tree():
    import jax.numpy as jnp

    return {"a": jnp.arange(32, dtype=jnp.float32).reshape(4, 8),
            "b": {"c": jnp.linspace(-1, 1, 6, dtype=jnp.float32)}}


def test_select_objects(tmp_path=None):
    pre = "krea2-fullft/run1/ckpt/5/"
    objs = [{"key": pre + rel, "size": n, "etag": f"x{n}"} for n, rel in
            enumerate(["_CHECKPOINT_METADATA", "commit_success.txt",
                       "params/_METADATA", "params/ocdbt.process_0/d/aa",
                       "ema/_METADATA", "ema/d/bb", "opt_state/_METADATA",
                       "opt_state/ocdbt.process_0/d/cc", "meta/metadata",
                       "meta/commit_success.txt", "params/"])]
    sel = mxf.select_ckpt_objects(objs, pre, "params")
    assert sorted(sel) == ["_CHECKPOINT_METADATA", "commit_success.txt",
                           "meta/commit_success.txt", "meta/metadata",
                           "params/_METADATA",
                           "params/ocdbt.process_0/d/aa"], sorted(sel)
    assert sel["params/_METADATA"] == {"key": pre + "params/_METADATA",
                                       "size": 2, "etag": "x2"}
    assert sorted(mxf.select_ckpt_objects(objs, pre, "ema")) == [
        "_CHECKPOINT_METADATA", "commit_success.txt", "ema/_METADATA",
        "ema/d/bb", "meta/commit_success.txt", "meta/metadata"]
    no_ema = [o for o in objs if "/ema/" not in o["key"]]
    for bad_objs, tree, needle in (
            (no_ema, "ema", "no ema/ objects"),
            ([o for o in objs if "/meta/" not in o["key"]], "params",
             "no meta/ objects"),
            (objs + [{"key": pre + "params/../x", "size": 1, "etag": "z"}],
             "params", "unsafe"),
            (objs, "opt_state", "unknown tree")):
        try:
            mxf.select_ckpt_objects(bad_objs, pre, tree)
        except ValueError as e:
            assert needle in str(e), (needle, e)
        else:
            raise AssertionError(f"{needle}: not refused")
    assert mxf.step_prefix("krea2-fullft", "run1", 5) == pre
    assert mxf.export_key("p", "run2", 7, "ema") == \
        "p/run2/exports/model_step7_ema.safetensors"
    lp = mxf.local_paths("/vol", "run1", 1513, "params")
    assert lp["ckpt_root"] == Path("/vol/ckpt/run1/params")
    assert lp["step_dir"] == Path("/vol/ckpt/run1/params/1513")
    assert lp["export"] == Path("/vol/exports/run1_1513_params.safetensors")
    assert lp["drift"] == Path("/vol/drift/run1_1513.json")
    assert not str(lp["marker"]).startswith(str(lp["ckpt_root"]) + "/")
    print("    selection: top-level + <tree>/ + meta/ only; refusals; layout")


def test_download_and_restore(tmp_path):
    import jax

    from krea2_jax import fullft_ckpt

    params = tiny_tree()
    ema = jax.tree.map(lambda x: 0.5 * x, params)
    src = tmp_path / "src" / "ckpt"
    save_train_state(src, 5, params, ema)
    save_train_state(tmp_path / "src50" / "ckpt", 50, ema, params)
    s3 = FakeS3(page=3)
    s3.put_dir(src / "5", "krea2-fullft/run1/ckpt/5/")
    s3.put_dir(tmp_path / "src50" / "ckpt" / "50", "krea2-fullft/run1/ckpt/50/")
    pre = mxf.step_prefix("krea2-fullft", "run1", 5)
    lp = mxf.local_paths(tmp_path / "vol", "run1", 5, "params")
    log = []

    def dl(tree="params", paths=lp):
        return mxf.download_ckpt_tree(s3, "bkt", pre, paths["step_dir"], tree,
                                      paths["marker"], transfer_config="cfg",
                                      workers=3, log=log.append)

    res = dl()
    got = sorted(k[len(pre):] for k in s3.downloads)
    assert all(k.startswith(pre) for k in s3.downloads), s3.downloads
    assert got and all(r.split("/")[0] in ("params", "meta") or "/" not in r
                       for r in got), got
    assert any(r.startswith("params/") for r in got)
    assert res["downloaded"] == res["objects"] == len(got), res
    assert s3.lists > 1   # paginated
    tree, meta = fullft_ckpt.restore_tree_on_host(str(lp["ckpt_root"]), 5,
                                                  item="params")
    assert meta["step"] == 5 and meta["item"] == "params"
    for x, y in zip(jax.tree.leaves(tree), jax.tree.leaves(params)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    marker = json.loads(lp["marker"].read_text())
    assert marker["complete"] and len(marker["objects"]) == res["objects"]
    # re-run: nothing downloaded
    n = len(s3.downloads)
    assert dl()["downloaded"] == 0 and len(s3.downloads) == n
    # a replaced object (same size, new ETag), a truncated local file and a
    # stale local file
    rel_a, rel_b = got[0], got[-1]
    s3.put(pre + rel_a, s3.objects[pre + rel_a], etag="replaced")
    p_b = lp["step_dir"] / rel_b
    p_b.write_bytes(p_b.read_bytes()[:-1])
    stale = lp["step_dir"] / "params" / "d" / "stale"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old")
    (lp["step_dir"] / "meta" / "x.part").write_bytes(b"p")
    res = dl()
    assert sorted(s3.downloads[n:]) == sorted([pre + rel_a, pre + rel_b]), \
        s3.downloads[n:]
    assert not stale.exists() and not (lp["step_dir"] / "meta" / "x.part").exists()
    assert json.loads(lp["marker"].read_text())["objects"][rel_a]["etag"] == \
        "replaced"
    # a marker entry lost (e.g. crash between rename and marker write)
    m = json.loads(lp["marker"].read_text())
    del m["objects"][rel_a]
    lp["marker"].write_text(json.dumps(m))
    n = len(s3.downloads)
    assert dl()["downloaded"] == 1 and s3.downloads[n:] == [pre + rel_a]
    # EMA into its own orbax dir (the params dir is untouched)
    lpe = mxf.local_paths(tmp_path / "vol", "run1", 5, "ema")
    dl("ema", lpe)
    assert not (lpe["step_dir"] / "params").exists()
    assert not (lp["step_dir"] / "ema").exists()
    tree, _ = fullft_ckpt.restore_tree_on_host(str(lpe["ckpt_root"]), 5,
                                               item="ema")
    for x, y in zip(jax.tree.leaves(tree), jax.tree.leaves(ema)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert not any("/opt_state/" in k for k in s3.downloads), s3.downloads
    assert not any("/ckpt/50/" in k for k in s3.downloads)
    # replaced during the download: error, no file, marker without it
    lpr = mxf.local_paths(tmp_path / "vol2", "run1", 5, "params")

    def replace(key):
        if key.endswith("params/_METADATA"):
            s3.etags[key] = "changed"
    s3.after_download = replace
    try:
        dl(paths=lpr)
    except RuntimeError as e:
        assert "changed during the download" in str(e), e
    else:
        raise AssertionError("concurrent replacement not detected")
    s3.after_download = None
    assert not (lpr["step_dir"] / "params" / "_METADATA").exists()
    assert not list(lpr["step_dir"].rglob("*.part"))
    mr = json.loads(lpr["marker"].read_text())
    assert not mr["complete"] and "params/_METADATA" not in mr["objects"]
    # the next run fetches what is missing (the ETag is now stable)
    s3.etags[pre + "params/_METADATA"] = "stable"
    assert dl(paths=lpr)["downloaded"] >= 1
    assert json.loads(lpr["marker"].read_text())["complete"]
    # a short download: error, no file
    lps = mxf.local_paths(tmp_path / "vol3", "run1", 5, "params")
    s3.short = {pre + "meta/metadata"}
    try:
        dl(paths=lps)
    except RuntimeError as e:
        assert "downloaded" in str(e) and "listing says" in str(e), e
    else:
        raise AssertionError("short download not detected")
    s3.short = set()
    assert not (lps["step_dir"] / "meta" / "metadata").exists()
    print("    download: minimal set restores, idempotent, replaced/truncated/"
          "stale handled, races and short reads refused")


def test_upload(tmp_path):
    s3 = FakeS3()
    f = tmp_path / "x.safetensors"
    f.write_bytes(b"abc" * 100)
    log = []
    meta = {"ckpt-fingerprint": "fp1", "sha256": "s1"}

    def up(path=f, metadata=meta, key="k/x", **kw):
        return mxf.upload_export(s3, "bkt", key, path, metadata,
                                 transfer_config="cfg", log=log.append, **kw)
    st, ident = up()
    assert st == "uploaded" and s3.objects["k/x"] == f.read_bytes()
    assert s3.metadata["k/x"] == meta
    assert ident["size"] == 300 and ident["key"] == "k/x"
    st, _ = up()
    assert st == "skipped" and len(s3.uploads) == 1
    st, _ = up(force=True)
    assert st == "uploaded" and len(s3.uploads) == 2
    # same size, other weights (other fingerprint / sha256): overwritten
    # without --force
    st, _ = up(metadata={"ckpt-fingerprint": "fp2", "sha256": "s1"})
    assert st == "uploaded" and len(s3.uploads) == 3
    st, _ = up(metadata={"ckpt-fingerprint": "fp2", "sha256": "s2"})
    assert st == "uploaded" and s3.metadata["k/x"]["sha256"] == "s2"
    # a same-size object without metadata (e.g. uploaded by hand)
    s3.put("k/x", s3.objects["k/x"])
    assert up(metadata={"ckpt-fingerprint": "fp2", "sha256": "s2"})[0] == \
        "uploaded"
    f.write_bytes(b"abcd" * 100)   # a different size replaces it
    assert up(metadata={"ckpt-fingerprint": "fp2", "sha256": "s2"})[0] == \
        "uploaded"
    s3.upload_size_delta = -1
    try:
        up(key="k/y")
    except RuntimeError as e:
        assert "uploaded object has" in str(e), e
    else:
        raise AssertionError("short upload not detected")

    class Denied(FakeS3):
        def head_object(self, Bucket, Key):
            raise FakeClientError("403")
    try:
        mxf.upload_export(Denied(), "bkt", "k/z", f, meta,
                          transfer_config="cfg")
    except FakeClientError:
        pass
    else:
        raise AssertionError("a 403 on head_object was swallowed")
    try:   # the fake enforces s3transfer's upload allowlist
        s3.upload_file(str(f), "bkt", "k/w", ExtraArgs={"IfMatch": "x"},
                       Config="cfg")
    except ValueError:
        pass
    else:
        raise AssertionError("IfMatch accepted by the fake upload_file")
    print("    upload: skip only on same size + metadata, other weights of "
          "the same size overwritten, --force, size check, head errors")


def test_drift_known(tmp_path):
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    shapes = {
        "blocks.0.attn.wq.weight": (8, 8), "blocks.0.attn.wk.weight": (4, 8),
        "blocks.0.mlp.up.weight": (16, 8), "blocks.0.prenorm.scale": (8,),
        "blocks.1.attn.wo.weight": (8, 8), "blocks.1.mod.lin": (24,),
        "first.weight": (8, 4), "txtfusion.refiner_blocks.0.attn.wv.weight":
        (6, 6), "zero.bias": (3,),
    }
    rel = {k: 0.01 * (i + 1) for i, k in enumerate(shapes)}
    base, ft = {}, {}
    for k, s in shapes.items():
        b = rng.standard_normal(s).astype(np.float32)
        if k == "zero.bias":
            b[:] = 0
        d = rng.standard_normal(s)
        d *= rel[k] * max(np.linalg.norm(b), 1.0) / np.linalg.norm(d)
        base[k], ft[k] = b, (b + d).astype(np.float32)
    # exact float32 deltas for the expected values
    want = {}
    for k in shapes:
        d = ft[k].astype(np.float64) - base[k]
        want[k] = (float((d * d).sum()),
                   float((base[k].astype(np.float64) ** 2).sum()))
    save_file(base, str(tmp_path / "base.safetensors"))
    save_file(ft, str(tmp_path / "ft.safetensors"))
    odd = {k: v for k, v in ft.items() if k != "first.weight"}
    odd["blocks.0.attn.wk.weight"] = np.zeros((8, 4), np.float32)
    odd["extra.weight"] = np.ones(2, np.float32)
    odd["blocks.1.mod.lin"] = np.full(24, np.nan, np.float32)
    save_file(odd, str(tmp_path / "odd.safetensors"))
    import ml_dtypes

    save_file({k: v.astype(ml_dtypes.bfloat16) for k, v in base.items()},
              str(tmp_path / "base_bf16.safetensors"))

    rep = mxf.drift_report(tmp_path / "base.safetensors",
                           [("ft", tmp_path / "ft.safetensors"),
                            ("odd", tmp_path / "odd.safetensors"),
                            ("same", tmp_path / "base.safetensors")],
                           log=lambda *_: None)
    f, o, same = rep["files"]
    assert f["n_tensors"] == len(shapes) and not f["missing"]
    for k, (d2, b2) in want.items():
        t = f["tensors"][k]
        assert abs(t["d2"] - d2) <= 1e-9 * max(d2, 1e-30), (k, t, d2)
        assert abs(t["b2"] - b2) <= 1e-9 * max(b2, 1e-30)
        if b2 > 0:
            assert abs(t["rel"] - np.sqrt(d2 / b2)) < 1e-9
            assert abs(t["rel"] - rel[k]) < 1e-5, (k, t["rel"], rel[k])
    assert f["tensors"]["zero.bias"]["rel"] is None   # ||W_base|| = 0
    nz = [k for k in shapes if k != "zero.bias"]
    g = np.sqrt(sum(want[k][0] for k in shapes) / sum(want[k][1] for k in nz))
    assert abs(f["global"] - g) < 1e-9, (f["global"], g)
    rels = np.array([np.sqrt(want[k][0] / want[k][1]) for k in nz])
    assert abs(f["median"] - np.median(rels)) < 1e-12
    assert abs(f["p90"] - np.percentile(rels, 90)) < 1e-12
    top = "txtfusion.refiner_blocks.0.attn.wv.weight"
    assert f["max_key"] == top and abs(f["max"] - rel[top]) < 1e-5

    def grp(keys):
        return np.sqrt(sum(want[k][0] for k in keys)
                       / sum(want[k][1] for k in keys))
    b0 = [k for k in shapes if k.startswith("blocks.0.")]
    assert abs(f["by_block"]["block00"] - grp(b0)) < 1e-9
    assert sorted(f["by_block"]) == ["block00", "block01", "other",
                                     "txtfusion"], f["by_block"]
    assert abs(f["by_block"]["other"] - grp(["first.weight", "zero.bias"])) \
        < 1e-9
    assert abs(f["by_kind"]["attn_q"] - rel["blocks.0.attn.wq.weight"]) < 1e-5
    assert abs(f["by_kind"]["norm"] - rel["blocks.0.prenorm.scale"]) < 1e-5
    assert abs(f["by_kind"]["modulation"] - rel["blocks.1.mod.lin"]) < 1e-5
    assert abs(f["by_kind"]["attn_v"] - rel[top]) < 1e-5
    assert list(f["by_kind"]) == [k for k in mxf.KINDS if k in f["by_kind"]]
    # missing / extra / shape mismatch / non-finite are reported, left out
    assert o["missing"] == ["first.weight"] and o["extra"] == ["extra.weight"]
    assert o["shape_mismatch"] == [["blocks.0.attn.wk.weight", [8, 4],
                                    [4, 8]]]
    assert o["nonfinite"] == ["blocks.1.mod.lin"]
    assert o["tensors"]["blocks.1.mod.lin"]["rel"] is None
    assert o["n_tensors"] == len(shapes) - 2
    assert abs(o["by_block"]["block01"]
               - rel["blocks.1.attn.wo.weight"]) < 1e-5   # NaN tensor left out
    assert same["global"] == 0.0 and same["max"] == 0.0
    # bf16 files are read too (base vs its bf16 rounding: tiny drift)
    rep2 = mxf.drift_report(tmp_path / "base.safetensors",
                            [("bf16", tmp_path / "base_bf16.safetensors")],
                            log=lambda *_: None)
    assert 0 < rep2["files"][0]["global"] < 4e-3, rep2["files"][0]["global"]
    json.dumps(rep)   # plain JSON (Modal return)
    table = mxf.drift_table(rep)
    for needle in ("[1] ft", "[3] same", "global", "by kind:", "by block:",
                   "block00", "attn_q", "1 missing", "1 nonfinite",
                   "global drift relative to [1]"):
        assert needle in table, (needle, table)
    print("    drift: per tensor / global / median / p90 / max / by block / "
          "by kind exact; odd files reported; bf16")


def test_kinds(tmp_path=None):
    import jax.numpy as jnp

    from krea2_jax.checkpoint import abstract_krea2_params, official_key_shapes
    from krea2_jax.config import KREA2_CONFIG

    keys = official_key_shapes(abstract_krea2_params(KREA2_CONFIG,
                                                     dtype=jnp.bfloat16))
    kinds = {k: mxf.tensor_kind(k) for k in keys}
    assert set(kinds.values()) == set(mxf.KINDS), set(kinds.values())
    for k, want in (
            ("blocks.3.attn.wq.weight", "attn_q"),
            ("blocks.3.attn.wk.weight", "attn_k"),
            ("blocks.3.attn.wv.weight", "attn_v"),
            ("blocks.3.attn.wo.weight", "attn_o"),
            ("blocks.3.attn.gate.weight", "attn_gate"),
            ("blocks.3.attn.qknorm.qnorm.scale", "norm"),
            ("blocks.3.mlp.gate.weight", "mlp"),
            ("blocks.3.prenorm.scale", "norm"),
            ("blocks.3.mod.lin", "modulation"),
            ("last.modulation.lin", "modulation"),
            ("last.norm.scale", "norm"), ("txtmlp.0.scale", "norm"),
            ("txtmlp.1.weight", "embed_final"),
            ("first.weight", "embed_final"),
            ("last.linear.bias", "embed_final"),
            ("tproj.1.weight", "embed_final"),
            ("txtfusion.projector.weight", "embed_final"),
            ("txtfusion.refiner_blocks.1.mlp.up.weight", "mlp")):
        assert kinds[k] == want, (k, kinds[k], want)
    groups = {mxf.tensor_group(k) for k in keys}
    assert groups == {f"block{i:02d}" for i in range(28)} | {"txtfusion",
                                                            "other"}, groups
    print(f"    {len(keys)} official keys -> {len(mxf.KINDS)} kinds, "
          f"{len(groups)} groups")


def test_export_impl(tmp_path):
    import jax
    import jax.numpy as jnp

    from krea2_jax.checkpoint import abstract_krea2_params, save_krea2_params
    from krea2_jax.config import Krea2Config

    abstract = abstract_krea2_params(Krea2Config(**TINY), dtype=jnp.float32)
    rng = np.random.default_rng(0)
    base = jax.tree.map(lambda s: jnp.asarray(
        (0.02 * rng.standard_normal(s.shape)).astype(np.float32)), abstract)
    delta = jax.tree.map(lambda s: jnp.asarray(
        (0.004 * rng.standard_normal(s.shape)).astype(np.float32)), abstract)
    params = jax.tree.map(lambda b, d: b + d, base, delta)
    ema = jax.tree.map(lambda b, d: b + 0.5 * d, base, delta)
    vol = tmp_path / "vol"
    (vol / "base").mkdir(parents=True)
    save_krea2_params(str(vol / mxf.BASE_REL), base, dtype=jnp.bfloat16)
    cmp_path = vol / "ft" / "run1" / "model.safetensors"
    cmp_path.parent.mkdir(parents=True)
    save_krea2_params(str(cmp_path), ema, dtype=jnp.bfloat16)
    save_train_state(tmp_path / "src" / "ckpt", 7, params, ema,
                     jax.tree.map(jnp.zeros_like, params))
    s3 = FakeS3(page=4)
    s3.put_dir(tmp_path / "src" / "ckpt" / "7", "krea2-fullft/run1/ckpt/7/")
    env = {"R2_BUCKET": "bkt"}
    kw = dict(vol=vol, client=s3, transfer_config="cfg", env=env,
              log=lambda *_: None)

    res = mxf.export_impl("run1", 7, "params", [str(cmp_path)], **kw)
    json.dumps(res)
    exp = vol / "exports" / "run1_7_params.safetensors"
    assert res["export"]["status"] == "exported" and exp.is_file(), res
    assert mxf.export_metadata(exp)["step"] == "7"
    assert mxf.export_metadata(exp)["source"] == "params"
    key = "krea2-fullft/run1/exports/model_step7_params.safetensors"
    assert res["upload"]["status"] == "uploaded" and res["upload"]["key"] == key
    assert s3.objects[key] == exp.read_bytes()
    side = json.loads(mxf.export_source_path(exp).read_text())
    import hashlib

    assert side["sha256"] == hashlib.sha256(exp.read_bytes()).hexdigest()
    assert side["size"] == exp.stat().st_size and side["step"] == 7
    assert side["bucket_sha256"] == hashlib.sha256(b"bkt").hexdigest()
    assert "bkt" not in json.dumps(side)
    assert s3.metadata[key] == {"ckpt-fingerprint": side["ckpt_fingerprint"],
                                "sha256": side["sha256"]}
    assert res["export"]["source_identity"] == side
    assert res["drift"]["export_source"] == side
    assert not any("/opt_state/" in k or "/ema/" in k for k in s3.downloads)
    f_params, f_ema = res["drift"]["files"]
    assert f_params["label"] == "exports/run1_7_params.safetensors"
    assert f_ema["label"] == "ft/run1/model.safetensors"
    assert res["drift"]["base"] == mxf.BASE_REL
    want = float(np.sqrt(
        sum(float(np.sum(np.asarray(d, np.float64) ** 2)) for d in
            jax.tree.leaves(delta))
        / sum(float(np.sum(np.asarray(b, np.float64) ** 2)) for b in
              jax.tree.leaves(base))))
    # bf16 rounding of base and export (~0.4% of each weight) adds noise
    assert abs(f_params["global"] - want) < 0.05 * want, (f_params["global"],
                                                          want)
    ratio = f_ema["global"] / f_params["global"]
    assert 0.45 < ratio < 0.56, ratio
    assert json.loads((vol / "drift" / "run1_7.json").read_text())["files"][0][
        "global"] == f_params["global"]
    # re-run: export and upload skipped, no download
    n_dl, n_up = len(s3.downloads), len(s3.uploads)
    res2 = mxf.export_impl("run1", 7, "params", [], **kw)
    assert res2["export"]["status"] == "skipped", res2["export"]
    assert res2["upload"]["status"] == "skipped"
    assert len(s3.downloads) == n_dl and len(s3.uploads) == n_up
    assert [f["label"] for f in res2["drift"]["files"]] == [
        "exports/run1_7_params.safetensors"]
    # the same step under another prefix: other checkpoint identity -> a new
    # download + export (not the cached file), uploaded under that prefix
    for k in [k for k in s3.objects if k.startswith("krea2-fullft/run1/ckpt/")]:
        s3.put("other/" + k[len("krea2-fullft/"):], s3.objects[k])
    n_dl = len(s3.downloads)
    resp = mxf.export_impl("run1", 7, "params", [], prefix="other", **kw)
    assert resp["export"]["status"] == "exported", resp["export"]
    assert len(s3.downloads) > n_dl and all(
        k.startswith("other/") for k in s3.downloads[n_dl:])
    side_o = resp["export"]["source_identity"]
    assert side_o["prefix"] == "other"
    assert side_o["ckpt_fingerprint"] != side["ckpt_fingerprint"]
    assert resp["upload"]["key"] == \
        "other/run1/exports/model_step7_params.safetensors"
    # back to the original prefix: the sidecar (now "other") does not match
    res2b = mxf.export_impl("run1", 7, "params", [], **kw)
    assert res2b["export"]["status"] == "exported"
    # (the local ckpt dir's marker was for "other": everything re-fetched)
    assert res2b["export"]["download"]["downloaded"] == \
        res2b["export"]["download"]["objects"]
    assert res2b["upload"]["status"] == "skipped"   # R2 object already this
    # a replaced checkpoint object at the same step: re-export (only that
    # object downloaded); the same bytes on R2 but a new fingerprint ->
    # re-uploaded without --force
    rk = next(k for k in s3.objects
              if k.startswith("krea2-fullft/run1/ckpt/7/params/"))
    s3.put(rk, s3.objects[rk], etag="replaced")
    n_dl, n_up = len(s3.downloads), len(s3.uploads)
    res2c = mxf.export_impl("run1", 7, "params", [], **kw)
    assert res2c["export"]["status"] == "exported"
    assert s3.downloads[n_dl:] == [rk], s3.downloads[n_dl:]
    assert res2c["upload"]["status"] == "uploaded"
    assert len(s3.uploads) == n_up + 1
    assert res2c["drift"]["export_source"]["ckpt_fingerprint"] not in (
        side["ckpt_fingerprint"], side_o["ckpt_fingerprint"])
    n_up = len(s3.uploads)
    # --force: export again (download complete -> nothing fetched), upload
    res3 = mxf.export_impl("run1", 7, "params", [], force=True, **kw)
    assert res3["export"]["status"] == "exported"
    assert res3["export"]["download"]["downloaded"] == 0
    assert res3["upload"]["status"] == "uploaded" and len(s3.uploads) == n_up + 1
    # --drift-only needs no R2 at all
    res4 = mxf.export_impl("run1", 7, "params", [str(cmp_path)],
                           drift_only=True, vol=vol, env={},
                           log=lambda *_: None)
    assert res4["export"] is None and res4["upload"] is None
    assert len(res4["drift"]["files"]) == 2
    # EMA export of the same step: its own orbax dir and export name
    res5 = mxf.export_impl("run1", 7, "ema", [], **kw)
    exp_e = vol / "exports" / "run1_7_ema.safetensors"
    assert res5["export"]["status"] == "exported" and exp_e.is_file()
    assert abs(res5["drift"]["files"][0]["global"] - f_ema["global"]) < \
        1e-6 * f_ema["global"]   # same bf16 tree as the compared file
    # refusals before any download
    for kwargs, needle in (
            (dict(compares=[str(vol / "nope.safetensors")]), "not on the volume"),
            (dict(vol=tmp_path / "empty"), "missing"),
            (dict(step=8, compares=[], drift_only=True), "nothing to compare")):
        args = {"run": "run1", "step": 7, "source": "params", "compares": [],
                **kw, **kwargs}
        try:
            mxf.export_impl(**args)
        except RuntimeError as e:
            assert needle in str(e), (needle, e)
        else:
            raise AssertionError(f"{needle}: not refused")
    print("    export_impl: fake R2 -> export_full -> upload -> drift (EMA / "
          f"params = {ratio:.3f}), idempotent, --force, --drift-only")


def test_cli(tmp_path=None):
    a = mxf.parse_args(["--step", "1513"])
    assert (a.run, a.step, a.source, a.compare, a.drift_only, a.force,
            a.prefix) == ("run1", 1513, "params", [], False, False,
                          "krea2-fullft")
    a = mxf.parse_args(["--run", "run2", "--step", "5", "--source", "ema",
                        "--compare", "/vol/ft/run1/model.safetensors",
                        "--compare", "/vol/exports/x.safetensors",
                        "--drift-only", "--force"])
    assert a.compare == ["/vol/ft/run1/model.safetensors",
                         "/vol/exports/x.safetensors"] and a.drift_only
    import contextlib
    import io

    for argv in ([], ["--step", "-1"], ["--step", "1", "--run", "../x"],
                 ["--step", "1", "--source", "opt_state"],
                 ["--step", "1", "--compare", "ft/x.safetensors"],
                 ["--step", "1", "--compare", "/vol/../etc/x"],
                 ["--step", "1", "--prefix", "a//b"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                mxf.parse_args(argv)
        except SystemExit as e:
            assert e.code == 2, (argv, e.code)
        else:
            raise AssertionError(f"{argv} accepted")
    try:
        mxf.local_main(["--step", "3"])
    except SystemExit as e:
        assert "modal run" in str(e)
    else:
        raise AssertionError("local_main did not exit")
    p = subprocess.run([sys.executable, str(REPO / "scripts" /
                                            "modal_export_full.py"), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "--compare" in p.stdout, p.stderr
    assert mxf.export_script() == REPO / "scripts" / "export_full.py"
    out = mxf.write_local({"run": "run1", "step": 3, "drift": {}},
                          str(Path(tempfile.gettempdir()) / "mxf_drift.json"))
    assert json.loads(out.read_text())["step"] == 3
    out.unlink()
    print("    CLI: defaults, repeatable --compare, validation, --help")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
    print(f"all {len(tests)} tests passed")
