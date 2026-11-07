"""Paid-run, credential and fairness guards of the Modal scripts, imported as
plain modules (no Modal SDK, no network)."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

import delta_compare as dc
import modal_cache_latents as mcl
import modal_eval_full as mef
import modal_export_full as mxf

OWNER = mcl.OUT_OWNER


def test_hf_token_pick_and_setup():
    assert mcl.pick_hf_token({"HUGGINGFACE_TOKEN": "c", "HUGGING_FACE_HUB_TOKEN": "b"}) == (
        "HUGGING_FACE_HUB_TOKEN", "b")
    assert mcl.pick_hf_token({"HF_TOKEN": "a", "HUGGING_FACE_HUB_TOKEN": "b"}) == ("HF_TOKEN", "a")
    env = {"HUGGINGFACE_TOKEN": "c"}
    assert mcl.setup_hf_token(env) == ("HUGGINGFACE_TOKEN", "c") and env["HF_TOKEN"] == "c"
    assert mcl.pick_hf_token({"HF_TOKEN": ""}) == (None, None)


def scope(etype, name, *perms):
    return [{"entity": {"type": etype, "name": name}, "permissions": ["repo.content.read", *perms]}]


@pytest.mark.parametrize("role,name,scoped,want", [
    ("write", OWNER, None, True), ("read", OWNER, None, False), ("write", "other", None, False),
    ("fineGrained", OWNER, scope("user", OWNER, "repo.write"), True),
    ("fineGrained", OWNER, scope("user", OWNER), False),
    ("fineGrained", OWNER, scope("dataset", f"{OWNER}/x", "repo.write"), None),
    ("mystery", OWNER, None, None), (None, None, None, None)])
def test_token_can_write(role, name, scoped, want):
    tok = {"role": role} if scoped is None else {"role": role, "fineGrained": {"scoped": scoped, "global": []}}
    who = {} if role is None else {"name": name, "auth": {"accessToken": tok}}
    assert mcl.token_can_write(who)[0] is want


def upload_pair(tmp_path):
    data = b'{"format": "krea2-latents-v1"}'
    (tmp_path / "m.json").write_bytes(data)
    sha, git = mcl.local_digests(tmp_path / "m.json")
    assert (sha, git) == (hashlib.sha256(data).hexdigest(), hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest())
    exp = {"m.json": {"size": len(data), "sha256": sha, "git_sha1": git},
           "s.safetensors": {"size": 9, "sha256": "a" * 64}}
    remote = {"m.json": {"size": len(data), "sha256": None, "blob_id": git},
              "s.safetensors": {"size": 9, "sha256": "a" * 64, "blob_id": "f" * 40}}
    return exp, remote


@pytest.mark.parametrize("path,field,value,problem", [
    (None, None, None, None), ("s.safetensors", "sha256", "0" * 64, "sha256 s.safetensors"),
    ("m.json", "blob_id", "1" * 40, "git blob m.json"), ("m.json", "size", 1, "size m.json"),
    ("s.safetensors", "sha256", None, "unverifiable s.safetensors"), ("s.safetensors", None, None, "missing")])
def test_verify_upload_content(tmp_path, path, field, value, problem):
    exp, remote = upload_pair(tmp_path)
    if field:
        remote[path][field] = value
    elif path:
        del remote[path]
    probs = mcl.verify_upload(exp, remote)
    assert probs == [] if problem is None else (len(probs) == 1 and probs[0].startswith(problem))


PRE = mxf.step_prefix("krea2-fullft", "run1", 5)
OBJS = [{"key": k, "size": n, "etag": f"x{n}"} for n, k in enumerate(
    [PRE + r for r in ("_CHECKPOINT_METADATA", "params/_METADATA", "params/d/aa", "ema/d/bb",
                       "opt_state/d/cc", "meta/metadata", "params/")]
    + [mxf.step_prefix("krea2-fullft", "run1", 50) + "params/d/zz"])]


def test_select_ckpt_objects_only_step_tree_and_meta():
    assert sorted(mxf.select_ckpt_objects(OBJS, PRE, "params")) == [
        "_CHECKPOINT_METADATA", "meta/metadata", "params/_METADATA", "params/d/aa"]
    assert sorted(mxf.select_ckpt_objects(OBJS, PRE, "ema")) == ["_CHECKPOINT_METADATA", "ema/d/bb", "meta/metadata"]


@pytest.mark.parametrize("objs,tree,match", [
    ([o for o in OBJS if "/ema/" not in o["key"]], "ema", "no ema/ objects"),
    ([o for o in OBJS if "/meta/" not in o["key"]], "params", "no meta/ objects"),
    (OBJS + [{"key": PRE + "params/../x", "size": 1, "etag": "z"}], "params", "unsafe"),
    (OBJS, "opt_state", "unknown tree")])
def test_select_ckpt_objects_refusals(objs, tree, match):
    with pytest.raises(ValueError, match=match):
        mxf.select_ckpt_objects(objs, PRE, tree)


class FakeS3:
    def __init__(self):
        self.objects, self.uploads, self.cut = {}, 0, 0

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise type("NotFound", (Exception,), {"response": {"Error": {"Code": "404"}}})()
        return {"ContentLength": len(self.objects[Key][0]), "ETag": '"e"', "Metadata": dict(self.objects[Key][1])}

    def upload_file(self, filename, bucket, key, ExtraArgs=None, Config=None):
        data = open(filename, "rb").read()
        self.objects[key] = (data[:len(data) - self.cut], (ExtraArgs or {}).get("Metadata") or {})
        self.uploads += 1


def test_upload_skip_only_on_same_size_and_metadata(tmp_path):
    s3, f = FakeS3(), tmp_path / "x.safetensors"
    f.write_bytes(b"abc" * 100)
    up = lambda meta, **kw: mxf.upload_export(s3, "bkt", "k/x", f, meta, transfer_config="cfg",
                                              log=lambda *_: None, **kw)[0]
    m1, m2 = {"ckpt-fingerprint": "fp1", "sha256": "s1"}, {"ckpt-fingerprint": "fp2", "sha256": "s1"}
    assert [up(m1), up(m1), up(m1, force=True), up(m2)] == ["uploaded", "skipped", "uploaded", "uploaded"]
    s3.objects["k/x"] = (s3.objects["k/x"][0], {})
    assert up(m2) == "uploaded" and s3.objects["k/x"][1] == m2 and s3.uploads == 4
    f.write_bytes(b"abcd" * 100)
    assert up(m2) == "uploaded" and len(s3.objects["k/x"][0]) == 400 and s3.uploads == 5
    s3.cut = 1
    with pytest.raises(RuntimeError, match="uploaded object has 399 bytes"):
        up(m2, force=True)


class FakeHf:
    def __init__(self, role, private, auth_ok):
        self.role, self.private, self.auth_ok, self.calls = role, private, auth_ok, []

    def __getattr__(self, name):
        def call(*a, **kw):
            self.calls.append(name)
            if name == "auth_check" and not self.auth_ok:
                raise PermissionError("403 Forbidden")
            return {"whoami": {"name": OWNER, "auth": {"accessToken": {"role": self.role}}}, "repo_exists": True,
                    "repo_info": SimpleNamespace(private=self.private, sha="c0ffee")}.get(name)
        return call


@pytest.mark.parametrize("role,private,auth_ok,err,last", [
    ("read", True, True, "token role read", "whoami"), ("write", True, False, "403 Forbidden", "auth_check"),
    ("write", False, True, "not private", "repo_info")])
def test_finalize_refuses_before_publishing(tmp_path, role, private, auth_ok, err, last):
    api, ran = FakeHf(role, private, auth_ok), []
    r = mcl.finalize_impl(SimpleNamespace(reload=lambda: None, commit=lambda: None), api=api, out_dir=tmp_path,
                          runner=lambda *a, **kw: ran.append(a), python="py")
    assert not r["ok"] and err in r["error"] and api.calls[-1] == last and not ran
    assert "upload_large_folder" not in api.calls and ("create_repo" in api.calls) == (last == "repo_info")



@pytest.mark.parametrize("flags", [{"bench": True}, {"build": True}, {"stage_only": True}, {"finalize_only": True},
                                   {"cache_name": "v3"}, {"src_drive": "p"}, {"dest_r2": "x"}, {"reset_upload_cache": True}])
def test_profile_refused_with_other_mode_or_cache_flag(flags):
    assert mcl.resolve_mode(profile=True) == "profile"
    with pytest.raises(SystemExit):   # main calls resolve_mode before any Modal call
        mcl.resolve_mode(profile=True, **flags)


@pytest.mark.parametrize("name,src,dest,mode", [
    ("", "selection_v3/package", "", "build"), ("", "", "latents_v3", "finalize"), ("v3", "", "latents_v3", "build"),
    ("v3", "selection_v3/package", "", "build"), ("v3", "a/../b", "latents_v3", "build"), ("../x", "p", "y", "stage")])
def test_named_cache_never_uses_v2_dirs_or_repo(name, src, dest, mode):
    with pytest.raises(SystemExit):
        mcl.resolve_job(name, src, dest, mode)
    job = mcl.resolve_job("v3", "selection_v3/package", "latents_v3/", "build")
    assert (job["out"], job["src"], job["src_remote"], job["dest"]) == (
        "/vol/krea2-v3", "/vol/src/v3", "gdrive:danbooru_pipeline/selection_v3/package", "latents_v3")
    assert mcl.worker_cmd(0, 2, 4, 8, out=job["out"], tar_dir=job["src"])[-2:] == ["--tar-dir", "/vol/src/v3"]


def fake_rclone(calls, listing=(), cat=b"another build", probe_rc=0):
    def run(cmd, **kw):
        calls.append(cmd)
        out = json.dumps([{"Path": p, "Size": n, "Hashes": {"md5": "m"}} for p, n in dict(listing).items()]).encode()
        rc = probe_rc if mcl.PROBE_PREFIX in str(cmd) and "copyto" in cmd else 0
        return SimpleNamespace(returncode=rc, stderr=b"", stdout=out if "lsjson" in cmd else cat)
    return run


MAN = json.dumps({"shards": ["s.safetensors"]})
LOCAL = {"metadata.parquet": 2, "latents/s.safetensors": 3, mcl.MANIFEST: len(MAN)}


@pytest.mark.parametrize("case,remote,cat,precheck,probe_rc,gen", [
    ("fresh", {}, b"", False, 0, "g"), ("foreign", {mcl.MANIFEST: 30, "latents/s.safetensors": 3}, b"x", False, 0, "g"),
    ("published", LOCAL, MAN.encode(), False, 0, "g"), ("read-only", {}, b"", True, 1, "g"), ("other input", {}, b"", False, 0, "h")])
def test_finalize_r2_never_hf_publishes_once_and_prechecks(tmp_path, monkeypatch, case, remote, cat, precheck, probe_rc, gen):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)   # any HF use fails the finalize
    monkeypatch.setattr(mcl, "VOL_OUT", str(tmp_path / "krea2"))
    monkeypatch.setattr(mcl, "VOL_SRC", str(tmp_path / "src"))
    out, calls = tmp_path / "krea2-v3", []
    (out / "latents").mkdir(parents=True), (tmp_path / "src/v3").mkdir(parents=True)
    (out / "metadata.parquet").write_bytes(b"pq"), (out / "latents/s.safetensors").write_bytes(b"abc")
    (out / mcl.MANIFEST).write_text(MAN), (out / mcl.GEN_JSON).write_text('{"generation": "g"}')
    (tmp_path / "src/v3" / mcl.STAGE_JSON).write_text(json.dumps({"generation": gen}))
    r = mcl.finalize_r2_impl(SimpleNamespace(reload=lambda: None, commit=lambda: None), "v3", "latents_v3/", python="py",
                             runner=fake_rclone(calls, remote, cat, probe_rc), env={k: "x" for k in mcl.R2_ENV},
                             precheck=precheck)
    up, flat = [c for c in calls if c[1] in ("copy", "copyto") and mcl.PROBE_PREFIX not in str(c)], sum(calls, [])
    assert r["ok"] is (case in ("fresh", "published")) and bool(up) is r["ok"] and ("--write-manifest" in flat) is r["ok"]
    assert case != "other input" or (calls == [] and "input generation" in r["error"])
    assert case != "fresh" or (up[0][-len(mcl.RCLONE_COMMON) - len(mcl.R2_UPLOAD_FLAGS):][0] == "--checksum"
                               and up[-1][1:4] == ["copyto", str(out / mcl.MANIFEST), f"r2:x/latents_v3/{mcl.MANIFEST}"])
    assert case != "published" or (len(up) == 1 and "--ignore-existing" in up[0])   # immutable: add only, never replace


@pytest.mark.parametrize("case", ["ok", "cache from another input", "tar not in the package", "unmarked shards"])
def test_stage_pins_input_generation_and_needs_every_tar_in_package(tmp_path, monkeypatch, case):
    pa, pq = pytest.importorskip("pyarrow"), pytest.importorskip("pyarrow.parquet")
    monkeypatch.setattr(mcl, "VOL_OUT", str(tmp_path / "krea2"))
    monkeypatch.setattr(mcl, "VOL_SRC", str(tmp_path / "src"))
    src, out, calls = tmp_path / "src/v3", tmp_path / "krea2-v3", []
    (src / "data").mkdir(parents=True), (out / "latents").mkdir(parents=True)
    tars = ["data/train-0000.tar"] + (["data/train-0001.tar"] if case == "tar not in the package" else [])
    pq.write_table(pa.table({"id": list(range(len(tars))), "tar": tars}), src / "metadata.parquet")
    for t in tars:   # on disk (a stale local tar too), but the package lists only train-0000
        (src / t).write_bytes(b"t")
    if case == "cache from another input":
        (out / mcl.GEN_JSON).write_text('{"generation": "old"}')
    (src / mcl.STAGE_JSON).write_text('{"generation": "A"}')   # an earlier staging: gone unless re-validated
    if case == "unmarked shards":
        (out / "latents/train-0000.safetensors").write_bytes(b"s")
    r = mcl.stage_impl(SimpleNamespace(reload=lambda: None, commit=lambda: None), "v3", "selection_v3/package",
                       runner=fake_rclone(calls, {"metadata.parquet": 9, "data/train-0000.tar": 1}),
                       env={"RCLONE_CONF_B64": "eA=="}, conf_path=str(tmp_path / "c.conf"))
    synced = [c for c in calls if "sync" in c]
    assert r["ok"] is (case == "ok") and bool(synced) is (case in ("ok", "tar not in the package"))
    assert r["ok"] or (src / mcl.STAGE_JSON).exists() is (not synced)   # a failed restage leaves no stale stage.json
    assert not r["ok"] or json.loads((out / mcl.GEN_JSON).read_text())["generation"] == r["generation"] == json.loads(
        (src / mcl.STAGE_JSON).read_text())["generation"]


@pytest.mark.parametrize("case", ["ok", "drive changed", "bound to factor 1"])
def test_stage_reuses_another_package_only_at_its_generation(tmp_path, monkeypatch, case):
    pa, pq = pytest.importorskip("pyarrow"), pytest.importorskip("pyarrow.parquet")
    monkeypatch.setattr(mcl, "VOL_OUT", str(tmp_path / "krea2"))
    monkeypatch.setattr(mcl, "VOL_SRC", str(tmp_path / "src"))
    src, out, calls = tmp_path / "src/v3", tmp_path / "krea2-v3-512", []
    (src / "data").mkdir(parents=True), (src / "data/train-0000.tar").write_bytes(b"t")
    pq.write_table(pa.table({"id": [0], "tar": ["data/train-0000.tar"]}), src / "metadata.parquet")
    listing = {"metadata.parquet": (src / "metadata.parquet").stat().st_size, "data/train-0000.tar": 1}
    files = {p: {"size": n, "md5": "m"} for p, n in listing.items()}
    gen = mcl.input_generation(files)
    (src / mcl.STAGE_JSON).write_text(st := json.dumps({"src_remote": mcl.resolve_drive_src("selection_v3/package"),
                                                         "generation": gen, "files": files}))
    if case == "bound to factor 1":
        out.mkdir(parents=True), (out / mcl.GEN_JSON).write_text(json.dumps({"generation": gen}))
    r = mcl.stage_impl(SimpleNamespace(reload=lambda: None, commit=lambda: None), "v3-512", "selection_v3/package",
                       runner=fake_rclone(calls, {**listing, **({"data/train-0000.tar": 2} if "drive" in case else {})}),
                       env={"RCLONE_CONF_B64": "eA=="}, conf_path=str(tmp_path / "c.conf"), src_name="v3", downscale=2)
    assert r["ok"] is (case == "ok") and not [c for c in calls if "sync" in c]
    assert r["ok"] or ("holds input generation" if "drive" in case else "is bound to") in r["error"]
    assert (src / mcl.STAGE_JSON).read_text() == st   # the reused package is never re-staged or touched
    job = mcl.resolve_job("v3-512", mode="worker", src_name="v3", downscale=2)
    assert case != "ok" or (mcl.staged_generation(job) == gen and json.loads((out / mcl.GEN_JSON).read_text())
                            == {"src_remote": json.loads(st)["src_remote"], "generation": gen, "src_name": "v3",
                                "downscale": 2})
    assert mcl.finalize_cmd(out=job["out"], tar_dir=job["src"], downscale=2)[-4:] == [
        "--downscale", "2", "--tar-dir", str(src)]
    for bad in ({"src_name": "v3"}, {"downscale": 2}):   # a worker / finalize of another source or factor
        with pytest.raises(SystemExit):
            mcl.staged_generation(mcl.resolve_job("v3-512", mode="worker", **bad))


TE = {"repo": mef.TEXT_REPO, "revision": "te-rev-1"}
HF_ID = {"repo": mef.BASE_REPO, "revision": "c0ffee", "file": mef.BASE_FILE, "size": 25_000}
R2_ID = {"key": "krea2-fullft/run2/model.safetensors", "etag": "e2", "size": 25_600}
S = mef.sampler_settings(28, 4.5, TE)
P = mef.finalize_prompts([{"source": "ood", "row_id": None, "variant": None, "artist": "",
                           "caption": "c", "width": 32, "height": 32}], seed=5)[0]


@pytest.mark.parametrize("s,p", [
    ({**S, "sampler_version": "v1"}, P), ({**S, "steps": 20}, P), ({**S, "cfg": 3.0}, P),
    ({**S, "cfg_batch": False}, P), ({**S, "attn": S["attn"] + "x"}, P), ({**S, "compact_text": not S["compact_text"]}, P),
    ({**S, "text_encoder": {**TE, "revision": "x"}}, P), (S, {**P, "seed": 99}), (S, {**P, "caption": "o"})])
def test_check_pairable_refuses_sampler_difference(s, p):
    mef.check_pairable(mef.entry_key(S, P, HF_ID), mef.entry_key(S, P, R2_ID))
    with pytest.raises(mef.UnfairPairError, match="only the model may differ"):
        mef.check_pairable(mef.entry_key(S, P, HF_ID), mef.entry_key(s, p, R2_ID))


def test_prompts_file_table_refusals_and_r2_deleted_fallback(tmp_path, monkeypatch):
    f, good = tmp_path / "p.json", [{"caption": "a", "width": 1024, "height": 768, "x": 1},
                                    {"caption": "b", "width": 512, "height": 512}]
    f.write_text(json.dumps({"format": 1, "notes": "x", "prompts": good}))
    table, src = mef.load_prompts_file(f, seed=7)
    assert table == mef.finalize_prompts([{"source": "file", "row_id": None, "variant": None, "artist": "",
                                           **{k: g[k] for k in ("caption", "width", "height")}} for g in good], 7)
    assert src["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest() and src["path"] == str(f)
    for fmt, bad in ((1, [good[0], {**good[1], "caption": "a"}]), (1, [{**good[0], "width": 1000}]), (1, []),
                     (1, [{**good[0], "height": 0}]), (1, [{**good[0], "caption": " "}]), (2, good), (True, good)):
        f.write_text(json.dumps({"format": fmt, "prompts": bad}))
        with pytest.raises(ValueError):
            mef.load_prompts_file(f, 7)
    with pytest.raises(ValueError, match="replaces --n-ood"):
        mef.resolve_prompt_args(str(f), None, 0, 7)
    # an R2 object deleted from the bucket: only a volume copy marked for this key at its size is used
    monkeypatch.setattr(mef, "VOL_FT_ROOT", str(tmp_path / "ft"))
    key, path = "krea2-fullft/run2/model.safetensors", tmp_path / "ft/run2/model.safetensors"
    path.parent.mkdir(parents=True), path.write_bytes(b"w" * 10)
    for ident, ok in (({**R2_ID, "size": 10}, True), ({**R2_ID, "size": 11}, False), ({**R2_ID, "key": "x", "size": 10}, False)):
        mef.r2_marker_path(path).write_text(json.dumps(ident))
        if ok:
            assert mef.r2_deleted_identity(key, log=lambda *_: None) == ident
        else:
            with pytest.raises(RuntimeError, match="no such object"):
                mef.r2_deleted_identity(key, log=lambda *_: None)


def test_turbo_fixed_mu_raw_rule_keys_and_turbo_refusals(tmp_path):
    f = tmp_path / "p.json"
    sizes = ((1024, 1024), (2048, 2048), (768, 1344))   # official Turbo: mu 1.15 at every resolution
    f.write_text(json.dumps({"format": 1, "prompts": [{"caption": str(w * h + w), "width": w, "height": h} for w, h in sizes]}))
    steps, cfg, off, mu = mef.resolve_sampler_args(None, None, None, True)
    assert (steps, cfg, off, mu) == (8, 0.0, 0.0, 1.15)
    table = mef.resolve_prompt_args(str(f), None, None, 0, off, mu)[2]
    assert [p["mu"] for p in table] == [1.15] * 3 and "mu fixed 1.15" in mef.prompt_table(table, mu_offset=off, mu=mu)
    raw = mef.resolve_prompt_args(str(f), None, None, 0, *mef.resolve_sampler_args(None, None, None, False)[2:])[2]
    assert raw[0]["mu"] == mef.resolved_mu(1024, 1024) == 0.90625 and raw[1]["mu"] > 1.15   # Raw: official rule
    # offset 0 / no --turbo: the cache key of an existing entry, pinned from before --mu-offset existed
    steps, cfg, off, mu = mef.resolve_sampler_args(None, None, None, False)
    p = mef.finalize_prompts([{"source": "ood", "row_id": None, "variant": None, "artist": "",
                               "caption": "a", "width": 1024, "height": 1024}], 0, off)[0]
    assert mef.entry_hash(mef.entry_key(mef.sampler_settings(steps, cfg), p, {"key": "x"})) == \
        "1e6f1fc8cfd878bb33838496d2b0da0d"
    for kw in ({"steps": 8}, {"cfg": 0.0}, {"mu_offset": 0.25}):
        with pytest.raises(ValueError, match="--turbo sets"):
            mef.resolve_sampler_args(**{"steps": None, "cfg": None, "mu_offset": None, **kw}, turbo=True)
    for bad in (float("nan"), 2.5):
        with pytest.raises(ValueError, match="--mu-offset"):
            mef.resolve_sampler_args(None, None, bad, False)

def test_delta_compare_cos_and_energy_weighted_aggregates(tmp_path):
    """Parallel deltas give cos 1, orthogonal ones 0; aggregates are energy-weighted (not a mean of cosines)."""
    import numpy as np
    from safetensors.numpy import save_file
    e, q, m = np.eye(4, dtype=np.float32), "blocks.0.attn.wq.weight", "blocks.0.mod.lin.weight"
    deltas = {"a": {q: 0.25 * e, m: 2 * e[0]}, "b": {q: 0.5 * e, m: e[1]}}   # b = 2a on q; b orthogonal to a on m
    paths = {}
    for name, d in (("base", {}), *deltas.items()):
        paths[name] = tmp_path / f"{name}.safetensors"
        save_file({k: np.ones(v.shape, np.float32) + d.get(k, 0) for k, v in deltas["a"].items()}, str(paths[name]))
    rep = dc.delta_compare(paths["base"], [("a", paths["a"]), ("b", paths["b"])], chunk=3, log=lambda *_: None)
    rows = {r["key"]: r for r in rep["tensors"]}
    assert rows[q]["cos"]["a|b"] == pytest.approx(1) and rows[m]["cos"]["a|b"] == 0
    assert rows[q]["rel"]["b"] == pytest.approx(0.5 * 2 / 4)
    agg = rep["aggregates"]["all"]
    p = agg["pairs"]["a|b"]   # dot 0.5, ||a||^2 4.25, ||b||^2 2; mean of the cosines would be 0.5
    assert p["cos"] == pytest.approx(0.5 / np.sqrt(4.25 * 2)) and p["cos_median"] == pytest.approx(0.5)
    assert p["proj_on_first"] == pytest.approx(0.5 / 4.25) and p["cos2"] == pytest.approx(0.25 / 8.5)
    assert agg["noise_floor"] == pytest.approx(1 / np.sqrt(20)) and set(rep["aggregates"]["by_dtype"]) == {"F32"}
    assert rep["aggregates"]["by_kind"]["attn_q"]["pairs"]["a|b"]["cos"] == pytest.approx(1)
    notes = []   # roundoff-sized negative squared norm -> 0; a real negative one -> None + note, never an error
    assert dc._sq_norm(-4.5e-13, 1.0, "x", notes) == 0.0 and dc._sq_norm(-1e-3, 1.0, "y", notes) is None and len(notes) == 1


def test_transplant_alpha0_bytes_alpha1_sum_and_dtypes(tmp_path):
    """alpha 0 reproduces Turbo byte for byte (-0.0 included), alpha 1 = turbo + ft - raw rounded once to Turbo's
    per-key dtype (BF16 stays BF16, F32 stays F32)."""
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
    import build_transplant as bt
    rng, bf = np.random.default_rng(0), ml_dtypes.bfloat16
    w, s = "blocks.0.attn.wq.weight", "last.norm.scale"
    t = {w: rng.normal(size=(64, 8)).astype(bf), s: rng.normal(size=(8,)).astype(np.float32)}
    t[w][0, 0] = -0.0
    files = {"turbo": t, "raw": {w: rng.normal(size=(64, 8)).astype(bf), s: rng.normal(size=(8,)).astype(np.float32)},
             "ft": {w: rng.normal(size=(64, 8)).astype(np.float32), s: rng.normal(size=(8,)).astype(np.float32)}}
    p = {n: str(tmp_path / f"{n}.safetensors") for n in files}
    for n, d in files.items():
        save_file(d, p[n])
    for a in (0, 1):
        bt.build(p["turbo"], p["raw"], p["ft"], a, tmp_path / f"a{a}.st", log=lambda *_: None)
        with safe_open(str(tmp_path / f"a{a}.st"), "numpy") as f:
            assert f.metadata()["alpha"] == f"{a}.0" and {k: f.get_slice(k).get_dtype() for k in f.keys()} == {w: "BF16", s: "F32"}
            for k, tk in t.items():
                want = tk if a == 0 else (tk.astype(np.float32) + (files["ft"][k] - files["raw"][k].astype(np.float32))).astype(tk.dtype)
                assert f.get_tensor(k).tobytes() == want.tobytes()
