"""Paid-run, credential and fairness guards of the Modal scripts, imported as
plain modules (no Modal SDK, no network)."""

import hashlib
from types import SimpleNamespace

import pytest

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
