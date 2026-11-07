#!/usr/bin/env python3
"""Hugging Face side of the release kit (huggingface_hub + hf_xet). The token is read from a file, never printed.

  hf_release.py whoami  --token-file F --expect-user U [--repos A,B]   token valid + owner; repos exist? (read only)
  hf_release.py prepare --token-file F --expect-user U --repos A,B     (laptop, BEFORE any billing) create the repos
                                                                       public (exist_ok) + gated="auto"; read back
  hf_release.py upload  --token-file F --repo R --folder DIR --manifest M
      upload DIR (upload_large_folder: resumable, Xet dedup), then the post-upload check: the remote file list
      (recursive) must contain exactly the manifest's paths with the manifest's sizes (no re-hashing).
      M = JSON {"files": {relpath: size}} written by the VM job from the staged folder.
Lines: HF_* ... ; failures HF_FAIL <why> (exit 1). Exception texts pass through redact() (token removed).
"""
import argparse, json, os, sys


def redact(s, tok):
    s = str(s)
    return s.replace(tok, "<token>") if tok else s


def api_for(token_file):
    from huggingface_hub import HfApi
    tok = open(token_file).read().strip()
    if not tok or any(c.isspace() for c in tok):
        raise SystemExit("HF_FAIL token file is empty or malformed")
    return HfApi(token=tok), tok


def remote_sizes(api, repo):
    from huggingface_hub.hf_api import RepoFile
    return {f.path: f.size for f in api.list_repo_tree(repo, recursive=True, repo_type="model") if isinstance(f, RepoFile)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["whoami", "prepare", "upload"])
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--expect-user", default="ricecake")
    ap.add_argument("--repos", default="")
    ap.add_argument("--repo"); ap.add_argument("--folder"); ap.add_argument("--manifest")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    api, tok = api_for(a.token_file)
    say = lambda s: print(redact(s, tok), flush=True)
    try:
        if a.cmd in ("whoami", "prepare"):
            me = api.whoami()
            name = me.get("name")
            if name != a.expect_user:
                say(f"HF_FAIL token belongs to {name!r}, expected {a.expect_user!r}"); return 1
            role = (me.get("auth", {}).get("accessToken", {}) or {}).get("role", "?")
            say(f"HF_WHOAMI_OK user={name} role={role}")
            for r in filter(None, a.repos.split(",")):
                if not r.startswith(a.expect_user + "/"):
                    say(f"HF_FAIL repo {r} is not under {a.expect_user}/"); return 1
                if a.cmd == "whoami":
                    say(f"HF_REPO {r} exists={api.repo_exists(r, repo_type='model')}")
                    continue
                api.create_repo(r, repo_type="model", private=False, exist_ok=True)
                api.update_repo_settings(r, gated="auto", private=False, repo_type="model")
                info = api.model_info(r)
                if info.private or info.gated != "auto":
                    say(f"HF_FAIL {r}: read back private={info.private} gated={info.gated}"); return 1
                say(f"HF_REPO_READY {r} private=False gated=auto")
            return 0
        if not (a.repo and a.folder and a.manifest):
            say("HF_FAIL upload needs --repo --folder --manifest"); return 1
        want = json.load(open(a.manifest))["files"]
        local = {}
        for root, dirs, files in os.walk(a.folder):
            dirs[:] = [d for d in dirs if d != ".cache"]
            for fn in files:
                p = os.path.join(root, fn)
                local[os.path.relpath(p, a.folder)] = os.path.getsize(p)
        if local != want:
            say(f"HF_FAIL staged folder != manifest: extra {sorted(set(local) - set(want))} missing {sorted(set(want) - set(local))}")
            return 1
        say(f"HF_UPLOAD_START {a.repo} files={len(want)} bytes={sum(want.values())}")
        api.upload_large_folder(repo_id=a.repo, folder_path=a.folder, repo_type="model", num_workers=a.workers,
                                ignore_patterns=[".cache/**"], print_report=True, print_report_every=120)
        got = remote_sizes(api, a.repo)
        got.pop(".gitattributes", None)
        bad = {p: (s, got.get(p)) for p, s in want.items() if got.get(p) != s}
        extra = sorted(set(got) - set(want))
        if bad or extra:
            say(f"HF_FAIL {a.repo} remote list != manifest: wrong/missing {bad} extra {extra}"); return 1
        say(f"HF_UPLOAD_OK {a.repo} files={len(want)} bytes={sum(want.values())} (remote list + sizes == manifest)")
        return 0
    except Exception as e:  # noqa: BLE001 - every failure is reported (redacted) and fails the step
        say(f"HF_FAIL {a.cmd}: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
