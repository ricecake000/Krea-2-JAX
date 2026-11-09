#!/usr/bin/env python3
"""Hugging Face side of the release kit (huggingface_hub + hf_xet). The token is read from a file, never printed.

  hf_release.py whoami  --token-file F --expect-user U [--repos A,B] [--cards DIR]
      token valid + owner; repos exist? each repo's README.md passes HF's validate-yaml (read only, no commit)
  hf_release.py prepare --token-file F --expect-user U --repos A,B --cards DIR
      (laptop, BEFORE any billing) README validate-yaml (refuse on any error), create the repos public (exist_ok) +
      gated="auto", read back, then ONE commit per repo of README.md, LICENSE.pdf, NOTICE from the laptop (skipped
      when the repo already holds exactly these bytes). DIR = {raw,turbo}/README.md, LICENSE.pdf, NOTICE; a repo
      ending in -Raw takes raw/, -Turbo takes turbo/.
  hf_release.py upload  --token-file F --repo R --folder DIR --manifest M --expect E
      upload DIR (the model files only; upload_large_folder: resumable, Xet dedup), then the post-upload check: the
      remote file list (recursive) must hold exactly E's paths with E's sizes (cards included; no re-hashing).
      M = JSON {"files": {relpath: size}} of DIR (written by the VM job); E = the same for the whole repo (plan).
  hf_release.py upload-pr --token-file F --repo R --file LOCAL --path-in-repo NAME --size N --title T
      one file as ONE pull request (main untouched): refuse if main already holds NAME; an open PR of ours titled T
      whose tree lists NAME with size N -> HF_PR_EXISTS (idempotent re-run); else create_commit(create_pr=True), then
      the PR tree must list NAME (size N) and every main file with its main size, main must not -> HF_PR_OK.
  hf_release.py merge-pr --token-file F --repo R --pr NUM --path-in-repo NAME --size N
      (laptop, after the user's OK) PR NUM must be an open PR whose tree lists NAME (size N); merge it; main must then
      list NAME with size N -> HF_MERGED.
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


def remote_sizes(api, repo, revision=None):
    from huggingface_hub.hf_api import RepoFile
    return {f.path: f.size for f in api.list_repo_tree(repo, recursive=True, repo_type="model", revision=revision)
            if isinstance(f, RepoFile)}


OPEN_PR = ("open", "draft")


def upload_pr(api, a, user, say):
    from huggingface_hub import CommitOperationAdd
    name, size = a.path_in_repo, a.size
    if os.path.getsize(a.file) != size:
        say(f"HF_FAIL {a.file}: local size {os.path.getsize(a.file)} != --size {size}"); return 1
    main = remote_sizes(api, a.repo)
    if name in main:
        say(f"HF_FAIL {a.repo}: main already holds {name} (size {main[name]}): nothing to propose"); return 1
    for d in api.get_repo_discussions(a.repo, author=user, discussion_type="pull_request", repo_type="model"):
        if d.status not in OPEN_PR or d.title != a.title or d.author != user:
            continue
        got = remote_sizes(api, a.repo, revision=d.git_reference).get(name)
        if got == size:
            say(f"HF_PR_EXISTS {a.repo} pr={d.num} url={d.url} ({name} size={size} already proposed; nothing uploaded)")
            return 0
        say(f"HF_PR_OTHER {a.repo} pr={d.num} status={d.status} same title, {name} size={got} != {size}: left alone")
    say(f"HF_PR_START {a.repo} {name} bytes={size}")
    c = api.create_commit(a.repo, operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=a.file)],
                          commit_message=a.title, repo_type="model", create_pr=True)
    if c.pr_num is None:
        say(f"HF_FAIL {a.repo}: create_commit opened no pull request (commit {c.oid})"); return 1
    pr = remote_sizes(api, a.repo, revision=c.pr_revision)
    bad = {p: (s, pr.get(p)) for p, s in main.items() if pr.get(p) != s}
    if pr.get(name) != size or bad:
        say(f"HF_FAIL {a.repo} pr={c.pr_num}: PR tree has {name}={pr.get(name)} (want {size}); main files changed {bad}")
        return 1
    if name in remote_sizes(api, a.repo):
        say(f"HF_FAIL {a.repo}: main lists {name} after a PR-only commit"); return 1
    say(f"HF_PR_OK {a.repo} pr={c.pr_num} url={c.pr_url} ({name} size={size}; main untouched)")
    return 0


def merge_pr(api, a, user, say):
    name, size = a.path_in_repo, a.size
    d = api.get_discussion_details(a.repo, a.pr, repo_type="model")
    if not d.is_pull_request:
        say(f"HF_FAIL {a.repo}: #{a.pr} is not a pull request"); return 1
    if d.author != user:   # only a PR this token's user opened (upload-pr) is ever merged
        say(f"HF_FAIL {a.repo}: PR #{a.pr} was opened by {d.author!r}, not {user!r}: not merging"); return 1
    if d.status == "merged" and remote_sizes(api, a.repo).get(name) == size:
        say(f"HF_MERGED {a.repo} pr={a.pr} (already merged; main lists {name} size={size})"); return 0
    if d.status not in OPEN_PR:
        say(f"HF_FAIL {a.repo}: PR #{a.pr} is {d.status}"); return 1
    pr, main = remote_sizes(api, a.repo, revision=d.git_reference), remote_sizes(api, a.repo)
    want = dict(main, **{name: size})
    if pr != want:   # the PR must add exactly NAME and leave every main file as it is (by path + size)
        diff = {p: (want.get(p), pr.get(p)) for p in set(pr) | set(want) if pr.get(p) != want.get(p)}
        say(f"HF_FAIL {a.repo}: PR #{a.pr} tree != main + {name} ({size} B): (expected, PR) {diff}: not merging"); return 1
    api.merge_pull_request(a.repo, a.pr, repo_type="model")
    got = remote_sizes(api, a.repo).get(name)
    if got != size:
        say(f"HF_FAIL {a.repo}: after the merge main has {name}={got}, expected {size}"); return 1
    say(f"HF_MERGED {a.repo} pr={a.pr} (main lists {name} size={size})")
    return 0


CARD_FILES = ("README.md", "LICENSE.pdf", "NOTICE")


def card_paths(cards, repo):
    sub = {"-raw": "raw", "-turbo": "turbo"}[next(k for k in ("-raw", "-turbo") if repo.lower().endswith(k))]
    return {"README.md": os.path.join(cards, sub, "README.md"), "LICENSE.pdf": os.path.join(cards, "LICENSE.pdf"),
            "NOTICE": os.path.join(cards, "NOTICE")}


def validate_readme(path):
    """HF's own model-card metadata validation (the check that rejects a commit): returns (errors, warnings)."""
    import urllib.error, urllib.request
    body = json.dumps({"content": open(path, encoding="utf-8").read(), "repoType": "model"}).encode()
    req = urllib.request.Request("https://huggingface.co/api/validate-yaml", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        d = json.loads(e.read())
    return d.get("errors", []), d.get("warnings", [])


def remote_bytes(api, repo, path):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        return open(hf_hub_download(repo, path, repo_type="model", token=api.token, force_download=True), "rb").read()
    except EntryNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["whoami", "prepare", "upload", "upload-pr", "merge-pr"])
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--expect-user", default="ricecake")
    ap.add_argument("--repos", default="")
    ap.add_argument("--repo"); ap.add_argument("--folder"); ap.add_argument("--manifest"); ap.add_argument("--expect")
    ap.add_argument("--cards")
    ap.add_argument("--file"); ap.add_argument("--path-in-repo"); ap.add_argument("--size", type=int)
    ap.add_argument("--title"); ap.add_argument("--pr", type=int)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    api, tok = api_for(a.token_file)
    say = lambda s: print(redact(s, tok), flush=True)
    try:
        if a.cmd in ("upload-pr", "merge-pr"):
            need = ["repo", "path_in_repo", "size"] + (["file", "title"] if a.cmd == "upload-pr" else ["pr"])
            miss = [n for n in need if getattr(a, n) is None]
            if miss:
                say(f"HF_FAIL {a.cmd} needs " + " ".join("--" + n.replace("_", "-") for n in miss)); return 1
            name = api.whoami().get("name")
            if name != a.expect_user or not a.repo.startswith(name + "/"):
                say(f"HF_FAIL token belongs to {name!r} (expected {a.expect_user!r}, repo {a.repo})"); return 1
            return upload_pr(api, a, name, say) if a.cmd == "upload-pr" else merge_pr(api, a, name, say)
        if a.cmd in ("whoami", "prepare"):
            me = api.whoami()
            name = me.get("name")
            if name != a.expect_user:
                say(f"HF_FAIL token belongs to {name!r}, expected {a.expect_user!r}"); return 1
            role = (me.get("auth", {}).get("accessToken", {}) or {}).get("role", "?")
            say(f"HF_WHOAMI_OK user={name} role={role}")
            repos = list(filter(None, a.repos.split(",")))
            if a.cmd == "prepare" and not a.cards:
                say("HF_FAIL prepare needs --cards"); return 1
            for r in repos:   # all cards are validated before anything is created or committed
                if not r.startswith(a.expect_user + "/"):
                    say(f"HF_FAIL repo {r} is not under {a.expect_user}/"); return 1
                if a.cards:
                    errs, warns = validate_readme(card_paths(a.cards, r)["README.md"])
                    if errs:
                        say(f"HF_FAIL {r}: README.md fails HF validate-yaml: {errs}"); return 1
                    say(f"HF_CARD_OK {r} README.md validate-yaml: 0 errors, {len(warns)} warnings {warns or ''}")
            for r in repos:
                if a.cmd == "whoami":
                    say(f"HF_REPO {r} exists={api.repo_exists(r, repo_type='model')}")
                    continue
                api.create_repo(r, repo_type="model", private=False, exist_ok=True)
                api.update_repo_settings(r, gated="auto", private=False, repo_type="model")
                info = api.model_info(r)
                if info.private or info.gated != "auto":
                    say(f"HF_FAIL {r}: read back private={info.private} gated={info.gated}"); return 1
                say(f"HF_REPO_READY {r} private=False gated=auto")
                from huggingface_hub import CommitOperationAdd
                cp = card_paths(a.cards, r)
                local = {n: open(cp[n], "rb").read() for n in CARD_FILES}
                if all(remote_bytes(api, r, n) == local[n] for n in CARD_FILES):
                    say(f"HF_CARDS_PRESENT {r} README.md, LICENSE.pdf, NOTICE already hold these bytes: no commit")
                    continue
                c = api.create_commit(r, repo_type="model", commit_message="Model card, license, notice",
                                      operations=[CommitOperationAdd(n, cp[n]) for n in CARD_FILES])
                got = remote_sizes(api, r)
                bad = [n for n in CARD_FILES if got.get(n) != len(local[n])]
                if bad:
                    say(f"HF_FAIL {r}: cards not listed with their sizes after the commit: {bad}"); return 1
                say(f"HF_CARDS_COMMITTED {r} {c.oid[:12]} " + ", ".join(f"{n}={len(local[n])}" for n in CARD_FILES))
            return 0
        if not (a.repo and a.folder and a.manifest and a.expect):
            say("HF_FAIL upload needs --repo --folder --manifest --expect"); return 1
        want = json.load(open(a.manifest))["files"]
        expect = json.load(open(a.expect))["files"]
        if any(expect.get(k) != v for k, v in want.items()):
            say("HF_FAIL the manifest is not a subset of the expected repo file list"); return 1
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
        bad = {p: (s, got.get(p)) for p, s in expect.items() if got.get(p) != s}
        extra = sorted(set(got) - set(expect))
        if bad or extra:
            say(f"HF_FAIL {a.repo} remote list != plan: wrong/missing {bad} extra {extra}"); return 1
        say(f"HF_UPLOAD_OK {a.repo} uploaded {len(want)} files / {sum(want.values())} B; remote list + sizes == plan "
            f"({len(expect)} files incl. cards)")
        return 0
    except Exception as e:  # noqa: BLE001 - every failure is reported (redacted) and fails the step
        say(f"HF_FAIL {a.cmd}: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
