"""Apply a data-filter patch (scripts/datafilter_apply.py) to the
`krea2-latents-v1` cache as a NEW dataset version, without touching (or
re-encoding) any existing shard: the VAE posterior sample is unseeded, so
existing shards are never rebuilt; only a patch shard is added.

Patch directory (written by scripts/datafilter_apply.py):
    images/<id>.webp   final bucket-sized WebP for re-cropped samples
    patch.parquet      one row per acted id: id, action (caption|crop|remove),
                       bucket_w, bucket_h, long/medium/short (new),
                       *_old (the dataset's captions the patch was made
                       against), long_tok/medium_tok/short_tok, status, ...
    patch.json         source dataset revision, review_id, counts, versions

Applied rows: status "ok" (any action), or "crop_unsure" for a crop (the
image is produced, only flagged in the review report). Every other status
(crop_too_small, over_budget, error, ...) is "unapplied": `build` fails
unless --unapplied keep (leave the row as it is) or --unapplied remove
(drop the sample).

Mechanism (read by krea2_jax/data.py::BucketDataset):
    latents/patch-NNNN.safetensors   NEW shard: "<id>" -> (16, h/8, w/8) bf16
                                     latents of the applied crops (next free
                                     NNNN; shard metadata source
                                     "patch:<review_id>", images_sha256)
    latents/manifest.json            base manifest + the patch shard in
                                     "shards"/"digests", n += its count,
                                     "superseded": {shard: [ids]} = keys the
                                     loader skips (removed ids, and cropped
                                     ids' old latents), "patches": [{...}]
                                     provenance (base revision + sha256s,
                                     patch file sha256s, counts)
    metadata.parquet                 base rows minus removed ids; captions,
                                     *_tok (and bucket_w/h for crops) from
                                     the patch; new column `latent_shard` =
                                     the shard holding each id's live latent
                                     (checked by the loader)
Old shards are byte-identical and keep their digests; the new version is
the base repo's files + these three, so it is uploaded as a commit on a
BRANCH of the same HF repo (old shards are not re-sent). The dataset
fingerprint changes (captions, samples, superseded, patch digest): a
training run on it starts fresh, it cannot resume a run of the old version.
`--verify-latents` (BucketDataset verify_digests) re-hashes every shard
incl. the patch shard, as before.

Commands (the parts that cost money or touch the Hub are explicit steps):
  (1) encode the crops on Modal (B200, same image / VAE / volume as
      scripts/modal_cache_latents.py; stdlib-only local side: the patch
      files are uploaded to the volume, read and encoded in the container):
        modal run scripts/modal_patch_latents.py --patch-dir P
      -> P/patch_latents.safetensors   (--vae-impl 2d default, --gpu B200)
      Local alternative / tests:
        python scripts/modal_patch_latents.py encode --patch-dir P [--vae-stub]
  (2) build the new version locally (base = metadata.parquet +
      latents/manifest.json of the base revision; shards optional):
        python scripts/modal_patch_latents.py build --base BASE \\
            --base-revision SHA --patch-dir P --out NEW
      -> NEW/{metadata.parquet, latents/manifest.json,
              latents/patch-NNNN.safetensors, build_report.json}
      With the base shards on disk, --link-base-shards symlinks them into
      NEW/latents and loads NEW with BucketDataset (full verify).
  (3) upload (separate, explicit; refuses branch main unless --allow-main):
        python scripts/modal_patch_latents.py upload --new NEW \\
            --branch datafilter-<review_id>
      creates the branch at the base revision, commits the three files,
      then checks every file of the new manifest on the Hub (sha256).

Importable without the Modal SDK, torch or pyarrow at module level.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

try:
    import modal
except ImportError:  # importable (tests, `python ... build`) without Modal
    modal = None

HERE = Path(__file__).resolve().parent
LOCAL_REPO = HERE.parent
REMOTE_REPO = "/repo"
for _p in (str(LOCAL_REPO), str(HERE), REMOTE_REPO, f"{REMOTE_REPO}/scripts"):
    if _p not in sys.path:
        sys.path.append(_p)

FORMAT = "krea2-latents-v1"
LAT_CHANNELS = 16
VAE_SCALE = 8
PATCH = 2
BUCKET_MULTIPLE = VAE_SCALE * PATCH
MAX_IMG_TOKENS = 4096
VAE_REPO = "Qwen/Qwen-Image"
LATENTS_REPO = "ricecake/danbooru-curated-v2-latents"
LATENT_SHARD_COLUMN = "latent_shard"
PATCH_SHARD_RE = re.compile(r"^patch-(\d{4})\.safetensors$")
PATCH_LATENTS = "patch_latents.safetensors"   # encode output in the patch dir
PATCH_FILES = ("patch.parquet", "patch.json")
ACTIONS = ("caption", "crop", "remove")
APPLY_STATUS = {"caption": {"ok"}, "crop": {"ok", "crop_unsure"},
                "remove": {"ok"}}
CAPTIONS = ("long", "medium", "short")
TOK_COLS = tuple(c + "_tok" for c in CAPTIONS)
REQUIRED_COLS = (("id", "action", "status", "bucket_w", "bucket_h")
                 + CAPTIONS + tuple(c + "_old" for c in CAPTIONS) + TOK_COLS)
UNAPPLIED_MODES = ("fail", "keep", "remove")
DEFAULT_GPU = "B200"
DEFAULT_VAE_IMPL = "2d"   # eager: no compile per (bucket, batch) shape
VAE_IMPLS = ("ref", "2d", "2d-compile")
VOL_PATCHES = "/vol/patches"
ENCODE_TIMEOUT_S = 60 * 60


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _examples(items, k=5):
    items = list(items)
    more = f", ... (+{len(items) - k})" if len(items) > k else ""
    return ", ".join(str(x) for x in items[:k]) + more


def _fail(title, problems):
    raise SystemExit(f"{title}:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------- patch dir
def read_patch(patch_dir):
    """-> {dir, rows (patch.parquet rows, id int), by_id, meta (patch.json),
    review_id, parquet_sha256, json_sha256}. SystemExit on a malformed
    patch (missing files/columns, duplicate ids, unknown actions)."""
    import pyarrow.parquet as pq

    d = Path(patch_dir)
    for f in PATCH_FILES:
        if not (d / f).is_file():
            raise SystemExit(f"{d / f} not found (patch dir from "
                             "scripts/datafilter_apply.py)")
    table = pq.read_table(d / "patch.parquet")
    missing = [c for c in REQUIRED_COLS if c not in table.column_names]
    if missing:
        raise SystemExit(f"{d / 'patch.parquet'}: missing columns {missing}")
    rows = table.to_pylist()
    problems, by_id = [], {}
    for r in rows:
        if r["id"] is None:
            problems.append("row with null id")
            continue
        r["id"] = int(r["id"])
        if r["id"] in by_id:
            problems.append(f"duplicate id {r['id']}")
        by_id[r["id"]] = r
        if r["action"] not in ACTIONS:
            problems.append(f"{r['id']}: unknown action {r['action']!r}")
    if problems:
        _fail(f"{d / 'patch.parquet'} is malformed", problems)
    meta = json.loads((d / "patch.json").read_text())
    if not isinstance(meta, dict):
        raise SystemExit(f"{d / 'patch.json'} is not a JSON object")
    return {"dir": d, "rows": rows, "by_id": by_id, "meta": meta,
            "review_id": str(meta.get("review_id") or "unknown"),
            "parquet_sha256": file_sha256(d / "patch.parquet"),
            "json_sha256": file_sha256(d / "patch.json")}


def is_applied(row):
    return row["status"] in APPLY_STATUS[row["action"]]


def bucket_problem(w, h):
    """None if (w, h) is a usable bucket, else the reason."""
    if w is None or h is None:
        return "null bucket"
    w, h = int(w), int(h)
    if w <= 0 or h <= 0 or w % BUCKET_MULTIPLE or h % BUCKET_MULTIPLE:
        return f"bucket {w}x{h} is not a positive multiple of {BUCKET_MULTIPLE}"
    tok = (w // BUCKET_MULTIPLE) * (h // BUCKET_MULTIPLE)
    if tok > MAX_IMG_TOKENS:
        return f"bucket {w}x{h} is {tok} image tokens > {MAX_IMG_TOKENS}"
    return None


def crop_items(patch):
    """Applied crop rows -> sorted [(id, image path, bucket_w, bucket_h)].
    Every applied crop needs images/<id>.webp of exactly its bucket size;
    images of unapplied crop rows are ignored; any other file in images/ is
    an error (SystemExit listing every problem)."""
    from PIL import Image

    d = patch["dir"] / "images"
    present = ({p.name: p for p in d.iterdir() if p.is_file()}
               if d.is_dir() else {})
    problems, items, used = [], [], set()
    for r in sorted(patch["rows"], key=lambda r: r["id"]):
        name = f"{r['id']}.webp"
        if r["action"] != "crop":
            continue
        used.add(name)  # an unapplied crop may still have its image
        if not is_applied(r):
            continue
        bp = bucket_problem(r["bucket_w"], r["bucket_h"])
        if bp:
            problems.append(f"{r['id']}: {bp}")
            continue
        w, h = int(r["bucket_w"]), int(r["bucket_h"])
        path = present.get(name)
        if path is None:
            problems.append(f"{r['id']}: no images/{name}")
            continue
        with Image.open(path) as im:
            size, fmt = im.size, im.format
        if size != (w, h):
            problems.append(f"{r['id']}: images/{name} is {size[0]}x{size[1]}"
                            f", patch bucket {w}x{h}")
        if fmt != "WEBP":
            problems.append(f"{r['id']}: images/{name} is {fmt}, not WEBP")
        items.append((r["id"], path, w, h))
    stray = sorted(set(present) - used)
    if stray:
        problems.append(f"{len(stray)} files in images/ belong to no crop row "
                        f"({_examples(stray)})")
    if problems:
        _fail(f"{patch['dir']}: patch images do not match patch.parquet",
              problems)
    return items


def images_digest(items):
    """sha256 over sorted [id, w, h, sha256(image bytes)]: what a patch
    shard was encoded from (stored as its metadata images_sha256)."""
    rec = sorted([int(i), int(w), int(h), file_sha256(p)]
                 for i, p, w, h in items)
    return hashlib.sha256(json.dumps(rec, separators=(",", ":")).encode()
                          ).hexdigest()


# ------------------------------------------------------------------ encode
def encode_items(items, encoder, out_path, *, vae_id, vae_impl, source,
                 batch=16, decode_threads=4, log=print):
    """VAE-encode crop items into one krea2-latents-v1 shard (atomic write,
    cache_latents.encode_samples: same decode / normalization / bf16 /
    contiguity as the base shards). -> {n, seconds, sha256, images_sha256}."""
    from concurrent.futures import ThreadPoolExecutor

    from safetensors.torch import save_file

    import cache_latents as cl

    t0 = time.time()
    digest = images_digest(items)
    samples = [{"id": int(i), "data": Path(p).read_bytes(), "w": int(w),
                "h": int(h)} for i, p, w, h in items]
    with ThreadPoolExecutor(max(1, decode_threads),
                            thread_name_prefix="patch_decode") as pool:
        tensors, resized = cl.encode_samples(samples, encoder, batch, pool,
                                             log=log, tag="patch")
    if resized:
        raise SystemExit(f"{resized} patch images were not bucket-sized")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    meta = {"format": FORMAT, "vae": vae_id, "source": source,
            "count": str(len(tensors)), "kind": "patch",
            "images_sha256": digest}
    if vae_impl is not None:
        meta["vae_impl"] = vae_impl
    save_file(tensors, str(tmp), metadata=meta)
    os.replace(tmp, out_path)
    return {"n": len(tensors), "seconds": time.time() - t0,
            "sha256": file_sha256(out_path), "images_sha256": digest}


def encode_patch_dir(patch_dir, out_path, encoder, *, vae_id, vae_impl,
                     batch=16, decode_threads=4, log=print):
    patch = read_patch(patch_dir)
    items = crop_items(patch)
    if not items:
        raise SystemExit(f"{patch_dir}: no applied crop rows: nothing to "
                         "encode (build needs no --latents)")
    log(f"encoding {len(items)} crops of review {patch['review_id']} "
        f"(vae {vae_id}, impl {vae_impl}, batch {batch})")
    return encode_items(items, encoder, out_path, vae_id=vae_id,
                        vae_impl=vae_impl,
                        source=f"patch:{patch['review_id']}", batch=batch,
                        decode_threads=decode_threads, log=log)


def load_encoder(vae_impl, device=None, dtype="auto", vae=VAE_REPO):
    import torch

    from cache_dataset import VaeEncoder
    from vae_fast import make_encoder

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == "auto":
        dt = torch.float32 if device == "cpu" else torch.bfloat16
    else:
        dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    return make_encoder(VaeEncoder(vae, device, dt), vae_impl), device


# ------------------------------------------------------------- shard checks
def read_shard_header(path):
    """-> (metadata dict, {key: (shape, dtype)})."""
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta = dict(f.metadata() or {})
        keys = {}
        for k in f.keys():
            s = f.get_slice(k)
            keys[k] = (tuple(s.get_shape()), s.get_dtype())
    return meta, keys


def check_patch_shard(path, items, vae, lat_channels=LAT_CHANNELS):
    """The encoded shard must hold exactly the applied crops, encoded from
    these exact images (images_sha256), bucket-shaped, bf16, finite, of the
    base's VAE. SystemExit listing every problem."""
    import numpy as np
    from safetensors import safe_open

    meta, keys = read_shard_header(path)
    problems = []
    for k, want in (("format", FORMAT), ("vae", vae), ("kind", "patch"),
                    ("count", str(len(keys)))):
        if meta.get(k) != want:
            problems.append(f"metadata {k} {meta.get(k)!r} != {want!r}")
    digest = images_digest(items)
    if meta.get("images_sha256") != digest:
        problems.append(f"metadata images_sha256 {meta.get('images_sha256')}"
                        f" != {digest} of the patch images (encoded from "
                        "other images: re-run the encode step)")
    want = {str(i): (lat_channels, h // VAE_SCALE, w // VAE_SCALE)
            for i, _, w, h in items}
    extra, absent = sorted(set(keys) - set(want)), sorted(set(want) - set(keys))
    if extra:
        problems.append(f"{len(extra)} latents of no applied crop "
                        f"({_examples(extra)})")
    if absent:
        problems.append(f"{len(absent)} applied crops without a latent "
                        f"({_examples(absent)})")
    bad = [f"{k}:{keys[k][0]}!={want[k]}" for k in sorted(set(keys) & set(want))
           if keys[k][0] != want[k]]
    if bad:
        problems.append(f"{len(bad)} latents not of their bucket shape "
                        f"({_examples(bad)})")
    bad = [f"{k}:{d}" for k, (_, d) in sorted(keys.items()) if d != "BF16"]
    if bad:
        problems.append(f"{len(bad)} latents not bf16 ({_examples(bad)})")
    if not problems:
        with safe_open(str(path), framework="numpy") as f:
            nonfinite = [k for k in sorted(keys) if not np.isfinite(
                f.get_tensor(k).astype(np.float32)).all()]
        if nonfinite:
            problems.append(f"{len(nonfinite)} latents not finite "
                            f"({_examples(nonfinite)})")
    if problems:
        _fail(f"{path}: patch shard check FAILED", problems)
    return meta


# ------------------------------------------------------------------- build
def shard_name(tar):
    """'data/train-0000.tar' -> 'train-0000.safetensors' (as
    scripts/cache_latents.py names shards)."""
    return Path(tar).stem + ".safetensors"


def next_patch_shard(shards):
    nums = [int(m.group(1)) for s in shards
            for m in [PATCH_SHARD_RE.match(s)] if m]
    return f"patch-{(max(nums) + 1) if nums else 0:04d}.safetensors"


def _local_digests(path):
    """-> {size, sha256, git_sha1} (git blob id: what the Hub reports for a
    non-LFS file)."""
    path = Path(path)
    sha, git = hashlib.sha256(), hashlib.sha1()
    git.update(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            sha.update(b)
            git.update(b)
    return {"size": path.stat().st_size, "sha256": sha.hexdigest(),
            "git_sha1": git.hexdigest()}


def read_base(base_dir):
    """Base version: metadata.parquet table + manifest.json (+ each id's
    current shard). Needs no shard file."""
    import pyarrow.parquet as pq

    from krea2_jax.hashing import is_sha256_hex

    base = Path(base_dir)
    mpath = base / "latents" / "manifest.json"
    ppath = base / "metadata.parquet"
    for p in (mpath, ppath):
        if not p.is_file():
            raise SystemExit(f"{p} not found (base = the base revision's "
                             "metadata.parquet + latents/manifest.json)")
    manifest = json.loads(mpath.read_text())
    if manifest.get("format") != FORMAT:
        raise SystemExit(f"{mpath}: format {manifest.get('format')!r}")
    shards = list(manifest.get("shards") or [])
    digests = manifest.get("digests")
    if not isinstance(digests, dict) or not all(
            is_sha256_hex(digests.get(s)) for s in shards):
        raise SystemExit(f"{mpath}: every shard needs a sha256 digest")
    table = pq.read_table(ppath)
    ids = [int(x) for x in table.column("id").to_pylist()]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{ppath}: duplicate ids")
    if LATENT_SHARD_COLUMN in table.column_names:
        where = table.column(LATENT_SHARD_COLUMN).to_pylist()
    else:
        where = [shard_name(t) for t in table.column("tar").to_pylist()]
    unknown = sorted({s for s in where if s not in shards})
    if unknown:
        raise SystemExit(f"{ppath}: ids live in shards the manifest does not "
                         f"list ({_examples(unknown)})")
    return {"dir": base, "manifest": manifest, "table": table, "ids": ids,
            "shard_of": dict(zip(ids, where)),
            "manifest_file": _local_digests(mpath),
            "metadata_file": _local_digests(ppath)}


def _norm(s):
    return "" if s is None else s


def validate_rows(patch, base, unapplied):
    """-> (applied rows, unapplied rows, removed ids). SystemExit listing
    every problem: ids not in the base, *_old captions that differ from the
    base (patch made against another revision), bad captions / token
    counts / buckets, unapplied rows under --unapplied fail."""
    cols = {c: base["table"].column(c).to_pylist()
            for c in CAPTIONS + ("bucket_w", "bucket_h")}
    pos = {i: j for j, i in enumerate(base["ids"])}
    buckets = set(zip(cols["bucket_w"], cols["bucket_h"]))
    problems, applied, unapplied_rows = [], [], []
    for r in sorted(patch["rows"], key=lambda r: r["id"]):
        i = r["id"]
        if i not in pos:
            problems.append(f"{i}: not in the base metadata")
            continue
        if not is_applied(r):
            unapplied_rows.append(r)
            continue
        applied.append(r)
        if r["action"] == "remove":
            continue
        j = pos[i]
        for c in CAPTIONS:
            if _norm(r[c + "_old"]) != _norm(cols[c][j]):
                problems.append(f"{i}: {c}_old differs from the base {c} "
                                "(patch made against another revision?)")
            if not isinstance(r[c], str) or not r[c].strip():
                problems.append(f"{i}: new {c} caption is empty")
        for c in TOK_COLS:
            v = r[c]
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and 0 <= v < float("inf") and int(v) == v):
                problems.append(f"{i}: {c} {v!r} is not a token count")
        if r["action"] == "caption":
            if (r["bucket_w"], r["bucket_h"]) != (cols["bucket_w"][j],
                                                  cols["bucket_h"][j]):
                problems.append(f"{i}: caption row changes the bucket")
        else:
            bp = bucket_problem(r["bucket_w"], r["bucket_h"])
            if bp:
                problems.append(f"{i}: {bp}")
            elif (int(r["bucket_w"]), int(r["bucket_h"])) not in buckets:
                problems.append(f"{i}: bucket {r['bucket_w']}x"
                                f"{r['bucket_h']} is not a dataset bucket")
    if unapplied_rows and unapplied == "fail":
        ex = [f"{r['id']}:{r['action']}/{r['status']}" for r in unapplied_rows]
        problems.append(
            f"{len(unapplied_rows)} unapplied rows (status not applied: "
            f"{_examples(ex)}); pass --unapplied keep (leave them as they "
            "are) or --unapplied remove (drop them)")
    if problems:
        _fail("patch does not apply to the base", problems)
    removed = {r["id"] for r in applied if r["action"] == "remove"}
    if unapplied == "remove":
        removed |= {r["id"] for r in unapplied_rows}
    return applied, unapplied_rows, removed


def simulate_resolution(base, new_ids, new_where, manifest, patch_keys):
    """The loader's id -> shard resolution, with base shard contents taken
    from the base metadata (keys of base shard s = base ids living in s)
    instead of shard headers. -> list of problems (empty = consistent)."""
    sup = {s: set(v) for s, v in (manifest.get("superseded") or {}).items()}
    contents = {}
    for i, s in base["shard_of"].items():
        contents.setdefault(s, set()).add(i)
    for s in base["manifest"].get("superseded") or {}:
        contents.setdefault(s, set()).update(
            base["manifest"]["superseded"][s])
    for s, keys in patch_keys.items():
        contents[s] = set(keys)
    problems = []
    live = {}
    for s in manifest["shards"]:
        keys = contents.get(s, set())
        gone = sup.get(s, set()) - keys
        if gone:
            problems.append(f"superseded ids not in {s}: {_examples(gone)}")
        for i in keys - sup.get(s, set()):
            if i in live:
                problems.append(f"{i} live in {live[i]} and {s}")
            live[i] = s
    rows = dict(zip(new_ids, new_where))
    if set(live) != set(rows):
        problems.append(f"live latents without a row: "
                        f"{_examples(sorted(set(live) - set(rows)))}; rows "
                        f"without a live latent: "
                        f"{_examples(sorted(set(rows) - set(live)))}")
    wrong = [i for i in rows if i in live and live[i] != rows[i]]
    if wrong:
        problems.append(f"{LATENT_SHARD_COLUMN} differs from the resolved "
                        f"shard for {_examples(wrong)}")
    n_keys = sum(len(contents.get(s, ())) for s in manifest["shards"])
    if manifest.get("n") is not None and manifest["n"] != n_keys:
        problems.append(f"manifest n {manifest['n']} != {n_keys} keys")
    return problems


def build_version(base_dir, patch_dir, out_dir, base_revision, latents=None,
                  unapplied="fail", link_base_shards=False,
                  lat_channels=LAT_CHANNELS, log=print):
    """Write the patched version into out_dir (see the module docstring).
    Never writes into base_dir or patch_dir. -> build report dict."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if unapplied not in UNAPPLIED_MODES:
        raise SystemExit(f"--unapplied must be one of {UNAPPLIED_MODES}")
    if not base_revision:
        raise SystemExit("--base-revision is required (the Hub commit the "
                         "base files were downloaded at)")
    base = read_base(base_dir)
    patch = read_patch(patch_dir)
    applied, unapplied_rows, removed = validate_rows(patch, base, unapplied)
    crops = {r["id"]: r for r in applied if r["action"] == "crop"}
    recaps = {r["id"]: r for r in applied if r["action"] in ("caption",
                                                             "crop")}
    man = json.loads(json.dumps(base["manifest"]))   # deep copy
    vae = man.get("vae")
    out = Path(out_dir)
    if out.resolve() in (Path(base_dir).resolve(), Path(patch_dir).resolve()):
        raise SystemExit("--out must be a new directory")
    mpath = out / "latents" / "manifest.json"
    if mpath.exists():   # written last: its presence = a finished build
        raise SystemExit(f"{out} already holds a version: pick a new --out")

    # 1. patch shard (checked against the patch images, not yet copied)
    src, patch_name, patch_keys, shard_info = None, None, {}, None
    if crops:
        items = crop_items(patch)
        src = Path(latents) if latents else patch["dir"] / PATCH_LATENTS
        if not src.is_file():
            raise SystemExit(f"{src} not found: run the encode step first "
                             "(modal run scripts/modal_patch_latents.py "
                             f"--patch-dir {patch['dir']})")
        smeta = check_patch_shard(src, items, vae, lat_channels)
        digest = file_sha256(src)
        patch_name = next_patch_shard(man["shards"])
        man["shards"].append(patch_name)
        man["digests"][patch_name] = digest
        man["n"] = int(man.get("n") or len(base["ids"])) + len(crops)
        patch_keys = {patch_name: set(crops)}
        shard_info = {"shard": patch_name, "sha256": digest,
                      "count": len(crops), "vae_impl": smeta.get("vae_impl"),
                      "images_sha256": smeta.get("images_sha256"),
                      "source": smeta.get("source")}
    elif latents:
        raise SystemExit("--latents given but the patch has no applied crop")

    # 2. superseded: removed ids and replaced (cropped) ids leave their shard
    sup = {s: set(v) for s, v in (man.get("superseded") or {}).items()}
    for i in sorted(removed | set(crops)):
        sup.setdefault(base["shard_of"][i], set()).add(i)
    if sup:
        man["superseded"] = {s: sorted(v) for s, v in sorted(sup.items())}

    # 3. metadata: base rows (base order) minus removed, patched columns
    table = base["table"]
    names = table.column_names
    data = {c: table.column(c).to_pylist() for c in names}
    keep = [j for j, i in enumerate(base["ids"]) if i not in removed]
    new_cols = {}
    for c in names:
        if c == LATENT_SHARD_COLUMN:
            continue
        vals = [data[c][j] for j in keep]
        if c in CAPTIONS + TOK_COLS + ("bucket_w", "bucket_h"):
            for k, j in enumerate(keep):
                r = recaps.get(base["ids"][j])
                if r is None or (c in ("bucket_w", "bucket_h")
                                 and r["action"] != "crop"):
                    continue
                vals[k] = int(r[c]) if c not in CAPTIONS else r[c]
        new_cols[c] = pa.array(vals, type=table.schema.field(c).type)
    new_ids = [base["ids"][j] for j in keep]
    new_where = [patch_name if i in crops else base["shard_of"][i]
                 for i in new_ids]
    new_cols[LATENT_SHARD_COLUMN] = pa.array(new_where, type=pa.string())

    # 4. manifest
    counts = {"replaced": len(crops),
              "recaptioned": sum(r["action"] == "caption" for r in applied),
              "removed": len(removed),
              "unapplied_kept": (len(unapplied_rows) if unapplied == "keep"
                                 else 0),
              "unapplied_removed": (len(unapplied_rows)
                                    if unapplied == "remove" else 0),
              "samples_before": len(base["ids"]),
              "samples_after": len(new_ids)}
    entry = {
        "review_id": patch["review_id"],
        "source_dataset_revision": patch["meta"].get("dataset_revision")
        or patch["meta"].get("source_revision"),
        "base_revision": base_revision,
        "base_manifest": base["manifest_file"],
        "base_metadata": base["metadata_file"],
        "patch_parquet_sha256": patch["parquet_sha256"],
        "patch_json_sha256": patch["json_sha256"],
        "unapplied": unapplied,
        "counts": counts,
        "shard": shard_info,
        "tool": "scripts/modal_patch_latents.py build",
    }
    man["patches"] = list(man.get("patches") or []) + [entry]
    if not new_ids:
        raise SystemExit("the patch would remove every sample: refusing to "
                         "build an empty version (nothing written)")
    problems = simulate_resolution(base, new_ids, new_where, man, patch_keys)
    if problems:
        _fail("internal consistency check FAILED (nothing written)", problems)

    # write: shard, parquet, then the manifest (the commit point)
    lat_out = out / "latents"
    lat_out.mkdir(parents=True, exist_ok=True)
    if src is not None:
        dst = lat_out / patch_name
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        if file_sha256(dst) != shard_info["sha256"]:
            raise SystemExit(f"{src} changed while being copied")
        log(f"patch shard {patch_name}: {len(crops)} latents, sha256 "
            f"{shard_info['sha256'][:16]}..")
    ppath = out / "metadata.parquet"
    tmp = ppath.with_name(ppath.name + ".tmp")
    pq.write_table(pa.table(new_cols), tmp)
    os.replace(tmp, ppath)
    entry["metadata"] = _local_digests(ppath)   # checked before upload
    tmp = mpath.with_name(mpath.name + ".tmp")
    tmp.write_text(json.dumps(man, indent=2))
    os.replace(tmp, mpath)

    # 5. optional full check with the training loader
    report = {"out": str(out), "manifest": str(mpath), "counts": counts,
              "patch_shard": shard_info,
              "unapplied": [{"id": r["id"], "action": r["action"],
                             "status": r["status"]} for r in unapplied_rows],
              "removed_ids": sorted(removed),
              "replaced_ids": sorted(crops),
              "recaptioned_ids": sorted(i for i, r in recaps.items()
                                        if r["action"] == "caption"),
              "loader_check": None}
    if link_base_shards:
        report["loader_check"] = loader_check(out, base["dir"],
                                              base["manifest"]["shards"],
                                              log=log)
    (out / "build_report.json").write_text(json.dumps(report, indent=2))
    log(f"built {out}: {counts}")
    return report


def loader_check(out, base_dir, base_shards, log=print):
    """Symlink the base shards into out/latents and load out with
    BucketDataset (all shard headers + every digest re-hashed)."""
    from krea2_jax.data import BucketDataset

    lat = Path(out) / "latents"
    absent = [s for s in base_shards
              if not (Path(base_dir) / "latents" / s).is_file()]
    if absent:
        raise SystemExit(f"--link-base-shards: {len(absent)} base shards are "
                         f"not on disk ({_examples(absent)})")
    for s in base_shards:
        link = lat / s
        if not link.exists():
            os.symlink((Path(base_dir) / "latents" / s).resolve(), link)
    t0 = time.time()
    ds = BucketDataset(out, template=None, img_tokens=MAX_IMG_TOKENS,
                       patch=PATCH, verify_digests=True)
    res = {"n": ds.n, "verified_shards": ds.verified_shards,
           "verified_bytes": ds.verified_bytes,
           "seconds": time.time() - t0}
    log(f"BucketDataset check OK: {res}")
    return res


# ------------------------------------------------------------------ upload
def remote_entries(files):
    """list_repo_tree items -> {path: {size, sha256 (LFS) or None,
    blob_id}} (folders skipped)."""
    out = {}
    for f in files:
        if getattr(f, "blob_id", None) is None:
            continue
        lfs = getattr(f, "lfs", None)
        out[f.path] = {"size": f.size, "blob_id": f.blob_id,
                       "sha256": getattr(lfs, "sha256", None) if lfs else None}
    return out


def _content_problems(path, want, remote):
    r = remote.get(path)
    if r is None:
        return [f"missing {path}"]
    if want.get("size") is not None and r["size"] != want["size"]:
        return [f"size {path}: remote {r['size']} != {want['size']}"]
    if r.get("sha256"):
        if r["sha256"] != want.get("sha256"):
            return [f"sha256 {path}: remote {r['sha256']} != "
                    f"{want.get('sha256')}"]
        return []
    if want.get("git_sha1"):
        if r["blob_id"] != want["git_sha1"]:
            return [f"git blob {path}: remote {r['blob_id']} != "
                    f"{want['git_sha1']}"]
        return []
    return [f"unverifiable {path}: no LFS sha256 on the remote"]


def expected_files(new_dir):
    """Check the local files against the manifest, then -> ({repo path:
    {size?, sha256, git_sha1?}} the branch must hold, manifest).

    The manifest is the root of trust: the patch shard must hash to its
    manifest digest and metadata.parquet to the digests the build recorded
    in the manifest's last "patches" entry; any mismatch (a file modified
    after build) is refused before anything is uploaded. Expected remote
    content comes from the manifest, never from re-hashing local files."""
    new = Path(new_dir)
    mpath = new / "latents" / "manifest.json"
    man = json.loads(mpath.read_text())
    entry = man["patches"][-1]
    shard = (entry.get("shard") or {}).get("shard")
    want_meta = entry.get("metadata")
    problems = []
    if not want_meta:
        problems.append("manifest has no metadata digests (built by an "
                        "older version of this script: rebuild)")
    else:
        got = _local_digests(new / "metadata.parquet")
        if got["sha256"] != want_meta["sha256"]:
            problems.append(f"metadata.parquet sha256 {got['sha256']} != "
                            f"{want_meta['sha256']} recorded at build")
    if shard:
        got = file_sha256(new / "latents" / shard)
        if got != man["digests"][shard]:
            problems.append(f"latents/{shard} sha256 {got} != manifest "
                            f"digest {man['digests'][shard]}")
    if problems:
        _fail(f"{new}: local files differ from the manifest (modified after "
              "build?); nothing uploaded", problems)
    exp = {"metadata.parquet": dict(want_meta),
           "latents/manifest.json": _local_digests(mpath)}
    for s in man["shards"]:
        exp[f"latents/{s}"] = {"sha256": man["digests"][s]}
    if shard:
        exp[f"latents/{shard}"]["size"] = (new / "latents" / shard).stat(
            ).st_size
    return exp, man


def base_problems(man, remote):
    """The base revision on the Hub must be exactly what was built on."""
    entry = man["patches"][-1]
    probs = (_content_problems("latents/manifest.json", entry["base_manifest"],
                               remote)
             + _content_problems("metadata.parquet", entry["base_metadata"],
                                 remote))
    shard = (entry.get("shard") or {}).get("shard")
    for s in man["shards"]:
        if s != shard:
            probs += _content_problems(f"latents/{s}",
                                       {"sha256": man["digests"][s]}, remote)
    return probs


def branch_problems(exp, written, remote):
    """An existing branch may only receive this commit if every file it
    keeps (all expected files except the `written` ones) already has the
    expected content and it holds no shard the new manifest does not list
    (another base / another patch)."""
    probs = []
    for k, w in sorted(exp.items()):
        if k not in written:
            probs += _content_problems(k, w, remote)
    return probs + unlisted_shards(exp, remote)


def unlisted_shards(exp, remote):
    """-> [problem] if `remote` holds a latents/*.safetensors the new
    manifest does not list (another base / another patch; a node download
    of the branch would carry it). Checked on the base, on an existing
    branch head and on the final head: no success path skips it."""
    extra = sorted(k for k in remote if k.startswith("latents/")
                   and k.endswith(".safetensors") and k not in exp)
    if extra:
        return [f"{len(extra)} shards not in the new manifest "
                f"({_examples(extra)})"]
    return []


def upload_version(new_dir, branch, repo=LATENTS_REPO, api=None,
                   allow_main=False, make_op=None, log=print):
    """Create `branch` at the base revision (if absent), commit
    metadata.parquet, latents/manifest.json and the patch shard to it (with
    parent_commit = the head that was checked), then check every file of
    the new manifest at the new head. Idempotent: a branch that already
    holds exactly this version is left alone; an existing branch whose
    other files do not fit this version is refused untouched."""
    if not branch:
        raise SystemExit("--branch is required")
    if branch == "main" and not allow_main:
        raise SystemExit(
            "refusing to commit to main: training nodes download main and a "
            "run resumed on the old version would see a different "
            "dataset_fingerprint. Use a branch (and --revision <branch> on "
            "download), or pass --allow-main deliberately.")
    new = Path(new_dir)
    exp, man = expected_files(new)   # local files vs manifest, first
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    if make_op is None:
        from huggingface_hub import CommitOperationAdd

        def make_op(path_in_repo, local):
            return CommitOperationAdd(path_in_repo=path_in_repo,
                                      path_or_fileobj=str(local))
    entry = man["patches"][-1]
    base_rev = entry["base_revision"]
    files = ["metadata.parquet", "latents/manifest.json"]
    shard = (entry.get("shard") or {}).get("shard")
    if shard:
        files.append(f"latents/{shard}")

    def tree(rev):
        return remote_entries(api.list_repo_tree(
            repo, repo_type="dataset", recursive=True, expand=True,
            revision=rev))

    def check(remote):   # every success path goes through this
        return [p for k, w in sorted(exp.items())
                for p in _content_problems(k, w, remote)] + unlisted_shards(
                    exp, remote)

    info = api.repo_info(repo, repo_type="dataset", revision=base_rev)
    if getattr(info, "private", None) is not True:
        raise SystemExit(f"{repo} is not private (private="
                         f"{getattr(info, 'private', None)!r}): not uploading")
    # the base may not hold any shard outside the base manifest (incl. a
    # file at the patch shard's path): a new branch would inherit it
    base_exp = {k: w for k, w in exp.items()
                if not (shard and k == f"latents/{shard}")}
    base_tree = tree(base_rev)
    probs = (base_problems(man, base_tree)
             + unlisted_shards(base_exp, base_tree))
    if probs:
        _fail(f"{repo}@{base_rev} is not the base this version was built on",
              probs)
    log(f"base {repo}@{base_rev[:12]} matches ({len(man['shards'])} shards)")
    refs = api.list_repo_refs(repo, repo_type="dataset")
    branches = {b.name: b.target_commit for b in refs.branches}
    if branch in branches:
        head = branches[branch]
        remote = tree(head)
        if not check(remote):
            log(f"{repo}@{branch} ({head[:12]}) already holds this version")
            return {"repo": repo, "branch": branch, "revision": head,
                    "uploaded": False, "problems": []}
        probs = branch_problems(exp, set(files), remote)
        if probs:
            _fail(f"branch {branch} ({head}) exists with content that does "
                  "not fit this version (nothing committed; use a new "
                  "--branch)", probs)
        log(f"branch {branch} exists at {head[:12]}: committing on top")
    else:
        api.create_branch(repo, branch=branch, revision=base_rev,
                          repo_type="dataset")
        head = base_rev
        log(f"created branch {branch} at {base_rev[:12]}")
    ops = [make_op(p, new / p) for p in files]
    c = entry["counts"]
    api.create_commit(
        repo, ops, repo_type="dataset", revision=branch, parent_commit=head,
        commit_message=(f"datafilter patch {entry['review_id']}: "
                        f"{c['replaced']} re-cropped, {c['recaptioned']} "
                        f"recaptioned, {c['removed']} removed"),
        commit_description=json.dumps(entry, indent=1))
    head = api.repo_info(repo, repo_type="dataset", revision=branch).sha
    probs = check(tree(head))
    if probs:
        _fail(f"remote check of {repo}@{branch} ({head}) FAILED", probs)
    log(f"uploaded {files} to {repo}@{branch} = {head}; all "
        f"{len(exp)} files verified")
    return {"repo": repo, "branch": branch, "revision": head,
            "uploaded": True, "problems": []}


# ------------------------------------------------------- Modal (stdlib side)
def patch_upload_list(patch_dir):
    """Files the Modal step uploads -> sorted [(relpath, Path)]: patch.parquet,
    patch.json and every images/*.webp (nothing else: reports stay local)."""
    d = Path(patch_dir)
    out = []
    for f in PATCH_FILES:
        if not (d / f).is_file():
            raise SystemExit(f"{d / f} not found")
        out.append((f, d / f))
    img = d / "images"
    if img.is_dir():
        out += [(f"images/{p.name}", p) for p in sorted(img.iterdir())
                if p.is_file() and p.suffix == ".webp"]
    return out


def upload_digests(files):
    """{relpath: sha256} and the content key (16 hex) of a patch upload."""
    digs = {rel: file_sha256(p) for rel, p in files}
    key = hashlib.sha256(json.dumps(sorted(digs.items())).encode()
                         ).hexdigest()[:16]
    return digs, key


def encode_remote_impl(key, digests, vae_impl, batch, decode_threads, force,
                       vol, root=VOL_PATCHES, loader=load_encoder, log=print):
    """Container side: check the uploaded patch files (sha256), encode the
    applied crops into <root>/<key>/patch_latents.safetensors (skipped when
    a shard of the same images and impl is already there), commit."""
    import subprocess

    d = Path(root) / key
    try:
        vol.reload()
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: volume reload failed: {e}")
    bad = [rel for rel, want in sorted(digests.items())
           if not (d / rel).is_file() or file_sha256(d / rel) != want]
    stray = sorted(str(p.relative_to(d)) for p in (d / "images").glob("*")
                   if str(p.relative_to(d)) not in digests) \
        if (d / "images").is_dir() else []
    if bad or stray:
        raise RuntimeError(f"uploaded patch files differ: {len(bad)} bad "
                           f"({bad[:5]}), {len(stray)} stray ({stray[:5]})")
    out = d / PATCH_LATENTS
    patch = read_patch(d)
    items = crop_items(patch)
    digest = images_digest(items)
    res = {"key": key, "n": len(items), "skipped": False}
    if out.is_file() and not force:
        meta, keys = read_shard_header(out)
        if (meta.get("images_sha256") == digest
                and meta.get("vae_impl") == vae_impl
                and len(keys) == len(items)):
            res.update(skipped=True, sha256=file_sha256(out),
                       size=out.stat().st_size)
            return res
    try:
        res["device_name"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        res["device_name"] = f"n/a ({e})"
    t0 = time.time()
    encoder, device = loader(vae_impl)
    res["load_s"] = time.time() - t0
    st = encode_items(items, encoder, out, vae_id=VAE_REPO, vae_impl=vae_impl,
                      source=f"patch:{patch['review_id']}", batch=batch,
                      decode_threads=decode_threads, log=log)
    vol.commit()
    res.update(st, size=out.stat().st_size, device=device)
    return res


if modal is not None:
    import modal_cache_latents as mcl

    app = modal.App("krea2-patch-latents", image=mcl.image)

    # same image / volume (VAE weights in /vol/hf) / secret as the cache
    # build; GPU only (no cpu=/memory= reservations: a few hundred images)
    @app.function(gpu=DEFAULT_GPU, timeout=ENCODE_TIMEOUT_S,
                  volumes={mcl.VOL: mcl.vol}, secrets=[mcl.hf_secret])
    def encode_remote(key: str, digests: dict, vae_impl: str, batch: int,
                      decode_threads: int, force: bool) -> dict:
        mcl.setup_hf_token()
        return encode_remote_impl(key, digests, vae_impl, batch,
                                  decode_threads, force, mcl.vol)

    @app.local_entrypoint()
    def main(patch_dir: str, out: str = "", gpu: str = DEFAULT_GPU,
             vae_impl: str = DEFAULT_VAE_IMPL, batch: int = 16,
             decode_threads: int = 4, force: bool = False):
        if vae_impl not in VAE_IMPLS:
            raise SystemExit(f"--vae-impl must be one of {VAE_IMPLS}")
        files = patch_upload_list(patch_dir)
        digests, key = upload_digests(files)
        n_img = sum(rel.startswith("images/") for rel in digests)
        if not n_img:
            raise SystemExit(f"{patch_dir}/images has no .webp: nothing to "
                             "encode (build needs no --latents)")
        dst = Path(out) if out else Path(patch_dir) / PATCH_LATENTS
        print(f"uploading {len(files)} files ({n_img} images) to volume "
              f"{mcl.VOLUME_NAME}:/patches/{key}")
        with mcl.vol.batch_upload(force=True) as up:
            for rel, p in files:
                up.put_file(str(p), f"/patches/{key}/{rel}")
        t0 = time.time()
        res = encode_remote.with_options(gpu=gpu).remote(
            key, digests, vae_impl, batch, decode_threads, force)
        print(f"encode: {json.dumps(res)} ({time.time() - t0:.0f}s wall)")
        tmp = dst.with_name(dst.name + ".part")
        with open(tmp, "wb") as f:
            for chunk in mcl.vol.read_file(f"/patches/{key}/{PATCH_LATENTS}"):
                f.write(chunk)
        got = file_sha256(tmp)
        if got != res["sha256"]:
            raise SystemExit(f"downloaded shard sha256 {got} != "
                             f"{res['sha256']} (kept {tmp})")
        os.replace(tmp, dst)
        print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB, {res['n']} "
              f"latents). Next: python scripts/modal_patch_latents.py build "
              f"--base BASE --base-revision SHA --patch-dir {patch_dir} "
              "--out NEW")


# --------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(
        description="Data-filter patch -> new krea2-latents-v1 version "
                    "(encode locally / build / upload). The Modal encode is "
                    "`modal run scripts/modal_patch_latents.py --patch-dir P`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="encode the applied crops locally")
    e.add_argument("--patch-dir", required=True)
    e.add_argument("--out", default=None,
                   help=f"default: <patch-dir>/{PATCH_LATENTS}")
    e.add_argument("--vae-impl", choices=VAE_IMPLS, default=DEFAULT_VAE_IMPL)
    e.add_argument("--device", default=None)
    e.add_argument("--dtype", choices=["auto", "bf16", "fp32"],
                   default="auto")
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--decode-threads", type=int, default=4)
    e.add_argument("--vae-stub", action="store_true",
                   help=argparse.SUPPRESS)  # tests: cache_latents.StubEncoder
    b = sub.add_parser("build", help="write the new version (local, CPU)")
    b.add_argument("--base", required=True,
                   help="dir with the base revision's metadata.parquet and "
                        "latents/manifest.json (shards optional)")
    b.add_argument("--base-revision", required=True,
                   help="Hub commit sha of the base files (recorded; the "
                        "upload branches from it)")
    b.add_argument("--patch-dir", required=True)
    b.add_argument("--latents", default=None,
                   help=f"encoded patch shard; default <patch-dir>/"
                        f"{PATCH_LATENTS}")
    b.add_argument("--out", required=True, help="new, empty output dir")
    b.add_argument("--unapplied", choices=UNAPPLIED_MODES, default="fail",
                   help="rows whose status is not applied: fail, keep them "
                        "unchanged, or remove the samples")
    b.add_argument("--link-base-shards", action="store_true",
                   help="symlink the base shards (must be on disk) into "
                        "<out>/latents and load <out> with BucketDataset "
                        "(full header + digest verification)")
    u = sub.add_parser("upload", help="commit the new version to a branch "
                                      "of the HF repo (token from the HF "
                                      "cache / HF_TOKEN)")
    u.add_argument("--new", required=True, help="build --out dir")
    u.add_argument("--branch", required=True)
    u.add_argument("--repo", default=LATENTS_REPO)
    u.add_argument("--allow-main", action="store_true")
    return ap


def cli(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "encode":
        if args.vae_stub:
            import cache_latents as cl
            encoder, vae_id, impl = cl.StubEncoder(), cl.STUB_VAE_ID, "stub"
        else:
            encoder, _ = load_encoder(args.vae_impl, args.device, args.dtype)
            vae_id, impl = VAE_REPO, args.vae_impl
        out = (Path(args.out) if args.out
               else Path(args.patch_dir) / PATCH_LATENTS)
        res = encode_patch_dir(args.patch_dir, out, encoder, vae_id=vae_id,
                               vae_impl=impl, batch=args.batch,
                               decode_threads=args.decode_threads)
        print(f"wrote {out}: {json.dumps(res)}")
        return res
    if args.cmd == "build":
        return build_version(args.base, args.patch_dir, args.out,
                             args.base_revision, latents=args.latents,
                             unapplied=args.unapplied,
                             link_base_shards=args.link_base_shards)
    return upload_version(args.new, args.branch, repo=args.repo,
                          allow_main=args.allow_main)


if __name__ == "__main__":
    cli()
