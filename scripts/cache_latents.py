"""Build the `krea2-latents-v1` VAE latent cache for the bucketed full
fine-tuning path (read by krea2_jax/data.py::BucketDataset).

Input: the HF dataset (default ricecake/danbooru-curated-v2: data/train-XXXX.tar
with `{id}.webp` + `{id}.json` (+ `{id}.txt`, ignored) per sample, and
metadata.parquet), or a local directory with the same layout (--tar-dir).
Images are pre-cropped to their aspect bucket; each is VAE-encoded at
(bucket_w, bucket_h) and stored normalized ((z - latents_mean) / latents_std,
exactly like cache_dataset.VaeEncoder) as bf16.

Output (<out>):
    metadata.parquet                    verbatim copy (--write-manifest pass)
    latents/manifest.json               {"format", "vae", "n", "shards", "digests"} (--write-manifest pass)
    latents/train-XXXX.safetensors      one per source tar; "<id>" -> (16, bucket_h/8, bucket_w/8)
                                        bf16; metadata {format, vae, source, count}

Shard files are written atomically (tmp + os.replace); a shard whose output
already opens with the right format/vae/source/count is skipped, so any run
can simply be restarted. Downloaded tars are deleted once their shard is
written (node disks are small and ephemeral) unless --keep-tars.

The --write-manifest pass also records the sha256 of every shard file
("digests": {"train-0000.safetensors": "<hex>", ...}); the trainer's dataset
fingerprint includes them, so a rebuilt cache (the VAE posterior sample is
unseeded) is refused on --resume unless its bytes are identical. Cost: every
shard is read once (~49 GB for the full dataset; streaming, 1 MB chunks).
When the existing manifest already has digests, the pass re-hashes and fails
on any mismatch (--no-verify-digests accepts the new content).

Usage (one process per GPU, then one manifest pass):
    for i in $(seq 0 7); do
      HIP_VISIBLE_DEVICES=$i python scripts/cache_latents.py --out /data/krea2 \
          --num-workers 8 --worker-index $i > cache_w$i.log 2>&1 &
    done; wait
    python scripts/cache_latents.py --out /data/krea2 --write-manifest
"""

import argparse
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_dataset import VaeEncoder, load_image  # noqa: E402
from krea2_jax.hashing import file_sha256, is_sha256_hex  # noqa: E402

FORMAT = "krea2-latents-v1"
LAT_CHANNELS = 16
VAE_SCALE = 8
BUCKET_MULTIPLE = 16  # VAE f8 x patch 2
IMAGE_EXTS = {"webp", "png", "jpg", "jpeg"}
STUB_VAE_ID = "stub"
_SHARD_RE = re.compile(r"train-(\d+)\.tar$")


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------
@torch.no_grad()
def vae_encode_batch(vae: VaeEncoder, imgs: torch.Tensor) -> torch.Tensor:
    """Batched `VaeEncoder.__call__`: (N, 3, H, W) in [-1, 1] -> normalized
    latents (N, 16, H/8, W/8). Same ops and normalization, batch kept."""
    x = imgs.to(vae.device, vae.dtype).unsqueeze(2)  # (n, c, 1, h, w)
    z = vae.vae.encode(x).latent_dist.sample()
    z = (z - vae.mean) / vae.std
    return z[:, :, 0]


class StubEncoder:
    """Deterministic stand-in for the VAE (tests / --vae-stub): 8x8 average
    pool, then a fixed-seed 3 -> 16 channel map. Records input shapes."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(LAT_CHANNELS, 3, generator=g)
        self.calls = []

    @torch.no_grad()
    def __call__(self, imgs):
        self.calls.append(tuple(imgs.shape))
        pooled = torch.nn.functional.avg_pool2d(imgs.float(), VAE_SCALE)
        return torch.einsum("oc,nchw->nohw", self.weight, pooled)


# ---------------------------------------------------------------------------
# shard processing
# ---------------------------------------------------------------------------
def shard_name(tar):
    """'data/train-0000.tar' -> 'train-0000.safetensors'."""
    return Path(tar).stem + ".safetensors"


def shard_number(tar):
    m = _SHARD_RE.search(str(tar))
    if not m:
        raise SystemExit(f"cannot parse a shard number from {tar!r}")
    return int(m.group(1))


def parse_shards(spec):
    """'0-99' / '0,3,5-7' -> sorted list of ints."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                raise SystemExit(f"bad shard range {part!r}")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    if not out:
        raise SystemExit(f"empty --shards {spec!r}")
    return sorted(out)


def shard_done(out_path, vae_id, source, expected_count):
    """True when `out_path` is a complete shard for exactly this job."""
    try:
        with safe_open(str(out_path), framework="numpy") as f:
            meta = f.metadata() or {}
            n_keys = len(list(f.keys()))
    except Exception:
        return False
    return (meta.get("format") == FORMAT and meta.get("vae") == vae_id
            and meta.get("source") == source
            and meta.get("count") == str(expected_count)
            and n_keys == expected_count)


def read_tar(tar_path):
    """Sequentially read a dataset tar -> list of samples
    {id, data (encoded image bytes), w, h} in tar order."""
    images, metas, order = {}, {}, []
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, _, ext = Path(m.name).name.partition(".")
            ext = ext.lower()
            if ext not in IMAGE_EXTS and ext != "json":
                continue  # .txt captions etc. live in the parquet
            data = tf.extractfile(m).read()
            if ext == "json":
                if stem in metas:
                    raise ValueError(f"{tar_path}: duplicate {m.name}")
                metas[stem] = json.loads(data)
            else:
                if stem in images:
                    raise ValueError(f"{tar_path}: duplicate image for {stem}")
                images[stem] = data
                order.append(stem)
    if set(images) != set(metas):
        no_json = sorted(set(images) - set(metas))
        no_img = sorted(set(metas) - set(images))
        raise ValueError(f"{tar_path}: {len(no_json)} images without json "
                         f"{no_json[:5]}, {len(no_img)} json without image "
                         f"{no_img[:5]}")
    samples = []
    for stem in order:
        meta = metas[stem]
        sid, w, h = int(meta["id"]), int(meta["bucket_w"]), int(meta["bucket_h"])
        if str(sid) != stem:
            raise ValueError(f"{tar_path}: member {stem!r} has json id {sid}")
        if w <= 0 or h <= 0 or w % BUCKET_MULTIPLE or h % BUCKET_MULTIPLE:
            raise ValueError(f"{tar_path}: id {sid} bucket {w}x{h} is not a "
                             f"positive multiple of {BUCKET_MULTIPLE}")
        samples.append({"id": sid, "data": images[stem], "w": w, "h": h})
    return samples


def decode(sample):
    """-> ((3, h, w) float32 in [-1, 1], resized?). An image whose size is not
    its bucket is cover-resized + center-cropped (cache_dataset.load_image)."""
    with Image.open(io.BytesIO(sample["data"])) as im:
        size = im.size  # header only
    img = load_image(io.BytesIO(sample["data"]), sample["w"], sample["h"])
    return img[0], size != (sample["w"], sample["h"])


def encode_samples(samples, encoder, batch, pool, lookahead=2, log=print,
                   tag=""):
    """Encode samples grouped by bucket, `batch` at a time, decoding up to
    `lookahead` batches ahead on `pool`. -> ({str(id): (16, h/8, w/8) bf16
    CPU tensor}, number of resized images)."""
    groups = defaultdict(list)
    for s in samples:
        groups[(s["w"], s["h"])].append(s)
    chunks = [g[i:i + batch] for key in sorted(groups)
              for g in (groups[key],) for i in range(0, len(g), batch)]
    pending, nxt = deque(), 0
    out, resized = {}, 0
    for j in range(len(chunks)):
        while nxt < len(chunks) and nxt <= j + lookahead:
            pending.append([pool.submit(decode, s) for s in chunks[nxt]])
            nxt += 1
        decoded = [f.result() for f in pending.popleft()]
        chunk = chunks[j]
        for s, (_, was_resized) in zip(chunk, decoded):
            if was_resized:
                resized += 1
                log(f"WARNING {tag}: id {s['id']} is not {s['w']}x{s['h']}; "
                    "cover-resized + center-cropped")
        imgs = torch.stack([d[0] for d in decoded])
        lat = encoder(imgs)
        w, h = chunk[0]["w"], chunk[0]["h"]
        want = (len(chunk), LAT_CHANNELS, h // VAE_SCALE, w // VAE_SCALE)
        if tuple(lat.shape) != want:
            raise ValueError(f"{tag}: encoder returned {tuple(lat.shape)}, "
                             f"expected {want}")
        if not torch.isfinite(lat).all():
            bad = [s["id"] for k, s in enumerate(chunk)
                   if not torch.isfinite(lat[k]).all()]
            raise ValueError(f"{tag}: non-finite latents for ids {bad}")
        lat = lat.detach().to(torch.bfloat16).cpu()
        for k, s in enumerate(chunk):
            out[str(s["id"])] = lat[k].clone()  # own storage (safetensors)
    return out, resized


def process_shard(tar_path, out_path, encoder, batch=8, *, vae_id, source,
                  expected_count=None, decode_threads=4, log=print, tag=None):
    """Encode one dataset tar into one `krea2-latents-v1` shard file.

    `encoder(imgs (N, 3, H, W) float32 in [-1, 1]) -> (N, 16, H/8, W/8)`.
    Skips the work when `out_path` is already complete for this job (with
    `expected_count` given this check needs no tar). Writes
    `<out_path>.tmp` then os.replace. Returns stats
    {skipped, n, resized, seconds}.
    """
    out_path = Path(out_path)
    tag = tag or out_path.stem
    if expected_count is not None and shard_done(out_path, vae_id, source,
                                                 expected_count):
        return {"skipped": True, "n": expected_count, "resized": 0,
                "seconds": 0.0}
    t0 = time.time()
    samples = read_tar(tar_path)
    if expected_count is not None and len(samples) != expected_count:
        raise ValueError(f"{tar_path}: {len(samples)} samples, metadata "
                         f"expects {expected_count}")
    if expected_count is None and shard_done(out_path, vae_id, source,
                                             len(samples)):
        return {"skipped": True, "n": len(samples), "resized": 0,
                "seconds": 0.0}
    with ThreadPoolExecutor(decode_threads,
                            thread_name_prefix="cache_latents_decode") as pool:
        tensors, resized = encode_samples(samples, encoder, batch, pool,
                                          log=log, tag=tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    save_file(tensors, str(tmp), metadata={
        "format": FORMAT, "vae": vae_id, "source": source,
        "count": str(len(tensors))})
    os.replace(tmp, out_path)
    return {"skipped": False, "n": len(tensors), "resized": resized,
            "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# metadata / manifest
# ---------------------------------------------------------------------------
def read_metadata(path):
    """-> dict of python lists: id, bucket_w, bucket_h, tar (parquet order)."""
    import pyarrow.parquet as pq

    t = pq.read_table(str(path), columns=["id", "bucket_w", "bucket_h", "tar"])
    cols = {k: t.column(k).to_pylist() for k in t.column_names}
    dup = [i for i, c in Counter(cols["id"]).items() if c > 1]
    if dup:
        raise SystemExit(f"{path}: {len(dup)} duplicate ids, e.g. {dup[:5]}")
    return cols


def scan_shards(lat_dir, names):
    """Header-only scan -> {name: {"meta": dict, "keys": {key: (shape,
    dtype)}}} or {name: "<error string>"} for missing / unreadable files."""
    out = {}
    for name in names:
        path = Path(lat_dir) / name
        if not path.exists():
            out[name] = "missing file"
            continue
        try:
            with safe_open(str(path), framework="numpy") as f:
                meta = dict(f.metadata() or {})
                keys = {}
                for k in f.keys():
                    s = f.get_slice(k)
                    keys[k] = (tuple(s.get_shape()), s.get_dtype())
        except Exception as e:  # noqa: BLE001 - report, don't crash
            out[name] = f"unreadable ({type(e).__name__}: {e})"
            continue
        out[name] = {"meta": meta, "keys": keys}
    return out


def verify_latents(cols, scan):
    """Check a scan against the metadata columns. Every parquet id must have
    exactly one latent, of shape (16, bucket_h/8, bucket_w/8), bf16, in the
    shard named after its tar; shard metadata must be consistent. Raises
    SystemExit listing every problem class with counts; returns (n, vae)."""
    problems = defaultdict(list)
    expect = {}
    for sid, w, h, tar in zip(cols["id"], cols["bucket_w"], cols["bucket_h"],
                              cols["tar"]):
        expect[sid] = ((LAT_CHANNELS, h // VAE_SCALE, w // VAE_SCALE),
                       shard_name(tar), tar)
    sources = {shard_name(t): t for t in cols["tar"]}
    seen = {}
    vaes = set()
    for name in sorted(scan):
        entry = scan[name]
        if isinstance(entry, str):
            problems["missing/unreadable shard"].append(f"{name}: {entry}")
            continue
        meta, keys = entry["meta"], entry["keys"]
        if meta.get("format") != FORMAT:
            problems["wrong format"].append(f"{name}: {meta.get('format')!r}")
        if meta.get("source") != sources.get(name):
            problems["wrong source"].append(
                f"{name}: {meta.get('source')!r} != {sources.get(name)!r}")
        if meta.get("count") != str(len(keys)):
            problems["count mismatch"].append(
                f"{name}: count {meta.get('count')!r}, {len(keys)} tensors")
        vaes.add(meta.get("vae"))
        for k, (shape, dtype) in keys.items():
            try:
                sid = int(k)
            except ValueError:
                problems["non-id key"].append(f"{name}: {k!r}")
                continue
            if str(sid) != k:
                problems["non-canonical key"].append(f"{name}: {k!r}")
            if sid in seen:
                problems["duplicate latent"].append(
                    f"{sid} in {seen[sid]} and {name}")
                continue
            seen[sid] = name
            if sid not in expect:
                problems["latent without parquet row"].append(f"{name}: {sid}")
                continue
            want_shape, want_shard, _ = expect[sid]
            if shape != want_shape:
                problems["wrong shape"].append(
                    f"{sid}: {shape} != {want_shape}")
            if dtype != "BF16":
                problems["wrong dtype"].append(f"{sid}: {dtype}")
            if name != want_shard:
                problems["latent in wrong shard"].append(
                    f"{sid}: in {name}, parquet tar -> {want_shard}")
    missing = [sid for sid in expect if sid not in seen]
    for sid in missing:
        problems["parquet id without latent"].append(
            f"{sid} ({expect[sid][2]})")
    if len(vaes) > 1:
        problems["mixed vae ids"].append(str(sorted(map(str, vaes))))
    if problems:
        lines = [f"  {k}: {len(v)}, e.g. {v[:5]}" for k, v in problems.items()]
        raise SystemExit("latent cache verification FAILED:\n"
                         + "\n".join(lines))
    vae = vaes.pop() if vaes else None
    return len(seen), vae


def shard_digests(lat_dir, names, log=print):
    """{name: sha256 of latents/<name>} (reads every shard once)."""
    out, t0, nbytes = {}, time.time(), 0
    for k, name in enumerate(names):
        path = Path(lat_dir) / name
        out[name] = file_sha256(path)
        nbytes += path.stat().st_size
        if (k + 1) % 10 == 0 or k + 1 == len(names):
            dt = max(time.time() - t0, 1e-9)
            log(f"hashed {k + 1}/{len(names)} shards ({nbytes / 1e9:.1f} GB, "
                f"{nbytes / 1e6 / dt:.0f} MB/s)")
    return out


def check_digests(old, new):
    """Raise SystemExit if any digest in `old` or `new` is not a 64-char
    lowercase hex sha256 string (e.g. null, upper case, truncated), or if a
    shard in both has a different digest. Returns (n_compared, added names,
    removed names)."""
    bad = sorted(f"{which}:{k}={v!r}" for which, d in (("existing", old),
                                                       ("new", new))
                 for k, v in d.items() if not is_sha256_hex(v))
    if bad:
        raise SystemExit(
            f"digest verification FAILED: {len(bad)} malformed digests "
            f"(not 64-char lowercase hex sha256), e.g. {bad[:5]}. Pass "
            "--no-verify-digests to replace them with freshly computed ones.")
    changed = sorted(k for k in old.keys() & new.keys() if old[k] != new[k])
    if changed:
        raise SystemExit(
            f"digest verification FAILED: {len(changed)} shards differ from "
            f"the existing manifest, e.g. {changed[:5]} (rebuilt or corrupted "
            "latents: a run resumed on them would see different data). Pass "
            "--no-verify-digests to accept the new content.")
    return (len(old.keys() & new.keys()), sorted(new.keys() - old.keys()),
            sorted(old.keys() - new.keys()))


def write_manifest(out_dir, parquet_src, log=print, verify_digests=None):
    """Verify every shard against `parquet_src`, hash every shard file
    (sha256), then copy the parquet to out/metadata.parquet and write
    out/latents/manifest.json (atomic) with the digests.

    verify_digests: None (default) = verify when the existing manifest has
    digests; True = verify (an existing manifest without digests is only
    reported); False = skip. Verification fails (SystemExit) when a shard's
    new digest differs from the recorded one; the manifest is then left
    untouched."""
    out_dir = Path(out_dir)
    lat_dir = out_dir / "latents"
    mpath = lat_dir / "manifest.json"
    cols = read_metadata(parquet_src)
    names = sorted({shard_name(t) for t in cols["tar"]})
    t0 = time.time()
    scan = scan_shards(lat_dir, names)
    n, vae = verify_latents(cols, scan)
    old_digests = None
    if mpath.exists():
        try:
            old_digests = json.loads(mpath.read_text()).get("digests")
        except (OSError, ValueError, AttributeError):
            old_digests = None
    digests = shard_digests(lat_dir, names, log=log)
    if verify_digests is None:
        verify_digests = isinstance(old_digests, dict)
    if verify_digests:
        if isinstance(old_digests, dict):
            same, added, removed = check_digests(old_digests, digests)
            log(f"digests verified: {same} shards match the existing "
                f"manifest" + (f"; {len(added)} new ({added[:5]})"
                               if added else "")
                + (f"; {len(removed)} no longer listed ({removed[:5]})"
                   if removed else ""))
        else:
            log("no digests in an existing manifest: nothing to verify")
    elif isinstance(old_digests, dict):
        log("WARNING: --no-verify-digests: existing digests are replaced "
            "without comparison")
    stray = sorted(p.name for p in lat_dir.glob("*.safetensors")
                   if p.name not in set(names))
    if stray:
        log(f"WARNING: {len(stray)} shard files not referenced by the "
            f"parquet are left out of the manifest: {stray[:5]}")
    dst = out_dir / "metadata.parquet"
    if not (dst.exists() and os.path.samefile(parquet_src, dst)):
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(parquet_src, tmp)
        os.replace(tmp, dst)
    manifest = {"format": FORMAT, "vae": vae, "n": n, "shards": names,
                "digests": digests}
    tmp = mpath.with_name(mpath.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, mpath)
    log(f"verified {n} latents in {len(names)} shards (vae {vae}) in "
        f"{time.time() - t0:.1f}s -> {mpath}")
    return manifest


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def fetch_file(repo, filename, download_dir):
    """Download one file of the HF dataset `repo` into `download_dir` (a real
    file, not an HF-cache symlink, so it can be deleted after use)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset",
                                local_dir=str(download_dir)))


def local_tar(tar_dir, tar):
    for cand in (Path(tar_dir) / tar, Path(tar_dir) / Path(tar).name):
        if cand.exists():
            return cand
    raise SystemExit(f"{tar} not found under {tar_dir}")


def build_parser():
    ap = argparse.ArgumentParser(
        description="Encode the curated dataset's tars into the "
                    f"{FORMAT} latent cache (one shard file per tar).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--repo", default="ricecake/danbooru-curated-v2",
                    help="HF dataset repo (private: token from the HF cache "
                         "or HF_TOKEN)")
    ap.add_argument("--tar-dir", default=None,
                    help="use local tars + metadata.parquet instead of --repo "
                         "(tars at <tar-dir>/<parquet tar> or <tar-dir>/"
                         "<basename>; never deleted)")
    ap.add_argument("--out", required=True, help="output data dir")
    ap.add_argument("--shards", default=None,
                    help="shard numbers, e.g. 0-99 or 0,3,5-7; None = every tar "
                         "named in the parquet")
    ap.add_argument("--num-workers", type=int, default=1,
                    help="split the shards across N processes (interleaved: "
                         "worker i takes shards i, i+N, ...); run one per GPU")
    ap.add_argument("--worker-index", type=int, default=0,
                    help="this process's index in [0, --num-workers)")
    ap.add_argument("--batch", type=int, default=8,
                    help="images per VAE call (same bucket)")
    ap.add_argument("--vae", default="Qwen/Qwen-Image",
                    help="diffusers repo with a vae/ subfolder "
                         "(AutoencoderKLQwenImage); recorded in the cache")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="default: cuda if available, else cpu. 'cuda' with several visible GPUs and --num-workers "
                         "> 1 picks cuda:<worker-index %% count>; prefer "
                         "HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES=<i>")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp32"],
                    default="auto",
                    help="VAE compute dtype; auto = bf16 on GPU, fp32 on CPU "
                         "(latents are stored bf16 either way)")
    ap.add_argument("--decode-threads", type=int, default=6,
                    help="CPU threads decoding WebP ahead of the VAE")
    ap.add_argument("--download-dir", default=None,
                    help="where --repo tars are downloaded; None = "
                         "<out>/_download/w<worker-index>, removed at the end "
                         "unless --keep-tars")
    ap.add_argument("--keep-tars", action="store_true",
                    help="keep downloaded tars instead of deleting each one "
                         "after its shard is written")
    ap.add_argument("--write-manifest", action="store_true",
                    help="final pass (no encoder): verify all shards against "
                         "the parquet, copy metadata.parquet into --out, "
                         "write latents/manifest.json with the sha256 of "
                         "every shard (reads all shards once, ~49 GB)")
    ap.add_argument("--verify-digests", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="with --write-manifest: re-hash and fail if a shard "
                         "differs from the digest in the existing manifest; "
                         "default: on when the existing manifest has digests")
    ap.add_argument("--vae-stub", action="store_true",
                    help=argparse.SUPPRESS)  # tests: StubEncoder, vae "stub"
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    lat_dir = out_dir / "latents"
    if not 0 <= args.worker_index < args.num_workers:
        raise SystemExit("--worker-index must be in [0, --num-workers)")
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    remote = args.tar_dir is None
    tag = f"[w{args.worker_index}/{args.num_workers}]"
    if args.download_dir is not None:
        dl_dir = Path(args.download_dir)
    else:
        sub = "manifest" if args.write_manifest else f"w{args.worker_index}"
        dl_dir = out_dir / "_download" / sub

    if remote:
        parquet = fetch_file(args.repo, "metadata.parquet", dl_dir)
    else:
        parquet = Path(args.tar_dir) / "metadata.parquet"
        if not parquet.exists():
            raise SystemExit(f"{parquet} not found")

    if args.write_manifest:
        write_manifest(out_dir, parquet, verify_digests=args.verify_digests)
        if remote and args.download_dir is None and not args.keep_tars:
            shutil.rmtree(dl_dir, ignore_errors=True)
        return

    cols = read_metadata(parquet)
    counts = Counter(cols["tar"])
    tars = sorted(counts)
    if args.shards is not None:
        want = parse_shards(args.shards)
        by_num = {shard_number(t): t for t in tars}
        absent = [k for k in want if k not in by_num]
        if absent:
            raise SystemExit(f"--shards {absent[:10]} not in the parquet")
        tars = [by_num[k] for k in want]
    mine = tars[args.worker_index::args.num_workers]
    vae_id = STUB_VAE_ID if args.vae_stub else args.vae
    todo = [t for t in mine
            if not shard_done(lat_dir / shard_name(t), vae_id, t, counts[t])]
    n_todo = sum(counts[t] for t in todo)
    print(f"{tag} {len(mine)} shards assigned, {len(mine) - len(todo)} "
          f"already done, {len(todo)} to encode ({n_todo} images)",
          flush=True)
    if not todo:
        return

    # Encoder loads lazily: a fully cached rerun never touches the model.
    if args.vae_stub:
        encoder = StubEncoder()
    else:
        device = args.device
        if (device == "cuda" and args.num_workers > 1
                and torch.cuda.device_count() > 1):
            device = f"cuda:{args.worker_index % torch.cuda.device_count()}"
        if args.dtype == "auto":
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
        else:
            dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
        print(f"{tag} loading VAE {args.vae} on {device} ({dtype})",
              flush=True)
        vae = VaeEncoder(args.vae, device, dtype)
        encoder = lambda imgs: vae_encode_batch(vae, imgs)  # noqa: E731

    def get_tar(t):
        t0 = time.time()
        path = (fetch_file(args.repo, t, dl_dir) if remote
                else local_tar(args.tar_dir, t))
        return path, time.time() - t0

    t_start, done_imgs = time.time(), 0
    # The next tar downloads while the current one is encoded.
    with ThreadPoolExecutor(1, thread_name_prefix="cache_latents_fetch") as dl:
        nxt = dl.submit(get_tar, todo[0])
        for k, t in enumerate(todo):
            t_wait = time.time()
            path, t_fetch = nxt.result()
            t_wait = time.time() - t_wait
            if k + 1 < len(todo):
                nxt = dl.submit(get_tar, todo[k + 1])
            stats = process_shard(
                path, lat_dir / shard_name(t), encoder, args.batch,
                vae_id=vae_id, source=t, expected_count=counts[t],
                decode_threads=args.decode_threads,
                tag=f"{tag} {Path(t).stem}")
            if remote and not args.keep_tars:
                path.unlink(missing_ok=True)
            done_imgs += stats["n"]
            elapsed = time.time() - t_start
            rate = done_imgs / max(elapsed, 1e-9)
            eta = (n_todo - done_imgs) / max(rate, 1e-9)
            print(f"{tag} {k + 1}/{len(todo)} {Path(t).stem}: {stats['n']} "
                  f"imgs in {stats['seconds']:.1f}s "
                  f"({stats['n'] / max(stats['seconds'], 1e-9):.2f} img/s), "
                  f"resized {stats['resized']}, fetch {t_fetch:.1f}s "
                  f"(waited {t_wait:.1f}s) | total {rate:.2f} img/s, "
                  f"ETA {eta / 60:.1f} min", flush=True)
    if remote and args.download_dir is None and not args.keep_tars:
        shutil.rmtree(dl_dir, ignore_errors=True)
    print(f"{tag} done: {done_imgs} images in "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
