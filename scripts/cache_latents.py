"""Build the `krea2-latents-v1` VAE latent cache for the bucketed full
fine-tuning path (read by krea2_jax/data.py::BucketDataset).

Input: the HF dataset (default ricecake/danbooru-curated-v2: data/train-XXXX.tar
with `{id}.webp` + `{id}.json` (+ `{id}.txt`, ignored) per sample, and
metadata.parquet), or a local directory with the same layout (--tar-dir).
Images are pre-cropped to their aspect bucket; each is VAE-encoded at
(bucket_w, bucket_h) and stored normalized ((z - latents_mean) / latents_std,
exactly like cache_dataset.VaeEncoder) as bf16.

VAE encode path (--vae-impl, scripts/vae_fast.py): "ref" = diffusers
vae.encode (default), "2d" = equivalent single-frame 2D conv encoder, "2d-compile"
= the same under torch.compile, "2d-compile-cl" = 2d-compile with module and
input in channels_last (latents still come out contiguous NCHW). Recorded as
shard metadata "vae_impl"
(informational: shard_done ignores it, so shards of any impl stay valid).

Output (<out>):
    metadata.parquet                    verbatim copy (--write-manifest pass)
    latents/manifest.json               {"format", "vae", "n", "shards", "digests"} (--write-manifest pass)
    latents/train-XXXX.safetensors      one per source tar; "<id>" -> (16, bucket_h/8, bucket_w/8)
                                        bf16; metadata {format, vae, source, count[, vae_impl]}

Pipelined (main()): a prefetch thread fetches + reads the next tar, the
decode pool decodes --decode-lookahead batches ahead (collated by a helper
thread into pinned memory on CUDA), the GPU gets batch j+1 before batch j's
latents are checked and copied out (encode_samples), and one writer thread
saves the previous shard while the next is encoded. Output (tensors, order,
RNG draws, metadata) is that of the synchronous loop (process_shard); the
per-shard progress line is printed only once that shard's save finished and
reports where the main thread was blocked (tar / decode / gpu / save).
Host memory: <= 2 tars of samples, lookahead + 1 collated batches, the shard
being encoded + 1 pending save.

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
from vae_fast import IMPLS as VAE_IMPLS, make_encoder  # noqa: E402
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
# seconds the main thread is blocked per shard, in pipeline order: next tar
# (fetch + read_tar), decode + collate, GPU results, the previous save
TIMING_KEYS = ("tar_wait", "decode_wait", "gpu_wait", "save_wait")
DECODE_LOOKAHEAD = 8   # batches decoded ahead of the encoder
RESOLVE_DEPTH = 2      # batches enqueued on the GPU before one is resolved


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


def encoder_device(encoder):
    """The torch.device an encoder computes on (FastEncoder / VaeEncoder-
    like: .vae.device; anything else, e.g. StubEncoder: cpu)."""
    dev = getattr(getattr(encoder, "vae", None), "device", None)
    return torch.device(dev if dev is not None else "cpu")


def _collate(futures, pin):
    """Decode futures of one batch -> ((N, 3, h, w) float32, [resized?]);
    into pinned host memory when `pin` (async H2D)."""
    decoded = [f.result() for f in futures]
    imgs = [d[0] for d in decoded]
    if pin:
        out = torch.empty((len(imgs),) + tuple(imgs[0].shape),
                          dtype=imgs[0].dtype, pin_memory=True)
        torch.stack(imgs, out=out)
    else:
        out = torch.stack(imgs)
    return out, [d[1] for d in decoded]


def encode_samples(samples, encoder, batch, pool, lookahead=DECODE_LOOKAHEAD,
                   log=print, tag="", timing=None):
    """Encode samples grouped by bucket, `batch` at a time, decoding up to
    `lookahead` batches ahead on `pool` (collated into one tensor per batch
    by a helper thread). -> ({str(id): (16, h/8, w/8) bf16 CPU tensor},
    number of resized images).

    On CUDA the main thread never waits for the GPU per batch: the batch is
    collated into pinned memory, copied H2D non_blocking, encoded, and its
    bf16 latents + per-sample isfinite flags are copied D2H non_blocking
    into pinned buffers behind a CUDA event; batch j is resolved (event
    wait, non-finite check, per-sample contiguous clone) only once batch
    j + RESOLVE_DEPTH has been enqueued, and all are drained before return.
    Batch order, bucket chunking, the encoder's RNG draw order, the stored
    tensors and the errors (same messages, raised before anything is saved)
    are those of the synchronous loop; on CPU nothing is pinned or deferred
    to an event. `timing` (dict, optional) accumulates the seconds the
    calling thread was blocked: "decode_wait" (decode + collate of the next
    batch) and "gpu_wait" (latents of a batch not back yet)."""
    if timing is None:
        timing = {}
    timing.setdefault("decode_wait", 0.0)
    timing.setdefault("gpu_wait", 0.0)
    device = encoder_device(encoder)
    cuda = device.type == "cuda"
    groups = defaultdict(list)
    for s in samples:
        groups[(s["w"], s["h"])].append(s)
    chunks = [g[i:i + batch] for key in sorted(groups)
              for g in (groups[key],) for i in range(0, len(g), batch)]
    out, resized = {}, 0

    def resolve(chunk, lat, fin, event):
        t = time.perf_counter()
        if event is not None:
            event.synchronize()    # lat / fin are the pinned D2H targets
        else:
            lat, fin = lat.cpu(), fin.cpu()
        timing["gpu_wait"] += time.perf_counter() - t
        if not fin.all():
            bad = [s["id"] for k, s in enumerate(chunk) if not fin[k]]
            raise ValueError(f"{tag}: non-finite latents for ids {bad}")
        for k, s in enumerate(chunk):
            # own, contiguous storage: safetensors refuses strided tensors
            # (e.g. channels_last latents from cudnn.benchmark / inductor)
            out[str(s["id"])] = lat[k].clone(
                memory_format=torch.contiguous_format)

    pending, inflight, nxt = deque(), deque(), 0
    with ThreadPoolExecutor(1, thread_name_prefix="cache_latents_collate"
                            ) as collator:
        for j in range(len(chunks)):
            while nxt < len(chunks) and nxt <= j + lookahead:
                futs = [pool.submit(decode, s) for s in chunks[nxt]]
                pending.append(collator.submit(_collate, futs, cuda))
                nxt += 1
            t = time.perf_counter()
            imgs, flags = pending.popleft().result()
            timing["decode_wait"] += time.perf_counter() - t
            chunk = chunks[j]
            for s, was_resized in zip(chunk, flags):
                if was_resized:
                    resized += 1
                    log(f"WARNING {tag}: id {s['id']} is not "
                        f"{s['w']}x{s['h']}; cover-resized + center-cropped")
            if cuda:
                imgs = imgs.to(device, non_blocking=True)
            lat = encoder(imgs)
            w, h = chunk[0]["w"], chunk[0]["h"]
            want = (len(chunk), LAT_CHANNELS, h // VAE_SCALE, w // VAE_SCALE)
            if tuple(lat.shape) != want:
                raise ValueError(f"{tag}: encoder returned "
                                 f"{tuple(lat.shape)}, expected {want}")
            fin = torch.isfinite(lat).flatten(1).all(1)
            lat = lat.detach().to(torch.bfloat16)
            event = None
            if lat.is_cuda:
                lat_h = torch.empty(lat.shape, dtype=lat.dtype,
                                    pin_memory=True)
                fin_h = torch.empty(fin.shape, dtype=fin.dtype,
                                    pin_memory=True)
                lat_h.copy_(lat, non_blocking=True)
                fin_h.copy_(fin, non_blocking=True)
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(lat.device))
                lat, fin = lat_h, fin_h
            inflight.append((chunk, lat, fin, event))
            while len(inflight) > RESOLVE_DEPTH:
                resolve(*inflight.popleft())
        while inflight:
            resolve(*inflight.popleft())
    return out, resized


def read_samples(tar_path, expected_count=None):
    """read_tar + the parquet's sample count for this tar (if given)."""
    samples = read_tar(tar_path)
    if expected_count is not None and len(samples) != expected_count:
        raise ValueError(f"{tar_path}: {len(samples)} samples, metadata "
                         f"expects {expected_count}")
    return samples


def write_shard(tensors, out_path, *, vae_id, source, vae_impl=None):
    """Atomic shard write (<out_path>.tmp, then os.replace) -> seconds."""
    t = time.perf_counter()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    meta = {"format": FORMAT, "vae": vae_id, "source": source,
            "count": str(len(tensors))}
    if vae_impl is not None:
        meta["vae_impl"] = vae_impl
    save_file(tensors, str(tmp), metadata=meta)
    os.replace(tmp, out_path)
    return time.perf_counter() - t


def process_shard(tar_path, out_path, encoder, batch=8, *, vae_id, source,
                  expected_count=None, decode_threads=4, log=print, tag=None,
                  vae_impl=None, lookahead=DECODE_LOOKAHEAD):
    """Encode one dataset tar into one `krea2-latents-v1` shard file
    (synchronously at the shard level: main() overlaps read / encode / save
    of consecutive shards with the same functions).

    `encoder(imgs (N, 3, H, W) float32 in [-1, 1]) -> (N, 16, H/8, W/8)`.
    Skips the work when `out_path` is already complete for this job (with
    `expected_count` given this check needs no tar). Writes
    `<out_path>.tmp` then os.replace. `vae_impl` (if given) is stored as
    extra metadata; it is not part of shard_done. Returns stats
    {skipped, n, resized, seconds} (+ when encoded, the seconds blocked on
    TIMING_KEYS: tar_wait = read_tar, decode_wait / gpu_wait
    (encode_samples), save_wait = write_shard).
    """
    out_path = Path(out_path)
    tag = tag or out_path.stem
    if expected_count is not None and shard_done(out_path, vae_id, source,
                                                 expected_count):
        return {"skipped": True, "n": expected_count, "resized": 0,
                "seconds": 0.0}
    t0 = time.time()
    t = time.perf_counter()
    samples = read_samples(tar_path, expected_count)
    timing = {"tar_wait": time.perf_counter() - t}
    if expected_count is None and shard_done(out_path, vae_id, source,
                                             len(samples)):
        return {"skipped": True, "n": len(samples), "resized": 0,
                "seconds": 0.0}
    with ThreadPoolExecutor(decode_threads,
                            thread_name_prefix="cache_latents_decode") as pool:
        tensors, resized = encode_samples(samples, encoder, batch, pool,
                                          lookahead=lookahead, log=log,
                                          tag=tag, timing=timing)
    timing["save_wait"] = write_shard(tensors, out_path, vae_id=vae_id,
                                      source=source, vae_impl=vae_impl)
    return {"skipped": False, "n": len(tensors), "resized": resized,
            "seconds": time.time() - t0, **timing}


def shard_breakdown(stats):
    """', blocked tar 0.0s decode 0.3s gpu 15.0s save 0.0s' from a shard's
    stats ('' for a skipped shard)."""
    keys = [k for k in TIMING_KEYS if k in stats]
    return (", blocked " + " ".join(f"{k[:-5]} {stats[k]:.1f}s"
                                    for k in keys) if keys else "")


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


PATCH_KEYS = ("superseded", "patches")  # scripts/modal_patch_latents.py


def refuse_patched(mpath):
    """SystemExit if `mpath` is the manifest of a patched version (written
    by scripts/modal_patch_latents.py build): --write-manifest would replace
    its metadata.parquet with the source dataset's and drop the patch."""
    try:
        old = json.loads(Path(mpath).read_text())
    except (OSError, ValueError):
        return
    if isinstance(old, dict) and any(k in old for k in PATCH_KEYS):
        raise SystemExit(
            f"{mpath} belongs to a patched version (has "
            f"{[k for k in PATCH_KEYS if k in old]}): --write-manifest would "
            "overwrite its metadata.parquet and drop the patch. Patched "
            "versions are built by scripts/modal_patch_latents.py build.")


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
    refuse_patched(mpath)
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
    ap.add_argument("--vae-impl", choices=VAE_IMPLS, default="ref",
                    help="VAE encode path (scripts/vae_fast.py): ref = "
                         "diffusers vae.encode, 2d = equivalent single-frame "
                         "2D encoder, 2d-compile = 2d under torch.compile, "
                         "2d-compile-cl = 2d-compile in channels_last; "
                         "stored as shard metadata vae_impl")
    ap.add_argument("--decode-threads", type=int, default=6,
                    help="CPU threads decoding WebP ahead of the VAE")
    ap.add_argument("--decode-lookahead", type=int, default=DECODE_LOOKAHEAD,
                    help="batches decoded (and collated) ahead of the VAE")
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
    if args.batch < 1 or args.decode_lookahead < 0:
        raise SystemExit("--batch must be >= 1, --decode-lookahead >= 0")
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
        vae_impl = STUB_VAE_ID
    else:
        device = args.device
        if (device == "cuda" and args.num_workers > 1
                and torch.cuda.device_count() > 1):
            device = f"cuda:{args.worker_index % torch.cuda.device_count()}"
        if args.dtype == "auto":
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
        else:
            dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
        print(f"{tag} loading VAE {args.vae} on {device} ({dtype}), impl "
              f"{args.vae_impl}", flush=True)
        vae = VaeEncoder(args.vae, device, dtype)
        encoder = make_encoder(vae, args.vae_impl)
        vae_impl = args.vae_impl

    def load_tar(t):
        """Prefetch thread: fetch + read_tar (+ count check) of tar t; a
        downloaded tar is deleted once read (its samples are in RAM)."""
        t0 = time.time()
        path = (fetch_file(args.repo, t, dl_dir) if remote
                else local_tar(args.tar_dir, t))
        t_fetch = time.time() - t0
        samples = read_samples(path, counts[t])
        if remote and not args.keep_tars:
            path.unlink(missing_ok=True)
        return samples, t_fetch, time.time() - t0 - t_fetch

    t_start, done_imgs = time.time(), 0

    def report(k, t, stats, save_s):
        """The per-shard progress line, printed only once the shard's
        save finished (modal_cache_latents commits the volume on it)."""
        nonlocal done_imgs
        done_imgs += stats["n"]
        rate = done_imgs / max(time.time() - t_start, 1e-9)
        eta = (n_todo - done_imgs) / max(rate, 1e-9)
        print(f"{tag} {k + 1}/{len(todo)} {Path(t).stem}: {stats['n']} "
              f"imgs in {stats['seconds']:.1f}s "
              f"({stats['n'] / max(stats['seconds'], 1e-9):.2f} img/s), "
              f"resized {stats['resized']}, fetch {stats['fetch']:.1f}s "
              f"read {stats['read']:.1f}s{shard_breakdown(stats)}, saved "
              f"in {save_s:.1f}s | total {rate:.2f} img/s, ETA "
              f"{eta / 60:.1f} min", flush=True)

    # Overlapped per shard: the next tar is fetched + read by `dl` while
    # this one is encoded; its save runs on `saver` (at most one in flight)
    # while the next one is encoded. Shard k's stats: main-thread wall time
    # from waiting for its samples to handing it to the saver, and the
    # seconds blocked within it (TIMING_KEYS).
    with (ThreadPoolExecutor(1, thread_name_prefix="cache_latents_fetch")
          as dl,
          ThreadPoolExecutor(1, thread_name_prefix="cache_latents_save")
          as saver,
          ThreadPoolExecutor(args.decode_threads,
                             thread_name_prefix="cache_latents_decode")
          as pool):
        nxt, saving = dl.submit(load_tar, todo[0]), None
        try:
            for k, t in enumerate(todo):
                t0 = time.perf_counter()
                samples, t_fetch, t_read = nxt.result()
                timing = {"tar_wait": time.perf_counter() - t0}
                if k + 1 < len(todo):
                    nxt = dl.submit(load_tar, todo[k + 1])
                tensors, resized = encode_samples(
                    samples, encoder, args.batch, pool,
                    lookahead=args.decode_lookahead,
                    tag=f"{tag} {Path(t).stem}", timing=timing)
                del samples
                t1 = time.perf_counter()
                if saving is not None:
                    prev, saving = saving, None
                    save_s = prev[0].result()   # a failed save raises here
                    timing["save_wait"] = time.perf_counter() - t1
                    report(*prev[1:], save_s)
                else:
                    timing["save_wait"] = 0.0
                stats = {"n": len(tensors), "resized": resized,
                         "fetch": t_fetch, "read": t_read, **timing,
                         "seconds": time.perf_counter() - t0}
                saving = (saver.submit(write_shard, tensors,
                                       lat_dir / shard_name(t), vae_id=vae_id,
                                       source=t, vae_impl=vae_impl),
                          k, t, stats)
                del tensors
            if saving is not None:
                prev, saving = saving, None
                report(*prev[1:], prev[0].result())
        finally:
            if saving is not None:   # failing elsewhere: report a save that
                try:                 # still completed, then re-raise
                    report(*saving[1:], saving[0].result())
                except Exception:  # noqa: BLE001 - the original error wins
                    pass
    if remote and args.download_dir is None and not args.keep_tars:
        shutil.rmtree(dl_dir, ignore_errors=True)
    print(f"{tag} done: {done_imgs} images in "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
