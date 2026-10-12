"""Streaming loader for caches written by scripts/cache_dataset.py.

The LoRA trainer loads the whole cache into host RAM; at ~100k samples that is
far too large (one 1024x1024 sample is ~0.5 MB of latent + up to ~30 MB of
text states), so full fine-tuning streams batches from disk instead:

* `CachedDataset` reads `meta.json` up front and one per-sample safetensors
  file per `__getitem__` (lazy `safe_open`, numpy framework, bf16 kept as
  `ml_dtypes.bfloat16` — never upcast).
* `BatchLoader` yields epoch-shuffled batches with the same order semantics as
  scripts/train_lora.py (each epoch is a fresh permutation, a step consumes
  `batch_size` consecutive indices, the incomplete tail is dropped), but with
  per-epoch seeding so a resumed run can fast-forward to any step
  arithmetically and see exactly the batches of an uninterrupted run.

Full fine-tuning on the curated aspect-bucketed dataset uses
`BucketDataset` / `BucketBatchLoader` instead: VAE latents only (format
`krea2-latents-v1`, written by scripts/cache_latents.py) plus raw captions
from metadata.parquet, tokenized per batch for the online text encoder.
"""

import collections
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open

from krea2_jax.hashing import file_sha256, is_sha256_hex

BF16 = ml_dtypes.bfloat16


class CachedDataset:
    """Random access to a cache directory (meta.json + one file per sample).

    txt_len: "auto" pads every sample's text states to the longest one in the
    cache (from meta.json's per-sample `txt_len`; files are opened — header
    only — only if that field is missing); an int pads/truncates to that
    length.

    Exposes `.n`, `.txt_len`, `.meta`, `.lat_hw` (latent h, w = image
    height, width // 8) and `.resolution` ("WxH"); `fingerprint()` hashes
    the sample manifest + preprocessing identity (for resume checks).
    """

    def __init__(self, cache_dir, txt_len="auto"):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — run scripts/cache_dataset.py (for "
                "sharded caching, the final merge pass writes meta.json)")
        self.meta = json.loads(meta_path.read_text())
        self.samples = self.meta["samples"]
        self.n = len(self.samples)
        width, height = int(self.meta["width"]), int(self.meta["height"])
        self.lat_hw = (height // 8, width // 8)
        self.resolution = self.meta.get("resolution", f"{width}x{height}")

        if isinstance(txt_len, str) and txt_len == "auto":
            if self.n == 0:
                raise ValueError(f"{meta_path} lists no samples")
            lens = [s.get("txt_len") for s in self.samples]
            if any(l is None for l in lens):
                lens = [self._file_txt_len(i) for i in range(self.n)]
            txt_len = max(lens)
        txt_len = int(txt_len)
        if txt_len <= 0:
            raise ValueError(f"txt_len must be positive, got {txt_len}")
        self.txt_len = txt_len

    def __len__(self):
        return self.n

    # meta.json fields that define what the cached tensors contain
    FINGERPRINT_FIELDS = ("width", "height", "max_length", "text_encoder",
                          "vae", "caption_prefix", "compute_dtype")

    def fingerprint(self):
        """Hex sha256 identifying the dataset: the ordered sample manifest
        ([file, image, txt_len] per sample; image is the source image
        filename, null when missing, so swapping source images behind the
        sequential cache filenames is caught) plus the preprocessing identity
        fields of meta.json (FINGERPRINT_FIELDS; missing ones count as null).
        Computed from meta.json alone (no sample file is read) and
        deterministic, so a resumed run can verify it sees the same samples
        in the same order."""
        ident = {k: self.meta.get(k) for k in self.FINGERPRINT_FIELDS}
        ident["samples"] = [
            [s.get("file"), s.get("image"), s.get("txt_len")]
            for s in self.samples]
        blob = json.dumps(ident, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def path(self, i):
        return self.cache_dir / self.samples[i]["file"]

    def _file_txt_len(self, i):
        with safe_open(str(self.path(i)), framework="numpy") as f:
            return f.get_slice("txt").get_shape()[0]

    def _read(self, i):
        """(latent, txt[:txt_len]) as stored on disk (bf16, unpadded)."""
        with safe_open(str(self.path(i)), framework="numpy") as f:
            latent = f.get_tensor("latent")
            sl = f.get_slice("txt")
            if sl.get_shape()[0] > self.txt_len:
                txt = sl[:self.txt_len]  # read only the rows we keep
            else:
                txt = f.get_tensor("txt")
        return latent, txt

    def read_into(self, i, lat_out, txt_out, mask_out):
        """Load sample i into preallocated rows (txt padded with zeros)."""
        latent, txt = self._read(i)
        if latent.shape != lat_out.shape or txt.shape[1:] != txt_out.shape[1:]:
            raise ValueError(
                f"{self.path(i)}: latent {latent.shape} / txt {txt.shape} do "
                f"not match the cache's latent {lat_out.shape} / txt "
                f"(*, {', '.join(map(str, txt_out.shape[1:]))})")
        k = txt.shape[0]
        lat_out[...] = latent
        txt_out[:k] = txt
        txt_out[k:] = 0
        mask_out[:k] = True
        mask_out[k:] = False

    def __getitem__(self, i):
        """-> (latent (C,h,w) bf16, txt (txt_len,layers,dim) bf16,
        mask (txt_len,) bool)."""
        if not -self.n <= i < self.n:
            raise IndexError(i)
        i %= self.n
        latent, txt = self._read(i)
        k = txt.shape[0]
        txt_p = np.zeros((self.txt_len,) + txt.shape[1:], dtype=txt.dtype)
        txt_p[:k] = txt
        mask = np.zeros((self.txt_len,), dtype=bool)
        mask[:k] = True
        return latent, txt_p, mask

    def sample_spec(self):
        """((latent shape, dtype), (txt row shape, dtype)) from the first
        file's header — the loader trusts the tensors, not meta.json."""
        with safe_open(str(self.path(0)), framework="numpy") as f:
            lat = f.get_tensor("latent")
            txt = f.get_slice("txt")[:1]
        return (lat.shape, lat.dtype), (txt.shape[1:], txt.dtype)


class BatchLoader:
    """Epoch-shuffled, prefetching batch iterator over a CachedDataset.

    Epoch e visits np.random.default_rng([seed, e]).permutation(n) in chunks
    of `batch_size`; the last n % batch_size indices of each epoch are
    dropped (train_lora.py's "new permutation when perm_pos + batch_size > n"
    rule). Global step s (0-based) therefore reads epoch s // steps_per_epoch
    at offset (s % steps_per_epoch) * batch_size, which is how `start_step`
    fast-forwards without loading anything: the first batch yielded is the
    (start_step+1)-th batch of a fresh loader.

    Samples are read by a ThreadPoolExecutor of `workers` threads straight
    into preallocated batch arrays; at most `prefetch` batches are in flight
    or waiting to be consumed. Iteration is endless (the caller stops it);
    worker exceptions are re-raised in the iterating thread. Call close()
    (or use as a context manager) to stop the workers; breaking out of the
    loop early is safe.

    Yields (latents (B,C,h,w), txt (B,L,layers,dim), mask (B,L) bool) numpy
    arrays in the cache's dtype (bf16).
    """

    def __init__(self, dataset, batch_size, seed, start_step=0, prefetch=4,
                 workers=4):
        batch_size, start_step = int(batch_size), int(start_step)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > dataset.n:
            raise ValueError(
                f"batch_size {batch_size} exceeds the dataset size "
                f"{dataset.n}: an epoch would not contain a single batch")
        if start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {start_step}")
        if prefetch < 1 or workers < 1:
            raise ValueError("prefetch and workers must be >= 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = int(seed)
        self.start_step = start_step
        self.prefetch = int(prefetch)
        self.steps_per_epoch = dataset.n // batch_size
        self._init_spec()
        self._closed = False
        self._lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(workers), thread_name_prefix="BatchLoader")

    def _init_spec(self):
        lat_spec, txt_spec = self.dataset.sample_spec()
        self._lat_shape, self._lat_dtype = lat_spec
        self._txt_row, self._txt_dtype = txt_spec

    def permutation(self, epoch):
        return np.random.default_rng([self.seed, epoch]).permutation(
            self.dataset.n)

    def indices(self, step):
        """Dataset indices of global step `step` (0-based)."""
        epoch, k = divmod(step, self.steps_per_epoch)
        pos = k * self.batch_size
        return self.permutation(epoch)[pos:pos + self.batch_size]

    def _schedule(self, idx, epoch):
        B, L = len(idx), self.dataset.txt_len
        lat = np.empty((B,) + tuple(self._lat_shape), dtype=self._lat_dtype)
        txt = np.empty((B, L) + tuple(self._txt_row), dtype=self._txt_dtype)
        mask = np.empty((B, L), dtype=bool)
        with self._lock:
            if self._closed:
                return None
            futs = [self._pool.submit(self.dataset.read_into, int(i),
                                      lat[j], txt[j], mask[j])
                    for j, i in enumerate(idx)]
        return futs, (lat, txt, mask)

    def __iter__(self):
        if self._closed:
            raise RuntimeError("BatchLoader is closed")
        inflight = collections.deque()
        step, epoch, perm = self.start_step, None, None
        try:
            while True:
                while len(inflight) < self.prefetch:
                    e, k = divmod(step, self.steps_per_epoch)
                    if e != epoch:
                        epoch, perm = e, self.permutation(e)
                    pos = k * self.batch_size
                    item = self._schedule(perm[pos:pos + self.batch_size],
                                          epoch)
                    if item is None:  # closed (possibly from another thread)
                        return
                    inflight.append(item)
                    step += 1
                futs, batch = inflight.popleft()
                try:
                    for f in futs:
                        f.result()
                except concurrent.futures.CancelledError:
                    if self._closed:
                        return
                    raise
                yield batch
        finally:
            for futs, _ in inflight:
                for f in futs:
                    f.cancel()

    def close(self, wait=True):
        """Cancel queued loads and join the worker threads (idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close(wait=False)
        except Exception:
            pass


LATENT_FORMAT = "krea2-latents-v1"
CAPTION_FIELDS = ("long", "medium", "short")
ARTIST_SUFFIX = ", drawn by {}"


def _examples(items, k=5):
    items = list(items)
    more = f", ... (+{len(items) - k})" if len(items) > k else ""
    return ", ".join(str(x) for x in items[:k]) + more


class BucketDataset:
    """Aspect-bucketed latent cache (`krea2-latents-v1`) + raw captions.

    data_dir holds metadata.parquet and latents/{manifest.json, *.safetensors}
    (one bf16 tensor (C, bucket_h // 8, bucket_w // 8) per sample, keyed by
    str(id)). Samples are ordered by `id` ascending; sample index i refers to
    `self.ids[i]`. Only safetensors headers are read at construction; every
    parquet id must have exactly one latent of the bucket's shape (and vice
    versa), else ValueError listing every problem with counts. The manifest
    must carry `digests` ({shard file: sha256}, written by
    scripts/cache_latents.py --write-manifest) for every shard, each a
    64-char lowercase hex string (null or malformed -> ValueError); they
    enter `fingerprint()`.

    verify_digests: re-hash every shard file (streaming sha256) at
    construction and raise ValueError listing every shard whose bytes differ
    from its manifest digest (expected vs actual, counts). The cache is then
    treated as immutable for the run: a cache rebuilt or corrupted (e.g. in
    transfer) under an unchanged manifest is refused instead of silently
    training on data the fingerprint does not describe. False skips the
    re-hash (headers only, cheap) and trusts the manifest. After
    construction `verified_shards`, `verified_bytes` and `verify_seconds`
    report the work done (0, 0, 0.0 when skipped).

    template: a text_template.CaptionTemplate (tokenizes captions per batch).
    img_tokens: static image-token rows per sample (h_tok * w_tok <= it).
    caption_probs: probabilities of the long / medium / short caption
    (must sum to 1 within 1e-6; stored normalized, p / sum).
    artist_prob: probability of appending ", drawn by {artist_name}" (never
    for an empty artist_name).
    """

    def __init__(self, data_dir, template, img_tokens=4096, patch=2,
                 caption_probs=(0.5, 0.3, 0.2), artist_prob=0.5,
                 verify_digests=True):
        import pyarrow.parquet as pq

        self.data_dir = Path(data_dir)
        self.template = template
        self.img_tokens = int(img_tokens)
        self.patch = int(patch)
        probs = np.asarray(caption_probs, dtype=np.float64)
        if (probs.shape != (len(CAPTION_FIELDS),) or np.any(probs < 0)
                or not abs(probs.sum() - 1.0) < 1e-6):
            raise ValueError(
                f"caption_probs must be {len(CAPTION_FIELDS)} non-negative "
                f"probabilities (long, medium, short) summing to 1, got "
                f"{caption_probs}")
        self.caption_probs = tuple(float(x) for x in probs / probs.sum())
        self.artist_prob = float(artist_prob)
        if not 0.0 <= self.artist_prob <= 1.0:
            raise ValueError(f"artist_prob must be in [0, 1], got {artist_prob}")
        if self.patch < 1 or self.img_tokens < 1:
            raise ValueError("patch and img_tokens must be >= 1")

        # --- metadata.parquet, reordered by id ---
        meta_path = self.data_dir / "metadata.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found")
        cols = ("id", "bucket_w", "bucket_h") + CAPTION_FIELDS + (
            "artist_name",)
        table = pq.read_table(meta_path, columns=list(cols))
        ids = table.column("id").to_numpy().astype(np.int64)
        if len(ids) == 0:
            raise ValueError(f"{meta_path} has no rows")
        order = np.argsort(ids, kind="stable")
        self.ids = ids[order]
        dup = self.ids[1:][self.ids[1:] == self.ids[:-1]]
        if len(dup):
            raise ValueError(f"{meta_path}: {len(dup)} duplicate ids "
                             f"({_examples(np.unique(dup))})")
        self.ids.setflags(write=False)
        self.n = len(self.ids)
        self.bucket_wh = np.stack(
            [table.column("bucket_w").to_numpy()[order],
             table.column("bucket_h").to_numpy()[order]],
            axis=1).astype(np.int64)
        order_l = order.tolist()

        def strings(name):
            col = table.column(name).to_pylist()
            return [col[j] or "" for j in order_l]  # null -> ""

        self._captions = tuple(strings(c) for c in CAPTION_FIELDS)
        self._artist = strings("artist_name")

        # --- latents/manifest.json + shard headers ---
        lat_dir = self.data_dir / "latents"
        man_path = lat_dir / "manifest.json"
        if not man_path.exists():
            raise FileNotFoundError(
                f"{man_path} not found — run scripts/cache_latents.py "
                "--write-manifest")
        self.manifest = json.loads(man_path.read_text())
        if self.manifest.get("format") != LATENT_FORMAT:
            raise ValueError(f"{man_path}: format "
                             f"{self.manifest.get('format')!r}, expected "
                             f"{LATENT_FORMAT!r}")
        self.vae = self.manifest.get("vae")
        self.shards = [str(x) for x in self.manifest.get("shards", [])]
        digests = self.manifest.get("digests")
        if not isinstance(digests, dict):
            raise ValueError(
                f"{man_path} has no shard \"digests\" (older manifest): "
                "rerun scripts/cache_latents.py --out <data-dir> "
                "--write-manifest to record the sha256 of every shard")
        self.digests = {name: digests[name] for name in self.shards
                        if name in digests}
        no_digest = [name for name in self.shards if name not in digests]
        bad_digest = [f"{name}={self.digests[name]!r}" for name in self.shards
                      if name in self.digests
                      and not is_sha256_hex(self.digests[name])]
        index = {int(x): j for j, x in enumerate(self.ids.tolist())}
        shard_of = np.full((self.n,), -1, dtype=np.int32)
        lat_shape = np.zeros((self.n, 3), dtype=np.int64)
        extra, dupes, bad_keys, bad_dtype, missing_files = [], [], [], [], []
        n_latents = 0
        for s_idx, name in enumerate(self.shards):
            path = lat_dir / name
            if not path.exists():
                missing_files.append(name)
                continue
            with safe_open(str(path), framework="numpy") as f:
                for key in f.keys():
                    n_latents += 1
                    try:
                        sid = int(key)
                    except ValueError:
                        bad_keys.append(f"{name}:{key}")
                        continue
                    j = index.get(sid)
                    if j is None:
                        extra.append(sid)
                        continue
                    if shard_of[j] >= 0:
                        dupes.append(sid)
                        continue
                    sl = f.get_slice(key)
                    shape = tuple(sl.get_shape())
                    if sl.get_dtype() != "BF16":
                        bad_dtype.append(f"{sid}:{sl.get_dtype()}")
                    if len(shape) != 3:
                        shape = (-1, -1, -1)
                    shard_of[j] = s_idx
                    lat_shape[j] = shape

        problems = []
        if no_digest:
            problems.append(f"{len(no_digest)} manifest shards have no digest "
                            f"({_examples(no_digest)}); rerun "
                            "scripts/cache_latents.py --write-manifest")
        if bad_digest:
            problems.append(f"{len(bad_digest)} manifest digests are not "
                            "64-char lowercase hex sha256 strings "
                            f"({_examples(bad_digest)}); rerun "
                            "scripts/cache_latents.py --write-manifest")
        if missing_files:
            problems.append(f"{len(missing_files)} manifest shards missing "
                            f"on disk ({_examples(missing_files)})")
        missing = self.ids[shard_of < 0]
        if len(missing):
            problems.append(f"{len(missing)} parquet ids have no latent "
                            f"({_examples(missing.tolist())})")
        for what, items in (("latents have no parquet row", extra),
                            ("ids appear in more than one shard", dupes),
                            ("latent keys are not integer ids", bad_keys),
                            ("latents are not bf16", bad_dtype)):
            if items:
                problems.append(f"{len(items)} {what} ({_examples(items)})")
        have = shard_of >= 0
        chans = np.unique(lat_shape[have, 0])
        if len(chans) > 1:
            problems.append(f"latents have mixed channel counts "
                            f"{chans.tolist()}")
        self.lat_channels = int(chans[0]) if len(chans) else 0
        expect = np.concatenate(
            [np.full((self.n, 1), self.lat_channels, np.int64),
             self.bucket_wh[:, ::-1] // 8], axis=1)  # (C, h // 8, w // 8)
        bad_shape = have & np.any(lat_shape != expect, axis=1)
        if bad_shape.any():
            ex = [f"{self.ids[j]}:{tuple(lat_shape[j].tolist())}!="
                  f"{tuple(expect[j].tolist())}"
                  for j in np.flatnonzero(bad_shape)]
            problems.append(f"{len(ex)} latents do not match their bucket "
                            f"(C, bucket_h//8, bucket_w//8) ({_examples(ex)})")
        lat_hw = expect[:, 1:]
        bad_patch = np.any(lat_hw % self.patch != 0, axis=1)
        if bad_patch.any():
            ex = [f"{self.ids[j]}:{tuple(lat_hw[j].tolist())}"
                  for j in np.flatnonzero(bad_patch)]
            problems.append(f"{len(ex)} latent grids are not divisible by "
                            f"patch {self.patch} ({_examples(ex)})")
        hw = lat_hw // self.patch
        too_big = hw[:, 0] * hw[:, 1] > self.img_tokens
        if too_big.any():
            ex = [f"{self.ids[j]}:{hw[j, 0]}x{hw[j, 1]}"
                  for j in np.flatnonzero(too_big)]
            problems.append(f"{len(ex)} samples exceed img_tokens "
                            f"{self.img_tokens} ({_examples(ex)})")
        man_n = self.manifest.get("n")
        if man_n is not None and int(man_n) != n_latents:
            problems.append(f"manifest n={man_n} but the shards hold "
                            f"{n_latents} latents")
        if problems:
            raise ValueError(f"{self.data_dir}: latent cache does not match "
                             "metadata.parquet:\n  " + "\n  ".join(problems))

        self.verified_shards, self.verified_bytes = 0, 0
        self.verify_seconds = 0.0
        if verify_digests:
            t0 = time.time()
            changed = []
            for name in self.shards:
                path = lat_dir / name
                actual = file_sha256(path)
                self.verified_bytes += path.stat().st_size
                self.verified_shards += 1
                if actual != self.digests[name]:
                    changed.append(f"{name}: expected {self.digests[name]}, "
                                   f"actual {actual}")
            self.verify_seconds = time.time() - t0
            if changed:
                raise ValueError(
                    f"{self.data_dir}: latent digest verification FAILED: "
                    f"{len(changed)} of {len(self.shards)} shards differ from "
                    f"{man_path} (rebuilt or corrupted latents; the cache is "
                    "immutable for a run):\n  " + "\n  ".join(changed))

        self.shard_of = shard_of
        self.hw = hw.astype(np.int32)
        self.hw.setflags(write=False)
        self.feat = self.lat_channels * self.patch * self.patch
        self._choice_lock = threading.Lock()
        self._choices = None  # ((seed, epoch), (variant, use_artist))
        self._fingerprint = None

    def __len__(self):
        return self.n

    def buckets(self):
        """{(bucket_w, bucket_h): sample count}."""
        out = collections.Counter(map(tuple, self.bucket_wh.tolist()))
        return dict(sorted(out.items()))

    def fingerprint(self):
        """Hex sha256 of the sample manifest (ordered [id, bucket_w,
        bucket_h, shard]), the shard content digests from manifest.json
        (sha256 of each shard file, so a rebuilt latent cache changes it),
        every caption field used (long, medium, short, artist_name; hashed
        separately and included as `captions_sha256`) and the preprocessing
        identity (format, vae, img_tokens, patch, tokenizer template ids,
        text_rows, normalized caption_probs, artist_prob). Reads no latent;
        deterministic."""
        if self._fingerprint is not None:
            return self._fingerprint
        h = hashlib.sha256()
        for j in range(self.n):
            for field in self._captions + (self._artist,):
                h.update(field[j].encode("utf-8"))
                h.update(b"\x1f")
            h.update(b"\x1e")
        t = self.template
        ident = {
            "format": self.manifest.get("format"),
            "vae": self.vae,
            "img_tokens": self.img_tokens,
            "patch": self.patch,
            "text_rows": t.text_rows,
            "prefix_ids": [int(x) for x in t.prefix_ids],
            "suffix_ids": [int(x) for x in t.suffix_ids],
            "pad_id": int(t.pad_id),
            "caption_probs": list(self.caption_probs),
            "artist_prob": self.artist_prob,
            "samples": [[i, w, hh, self.shards[s]] for i, (w, hh), s in zip(
                self.ids.tolist(), self.bucket_wh.tolist(),
                self.shard_of.tolist())],
            "captions_sha256": h.hexdigest(),
            "shard_digests": {name: self.digests[name] for name in self.shards},
        }
        blob = json.dumps(ident, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True)
        self._fingerprint = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        return self._fingerprint

    def epoch_choices(self, seed, epoch):
        """(variant (n,) int8 in {0: long, 1: medium, 2: short}, use_artist
        (n,) bool) for every sample of `epoch`, drawn from
        default_rng([seed, epoch, 1]); memoized for the last (seed, epoch).
        The arrays are read-only."""
        key = (int(seed), int(epoch))
        with self._choice_lock:
            if self._choices is not None and self._choices[0] == key:
                return self._choices[1]
            rng = np.random.default_rng([key[0], key[1], 1])
            variant = rng.choice(len(CAPTION_FIELDS), size=self.n,
                                 p=self.caption_probs).astype(np.int8)
            use_artist = rng.random(self.n) < self.artist_prob
            variant.setflags(write=False)
            use_artist.setflags(write=False)
            self._choices = (key, (variant, use_artist))
            return variant, use_artist

    def caption(self, i, variant, use_artist):
        """Caption text of sample i: long/medium/short (variant 0/1/2) plus
        ", drawn by {artist_name}" if use_artist and the artist is known."""
        text = self._captions[int(variant)][i]
        artist = self._artist[i]
        if use_artist and artist:
            text += ARTIST_SUFFIX.format(artist)
        return text

    def captions(self, idx, seed, epoch):
        """Caption texts of samples `idx` as seen in `epoch`."""
        variant, use_artist = self.epoch_choices(seed, epoch)
        return [self.caption(int(i), variant[i], use_artist[i]) for i in idx]

    def latent_path(self, i):
        return self.data_dir / "latents" / self.shards[self.shard_of[i]]

    def read_latent(self, i):
        """(C, h, w) bf16 latent of sample i, as stored."""
        with safe_open(str(self.latent_path(i)), framework="numpy") as f:
            return f.get_tensor(str(int(self.ids[i])))

    def read_latent_into(self, i, out_row):
        """Host-patchify sample i's latent into out_row (img_tokens, C*p*p):
        rows [:h_tok*w_tok] in krea2_jax.model.patchify layout (token-major
        (h_tok, w_tok), features (C, p, p)), the rest zeroed."""
        lat = self.read_latent(i)
        C, p = self.lat_channels, self.patch
        h, w = (int(x) for x in self.hw[i])
        if lat.shape != (C, h * p, w * p):
            raise ValueError(
                f"{self.latent_path(i)}: latent {self.ids[i]} has shape "
                f"{lat.shape}, expected {(C, h * p, w * p)}")
        n_img = h * w
        out_row[:n_img].reshape(h, w, C, p, p)[...] = lat.reshape(
            C, h, p, w, p).transpose(1, 3, 0, 2, 4)
        out_row[n_img:] = 0


class BucketBatchLoader(BatchLoader):
    """Epoch-shuffled, prefetching batch iterator over a BucketDataset.

    Same order / tail-drop / `start_step` fast-forward / threading / close
    semantics as BatchLoader. Captions of epoch e use
    dataset.epoch_choices(seed, e) and are tokenized once per batch (in the
    iterating thread) with dataset.template.encode_captions; latents are read
    and patchified by the worker threads into a preallocated batch array.

    Yields numpy (img (B, img_tokens, C*p*p) bf16 — rows >= h_tok*w_tok are
    zero, hw (B, 2) int32 (h_tok, w_tok), ids (B, template.total_len) int32,
    txt_len (B,) int32).
    """

    def _init_spec(self):
        pass

    def _schedule(self, idx, epoch):
        ds = self.dataset
        B = len(idx)
        img = np.empty((B, ds.img_tokens, ds.feat), dtype=BF16)
        hw = np.ascontiguousarray(ds.hw[idx], dtype=np.int32)
        ids, txt_len = ds.template.encode_captions(
            ds.captions(idx, self.seed, epoch))
        with self._lock:
            if self._closed:
                return None
            futs = [self._pool.submit(ds.read_latent_into, int(i), img[j])
                    for j, i in enumerate(idx)]
        return futs, (img, hw, ids, txt_len)
