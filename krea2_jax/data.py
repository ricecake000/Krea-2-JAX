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
"""

import collections
import concurrent.futures
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
from safetensors import safe_open


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
        lat_spec, txt_spec = dataset.sample_spec()
        self._lat_shape, self._lat_dtype = lat_spec
        self._txt_row, self._txt_dtype = txt_spec
        self._closed = False
        self._lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(workers), thread_name_prefix="BatchLoader")

    def permutation(self, epoch):
        return np.random.default_rng([self.seed, epoch]).permutation(
            self.dataset.n)

    def indices(self, step):
        """Dataset indices of global step `step` (0-based)."""
        epoch, k = divmod(step, self.steps_per_epoch)
        pos = k * self.batch_size
        return self.permutation(epoch)[pos:pos + self.batch_size]

    def _schedule(self, idx):
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
                    item = self._schedule(perm[pos:pos + self.batch_size])
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
