"""Cloudflare R2 (S3 API) helpers shared by the Modal scripts
(scripts/modal_eval_full.py, scripts/modal_export_full.py).

Plain functions over a boto3-style client (tests pass a fake one): the
client from the `krea2-r2` secret env, object identities (key, ETag, size,
LastModified), an identity-checked single-object download with a sidecar
marker, and a read-only safetensors reader over ETag-pinned ranged GETs
(R2Safetensors, R2Fetch: scripts/modal_average_full.py). Credentials come
from the environment and are never printed.
"""

import json
import os
import time
from pathlib import Path

R2_ENV = ("R2_ACCOUNT_ID", "R2_BUCKET", "R2_ACCESS_KEY_ID",
          "R2_SECRET_ACCESS_KEY")
R2_TRANSFER = dict(max_concurrency=32, multipart_threshold=64 * 2**20,
                   multipart_chunksize=64 * 2**20)


def r2_client(env):
    """boto3 S3 client for Cloudflare R2 from the `krea2-r2` secret env."""
    missing = [k for k in R2_ENV if not env.get(k)]
    if missing:
        raise RuntimeError(f"Modal secret krea2-r2 lacks {missing}")
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      max_pool_connections=64,
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"),
    )


R2_PLACEHOLDERS = {"R2_ACCOUNT_ID": "<account>", "R2_BUCKET": "<bucket>",
                   "R2_ACCESS_KEY_ID": "<key-id>",
                   "R2_SECRET_ACCESS_KEY": "<secret>"}


def scrub(text, secrets):
    """text with every (value, placeholder) of `secrets` replaced (longest
    value first; empty values skipped)."""
    text = str(text)
    for value, ph in sorted(secrets, key=lambda vp: -len(str(vp[0] or ""))):
        if value:
            text = text.replace(str(value), ph)
    return text


def env_secrets(env):
    """[(value, placeholder)] of the `krea2-r2` secret env (for scrub)."""
    return [(env.get(k), ph) for k, ph in R2_PLACEHOLDERS.items()]


def client_secrets(client, bucket):
    """[(value, placeholder)] of a client's endpoint host / account id and
    the bucket (for scrub)."""
    from urllib.parse import urlparse

    host = urlparse(str(getattr(getattr(client, "meta", None),
                                "endpoint_url", "") or "")).hostname or ""
    out = [(bucket, "<bucket>")]
    if host.endswith(".r2.cloudflarestorage.com"):
        out += [(host, "<endpoint>"), (host.split(".")[0], "<account>")]
    return out


def make_transfer_config(**overrides):
    """boto3 TransferConfig with R2_TRANSFER (+ overrides)."""
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(**{**R2_TRANSFER, **overrides})


def r2_marker_path(dest):
    """Sidecar with the R2 object identity of a complete download."""
    dest = Path(dest)
    return dest.with_name(dest.name + ".r2.json")


def r2_identity(head, key):
    """head_object response -> {"key", "etag", "size", "last_modified"}
    (plain JSON types)."""
    lm = head.get("LastModified")
    return {"key": key, "etag": str(head.get("ETag", "")).strip('"'),
            "size": int(head["ContentLength"]),
            "last_modified": (lm.isoformat() if hasattr(lm, "isoformat")
                              else None if lm is None else str(lm))}


def read_r2_marker(dest):
    """The identity recorded next to dest, or None."""
    try:
        return json.loads(r2_marker_path(dest).read_text())
    except (OSError, ValueError):
        return None


def head_or_none(client, bucket, key):
    """head_object, or None when the object does not exist (any other error
    is raised)."""
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception as e:  # botocore ClientError (not importable w/o boto3)
        code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def list_objects(client, bucket, prefix):
    """Every object under prefix (list_objects_v2, paginated) ->
    [{"key", "size", "etag"}]."""
    out, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = client.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            out.append({"key": o["Key"], "size": int(o["Size"]),
                        "etag": str(o.get("ETag", "")).strip('"')})
        if not resp.get("IsTruncated"):
            return out
        token = resp["NextContinuationToken"]


def r2_fetch(client, bucket, key, dest, log=print, transfer_config=None):
    """Download s3://bucket/key to dest (multipart, via dest.part) unless dest
    is a complete download of the CURRENT object: its sidecar marker
    (r2_marker_path, written only after a complete, size-checked download)
    must equal the object's head_object identity (key, ETag, size,
    LastModified) and dest must have that size. A replaced object (even of
    the same size) is downloaded again. s3transfer pins every ranged part
    GET to the ETag of its own HeadObject (IfMatch); head_object is asked
    again after the download, and an identity that changed meanwhile is an
    error (no file, no sidecar). -> (status, identity)."""
    dest = Path(dest)
    head = client.head_object(Bucket=bucket, Key=key)
    ident = r2_identity(head, key)
    size = ident["size"]
    if (dest.is_file() and dest.stat().st_size == size
            and read_r2_marker(dest) == ident):
        log(f"r2:{key}: complete ({size / 2**30:.2f} GiB, etag "
            f"{ident['etag']}), skipped")
        return "skipped", ident
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = r2_marker_path(dest)
    marker.unlink(missing_ok=True)   # dest is not trusted until re-marked
    part = dest.with_name(dest.name + ".part")
    t0 = time.time()
    if transfer_config is None:
        transfer_config = make_transfer_config()
    try:
        client.download_file(bucket, key, str(part), Config=transfer_config)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    after = r2_identity(client.head_object(Bucket=bucket, Key=key), key)
    if after != ident:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"r2:{key}: the object changed during the download "
                           f"(before {ident}, after {after}); run prepare "
                           "again")
    got = part.stat().st_size
    if got != size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"r2:{key}: downloaded {got} bytes, object has "
                           f"{size}")
    os.replace(part, dest)
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text(json.dumps(ident, indent=1))
    os.replace(tmp, marker)
    dt = time.time() - t0
    log(f"r2:{key}: {size / 2**30:.2f} GiB (etag {ident['etag']}) in "
        f"{dt:.0f} s ({size / 2**20 / max(dt, 1e-6):.0f} MiB/s) -> {dest}")
    return "downloaded", ident


# ------------------------------------------------- safetensors over ranged GETs
_ST_NUMPY = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8",
             "I32": "<i4", "I16": "<i2", "I8": "i1", "U8": "u1", "BOOL": "?"}
ST_CHUNK = 64 * 2**20          # bytes per ranged GET of a tensor
ST_MAX_HEADER = 100 * 2**20    # safetensors' own header limit
ST_ATTEMPTS = 4                # tries per ranged GET (mid-stream resets)
_READ_BLOCK = 8 * 2**20


def _st_dtype(name):
    import numpy as np

    if name == "BF16":
        import ml_dtypes
        return np.dtype(ml_dtypes.bfloat16)
    if name not in _ST_NUMPY:
        raise ValueError(f"unsupported safetensors dtype {name!r}")
    return np.dtype(_ST_NUMPY[name])


def _precondition_failed(e):
    resp = getattr(e, "response", None) or {}
    code = str(resp.get("Error", {}).get("Code", ""))
    status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in ("PreconditionFailed", "412") or status == 412


class _Slice:
    def __init__(self, info):
        self._info = info

    def get_shape(self):
        return list(self._info["shape"])

    def get_dtype(self):
        return self._info["dtype"]


class _PendingTensor:
    """One tensor being fetched: chunk futures writing into one buffer."""

    def __init__(self, arr_shape, dtype, buf, futures):
        self._shape, self._dtype = arr_shape, dtype
        self._buf, self._futures = buf, futures

    def result(self):
        for f in self._futures:
            f.result()
        buf, self._buf, self._futures = self._buf, None, None
        return buf.view(self._dtype).reshape(self._shape)


class R2Safetensors:
    """Read-only safetensors reader over one R2 object (the subset of
    safetensors.safe_open used by scripts/average_snapshots.py: keys,
    metadata, get_slice(k).get_shape()/get_dtype(), get_tensor -> numpy).

    The object is pinned at construction (head_object ETag + size): every
    ranged GET sends IfMatch=<ETag>, so a replaced object fails with
    RuntimeError ("object changed") instead of mixing versions. The header
    is validated (length, dtype, data_offsets inside the file, nbytes ==
    prod(shape) * itemsize). A tensor is fetched as `chunk`-byte ranged GETs
    on `executor` (shared, its max_workers bounds the HTTP connections),
    written into one preallocated buffer, each chunk length-checked and
    retried (ST_ATTEMPTS) on errors other than a failed precondition.
    stats: {"read_bytes"} (shared dict, updated under a lock)."""

    def __init__(self, client, bucket, key, executor=None, chunk=ST_CHUNK,
                 stats=None, lock=None):
        import struct
        import threading

        self.client, self.bucket, self.key = client, bucket, key
        self.executor, self.chunk = executor, int(chunk)
        self.stats = stats if stats is not None else {"read_bytes": 0}
        self._lock = lock or threading.Lock()
        self._secrets = client_secrets(client, bucket)
        head = client.head_object(Bucket=bucket, Key=key)
        self.etag = head["ETag"]            # as sent back (quoted) for IfMatch
        self.size = int(head["ContentLength"])
        if self.size < 8:
            raise ValueError(f"r2:{key}: {self.size} bytes, not a safetensors")
        n = struct.unpack("<Q", self._read(0, 8))[0]
        if not 2 <= n <= ST_MAX_HEADER or 8 + n > self.size:
            raise ValueError(f"r2:{key}: bad safetensors header length {n} "
                             f"(object {self.size} bytes)")
        hdr = json.loads(self._read(8, 8 + n))
        if not isinstance(hdr, dict):
            raise ValueError(f"r2:{key}: header is not a JSON object")
        meta = hdr.pop("__metadata__", None)
        self._meta = None if meta is None else {str(k): str(v)
                                                for k, v in meta.items()}
        self.data_start = 8 + n
        data_len = self.size - self.data_start
        self._info = {}
        for k, v in hdr.items():
            try:
                dt = _st_dtype(v["dtype"])
                shape = tuple(int(d) for d in v["shape"])
                a, b = (int(x) for x in v["data_offsets"])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"r2:{key}: bad header entry {k!r}: {e}")
            count = 1
            for d in shape:
                if d < 0:
                    raise ValueError(f"r2:{key}: {k}: negative dim {shape}")
                count *= d
            if not 0 <= a <= b <= data_len or b - a != count * dt.itemsize:
                raise ValueError(f"r2:{key}: {k}: data_offsets {[a, b]} do not "
                                 f"fit shape {shape} {v['dtype']} in "
                                 f"{data_len} data bytes")
            self._info[k] = {"dtype": v["dtype"], "np": dt, "shape": shape,
                             "start": a, "nbytes": b - a}

    # -- safe_open interface
    def keys(self):
        return sorted(self._info)

    def metadata(self):
        return self._meta

    def get_slice(self, k):
        return _Slice(self._info[k])

    def nbytes(self, k):
        return self._info[k]["nbytes"]

    def get_tensor(self, k):
        return self.start(k).result()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # -- fetching
    def start(self, k):
        """Submit the ranged GETs of tensor k -> object with .result()."""
        import numpy as np
        from concurrent.futures import Future

        info = self._info[k]
        n, base = info["nbytes"], self.data_start + info["start"]
        buf = np.empty(n, np.uint8)
        futures = []
        for a in range(0, n, self.chunk):
            b = min(a + self.chunk, n)
            if self.executor is None:
                f = Future()
                self._read_into(buf[a:b], base + a)
                f.set_result(None)
            else:
                f = self.executor.submit(self._read_into, buf[a:b], base + a)
            futures.append(f)
        return _PendingTensor(info["shape"], info["np"], buf, futures)

    def _get(self, start, end):
        """Ranged GET [start, end) pinned to the ETag -> streaming body."""
        try:
            return self.client.get_object(
                Bucket=self.bucket, Key=self.key, IfMatch=self.etag,
                Range=f"bytes={start}-{end - 1}")["Body"]
        except Exception as e:
            if _precondition_failed(e):
                raise RuntimeError(
                    f"r2:{self.key}: object changed (ETag no longer "
                    f"{self.etag}); re-run") from None
            raise

    def _read(self, start, end):
        out = bytearray(end - start)
        self._read_into(memoryview(out), start)
        return bytes(out)

    def _read_into(self, dest, start):
        """Fill dest (writable buffer) from [start, start + len(dest)),
        retried on transient errors; the length is checked."""
        mv = memoryview(dest).cast("B")
        n = len(mv)
        for attempt in range(ST_ATTEMPTS):
            got = 0
            try:
                body = self._get(start, start + n)
                try:
                    while got < n:
                        b = body.read(min(_READ_BLOCK, n - got))
                        if not b:
                            break
                        if got + len(b) > n:
                            raise IOError(f"more than {n} bytes")
                        mv[got:got + len(b)] = b
                        got += len(b)
                finally:
                    close = getattr(body, "close", None)
                    if close is not None:
                        close()
                if got != n:
                    raise IOError(f"short read: {got} of {n} bytes")
                with self._lock:
                    self.stats["read_bytes"] = self.stats.get(
                        "read_bytes", 0) + n
                return
            except RuntimeError:
                raise                      # the object changed: never retry
            except Exception as e:
                if attempt + 1 == ST_ATTEMPTS:
                    msg = scrub(f"{type(e).__name__}: {e}", self._secrets)
                    raise IOError(f"r2:{self.key}: bytes {start}-"
                                  f"{start + n - 1} failed {ST_ATTEMPTS} "
                                  f"times: {msg}") from None
                time.sleep(2 ** attempt)


class R2Fetch:
    """fetch(key, order) for average_snapshots.average over R2Safetensors
    readers: yields the inputs' arrays of `key` in input order while the next
    (key, input) pairs in `order` (key-major, input order) are prefetched,
    across tensor boundaries, as long as the fetched-but-unconsumed bytes
    stay within `budget` (one pair is always allowed, whatever its size); a
    pair's bytes count until the consumer requests the next array or closes
    the generator.
    The readers' shared executor bounds the HTTP connections. A key requested
    out of the expected order is read directly (correct, slower; logged
    once). Prefetched data is never reused for another key."""

    def __init__(self, readers, budget=4 * 2**30, max_pending=512, log=print):
        import collections

        self.readers, self.budget, self.max_pending = readers, budget, max_pending
        self.log = log
        self._order = None
        self._next = 0
        self._pending = collections.deque()  # (key, i, nbytes, pending)
        self._held = 0
        self._warned = False

    def _items(self, j):
        n = len(self.readers)
        return self._order[j // n], j % n

    def _fill(self):
        total = len(self._order) * len(self.readers)
        while self._next < total and len(self._pending) < self.max_pending:
            k, i = self._items(self._next)
            nb = self.readers[i].nbytes(k)
            if self._pending and self._held + nb > self.budget:
                return
            self._pending.append((k, i, nb, self.readers[i].start(k)))
            self._held += nb
            self._next += 1

    def _expected(self):
        if self._pending:
            return self._pending[0][:2]
        total = len(self._order or ()) * len(self.readers)
        return self._items(self._next) if self._next < total else None

    def __call__(self, key, order):
        if self._order is None:
            self._order = list(order)
        if self._expected() != (key, 0):
            if not self._warned:
                self.log(f"R2Fetch: {key} requested out of the expected "
                         "order: direct reads from here on when out of order")
                self._warned = True
            for r in self.readers:
                yield r.get_tensor(key)
            return
        for i in range(len(self.readers)):
            self._fill()
            k, j, nb, p = self._pending.popleft()
            assert (k, j) == (key, i), ((k, j), (key, i))
            arr = p.result()
            del p
            try:
                yield arr
            finally:   # the consumer is done with arr (next request / close)
                del arr
                self._held -= nb
