"""Streaming file hashing shared by the latent cache (scripts/cache_latents.py,
krea2_jax/data.py::BucketDataset) and the text-encoder identity
(krea2_jax/qwen3_text.py::text_encoder_identity). Standard library only, so
the torch-side caching scripts can import it without JAX."""

import hashlib
import re

CHUNK = 1 << 20  # 1 MiB reads
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def file_sha256(path, chunk=CHUNK, h=None):
    """Hex sha256 of a file's bytes, streamed in `chunk`-byte reads.

    With `h` (a hashlib object) the bytes are fed into it instead and the
    number of bytes read is returned (for hashing several files into one
    digest)."""
    own = h is None
    if own:
        h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest() if own else n


def is_sha256_hex(x):
    """True for a 64-char lowercase hex string (a sha256 hexdigest)."""
    return isinstance(x, str) and _SHA256_HEX.fullmatch(x) is not None
