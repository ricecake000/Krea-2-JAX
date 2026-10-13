"""Cloudflare R2 (S3 API) helpers shared by the Modal scripts
(scripts/modal_eval_full.py, scripts/modal_export_full.py).

Plain functions over a boto3-style client (tests pass a fake one): the
client from the `krea2-r2` secret env, object identities (key, ETag, size,
LastModified) and an identity-checked single-object download with a sidecar
marker. Credentials come from the environment and are never printed.
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
