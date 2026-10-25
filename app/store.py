"""Storage: a local folder in development, an S3 bucket on AWS (set BUCKET)."""

from __future__ import annotations

import json
import os
from pathlib import Path

LOCAL = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data" / "store"))


def _bucket():
    return os.environ.get("BUCKET")


_s3 = None


def _client():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client("s3")
    return _s3


def put(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    if _bucket():
        _client().put_object(Bucket=_bucket(), Key=key, Body=data, ContentType=content_type)
    else:
        p = LOCAL / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def get(key: str) -> bytes | None:
    if _bucket():
        try:
            return _client().get_object(Bucket=_bucket(), Key=key)["Body"].read()
        except _client().exceptions.NoSuchKey:
            return None
    p = LOCAL / key
    return p.read_bytes() if p.exists() else None


def put_json(key: str, obj) -> None:
    put(key, json.dumps(obj, indent=1, default=str).encode(), "application/json")


def get_json(key: str, default=None):
    raw = get(key)
    return json.loads(raw) if raw is not None else default


def keys(prefix: str) -> list[str]:
    if _bucket():
        out, token = [], None
        while True:
            kw = {"Bucket": _bucket(), "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            r = _client().list_objects_v2(**kw)
            out += [o["Key"] for o in r.get("Contents", [])]
            token = r.get("NextContinuationToken")
            if not token:
                return out
    base = LOCAL / prefix
    root = LOCAL
    if not base.parent.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in root.glob(prefix + "*") if p.is_file())
