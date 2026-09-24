from __future__ import annotations

import hashlib
from pathlib import Path

from .models import FileProvenance


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: str | Path, include_hash: bool = False) -> FileProvenance:
    target = Path(path)
    stat = target.stat()
    return FileProvenance(
        path=str(target),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=sha256_file(target) if include_hash else None,
    )
