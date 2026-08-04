"""Integrity-checked, atomically-written artifact cache.

Each artifact is two files in the cache dir:
  ``<key>.npz``  — the numpy arrays (the heavy payload)
  ``<key>.json`` — a sidecar with the full spec, a content checksum, the schema
                   version, and a ``complete`` flag.

Reads verify ``complete`` + schema version + checksum; any mismatch is treated as a
miss so a partial/corrupted/stale artifact is never served as real (FR-006, FR-007,
edge cases: interrupted precompute, corrupted cache). Writes are atomic (temp file +
``os.replace``) and the ``complete`` flag is only set once both files are written.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CACHE_DIR, SCHEMA_VERSION
from .keys import _canonical


class CacheStore:
    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Per-process memo of payloads that already passed the (expensive) checksum check,
        # keyed by absolute npz path → (size, mtime_ns). A cache hit then only needs a cheap
        # stat instead of re-hashing the whole (potentially 25 MB+) payload on every read —
        # which is what made the SECOND load of a big animation artifact slow. Any real change
        # to the file changes size/mtime and forces a fresh integrity check.
        self._validated: dict[str, tuple[int, int]] = {}

    def _npz_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    def _json_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def _checksum(npz_bytes: bytes, meta: dict[str, Any]) -> str:
        h = hashlib.sha256()
        h.update(npz_bytes)
        h.update(_canonical(meta).encode("utf-8"))
        return h.hexdigest()

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def stale_schema_version(self, key: str) -> int | None:
        """The schema version of an artifact this build refuses to read, or ``None``.

        ``get`` renders a version mismatch as a plain miss (FR-007), which is right for a
        DERIVED artifact — recompute it and nobody needs to know. It is wrong as an
        *explanation* when the key names something the user made: a `weights_token` is a
        model they trained, and "unknown (never minted here, or evicted)" describes
        neither what happened nor what to do about it. Callers that own such a key use
        this to say "the format moved" instead of guessing at eviction.
        """
        try:
            sidecar = json.loads(self._json_path(key).read_text())
        except (json.JSONDecodeError, OSError):
            return None
        version = sidecar.get("schema_version")
        if isinstance(version, int) and version != SCHEMA_VERSION:
            return version
        return None

    def get(self, key: str) -> dict[str, Any] | None:
        """Return ``{"meta", "arrays", "spec"}`` or ``None`` on miss/invalid."""
        json_path, npz_path = self._json_path(key), self._npz_path(key)
        if not json_path.exists() or not npz_path.exists():
            return None
        try:
            sidecar = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not sidecar.get("complete"):
            return None  # interrupted precompute
        if sidecar.get("schema_version") != SCHEMA_VERSION:
            return None  # format changed -> recompute
        try:
            npz_bytes = npz_path.read_bytes()
            st = npz_path.stat()
        except OSError:
            return None
        meta = sidecar.get("meta", {})
        sig = (st.st_size, st.st_mtime_ns)
        path_key = str(npz_path)
        if self._validated.get(path_key) != sig:
            # not yet validated in this process (or the file changed) -> full integrity check
            if self._checksum(npz_bytes, meta) != sidecar.get("checksum"):
                self._validated.pop(path_key, None)
                return None  # corruption -> recompute
            self._validated[path_key] = sig
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        return {"meta": meta, "arrays": arrays, "spec": sidecar.get("spec", {})}

    def put(
        self,
        key: str,
        spec: dict[str, Any],
        meta: dict[str, Any],
        arrays: dict[str, Any],
    ) -> None:
        buf = io.BytesIO()
        np.savez(buf, **{name: np.asarray(value) for name, value in arrays.items()})
        npz_bytes = buf.getvalue()
        checksum = self._checksum(npz_bytes, meta)
        # Write the heavy payload first; only then write the sidecar that marks the
        # artifact complete. A crash between the two leaves no complete sidecar.
        self._atomic_write_bytes(self._npz_path(key), npz_bytes)
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "spec": spec,
            "meta": meta,
            "checksum": checksum,
            "complete": True,
        }
        self._atomic_write_bytes(self._json_path(key), _canonical(sidecar).encode("utf-8"))

    def delete(self, key: str) -> None:
        for path in (self._npz_path(key), self._json_path(key)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self.cache_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
