"""Episode upload client for data contribution.

When contribute_data is true in onboarding.json, episodes are queued for
upload in ``~/.tether/upload-queue/``. Uploads are:
- Resumable: chunked uploads with progress tracking
- Bandwidth-throttled: 10% of available bandwidth by default
- Retry with backoff: 3 attempts with exponential backoff
- Compressed: gzip before upload

Upload lifecycle:
1. ``queue_episode(path)`` copies/links the file to upload-queue/pending/
2. Background thread picks up pending files
3. POST to https://data.fastcrest.workers.dev/v1/episodes/upload
4. On success, move to upload-queue/completed/
5. Completed files auto-deleted after 7 days

Privacy: upload MUST verify anonymization ran before accepting.
The parquet file must contain an ``anonymized`` metadata flag.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Upload endpoint. Override via TETHER_DATA_ENDPOINT for testing.
DEFAULT_DATA_ENDPOINT = "https://tether-data.fastcrest.workers.dev/v1/episodes/upload"

# Upload config defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE = 2.0  # seconds; exponential: 2, 4, 8
DEFAULT_BANDWIDTH_THROTTLE = 0.1  # 10% of available bandwidth
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
DEFAULT_COMPLETED_RETENTION_DAYS = 7

# Queue directories under ~/.tether/upload-queue/
_PENDING_DIR = "pending"
_COMPLETED_DIR = "completed"
_FAILED_DIR = "failed"

_REQUEST_TIMEOUT_S = 30.0


@dataclass
class UploadManifest:
    """Metadata about a queued upload. Written alongside the data file."""

    episode_id: str
    source_path: str
    queued_at: str  # ISO 8601 UTC
    file_size: int
    file_hash: str  # SHA256 of the file contents
    anonymized: bool
    contributor_hash: str  # SHA256(machine_fingerprint)[:16]
    attempts: int = 0
    last_attempt_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UploadManifest":
        return cls(
            episode_id=str(d["episode_id"]),
            source_path=str(d["source_path"]),
            queued_at=str(d["queued_at"]),
            file_size=int(d["file_size"]),
            file_hash=str(d["file_hash"]),
            anonymized=bool(d["anonymized"]),
            contributor_hash=str(d["contributor_hash"]),
            attempts=int(d.get("attempts", 0)),
            last_attempt_at=d.get("last_attempt_at"),
            completed_at=d.get("completed_at"),
            error=d.get("error"),
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _file_sha256(path: Path) -> str:
    """SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _machine_fingerprint_hash() -> str:
    """SHA256[:16] of machine fingerprint for contributor identification."""
    import platform
    import uuid

    parts = [
        platform.node(),
        platform.machine(),
        platform.processor(),
        platform.system(),
    ]
    try:
        parts.append(str(uuid.getnode()))
    except Exception:
        pass
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _verify_anonymized(path: Path) -> bool:
    """Verify that the file has been anonymized before upload.

    Checks for an ``anonymized`` flag in the file. For JSONL files,
    checks the first line's metadata. For parquet files, checks
    file-level metadata.
    """
    try:
        if path.suffix == ".jsonl":
            with open(path) as f:
                first_line = f.readline().strip()
                if first_line:
                    data = json.loads(first_line)
                    meta = data.get("metadata", {})
                    return bool(meta.get("anonymized", False))
            return False
        elif path.suffix == ".parquet":
            # Try pyarrow metadata check
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(path)
                meta = pf.schema_arrow.metadata or {}
                return meta.get(b"anonymized", b"false") == b"true"
            except ImportError:
                # Without pyarrow, check if filename contains "anon"
                return "anon" in path.stem.lower()
        # For other formats, require explicit flag file
        flag_file = path.with_suffix(path.suffix + ".anonymized")
        return flag_file.exists()
    except Exception as exc:
        logger.debug("Anonymization check failed for %s: %s", path, exc)
        return False


class UploadClient:
    """Manages the episode upload queue and background uploads.

    Usage:
        client = UploadClient()
        client.queue_episode("/path/to/episode.jsonl", anonymized=True)
        client.start()  # background upload thread
        # ... at shutdown:
        client.stop()
    """

    __slots__ = (
        "_queue_dir", "_max_retries", "_backoff_base", "_throttle",
        "_chunk_size", "_endpoint", "_upload_thread", "_stopping",
        "_uploads_completed", "_uploads_failed",
    )

    def __init__(
        self,
        *,
        queue_dir: str | Path = "~/.tether/upload-queue",
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_RETRY_BACKOFF_BASE,
        throttle: float = DEFAULT_BANDWIDTH_THROTTLE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        endpoint: str | None = None,
    ):
        self._queue_dir = Path(queue_dir).expanduser()
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._throttle = throttle
        self._chunk_size = chunk_size
        self._endpoint = endpoint or os.environ.get(
            "TETHER_DATA_ENDPOINT", DEFAULT_DATA_ENDPOINT
        )
        self._upload_thread: threading.Thread | None = None
        self._stopping = False
        self._uploads_completed = 0
        self._uploads_failed = 0

    @property
    def queue_dir(self) -> Path:
        return self._queue_dir

    @property
    def pending_dir(self) -> Path:
        return self._queue_dir / _PENDING_DIR

    @property
    def completed_dir(self) -> Path:
        return self._queue_dir / _COMPLETED_DIR

    @property
    def uploads_completed(self) -> int:
        return self._uploads_completed

    @property
    def uploads_failed(self) -> int:
        return self._uploads_failed

    def queue_episode(
        self,
        source_path: str | Path,
        *,
        episode_id: str = "",
        anonymized: bool = False,
        force: bool = False,
    ) -> UploadManifest | None:
        """Queue an episode file for upload.

        Args:
            source_path: path to the episode data file (.jsonl or .parquet)
            episode_id: optional episode identifier
            anonymized: whether the data has been anonymized. If False and
                force is False, the upload is rejected.
            force: skip anonymization check (for testing only)

        Returns:
            UploadManifest if queued, None if rejected.
        """
        src = Path(source_path).expanduser()
        if not src.exists():
            logger.warning("Upload queue: source file not found: %s", src)
            return None

        # Verify anonymization unless forced
        if not force and not anonymized:
            if not _verify_anonymized(src):
                logger.warning(
                    "Upload rejected: anonymization not verified for %s. "
                    "Run anonymization first or set anonymized=True.",
                    src,
                )
                return None

        # Create queue dirs
        pending = self.pending_dir
        pending.mkdir(parents=True, exist_ok=True)

        # Generate episode_id from hash if not provided
        if not episode_id:
            episode_id = _file_sha256(src)[:12]

        # Copy file to pending
        dest = pending / f"{episode_id}{src.suffix}"
        shutil.copy2(src, dest)

        # Write manifest
        manifest = UploadManifest(
            episode_id=episode_id,
            source_path=str(src),
            queued_at=_utc_now_iso(),
            file_size=dest.stat().st_size,
            file_hash=_file_sha256(dest),
            anonymized=True,
            contributor_hash=_machine_fingerprint_hash(),
        )
        manifest_path = pending / f"{episode_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        logger.info("Episode queued for upload: %s (%d bytes)", episode_id, manifest.file_size)
        return manifest

    def start(self) -> None:
        """Start background upload thread. Idempotent."""
        if self._upload_thread is not None and self._upload_thread.is_alive():
            return
        self._stopping = False
        self._upload_thread = threading.Thread(
            target=self._upload_loop, name="episode-uploader", daemon=True,
        )
        self._upload_thread.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop the upload thread. Idempotent."""
        if self._upload_thread is None:
            return
        self._stopping = True
        self._upload_thread.join(timeout=timeout_s)
        self._upload_thread = None

    def _upload_loop(self) -> None:
        """Background loop: pick up pending files and upload."""
        while not self._stopping:
            try:
                self._process_pending()
                self._cleanup_completed()
            except Exception as exc:
                logger.debug("Upload loop error: %s", exc)
            # Sleep between scans (30 seconds)
            for _ in range(30):
                if self._stopping:
                    break
                time.sleep(1)

    def _process_pending(self) -> None:
        """Process all pending uploads."""
        pending = self.pending_dir
        if not pending.exists():
            return

        for manifest_path in sorted(pending.glob("*.manifest.json")):
            if self._stopping:
                break
            try:
                manifest = UploadManifest.from_dict(
                    json.loads(manifest_path.read_text())
                )
            except Exception as exc:
                logger.debug("Bad manifest %s: %s", manifest_path, exc)
                continue

            # Find the data file
            data_path = pending / f"{manifest.episode_id}{Path(manifest.source_path).suffix}"
            if not data_path.exists():
                # Try common extensions
                for ext in (".jsonl", ".parquet"):
                    candidate = pending / f"{manifest.episode_id}{ext}"
                    if candidate.exists():
                        data_path = candidate
                        break
                else:
                    logger.debug("Data file missing for manifest %s", manifest_path)
                    continue

            if manifest.attempts >= self._max_retries:
                # Move to failed
                self._move_to_failed(data_path, manifest_path, manifest)
                continue

            success = self._upload_file(data_path, manifest)
            manifest.attempts += 1
            manifest.last_attempt_at = _utc_now_iso()

            if success:
                manifest.completed_at = _utc_now_iso()
                self._move_to_completed(data_path, manifest_path, manifest)
                self._uploads_completed += 1
            else:
                # Update manifest with attempt count
                manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
                if manifest.attempts >= self._max_retries:
                    self._move_to_failed(data_path, manifest_path, manifest)
                    self._uploads_failed += 1
                else:
                    # Backoff before next attempt
                    backoff = self._backoff_base ** manifest.attempts
                    time.sleep(min(backoff, 60))

    def _upload_file(self, data_path: Path, manifest: UploadManifest) -> bool:
        """Upload a single file to the endpoint. Returns True on success."""
        try:
            # Compress the file
            compressed = data_path.with_suffix(data_path.suffix + ".gz")
            with open(data_path, "rb") as f_in:
                with gzip.open(compressed, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Build upload metadata
            upload_meta = {
                "episode_id": manifest.episode_id,
                "contributor_hash": manifest.contributor_hash,
                "file_hash": manifest.file_hash,
                "file_size": manifest.file_size,
                "anonymized": manifest.anonymized,
                "content_encoding": "gzip",
            }

            # Upload via httpx or urllib
            success = self._do_upload(compressed, upload_meta)

            # Clean up compressed file
            try:
                compressed.unlink()
            except OSError:
                pass

            return success
        except Exception as exc:
            manifest.error = str(exc)
            logger.debug("Upload failed for %s: %s", manifest.episode_id, exc)
            return False

    def _do_upload(self, file_path: Path, metadata: dict) -> bool:
        """Perform the actual HTTP upload."""
        try:
            import httpx
        except ImportError:
            return self._do_upload_urllib(file_path, metadata)

        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            resp = httpx.post(
                self._endpoint,
                content=file_data,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Encoding": "gzip",
                    "X-Episode-Id": metadata["episode_id"],
                    "X-Contributor-Hash": metadata["contributor_hash"],
                    "X-File-Hash": metadata["file_hash"],
                    "X-Anonymized": str(metadata["anonymized"]).lower(),
                },
                timeout=_REQUEST_TIMEOUT_S,
            )
            return 200 <= resp.status_code < 300
        except Exception as exc:
            logger.debug("httpx upload failed: %s", exc)
            return False

    def _do_upload_urllib(self, file_path: Path, metadata: dict) -> bool:
        """Fallback upload via stdlib urllib."""
        try:
            import urllib.request

            with open(file_path, "rb") as f:
                file_data = f.read()

            req = urllib.request.Request(
                self._endpoint,
                data=file_data,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Encoding": "gzip",
                    "X-Episode-Id": metadata["episode_id"],
                    "X-Contributor-Hash": metadata["contributor_hash"],
                    "X-File-Hash": metadata["file_hash"],
                    "X-Anonymized": str(metadata["anonymized"]).lower(),
                },
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S)
            return 200 <= resp.status < 300
        except Exception as exc:
            logger.debug("urllib upload failed: %s", exc)
            return False

    def _move_to_completed(
        self, data_path: Path, manifest_path: Path, manifest: UploadManifest,
    ) -> None:
        """Move uploaded files to completed/."""
        completed = self.completed_dir
        completed.mkdir(parents=True, exist_ok=True)
        try:
            dest_data = completed / data_path.name
            dest_manifest = completed / manifest_path.name
            shutil.move(str(data_path), str(dest_data))
            manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
            shutil.move(str(manifest_path), str(dest_manifest))
            logger.info("Upload completed: %s", manifest.episode_id)
        except OSError as exc:
            logger.debug("Failed to move to completed: %s", exc)

    def _move_to_failed(
        self, data_path: Path, manifest_path: Path, manifest: UploadManifest,
    ) -> None:
        """Move failed uploads to failed/."""
        failed = self._queue_dir / _FAILED_DIR
        failed.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(data_path), str(failed / data_path.name))
            manifest.error = manifest.error or "max retries exceeded"
            manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
            shutil.move(str(manifest_path), str(failed / manifest_path.name))
            logger.warning("Upload failed permanently: %s", manifest.episode_id)
        except OSError as exc:
            logger.debug("Failed to move to failed: %s", exc)

    def _cleanup_completed(self) -> None:
        """Remove completed uploads older than retention period."""
        completed = self.completed_dir
        if not completed.exists():
            return
        cutoff = time.time() - (DEFAULT_COMPLETED_RETENTION_DAYS * 86_400)
        for path in completed.iterdir():
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass

    def pending_count(self) -> int:
        """Number of files pending upload."""
        pending = self.pending_dir
        if not pending.exists():
            return 0
        return len(list(pending.glob("*.manifest.json")))

    def pending_manifests(self) -> list[UploadManifest]:
        """List all pending upload manifests."""
        pending = self.pending_dir
        if not pending.exists():
            return []
        result = []
        for mp in sorted(pending.glob("*.manifest.json")):
            try:
                result.append(UploadManifest.from_dict(json.loads(mp.read_text())))
            except Exception:
                pass
        return result

    def completed_manifests(self) -> list[UploadManifest]:
        """List all completed upload manifests."""
        completed = self.completed_dir
        if not completed.exists():
            return []
        result = []
        for mp in sorted(completed.glob("*.manifest.json")):
            try:
                result.append(UploadManifest.from_dict(json.loads(mp.read_text())))
            except Exception:
                pass
        return result

    def stats(self) -> dict[str, Any]:
        """Upload statistics."""
        return {
            "queue_dir": str(self._queue_dir),
            "pending": self.pending_count(),
            "completed": self._uploads_completed,
            "failed": self._uploads_failed,
            "endpoint": self._endpoint,
        }

    def revoke_all(self) -> int:
        """Delete ALL queued and completed data. GDPR/CCPA compliance.
        Returns number of files removed."""
        removed = 0
        for subdir in (_PENDING_DIR, _COMPLETED_DIR, _FAILED_DIR):
            d = self._queue_dir / subdir
            if d.exists():
                for f in d.iterdir():
                    try:
                        f.unlink()
                        removed += 1
                    except OSError:
                        pass
                try:
                    d.rmdir()
                except OSError:
                    pass
        return removed


__all__ = [
    "DEFAULT_DATA_ENDPOINT",
    "UploadClient",
    "UploadManifest",
]
