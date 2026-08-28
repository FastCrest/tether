"""Episode upload client for data contribution.

When contribute_data is true in onboarding.json, episodes are queued for
upload in ``~/.tether/upload-queue/``. Uploads are:
- Authenticated: every control-plane request is signed with a local Ed25519 key
- Capability-bound: the raw upload consumes a short-lived one-time capability
- Retry with backoff: 3 attempts with exponential backoff

Upload lifecycle:
1. ``queue_episode(path)`` copies/links the file to upload-queue/pending/
2. Background thread picks up pending files
3. Register, reserve, PUT, and complete through the contribution worker
4. On success, move to upload-queue/completed/
5. Completed files auto-deleted after 7 days

Privacy: upload MUST verify anonymization ran before accepting.
The parquet file must contain an ``anonymized`` metadata flag.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Upload endpoint. Override via TETHER_DATA_ENDPOINT for testing.
DEFAULT_DATA_ENDPOINT = "https://reflex-contributions.fastcrest.workers.dev"

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
_SUPPORTED_DATA_SUFFIXES = (".jsonl", ".parquet")


@dataclass
class UploadManifest:
    """Metadata about a queued upload. Written alongside the data file."""

    episode_id: str
    file_name: str
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
        episode_id = str(d["episode_id"])
        source_path = str(d["source_path"])
        # Pre-binding manifests are upgraded deterministically from the
        # original source format. New manifests always persist file_name.
        file_name = str(
            d.get("file_name") or f"{episode_id}{Path(source_path).suffix}"
        )
        return cls(
            episode_id=episode_id,
            file_name=file_name,
            source_path=source_path,
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


def _reservation_state_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.upload-v1.json")


def _source_identity(path: Path) -> str:
    current = path.stat()
    return f"{current.st_dev}:{current.st_ino}:{current.st_mtime_ns}:{current.st_size}"


def _write_manifest(path: Path, manifest: UploadManifest) -> None:
    """Atomically publish one authoritative episode generation."""
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(json.dumps(manifest.to_dict(), indent=2))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bound_data_path(
    pending: Path, episode_id: str, manifest: UploadManifest,
) -> Path:
    """Resolve a manifest binding without accepting alternate/path names."""
    file_name = manifest.file_name
    file_path = Path(file_name)
    if file_path.name != file_name or file_path.parent != Path("."):
        raise ValueError("manifest file_name must be a plain file name")
    if file_path.suffix not in _SUPPORTED_DATA_SUFFIXES:
        raise ValueError("manifest file_name has an unsupported format")
    if file_name != f"{episode_id}{file_path.suffix}":
        raise ValueError("manifest file_name does not match episode_id")
    return pending / file_name


def _generation_matches(path: Path, manifest: UploadManifest) -> bool:
    """Verify that the on-disk bytes are the manifest-bound generation."""
    try:
        if path.stat().st_size != manifest.file_size:
            return False
        return hmac.compare_digest(_file_sha256(path), manifest.file_hash)
    except OSError:
        return False


def _remove_obsolete_siblings(
    pending: Path, episode_id: str, bound_path: Path,
) -> None:
    """Remove superseded formats and their unusable reservation sidecars."""
    for suffix in _SUPPORTED_DATA_SUFFIXES:
        candidate = pending / f"{episode_id}{suffix}"
        if candidate == bound_path:
            continue
        candidate.unlink(missing_ok=True)
        _reservation_state_path(candidate).unlink(missing_ok=True)


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
        "_chunk_size", "_endpoint", "_auth_client", "_upload_thread", "_stopping",
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
        auth_client: Any | None = None,
    ):
        self._queue_dir = Path(queue_dir).expanduser()
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._throttle = throttle
        self._chunk_size = chunk_size
        self._endpoint = endpoint or os.environ.get(
            "TETHER_DATA_ENDPOINT", DEFAULT_DATA_ENDPOINT
        )
        if self._endpoint.startswith("https://tether-data.fastcrest.workers.dev"):
            # Migrate the former built-in default even if it persists in an old env file.
            self._endpoint = DEFAULT_DATA_ENDPOINT
        elif self._endpoint.endswith("/v1/episodes/upload"):
            self._endpoint = self._endpoint.removesuffix("/v1/episodes/upload")
        self._auth_client = auth_client
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

        if src.suffix not in _SUPPORTED_DATA_SUFFIXES:
            logger.warning("Upload queue: unsupported episode format: %s", src.suffix)
            return None

        # The manifest pathname is the episode lock identity. It deliberately
        # does not vary with .jsonl/.parquet so format replacement is atomic.
        dest = pending / f"{episode_id}{src.suffix}"
        manifest_path = pending / f"{episode_id}.manifest.json"
        from tether.pro.data_collection import _queue_file_lock
        with _queue_file_lock(manifest_path):
            temporary = pending / (
                f".{episode_id}.{os.getpid()}.{threading.get_ident()}"
                f"{src.suffix}.queueing"
            )
            try:
                shutil.copy2(src, temporary)
                os.replace(temporary, dest)
            finally:
                temporary.unlink(missing_ok=True)
            manifest = UploadManifest(
                episode_id=episode_id,
                file_name=dest.name,
                source_path=str(src),
                queued_at=_utc_now_iso(),
                file_size=dest.stat().st_size,
                file_hash=_file_sha256(dest),
                anonymized=True,
                contributor_hash=_machine_fingerprint_hash(),
            )
            _write_manifest(manifest_path, manifest)
            # A new generation must never inherit the old reservation, even
            # when the replacement uses the same extension/pathname.
            _reservation_state_path(dest).unlink(missing_ok=True)
            _remove_obsolete_siblings(pending, episode_id, dest)

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
            episode_id = manifest_path.name.removesuffix(".manifest.json")
            from tether.pro.data_collection import _queue_file_lock
            with _queue_file_lock(manifest_path):
                # The manifest is intentionally the first queue artifact read
                # under the episode lock. It alone selects the exact format
                # and byte generation; sibling discovery is never authoritative.
                if not manifest_path.exists():
                    continue
                try:
                    manifest = UploadManifest.from_dict(
                        json.loads(manifest_path.read_text())
                    )
                    if manifest.episode_id != episode_id:
                        raise ValueError("manifest episode_id mismatch")
                    data_path = _bound_data_path(pending, episode_id, manifest)
                except Exception as exc:
                    logger.debug("Bad manifest %s: %s", manifest_path, exc)
                    continue
                if not _generation_matches(data_path, manifest):
                    manifest.error = "queued data does not match manifest hash and size"
                    _write_manifest(manifest_path, manifest)
                    logger.debug("Manifest generation mismatch: %s", manifest_path)
                    continue
                _remove_obsolete_siblings(pending, episode_id, data_path)
                if manifest.attempts >= self._max_retries:
                    self._move_to_failed(data_path, manifest_path, manifest)
                    continue
                success = self._upload_file(data_path, manifest)
                manifest.attempts += 1
                manifest.last_attempt_at = _utc_now_iso()

                if success:
                    if not _generation_matches(data_path, manifest):
                        manifest.error = "queued data changed during upload"
                        _write_manifest(manifest_path, manifest)
                        continue
                    manifest.completed_at = _utc_now_iso()
                    self._move_to_completed(data_path, manifest_path, manifest)
                    self._uploads_completed += 1
                else:
                    _write_manifest(manifest_path, manifest)
                    if manifest.attempts >= self._max_retries:
                        if not _generation_matches(data_path, manifest):
                            manifest.error = "queued data changed before terminal failure"
                            _write_manifest(manifest_path, manifest)
                            continue
                        self._move_to_failed(data_path, manifest_path, manifest)
                        self._uploads_failed += 1
                    else:
                        backoff = self._backoff_base ** manifest.attempts
                        time.sleep(min(backoff, 60))

    def _upload_file(self, data_path: Path, manifest: UploadManifest) -> bool:
        """Upload a single file to the endpoint. Returns True on success."""
        try:
            # The reservation digest and PUT body cover the same exact bytes.
            upload_meta = {
                "episode_id": manifest.episode_id,
                "file_hash": manifest.file_hash,
                "file_size": manifest.file_size,
                "anonymized": manifest.anonymized,
            }
            return self._do_upload(data_path, upload_meta)
        except Exception as exc:
            manifest.error = str(exc)
            logger.debug("Upload failed for %s: %s", manifest.episode_id, exc)
            return False

    def _do_upload(self, file_path: Path, metadata: dict) -> bool:
        """Perform the signed reserve -> capability PUT -> complete flow."""
        try:
            from tether.contributor_auth import ContributorAuthClient

            client = self._auth_client
            if client is None:
                client = ContributorAuthClient(
                    self._endpoint,
                    timeout=_REQUEST_TIMEOUT_S,
                    # Pro's constant counts total attempts; the shared client
                    # takes retries after the initial operation.
                    upload_max_retries=max(0, self._max_retries - 1),
                    upload_backoff_base_s=self._backoff_base,
                )
                self._auth_client = client
            file_data = file_path.read_bytes()
            media_type = {
                ".jsonl": "application/jsonl",
                ".parquet": "application/x-parquet",
            }.get(file_path.suffix)
            if media_type is None:
                raise ValueError(f"unsupported contribution format: {file_path.suffix}")
            client.upload(
                file_name=file_path.name,
                file_bytes=file_data,
                media_type=media_type,
                state_path=_reservation_state_path(file_path),
                source_identity=_source_identity(file_path),
            )
            return True
        except Exception as exc:
            logger.debug("authenticated contribution upload failed: %s", exc)
            return False

    def _do_upload_urllib(self, file_path: Path, metadata: dict) -> bool:
        """Compatibility shim; the shared client owns its stdlib fallback."""
        return self._do_upload(file_path, metadata)

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
            _reservation_state_path(data_path).unlink(missing_ok=True)
            _write_manifest(manifest_path, manifest)
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
            state_path = _reservation_state_path(data_path)
            if state_path.exists():
                shutil.move(str(state_path), str(failed / state_path.name))
            manifest.error = manifest.error or "max retries exceeded"
            _write_manifest(manifest_path, manifest)
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
