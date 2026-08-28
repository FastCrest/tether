"""Contributor Authentication v1 client and local Ed25519 credentials.

The private key is generated on first use and stored only in the local
credential file with owner-only permissions.  Consent receipts and upload
manifests contain only the derived public contributor identifier.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlsplit
from urllib.parse import quote

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

AUTH_DOMAIN = "tether.contrib.request"
AUTH_VERSION = "1"
DEFAULT_CONTRIBUTION_WORKER = "https://reflex-contributions.fastcrest.workers.dev"
DEFAULT_CREDENTIAL_PATH = "~/.tether/contributor-auth-v1.json"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DEFAULT_UPLOAD_MAX_RETRIES = 2
DEFAULT_UPLOAD_BACKOFF_BASE_S = 2.0
MAX_UPLOAD_BACKOFF_S = 30.0
TRANSIENT_STATUS_CODES = frozenset({500, 502, 503, 504})
UPLOAD_STATE_SCHEMA_VERSION = 1
DEFAULT_STATE_LOCK_TIMEOUT_S = 30.0
STATE_LOCK_POLL_S = 0.05


class ContributorAuthError(RuntimeError):
    """A signed contributor request failed."""

    def __init__(self, status: int, body: dict[str, Any] | None = None):
        self.status = status
        self.body = body or {}
        super().__init__(f"contributor_auth status={status} body={self.body}")


class ContributorTransportError(RuntimeError):
    """The request outcome is unknown because its transport failed."""


class _UploadStateLock:
    """Crash-safe exclusive election for one persistent upload state path.

    The lock inode is never deleted, so a waiter cannot switch to a new inode
    while a live owner still holds the old one. Kernel advisory locks are
    released automatically if the process exits or crashes.
    """

    def __init__(self, state_path: Path, timeout_s: float):
        self.path = state_path.with_name(f"{state_path.name}.lock")
        self.timeout_s = max(0.0, float(timeout_s))
        self.fd: int | None = None

    def __enter__(self) -> _UploadStateLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.path, flags, 0o600)
        self.fd = fd
        try:
            mode = os.fstat(fd).st_mode
            if not stat.S_ISREG(mode):
                raise PermissionError(f"upload state lock is not a regular file: {self.path}")
            if os.name == "posix":
                os.fchmod(fd, 0o600)
            else:
                # msvcrt.locking requires at least one byte to lock.
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                    os.fsync(fd)
            self._acquire(fd)
            return self
        except Exception:
            os.close(fd)
            self.fd = None
            raise

    def _acquire(self, fd: int) -> None:
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
                return
            except (BlockingIOError, OSError):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for upload state lock: {self.path}"
                    ) from None
                time.sleep(min(STATE_LOCK_POLL_S, remaining))

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.fd is None:
            return
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(self.fd, fcntl.LOCK_UN)
            else:
                import msvcrt

                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        finally:
            os.close(self.fd)
            self.fd = None


@dataclass(frozen=True)
class ContributorCredentials:
    private_key_seed: bytes = field(repr=False)
    public_key: str
    contributor_id: str
    key_id: str


@dataclass(frozen=True)
class _Response:
    status_code: int
    content: bytes

    def json(self) -> dict[str, Any]:
        value = json.loads(self.content.decode("utf-8"))
        return value if isinstance(value, dict) else {"value": value}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


Transport = Callable[[str, str, bytes, dict[str, str], float], Any]


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value) or "=" in value:
        raise ValueError("base64url value must be non-empty and unpadded")
    decoded = base64.urlsafe_b64decode(value + "=" * ((4 - len(value) % 4) % 4))
    if _b64url(decoded) != value:
        raise ValueError("base64url value is not canonical")
    return decoded


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize the integer/string-only protocol values in RFC 8785 form."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def derive_identifiers(public_key: bytes) -> tuple[str, str]:
    if len(public_key) != 32:
        raise ValueError("Ed25519 public key must be 32 bytes")
    prefix = hashlib.sha256(public_key).hexdigest()[:32]
    return f"ctr_{prefix}", f"key_{prefix}"


def generate_credentials() -> ContributorCredentials:
    private = Ed25519PrivateKey.generate()
    seed = private.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    contributor_id, key_id = derive_identifiers(public)
    return ContributorCredentials(seed, _b64url(public), contributor_id, key_id)


def load_or_create_credentials(path: str | Path | None = None) -> ContributorCredentials:
    configured = path or os.environ.get("TETHER_CONTRIBUTOR_CREDENTIALS", DEFAULT_CREDENTIAL_PATH)
    credential_path = Path(configured).expanduser()
    if credential_path.exists():
        if os.name == "posix" and stat.S_IMODE(credential_path.stat().st_mode) & 0o077:
            raise PermissionError(
                f"contributor credential file must be owner-only: {credential_path}"
            )
        data = json.loads(credential_path.read_text(encoding="utf-8"))
        seed = _b64url_decode(str(data["private_key_seed"]))
        if len(seed) != 32:
            raise ValueError("invalid contributor private key seed")
        private = Ed25519PrivateKey.from_private_bytes(seed)
        public = private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        contributor_id, key_id = derive_identifiers(public)
        if data.get("public_key") != _b64url(public):
            raise ValueError("credential public key does not match private key")
        if data.get("contributor_id") != contributor_id or data.get("key_id") != key_id:
            raise ValueError("credential identifiers do not match public key")
        return ContributorCredentials(seed, _b64url(public), contributor_id, key_id)

    credential_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(credential_path.parent, 0o700)
    credentials = generate_credentials()
    payload = (
        canonical_json_bytes(
            {
                "contributor_id": credentials.contributor_id,
                "key_id": credentials.key_id,
                "private_key_seed": _b64url(credentials.private_key_seed),
                "public_key": credentials.public_key,
                "schema_version": 1,
            }
        )
        + b"\n"
    )
    fd, temporary = tempfile.mkstemp(prefix=f".{credential_path.name}.", dir=credential_path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        try:
            # Atomic no-replace publication: concurrent first-use processes
            # converge on the one credential that actually reached the path.
            os.link(temporary, credential_path)
        except FileExistsError:
            return load_or_create_credentials(credential_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return credentials


def signed_headers(
    credentials: ContributorCredentials,
    method: str,
    url: str,
    body: bytes = b"",
    *,
    timestamp: int | None = None,
    nonce: bytes | None = None,
) -> dict[str, str]:
    parsed = urlsplit(url)
    timestamp = int(time.time()) if timestamp is None else timestamp
    nonce = secrets.token_bytes(16) if nonce is None else nonce
    if len(nonce) != 16:
        raise ValueError("Contributor Auth nonce must be 16 bytes")
    digest = hashlib.sha256(body).hexdigest()
    envelope = {
        "body_sha256": digest,
        "contributor_id": credentials.contributor_id,
        "domain": AUTH_DOMAIN,
        "key_id": credentials.key_id,
        "method": method.upper(),
        "nonce": _b64url(nonce),
        "path": parsed.path or "/",
        "query": sorted(parse_qsl(parsed.query, keep_blank_values=True)),
        "timestamp": timestamp,
        "v": 1,
    }
    signature = Ed25519PrivateKey.from_private_bytes(credentials.private_key_seed).sign(
        canonical_json_bytes(envelope)
    )
    return {
        "X-Tether-Auth-Version": AUTH_VERSION,
        "X-Tether-Contributor-Id": credentials.contributor_id,
        "X-Tether-Key-Id": credentials.key_id,
        "X-Tether-Timestamp": str(timestamp),
        "X-Tether-Nonce": _b64url(nonce),
        "X-Tether-Content-SHA256": digest,
        "X-Tether-Signature": _b64url(signature),
    }


def _default_transport(
    method: str, url: str, body: bytes, headers: dict[str, str], timeout: float
) -> Any:
    try:
        import httpx

        return httpx.request(method, url, content=body, headers=headers, timeout=timeout)
    except ImportError:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, data=body or None, headers=headers, method=method)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
            return _Response(response.status, response.read())
        except urllib.error.HTTPError as error:
            return _Response(error.code, error.read())


class ContributorAuthClient:
    """Signed registration, reservation, capability upload, and completion."""

    def __init__(
        self,
        worker_url: str = DEFAULT_CONTRIBUTION_WORKER,
        *,
        credentials: ContributorCredentials | None = None,
        credential_path: str | Path | None = None,
        transport: Transport | None = None,
        timeout: float = 30.0,
        upload_max_retries: int = DEFAULT_UPLOAD_MAX_RETRIES,
        upload_backoff_base_s: float = DEFAULT_UPLOAD_BACKOFF_BASE_S,
        state_lock_timeout_s: float = DEFAULT_STATE_LOCK_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.worker_url = worker_url.rstrip("/")
        self.credentials = credentials or load_or_create_credentials(credential_path)
        self.transport = transport or _default_transport
        self.timeout = timeout
        self.upload_max_retries = max(0, int(upload_max_retries))
        self.upload_backoff_base_s = max(0.0, float(upload_backoff_base_s))
        self.state_lock_timeout_s = max(0.0, float(state_lock_timeout_s))
        self._sleep = sleep
        self._registered = False

    def _request(
        self,
        method: str,
        path_or_url: str,
        body: bytes = b"",
        *,
        extra_headers: dict[str, str] | None = None,
        signed: bool = True,
    ) -> Any:
        url = (
            path_or_url
            if path_or_url.startswith(("http://", "https://"))
            else f"{self.worker_url}{path_or_url}"
        )
        headers = signed_headers(self.credentials, method, url, body) if signed else {}
        headers.update(extra_headers or {})
        try:
            return self.transport(method, url, body, headers, self.timeout)
        except Exception as error:
            raise ContributorTransportError(
                f"contributor transport failed for {method.upper()} {urlsplit(url).path}"
            ) from error

    @staticmethod
    def _json(response: Any) -> dict[str, Any]:
        try:
            return response.json()
        except Exception:
            return {"raw": getattr(response, "text", "")[:500]}

    def register(self) -> dict[str, Any]:
        response = self._request(
            "POST",
            "/v1/contributors/register",
            extra_headers={"X-Tether-Public-Key": self.credentials.public_key},
        )
        body = self._json(response)
        if response.status_code not in (200, 201):
            raise ContributorAuthError(response.status_code, body)
        if body.get("contributor_id") != self.credentials.contributor_id:
            raise ContributorAuthError(
                response.status_code, {"error": "registration_identity_mismatch"}
            )
        self._registered = True
        return body

    def ensure_registered(self) -> None:
        if not self._registered:
            self.register()

    def reserve(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        media_type: str = "application/jsonl",
        removed_fields: dict[str, int] | None = None,
        anonymizer_version: str = "tether-anonymizer-v1",
        scanner_version: str = "tether-scanner-v1",
        reservation_key: str | None = None,
        scan_timestamp: int | None = None,
        canonical_body: bytes | None = None,
    ) -> dict[str, Any]:
        if not file_bytes or len(file_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload size must be between 1 and {MAX_UPLOAD_BYTES} bytes")
        self.ensure_registered()
        reservation_key = reservation_key or _b64url(secrets.token_bytes(32))
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", reservation_key):
            raise ValueError("reservation_key must be 32-byte unpadded base64url")
        payload = {
            "byte_size": len(file_bytes),
            "file_name": file_name,
            "reservation_key": reservation_key,
            "manifest": {
                "anonymizer_version": anonymizer_version,
                "content_sha256": hashlib.sha256(file_bytes).hexdigest(),
                "domain": "tether.anonymization.manifest",
                "media_type": media_type,
                "removed_fields": removed_fields or {"email": 0, "face": 0, "name": 0},
                "scan_timestamp": int(time.time()) if scan_timestamp is None else scan_timestamp,
                "scanner_version": scanner_version,
                "schema_version": 1,
            },
        }
        body = canonical_body if canonical_body is not None else canonical_json_bytes(payload)
        if canonical_body is not None:
            persisted = json.loads(canonical_body)
            if persisted != payload or canonical_json_bytes(persisted) != canonical_body:
                raise ValueError("persisted reservation body is not the expected canonical request")
        response = self._request(
            "POST",
            "/v1/uploads/sign",
            body,
            extra_headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        result = self._json(response)
        if response.status_code != 200:
            raise ContributorAuthError(response.status_code, result)
        self._validate_reservation(result)
        result["reservation_key"] = reservation_key
        return result

    def _validate_reservation(
        self, result: dict[str, Any], *, require_capability: bool = True
    ) -> None:
        upload_id = str(result.get("upload_id", ""))
        capability = str(result.get("upload_capability", ""))
        put_url = str(result.get("put_url", ""))
        base = urlsplit(self.worker_url)
        put = urlsplit(put_url)
        if not re.fullmatch(r"upl_[0-9a-f]{32}", upload_id):
            raise ContributorAuthError(200, {"error": "invalid_reservation_upload_id"})
        if require_capability:
            try:
                capability_bytes = _b64url_decode(capability)
            except ValueError as exc:
                raise ContributorAuthError(
                    200, {"error": "invalid_reservation_capability"}
                ) from exc
            if len(capability_bytes) != 32:
                raise ContributorAuthError(200, {"error": "invalid_reservation_capability"})
        if (
            put.scheme != base.scheme
            or put.netloc != base.netloc
            or put.path != f"/v1/uploads/put/{upload_id}"
            or put.query
            or put.fragment
        ):
            raise ContributorAuthError(200, {"error": "invalid_reservation_put_url"})

    @staticmethod
    def _reservation_expiry(reservation: dict[str, Any]) -> float:
        value = str(reservation.get("expires_at", ""))
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as error:
            raise ContributorAuthError(200, {"error": "invalid_reservation_expiry"}) from error

    def _state_record(
        self,
        reservation: dict[str, Any],
        *,
        file_name: str,
        file_bytes: bytes,
        media_type: str,
        phase: str,
        source_identity: str | None = None,
    ) -> dict[str, Any]:
        return {
            "content_sha256": hashlib.sha256(file_bytes).hexdigest(),
            "contributor_id": self.credentials.contributor_id,
            "file_name": file_name,
            "media_type": media_type,
            "phase": phase,
            "reservation": reservation,
            "schema_version": UPLOAD_STATE_SCHEMA_VERSION,
            "source_identity": source_identity,
            "worker_url": self.worker_url,
        }

    @staticmethod
    def _write_state(path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes(state) + b"\n"
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
            if os.name == "posix":
                os.chmod(path, 0o600)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _load_state(
        self,
        path: Path,
        *,
        file_name: str,
        file_bytes: bytes,
        media_type: str,
        source_identity: str | None = None,
    ) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise PermissionError(f"upload reservation state must be owner-only: {path}")
        state = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "content_sha256": hashlib.sha256(file_bytes).hexdigest(),
            "contributor_id": self.credentials.contributor_id,
            "file_name": file_name,
            "media_type": media_type,
            "schema_version": UPLOAD_STATE_SCHEMA_VERSION,
            "worker_url": self.worker_url,
        }
        expected["source_identity"] = source_identity
        if state.get("phase") == "completed" and any(
            state.get(key) != value for key, value in expected.items()
        ):
            # A completed marker belongs to an older filesystem generation.
            # Replacing it under the same lock permits a legitimately new file
            # at the same queue path without weakening concurrent election.
            self._remove_state(path)
            return None
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"upload reservation state mismatch: {key}")
        if state.get("phase") not in {"reserving", "reserved", "uploaded", "completed"}:
            raise ValueError("invalid upload reservation state phase")
        reservation = state.get("reservation")
        if state.get("phase") == "reserving":
            if (
                reservation is not None
                or not re.fullmatch(r"[A-Za-z0-9_-]{43}", str(state.get("reservation_key", "")))
                or not isinstance(state.get("scan_timestamp"), int)
            ):
                raise ValueError("invalid reserving upload state")
            sign_body = str(state.get("sign_body", "")).encode("utf-8")
            try:
                if canonical_json_bytes(json.loads(sign_body)) != sign_body:
                    raise ValueError
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError("invalid canonical reserving request body") from error
            return state
        if not isinstance(reservation, dict):
            raise ValueError("invalid upload reservation state")
        expiry = self._reservation_expiry(reservation)
        self._validate_reservation(
            reservation,
            require_capability=state.get("phase") != "completed" and expiry > time.time(),
        )
        if expiry <= time.time() and reservation.get("upload_capability"):
            # The capability cannot be consumed after its server-owned expiry.
            # Redact it at rest, but retain upload_id for signed completion and
            # recovery reconciliation before any replacement reservation.
            reservation = dict(reservation)
            reservation["upload_capability"] = ""
            state = dict(state)
            state["reservation"] = reservation
            self._write_state(path, state)
        return state

    def _mark_completed_state(
        self,
        path: Path | None,
        reservation: dict[str, Any],
        *,
        file_name: str,
        file_bytes: bytes,
        media_type: str,
        result: dict[str, Any],
        source_identity: str | None,
    ) -> None:
        if path is None:
            return
        redacted = dict(reservation)
        redacted["upload_capability"] = ""
        state = self._state_record(
            redacted,
            file_name=file_name,
            file_bytes=file_bytes,
            media_type=media_type,
            phase="completed",
            source_identity=source_identity,
        )
        state["completion_result"] = result
        self._write_state(path, state)

    @staticmethod
    def _remove_state(path: Path | None) -> None:
        if path is not None:
            path.unlink(missing_ok=True)

    def put(self, reservation: dict[str, Any], file_bytes: bytes, *, media_type: str) -> int:
        capability = str(reservation.get("upload_capability", ""))
        if not capability:
            raise ValueError("reservation omitted upload capability")
        response = self._request(
            "PUT",
            str(reservation["put_url"]),
            file_bytes,
            signed=False,
            extra_headers={
                "Authorization": f"Upload {capability}",
                "Content-Length": str(len(file_bytes)),
                "Content-Type": media_type,
            },
        )
        result = self._json(response)
        if response.status_code != 200:
            raise ContributorAuthError(response.status_code, result)
        return int(result.get("bytes_received", len(file_bytes)))

    def complete(self, upload_id: str) -> dict[str, Any]:
        body = canonical_json_bytes({"upload_id": upload_id})
        response = self._request(
            "POST",
            "/v1/uploads/complete",
            body,
            extra_headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        result = self._json(response)
        if response.status_code != 200:
            raise ContributorAuthError(response.status_code, result)
        return result

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        return isinstance(error, ContributorTransportError) or (
            isinstance(error, ContributorAuthError) and error.status in TRANSIENT_STATUS_CODES
        )

    @staticmethod
    def _is_ambiguous_put(error: Exception) -> bool:
        return isinstance(error, ContributorTransportError) or (
            isinstance(error, ContributorAuthError)
            and (error.status in TRANSIENT_STATUS_CODES or error.status == 409)
        )

    def _backoff(self, retry_index: int) -> None:
        delay = min(
            self.upload_backoff_base_s * (2**retry_index),
            MAX_UPLOAD_BACKOFF_S,
        )
        if delay > 0:
            self._sleep(delay)

    def _complete_with_retry(self, upload_id: str) -> dict[str, Any]:
        """Retry only the idempotent completion mutation after transient loss."""
        for retry_index in range(self.upload_max_retries + 1):
            try:
                return self.complete(upload_id)
            except Exception as error:
                if not self._is_transient(error) or retry_index >= self.upload_max_retries:
                    raise
                self._backoff(retry_index)
        raise AssertionError("unreachable completion retry state")

    def _reconcile_ambiguous_put(
        self,
        reservation: dict[str, Any],
        file_bytes: bytes,
        *,
        media_type: str,
        initial_error: Exception,
    ) -> dict[str, Any]:
        """Resolve a lost PUT response without creating another reservation.

        Completion is both an idempotent mutation and the authenticated state
        probe available in Upload Reservation v1. If it reports ``pending``,
        the original PUT never claimed the capability and the same reservation
        can safely be retried. If it reports ``uploading``, another PUT remains
        in flight, so only completion is retried.
        """
        upload_id = str(reservation["upload_id"])
        last_error: Exception = initial_error
        for retry_index in range(self.upload_max_retries + 1):
            try:
                return self.complete(upload_id)
            except ContributorAuthError as error:
                last_error = error
                if error.status == 409 and error.body.get("error") == "upload_not_ready":
                    status = error.body.get("status")
                    if status == "pending":
                        try:
                            self.put(reservation, file_bytes, media_type=media_type)
                            return self._complete_with_retry(upload_id)
                        except Exception as put_error:
                            if not self._is_ambiguous_put(put_error):
                                raise
                            last_error = put_error
                    elif status != "uploading":
                        raise
                elif not self._is_transient(error):
                    raise
            except Exception as error:
                last_error = error
                if not self._is_transient(error):
                    raise

            if retry_index >= self.upload_max_retries:
                raise last_error
            self._backoff(retry_index)
        raise AssertionError("unreachable PUT reconciliation state")

    def upload(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        media_type: str,
        state_path: str | Path | None = None,
        source_identity: str | None = None,
    ) -> dict[str, Any]:
        persistent_path = (
            Path(state_path).expanduser().resolve(strict=False) if state_path is not None else None
        )
        if persistent_path is not None:
            with _UploadStateLock(persistent_path, self.state_lock_timeout_s):
                return self._upload_locked(
                    file_name=file_name,
                    file_bytes=file_bytes,
                    media_type=media_type,
                    persistent_path=persistent_path,
                    source_identity=source_identity,
                )
        return self._upload_locked(
            file_name=file_name,
            file_bytes=file_bytes,
            media_type=media_type,
            persistent_path=None,
            source_identity=source_identity,
        )

    @staticmethod
    def _fresh_sign_request(
        file_name: str,
        file_bytes: bytes,
        media_type: str,
    ) -> tuple[str, int, bytes]:
        reservation_key = _b64url(secrets.token_bytes(32))
        scan_timestamp = int(time.time())
        sign_body = canonical_json_bytes(
            {
                "byte_size": len(file_bytes),
                "file_name": file_name,
                "manifest": {
                    "anonymizer_version": "tether-anonymizer-v1",
                    "content_sha256": hashlib.sha256(file_bytes).hexdigest(),
                    "domain": "tether.anonymization.manifest",
                    "media_type": media_type,
                    "removed_fields": {"email": 0, "face": 0, "name": 0},
                    "scan_timestamp": scan_timestamp,
                    "scanner_version": "tether-scanner-v1",
                    "schema_version": 1,
                },
                "reservation_key": reservation_key,
            }
        )
        return reservation_key, scan_timestamp, sign_body

    def _persist_reserving_request(
        self,
        path: Path,
        *,
        file_name: str,
        file_bytes: bytes,
        media_type: str,
        source_identity: str | None,
        reservation_key: str,
        scan_timestamp: int,
        sign_body: bytes,
    ) -> None:
        reserving = self._state_record(
            {},
            file_name=file_name,
            file_bytes=file_bytes,
            media_type=media_type,
            phase="reserving",
            source_identity=source_identity,
        )
        reserving.update(
            {
                "reservation": None,
                "reservation_key": reservation_key,
                "scan_timestamp": scan_timestamp,
                "sign_body": sign_body.decode("utf-8"),
            }
        )
        self._write_state(path, reserving)

    def _upload_locked(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        media_type: str,
        persistent_path: Path | None,
        source_identity: str | None,
    ) -> dict[str, Any]:
        state = (
            self._load_state(
                persistent_path,
                file_name=file_name,
                file_bytes=file_bytes,
                media_type=media_type,
                source_identity=source_identity,
            )
            if persistent_path is not None
            else None
        )
        if state is not None:
            if state["phase"] == "reserving":
                assert persistent_path is not None
                try:
                    reservation = self.reserve(
                        file_name=file_name,
                        file_bytes=file_bytes,
                        media_type=media_type,
                        reservation_key=str(state["reservation_key"]),
                        scan_timestamp=int(state["scan_timestamp"]),
                        canonical_body=str(state["sign_body"]).encode("utf-8"),
                    )
                except ContributorAuthError as error:
                    if not (error.status == 400 and error.body.get("error") == "stale_manifest"):
                        raise
                    # The Worker checks committed idempotent replay before
                    # manifest freshness. This response therefore proves no
                    # reservation exists for the old key, so rotating the
                    # request cannot double quota.
                    reservation_key, scan_timestamp, sign_body = self._fresh_sign_request(
                        file_name,
                        file_bytes,
                        media_type,
                    )
                    self._persist_reserving_request(
                        persistent_path,
                        file_name=file_name,
                        file_bytes=file_bytes,
                        media_type=media_type,
                        source_identity=source_identity,
                        reservation_key=reservation_key,
                        scan_timestamp=scan_timestamp,
                        sign_body=sign_body,
                    )
                    reservation = self.reserve(
                        file_name=file_name,
                        file_bytes=file_bytes,
                        media_type=media_type,
                        reservation_key=reservation_key,
                        scan_timestamp=scan_timestamp,
                        canonical_body=sign_body,
                    )
                self._write_state(
                    persistent_path,
                    self._state_record(
                        reservation,
                        file_name=file_name,
                        file_bytes=file_bytes,
                        media_type=media_type,
                        phase="reserved",
                        source_identity=source_identity,
                    ),
                )
                state = {"phase": "reserved", "reservation": reservation}
            reservation = state["reservation"]
            if state["phase"] == "completed":
                result = state.get("completion_result")
                if isinstance(result, dict):
                    return result
                return {
                    "idempotent": True,
                    "local_reconciled": True,
                    "status": "completed",
                    "upload_id": reservation["upload_id"],
                }
            try:
                if state["phase"] == "uploaded":
                    result = self._complete_with_retry(str(reservation["upload_id"]))
                else:
                    result = self._reconcile_ambiguous_put(
                        reservation,
                        file_bytes,
                        media_type=media_type,
                        initial_error=ContributorTransportError(
                            "resuming persisted upload reservation"
                        ),
                    )
            except ContributorAuthError as error:
                if not (
                    error.status == 410 and error.body.get("error") == "upload_reservation_expired"
                ):
                    raise
                # A server-confirmed pending/expired reservation cannot own an
                # object. It is safe to discard and reserve again.
                self._remove_state(persistent_path)
            else:
                self._mark_completed_state(
                    persistent_path,
                    reservation,
                    file_name=file_name,
                    file_bytes=file_bytes,
                    media_type=media_type,
                    result=result,
                    source_identity=source_identity,
                )
                return result

        reservation_key, scan_timestamp, sign_body = self._fresh_sign_request(
            file_name,
            file_bytes,
            media_type,
        )
        if persistent_path is not None:
            self._persist_reserving_request(
                persistent_path,
                file_name=file_name,
                file_bytes=file_bytes,
                media_type=media_type,
                source_identity=source_identity,
                reservation_key=reservation_key,
                scan_timestamp=scan_timestamp,
                sign_body=sign_body,
            )
        reservation = self.reserve(
            file_name=file_name,
            file_bytes=file_bytes,
            media_type=media_type,
            reservation_key=reservation_key,
            scan_timestamp=scan_timestamp,
            canonical_body=sign_body,
        )
        if persistent_path is not None:
            self._reservation_expiry(reservation)
            self._write_state(
                persistent_path,
                self._state_record(
                    reservation,
                    file_name=file_name,
                    file_bytes=file_bytes,
                    media_type=media_type,
                    phase="reserved",
                    source_identity=source_identity,
                ),
            )
        try:
            self.put(reservation, file_bytes, media_type=media_type)
        except Exception as error:
            if not self._is_ambiguous_put(error):
                raise
            result = self._reconcile_ambiguous_put(
                reservation,
                file_bytes,
                media_type=media_type,
                initial_error=error,
            )
            self._mark_completed_state(
                persistent_path,
                reservation,
                file_name=file_name,
                file_bytes=file_bytes,
                media_type=media_type,
                result=result,
                source_identity=source_identity,
            )
            return result
        if persistent_path is not None:
            self._write_state(
                persistent_path,
                self._state_record(
                    reservation,
                    file_name=file_name,
                    file_bytes=file_bytes,
                    media_type=media_type,
                    phase="uploaded",
                    source_identity=source_identity,
                ),
            )
        result = self._complete_with_retry(str(reservation["upload_id"]))
        self._mark_completed_state(
            persistent_path,
            reservation,
            file_name=file_name,
            file_bytes=file_bytes,
            media_type=media_type,
            result=result,
            source_identity=source_identity,
        )
        return result

    def stats(self) -> dict[str, Any]:
        self.ensure_registered()
        response = self._request("GET", f"/v1/contributors/{self.credentials.contributor_id}/stats")
        result = self._json(response)
        if response.status_code != 200:
            raise ContributorAuthError(response.status_code, result)
        return result

    def revoke_status(self, request_id: str) -> dict[str, Any]:
        self.ensure_registered()
        response = self._request("GET", f"/v1/revoke/cascade-status/{quote(request_id, safe='')}")
        result = self._json(response)
        if response.status_code != 200:
            raise ContributorAuthError(response.status_code, result)
        return result
