from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from tether.contributor_auth import (
    DEFAULT_CONTRIBUTION_WORKER,
    ContributorAuthClient,
    ContributorAuthError,
    ContributorCredentials,
    ContributorTransportError,
    _UploadStateLock,
    canonical_json_bytes,
    load_or_create_credentials,
    signed_headers,
)
from tether.curate.uploader import DEFAULT_WORKER_URL
from tether.pro.upload import DEFAULT_DATA_ENDPOINT


VECTORS = (
    Path(__file__).parents[1]
    / "infra"
    / "contribution-worker"
    / "test"
    / "fixtures"
    / "contributor-auth-v1.json"
)


def _decode_base64url(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_fixed_contributor_auth_vectors_verify_in_python() -> None:
    payload = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["vectors"]) >= 2

    for vector in payload["vectors"]:
        seed = bytes.fromhex(vector["private_key_seed_hex"])
        private_key = Ed25519PrivateKey.from_private_bytes(seed)
        public_key = private_key.public_key()
        public_bytes = public_key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        assert public_bytes == _decode_base64url(vector["public_key"])
        digest = hashlib.sha256(public_bytes).hexdigest()
        assert vector["contributor_id"] == f"ctr_{digest[:32]}"
        assert vector["key_id"] == f"key_{digest[:32]}"

        body = vector["body_utf8"].encode("utf-8")
        assert hashlib.sha256(body).hexdigest() == vector["body_sha256"]
        canonical = _canonical_json(vector["envelope"])
        assert canonical.decode("utf-8") == vector["canonical_envelope_utf8"]
        public_key.verify(_decode_base64url(vector["signature"]), canonical)


def test_production_signer_matches_cross_language_vectors() -> None:
    payload = json.loads(VECTORS.read_text(encoding="utf-8"))
    for vector in payload["vectors"]:
        credentials = ContributorCredentials(
            bytes.fromhex(vector["private_key_seed_hex"]),
            vector["public_key"],
            vector["contributor_id"],
            vector["key_id"],
        )
        envelope = vector["envelope"]
        query = "&".join(f"{key}={value}" for key, value in envelope["query"])
        url = f"https://example.test{envelope['path']}"
        if query:
            url += f"?{query}"
        headers = signed_headers(
            credentials,
            envelope["method"],
            url,
            vector["body_utf8"].encode("utf-8"),
            timestamp=envelope["timestamp"],
            nonce=_decode_base64url(envelope["nonce"]),
        )
        assert headers["X-Tether-Signature"] == vector["signature"]
        assert headers["X-Tether-Content-SHA256"] == vector["body_sha256"]


def test_credentials_are_owner_only_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "auth" / "credentials.json"
    first = load_or_create_credentials(path)
    second = load_or_create_credentials(path)
    assert first == second
    assert "private_key_seed" in json.loads(path.read_text(encoding="utf-8"))
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


class _FakeResponse:
    def __init__(self, status_code: int, body: dict[str, object]):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> dict[str, object]:
        return self._body


def test_full_client_flow_is_signed_and_never_sends_client_tier() -> None:
    vector = json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"][0]
    credentials = ContributorCredentials(
        bytes.fromhex(vector["private_key_seed_hex"]),
        vector["public_key"],
        vector["contributor_id"],
        vector["key_id"],
    )
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    def transport(method, url, body, headers, _timeout):
        calls.append((method, url, body, headers))
        if url.endswith("/register"):
            return _FakeResponse(
                201,
                {
                    "contributor_id": credentials.contributor_id,
                    "key_id": credentials.key_id,
                    "tier": "free",
                },
            )
        if url.endswith("/sign"):
            return _FakeResponse(
                200,
                {
                    "upload_id": f"upl_{'1' * 32}",
                    "put_url": f"https://worker.test/v1/uploads/put/upl_{'1' * 32}",
                    "upload_capability": base64.urlsafe_b64encode(b"c" * 32).rstrip(b"=").decode(),
                },
            )
        if method == "PUT":
            return _FakeResponse(200, {"bytes_received": len(body)})
        return _FakeResponse(200, {"status": "completed"})

    client = ContributorAuthClient(
        "https://worker.test", credentials=credentials, transport=transport
    )
    client.upload(file_name="episode.jsonl", file_bytes=b"{}\n", media_type="application/jsonl")

    assert [call[0] for call in calls] == ["POST", "POST", "PUT", "POST"]
    for method, _url, _body, headers in (calls[0], calls[1], calls[3]):
        assert method in {"POST"}
        assert headers["X-Tether-Contributor-Id"] == credentials.contributor_id
        assert "X-Tether-Signature" in headers
    reservation = json.loads(calls[1][2])
    assert set(reservation) == {"byte_size", "file_name", "manifest", "reservation_key"}
    assert "tier" not in reservation and "contributor_id" not in reservation
    expected_capability = base64.urlsafe_b64encode(b"c" * 32).rstrip(b"=").decode()
    assert calls[2][3]["Authorization"] == f"Upload {expected_capability}"
    assert calls[2][3]["Content-Length"] == "3"
    assert calls[3][2] == canonical_json_bytes({"upload_id": f"upl_{'1' * 32}"})


def test_default_worker_url_matches_deployed_wrangler_service() -> None:
    expected = "https://reflex-contributions.fastcrest.workers.dev"
    assert DEFAULT_CONTRIBUTION_WORKER == expected
    assert DEFAULT_WORKER_URL == expected
    assert DEFAULT_DATA_ENDPOINT == expected
    root = Path(__file__).parents[1]
    assert 'name = "reflex-contributions"' in (
        root / "infra" / "contribution-worker" / "wrangler.toml"
    ).read_text(encoding="utf-8")
    assert expected in (root / "infra" / "contribution-worker" / "README.md").read_text(
        encoding="utf-8"
    )


def test_lost_put_response_reconciles_without_duplicate_reservation_or_object() -> None:
    vector = json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"][0]
    credentials = ContributorCredentials(
        bytes.fromhex(vector["private_key_seed_hex"]),
        vector["public_key"],
        vector["contributor_id"],
        vector["key_id"],
    )
    upload_id = f"upl_{'3' * 32}"
    capability = base64.urlsafe_b64encode(b"p" * 32).rstrip(b"=").decode()
    counts = {"register": 0, "reserve": 0, "put": 0, "complete": 0}
    objects: dict[str, bytes] = {}

    def transport(method, url, body, _headers, _timeout):
        if url.endswith("/register"):
            counts["register"] += 1
            return _FakeResponse(201, {"contributor_id": credentials.contributor_id})
        if url.endswith("/sign"):
            counts["reserve"] += 1
            return _FakeResponse(
                200,
                {
                    "upload_id": upload_id,
                    "put_url": f"https://worker.test/v1/uploads/put/{upload_id}",
                    "upload_capability": capability,
                },
            )
        if method == "PUT":
            counts["put"] += 1
            objects[upload_id] = body
            raise TimeoutError("response lost after authoritative PUT")
        counts["complete"] += 1
        assert objects.get(upload_id) == b"payload"
        return _FakeResponse(200, {"status": "completed", "upload_id": upload_id})

    sleeps: list[float] = []
    client = ContributorAuthClient(
        "https://worker.test",
        credentials=credentials,
        transport=transport,
        sleep=sleeps.append,
    )
    result = client.upload(
        file_name="lost-put.jsonl",
        file_bytes=b"payload",
        media_type="application/jsonl",
    )
    assert result["status"] == "completed"
    assert counts == {"register": 1, "reserve": 1, "put": 1, "complete": 1}
    assert objects == {upload_id: b"payload"}
    assert sleeps == []


def test_lost_completion_response_retries_idempotently_without_new_reservation() -> None:
    vector = json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"][0]
    credentials = ContributorCredentials(
        bytes.fromhex(vector["private_key_seed_hex"]),
        vector["public_key"],
        vector["contributor_id"],
        vector["key_id"],
    )
    upload_id = f"upl_{'4' * 32}"
    capability = base64.urlsafe_b64encode(b"q" * 32).rstrip(b"=").decode()
    counts = {"register": 0, "reserve": 0, "put": 0, "complete": 0}
    objects: dict[str, bytes] = {}
    completed: set[str] = set()

    def transport(method, url, body, _headers, _timeout):
        if url.endswith("/register"):
            counts["register"] += 1
            return _FakeResponse(201, {"contributor_id": credentials.contributor_id})
        if url.endswith("/sign"):
            counts["reserve"] += 1
            return _FakeResponse(
                200,
                {
                    "upload_id": upload_id,
                    "put_url": f"https://worker.test/v1/uploads/put/{upload_id}",
                    "upload_capability": capability,
                },
            )
        if method == "PUT":
            counts["put"] += 1
            objects[upload_id] = body
            return _FakeResponse(200, {"bytes_received": len(body)})
        counts["complete"] += 1
        completed.add(upload_id)
        if counts["complete"] == 1:
            raise TimeoutError("response lost after authoritative completion")
        return _FakeResponse(
            200,
            {
                "status": "completed",
                "upload_id": upload_id,
                "idempotent": True,
            },
        )

    sleeps: list[float] = []
    client = ContributorAuthClient(
        "https://worker.test",
        credentials=credentials,
        transport=transport,
        sleep=sleeps.append,
    )
    result = client.upload(
        file_name="lost-complete.jsonl",
        file_bytes=b"payload",
        media_type="application/jsonl",
    )
    assert result["idempotent"] is True
    assert counts == {"register": 1, "reserve": 1, "put": 1, "complete": 2}
    assert objects == {upload_id: b"payload"}
    assert completed == {upload_id}
    assert sleeps == [2.0]


def test_lost_completion_persists_across_queue_retry_and_client_restart(
    tmp_path: Path,
) -> None:
    vector = json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"][0]
    credentials = ContributorCredentials(
        bytes.fromhex(vector["private_key_seed_hex"]),
        vector["public_key"],
        vector["contributor_id"],
        vector["key_id"],
    )
    upload_id = f"upl_{'5' * 32}"
    capability = base64.urlsafe_b64encode(b"r" * 32).rstrip(b"=").decode()
    counts = {"register": 0, "reserve": 0, "put": 0, "complete": 0}
    objects: dict[str, bytes] = {}
    state_path = tmp_path / "queue" / ".episode.jsonl.upload-v1.json"

    def transport(method, url, body, _headers, _timeout):
        if url.endswith("/register"):
            counts["register"] += 1
            return _FakeResponse(201, {"contributor_id": credentials.contributor_id})
        if url.endswith("/sign"):
            counts["reserve"] += 1
            return _FakeResponse(
                200,
                {
                    "expires_at": "2099-01-01T00:00:00.000Z",
                    "upload_id": upload_id,
                    "put_url": f"https://worker.test/v1/uploads/put/{upload_id}",
                    "upload_capability": capability,
                },
            )
        if method == "PUT":
            counts["put"] += 1
            objects[upload_id] = body
            return _FakeResponse(200, {"bytes_received": len(body)})
        counts["complete"] += 1
        if counts["complete"] == 1:
            raise TimeoutError("completion committed but its response was lost")
        return _FakeResponse(
            200,
            {
                "status": "completed",
                "upload_id": upload_id,
                "idempotent": True,
            },
        )

    first = ContributorAuthClient(
        "https://worker.test",
        credentials=credentials,
        transport=transport,
        upload_max_retries=0,
    )
    with pytest.raises(Exception, match="contributor transport failed"):
        first.upload(
            file_name="episode.jsonl",
            file_bytes=b"persisted",
            media_type="application/jsonl",
            state_path=state_path,
        )
    assert state_path.exists()
    if os.name == "posix":
        assert state_path.stat().st_mode & 0o777 == 0o600
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["phase"] == "uploaded"
    assert "private_key_seed" not in state_path.read_text(encoding="utf-8")

    # Simulate the next queue pass in a fresh process/client. It loads the
    # upload_id and completion phase before considering POST /sign.
    second = ContributorAuthClient(
        "https://worker.test",
        credentials=credentials,
        transport=transport,
        upload_max_retries=0,
    )
    result = second.upload(
        file_name="episode.jsonl",
        file_bytes=b"persisted",
        media_type="application/jsonl",
        state_path=state_path,
    )
    assert result["idempotent"] is True
    assert counts == {"register": 1, "reserve": 1, "put": 1, "complete": 2}
    assert objects == {upload_id: b"persisted"}
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["phase"] == "completed"
    assert persisted["reservation"]["upload_capability"] == ""


def test_lost_reservation_response_reuses_persisted_key_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"][0]
    credentials = ContributorCredentials(
        bytes.fromhex(vector["private_key_seed_hex"]),
        vector["public_key"],
        vector["contributor_id"],
        vector["key_id"],
    )
    upload_id = f"upl_{'8' * 32}"
    capability = base64.urlsafe_b64encode(b"u" * 32).rstrip(b"=").decode()
    state_path = tmp_path / ".lost-sign.upload-v1.json"
    seen_keys: list[str] = []
    seen_bodies: list[bytes] = []
    clock = [1_800_000_000]
    monkeypatch.setattr("tether.contributor_auth.time.time", lambda: clock[0])
    counts = {"sign": 0, "put": 0, "complete": 0}

    def transport(method, url, body, _headers, _timeout):
        if url.endswith("/register"):
            return _FakeResponse(201, {"contributor_id": credentials.contributor_id})
        if url.endswith("/sign"):
            counts["sign"] += 1
            seen_keys.append(json.loads(body)["reservation_key"])
            seen_bodies.append(body)
            if counts["sign"] == 1:
                raise TimeoutError("reservation committed; response lost")
            return _FakeResponse(
                200,
                {
                    "expires_at": "2099-01-01T00:00:00.000Z",
                    "upload_id": upload_id,
                    "put_url": f"https://worker.test/v1/uploads/put/{upload_id}",
                    "upload_capability": capability,
                },
            )
        if method == "PUT":
            counts["put"] += 1
            return _FakeResponse(200, {"bytes_received": len(body)})
        counts["complete"] += 1
        if counts["complete"] == 1:
            return _FakeResponse(409, {"error": "upload_not_ready", "status": "pending"})
        return _FakeResponse(200, {"status": "completed", "upload_id": upload_id})

    kwargs = dict(
        file_name="lost-sign.jsonl",
        file_bytes=b"one quota row",
        media_type="application/jsonl",
        state_path=state_path,
    )
    with pytest.raises(Exception, match="contributor transport failed"):
        ContributorAuthClient(
            "https://worker.test",
            credentials=credentials,
            transport=transport,
            upload_max_retries=0,
        ).upload(**kwargs)
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["phase"] == "reserving"
    assert saved["sign_body"].encode("utf-8") == seen_bodies[0]
    clock[0] += 360
    result = ContributorAuthClient(
        "https://worker.test",
        credentials=credentials,
        transport=transport,
        upload_max_retries=0,
    ).upload(**kwargs)
    assert result["status"] == "completed"
    assert len(set(seen_keys)) == 1
    assert seen_bodies[0] == seen_bodies[1]
    assert counts == {"sign": 2, "put": 1, "complete": 2}


def test_stale_uncommitted_reserving_request_rotates_once_without_sticking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"][0]
    credentials = ContributorCredentials(
        bytes.fromhex(vector["private_key_seed_hex"]),
        vector["public_key"],
        vector["contributor_id"],
        vector["key_id"],
    )
    clock = [1_800_000_000]
    monkeypatch.setattr("tether.contributor_auth.time.time", lambda: clock[0])
    upload_id = f"upl_{'9' * 32}"
    capability = base64.urlsafe_b64encode(b"v" * 32).rstrip(b"=").decode()
    state_path = tmp_path / ".stale-uncommitted.upload-v1.json"
    bodies: list[dict] = []
    counts = {"sign": 0, "put": 0, "complete": 0}

    def transport(method, url, body, _headers, _timeout):
        if url.endswith("/register"):
            return _FakeResponse(201, {"contributor_id": credentials.contributor_id})
        if url.endswith("/sign"):
            counts["sign"] += 1
            bodies.append(json.loads(body))
            if counts["sign"] == 1:
                raise TimeoutError("request was never committed")
            if counts["sign"] == 2:
                return _FakeResponse(400, {"error": "stale_manifest"})
            return _FakeResponse(
                200,
                {
                    "expires_at": "2099-01-01T00:00:00.000Z",
                    "upload_id": upload_id,
                    "put_url": f"https://worker.test/v1/uploads/put/{upload_id}",
                    "upload_capability": capability,
                },
            )
        if method == "PUT":
            counts["put"] += 1
            return _FakeResponse(200, {"bytes_received": len(body)})
        counts["complete"] += 1
        if counts["complete"] == 1:
            return _FakeResponse(409, {"error": "upload_not_ready", "status": "pending"})
        return _FakeResponse(200, {"status": "completed", "upload_id": upload_id})

    kwargs = dict(
        file_name="stale.jsonl",
        file_bytes=b"uncommitted",
        media_type="application/jsonl",
        state_path=state_path,
    )
    with pytest.raises(ContributorTransportError):
        ContributorAuthClient(
            "https://worker.test",
            credentials=credentials,
            transport=transport,
            upload_max_retries=0,
        ).upload(**kwargs)
    clock[0] += 301
    result = ContributorAuthClient(
        "https://worker.test",
        credentials=credentials,
        transport=transport,
        upload_max_retries=0,
    ).upload(**kwargs)
    assert result["status"] == "completed"
    assert bodies[0] == bodies[1]
    assert bodies[2]["reservation_key"] != bodies[1]["reservation_key"]
    assert bodies[2]["manifest"]["scan_timestamp"] == clock[0]
    assert counts == {"sign": 3, "put": 1, "complete": 2}
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["phase"] == "completed"


def test_concurrent_clients_elect_one_persistent_upload_owner(tmp_path: Path) -> None:
    vector = json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"][0]
    credentials = ContributorCredentials(
        bytes.fromhex(vector["private_key_seed_hex"]),
        vector["public_key"],
        vector["contributor_id"],
        vector["key_id"],
    )
    upload_id = f"upl_{'7' * 32}"
    capability = base64.urlsafe_b64encode(b"t" * 32).rstrip(b"=").decode()
    state_path = tmp_path / ".same-file.upload-v1.json"
    counts = {"register": 0, "reserve": 0, "put": 0, "complete": 0}
    objects: dict[str, bytes] = {}
    transport_guard = threading.Lock()
    first_reserved = threading.Event()
    release_first = threading.Event()

    def transport(method, url, body, _headers, _timeout):
        with transport_guard:
            if url.endswith("/register"):
                counts["register"] += 1
                return _FakeResponse(201, {"contributor_id": credentials.contributor_id})
            if url.endswith("/sign"):
                counts["reserve"] += 1
                first_reserved.set()
            elif method == "PUT":
                counts["put"] += 1
                objects[upload_id] = body
                return _FakeResponse(200, {"bytes_received": len(body)})
            else:
                counts["complete"] += 1
                return _FakeResponse(
                    200,
                    {
                        "status": "completed",
                        "upload_id": upload_id,
                    },
                )
        assert release_first.wait(2)
        return _FakeResponse(
            200,
            {
                "expires_at": "2099-01-01T00:00:00.000Z",
                "upload_id": upload_id,
                "put_url": f"https://worker.test/v1/uploads/put/{upload_id}",
                "upload_capability": capability,
            },
        )

    clients = [
        ContributorAuthClient(
            "https://worker.test",
            credentials=credentials,
            transport=transport,
            upload_max_retries=0,
            state_lock_timeout_s=2,
        )
        for _ in range(2)
    ]
    kwargs = {
        "file_name": "same-file.jsonl",
        "file_bytes": b"one object",
        "media_type": "application/jsonl",
        "state_path": state_path,
        "source_identity": "device:inode:mtime:size",
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(clients[0].upload, **kwargs)
        assert first_reserved.wait(2)
        second = pool.submit(clients[1].upload, **kwargs)
        release_first.set()
        results = [first.result(timeout=2), second.result(timeout=2)]

    assert all(result["status"] == "completed" for result in results)
    assert counts == {"register": 1, "reserve": 1, "put": 1, "complete": 1}
    assert objects == {upload_id: b"one object"}
    marker = json.loads(state_path.read_text(encoding="utf-8"))
    assert marker["phase"] == "completed"
    assert marker["reservation"]["upload_capability"] == ""
    lock_path = state_path.with_name(f"{state_path.name}.lock")
    assert lock_path.exists()
    if os.name == "posix":
        assert lock_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-death lock semantics")
def test_upload_state_lock_survives_inode_and_releases_on_process_death(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / ".crash.upload-v1.json"
    code = """
import sys, time
from pathlib import Path
from tether.contributor_auth import _UploadStateLock
with _UploadStateLock(Path(sys.argv[1]), 2):
    print("locked", flush=True)
    time.sleep(30)
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    owner = subprocess.Popen(
        [sys.executable, "-c", code, str(state_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert owner.stdout is not None
        assert owner.stdout.readline().strip() == "locked"
        lock_path = state_path.with_name(f"{state_path.name}.lock")
        inode = lock_path.stat().st_ino
        with pytest.raises(TimeoutError, match="timed out waiting"):
            with _UploadStateLock(state_path, 0.05):
                pass
        assert lock_path.stat().st_ino == inode
        owner.terminate()
        owner.wait(timeout=2)
        with _UploadStateLock(state_path, 1):
            assert lock_path.stat().st_ino == inode
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=2)


def test_expired_persisted_capability_is_redacted_but_upload_id_is_retained(
    tmp_path: Path,
) -> None:
    vector = json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"][0]
    credentials = ContributorCredentials(
        bytes.fromhex(vector["private_key_seed_hex"]),
        vector["public_key"],
        vector["contributor_id"],
        vector["key_id"],
    )
    client = ContributorAuthClient("https://worker.test", credentials=credentials)
    upload_id = f"upl_{'6' * 32}"
    reservation = {
        "expires_at": "2000-01-01T00:00:00.000Z",
        "upload_id": upload_id,
        "put_url": f"https://worker.test/v1/uploads/put/{upload_id}",
        "upload_capability": base64.urlsafe_b64encode(b"s" * 32).rstrip(b"=").decode(),
    }
    path = tmp_path / ".expired-upload.json"
    client._write_state(
        path,
        client._state_record(
            reservation,
            file_name="expired.jsonl",
            file_bytes=b"expired",
            media_type="application/jsonl",
            phase="uploaded",
        ),
    )
    loaded = client._load_state(
        path,
        file_name="expired.jsonl",
        file_bytes=b"expired",
        media_type="application/jsonl",
    )
    assert loaded is not None
    assert loaded["reservation"]["upload_id"] == upload_id
    assert loaded["reservation"]["upload_capability"] == ""
    assert reservation["upload_capability"] not in path.read_text(encoding="utf-8")


def test_reservation_rejects_cross_origin_put_url() -> None:
    vector = json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"][0]
    credentials = ContributorCredentials(
        bytes.fromhex(vector["private_key_seed_hex"]),
        vector["public_key"],
        vector["contributor_id"],
        vector["key_id"],
    )

    def transport(_method, url, _body, _headers, _timeout):
        if url.endswith("/register"):
            return _FakeResponse(201, {"contributor_id": credentials.contributor_id})
        upload_id = f"upl_{'2' * 32}"
        return _FakeResponse(
            200,
            {
                "upload_id": upload_id,
                "put_url": f"https://attacker.test/v1/uploads/put/{upload_id}",
                "upload_capability": base64.urlsafe_b64encode(b"c" * 32).rstrip(b"=").decode(),
            },
        )

    client = ContributorAuthClient(
        "https://worker.test",
        credentials=credentials,
        transport=transport,
    )
    with pytest.raises(ContributorAuthError, match="invalid_reservation_put_url"):
        client.reserve(file_name="x.jsonl", file_bytes=b"x")


def test_obsolete_live_stress_harness_is_explicitly_retired() -> None:
    script = Path(__file__).parents[1] / "scripts" / "stress_test_contribution_worker.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "RETIRED" in result.stderr
    assert "no requests were sent" in result.stderr
