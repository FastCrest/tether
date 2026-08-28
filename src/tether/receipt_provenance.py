"""Trusted production and consumption of hardware parity receipts.

The receipt payload and its manifest are separate GitHub artifacts.  That
allows the manifest to bind the payload's server-assigned artifact id and
digest without making the manifest self-referential.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = 1
WORKFLOW_PATH = ".github/workflows/parity-receipts.yml"
REQUIRED_PARITY_CELLS = frozenset({"pi05_teacher_n1", "pi05_teacher_n10"})
EXPORT_FILES = (
    "tether_config.json",
    "vlm_prefix.onnx",
    "expert_denoise.onnx",
)
PAYLOAD_FILES = (
    "per_step_parity_last_run.json",
    "per_step_overhead_last_run.json",
    "per_step_e2e_latency_last_run.json",
)
ALLOWED_EVENTS = frozenset({"schedule", "workflow_dispatch", "push"})
RECEIPT_RUNS_DIR = "receipt_runs"
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_RECEIPT_NAMESPACE_RE = re.compile(
    r"^gh-[0-9a-f]{32}-sha-(?:[0-9a-f]{40}|[0-9a-f]{64})-"
    r"run-[1-9][0-9]*-attempt-[1-9][0-9]*$"
)


class ReceiptVerificationError(ValueError):
    """Raised when a receipt or its GitHub provenance is not trusted."""


def build_receipt_namespace(
    repository: str, source_sha: str, workflow_run_id: int, workflow_run_attempt: int
) -> str:
    """Build the immutable, path-safe namespace for one GitHub run attempt."""
    if not isinstance(repository, str) or not _REPOSITORY_RE.fullmatch(repository):
        raise ReceiptVerificationError("repository must be an owner/name slug")
    normalized_sha = source_sha.lower() if isinstance(source_sha, str) else ""
    if len(normalized_sha) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in normalized_sha
    ):
        raise ReceiptVerificationError("source_sha must be an immutable Git SHA")
    if (
        isinstance(workflow_run_id, bool)
        or not isinstance(workflow_run_id, int)
        or workflow_run_id <= 0
    ):
        raise ReceiptVerificationError("workflow_run_id must be a positive integer")
    if (
        isinstance(workflow_run_attempt, bool)
        or not isinstance(workflow_run_attempt, int)
        or workflow_run_attempt <= 0
    ):
        raise ReceiptVerificationError("workflow_run_attempt must be a positive integer")
    repository_digest = _sha256_bytes(repository.lower().encode())[:32]
    return (
        f"gh-{repository_digest}-sha-{normalized_sha}-"
        f"run-{workflow_run_id}-attempt-{workflow_run_attempt}"
    )


def receipt_namespace_binding(
    *,
    receipt_namespace: str,
    repository: str,
    source_sha: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
) -> dict[str, object]:
    """Validate and serialize the complete namespace provenance binding."""
    expected = build_receipt_namespace(
        repository, source_sha, workflow_run_id, workflow_run_attempt
    )
    if receipt_namespace != expected:
        raise ReceiptVerificationError(
            f"receipt namespace does not match immutable run binding: {receipt_namespace!r}"
        )
    return {
        "value": expected,
        "repository": repository,
        "source_sha": source_sha.lower(),
        "workflow_run_id": workflow_run_id,
        "workflow_run_attempt": workflow_run_attempt,
    }


def receipt_mode_binding(
    *,
    receipt_namespace: str = "",
    repository: str = "",
    source_sha: str = "",
    workflow_run_id: int = 0,
    workflow_run_attempt: int = 0,
) -> dict[str, object] | None:
    """Return a binding for receipt mode or explicit legacy mode outside it."""
    supplied = (
        bool(receipt_namespace),
        bool(repository),
        bool(source_sha),
        workflow_run_id != 0,
        workflow_run_attempt != 0,
    )
    if not any(supplied):
        return None
    if not all(supplied):
        raise ReceiptVerificationError(
            "receipt mode requires namespace, repository, SHA, run id, and run attempt"
        )
    return receipt_namespace_binding(
        receipt_namespace=receipt_namespace,
        repository=repository,
        source_sha=source_sha,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
    )


def receipt_run_dir(root: Path, receipt_namespace: str) -> Path:
    """Resolve a namespace beneath a run root without accepting path syntax."""
    if not isinstance(receipt_namespace, str) or not _RECEIPT_NAMESPACE_RE.fullmatch(
        receipt_namespace
    ):
        raise ReceiptVerificationError("receipt namespace is unsafe or non-canonical")
    return root / RECEIPT_RUNS_DIR / receipt_namespace


def receipt_artifact_names(receipt_namespace: str) -> tuple[str, str]:
    """Return the exact payload and manifest artifact names for one attempt."""
    if not isinstance(receipt_namespace, str) or not _RECEIPT_NAMESPACE_RE.fullmatch(
        receipt_namespace
    ):
        raise ReceiptVerificationError("receipt namespace is unsafe or non-canonical")
    return (
        f"parity-receipts-payload-{receipt_namespace}",
        f"parity-receipt-manifest-{receipt_namespace}",
    )


def _validate_namespace_binding(value: object, expected: Mapping[str, object], field: str) -> None:
    binding = _require_mapping(value, field)
    if set(binding) != set(expected) or dict(binding) != dict(expected):
        raise ReceiptVerificationError(f"{field} does not match immutable run namespace")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ReceiptVerificationError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReceiptVerificationError(f"{field} is not valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ReceiptVerificationError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_export_set(export_dir: Path) -> dict[str, dict[str, object]]:
    """Hash the complete runtime export set used by receipt producers."""
    result: dict[str, dict[str, object]] = {}
    for name in EXPORT_FILES:
        path = export_dir / name
        if not path.is_file() or path.is_symlink():
            raise ReceiptVerificationError(f"required export file is missing: {path}")
        result[name] = {
            "path": name,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    return result


def require_cuda_provider_lists(
    prefix_providers: object, expert_providers: object
) -> tuple[list[str], list[str]]:
    """Require CUDA to be the active provider for both decomposed sessions."""
    if not isinstance(prefix_providers, list) or not all(
        isinstance(provider, str) for provider in prefix_providers
    ):
        raise ReceiptVerificationError("prefix provider list is invalid")
    if not isinstance(expert_providers, list) or not all(
        isinstance(provider, str) for provider in expert_providers
    ):
        raise ReceiptVerificationError("expert provider list is invalid")
    if not prefix_providers or prefix_providers[0] != "CUDAExecutionProvider":
        raise ReceiptVerificationError(
            f"vlm_prefix did not use CUDAExecutionProvider: {prefix_providers!r}"
        )
    if not expert_providers or expert_providers[0] != "CUDAExecutionProvider":
        raise ReceiptVerificationError(
            f"expert_denoise did not use CUDAExecutionProvider: {expert_providers!r}"
        )
    return list(prefix_providers), list(expert_providers)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _normalize_digest(value: object, field: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        if required:
            raise ReceiptVerificationError(f"{field} must be a SHA-256 digest")
        return ""
    digest = value.removeprefix("sha256:").lower()
    if not digest and not required:
        return ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ReceiptVerificationError(f"{field} must be a SHA-256 digest")
    return digest


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptVerificationError(f"{field} must be an object")
    return value


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise ReceiptVerificationError("receipt file path must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ReceiptVerificationError(f"unsafe receipt file path: {value!r}")
    return path


def _read_payload(payload_dir: Path) -> dict[str, dict[str, Any]]:
    measurements: dict[str, dict[str, Any]] = {}
    for name in PAYLOAD_FILES:
        path = payload_dir / name
        if not path.is_file() or path.is_symlink():
            raise ReceiptVerificationError(f"required payload file is missing: {name}")
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ReceiptVerificationError(f"invalid JSON payload: {name}") from exc
        if not isinstance(value, dict):
            raise ReceiptVerificationError(f"payload must be a JSON object: {name}")
        measurements[name] = value
    return measurements


def _validate_payload_namespace(
    measurements: Mapping[str, Mapping[str, Any]], expected: Mapping[str, object]
) -> None:
    for name in PAYLOAD_FILES:
        measurement = _require_mapping(measurements.get(name), name)
        _validate_namespace_binding(
            measurement.get("receipt_namespace"), expected, f"{name}.receipt_namespace"
        )
    e2e = _require_mapping(
        measurements.get("per_step_e2e_latency_last_run.json"), "e2e measurement"
    )
    for variant in ("baked", "per_step"):
        nested = _require_mapping(e2e.get(variant), f"e2e {variant}")
        _validate_namespace_binding(
            nested.get("receipt_namespace"),
            expected,
            f"e2e {variant}.receipt_namespace",
        )


def _normalize_export_set(value: object, field: str) -> dict[str, str]:
    export_set = _require_mapping(value, field)
    if set(export_set) != set(EXPORT_FILES):
        raise ReceiptVerificationError(f"{field} must contain exactly {', '.join(EXPORT_FILES)}")
    normalized: dict[str, str] = {}
    for name in EXPORT_FILES:
        file_info = _require_mapping(export_set.get(name), f"{field}.{name}")
        if file_info.get("path") != name:
            raise ReceiptVerificationError(f"{field}.{name}.path is not canonical")
        if not isinstance(file_info.get("size"), int) or int(file_info["size"]) <= 0:
            raise ReceiptVerificationError(f"{field}.{name}.size must be positive")
        normalized[name] = _normalize_digest(file_info.get("sha256"), f"{field}.{name}.sha256")
    return normalized


def _export_identity(measurements: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    parity = _require_mapping(
        measurements.get("per_step_parity_last_run.json"), "parity measurement"
    )
    cells = _require_mapping(parity.get("cells"), "parity cells")
    if set(cells) != REQUIRED_PARITY_CELLS:
        raise ReceiptVerificationError(
            "parity receipt cells must be exactly "
            f"{sorted(REQUIRED_PARITY_CELLS)!r}; got {sorted(str(key) for key in cells)!r}"
        )
    artifacts: dict[str, Any] = {}
    for label, raw_cell in sorted(cells.items()):
        cell = _require_mapping(raw_cell, f"parity cell {label}")
        for provider_field in ("used_provider_baked", "used_provider_per_step"):
            if cell.get(provider_field) != "CUDAExecutionProvider":
                raise ReceiptVerificationError(
                    f"parity cell {label}.{provider_field} did not use CUDA"
                )
        cell_exports = _require_mapping(cell.get("exports"), f"parity cell {label}.exports")
        normalized: dict[str, dict[str, str]] = {}
        for variant in ("baked", "per_step"):
            normalized[variant] = _normalize_export_set(
                cell_exports.get(variant), f"parity cell {label}.exports.{variant}"
            )
        artifacts[str(label)] = normalized

    expected_runtime_exports = artifacts.get("pi05_teacher_n10")
    if not isinstance(expected_runtime_exports, Mapping):
        raise ReceiptVerificationError("parity receipt has no pi05_teacher_n10 exports")
    overhead = _require_mapping(
        measurements.get("per_step_overhead_last_run.json"), "overhead measurement"
    )
    overhead_exports = _require_mapping(overhead.get("exports"), "overhead exports")
    overhead_providers = _require_mapping(overhead.get("providers"), "overhead providers")
    e2e_identity: dict[str, Any] = {}
    for variant in ("baked", "per_step"):
        if overhead_providers.get(variant) != "CUDAExecutionProvider":
            raise ReceiptVerificationError(f"overhead {variant} did not use CUDA")
    e2e = _require_mapping(
        measurements.get("per_step_e2e_latency_last_run.json"), "e2e measurement"
    )
    for variant in ("baked", "per_step"):
        overhead_export = _normalize_export_set(
            overhead_exports.get(variant), f"overhead exports.{variant}"
        )
        e2e_variant = _require_mapping(e2e.get(variant), f"e2e {variant}")
        e2e_export = _normalize_export_set(e2e_variant.get("export"), f"e2e {variant}.export")
        expected_export = expected_runtime_exports[variant]
        if overhead_export != expected_export:
            raise ReceiptVerificationError(
                f"overhead {variant} export does not match parity export"
            )
        if e2e_export != expected_export:
            raise ReceiptVerificationError(f"e2e {variant} export does not match parity export")
        e2e_providers = _require_mapping(e2e_variant.get("providers"), f"e2e {variant}.providers")
        require_cuda_provider_lists(
            e2e_providers.get("vlm_prefix"),
            e2e_providers.get("expert_denoise"),
        )
        e2e_identity[variant] = {
            "export": e2e_export,
            "receipt_namespace": dict(
                _require_mapping(
                    e2e_variant.get("receipt_namespace"),
                    f"e2e {variant}.receipt_namespace",
                )
            ),
        }
    return "pi05-per-step-expert", _canonical_digest(
        {"parity_exports": artifacts, "e2e": e2e_identity}
    )


def _measurement_model_ids(
    measurements: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    parity = _require_mapping(
        measurements.get("per_step_parity_last_run.json"), "parity measurement"
    )
    cells = _require_mapping(parity.get("cells"), "parity cells")
    model_ids: set[str] = set()
    for label, raw_cell in cells.items():
        cell = _require_mapping(raw_cell, f"parity cell {label}")
        model_id = cell.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ReceiptVerificationError(f"parity cell {label} has no model_id")
        model_ids.add(model_id)
    return model_ids


def _measurement_model_digests(
    measurements: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    parity = _require_mapping(
        measurements.get("per_step_parity_last_run.json"), "parity measurement"
    )
    cells = _require_mapping(parity.get("cells"), "parity cells")
    return {
        _normalize_digest(
            _require_mapping(raw_cell, f"parity cell {label}").get("model_digest"),
            f"parity cell {label}.model_digest",
        )
        for label, raw_cell in cells.items()
    }


def _measurement_model_revision(
    measurements: Mapping[str, Mapping[str, Any]],
) -> str:
    parity = _require_mapping(
        measurements.get("per_step_parity_last_run.json"), "parity measurement"
    )
    cells = _require_mapping(parity.get("cells"), "parity cells")
    raw_revisions = [
        _require_mapping(raw_cell, f"parity cell {label}").get("model_revision")
        for label, raw_cell in cells.items()
    ]
    if any(not isinstance(revision, str) for revision in raw_revisions):
        raise ReceiptVerificationError("payload model revision must be a string")
    revisions = set(raw_revisions)
    if len(revisions) != 1:
        raise ReceiptVerificationError("payload has inconsistent model revisions")
    revision = revisions.pop()
    if (
        not isinstance(revision, str)
        or len(revision) not in {40, 64}
        or any(char not in "0123456789abcdef" for char in revision.lower())
    ):
        raise ReceiptVerificationError("payload model revision is not an immutable hash")
    return revision.lower()


def build_receipt_manifest(
    *,
    payload_dir: Path,
    repository: str,
    source_sha: str,
    source_ref: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    receipt_namespace: str,
    event: str,
    payload_artifact_id: int,
    payload_artifact_name: str,
    payload_artifact_digest: str,
    model_id: str,
    model_digest: str,
    expires_in_hours: int = 168,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a manifest after the payload artifact has been uploaded."""
    if expires_in_hours <= 0:
        raise ReceiptVerificationError("expires_in_hours must be positive")
    if event not in ALLOWED_EVENTS:
        raise ReceiptVerificationError(f"untrusted producer event: {event}")
    if not _allowed_ref(source_ref):
        raise ReceiptVerificationError(f"untrusted producer ref: {source_ref}")

    namespace = receipt_namespace_binding(
        receipt_namespace=receipt_namespace,
        repository=repository,
        source_sha=source_sha,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
    )
    expected_payload_name, _ = receipt_artifact_names(receipt_namespace)
    if payload_artifact_name != expected_payload_name:
        raise ReceiptVerificationError(
            "payload artifact name does not match immutable run namespace"
        )
    measurements = _read_payload(payload_dir)
    _validate_payload_namespace(measurements, namespace)
    found_model_ids = _measurement_model_ids(measurements)
    if found_model_ids != {model_id}:
        raise ReceiptVerificationError(
            f"payload model ids {sorted(found_model_ids)!r} do not match {model_id!r}"
        )
    normalized_model_digest = _normalize_digest(model_digest, "model.digest")
    model_revision = _measurement_model_revision(measurements)
    derived_model_digest = _sha256_bytes(f"huggingface:{model_id}@{model_revision}".encode())
    if normalized_model_digest != derived_model_digest:
        raise ReceiptVerificationError(
            "approved model digest does not bind the payload model revision"
        )
    if _measurement_model_digests(measurements) != {normalized_model_digest}:
        raise ReceiptVerificationError(
            "payload model digest does not match the approved model digest"
        )
    export_id, export_digest = _export_identity(measurements)
    generated = (generated_at or _utc_now()).astimezone(timezone.utc)

    files = []
    for name in PAYLOAD_FILES:
        path = payload_dir / name
        files.append({"path": name, "sha256": sha256_file(path), "size": path.stat().st_size})

    thresholds = {name: value.get("thresholds", {}) for name, value in measurements.items()}
    try:
        modal_version = importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        modal_version = "unknown"

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"repository": repository, "sha": source_sha, "ref": source_ref},
        "receipt_namespace": namespace,
        "producer": {
            "workflow_path": WORKFLOW_PATH,
            "workflow_run_id": workflow_run_id,
            "workflow_run_attempt": workflow_run_attempt,
            "event": event,
        },
        "artifact": {
            "id": payload_artifact_id,
            "name": payload_artifact_name,
            "digest": f"sha256:{_normalize_digest(payload_artifact_digest, 'artifact.digest')}",
        },
        "generated_at": _isoformat(generated),
        "expires_at": _isoformat(generated + timedelta(hours=expires_in_hours)),
        "model": {
            "id": model_id,
            "revision": model_revision,
            "digest": f"sha256:{normalized_model_digest}",
        },
        "export": {"id": export_id, "digest": f"sha256:{export_digest}"},
        "hardware": {"provider": "Modal", "accelerator": "A100-80GB"},
        "backend": {
            "runtime": "onnxruntime-gpu",
            "execution_provider": "CUDAExecutionProvider",
        },
        "thresholds": thresholds,
        "measurements": measurements,
        "files": files,
        "producer_tools": {
            "python": sys.version.split()[0],
            "modal": modal_version,
            "upload_artifact": "actions/upload-artifact@v4",
        },
    }


def _branch_from_ref(ref: str) -> str:
    prefix = "refs/heads/"
    return ref[len(prefix) :] if ref.startswith(prefix) else ref


def _allowed_ref(ref: str) -> bool:
    branch = _branch_from_ref(ref)
    return branch == "main" or branch.startswith("release/")


def validate_trusted_run(
    run: Mapping[str, Any], *, expected_repository: str, expected_sha: str
) -> None:
    """Validate server-returned run metadata before trusting a run id."""
    if run.get("head_sha") != expected_sha:
        raise ReceiptVerificationError("workflow run head_sha does not match target SHA")
    if run.get("event") not in ALLOWED_EVENTS or run.get("event") == "pull_request":
        raise ReceiptVerificationError("workflow run event is not allowlisted")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ReceiptVerificationError("workflow run did not complete successfully")
    path = run.get("path")
    if not isinstance(path, str) or path.split("@", 1)[0] != WORKFLOW_PATH:
        raise ReceiptVerificationError("workflow run path is not the allowlisted producer")
    branch = run.get("head_branch")
    if not isinstance(branch, str) or not _allowed_ref(branch):
        raise ReceiptVerificationError("workflow run ref is not allowlisted")
    event = run.get("event")
    if event == "schedule" and branch != "main":
        raise ReceiptVerificationError("scheduled receipts must be produced from main")
    if event == "push" and not branch.startswith("release/"):
        raise ReceiptVerificationError("push receipts are allowed only on release branches")
    head_repo = _require_mapping(run.get("head_repository"), "workflow head_repository")
    if head_repo.get("full_name") != expected_repository:
        raise ReceiptVerificationError("workflow run came from an untrusted repository")
    if not isinstance(run.get("id"), int) or int(run["id"]) <= 0:
        raise ReceiptVerificationError("workflow run id is invalid")
    if not isinstance(run.get("run_attempt"), int) or int(run["run_attempt"]) <= 0:
        raise ReceiptVerificationError("workflow run attempt is invalid")


def select_trusted_run(
    runs: Sequence[Mapping[str, Any]], *, expected_repository: str, expected_sha: str
) -> Mapping[str, Any]:
    trusted = []
    for run in runs:
        try:
            validate_trusted_run(
                run, expected_repository=expected_repository, expected_sha=expected_sha
            )
        except ReceiptVerificationError:
            continue
        trusted.append(run)
    if not trusted:
        raise ReceiptVerificationError(f"no trusted {WORKFLOW_PATH} run exists for {expected_sha}")
    return max(
        trusted,
        key=lambda item: (
            str(item.get("created_at", "")),
            int(item.get("run_attempt", 0)),
        ),
    )


def verify_receipt_manifest(
    manifest: Mapping[str, Any],
    *,
    payload_dir: Path,
    trusted_run: Mapping[str, Any],
    trusted_artifact: Mapping[str, Any],
    expected_repository: str,
    expected_sha: str,
    expected_model_id: str,
    expected_model_digest: str,
    expected_export_id: str,
    expected_export_digest: str,
    now: datetime | None = None,
) -> None:
    """Verify provenance, freshness, identity, and every payload byte."""
    validate_trusted_run(
        trusted_run, expected_repository=expected_repository, expected_sha=expected_sha
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReceiptVerificationError("unsupported receipt schema_version")

    source = _require_mapping(manifest.get("source"), "source")
    if source.get("repository") != expected_repository or source.get("sha") != expected_sha:
        raise ReceiptVerificationError("receipt source does not match target")
    branch = str(trusted_run.get("head_branch"))
    if _branch_from_ref(str(source.get("ref", ""))) != branch:
        raise ReceiptVerificationError("receipt source ref does not match workflow run")

    producer = _require_mapping(manifest.get("producer"), "producer")
    if producer.get("workflow_path") != WORKFLOW_PATH:
        raise ReceiptVerificationError("receipt workflow is not allowlisted")
    if producer.get("workflow_run_id") != trusted_run.get("id"):
        raise ReceiptVerificationError("receipt contains a forged workflow run id")
    if producer.get("workflow_run_attempt") != trusted_run.get("run_attempt"):
        raise ReceiptVerificationError("receipt workflow run attempt does not match GitHub")
    if producer.get("event") != trusted_run.get("event"):
        raise ReceiptVerificationError("receipt event does not match workflow run")
    namespace = receipt_namespace_binding(
        receipt_namespace=str(
            _require_mapping(manifest.get("receipt_namespace"), "receipt_namespace").get(
                "value", ""
            )
        ),
        repository=expected_repository,
        source_sha=expected_sha,
        workflow_run_id=int(trusted_run["id"]),
        workflow_run_attempt=int(trusted_run["run_attempt"]),
    )
    _validate_namespace_binding(manifest.get("receipt_namespace"), namespace, "receipt_namespace")
    expected_payload_name, _ = receipt_artifact_names(str(namespace["value"]))

    generated_at = _parse_time(manifest.get("generated_at"), "generated_at")
    expires_at = _parse_time(manifest.get("expires_at"), "expires_at")
    current = (now or _utc_now()).astimezone(timezone.utc)
    if expires_at <= generated_at:
        raise ReceiptVerificationError("receipt expiry is not after generation")
    if generated_at > current + timedelta(minutes=5):
        raise ReceiptVerificationError("receipt generated_at is in the future")
    if current >= expires_at:
        raise ReceiptVerificationError("receipt has expired")

    model = _require_mapping(manifest.get("model"), "model")
    if model.get("id") != expected_model_id:
        raise ReceiptVerificationError("receipt model id does not match expected model")
    if _normalize_digest(model.get("digest"), "model.digest") != _normalize_digest(
        expected_model_digest, "expected_model_digest"
    ):
        raise ReceiptVerificationError("receipt model digest does not match expected model")
    model_revision = model.get("revision")
    derived_model_digest = _sha256_bytes(
        f"huggingface:{expected_model_id}@{model_revision}".encode()
    )
    if derived_model_digest != _normalize_digest(expected_model_digest, "expected_model_digest"):
        raise ReceiptVerificationError("receipt model revision does not match model digest")

    export = _require_mapping(manifest.get("export"), "export")
    if export.get("id") != expected_export_id:
        raise ReceiptVerificationError("receipt export id does not match expected export")
    if _normalize_digest(export.get("digest"), "export.digest") != _normalize_digest(
        expected_export_digest, "expected_export_digest"
    ):
        raise ReceiptVerificationError("receipt export digest does not match expected export")

    artifact = _require_mapping(manifest.get("artifact"), "artifact")
    if artifact.get("id") != trusted_artifact.get("id"):
        raise ReceiptVerificationError("receipt payload artifact id does not match GitHub")
    if artifact.get("name") != trusted_artifact.get("name"):
        raise ReceiptVerificationError("receipt payload artifact name does not match GitHub")
    if artifact.get("name") != expected_payload_name:
        raise ReceiptVerificationError(
            "receipt payload artifact name does not match immutable run namespace"
        )
    receipt_artifact_digest = _normalize_digest(artifact.get("digest"), "artifact.digest")
    api_digest = _normalize_digest(
        trusted_artifact.get("digest"), "GitHub artifact digest", required=False
    )
    if api_digest and receipt_artifact_digest != api_digest:
        raise ReceiptVerificationError("receipt payload artifact digest does not match GitHub")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ReceiptVerificationError("files must be a list")
    expected_paths = set(PAYLOAD_FILES)
    seen: set[str] = set()
    for raw_file in files:
        file_info = _require_mapping(raw_file, "files entry")
        relative = _safe_relative_path(file_info.get("path"))
        relative_name = relative.as_posix()
        if relative_name in seen:
            raise ReceiptVerificationError(f"duplicate receipt file: {relative_name}")
        seen.add(relative_name)
        path = payload_dir.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ReceiptVerificationError(f"receipt payload file is missing: {relative_name}")
        if path.stat().st_size != file_info.get("size"):
            raise ReceiptVerificationError(f"receipt payload size changed: {relative_name}")
        expected_hash = _normalize_digest(file_info.get("sha256"), "file sha256")
        if sha256_file(path) != expected_hash:
            raise ReceiptVerificationError(f"receipt payload hash changed: {relative_name}")
    if seen != expected_paths:
        raise ReceiptVerificationError("receipt payload file set is incomplete or unexpected")

    measurements = _read_payload(payload_dir)
    _validate_payload_namespace(measurements, namespace)
    actual_payload_files = {
        path.relative_to(payload_dir).as_posix()
        for path in payload_dir.rglob("*")
        if path.is_file()
    }
    if actual_payload_files != expected_paths:
        raise ReceiptVerificationError("receipt payload contains unexpected files")
    if manifest.get("measurements") != measurements:
        raise ReceiptVerificationError("receipt measurements do not match payload files")
    expected_thresholds = {
        name: value.get("thresholds", {}) for name, value in measurements.items()
    }
    if manifest.get("thresholds") != expected_thresholds:
        raise ReceiptVerificationError("receipt thresholds do not match payload files")
    payload_export_id, payload_export_digest = _export_identity(measurements)
    if payload_export_id != expected_export_id or payload_export_digest != _normalize_digest(
        expected_export_digest, "expected_export_digest"
    ):
        raise ReceiptVerificationError("payload export identity does not match expected export")
    if _measurement_model_ids(measurements) != {expected_model_id}:
        raise ReceiptVerificationError("payload model identity does not match expected model")
    if _measurement_model_revision(measurements) != model_revision:
        raise ReceiptVerificationError("payload model revision does not match receipt")
    if _measurement_model_digests(measurements) != {
        _normalize_digest(expected_model_digest, "expected_model_digest")
    }:
        raise ReceiptVerificationError("payload model digest does not match expected model")


class GitHubActionsClient:
    """Minimal GitHub Actions API client with an injectable transport."""

    def __init__(
        self,
        token: str,
        *,
        api_url: str = "https://api.github.com",
        transport: Callable[[urllib.request.Request], bytes] | None = None,
    ) -> None:
        if not token:
            raise ReceiptVerificationError("a GitHub token is required")
        self._token = token
        self._api_url = api_url.rstrip("/")
        self._transport = transport or self._urlopen

    @staticmethod
    def _urlopen(request: urllib.request.Request) -> bytes:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return response.read()

    def get_bytes(self, path: str, query: Mapping[str, str] | None = None) -> bytes:
        url = f"{self._api_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "tether-receipt-verifier",
            },
        )
        return self._transport(request)

    def get_json(self, path: str, query: Mapping[str, str] | None = None) -> dict[str, Any]:
        try:
            value = json.loads(self.get_bytes(path, query))
        except json.JSONDecodeError as exc:
            raise ReceiptVerificationError("GitHub returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ReceiptVerificationError("GitHub returned an unexpected response")
        return value


def _verify_download_digest(data: bytes, artifact: Mapping[str, Any]) -> None:
    digest = _normalize_digest(artifact.get("digest"), "GitHub artifact digest", required=False)
    if digest and _sha256_bytes(data) != digest:
        raise ReceiptVerificationError("downloaded artifact digest does not match GitHub")


def _extract_zip(data: bytes, destination: Path) -> None:
    archive_path = destination / "artifact.zip"
    archive_path.write_bytes(data)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = _safe_relative_path(info.filename)
            target = destination.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    archive_path.unlink()


def download_verified_receipts(
    *,
    repository: str,
    expected_sha: str,
    expected_model_id: str,
    expected_model_digest: str,
    expected_export_id: str,
    expected_export_digest: str,
    destination: Path,
    token: str,
    client: GitHubActionsClient | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Discover, download, and verify receipts for an exact target SHA."""
    api = client or GitHubActionsClient(token)
    encoded_workflow = urllib.parse.quote(WORKFLOW_PATH, safe="")
    runs_response = api.get_json(
        f"/repos/{repository}/actions/workflows/{encoded_workflow}/runs",
        {
            "head_sha": expected_sha,
            "status": "success",
            "exclude_pull_requests": "true",
            "per_page": "100",
        },
    )
    runs = runs_response.get("workflow_runs")
    if not isinstance(runs, list):
        raise ReceiptVerificationError("GitHub workflow run response is malformed")
    run = select_trusted_run(runs, expected_repository=repository, expected_sha=expected_sha)
    run_id = int(run["id"])
    namespace = build_receipt_namespace(
        repository,
        expected_sha,
        run_id,
        int(run["run_attempt"]),
    )
    expected_payload_name, expected_manifest_name = receipt_artifact_names(namespace)
    artifacts_response = api.get_json(
        f"/repos/{repository}/actions/runs/{run_id}/artifacts", {"per_page": "100"}
    )
    artifacts = artifacts_response.get("artifacts")
    if not isinstance(artifacts, list):
        raise ReceiptVerificationError("GitHub artifact response is malformed")

    manifest_candidates = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("name") == expected_manifest_name
        and not item.get("expired")
    ]
    if len(manifest_candidates) != 1:
        raise ReceiptVerificationError("trusted run has no unique receipt manifest artifact")
    manifest_artifact = manifest_candidates[0]

    manifest_zip = api.get_bytes(
        f"/repos/{repository}/actions/artifacts/{int(manifest_artifact['id'])}/zip"
    )
    _verify_download_digest(manifest_zip, manifest_artifact)

    with tempfile.TemporaryDirectory(prefix="tether-receipt-") as temp_name:
        temp = Path(temp_name)
        manifest_dir = temp / "manifest"
        manifest_dir.mkdir()
        _extract_zip(manifest_zip, manifest_dir)
        manifest_path = manifest_dir / "receipt-manifest.json"
        if not manifest_path.is_file():
            raise ReceiptVerificationError("manifest artifact has no receipt-manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ReceiptVerificationError("receipt manifest is invalid JSON") from exc
        manifest = _require_mapping(manifest, "receipt manifest")
        artifact_binding = _require_mapping(manifest.get("artifact"), "artifact")
        if artifact_binding.get("name") != expected_payload_name:
            raise ReceiptVerificationError(
                "manifest payload artifact name does not match selected run namespace"
            )
        payload_candidates = [
            item
            for item in artifacts
            if isinstance(item, Mapping)
            and item.get("id") == artifact_binding.get("id")
            and item.get("name") == expected_payload_name
            and not item.get("expired")
        ]
        if len(payload_candidates) != 1:
            raise ReceiptVerificationError("manifest payload artifact is absent or ambiguous")
        payload_artifact = payload_candidates[0]
        payload_zip = api.get_bytes(
            f"/repos/{repository}/actions/artifacts/{int(payload_artifact['id'])}/zip"
        )
        _verify_download_digest(payload_zip, payload_artifact)
        payload_dir = temp / "payload"
        payload_dir.mkdir()
        _extract_zip(payload_zip, payload_dir)
        verify_receipt_manifest(
            manifest,
            payload_dir=payload_dir,
            trusted_run=run,
            trusted_artifact=payload_artifact,
            expected_repository=repository,
            expected_sha=expected_sha,
            expected_model_id=expected_model_id,
            expected_model_digest=expected_model_digest,
            expected_export_id=expected_export_id,
            expected_export_digest=expected_export_digest,
            now=now,
        )
        destination.mkdir(parents=True, exist_ok=True)
        for name in PAYLOAD_FILES:
            shutil.copy2(payload_dir / name, destination / name)
        return dict(manifest)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _build_command(args: argparse.Namespace) -> int:
    manifest = build_receipt_manifest(
        payload_dir=args.payload_dir,
        repository=args.repository,
        source_sha=args.source_sha,
        source_ref=args.source_ref,
        workflow_run_id=args.run_id,
        workflow_run_attempt=args.run_attempt,
        receipt_namespace=args.receipt_namespace,
        event=args.event,
        payload_artifact_id=args.artifact_id,
        payload_artifact_name=args.artifact_name,
        payload_artifact_digest=args.artifact_digest,
        model_id=args.model_id,
        model_digest=args.model_digest,
        expires_in_hours=args.expires_in_hours,
    )
    _write_json(args.output, manifest)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a") as output:
            output.write(f"export_id={manifest['export']['id']}\n")
            output.write(f"export_digest={manifest['export']['digest']}\n")
    return 0


def _namespace_command(args: argparse.Namespace) -> int:
    namespace = build_receipt_namespace(
        args.repository, args.source_sha, args.run_id, args.run_attempt
    )
    payload_name, manifest_name = receipt_artifact_names(namespace)
    print(namespace)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a") as output:
            output.write(f"receipt_namespace={namespace}\n")
            output.write(f"payload_artifact_name={payload_name}\n")
            output.write(f"manifest_artifact_name={manifest_name}\n")
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text())
    trusted_run = json.loads(args.trusted_run.read_text())
    trusted_artifact = json.loads(args.trusted_artifact.read_text())
    verify_receipt_manifest(
        manifest,
        payload_dir=args.payload_dir,
        trusted_run=trusted_run,
        trusted_artifact=trusted_artifact,
        expected_repository=args.repository,
        expected_sha=args.source_sha,
        expected_model_id=args.model_id,
        expected_model_digest=args.model_digest,
        expected_export_id=args.export_id,
        expected_export_digest=args.export_digest,
    )
    return 0


def _verify_local_command(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text())
    branch = _branch_from_ref(args.source_ref)
    trusted_run = {
        "id": args.run_id,
        "path": f"{WORKFLOW_PATH}@{args.source_ref}",
        "head_sha": args.source_sha,
        "head_branch": branch,
        "head_repository": {"full_name": args.repository},
        "event": args.event,
        "status": "completed",
        "conclusion": "success",
        "run_attempt": args.run_attempt,
    }
    trusted_artifact = {
        "id": args.artifact_id,
        "name": args.artifact_name,
        "digest": args.artifact_digest,
        "expired": False,
    }
    verify_receipt_manifest(
        manifest,
        payload_dir=args.payload_dir,
        trusted_run=trusted_run,
        trusted_artifact=trusted_artifact,
        expected_repository=args.repository,
        expected_sha=args.source_sha,
        expected_model_id=args.model_id,
        expected_model_digest=args.model_digest,
        expected_export_id=args.export_id,
        expected_export_digest=args.export_digest,
    )
    return 0


def _download_command(args: argparse.Namespace) -> int:
    download_verified_receipts(
        repository=args.repository,
        expected_sha=args.source_sha,
        expected_model_id=args.model_id,
        expected_model_digest=args.model_digest,
        expected_export_id=args.export_id,
        expected_export_digest=args.export_digest,
        destination=args.destination,
        token=args.token or os.environ.get("GITHUB_TOKEN", ""),
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    namespace = subparsers.add_parser(
        "namespace", help="derive the immutable namespace for one workflow run"
    )
    namespace.add_argument("--repository", required=True)
    namespace.add_argument("--source-sha", required=True)
    namespace.add_argument("--run-id", type=int, required=True)
    namespace.add_argument("--run-attempt", type=int, required=True)
    namespace.set_defaults(handler=_namespace_command)

    build = subparsers.add_parser("build", help="build a provenance manifest")
    build.add_argument("--payload-dir", type=Path, required=True)
    build.add_argument("--repository", required=True)
    build.add_argument("--source-sha", required=True)
    build.add_argument("--source-ref", required=True)
    build.add_argument("--run-id", type=int, required=True)
    build.add_argument("--run-attempt", type=int, required=True)
    build.add_argument("--receipt-namespace", required=True)
    build.add_argument("--event", choices=sorted(ALLOWED_EVENTS), required=True)
    build.add_argument("--artifact-id", type=int, required=True)
    build.add_argument("--artifact-name", required=True)
    build.add_argument("--artifact-digest", required=True)
    build.add_argument("--model-id", required=True)
    build.add_argument("--model-digest", required=True)
    build.add_argument("--expires-in-hours", type=int, default=168)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(handler=_build_command)

    verify = subparsers.add_parser("verify", help="verify local downloaded artifacts")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--payload-dir", type=Path, required=True)
    verify.add_argument("--trusted-run", type=Path, required=True)
    verify.add_argument("--trusted-artifact", type=Path, required=True)
    verify.add_argument("--repository", required=True)
    verify.add_argument("--source-sha", required=True)
    verify.add_argument("--model-id", required=True)
    verify.add_argument("--model-digest", required=True)
    verify.add_argument("--export-id", required=True)
    verify.add_argument("--export-digest", required=True)
    verify.set_defaults(handler=_verify_command)

    verify_local = subparsers.add_parser(
        "verify-local", help="verify artifacts downloaded from the current trusted run"
    )
    verify_local.add_argument("--manifest", type=Path, required=True)
    verify_local.add_argument("--payload-dir", type=Path, required=True)
    verify_local.add_argument("--repository", required=True)
    verify_local.add_argument("--source-sha", required=True)
    verify_local.add_argument("--source-ref", required=True)
    verify_local.add_argument("--run-id", type=int, required=True)
    verify_local.add_argument("--run-attempt", type=int, required=True)
    verify_local.add_argument("--event", choices=sorted(ALLOWED_EVENTS), required=True)
    verify_local.add_argument("--artifact-id", type=int, required=True)
    verify_local.add_argument("--artifact-name", required=True)
    verify_local.add_argument("--artifact-digest", required=True)
    verify_local.add_argument("--model-id", required=True)
    verify_local.add_argument("--model-digest", required=True)
    verify_local.add_argument("--export-id", required=True)
    verify_local.add_argument("--export-digest", required=True)
    verify_local.set_defaults(handler=_verify_local_command)

    download = subparsers.add_parser("download", help="download a trusted receipt from GitHub")
    download.add_argument("--repository", required=True)
    download.add_argument("--source-sha", required=True)
    download.add_argument("--model-id", required=True)
    download.add_argument("--model-digest", required=True)
    download.add_argument("--export-id", required=True)
    download.add_argument("--export-digest", required=True)
    download.add_argument("--destination", type=Path, required=True)
    download.add_argument("--token")
    download.set_defaults(handler=_download_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except ReceiptVerificationError as exc:
        print(f"receipt verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
