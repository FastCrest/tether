"""Tether-owned Improve worker contract and deterministic dry-run runner.

This module is intentionally stdlib-only. It is the first Tether-side slice for
the FastCrest Improve loop: consume a Cloud-built worker input, locally enforce
the selected bounded evidence contract, materialize a selected-evidence
manifest, and return the worker-result envelope that Cloud already ingests.

Real GPU training stays behind a follow-up Modal command until credentials,
storage, and model targets are configured:

    modal run scripts/real_improve_worker_modal.py --worker-input <worker-input.json> \
        --output-uri s3://<bucket>/improve/<job_id>/ --gpu A10G
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKER_INPUT_CONTRACT_VERSION = "fastcrest.improve.worker_input.v1"
WORKER_RESULT_CONTRACT_VERSION = "fastcrest.improve.worker_result.v1"
WORKER_RUNNER = "tether_dry_run"
MAX_WORKER_INPUT_BYTES = 128 * 1024

MODAL_GPU_FOLLOW_UP_COMMAND = (
    "modal run scripts/real_improve_worker_modal.py "
    "--worker-input <worker-input.json> "
    "--output-uri s3://<bucket>/improve/<job_id>/ "
    "--gpu A10G"
)

_BOUNDED_EVIDENCE_FIELDS = (
    ("action_chunks", "action_chunks_uri", "action_chunks_sha256"),
    ("trace", "trace_uri", "trace_sha256"),
    ("frames", "frames_uri", "frames_sha256"),
)
_RAW_EVIDENCE_FIELDS = {
    "actions",
    "action_chunks",
    "frames",
    "image_bytes",
    "observations",
    "payload",
    "raw_payload",
    "trace",
    "video",
    "video_bytes",
}
_FALSEY_FAILURE_MODES = {"", "0", "false", "no", "none", "ok", "success"}


class ImproveWorkerError(ValueError):
    """A typed, envelope-safe worker failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BoundedEvidenceRef:
    kind: str
    uri: str
    sha256: str


@dataclass(frozen=True)
class SelectedEvidence:
    failure_id: str
    fields: dict[str, Any]
    refs: tuple[BoundedEvidenceRef, ...]


@dataclass(frozen=True)
class MaterializedEvidence:
    output_dir: Path
    manifest_path: Path
    manifest_sha256: str
    selected: tuple[SelectedEvidence, ...]


def run_improve_worker(
    worker_input: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Run the deterministic local Improve worker.

    All contract, trainability, and materialization failures are returned as
    ``ok: false`` worker-result envelopes. Unexpected local IO/runtime failures
    are also contained so callers never get a traceback-shaped result.
    """

    context = _best_effort_context(worker_input)
    try:
        payload = validate_worker_input(worker_input)
        context = _best_effort_context(payload)
        runner_config = _object(payload.get("runner_config"))
        failure_mode = _requested_failure_mode(payload, runner_config)
        if failure_mode is not None:
            return _failure_result(
                payload,
                "requested_failure",
                f"dry-run worker failed by request: {failure_mode}",
                failure_mode=failure_mode,
            )
        materialized = materialize_selected_evidence(
            payload,
            output_dir=output_dir,
            now=time.time() if now is None else now,
        )
        return _success_result(payload, materialized)
    except ImproveWorkerError as exc:
        return _failure_result(context, exc.code, str(exc))
    except (OSError, TypeError, ValueError) as exc:
        return _failure_result(context, "worker_failure", str(exc))


def validate_worker_input(
    worker_input: Mapping[str, Any],
    *,
    max_bytes: int = MAX_WORKER_INPUT_BYTES,
) -> dict[str, Any]:
    """Validate and normalize a Cloud Improve worker input payload."""

    payload = _bounded_json_object(worker_input, "worker input", max_bytes)
    version = payload.get("contract_version") or payload.get("schema_version")
    if version != WORKER_INPUT_CONTRACT_VERSION:
        raise ImproveWorkerError("invalid_worker_input", "contract_version is not supported")
    payload["contract_version"] = WORKER_INPUT_CONTRACT_VERSION

    request = _required_object(payload, "request")
    job = _required_object(payload, "job")
    parent = _required_object(payload, "parent_artifact")
    failures = payload.get("failure_evidence")
    if not isinstance(failures, list):
        raise ImproveWorkerError("invalid_worker_input", "failure_evidence must be a list")
    for index, row in enumerate(failures):
        if not isinstance(row, dict):
            raise ImproveWorkerError(
                "invalid_worker_input",
                f"failure_evidence[{index}] must be an object",
            )

    workspace_id = _required_str(request, "workspace_id", field="request.workspace_id")
    request_id = _required_str(request, "id", field="request.id")
    cluster_id = _required_str(request, "cluster_id", field="request.cluster_id")
    parent_artifact_id = _required_str(
        request,
        "parent_artifact_id",
        field="request.parent_artifact_id",
    )
    job_id = _required_str(job, "id", field="job.id")
    job_request_id = _required_str(job, "request_id", field="job.request_id")
    parent_id = _required_str(parent, "id", field="parent_artifact.id")
    _required_str(parent, "optimized_artifact_url", field="parent_artifact.optimized_artifact_url")
    _require_sha256(
        parent,
        "optimized_artifact_digest",
        field="parent_artifact.optimized_artifact_digest",
    )

    selected_failure_ids = _selected_failure_ids(payload)
    request_failure_ids = request.get("selected_failure_ids")
    if not isinstance(request_failure_ids, list):
        raise ImproveWorkerError(
            "invalid_worker_input",
            "request.selected_failure_ids must be a list",
        )
    request_ids = [_clean_identifier(value, "request.selected_failure_ids") for value in request_failure_ids]
    if request_ids != selected_failure_ids:
        raise ImproveWorkerError(
            "selected_id_mismatch",
            "selected_failure_ids must match request.selected_failure_ids",
        )

    top_level_matches = {
        "workspace_id": workspace_id,
        "request_id": request_id,
        "job_id": job_id,
        "cluster_id": cluster_id,
        "parent_artifact_id": parent_artifact_id,
    }
    for key, expected in top_level_matches.items():
        if payload.get(key) != expected:
            raise ImproveWorkerError(
                "invalid_worker_input",
                f"{key} must match nested payload identifiers",
            )
    if job.get("workspace_id") != workspace_id:
        raise ImproveWorkerError(
            "invalid_worker_input",
            "job.workspace_id must match request.workspace_id",
        )
    if job_request_id != request_id:
        raise ImproveWorkerError("invalid_worker_input", "job.request_id must match request.id")
    if parent.get("workspace_id") != workspace_id:
        raise ImproveWorkerError(
            "invalid_worker_input",
            "parent_artifact.workspace_id must match request.workspace_id",
        )
    if parent_id != parent_artifact_id:
        raise ImproveWorkerError(
            "invalid_worker_input",
            "parent_artifact.id must match request.parent_artifact_id",
        )

    for field in ("train_config", "validation_config", "runner_config"):
        value = payload.get(field, {})
        if value is not None and not isinstance(value, dict):
            raise ImproveWorkerError("invalid_worker_input", f"{field} must be an object")
        payload[field] = dict(value or {})
    return payload


def materialize_selected_evidence(
    payload: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    now: float | None = None,
) -> MaterializedEvidence:
    """Write a selected-evidence manifest for dry-run/local worker execution."""

    selected = _validated_selected_evidence(payload, now=time.time() if now is None else now)
    run_dir = _resolve_output_dir(payload, output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "fastcrest.improve.evidence_manifest.v1",
        "workspace_id": payload["workspace_id"],
        "request_id": payload["request_id"],
        "job_id": payload["job_id"],
        "cluster_id": payload["cluster_id"],
        "parent_artifact_id": payload["parent_artifact_id"],
        "selected_failure_ids": list(payload["selected_failure_ids"]),
        "evidence": [_evidence_manifest_row(item) for item in selected],
    }
    manifest_sha256 = _sha256_bytes(_canonical_json(manifest))
    manifest_with_hash = {**manifest, "manifest_sha256": manifest_sha256}
    manifest_path = run_dir / "selected_evidence_manifest.json"
    _write_json(manifest_path, manifest_with_hash)
    return MaterializedEvidence(
        output_dir=run_dir,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        selected=tuple(selected),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Tether Improve dry-run worker.")
    parser.add_argument("--worker-input", required=True, help="Worker input JSON file, or '-' for stdin.")
    parser.add_argument("--output-dir", help="Directory for local dry-run artifacts.")
    parser.add_argument("--result-output", "--output", dest="result_output", help="Write result JSON here.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print result JSON.")
    args = parser.parse_args(argv)

    try:
        payload = _read_json_input(args.worker_input)
    except (OSError, json.JSONDecodeError) as exc:
        result = _failure_result({}, "invalid_worker_input", f"could not read worker input: {exc}")
    else:
        result = run_improve_worker(payload, output_dir=args.output_dir)

    encoded = json.dumps(
        result,
        indent=2 if args.pretty else None,
        sort_keys=True,
        separators=None if args.pretty else (",", ":"),
    )
    if args.result_output:
        Path(args.result_output).write_text(encoded + "\n", encoding="utf-8")
    else:
        sys.stdout.write(encoded + "\n")
    return 0


def _success_result(
    payload: Mapping[str, Any],
    materialized: MaterializedEvidence,
) -> dict[str, Any]:
    artifact_path = materialized.output_dir / "candidate_artifact.json"
    metrics = _metrics(payload)
    artifact_payload = {
        "schema_version": "fastcrest.improve.dry_run_artifact.v1",
        "runner": WORKER_RUNNER,
        "workspace_id": payload["workspace_id"],
        "request_id": payload["request_id"],
        "job_id": payload["job_id"],
        "parent_artifact_id": payload["parent_artifact_id"],
        "parent_artifact_digest": _object(payload["parent_artifact"]).get(
            "optimized_artifact_digest",
        ),
        "selected_failure_ids": list(payload["selected_failure_ids"]),
        "evidence_manifest_sha256": materialized.manifest_sha256,
        "metrics": metrics,
        "train_config": _object(payload.get("train_config")),
        "validation_config": _object(payload.get("validation_config")),
        "deterministic": True,
    }
    artifact_bytes = _canonical_json(artifact_payload)
    artifact_sha256 = _sha256_bytes(artifact_bytes)
    artifact_path.write_bytes(artifact_bytes + b"\n")

    log_path = materialized.output_dir / "training_log.jsonl"
    verification_path = materialized.output_dir / "VERIFICATION.md"
    modal_command = _modal_follow_up_command(payload)
    _write_training_log(log_path, payload, materialized, artifact_sha256, modal_command)
    _write_verification_md(verification_path, payload, artifact_sha256, modal_command)

    parent = _object(payload.get("parent_artifact"))
    result = {
        "schema_version": WORKER_RESULT_CONTRACT_VERSION,
        "ok": True,
        "runner": WORKER_RUNNER,
        "request_id": payload["request_id"],
        "job_id": payload["job_id"],
        "artifact_uri": artifact_path.resolve().as_uri(),
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": artifact_path.stat().st_size,
        "target_hardware_class": parent.get("target_hardware_class") or "orin",
        "runtime": parent.get("runtime") or "tensorrt",
        "precision": parent.get("precision") or "fp16",
        "shape_mode": parent.get("shape_mode") or "static",
        "batch_size": parent.get("batch_size"),
        "action_chunk_size": parent.get("action_chunk_size") or 16,
        "cert_status": "dry_run",
        "builder_version": "tether-improve-worker-dry-run",
        "tether_version": "local",
        "metrics": metrics,
        "outputs": {
            "manifest_uri": materialized.manifest_path.resolve().as_uri(),
            "training_log_uri": log_path.resolve().as_uri(),
            "verification_md_uri": verification_path.resolve().as_uri(),
            "final_checkpoint_uri": artifact_path.resolve().as_uri(),
        },
        "metadata": {
            "source": "tether_improve_worker",
            "dry_run": True,
            "deterministic": True,
            "workspace_id": payload["workspace_id"],
            "cluster_id": payload["cluster_id"],
            "selected_failure_ids": list(payload["selected_failure_ids"]),
            "parent_artifact_id": payload["parent_artifact_id"],
            "evidence_manifest_sha256": materialized.manifest_sha256,
            "materialized_evidence_count": len(materialized.selected),
            "modal_gpu_follow_up_command": modal_command,
        },
    }
    result["worker_result_hash"] = _hash_payload(result)
    return result


def _failure_result(
    payload: Mapping[str, Any],
    code: str,
    message: str,
    *,
    failure_mode: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": WORKER_RESULT_CONTRACT_VERSION,
        "ok": False,
        "runner": WORKER_RUNNER,
        "error_code": code,
        "error": message,
        "metadata": {
            "source": "tether_improve_worker",
            "dry_run": True,
            "failure_mode": failure_mode or code,
            "modal_gpu_follow_up_command": _modal_follow_up_command(payload),
        },
    }
    for key in ("workspace_id", "request_id", "job_id", "cluster_id", "parent_artifact_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            result[key] = value
    selected = payload.get("selected_failure_ids")
    if isinstance(selected, list):
        result["metadata"]["selected_failure_ids"] = [
            value for value in selected if isinstance(value, str)
        ]
    result["worker_result_hash"] = _hash_payload(result)
    return result


def _validated_selected_evidence(
    payload: Mapping[str, Any],
    *,
    now: float,
) -> list[SelectedEvidence]:
    selected_ids = _selected_failure_ids(payload)
    selected_set = set(selected_ids)
    failures = payload.get("failure_evidence")
    if not isinstance(failures, list):
        raise ImproveWorkerError("invalid_worker_input", "failure_evidence must be a list")

    by_id: dict[str, SelectedEvidence] = {}
    for index, row in enumerate(failures):
        evidence = _object(row)
        failure_id = _failure_id(evidence, index)
        if failure_id not in selected_set:
            raise ImproveWorkerError(
                "selected_id_mismatch",
                f"failure_evidence contains unselected failure id {failure_id}",
            )
        if failure_id in by_id:
            raise ImproveWorkerError(
                "selected_id_mismatch",
                f"failure_evidence contains duplicate selected failure id {failure_id}",
            )
        _validate_evidence_scope(payload, evidence, failure_id)
        _validate_trainability(evidence, failure_id, now=now)
        refs = _bounded_refs(evidence, failure_id)
        by_id[failure_id] = SelectedEvidence(
            failure_id=failure_id,
            fields=_bounded_evidence_fields(evidence, failure_id),
            refs=tuple(refs),
        )

    missing = [failure_id for failure_id in selected_ids if failure_id not in by_id]
    if missing:
        raise ImproveWorkerError(
            "insufficient_evidence",
            f"missing selected failure evidence: {', '.join(missing)}",
        )
    return [by_id[failure_id] for failure_id in selected_ids]


def _validate_evidence_scope(
    payload: Mapping[str, Any],
    evidence: Mapping[str, Any],
    failure_id: str,
) -> None:
    for raw_field in sorted(_RAW_EVIDENCE_FIELDS):
        if raw_field in evidence:
            raise ImproveWorkerError(
                "unbounded_evidence_payload",
                f"failure_evidence {failure_id} includes raw field {raw_field}",
            )
    workspace_id = evidence.get("workspace_id")
    if workspace_id is not None and workspace_id != payload["workspace_id"]:
        raise ImproveWorkerError(
            "invalid_worker_input",
            f"failure_evidence {failure_id} workspace_id must match request.workspace_id",
        )
    cluster_id = evidence.get("cluster_id")
    if cluster_id is not None and cluster_id != payload["cluster_id"]:
        raise ImproveWorkerError(
            "invalid_worker_input",
            f"failure_evidence {failure_id} cluster_id must match request.cluster_id",
        )
    artifact_id = evidence.get("artifact_id")
    if artifact_id is not None and artifact_id != payload["parent_artifact_id"]:
        raise ImproveWorkerError(
            "invalid_worker_input",
            f"failure_evidence {failure_id} artifact_id must match parent_artifact_id",
        )


def _validate_trainability(evidence: Mapping[str, Any], failure_id: str, *, now: float) -> None:
    if evidence.get("do_not_train") is True:
        raise ImproveWorkerError(
            "untrainable_evidence",
            f"failure_evidence {failure_id} is marked do_not_train",
        )
    if evidence.get("deleted_at") is not None:
        raise ImproveWorkerError(
            "untrainable_evidence",
            f"failure_evidence {failure_id} is deleted",
        )
    retention_expires_at = evidence.get("retention_expires_at")
    if retention_expires_at is not None:
        try:
            expires_at = float(retention_expires_at)
        except (TypeError, ValueError) as exc:
            raise ImproveWorkerError(
                "invalid_worker_input",
                f"failure_evidence {failure_id} retention_expires_at must be numeric",
            ) from exc
        if expires_at <= now:
            raise ImproveWorkerError(
                "retention_expired",
                f"failure_evidence {failure_id} retention has expired",
            )


def _bounded_refs(
    evidence: Mapping[str, Any],
    failure_id: str,
) -> list[BoundedEvidenceRef]:
    refs: list[BoundedEvidenceRef] = []
    for kind, uri_field, sha_field in _BOUNDED_EVIDENCE_FIELDS:
        uri = evidence.get(uri_field)
        sha = evidence.get(sha_field)
        if uri is None and sha is None:
            continue
        if not isinstance(uri, str) or not uri.strip():
            raise ImproveWorkerError(
                "insufficient_evidence",
                f"failure_evidence {failure_id} missing {uri_field}",
            )
        refs.append(BoundedEvidenceRef(kind=kind, uri=uri.strip(), sha256=_sha256_value(sha, sha_field)))
    if not refs:
        raise ImproveWorkerError(
            "insufficient_evidence",
            f"failure_evidence {failure_id} has no bounded evidence URI/hash pairs",
        )
    return refs


def _bounded_evidence_fields(
    evidence: Mapping[str, Any],
    failure_id: str,
) -> dict[str, Any]:
    kept: dict[str, Any] = {"failure_id": failure_id}
    for key in (
        "id",
        "workspace_id",
        "cluster_id",
        "artifact_id",
        "assignment_id",
        "device_id",
        "event_type",
        "severity",
        "source",
        "started_at",
        "ended_at",
        "rollout_id",
        "rollout_stage_id",
        "rollout_device_step_id",
    ):
        if key in evidence and _json_safe_scalar(evidence[key]):
            kept[key] = evidence[key]
    return kept


def _evidence_manifest_row(item: SelectedEvidence) -> dict[str, Any]:
    return {
        **item.fields,
        "bounded_refs": [
            {"kind": ref.kind, "uri": ref.uri, "sha256": ref.sha256}
            for ref in item.refs
        ],
    }


def _metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    runner_config = _object(payload.get("runner_config"))
    parent_metadata = _object(_object(payload.get("parent_artifact")).get("metadata"))
    train_config = _object(payload.get("train_config"))
    p99_ms = _metric_value(runner_config, parent_metadata, "p99_ms", default=50.0, multiplier=0.8)
    peak_memory_mb = _metric_value(
        runner_config,
        parent_metadata,
        "peak_memory_mb",
        default=1000.0,
        multiplier=0.9,
    )
    mmd_score = _configured_float(runner_config, "mmd_score", default=0.02)
    return {
        "p50_ms": round(p99_ms * 0.6, 3),
        "p95_ms": round(p99_ms * 0.85, 3),
        "p99_ms": p99_ms,
        "peak_memory_mb": peak_memory_mb,
        "mmd": mmd_score,
        "mmd_score": mmd_score,
        "training_steps": float(_training_steps(train_config, runner_config)),
        "modal_cost_usd": float(runner_config.get("modal_cost_usd") or 0.0),
    }


def _metric_value(
    runner_config: Mapping[str, Any],
    parent_metadata: Mapping[str, Any],
    key: str,
    *,
    default: float,
    multiplier: float,
) -> float:
    if key in runner_config:
        return _non_negative_float(runner_config[key], f"runner_config.{key}")
    if key in parent_metadata:
        return round(
            _non_negative_float(parent_metadata[key], f"parent_artifact.metadata.{key}") * multiplier,
            3,
        )
    return default


def _configured_float(
    runner_config: Mapping[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    if key in runner_config:
        return _non_negative_float(runner_config[key], f"runner_config.{key}")
    fallback = "mmd" if key == "mmd_score" else key
    if fallback in runner_config:
        return _non_negative_float(runner_config[fallback], f"runner_config.{fallback}")
    return default


def _training_steps(
    train_config: Mapping[str, Any],
    runner_config: Mapping[str, Any],
) -> int:
    for source in (runner_config, train_config):
        for key in ("training_steps", "steps", "num_steps"):
            if key in source:
                value = source[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ImproveWorkerError(
                        "invalid_runner_config",
                        f"{key} must be numeric",
                    )
                steps = int(value)
                if steps < 0:
                    raise ImproveWorkerError(
                        "invalid_runner_config",
                        f"{key} must be non-negative",
                    )
                return steps
    return 10


def _write_training_log(
    path: Path,
    payload: Mapping[str, Any],
    materialized: MaterializedEvidence,
    artifact_sha256: str,
    modal_command: str,
) -> None:
    rows = [
        {"event": "worker_start", "runner": WORKER_RUNNER, "job_id": payload["job_id"]},
        {
            "event": "evidence_materialized",
            "manifest_sha256": materialized.manifest_sha256,
            "selected_failure_ids": list(payload["selected_failure_ids"]),
        },
        {"event": "dry_run_complete", "artifact_sha256": artifact_sha256},
        {"event": "modal_gpu_follow_up", "command": modal_command},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _write_verification_md(
    path: Path,
    payload: Mapping[str, Any],
    artifact_sha256: str,
    modal_command: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Tether Improve Worker Dry Run",
                "",
                f"- job_id: {payload['job_id']}",
                f"- request_id: {payload['request_id']}",
                f"- artifact_sha256: {artifact_sha256}",
                "- GPU training launched: false",
                "",
                "Run the credentialed GPU smoke with:",
                "",
                f"```bash\n{modal_command}\n```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _resolve_output_dir(payload: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return Path.cwd() / "tether_improve_worker" / str(payload.get("job_id") or "unknown_job")
    return Path(output_dir) / str(payload.get("job_id") or "unknown_job")


def _modal_follow_up_command(payload: Mapping[str, Any]) -> str:
    job_id = str(payload.get("job_id") or "<job_id>")
    return MODAL_GPU_FOLLOW_UP_COMMAND.replace("<job_id>", job_id)


def _requested_failure_mode(
    payload: Mapping[str, Any],
    runner_config: Mapping[str, Any],
) -> str | None:
    for source in (payload, runner_config):
        for key in ("failure_mode", "requested_failure_mode"):
            value = source.get(key)
            if value is None:
                continue
            mode = str(value).strip()
            if mode.lower() not in _FALSEY_FAILURE_MODES:
                return mode
    if payload.get("fail") is True or payload.get("ok") is False or runner_config.get("fail") is True:
        return "requested_failure"
    return None


def _selected_failure_ids(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("selected_failure_ids")
    if not isinstance(values, list) or not values:
        raise ImproveWorkerError(
            "invalid_worker_input",
            "selected_failure_ids must contain at least one id",
        )
    return [_clean_identifier(value, "selected_failure_ids") for value in values]


def _failure_id(evidence: Mapping[str, Any], index: int) -> str:
    raw = evidence.get("failure_id") or evidence.get("id")
    return _clean_identifier(raw, f"failure_evidence[{index}].id")


def _clean_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImproveWorkerError(
            "invalid_worker_input",
            f"{field} must contain non-empty string identifiers",
        )
    return value.strip()


def _required_object(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ImproveWorkerError("invalid_worker_input", f"{key} must be an object")
    return dict(value)


def _required_str(data: Mapping[str, Any], key: str, *, field: str | None = None) -> str:
    name = field or key
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ImproveWorkerError("invalid_worker_input", f"{name} is required")
    return value.strip()


def _require_sha256(data: Mapping[str, Any], key: str, *, field: str | None = None) -> str:
    return _sha256_value(data.get(key), field or key)


def _sha256_value(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ImproveWorkerError("invalid_worker_input", f"{field} must be a sha256 string")
    cleaned = value.strip().lower()
    if len(cleaned) != 64 or any(char not in "0123456789abcdef" for char in cleaned):
        raise ImproveWorkerError(
            "invalid_worker_input",
            f"{field} must be a 64-character hex sha256",
        )
    return cleaned


def _bounded_json_object(
    value: Mapping[str, Any],
    field: str,
    max_bytes: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ImproveWorkerError("invalid_worker_input", f"{field} must be an object")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ImproveWorkerError(
            "invalid_worker_input",
            f"{field} must be JSON serializable",
        ) from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise ImproveWorkerError("invalid_worker_input", f"{field} is too large")
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ImproveWorkerError("invalid_worker_input", f"{field} must be an object")
    return decoded


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json_safe_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _non_negative_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImproveWorkerError("invalid_runner_config", f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ImproveWorkerError("invalid_runner_config", f"{field} must be finite and non-negative")
    return parsed


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json_input(worker_input_path: str) -> dict[str, Any]:
    if worker_input_path == "-":
        data = sys.stdin.read()
    else:
        data = Path(worker_input_path).read_text(encoding="utf-8")
    decoded = json.loads(data)
    if not isinstance(decoded, dict):
        raise json.JSONDecodeError("worker input must be an object", data, 0)
    return decoded


def _best_effort_context(worker_input: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(worker_input, Mapping):
        return {}
    context: dict[str, Any] = {}
    for key in ("workspace_id", "request_id", "job_id", "cluster_id", "parent_artifact_id"):
        value = worker_input.get(key)
        if isinstance(value, str) and value:
            context[key] = value
    selected = worker_input.get("selected_failure_ids")
    if isinstance(selected, list):
        context["selected_failure_ids"] = [value for value in selected if isinstance(value, str)]
    return context


__all__ = [
    "ImproveWorkerError",
    "MODAL_GPU_FOLLOW_UP_COMMAND",
    "WORKER_INPUT_CONTRACT_VERSION",
    "WORKER_RESULT_CONTRACT_VERSION",
    "WORKER_RUNNER",
    "materialize_selected_evidence",
    "run_improve_worker",
    "validate_worker_input",
]


if __name__ == "__main__":
    raise SystemExit(main())
