from __future__ import annotations

import json
import os
import string
import subprocess
import sys
from pathlib import Path

from tether.finetune.improve_worker import (
    MODAL_GPU_FOLLOW_UP_COMMAND,
    WORKER_RESULT_CONTRACT_VERSION,
    run_improve_worker,
)


FUTURE_RETENTION = 4_102_444_800.0


def _failure(failure_id: str, *, with_evidence: bool = True) -> dict[str, object]:
    row: dict[str, object] = {
        "id": failure_id,
        "workspace_id": "ws_1",
        "cluster_id": "fcl_1",
        "artifact_id": "art_parent",
        "event_type": "safety_clamp",
        "severity": "critical",
        "device_id": "edge_1",
        "do_not_train": False,
        "deleted_at": None,
        "retention_expires_at": FUTURE_RETENTION,
    }
    if with_evidence:
        row.update(
            {
                "action_chunks_uri": f"s3://bucket/failures/{failure_id}/chunks.jsonl",
                "action_chunks_sha256": "b" * 64,
            }
        )
    return row


def _worker_input(
    *,
    selected_failure_ids: list[str] | None = None,
    failure_evidence: list[dict[str, object]] | None = None,
    runner_config: dict[str, object] | None = None,
) -> dict[str, object]:
    selected = selected_failure_ids or ["fail_1"]
    failures = failure_evidence if failure_evidence is not None else [_failure(selected[0])]
    return {
        "contract_version": "fastcrest.improve.worker_input.v1",
        "workspace_id": "ws_1",
        "request_id": "imp_req_1",
        "job_id": "imp_job_1",
        "cluster_id": "fcl_1",
        "parent_artifact_id": "art_parent",
        "selected_failure_ids": list(selected),
        "request": {
            "id": "imp_req_1",
            "workspace_id": "ws_1",
            "cluster_id": "fcl_1",
            "selected_failure_ids": list(selected),
            "parent_artifact_id": "art_parent",
            "attestation": "operator reviewed bounded edge evidence for training",
            "trainability_snapshot": {"status": "trainable"},
            "retention_snapshot": {"retention_expires_at": FUTURE_RETENTION},
            "metadata": {},
            "created_at": 1.0,
        },
        "job": {
            "id": "imp_job_1",
            "workspace_id": "ws_1",
            "request_id": "imp_req_1",
            "recipe": "worker_launch",
            "train_config": {"steps": 25},
            "validation_config": {"max_p99_ms": 90.0},
            "metadata": {},
            "created_at": 1.0,
        },
        "parent_artifact": {
            "id": "art_parent",
            "workspace_id": "ws_1",
            "source_model_url": "hf://lerobot/smolvla",
            "source_model_digest": "d" * 64,
            "optimized_artifact_url": "s3://bucket/current.tar.gz",
            "optimized_artifact_digest": "a" * 64,
            "artifact_size_bytes": 1234,
            "target_hardware_class": "orin",
            "runtime": "tensorrt",
            "precision": "fp16",
            "shape_mode": "static",
            "batch_size": 1,
            "action_chunk_size": 16,
            "metadata": {"p99_ms": 100.0, "peak_memory_mb": 1000.0},
        },
        "failure_evidence": failures,
        "failures": failures,
        "runner": "tether_modal",
        "recipe": "worker_launch",
        "train_config": {"steps": 25},
        "validation_config": {"max_p99_ms": 90.0},
        "runner_config": runner_config or {},
        "trainability_snapshot": {"status": "trainable"},
        "retention_snapshot": {"retention_expires_at": FUTURE_RETENTION},
    }


def test_valid_worker_input_materializes_selected_evidence_and_result(tmp_path: Path) -> None:
    result = run_improve_worker(_worker_input(), output_dir=tmp_path, now=1.0)

    assert result["schema_version"] == WORKER_RESULT_CONTRACT_VERSION
    assert result["ok"] is True
    assert result["runner"] == "tether_dry_run"
    assert result["artifact_uri"].startswith("file://")
    assert len(result["artifact_sha256"]) == 64
    assert set(result["artifact_sha256"]) <= set(string.hexdigits.lower())
    assert result["metrics"]["p99_ms"] == 80.0
    assert result["metrics"]["peak_memory_mb"] == 900.0
    assert result["metrics"]["training_steps"] == 25.0
    assert "modal run scripts/real_improve_worker_modal.py" in (
        result["metadata"]["modal_gpu_follow_up_command"]
    )

    manifest_path = tmp_path / "imp_job_1" / "selected_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_failure_ids"] == ["fail_1"]
    assert [row["failure_id"] for row in manifest["evidence"]] == ["fail_1"]
    assert manifest["evidence"][0]["bounded_refs"] == [
        {
            "kind": "action_chunks",
            "sha256": "b" * 64,
            "uri": "s3://bucket/failures/fail_1/chunks.jsonl",
        }
    ]


def test_selected_id_enforcement_rejects_unselected_evidence(tmp_path: Path) -> None:
    payload = _worker_input(failure_evidence=[_failure("fail_1"), _failure("fail_2")])

    result = run_improve_worker(payload, output_dir=tmp_path, now=1.0)

    assert result["ok"] is False
    assert result["error_code"] == "selected_id_mismatch"
    assert "unselected failure id fail_2" in result["error"]
    assert "artifact_uri" not in result
    assert not (tmp_path / "imp_job_1" / "selected_evidence_manifest.json").exists()


def test_insufficient_evidence_returns_typed_failure_envelope(tmp_path: Path) -> None:
    payload = _worker_input(failure_evidence=[_failure("fail_1", with_evidence=False)])

    result = run_improve_worker(payload, output_dir=tmp_path, now=1.0)

    assert result["ok"] is False
    assert result["error_code"] == "insufficient_evidence"
    assert "no bounded evidence URI/hash pairs" in result["error"]
    assert "artifact_sha256" not in result


def test_success_and_requested_failure_are_deterministic(tmp_path: Path) -> None:
    payload = _worker_input(
        runner_config={
            "p99_ms": 44.0,
            "peak_memory_mb": 512.0,
            "mmd_score": 0.01,
            "training_steps": 3,
        }
    )
    first = run_improve_worker(payload, output_dir=tmp_path, now=1.0)
    second = run_improve_worker(payload, output_dir=tmp_path, now=1.0)

    assert first["ok"] is True
    assert second["ok"] is True
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert first["worker_result_hash"] == second["worker_result_hash"]
    assert first["metrics"] == second["metrics"]

    failure_payload = _worker_input(runner_config={"failure_mode": "training_timeout"})
    failure_a = run_improve_worker(failure_payload, output_dir=tmp_path / "a", now=1.0)
    failure_b = run_improve_worker(failure_payload, output_dir=tmp_path / "b", now=1.0)
    assert failure_a["ok"] is False
    assert failure_a == failure_b
    assert failure_a["metadata"]["failure_mode"] == "training_timeout"


def test_malformed_input_returns_failure_envelope_not_traceback() -> None:
    payload = _worker_input()
    payload["job"] = {}

    result = run_improve_worker(payload)

    assert result["ok"] is False
    assert result["error_code"] == "invalid_worker_input"
    assert "job.id is required" in result["error"]
    assert "artifact_uri" not in result


def test_module_cli_writes_worker_result(tmp_path: Path) -> None:
    input_path = tmp_path / "worker_input.json"
    output_path = tmp_path / "worker_result.json"
    input_path.write_text(json.dumps(_worker_input()), encoding="utf-8")

    completed = _run_module_cli(
        tmp_path,
        "--worker-input",
        str(input_path),
        "--output-dir",
        str(tmp_path / "runs"),
        "--result-output",
        str(output_path),
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["outputs"]["manifest_uri"].startswith("file://")
    assert MODAL_GPU_FOLLOW_UP_COMMAND.split(" --worker-input")[0] in (
        result["metadata"]["modal_gpu_follow_up_command"]
    )


def test_module_cli_malformed_input_prints_failure_without_traceback(tmp_path: Path) -> None:
    input_path = tmp_path / "bad_worker_input.json"
    input_path.write_text("{}", encoding="utf-8")

    completed = _run_module_cli(tmp_path, "--worker-input", str(input_path))

    assert completed.returncode == 0
    assert "Traceback" not in completed.stderr
    result = json.loads(completed.stdout)
    assert result["ok"] is False
    assert result["error_code"] == "invalid_worker_input"


def _run_module_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(repo_root / "src"), env.get("PYTHONPATH", ""))
        if part
    )
    return subprocess.run(
        [sys.executable, "-m", "tether.finetune.improve_worker", *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
