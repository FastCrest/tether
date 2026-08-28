from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tether.receipt_provenance import (
    EXPORT_FILES,
    WORKFLOW_PATH,
    ReceiptVerificationError,
    build_receipt_namespace,
    build_receipt_manifest,
    download_verified_receipts,
    hash_export_set,
    receipt_artifact_names,
    receipt_mode_binding,
    receipt_namespace_binding,
    receipt_run_dir,
    select_trusted_run,
    verify_receipt_manifest,
)
from receipt_test_support import require_receipt


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
REPOSITORY = "FastCrest/tether"
SOURCE_SHA = "1" * 40
MODEL_ID = "lerobot/pi05_libero_finetuned_v044"
MODEL_REVISION = "a" * 40
MODEL_DIGEST = hashlib.sha256(f"huggingface:{MODEL_ID}@{MODEL_REVISION}".encode()).hexdigest()
ARTIFACT_DIGEST = "3" * 64
RECEIPT_NAMESPACE = build_receipt_namespace(REPOSITORY, SOURCE_SHA, 101, 1)
RECEIPT_NAMESPACE_BINDING = receipt_namespace_binding(
    receipt_namespace=RECEIPT_NAMESPACE,
    repository=REPOSITORY,
    source_sha=SOURCE_SHA,
    workflow_run_id=101,
    workflow_run_attempt=1,
)
PAYLOAD_ARTIFACT_NAME, MANIFEST_ARTIFACT_NAME = receipt_artifact_names(RECEIPT_NAMESPACE)


def _export_set(seed: int) -> dict:
    names = ("tether_config.json", "vlm_prefix.onnx", "expert_denoise.onnx")
    return {
        name: {
            "path": name,
            "sha256": format((seed + index) % 16, "x") * 64,
            "size": 100 + index,
        }
        for index, name in enumerate(names)
    }


def _write_payload(
    path: Path,
    namespace_binding: dict[str, object] = RECEIPT_NAMESPACE_BINDING,
) -> None:
    path.mkdir()
    n10_baked = _export_set(4)
    n10_per_step = _export_set(8)
    parity = {
        "receipt_namespace": namespace_binding,
        "cells": {
            "pi05_teacher_n10": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_digest": MODEL_DIGEST,
                "exports": {
                    "baked": n10_baked,
                    "per_step": n10_per_step,
                },
                "cos": 0.999999,
                "used_provider_baked": "CUDAExecutionProvider",
                "used_provider_per_step": "CUDAExecutionProvider",
                "passes_overall": True,
            },
            "pi05_teacher_n1": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_digest": MODEL_DIGEST,
                "exports": {
                    "baked": _export_set(12),
                    "per_step": _export_set(16),
                },
                "cos": 1.0,
                "used_provider_baked": "CUDAExecutionProvider",
                "used_provider_per_step": "CUDAExecutionProvider",
                "passes_overall": True,
            },
        },
        "thresholds": {"cos_min": 0.99999, "max_abs_max": 1e-5},
    }
    overhead = {
        "receipt_namespace": namespace_binding,
        "iobinding_gate": {"passes_overall": True},
        "providers": {
            "baked": "CUDAExecutionProvider",
            "per_step": "CUDAExecutionProvider",
        },
        "exports": {
            "baked": n10_baked,
            "per_step": n10_per_step,
        },
        "thresholds": {"median_overhead_pct_max": 0.2, "p99_ratio_max": 1.3},
    }
    e2e = {
        "receipt_namespace": namespace_binding,
        "baked": {
            "receipt_namespace": namespace_binding,
            "export": n10_baked,
            "providers": {
                "vlm_prefix": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "expert_denoise": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            },
        },
        "per_step": {
            "receipt_namespace": namespace_binding,
            "export": n10_per_step,
            "providers": {
                "vlm_prefix": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "expert_denoise": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            },
        },
        "passes_overall": True,
        "thresholds": {"median_overhead_pct_max": 0.2, "p99_ratio_max": 1.3},
    }
    values = {
        "per_step_parity_last_run.json": parity,
        "per_step_overhead_last_run.json": overhead,
        "per_step_e2e_latency_last_run.json": e2e,
    }
    for name, value in values.items():
        (path / name).write_text(json.dumps(value, sort_keys=True))


def test_hash_export_set_binds_config_prefix_and_expert(tmp_path):
    for index, name in enumerate(EXPORT_FILES, start=1):
        (tmp_path / name).write_bytes(bytes([index]) * index)

    result = hash_export_set(tmp_path)

    assert set(result) == set(EXPORT_FILES)
    for name in EXPORT_FILES:
        assert result[name]["path"] == name
        assert result[name]["size"] == (tmp_path / name).stat().st_size
        assert len(str(result[name]["sha256"])) == 64


def test_immutable_namespaces_separate_runs_and_attempts(tmp_path):
    run_101 = build_receipt_namespace(REPOSITORY, SOURCE_SHA, 101, 1)
    run_102 = build_receipt_namespace(REPOSITORY, SOURCE_SHA, 102, 1)
    attempt_2 = build_receipt_namespace(REPOSITORY, SOURCE_SHA, 101, 2)

    assert len({run_101, run_102, attempt_2}) == 3
    assert receipt_run_dir(tmp_path, run_101) != receipt_run_dir(tmp_path, run_102)
    assert receipt_run_dir(tmp_path, run_101) != receipt_run_dir(tmp_path, attempt_2)


@pytest.mark.parametrize(
    "unsafe_namespace",
    ["../other-run", "/tmp/receipt", "gh-invalid", RECEIPT_NAMESPACE + "/x"],
)
def test_receipt_run_dir_rejects_unsafe_or_noncanonical_namespaces(tmp_path, unsafe_namespace):
    with pytest.raises(ReceiptVerificationError, match="namespace"):
        receipt_run_dir(tmp_path, unsafe_namespace)


def test_receipt_mode_has_explicit_legacy_mode_and_rejects_partial_binding():
    assert receipt_mode_binding() is None
    with pytest.raises(ReceiptVerificationError, match="receipt mode requires"):
        receipt_mode_binding(receipt_namespace=RECEIPT_NAMESPACE)


def test_export_measurement_hashes_are_confined_to_each_namespace(tmp_path):
    other_namespace = build_receipt_namespace(REPOSITORY, SOURCE_SHA, 102, 1)
    hashes = []
    directories = []
    for namespace, byte in (
        (RECEIPT_NAMESPACE, b"a"),
        (other_namespace, b"b"),
    ):
        export_dir = receipt_run_dir(tmp_path, namespace) / "export"
        export_dir.mkdir(parents=True)
        for name in EXPORT_FILES:
            (export_dir / name).write_bytes(byte + name.encode())
        directories.append(export_dir)
        hashes.append(hash_export_set(export_dir))

    assert directories[0] != directories[1]
    assert hashes[0] != hashes[1]


def test_build_rejects_payload_from_a_different_run_namespace(tmp_path):
    payload_dir = tmp_path / "payload"
    _write_payload(payload_dir)
    other_namespace = build_receipt_namespace(REPOSITORY, SOURCE_SHA, 102, 1)
    parity_path = payload_dir / "per_step_parity_last_run.json"
    parity = json.loads(parity_path.read_text())
    parity["receipt_namespace"] = receipt_namespace_binding(
        receipt_namespace=other_namespace,
        repository=REPOSITORY,
        source_sha=SOURCE_SHA,
        workflow_run_id=102,
        workflow_run_attempt=1,
    )
    parity_path.write_text(json.dumps(parity))

    with pytest.raises(ReceiptVerificationError, match="receipt_namespace"):
        build_receipt_manifest(
            payload_dir=payload_dir,
            receipt_namespace=RECEIPT_NAMESPACE,
            repository=REPOSITORY,
            source_sha=SOURCE_SHA,
            source_ref="refs/heads/main",
            workflow_run_id=101,
            workflow_run_attempt=1,
            event="schedule",
            payload_artifact_id=202,
            payload_artifact_name=PAYLOAD_ARTIFACT_NAME,
            payload_artifact_digest=ARTIFACT_DIGEST,
            model_id=MODEL_ID,
            model_digest=MODEL_DIGEST,
            generated_at=NOW,
        )


@pytest.mark.parametrize("mutation", ["missing", "partial", "forged", "cross_run"])
def test_build_rejects_invalid_nested_e2e_namespace(tmp_path, mutation):
    payload_dir = tmp_path / mutation
    _write_payload(payload_dir)
    e2e_path = payload_dir / "per_step_e2e_latency_last_run.json"
    e2e = json.loads(e2e_path.read_text())
    if mutation == "missing":
        del e2e["baked"]["receipt_namespace"]
    elif mutation == "partial":
        e2e["baked"]["receipt_namespace"] = {"value": RECEIPT_NAMESPACE}
    elif mutation == "forged":
        e2e["baked"]["receipt_namespace"]["workflow_run_id"] = 999
    else:
        other_namespace = build_receipt_namespace(REPOSITORY, SOURCE_SHA, 102, 1)
        e2e["baked"]["receipt_namespace"] = receipt_namespace_binding(
            receipt_namespace=other_namespace,
            repository=REPOSITORY,
            source_sha=SOURCE_SHA,
            workflow_run_id=102,
            workflow_run_attempt=1,
        )
    e2e_path.write_text(json.dumps(e2e))

    with pytest.raises(ReceiptVerificationError, match=r"e2e baked\.receipt_namespace"):
        build_receipt_manifest(
            payload_dir=payload_dir,
            receipt_namespace=RECEIPT_NAMESPACE,
            repository=REPOSITORY,
            source_sha=SOURCE_SHA,
            source_ref="refs/heads/main",
            workflow_run_id=101,
            workflow_run_attempt=1,
            event="schedule",
            payload_artifact_id=202,
            payload_artifact_name=PAYLOAD_ARTIFACT_NAME,
            payload_artifact_digest=ARTIFACT_DIGEST,
            model_id=MODEL_ID,
            model_digest=MODEL_DIGEST,
            generated_at=NOW,
        )


def test_nested_e2e_namespace_is_part_of_export_identity(tmp_path):
    first_dir = tmp_path / "first"
    _write_payload(first_dir)
    first = build_receipt_manifest(
        payload_dir=first_dir,
        receipt_namespace=RECEIPT_NAMESPACE,
        repository=REPOSITORY,
        source_sha=SOURCE_SHA,
        source_ref="refs/heads/main",
        workflow_run_id=101,
        workflow_run_attempt=1,
        event="schedule",
        payload_artifact_id=202,
        payload_artifact_name=PAYLOAD_ARTIFACT_NAME,
        payload_artifact_digest=ARTIFACT_DIGEST,
        model_id=MODEL_ID,
        model_digest=MODEL_DIGEST,
        generated_at=NOW,
    )

    second_namespace = build_receipt_namespace(REPOSITORY, SOURCE_SHA, 101, 2)
    second_binding = receipt_namespace_binding(
        receipt_namespace=second_namespace,
        repository=REPOSITORY,
        source_sha=SOURCE_SHA,
        workflow_run_id=101,
        workflow_run_attempt=2,
    )
    second_payload_name, _ = receipt_artifact_names(second_namespace)
    second_dir = tmp_path / "second"
    _write_payload(second_dir, second_binding)
    second = build_receipt_manifest(
        payload_dir=second_dir,
        receipt_namespace=second_namespace,
        repository=REPOSITORY,
        source_sha=SOURCE_SHA,
        source_ref="refs/heads/main",
        workflow_run_id=101,
        workflow_run_attempt=2,
        event="schedule",
        payload_artifact_id=203,
        payload_artifact_name=second_payload_name,
        payload_artifact_digest=ARTIFACT_DIGEST,
        model_id=MODEL_ID,
        model_digest=MODEL_DIGEST,
        generated_at=NOW,
    )

    assert first["export"]["digest"] != second["export"]["digest"]


def test_build_rejects_payload_artifact_name_outside_namespace(tmp_path):
    payload_dir = tmp_path / "payload"
    _write_payload(payload_dir)
    with pytest.raises(ReceiptVerificationError, match="artifact name"):
        build_receipt_manifest(
            payload_dir=payload_dir,
            receipt_namespace=RECEIPT_NAMESPACE,
            repository=REPOSITORY,
            source_sha=SOURCE_SHA,
            source_ref="refs/heads/main",
            workflow_run_id=101,
            workflow_run_attempt=1,
            event="schedule",
            payload_artifact_id=202,
            payload_artifact_name="parity-receipts-payload-forged",
            payload_artifact_digest=ARTIFACT_DIGEST,
            model_id=MODEL_ID,
            model_digest=MODEL_DIGEST,
            generated_at=NOW,
        )


def test_workflow_protects_paid_producer_environment_and_digest_source():
    workflow = (Path(__file__).parent.parent / ".github/workflows/parity-receipts.yml").read_text()
    assert "environment: parity-receipts-production" in workflow
    assert "PARITY_RECEIPTS_ENVIRONMENT_PROTECTED" in workflow
    assert "DISPATCH_MODEL_DIGEST" not in workflow
    assert workflow.count("--receipt-namespace") == 4
    assert "receipt_runs/$RECEIPT_NAMESPACE" in workflow
    assert "model_digest:" not in workflow.split("workflow_dispatch:", 1)[1].split("push:", 1)[0]


def _trusted_run(**overrides) -> dict:
    run = {
        "id": 101,
        "path": f"{WORKFLOW_PATH}@refs/heads/main",
        "head_sha": SOURCE_SHA,
        "head_branch": "main",
        "head_repository": {"full_name": REPOSITORY},
        "event": "schedule",
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        "created_at": "2026-08-28T11:00:00Z",
    }
    run.update(overrides)
    return run


@pytest.fixture
def receipt_case(tmp_path):
    payload_dir = tmp_path / "payload"
    _write_payload(payload_dir)
    manifest = build_receipt_manifest(
        payload_dir=payload_dir,
        receipt_namespace=RECEIPT_NAMESPACE,
        repository=REPOSITORY,
        source_sha=SOURCE_SHA,
        source_ref="refs/heads/main",
        workflow_run_id=101,
        workflow_run_attempt=1,
        event="schedule",
        payload_artifact_id=202,
        payload_artifact_name=PAYLOAD_ARTIFACT_NAME,
        payload_artifact_digest=ARTIFACT_DIGEST,
        model_id=MODEL_ID,
        model_digest=MODEL_DIGEST,
        generated_at=NOW - timedelta(hours=1),
    )
    artifact = {
        "id": 202,
        "name": PAYLOAD_ARTIFACT_NAME,
        "digest": f"sha256:{ARTIFACT_DIGEST}",
        "expired": False,
    }
    return payload_dir, manifest, _trusted_run(), artifact


def _verify(case, **overrides) -> None:
    payload_dir, manifest, run, artifact = case
    arguments = {
        "manifest": manifest,
        "payload_dir": payload_dir,
        "trusted_run": run,
        "trusted_artifact": artifact,
        "expected_repository": REPOSITORY,
        "expected_sha": SOURCE_SHA,
        "expected_model_id": MODEL_ID,
        "expected_model_digest": MODEL_DIGEST,
        "expected_export_id": manifest["export"]["id"],
        "expected_export_digest": manifest["export"]["digest"],
        "now": NOW,
    }
    arguments.update(overrides)
    verify_receipt_manifest(**arguments)


def test_valid_receipt_binds_run_artifact_and_payload(receipt_case):
    _verify(receipt_case)


def test_rejects_self_asserted_forged_run_id(receipt_case):
    _, manifest, _, _ = receipt_case
    forged = copy.deepcopy(manifest)
    forged["producer"]["workflow_run_id"] = 999999
    with pytest.raises(ReceiptVerificationError, match="forged workflow run id"):
        _verify(receipt_case, manifest=forged)


def test_rejects_forged_namespace_or_run_attempt(receipt_case):
    _, manifest, _, _ = receipt_case
    forged = copy.deepcopy(manifest)
    forged["receipt_namespace"]["workflow_run_attempt"] = 2
    with pytest.raises(ReceiptVerificationError, match="namespace"):
        _verify(receipt_case, manifest=forged)

    with pytest.raises(ReceiptVerificationError, match="run attempt"):
        _verify(receipt_case, trusted_run=_trusted_run(run_attempt=2))


def test_rejects_pull_request_run(receipt_case):
    with pytest.raises(ReceiptVerificationError, match="event is not allowlisted"):
        _verify(receipt_case, trusted_run=_trusted_run(event="pull_request"))


@pytest.mark.parametrize(
    ("run_override", "message"),
    [
        ({"head_sha": "9" * 40}, "head_sha"),
        ({"head_branch": "feature/untrusted"}, "ref is not allowlisted"),
        ({"path": ".github/workflows/pytest.yml@refs/heads/main"}, "workflow run path"),
        ({"head_repository": {"full_name": "attacker/tether"}}, "untrusted repository"),
    ],
)
def test_rejects_wrong_sha_ref_workflow_or_repository(receipt_case, run_override, message):
    with pytest.raises(ReceiptVerificationError, match=message):
        _verify(receipt_case, trusted_run=_trusted_run(**run_override))


def test_rejects_expired_receipt(receipt_case):
    _, manifest, _, _ = receipt_case
    expired = copy.deepcopy(manifest)
    expired["expires_at"] = "2026-08-28T11:59:59Z"
    with pytest.raises(ReceiptVerificationError, match="expired"):
        _verify(receipt_case, manifest=expired)


def test_rejects_altered_payload(receipt_case):
    payload_dir, _, _, _ = receipt_case
    path = payload_dir / "per_step_overhead_last_run.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ReceiptVerificationError, match="size changed"):
        _verify(receipt_case)


def test_rejects_unhashed_extra_payload_file(receipt_case):
    payload_dir, _, _, _ = receipt_case
    (payload_dir / "untrusted.json").write_text("{}")
    with pytest.raises(ReceiptVerificationError, match="unexpected files"):
        _verify(receipt_case)


def test_rejects_model_digest_mismatch(receipt_case):
    with pytest.raises(ReceiptVerificationError, match="model digest"):
        _verify(receipt_case, expected_model_digest="8" * 64)


def test_rejects_export_digest_mismatch(receipt_case):
    with pytest.raises(ReceiptVerificationError, match="export digest"):
        _verify(receipt_case, expected_export_digest="8" * 64)


def test_build_rejects_measurements_from_a_different_export(tmp_path):
    payload_dir = tmp_path / "payload"
    _write_payload(payload_dir)
    e2e_path = payload_dir / "per_step_e2e_latency_last_run.json"
    e2e = json.loads(e2e_path.read_text())
    e2e["per_step"]["export"]["expert_denoise.onnx"]["sha256"] = "f" * 64
    e2e_path.write_text(json.dumps(e2e))
    with pytest.raises(ReceiptVerificationError, match="e2e per_step export"):
        build_receipt_manifest(
            payload_dir=payload_dir,
            receipt_namespace=RECEIPT_NAMESPACE,
            repository=REPOSITORY,
            source_sha=SOURCE_SHA,
            source_ref="refs/heads/main",
            workflow_run_id=101,
            workflow_run_attempt=1,
            event="schedule",
            payload_artifact_id=202,
            payload_artifact_name=PAYLOAD_ARTIFACT_NAME,
            payload_artifact_digest=ARTIFACT_DIGEST,
            model_id=MODEL_ID,
            model_digest=MODEL_DIGEST,
            generated_at=NOW,
        )


def test_build_rejects_incomplete_or_extra_parity_cell_set(tmp_path):
    for mutation in ("missing", "extra"):
        payload_dir = tmp_path / mutation
        _write_payload(payload_dir)
        parity_path = payload_dir / "per_step_parity_last_run.json"
        parity = json.loads(parity_path.read_text())
        if mutation == "missing":
            del parity["cells"]["pi05_teacher_n1"]
        else:
            parity["cells"]["unexpected"] = copy.deepcopy(parity["cells"]["pi05_teacher_n1"])
        parity_path.write_text(json.dumps(parity))
        with pytest.raises(ReceiptVerificationError, match="cells must be exactly"):
            build_receipt_manifest(
                payload_dir=payload_dir,
                receipt_namespace=RECEIPT_NAMESPACE,
                repository=REPOSITORY,
                source_sha=SOURCE_SHA,
                source_ref="refs/heads/main",
                workflow_run_id=101,
                workflow_run_attempt=1,
                event="schedule",
                payload_artifact_id=202,
                payload_artifact_name=PAYLOAD_ARTIFACT_NAME,
                payload_artifact_digest=ARTIFACT_DIGEST,
                model_id=MODEL_ID,
                model_digest=MODEL_DIGEST,
                generated_at=NOW,
            )


def test_build_rejects_incomplete_export_set_and_cpu_e2e_provider(tmp_path):
    incomplete_dir = tmp_path / "incomplete"
    _write_payload(incomplete_dir)
    parity_path = incomplete_dir / "per_step_parity_last_run.json"
    parity = json.loads(parity_path.read_text())
    del parity["cells"]["pi05_teacher_n10"]["exports"]["baked"]["tether_config.json"]
    parity_path.write_text(json.dumps(parity))
    with pytest.raises(ReceiptVerificationError, match="must contain exactly"):
        build_receipt_manifest(
            payload_dir=incomplete_dir,
            receipt_namespace=RECEIPT_NAMESPACE,
            repository=REPOSITORY,
            source_sha=SOURCE_SHA,
            source_ref="refs/heads/main",
            workflow_run_id=101,
            workflow_run_attempt=1,
            event="schedule",
            payload_artifact_id=202,
            payload_artifact_name=PAYLOAD_ARTIFACT_NAME,
            payload_artifact_digest=ARTIFACT_DIGEST,
            model_id=MODEL_ID,
            model_digest=MODEL_DIGEST,
            generated_at=NOW,
        )

    cpu_dir = tmp_path / "cpu"
    _write_payload(cpu_dir)
    e2e_path = cpu_dir / "per_step_e2e_latency_last_run.json"
    e2e = json.loads(e2e_path.read_text())
    e2e["baked"]["providers"]["expert_denoise"] = ["CPUExecutionProvider"]
    e2e_path.write_text(json.dumps(e2e))
    with pytest.raises(ReceiptVerificationError, match="expert_denoise did not use"):
        build_receipt_manifest(
            payload_dir=cpu_dir,
            receipt_namespace=RECEIPT_NAMESPACE,
            repository=REPOSITORY,
            source_sha=SOURCE_SHA,
            source_ref="refs/heads/main",
            workflow_run_id=101,
            workflow_run_attempt=1,
            event="schedule",
            payload_artifact_id=202,
            payload_artifact_name=PAYLOAD_ARTIFACT_NAME,
            payload_artifact_digest=ARTIFACT_DIGEST,
            model_id=MODEL_ID,
            model_digest=MODEL_DIGEST,
            generated_at=NOW,
        )


def test_rejects_artifact_id_or_digest_mismatch(receipt_case):
    _, _, _, artifact = receipt_case
    wrong_id = {**artifact, "id": 303}
    with pytest.raises(ReceiptVerificationError, match="artifact id"):
        _verify(receipt_case, trusted_artifact=wrong_id)
    wrong_digest = {**artifact, "digest": f"sha256:{'8' * 64}"}
    with pytest.raises(ReceiptVerificationError, match="artifact digest"):
        _verify(receipt_case, trusted_artifact=wrong_digest)


def test_select_trusted_run_ignores_untrusted_candidates():
    forged = _trusted_run(id=999, event="pull_request", created_at="2026-08-28T12:00:00Z")
    trusted = _trusted_run(id=101)
    assert (
        select_trusted_run(
            [forged, trusted], expected_repository=REPOSITORY, expected_sha=SOURCE_SHA
        )["id"]
        == 101
    )


def test_release_branch_fails_loud_when_receipt_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_REF_NAME", "release/v1.0")
    with pytest.raises(pytest.fail.Exception, match="No trusted receipt"):
        require_receipt(tmp_path / "missing.json", "the producer")


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, contents in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 28, 12, 0, 0))
            archive.writestr(info, contents)
    return output.getvalue()


def test_download_discovers_exact_run_and_verifies_artifacts(tmp_path):
    selected_namespace = build_receipt_namespace(REPOSITORY, SOURCE_SHA, 101, 2)
    selected_binding = receipt_namespace_binding(
        receipt_namespace=selected_namespace,
        repository=REPOSITORY,
        source_sha=SOURCE_SHA,
        workflow_run_id=101,
        workflow_run_attempt=2,
    )
    selected_payload_name, selected_manifest_name = receipt_artifact_names(selected_namespace)
    payload_dir = tmp_path / "source-payload"
    _write_payload(payload_dir, selected_binding)
    payload_zip = _zip_bytes(
        {
            name: (payload_dir / name).read_bytes()
            for name in sorted(p.name for p in payload_dir.iterdir())
        }
    )
    payload_digest = hashlib.sha256(payload_zip).hexdigest()
    manifest = build_receipt_manifest(
        payload_dir=payload_dir,
        receipt_namespace=selected_namespace,
        repository=REPOSITORY,
        source_sha=SOURCE_SHA,
        source_ref="refs/heads/main",
        workflow_run_id=101,
        workflow_run_attempt=2,
        event="schedule",
        payload_artifact_id=202,
        payload_artifact_name=selected_payload_name,
        payload_artifact_digest=payload_digest,
        model_id=MODEL_ID,
        model_digest=MODEL_DIGEST,
        generated_at=NOW - timedelta(hours=1),
    )
    manifest_zip = _zip_bytes({"receipt-manifest.json": json.dumps(manifest).encode()})
    manifest_digest = hashlib.sha256(manifest_zip).hexdigest()
    payload_artifact = {
        "id": 202,
        "name": selected_payload_name,
        "digest": f"sha256:{payload_digest}",
        "expired": False,
    }
    manifest_artifact = {
        "id": 303,
        "name": selected_manifest_name,
        "digest": f"sha256:{manifest_digest}",
        "expired": False,
    }

    class FakeClient:
        def __init__(self):
            self.calls = []

        def get_json(self, path, query=None):
            self.calls.append(("json", path, query))
            if "/workflows/" in path:
                return {
                    "workflow_runs": [
                        _trusted_run(run_attempt=1),
                        _trusted_run(run_attempt=2),
                    ]
                }
            return {
                "artifacts": [
                    {
                        "id": 102,
                        "name": PAYLOAD_ARTIFACT_NAME,
                        "expired": False,
                    },
                    {
                        "id": 103,
                        "name": MANIFEST_ARTIFACT_NAME,
                        "expired": False,
                    },
                    manifest_artifact,
                    payload_artifact,
                ]
            }

        def get_bytes(self, path, query=None):
            self.calls.append(("bytes", path, query))
            return manifest_zip if path.endswith("/303/zip") else payload_zip

    client = FakeClient()
    destination = tmp_path / "downloaded"
    downloaded = download_verified_receipts(
        repository=REPOSITORY,
        expected_sha=SOURCE_SHA,
        expected_model_id=MODEL_ID,
        expected_model_digest=MODEL_DIGEST,
        expected_export_id=manifest["export"]["id"],
        expected_export_digest=manifest["export"]["digest"],
        destination=destination,
        token="unused-by-injected-client",
        client=client,
        now=NOW,
    )

    assert downloaded["producer"]["workflow_run_id"] == 101
    assert downloaded["producer"]["workflow_run_attempt"] == 2
    assert downloaded["receipt_namespace"]["value"] == selected_namespace
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        [
            "per_step_parity_last_run.json",
            "per_step_overhead_last_run.json",
            "per_step_e2e_latency_last_run.json",
        ]
    )
    run_call = client.calls[0]
    assert run_call[2]["head_sha"] == SOURCE_SHA
    assert run_call[2]["status"] == "success"
    assert any(call[1].endswith("/202/zip") for call in client.calls)
    assert any(call[1].endswith("/303/zip") for call in client.calls)


@pytest.mark.parametrize("case", ["forged", "duplicate"])
def test_download_rejects_forged_or_duplicate_manifest_artifacts(tmp_path, case):
    exact = {
        "id": 303,
        "name": MANIFEST_ARTIFACT_NAME,
        "expired": False,
    }
    if case == "forged":
        artifacts = [{**exact, "name": MANIFEST_ARTIFACT_NAME + "-forged"}]
    else:
        artifacts = [exact, {**exact, "id": 304}]

    class FakeClient:
        def get_json(self, path, query=None):
            if "/workflows/" in path:
                return {"workflow_runs": [_trusted_run()]}
            return {"artifacts": artifacts}

        def get_bytes(self, path, query=None):
            raise AssertionError("ambiguous artifact must not be downloaded")

    with pytest.raises(ReceiptVerificationError, match="unique receipt manifest"):
        download_verified_receipts(
            repository=REPOSITORY,
            expected_sha=SOURCE_SHA,
            expected_model_id=MODEL_ID,
            expected_model_digest=MODEL_DIGEST,
            expected_export_id="pi05-per-step-expert",
            expected_export_digest="0" * 64,
            destination=tmp_path / "downloaded",
            token="unused-by-injected-client",
            client=FakeClient(),
            now=NOW,
        )


@pytest.mark.parametrize("case", ["forged", "duplicate"])
def test_download_rejects_forged_or_duplicate_payload_artifacts(tmp_path, case):
    payload_dir = tmp_path / "source-payload"
    _write_payload(payload_dir)
    payload_zip = _zip_bytes(
        {
            name: (payload_dir / name).read_bytes()
            for name in sorted(path.name for path in payload_dir.iterdir())
        }
    )
    payload_digest = hashlib.sha256(payload_zip).hexdigest()
    manifest = build_receipt_manifest(
        payload_dir=payload_dir,
        receipt_namespace=RECEIPT_NAMESPACE,
        repository=REPOSITORY,
        source_sha=SOURCE_SHA,
        source_ref="refs/heads/main",
        workflow_run_id=101,
        workflow_run_attempt=1,
        event="schedule",
        payload_artifact_id=202,
        payload_artifact_name=PAYLOAD_ARTIFACT_NAME,
        payload_artifact_digest=payload_digest,
        model_id=MODEL_ID,
        model_digest=MODEL_DIGEST,
        generated_at=NOW - timedelta(hours=1),
    )
    if case == "forged":
        manifest["artifact"]["name"] = PAYLOAD_ARTIFACT_NAME + "-forged"
    manifest_zip = _zip_bytes({"receipt-manifest.json": json.dumps(manifest).encode()})
    manifest_artifact = {
        "id": 303,
        "name": MANIFEST_ARTIFACT_NAME,
        "digest": f"sha256:{hashlib.sha256(manifest_zip).hexdigest()}",
        "expired": False,
    }
    payload_artifact = {
        "id": 202,
        "name": PAYLOAD_ARTIFACT_NAME,
        "digest": f"sha256:{payload_digest}",
        "expired": False,
    }
    artifacts = [manifest_artifact, payload_artifact]
    if case == "duplicate":
        artifacts.append(dict(payload_artifact))

    class FakeClient:
        def get_json(self, path, query=None):
            if "/workflows/" in path:
                return {"workflow_runs": [_trusted_run()]}
            return {"artifacts": artifacts}

        def get_bytes(self, path, query=None):
            if path.endswith("/303/zip"):
                return manifest_zip
            return payload_zip

    message = "selected run namespace" if case == "forged" else "absent or ambiguous"
    with pytest.raises(ReceiptVerificationError, match=message):
        download_verified_receipts(
            repository=REPOSITORY,
            expected_sha=SOURCE_SHA,
            expected_model_id=MODEL_ID,
            expected_model_digest=MODEL_DIGEST,
            expected_export_id=manifest["export"]["id"],
            expected_export_digest=manifest["export"]["digest"],
            destination=tmp_path / "downloaded",
            token="unused-by-injected-client",
            client=FakeClient(),
            now=NOW,
        )
