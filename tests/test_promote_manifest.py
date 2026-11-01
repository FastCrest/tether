"""Promotion must bind the files it consumes when hashes are required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tether.deploy_proof import write_deploy_proof_packet
from tether.promote import decide_promotion


def _policy_report(verdict: str = "pass") -> dict:
    return {
        "kind": "tether.policy_diff",
        "summary": {"verdict": verdict, "compared": 1},
    }


def _packet(tmp_path: Path, *, policy: str | None = None, passed: bool = True) -> Path:
    receipt = {
        "schema_version": 1,
        "kind": "tether.deployment_proof",
        "passed": passed,
        "checks": [{"name": "synthetic_fixture", "status": "pass"}],
    }
    if policy is not None:
        receipt["policy_diff"] = {"enabled": True, "report": _policy_report(policy)}
    packet = tmp_path / "proof"
    write_deploy_proof_packet(receipt, packet)
    return packet


def _read_manifest(packet: Path) -> dict:
    return json.loads((packet / "MANIFEST.json").read_text(encoding="utf-8"))


def _write_manifest(packet: Path, manifest: dict) -> None:
    (packet / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def _omit(packet: Path, name: str) -> None:
    manifest = _read_manifest(packet)
    manifest["files"] = [item for item in manifest["files"] if item["name"] != name]
    _write_manifest(packet, manifest)


def _assert_integrity_failure(packet: Path, *, active: bool = False) -> None:
    report = decide_promotion(packet, candidate_active=active)
    assert report["decision"] == ("ROLLBACK" if active else "BLOCK")
    assert "packet_manifest_hashes" in report["summary"]["failed_checks"]


@pytest.mark.parametrize("variant", ["missing", "null", "empty"])
def test_required_manifest_cannot_omit_file_list(tmp_path: Path, variant: str) -> None:
    packet = _packet(tmp_path)
    manifest = _read_manifest(packet)
    if variant == "missing":
        del manifest["files"]
    else:
        manifest["files"] = None if variant == "null" else []
    _write_manifest(packet, manifest)
    _assert_integrity_failure(packet)


@pytest.mark.parametrize("variant", ["missing", "null", "empty"])
def test_required_hash_cannot_be_absent(tmp_path: Path, variant: str) -> None:
    packet = _packet(tmp_path)
    manifest = _read_manifest(packet)
    for item in manifest["files"]:
        if item["name"] == "deployment-proof.json":
            if variant == "missing":
                del item["sha256"]
            else:
                item["sha256"] = None if variant == "null" else ""
    _write_manifest(packet, manifest)
    _assert_integrity_failure(packet)


@pytest.mark.parametrize("name", ["deployment-proof.json", "policy-diff.json"])
def test_consumed_artifact_must_be_listed(tmp_path: Path, name: str) -> None:
    packet = _packet(tmp_path, policy="pass")
    _omit(packet, name)
    _assert_integrity_failure(packet)


def test_unlisted_sidecar_cannot_override_embedded_failure(tmp_path: Path) -> None:
    packet = _packet(tmp_path, policy="fail")
    assert decide_promotion(packet)["decision"] == "BLOCK"
    _omit(packet, "policy-diff.json")
    (packet / "policy-diff.json").write_text(json.dumps(_policy_report("pass")), encoding="utf-8")
    _assert_integrity_failure(packet)


def test_edited_proof_without_hash_or_size_is_rejected(tmp_path: Path) -> None:
    packet = _packet(tmp_path, passed=False)
    assert decide_promotion(packet)["decision"] == "BLOCK"
    path = packet / "deployment-proof.json"
    proof = json.loads(path.read_text(encoding="utf-8"))
    proof["passed"] = True
    path.write_text(json.dumps(proof), encoding="utf-8")
    manifest = _read_manifest(packet)
    for item in manifest["files"]:
        if item["name"] == path.name:
            del item["sha256"]
            del item["size_bytes"]
    _write_manifest(packet, manifest)
    _assert_integrity_failure(packet)


def test_missing_binding_rolls_back_active_candidate(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    _omit(packet, "deployment-proof.json")
    _assert_integrity_failure(packet, active=True)


@pytest.mark.parametrize("mode", ["none", "sidecar", "embedded"])
def test_valid_packet_still_promotes(tmp_path: Path, mode: str) -> None:
    packet = _packet(tmp_path, policy=None if mode == "none" else "pass")
    if mode == "embedded":
        (packet / "policy-diff.json").unlink()
        _omit(packet, "policy-diff.json")
    report = decide_promotion(packet)
    assert report["decision"] == "PROMOTE"
    assert report["summary"]["fail"] == 0


@pytest.mark.parametrize("variant", ["missing_hash", "empty_list"])
def test_explicit_optional_hash_profile_is_preserved(tmp_path: Path, variant: str) -> None:
    packet = _packet(tmp_path)
    manifest = _read_manifest(packet)
    if variant == "empty_list":
        manifest["files"] = []
    else:
        for item in manifest["files"]:
            del item["sha256"]
    _write_manifest(packet, manifest)
    profile = tmp_path / "optional.json"
    profile.write_text(
        json.dumps({"thresholds": {"require_manifest_hashes": False}}),
        encoding="utf-8",
    )
    report = decide_promotion(packet, profile_path=profile)
    assert report["decision"] == "PROMOTE"
    assert report["summary"]["fail"] == 0


def test_unconsumed_operator_note_need_not_be_listed(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    (packet / "operator-notes.txt").write_text("Review pending.\n", encoding="utf-8")
    assert decide_promotion(packet)["decision"] == "PROMOTE"


def test_wrong_digest_still_blocks(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    manifest = _read_manifest(packet)
    manifest["files"][0]["sha256"] = "0" * 64
    _write_manifest(packet, manifest)
    _assert_integrity_failure(packet)


def test_missing_manifest_still_blocks(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    (packet / "MANIFEST.json").unlink()
    report = decide_promotion(packet)
    assert report["decision"] == "BLOCK"
    assert "packet_manifest_present" in report["summary"]["failed_checks"]
