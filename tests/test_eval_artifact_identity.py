import hashlib
import json
from pathlib import Path

import pytest

from tether.eval.artifact_identity import (
    ArtifactIdentityError,
    artifact_digest,
    build_artifact_identity,
    collect_artifacts,
)


def write(root: Path, relative: str, data: bytes):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_receipt_matches_studio_training_import_digest(tmp_path):
    write(tmp_path, "adapter_config.json", b"{}")
    write(tmp_path, "adapter_model.safetensors", b"weights")
    write(tmp_path, "nested/policy_postprocessor.json", b'{"x":1}')
    rows = collect_artifacts(tmp_path)
    expected = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    receipt = build_artifact_identity(
        tmp_path,
        source="/remote/candidate",
        kind="smolvla-lora",
    )
    assert receipt["artifact_sha256"] == expected == artifact_digest(rows)
    assert receipt["file_count"] == 3
    assert receipt["size_bytes"] == sum(row["size_bytes"] for row in rows)
    assert [row["path"] for row in rows] == sorted(row["path"] for row in rows)


def test_artifact_mutation_changes_identity(tmp_path):
    target = write(tmp_path, "adapter_model.safetensors", b"one")
    first = build_artifact_identity(tmp_path, source="candidate")
    target.write_bytes(b"two")
    second = build_artifact_identity(tmp_path, source="candidate")
    assert first["artifact_sha256"] != second["artifact_sha256"]


def test_symlinks_are_excluded_to_match_studio_candidate_hash(tmp_path):
    real = write(tmp_path, "real.bin", b"data")
    link = tmp_path / "link.bin"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks unavailable")
    rows = collect_artifacts(tmp_path)
    assert [row["path"] for row in rows] == ["real.bin"]


def test_empty_or_missing_artifact_root_fails_closed(tmp_path):
    with pytest.raises(ArtifactIdentityError, match="no regular"):
        collect_artifacts(tmp_path)
    with pytest.raises(ArtifactIdentityError, match="not a directory"):
        collect_artifacts(tmp_path / "missing")


def test_receipt_preserves_exact_source_revision_and_hash_contract(tmp_path):
    write(tmp_path, "config.json", b"{}")
    receipt = build_artifact_identity(
        tmp_path,
        source="lerobot/smolvla_base",
        revision="abc123",
        kind="parent-checkpoint",
    )
    assert receipt["source"] == "lerobot/smolvla_base"
    assert receipt["revision"] == "abc123"
    assert receipt["hash_contract"] == "studio-training-artifacts-v1"
    assert receipt["symlink_policy"] == "excluded"
