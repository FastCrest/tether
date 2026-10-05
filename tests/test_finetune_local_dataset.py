"""Synthetic admission tests, not a LeRobot writer/loader/training qualification.

Parquet slots contain labeled opaque bytes: the admission layer verifies their
pinned integrity and retained evidence. It deliberately does not decode them.
Base-config fixtures declare their dimensions explicitly; no model is fetched.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tether.finetune.config import FinetuneConfig
from tether.finetune.local_dataset import (
    LocalDatasetError,
    NORMALIZATION_FILES,
    WRITER_FILES,
    WRITER_SOURCE,
    WRITER_WHEEL,
    verify_local_export,
)
from tether.finetune.preflight import run_preflight
from tether.finetune.preflight.dataset_size import check_dataset_size
from tether.finetune.preflight.schema import check_schema
from tether.finetune.run import _build_lerobot_command, _validate_config, run_finetune


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def seal(export, receipt, *, inventory=True):
    if inventory:
        receipt["artifacts"] = [
            {
                "path": p.relative_to(export).as_posix(),
                "size_bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(export.rglob("*"))
            if p.is_file() and p.name != "export-receipt.json"
        ]
    receipt["artifact_manifest_sha256"] = canonical(receipt["artifacts"])
    receipt["sha256"] = canonical({k: v for k, v in receipt.items() if k != "sha256"})
    write_json(export / "export-receipt.json", receipt)
    return receipt["sha256"]


@pytest.fixture
def local_export(tmp_path):
    export = tmp_path / "export"
    root = export / "dataset"
    camera = "observation.images.wrist"
    features = {
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
        camera: {"dtype": "image", "shape": [6, 8, 3], "names": ["height", "width", "channels"]},
        **{
            key: {
                "dtype": "float32" if key == "timestamp" else "int64",
                "shape": [1],
                "names": None,
            }
            for key in ("timestamp", "index", "episode_index", "frame_index", "task_index")
        },
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "synthetic-arm",
        "total_frames": 2,
        "total_episodes": 1,
        "total_tasks": 1,
        "fps": 10,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": features,
    }
    stats = {}
    for key, feature in features.items():
        values = [[[0.0]], [[0.0]], [[0.0]]] if key == camera else [0.0] * feature["shape"][0]
        stats[key] = {
            "count": [2],
            **{name: copy.deepcopy(values) for name in ("min", "max", "mean", "std")},
        }
    write_json(root / "meta/info.json", info)
    write_json(root / "meta/stats.json", stats)
    opaque_paths = [
        "dataset/data/chunk-000/file-000.parquet",
        "dataset/meta/episodes/chunk-000/file-000.parquet",
        "dataset/meta/tasks.parquet",
        "provenance/upstream-statistics/meta/episodes/chunk-000/file-000.parquet",
        "provenance/upstream-statistics/meta/stats.json",
    ]
    for relative in opaque_paths:
        path = export / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC opaque artifact slot; no loader was run.\n")
    identity = {
        "id": "synthetic-corrected-source",
        "sha256": "a" * 64,
        "source_type": "materialized-dataset-edit",
        "format": "tether-recording-dataset-v1",
    }
    case = {
        "task": "synthetic task",
        "seed": 3,
        "episode": 0,
        "initial_state": 1,
        "embodiment": "synthetic-arm",
    }
    split = {
        "schema": 2,
        "identity_fields": ["task", "seed", "episode", "initial_state", "embodiment"],
        "dataset_id": identity["id"],
        "dataset_sha256": identity["sha256"],
        "development": [case],
        "holdout": [{**case, "episode": 1}],
        "test": [],
    }
    split["sha256"] = canonical(split)
    write_json(export / "provenance/source-identity.json", identity)
    write_json(export / "provenance/split.json", split)
    frames = [
        {
            "source_index": index,
            "episode_index": 0,
            "frame_index": index,
            "example": {
                "task": case["task"],
                "embodiment": case["embodiment"],
                "source": {"episode": 0, "seed": 3, "initial_state": 1},
            },
        }
        for index in range(2)
    ]
    rows = [
        {
            "source_index": i,
            "partition": "development",
            "example_sha256": canonical(frame["example"]),
        }
        for i, frame in enumerate(frames)
    ]
    rows.append({"source_index": 2, "partition": "holdout", "example_sha256": "d" * 64})
    write_json(export / "provenance/partition-rows.json", rows)
    (export / "provenance/frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in frames)
    )

    def file_hash(path):
        return hashlib.sha256((export / path).read_bytes()).hexdigest()

    replacement_paths = [
        "dataset/meta/stats.json",
        "dataset/meta/episodes/chunk-000/file-000.parquet",
    ]
    adapter = {
        "policy": "verified-moments-v1",
        "adapter_source_sha256": "11c0a0cc0d5f4cd81de9904ec0d08f0305ba9ecec6e5952c25fe435c857afdf3",
        "statistics_sha256": canonical(stats),
        "source_dataset_sha256": identity["sha256"],
        "split_sha256": split["sha256"],
        "training_partition_sha256": canonical(split["development"]),
        "training_rows_sha256": canonical(rows[:2]),
        "replacement_statistics": [{"path": p, "sha256": file_hash(p)} for p in replacement_paths],
        "original_statistics": [
            {
                "output_path": p,
                "retained_path": "provenance/upstream-statistics/" + p.removeprefix("dataset/"),
                "sha256": file_hash("provenance/upstream-statistics/" + p.removeprefix("dataset/")),
            }
            for p in replacement_paths
        ],
    }
    receipt = {
        "schema": 1,
        "kind": "studio-corrected-lerobot-offline-export",
        "status": "complete",
        "source_dataset_id": identity["id"],
        "source_dataset_sha256": identity["sha256"],
        "source_manifest_sha256": "b" * 64,
        "split_sha256": split["sha256"],
        "source_partition": "development",
        "lerobot_split": "train",
        "partition_row_counts": {"development": 2, "holdout": 1, "test": 0},
        "episodes": [
            {
                "case": {
                    **case,
                    "identity_sources": {
                        "task": "example.task",
                        "embodiment": "example.embodiment",
                        "seed": "source.seed",
                        "episode": "source.episode",
                        "initial_state": "source.initial_state",
                    },
                },
                "partition": "development",
                "indexes": [0, 1],
            }
        ],
        "contract": {
            "state_names": ["x", "y"],
            "action_names": ["a", "b"],
            "state_key": "state",
            "camera_key": camera,
            "fps": 10,
        },
        "writer": {
            "version": "0.5.1",
            "source_commit": WRITER_SOURCE,
            "wheel_sha256": WRITER_WHEEL,
            "source_files": dict(WRITER_FILES),
            "runtime_versions": {"lerobot": "0.5.1"},
        },
        "statistics_policy": "verified-moments-v1",
        "statistics_adapter": adapter,
        "normalization_processor": {
            "status": "passed",
            "profile": "smolvla",
            "epsilon": 1e-8,
            "source_files": dict(NORMALIZATION_FILES),
            "numeric_frames": 2,
            "image_frame_indexes": [0, 1],
            "maximum_image_batch": 1,
            "resolved_feature_modes": {
                "action": "MEAN_STD",
                "observation.state": "MEAN_STD",
                camera: "IDENTITY",
            },
            "image_statistics_used": False,
            "imagenet_override_applied_in_this_check": False,
            "trainer_use_imagenet_stats_default": True,
        },
        "loader_roundtrip": {"status": "passed", "frames_read": 2, "episodes_read": 1},
    }
    digest = seal(export, receipt)
    return root, receipt, digest


@pytest.fixture
def local_cfg(local_export, tmp_path):
    root, _, digest = local_export
    base = tmp_path / "synthetic-base"
    write_json(
        base / "config.json",
        {
            "type": "smolvla",
            "max_action_dim": 4,
            "max_state_dim": 4,
            "output_features": {"action": {"shape": [3]}},
            "normalization_mapping": {
                "VISUAL": "IDENTITY",
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD",
            },
        },
    )
    return FinetuneConfig(
        base=str(base),
        dataset="studio-local/synthetic",
        output=tmp_path / "run",
        dataset_root=root,
        dataset_manifest_sha256=digest,
        dry_run=True,
    )


def test_valid_synthetic_admission(local_export):
    root, receipt, digest = local_export
    result = verify_local_export(root, digest)
    assert result == {
        "receipt": receipt,
        "info": json.loads((root / "meta/info.json").read_text()),
        "dataset_root": str(root),
        "manifest_sha256": digest,
    }


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("schema", 2, "receipt-schema"),
        ("status", "incomplete", "receipt-schema"),
        ("kind", "other", "receipt-schema"),
        ("source_partition", "holdout", "split-leakage"),
        ("lerobot_split", "test", "split-leakage"),
        ("statistics_policy", "upstream", "statistics-policy"),
        ("artifact_manifest_sha256", "0" * 64, "artifact-manifest"),
    ],
)
def test_receipt_semantics(local_export, field, value, code):
    root, receipt, _ = local_export
    receipt[field] = value
    if field == "artifact_manifest_sha256":
        receipt["sha256"] = canonical({k: v for k, v in receipt.items() if k != "sha256"})
        write_json(root.parent / "export-receipt.json", receipt)
        digest = receipt["sha256"]
    else:
        digest = seal(root.parent, receipt)
    with pytest.raises(LocalDatasetError) as error:
        verify_local_export(root, digest)
    assert error.value.code == code


@pytest.mark.parametrize("digest", [None, "", "0" * 64, "A" * 64, "1234"])
def test_missing_or_mismatched_digest(local_export, digest):
    with pytest.raises(LocalDatasetError, match="SHA-256"):
        verify_local_export(local_export[0], digest)


@pytest.mark.parametrize(
    "relative",
    [
        "incomplete.json",
        "export-receipt.pending.json",
        "dataset/meta/info.incomplete.json",
        "quarantine.json",
    ],
)
def test_marker_refused_even_if_inventoried(local_export, relative):
    root, receipt, _ = local_export
    (root.parent / relative).write_text("unfinished")
    digest = seal(root.parent, receipt)
    with pytest.raises(LocalDatasetError) as error:
        verify_local_export(root, digest)
    assert error.value.code == "incomplete-export"


@pytest.mark.parametrize(
    "mutation",
    [
        "changed",
        "deleted",
        "extra",
        "duplicate",
        "traversal",
        "absolute",
        "symlink_file",
        "symlink_dir",
        "symlink_root",
        "symlink_receipt",
    ],
)
def test_inventory_and_paths(local_export, tmp_path, mutation):
    root, receipt, digest = local_export
    data = root / "data/chunk-000/file-000.parquet"
    if mutation == "changed":
        data.write_bytes(b"corrupt")
    elif mutation == "deleted":
        data.unlink()
    elif mutation == "extra":
        (root / "unlisted.txt").write_text("extra")
    elif mutation in ("duplicate", "traversal", "absolute"):
        if mutation == "duplicate":
            receipt["artifacts"].append(copy.deepcopy(receipt["artifacts"][0]))
        else:
            receipt["artifacts"][0]["path"] = (
                "../outside" if mutation == "traversal" else "/tmp/outside"
            )
        digest = seal(root.parent, receipt, inventory=False)
    elif mutation == "symlink_file":
        outside = tmp_path / "outside"
        outside.write_bytes(data.read_bytes())
        data.unlink()
        data.symlink_to(outside)
    elif mutation == "symlink_dir":
        (root / "extra-dir").symlink_to(tmp_path, target_is_directory=True)
    elif mutation == "symlink_root":
        link = tmp_path / "linked-export"
        link.symlink_to(root.parent, target_is_directory=True)
        root = link / "dataset"
    elif mutation == "symlink_receipt":
        path = root.parent / "export-receipt.json"
        outside = tmp_path / "outside-receipt"
        path.rename(outside)
        path.symlink_to(outside)
    with pytest.raises(LocalDatasetError):
        verify_local_export(root, digest)


@pytest.mark.parametrize(
    "part,field,value",
    [
        ("writer", "version", "0.5.2"),
        ("writer", "source_commit", "0" * 40),
        ("writer", "source_files", {}),
        ("writer", "wheel_sha256", "0" * 64),
        ("normalization_processor", "profile", "act"),
        ("normalization_processor", "status", "failed"),
        ("normalization_processor", "image_statistics_used", True),
        ("normalization_processor", "resolved_feature_modes", {}),
        ("normalization_processor", "source_files", {}),
        ("loader_roundtrip", "frames_read", 999),
        ("loader_roundtrip", "status", "failed"),
        ("statistics_adapter", "training_rows_sha256", "0" * 64),
        ("statistics_adapter", "training_partition_sha256", "0" * 64),
        ("statistics_adapter", "source_dataset_sha256", "0" * 64),
        ("statistics_adapter", "split_sha256", "0" * 64),
        ("statistics_adapter", "statistics_sha256", "0" * 64),
    ],
)
def test_bound_evidence(local_export, part, field, value):
    root, receipt, _ = local_export
    receipt[part][field] = value
    digest = seal(root.parent, receipt)
    with pytest.raises(LocalDatasetError):
        verify_local_export(root, digest)


@pytest.mark.parametrize(
    "mutation",
    [
        "shape",
        "quantile",
        "nan",
        "count",
        "negative_std",
        "mean_bounds",
        "feature_names",
        "v2",
        "split",
        "frame_binding",
        "source_identity",
        "partition_rows",
    ],
)
def test_metadata_consistency_after_rehash(local_export, mutation):
    root, receipt, _ = local_export
    info = json.loads((root / "meta/info.json").read_text())
    stats = json.loads((root / "meta/stats.json").read_text())
    if mutation == "shape":
        stats["action"]["mean"] = [0]
    elif mutation == "quantile":
        stats["action"]["q01"] = [0, 0]
    elif mutation == "nan":
        (root / "meta/stats.json").write_text('{"action": NaN}')
    elif mutation == "count":
        stats["action"]["count"] = [1]
    elif mutation == "negative_std":
        stats["action"]["std"] = [-1, 0]
    elif mutation == "mean_bounds":
        stats["action"]["mean"] = [9, 0]
    elif mutation == "feature_names":
        info["features"]["action"]["names"] = ["wrong", "names"]
    elif mutation == "v2":
        info["codebase_version"] = "v2.1"
    elif mutation == "split":
        info["splits"] = {"train": "0:1", "test": "0:1"}
    elif mutation == "frame_binding":
        path = root.parent / "provenance/frames.jsonl"
        path.write_text(path.read_text().replace('"source_index": 0', '"source_index": 2'))
    elif mutation == "source_identity":
        path = root.parent / "provenance/source-identity.json"
        value = json.loads(path.read_text())
        value["sha256"] = "0" * 64
        write_json(path, value)
    elif mutation == "partition_rows":
        path = root.parent / "provenance/partition-rows.json"
        value = json.loads(path.read_text())
        value[0]["partition"] = "holdout"
        write_json(path, value)
    write_json(root / "meta/info.json", info)
    if mutation != "nan":
        write_json(root / "meta/stats.json", stats)
        receipt["statistics_adapter"]["statistics_sha256"] = canonical(stats)
    for replacement in receipt["statistics_adapter"]["replacement_statistics"]:
        replacement["sha256"] = hashlib.sha256(
            (root.parent / replacement["path"]).read_bytes()
        ).hexdigest()
    digest = seal(root.parent, receipt)
    with pytest.raises(LocalDatasetError):
        verify_local_export(root, digest)


def test_preflight_is_local_and_padding_is_real(local_cfg):
    with (
        patch(
            "tether.finetune.preflight.schema._fetch_dataset_features",
            side_effect=AssertionError("Hub forbidden"),
        ),
        patch(
            "tether.finetune.preflight.dataset_size._fetch_dataset_info",
            side_effect=AssertionError("Hub forbidden"),
        ),
    ):
        report = run_preflight(local_cfg)
        assert not report.has_failures
        assert check_schema(local_cfg).detail["uses_action_padding"] is True
        assert check_dataset_size(local_cfg).detail["dataset_episodes"] == 1
        command = _build_lerobot_command(local_cfg)
    assert f"--dataset.root={local_cfg.dataset_root}" in command
    assert "--dataset.repo_id=studio-local/synthetic" in command
    assert "--policy.type=smolvla" not in command
    assert not any("use_imagenet_stats" in arg for arg in command)
    detail = report.checks[0].detail
    assert detail["effective_use_imagenet_stats"] is True
    assert detail["visual_normalization"] == "IDENTITY"


def test_explicit_imagenet_setting_preserved(local_cfg):
    local_cfg.extra_lerobot_args = {"dataset.use_imagenet_stats": "false"}
    assert run_preflight(local_cfg).checks[0].detail["effective_use_imagenet_stats"] is False
    assert "--dataset.use_imagenet_stats=false" in _build_lerobot_command(local_cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("skip_preflight", True),
        ("resume", True),
        ("dataset_root", None),
        ("dataset_manifest_sha256", None),
        ("dataset_revision", "revision"),
        ("policy", "act"),
        ("mode", "full"),
        ("phase", "distill"),
        ("base_dataset", "other/dataset"),
    ],
)
def test_local_escape_hatches_block_before_training(local_cfg, field, value):
    setattr(local_cfg, field, value)
    with patch("tether.finetune.run._run_lerobot_training") as train:
        assert _validate_config(local_cfg)
        assert run_finetune(local_cfg).status == "aborted"
        assert run_preflight(local_cfg).has_failures
        with pytest.raises(LocalDatasetError):
            _build_lerobot_command(local_cfg)
    train.assert_not_called()


@pytest.mark.parametrize(
    "key",
    [
        "dataset.root",
        "dataset.repo_id",
        "dataset.revision",
        "dataset",
        "policy.normalization_mapping",
        "policy.normalization_mapping.VISUAL",
        "policy.pretrained_path",
        "config_path",
        "output_dir",
        "policy.input_features.observation.state",
        "policy.output_features.action",
        "rename_map",
        "peft",
        "peft.method_type",
        "peft.method_type.other",
        "rename_map.action",
        "resume",
        "dataset.episodes",
        "policy.preprocessor_overrides",
        "dataset.streaming",
        "dataset.image_transforms",
        "policy.max_action_dim",
        "policy.max_state_dim",
        "--dataset.root",
        "dataset.root=evil",
    ],
)
def test_extra_args_cannot_bypass(local_cfg, key):
    local_cfg.extra_lerobot_args = {key: "conflict"}
    assert _validate_config(local_cfg)
    with pytest.raises(LocalDatasetError):
        _build_lerobot_command(local_cfg)


@pytest.mark.parametrize(
    "change",
    [
        {"type": "act"},
        {"max_action_dim": 1},
        {"max_state_dim": 1},
        {"normalization_mapping": {}},
        {"max_action_dim": None},
    ],
)
def test_actual_base_contract_blocks(local_cfg, change):
    path = Path(local_cfg.base) / "config.json"
    config = json.loads(path.read_text())
    config.update(change)
    write_json(path, config)
    assert run_preflight(local_cfg).has_failures
    with pytest.raises(LocalDatasetError):
        _build_lerobot_command(local_cfg)


def test_missing_base_metadata_fails_local(local_cfg):
    (Path(local_cfg.base) / "config.json").unlink()
    assert run_preflight(local_cfg).has_failures


@pytest.mark.parametrize(
    "error", [LocalDatasetError("integrity", "corrupt"), RuntimeError("unexpected")]
)
def test_verifier_failures_never_become_warning(local_cfg, error):
    with (
        patch("tether.finetune.local_dataset.verify_local_export", side_effect=error),
        patch("tether.finetune.run._run_lerobot_training") as train,
    ):
        report = run_preflight(local_cfg)
        assert report.has_failures and report.checks[0].severity == "fail"
        assert run_finetune(local_cfg).status == "aborted"
    train.assert_not_called()


def test_schema_crash_fails_local(local_cfg):
    with patch(
        "tether.finetune.preflight.runner.check_schema",
        autospec=True,
        side_effect=RuntimeError("unexpected"),
    ):
        assert run_preflight(local_cfg).has_failures


def test_dry_run_never_trains(local_cfg):
    with patch("tether.finetune.run._run_lerobot_training") as train:
        result = run_finetune(local_cfg)
    assert result.status == "ok"
    train.assert_not_called()
    assert (
        "effective_use_imagenet_stats: True"
        in (local_cfg.output / "preflight_report.txt").read_text()
    )


def test_cli_flags_and_local_dry_run(local_cfg):
    import typer
    from typer.testing import CliRunner
    from tether.finetune.cli import finetune_command

    app = typer.Typer()
    app.command()(finetune_command)
    runner = CliRunner()
    with patch("tether.finetune.run._run_lerobot_training") as train:
        result = runner.invoke(
            app,
            [
                "--base",
                local_cfg.base,
                "--dataset",
                local_cfg.dataset,
                "--dataset-root",
                str(local_cfg.dataset_root),
                "--dataset-manifest-sha256",
                local_cfg.dataset_manifest_sha256,
                "--output",
                str(local_cfg.output),
                "--dry-run",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "status: ok" in result.output
    train.assert_not_called()


def test_output_cannot_mutate_export(local_cfg):
    local_cfg.output = local_cfg.dataset_root.parent / "new-training-output"
    assert run_finetune(local_cfg).status == "aborted"
    assert not local_cfg.output.exists()


def test_missing_receipt_and_wrong_root(local_export):
    root, _, digest = local_export
    with pytest.raises(LocalDatasetError):
        verify_local_export(root.parent, digest)
    (root.parent / "export-receipt.json").unlink()
    with pytest.raises(LocalDatasetError) as error:
        verify_local_export(root, digest)
    assert error.value.code == "missing-receipt"


def test_duplicate_json_fields_refused(local_export):
    root, _, digest = local_export
    path = root.parent / "export-receipt.json"
    path.write_text(path.read_text().replace('"schema": 1', '"schema": 1, "schema": 1'))
    with pytest.raises(LocalDatasetError) as error:
        verify_local_export(root, digest)
    assert error.value.code == "invalid-json"


def test_unsupported_requested_profile(local_export):
    root, _, digest = local_export
    with pytest.raises(LocalDatasetError) as error:
        verify_local_export(root, digest, expected_profile="act")
    assert error.value.code == "unsupported-profile"


def test_receipt_changed_after_preflight_blocks_command(local_cfg):
    assert not run_preflight(local_cfg).has_failures
    (local_cfg.dataset_root / "data/chunk-000/file-000.parquet").write_bytes(b"changed later")
    with pytest.raises(LocalDatasetError) as error:
        _build_lerobot_command(local_cfg)
    assert error.value.code == "artifact-integrity"


def test_omitted_base_modes_use_only_pinned_defaults(local_cfg):
    path = Path(local_cfg.base) / "config.json"
    config = json.loads(path.read_text())
    del config["normalization_mapping"]
    write_json(path, config)
    result = check_schema(local_cfg)
    assert result.severity == "ok"
    assert (
        result.detail["normalization_mapping_source"]
        == "pinned LeRobot 0.5.1 SmolVLAConfig defaults"
    )
    config["normalization_mapping"] = None
    write_json(path, config)
    assert check_schema(local_cfg).severity == "fail"


def test_declared_source_config_fallback_remains_receipt_bound(local_export):
    root, receipt, _ = local_export
    case = receipt["episodes"][0]["case"]
    case["identity_sources"].update(
        {
            "task": "source_config.task",
            "embodiment": "source_config.embodiment",
            "initial_state": "source_config.initial_state",
        }
    )
    path = root.parent / "provenance/frames.jsonl"
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    rows_path = root.parent / "provenance/partition-rows.json"
    rows = json.loads(rows_path.read_text())
    for index, frame in enumerate(frames):
        del frame["example"]["task"]
        del frame["example"]["embodiment"]
        del frame["example"]["source"]["initial_state"]
        rows[index]["example_sha256"] = canonical(frame["example"])
    path.write_text("".join(json.dumps(frame) + "\n" for frame in frames))
    write_json(rows_path, rows)
    receipt["statistics_adapter"]["training_rows_sha256"] = canonical(rows[:2])
    digest = seal(root.parent, receipt)
    assert verify_local_export(root, digest)["manifest_sha256"] == digest
    case["identity_sources"]["task"] = "derived-task"
    digest = seal(root.parent, receipt)
    with pytest.raises(LocalDatasetError):
        verify_local_export(root, digest)


def test_output_extra_cannot_target_immutable_export(local_cfg):
    local_cfg.extra_lerobot_args = {"output_dir": str(local_cfg.dataset_root.parent / "training")}
    assert run_finetune(local_cfg).status == "aborted"
    with pytest.raises(LocalDatasetError):
        _build_lerobot_command(local_cfg)
    assert not (local_cfg.dataset_root.parent / "training").exists()


def test_existing_positional_config_arguments_stay_compatible(tmp_path):
    cfg = FinetuneConfig("base", "owner/dataset", tmp_path, "base-rev", "data-rev", 123)
    assert cfg.num_steps == 123
    assert cfg.dataset_root is None
    assert cfg.dataset_manifest_sha256 is None
