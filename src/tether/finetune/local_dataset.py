"""Offline admission of one Studio verified-moments-v1 SmolVLA export.

The caller supplies the export receipt's canonical SHA-256, not an arbitrary
receipt path. This verifies retained evidence and bytes, without importing
LeRobot, reading model weights, or claiming to re-run the exporter/loader.
Exports must remain immutable after verification, including during training.
"""

from __future__ import annotations

from tether._lerobot_version import QUALIFIED_LEROBOT_VERSION

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, NoReturn


ADAPTER_SOURCE = "af6969a6663d0b288db781ad46736eb62d33a361dbe75cd578da6c163b075307"
WRITER_SOURCE = "7e241bd630a3719a56157a497ce5d08f244784f1"
WRITER_WHEEL = "1894516040c65f80a45bd9741f8174aae90ed5d93da0627ab4f1a85fd8d75e90"
WRITER_FILES = {
    "lerobot/datasets/lerobot_dataset.py": "07523e2a4f9c53b6a96df9255ad131ee30e707fd8610ec9d2178d3741023ae3a",
    "lerobot/datasets/utils.py": "1fd3eddf70baf1c5c7f4e745c78338c580385283fad00e853d49cd0f34dd57e3",
    "lerobot/datasets/compute_stats.py": "d546a5c690eff716a10e9b13d70cd5d1ef7fed88f2f9c40c99f2f7f50845cf45",
    "lerobot/datasets/dataset_writer.py": "319f4efb53e892ceeb439aae6d25209d14564e3dc91134c7cc2c6c0b5e71ebc5",
    "lerobot/datasets/dataset_reader.py": "3a477dae6aa887449f09be25b385485b065861d13c5448b9369a0ba09f01bb5d",
    "lerobot/datasets/dataset_metadata.py": "23c927553a53e320161f35f9b36e8b61a446833ca17a4287b9ef5cdc8d19c50f",
    "lerobot/datasets/io_utils.py": "af08c7875f34f8469f80c4fd903af12971d6fa96c179be984887b93ef2f6f91c",
    "lerobot/datasets/feature_utils.py": "928369cff50f0d663a125abfc67683ccbae0e552ed2eb22edde9e6bcff2aef81",
    "lerobot/datasets/image_writer.py": "54cad218c99caad105ea8738507ee87fc8c2858832d31aa5b21d324c97360631",
}
NORMALIZATION_FILES = {
    "lerobot/processor/normalize_processor.py": "f0cd88bec4c954a93ba1cc90300473b96eb177762789c1780cf07801fc1af9d6",
    "lerobot/policies/act/configuration_act.py": "35a60c0dd3c811b137b5eb035bdbd40d40e2c24ba0cda61756052e6e1c829282",
    "lerobot/policies/smolvla/configuration_smolvla.py": "2fb637cb428fa2fdf1d114646dcffaf4728216bfe5b7039d5d0cac4857ffc4e0",
    "lerobot/configs/default.py": "e8307a48aa97c1b3fd0722e9a849deb1bc1bb48b63e3109b63683eb7ffe08105",
}
IDENTITY_FIELDS = ["task", "seed", "episode", "initial_state", "embodiment"]
PARTITIONS = ("development", "holdout", "test")
MOMENTS = {"count", "min", "max", "mean", "std"}


class LocalDatasetError(ValueError):
    """A stable refusal code for local admission callers, including Studio."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _fail(code: str, message: str) -> NoReturn:
    raise LocalDatasetError(code, message)


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        _fail(code, message)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _object(value: Any, label: str) -> dict:
    _require(isinstance(value, dict), "invalid-metadata", f"{label} must be an object.")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "invalid-json", f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_pairs,
        parse_constant=lambda v: _fail("invalid-json", f"Nonfinite JSON: {v}"),
    )


def _json(path: Path) -> Any:
    return _loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: Any) -> str:
    _require(
        isinstance(value, str) and bool(value) and "\\" not in value,
        "unsafe-path",
        "Artifacts require relative POSIX paths.",
    )
    path = PurePosixPath(value)
    _require(
        not path.is_absolute() and ".." not in path.parts and str(path) == value,
        "unsafe-path",
        f"Unsafe artifact path: {value!r}",
    )
    return value


def _root(path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    for part in [*reversed(path.parents), path]:
        _require(not part.is_symlink(), "unsafe-path", f"Symlink path component: {part}")
    path = path.resolve(strict=True)
    _require(
        path.name == "dataset" and path.is_dir(),
        "dataset-root",
        "dataset_root must be the actual EXPORT/dataset directory.",
    )
    return path


def _inventory(export: Path, receipt: dict) -> dict[str, dict]:
    artifacts = receipt.get("artifacts")
    _require(
        isinstance(artifacts, list) and bool(artifacts),
        "artifact-inventory",
        "A complete artifact inventory is required.",
    )
    indexed = {}
    for item in artifacts:
        item = _object(item, "artifact")
        relative = _safe_relative(item.get("path"))
        _require(
            relative != "export-receipt.json" and relative not in indexed,
            "artifact-inventory",
            "Artifact paths must be unique and exclude the receipt.",
        )
        _require(
            set(item) == {"path", "size_bytes", "sha256"}
            and _integer(item.get("size_bytes"), 1)
            and _digest(item.get("sha256")),
            "artifact-inventory",
            f"Invalid artifact metadata: {relative}",
        )
        indexed[relative] = item
    _require(
        receipt.get("artifact_manifest_sha256") == _hash(artifacts),
        "artifact-manifest",
        "Artifact manifest digest differs from the receipt.",
    )
    actual = set()
    # os.walk does not follow directory symlinks; explicitly reject every entry.
    for directory, dirs, files in os.walk(export, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            relative = path.relative_to(export).as_posix()
            _require(
                not any(
                    marker in name.lower() for marker in ("incomplete", "pending", "quarantine")
                ),
                "incomplete-export",
                f"Unfinished export marker: {relative}",
            )
            mode = path.lstat().st_mode
            _require(
                stat.S_ISREG(mode) or stat.S_ISDIR(mode),
                "unsafe-path",
                f"Only regular files/directories are allowed: {relative}",
            )
            if stat.S_ISREG(mode) and relative != "export-receipt.json":
                actual.add(relative)
    _require(
        actual == set(indexed),
        "artifact-inventory",
        "Actual export files differ from the complete receipt inventory.",
    )
    for relative, item in indexed.items():
        path = export / relative
        _require(
            path.stat().st_size == item["size_bytes"] and _file_hash(path) == item["sha256"],
            "artifact-integrity",
            f"Artifact size or SHA-256 mismatch: {relative}",
        )
    required = {
        "dataset/meta/info.json",
        "dataset/meta/stats.json",
        "dataset/meta/tasks.parquet",
        "provenance/source-identity.json",
        "provenance/split.json",
        "provenance/partition-rows.json",
        "provenance/frames.jsonl",
    }
    _require(
        required <= actual
        and any(re.fullmatch(r"dataset/data/chunk-\d{3,}/file-\d{3,}\.parquet", p) for p in actual)
        and any(
            re.fullmatch(r"dataset/meta/episodes/chunk-\d{3,}/file-\d{3,}\.parquet", p)
            for p in actual
        ),
        "missing-artifact",
        "Required LeRobot v3 metadata, data, and provenance must be local.",
    )
    return indexed


def _shape(value: Any, shape: list[int], label: str) -> list[float]:
    if not shape:
        _require(
            type(value) in (int, float) and math.isfinite(value),
            "statistics-shape",
            f"{label} must contain finite numbers.",
        )
        return [value]
    _require(
        isinstance(value, list) and len(value) == shape[0],
        "statistics-shape",
        f"Invalid {label} shape; expected {shape}.",
    )
    return [number for child in value for number in _shape(child, shape[1:], label)]


def _features_and_stats(info: dict, stats: dict, receipt: dict) -> str:
    _require(
        info.get("codebase_version") == "v3.0", "dataset-version", "Only LeRobot v3.0 is admitted."
    )
    frames, episodes = info.get("total_frames"), info.get("total_episodes")
    _require(
        _integer(frames, 1)
        and _integer(episodes, 1)
        and _integer(info.get("total_tasks"), 1)
        and info["total_tasks"] <= episodes <= frames,
        "dataset-counts",
        "Invalid local dataset counts.",
    )
    _require(
        info.get("splits") == {"train": f"0:{episodes}"}
        and info.get("data_path") == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        and info.get("video_path") is None,
        "dataset-layout",
        "Only the exported development/train still-image layout is admitted.",
    )
    contract = _object(receipt.get("contract"), "contract")
    camera = contract.get("camera_key")
    _require(
        isinstance(camera, str)
        and re.fullmatch(r"observation\.images\.[A-Za-z0-9_]+", camera) is not None,
        "feature-contract",
        "An explicit single camera key is required.",
    )
    _require(
        _integer(contract.get("fps"), 1)
        and contract["fps"] <= 1000
        and info.get("fps") == contract["fps"]
        and isinstance(contract.get("state_key"), str)
        and bool(contract["state_key"]),
        "feature-contract",
        "State key and retained sample rate must agree.",
    )
    features = _object(info.get("features"), "features")
    expected = {
        "action",
        "observation.state",
        camera,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    _require(
        set(features) == expected and set(stats) == expected,
        "feature-contract",
        "The complete exported feature/statistics sets must agree.",
    )
    for key, names_key in (("action", "action_names"), ("observation.state", "state_names")):
        feature = _object(features[key], key)
        names = contract.get(names_key)
        _require(
            isinstance(names, list)
            and bool(names)
            and all(isinstance(n, str) and n.strip() for n in names)
            and len(set(names)) == len(names)
            and feature.get("names") == names
            and feature.get("shape") == [len(names)]
            and feature.get("dtype") == "float32",
            "feature-contract",
            f"{key} shape/names must match the retained contract.",
        )
    visual = _object(features[camera], camera)
    shape = visual.get("shape")
    _require(
        isinstance(shape, list)
        and len(shape) == 3
        and all(_integer(n, 1) for n in shape)
        and shape[2] == 3
        and visual.get("dtype") == "image"
        and visual.get("names") == ["height", "width", "channels"],
        "feature-contract",
        "One retained HWC RGB still-image feature is required.",
    )
    for key in expected - {"action", "observation.state", camera}:
        feature = _object(features[key], key)
        _require(
            feature.get("shape") == [1]
            and feature.get("dtype") == ("float32" if key == "timestamp" else "int64"),
            "feature-contract",
            f"Invalid index/timestamp feature: {key}",
        )
    for key, feature in features.items():
        values = _object(stats[key], f"statistics/{key}")
        _require(
            set(values) == MOMENTS,
            "statistics-policy",
            f"{key} requires moments-only statistics; quantiles are unsupported.",
        )
        stat_shape = [3, 1, 1] if key == camera else feature["shape"]
        flattened = {
            name: _shape(values[name], stat_shape, f"{key}/{name}")
            for name in ("min", "max", "mean", "std")
        }
        count = values["count"]
        _require(
            isinstance(count, list)
            and len(count) == 1
            and _integer(count[0], 1)
            and (count[0] <= frames if key == camera else count[0] == frames),
            "statistics-count",
            f"Invalid retained {key} statistics count.",
        )
        _require(
            all(
                lo <= mean <= hi and std >= 0
                for lo, mean, hi, std in zip(
                    flattened["min"], flattened["mean"], flattened["max"], flattened["std"]
                )
            ),
            "statistics-values",
            f"Invalid {key} moment bounds.",
        )
    return camera


def _verify_frame_identity(example: dict, case: dict) -> None:
    """Check retained row identities using the exporter's explicit source labels.

    source_config is not copied into the export: those fallback values are bound
    by the selected receipt's case, split, and original source-manifest digest.
    Do not pretend to independently recover absent source configuration.
    """
    source = _object(example.get("source"), "frame source")
    origins = _object(case.get("identity_sources"), "case identity sources")
    candidates = {
        "task": [
            (example.get("task"), "example.task"),
            (example.get("label"), "example.label"),
            (source.get("task"), "source.task"),
        ],
        "embodiment": [
            (example.get("embodiment"), "example.embodiment"),
            (source.get("embodiment"), "source.embodiment"),
        ],
    }
    config_origins = {
        "task": {"source_config.task", "source_config.instruction"},
        "embodiment": {"source_config.embodiment", "source_config.policy.embodiment"},
        "initial_state": {"source_config.initial_state"},
    }
    for key in IDENTITY_FIELDS:
        if key in candidates:
            found = next(
                (
                    (v.strip(), origin)
                    for v, origin in candidates[key]
                    if isinstance(v, str) and v.strip()
                ),
                None,
            )
        else:
            value = source.get(key)
            found = (value, "source." + key) if value is not None else None
        if found is not None:
            valid = found == (case.get(key), origins.get(key))
        else:
            valid = origins.get(key) in config_origins.get(key, set())
        _require(
            valid,
            "split-leakage",
            "Frame identity differs from its retained development case/source.",
        )


def _provenance(export: Path, receipt: dict, info: dict, adapter: dict) -> None:
    identity = _object(_json(export / "provenance/source-identity.json"), "source identity")
    split = _object(_json(export / "provenance/split.json"), "split")
    _require(
        all(
            _digest(receipt.get(k))
            for k in ("source_dataset_sha256", "source_manifest_sha256", "split_sha256")
        ),
        "source-binding",
        "Retained source/split digests are required.",
    )
    _require(
        identity.get("id") == receipt.get("source_dataset_id")
        and isinstance(identity.get("id"), str)
        and bool(identity["id"])
        and identity.get("sha256") == receipt["source_dataset_sha256"]
        and identity.get("source_type") == "materialized-dataset-edit"
        and identity.get("format") == "tether-recording-dataset-v1",
        "source-binding",
        "Retained corrected source identity differs from the receipt.",
    )
    _require(
        split.get("schema") == 2
        and split.get("identity_fields") == IDENTITY_FIELDS
        and split.get("dataset_id") == identity["id"]
        and split.get("dataset_sha256") == identity["sha256"]
        and split.get("sha256") == receipt["split_sha256"]
        and _hash({k: v for k, v in split.items() if k not in {"created_at", "sha256"}})
        == split["sha256"],
        "split-binding",
        "Retained split does not match source identity and canonical digest.",
    )
    locations = {}
    for partition in PARTITIONS:
        cases = split.get(partition)
        _require(
            isinstance(cases, list) and (bool(cases) or partition == "test"),
            "split-binding",
            "Development and holdout require explicit cases.",
        )
        for case in cases:
            case = _object(case, "split case")
            _require(
                set(case) == set(IDENTITY_FIELDS)
                and all(
                    isinstance(case[k], str) and case[k].strip() for k in ("task", "embodiment")
                )
                and all(_integer(case[k]) for k in ("seed", "episode"))
                and (case["initial_state"] is None or _integer(case["initial_state"])),
                "split-binding",
                "Invalid retained split identity.",
            )
            key = _hash(case)
            _require(key not in locations, "split-leakage", "Split case appears more than once.")
            locations[key] = partition
    rows = _json(export / "provenance/partition-rows.json")
    _require(
        isinstance(rows, list) and bool(rows),
        "source-binding",
        "Retained partition rows are required.",
    )
    counts = dict.fromkeys(PARTITIONS, 0)
    for index, row in enumerate(rows):
        row = _object(row, "partition row")
        _require(
            row.get("source_index") == index
            and type(row.get("source_index")) is int
            and row.get("partition") in PARTITIONS
            and _digest(row.get("example_sha256")),
            "source-binding",
            "Invalid partition row identity.",
        )
        counts[row["partition"]] += 1
    _require(
        counts == receipt.get("partition_row_counts")
        and counts["development"] == info["total_frames"]
        and adapter.get("training_partition_sha256") == _hash(split["development"])
        and adapter.get("training_rows_sha256")
        == _hash([r for r in rows if r["partition"] == "development"])
        and adapter.get("source_dataset_sha256") == identity["sha256"]
        and adapter.get("split_sha256") == split["sha256"],
        "source-binding",
        "Statistics/source/development partition binding differs.",
    )
    episodes = receipt.get("episodes")
    _require(
        isinstance(episodes, list) and len(episodes) == info["total_episodes"],
        "source-binding",
        "Retained episode count differs from local metadata.",
    )
    ordered = []
    for episode_index, episode in enumerate(episodes):
        episode = _object(episode, "episode")
        case = _object(episode.get("case"), "episode case")
        canonical_case = {key: case.get(key) for key in IDENTITY_FIELDS}
        _require(
            episode.get("partition") == "development"
            and locations.get(_hash(canonical_case)) == "development"
            and canonical_case["embodiment"] == info.get("robot_type")
            and _integer(canonical_case["initial_state"]),
            "split-leakage",
            "Every training episode must belong to development.",
        )
        indexes = episode.get("indexes")
        _require(isinstance(indexes, list) and bool(indexes), "source-binding", "Empty episode.")
        for frame_index, index in enumerate(indexes):
            _require(
                _integer(index) and index < len(rows) and rows[index]["partition"] == "development",
                "split-leakage",
                "Training episode contains non-development rows.",
            )
            ordered.append((index, episode_index, frame_index))
    _require(
        len({row[0] for row in ordered}) == len(ordered)
        and {row[0] for row in ordered}
        == {r["source_index"] for r in rows if r["partition"] == "development"},
        "source-binding",
        "Training episodes must cover development rows exactly once.",
    )
    with (export / "provenance/frames.jsonl").open(encoding="utf-8") as stream:
        frame_count = 0
        for line in stream:
            _require(
                bool(line.strip()) and frame_count < len(ordered),
                "source-binding",
                "Unexpected provenance frame.",
            )
            frame = _object(_loads(line), "frame")
            source_index, episode_index, frame_index = ordered[frame_count]
            example = _object(frame.get("example"), "frame example")
            _require(
                frame.get("source_index") == source_index
                and frame.get("episode_index") == episode_index
                and frame.get("frame_index") == frame_index
                and _hash(example) == rows[source_index]["example_sha256"],
                "source-binding",
                "Provenance frame differs from retained development row.",
            )
            _verify_frame_identity(example, episodes[episode_index]["case"])
            frame_count += 1
    _require(
        frame_count == info["total_frames"],
        "source-binding",
        "Missing retained development frames.",
    )


def _verify(dataset_root: Path, expected_manifest_sha256: str, expected_profile: str) -> dict:
    _require(
        expected_profile == "smolvla",
        "unsupported-profile",
        "Only SmolVLA local admission is qualified.",
    )
    _require(
        _digest(expected_manifest_sha256),
        "manifest-digest",
        "Supply the canonical export receipt SHA-256.",
    )
    root = _root(dataset_root)
    export = root.parent
    receipt_path = export / "export-receipt.json"
    _require(
        not receipt_path.is_symlink() and receipt_path.is_file(),
        "missing-receipt",
        "The sibling EXPORT/export-receipt.json is required.",
    )
    receipt = _object(_json(receipt_path), "export receipt")
    _require(
        receipt.get("sha256") == expected_manifest_sha256
        and _hash({k: v for k, v in receipt.items() if k != "sha256"}) == expected_manifest_sha256,
        "manifest-digest",
        "Export receipt does not match the selected canonical SHA-256.",
    )
    _require(
        type(receipt.get("schema")) is int
        and receipt["schema"] == 1
        and receipt.get("kind") == "studio-corrected-lerobot-offline-export"
        and receipt.get("status") == "complete",
        "receipt-schema",
        "A complete supported Studio export receipt is required.",
    )
    _require(
        receipt.get("source_partition") == "development"
        and receipt.get("lerobot_split") == "train",
        "split-leakage",
        "Only development exported to train is admitted.",
    )
    _require(
        receipt.get("statistics_policy") == "verified-moments-v1",
        "statistics-policy",
        "The verified-moments-v1 export statistics policy is required.",
    )
    inventory = _inventory(export, receipt)
    writer = _object(receipt.get("writer"), "writer")
    _require(
        writer.get("version") == QUALIFIED_LEROBOT_VERSION
        and writer.get("source_commit") == WRITER_SOURCE
        and writer.get("wheel_sha256") == WRITER_WHEEL
        and writer.get("source_files") == WRITER_FILES
        and _object(writer.get("runtime_versions"), "writer runtime").get("lerobot") == QUALIFIED_LEROBOT_VERSION,
        "writer-pin",
        f"Export writer/source pins must match qualified LeRobot {QUALIFIED_LEROBOT_VERSION}.",
    )
    info = _object(_json(root / "meta/info.json"), "dataset info")
    stats = _object(_json(root / "meta/stats.json"), "dataset statistics")
    camera = _features_and_stats(info, stats, receipt)
    normalization = _object(receipt.get("normalization_processor"), "normalization processor")
    _require(
        normalization.get("profile") == expected_profile,
        "unsupported-profile",
        "Receipt profile must be SmolVLA.",
    )
    _require(
        normalization.get("status") == "passed"
        and normalization.get("resolved_feature_modes")
        == {"action": "MEAN_STD", "observation.state": "MEAN_STD", camera: "IDENTITY"}
        and normalization.get("source_files") == NORMALIZATION_FILES
        and normalization.get("epsilon") == 1e-8
        and normalization.get("numeric_frames") == info["total_frames"]
        and normalization.get("image_statistics_used") is False
        and normalization.get("imagenet_override_applied_in_this_check") is False
        and normalization.get("trainer_use_imagenet_stats_default") is True,
        "normalization-contract",
        "Pinned SmolVLA normalization evidence must pass and retain VISUAL=IDENTITY.",
    )
    loader = _object(receipt.get("loader_roundtrip"), "loader roundtrip")
    _require(
        loader.get("status") == "passed"
        and loader.get("frames_read") == info["total_frames"]
        and loader.get("episodes_read") == info["total_episodes"],
        "loader-contract",
        "Passed loader evidence must agree with local counts.",
    )
    adapter = _object(receipt.get("statistics_adapter"), "statistics adapter")
    _require(
        adapter.get("policy") == "verified-moments-v1"
        and adapter.get("adapter_source_sha256") == ADAPTER_SOURCE
        and adapter.get("statistics_sha256") == _hash(stats),
        "statistics-binding",
        "Actual local statistics differ from the verified adapter evidence.",
    )
    replacements = adapter.get("replacement_statistics")
    originals = adapter.get("original_statistics")
    targets = {"dataset/meta/stats.json"} | {
        p for p in inventory if p.startswith("dataset/meta/episodes/")
    }
    _require(
        isinstance(replacements, list)
        and isinstance(originals, list)
        and len(replacements) == len(targets)
        and len(originals) == len(targets),
        "statistics-binding",
        "Complete original and replacement statistics bindings are required.",
    )
    _require(
        {r.get("path") for r in replacements} == targets
        and {r.get("output_path") for r in originals} == targets,
        "statistics-binding",
        "Statistics paths differ from the local inventory.",
    )
    for item in replacements:
        _require(
            item.get("sha256") == inventory[item["path"]]["sha256"],
            "statistics-binding",
            "Replacement statistics digest differs.",
        )
    for item in originals:
        retained = "provenance/upstream-statistics/" + item["output_path"].removeprefix("dataset/")
        _require(
            item.get("retained_path") == retained
            and retained in inventory
            and item.get("sha256") == inventory[retained]["sha256"],
            "statistics-binding",
            "Retained upstream statistics digest differs.",
        )
    _provenance(export, receipt, info, adapter)
    image_indexes, offset = [], 0
    for episode in receipt["episodes"]:
        length = len(episode["indexes"])
        image_indexes.extend(sorted({offset, offset + length - 1}))
        offset += length
    _require(
        normalization.get("image_frame_indexes") == image_indexes
        and normalization.get("maximum_image_batch") == 1,
        "normalization-contract",
        "Representative image check evidence differs from exported episodes.",
    )
    return {
        "receipt": receipt,
        "info": info,
        "dataset_root": str(root),
        "manifest_sha256": expected_manifest_sha256,
    }


def verify_local_export(
    dataset_root: Path, expected_manifest_sha256: str, *, expected_profile: str = "smolvla"
) -> dict:
    """Verify local bytes and retained evidence; never fall back to the Hub.

    Raises LocalDatasetError with a stable code on missing, malformed, unsafe,
    unsupported or inconsistent input. No external packages/network are used.
    """
    try:
        return _verify(dataset_root, expected_manifest_sha256, expected_profile)
    except LocalDatasetError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise LocalDatasetError(
            "invalid-local-export", f"Cannot verify local export: {exc}"
        ) from exc


def validate_local_config(cfg: Any) -> None:
    """Reject local escape hatches before any preflight or command is built."""
    root, digest = cfg.dataset_root, cfg.dataset_manifest_sha256
    content = getattr(cfg, "dataset_content_sha256", None)
    raw = root is not None and digest is None
    if root is None and digest is None and content is None:
        return
    _require(
        root is not None and (digest is None or _digest(digest)) and (content is None or _digest(content)) and not (digest and content),
        "local-config",
        "dataset_root is required with a valid optional dataset_manifest_sha256 or dataset_content_sha256.",
    )
    _require(
        not cfg.skip_preflight and (not cfg.resume or
            (raw and cfg.policy in {"diffusion", "vqbet", "tdmpc"} and cfg.mode == "full")),
        "local-config",
        "Local dataset admission does not support skip_preflight or resume outside the state-bound local Diffusion/VQ-BeT/TDMPC recipes.",
    )
    _require(
        cfg.phase == "train"
        and cfg.backend == "lerobot"
        and ((cfg.policy in {"act", "diffusion", "vqbet", "tdmpc"} and cfg.mode == "full") or (cfg.policy == "auto" and cfg.mode == "lora" and bool(cfg.base)))
        and (raw or cfg.policy == "auto")
        and bool(cfg.dataset)
        and not cfg.base_dataset,
        "unsupported-profile",
        "Local datasets require ACT, Diffusion, VQ-BeT or TDMPC from scratch (policy=act|diffusion|vqbet|tdmpc, mode=full) or pretrained SmolVLA LoRA (policy=auto, mode=lora), phase=train.",
    )
    _require(not raw or not (Path(root).expanduser().resolve().parent / "export-receipt.json").exists(), "local-config", "dataset_manifest_sha256: a Studio export receipt requires its explicit digest; it cannot be downgraded to raw admission.")
    export = Path(root).expanduser().resolve()
    if not raw:
        export = export.parent
    _require(
        not Path(cfg.output).expanduser().resolve().is_relative_to(export),
        "local-config",
        "Training output must be outside the immutable export directory.",
    )
    _require(
        not cfg.dataset_revision,
        "local-config",
        "Local datasets cannot select a Hub dataset revision.",
    )
    for key in cfg.extra_lerobot_args:
        _require(
            isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", key) is not None,
            "local-config",
            "Local extra argument names must be canonical dotted keys.",
        )
        normalized = key
        _require(
            normalized
            not in {
                "dataset",
                "dataset.root",
                "dataset.repo_id",
                "dataset.revision",
                "config_path",
                "output_dir",
                "resume",
                "rename_map",
                "peft",
                "peft.method_type",
                # Raw v3 admission checks video frames at LeRobot's default.
                "tolerance_s",
            }
            and (
                not normalized.startswith("dataset.") or normalized == "dataset.use_imagenet_stats"
            )
            and not normalized.startswith(
                (
                    "policy.input_features.",
                    "policy.output_features.",
                    "rename_map.",
                    "peft.method_type.",
                )
            )
            and ((raw and normalized == "policy.normalization_mapping") or "normaliz" not in normalized.lower())
            and "processor" not in normalized.lower()
            and normalized
            not in {
                "policy",
                "policy.type",
                "policy.path",
                "policy.pretrained_path",
                "policy.pretrained_model_path",
                "policy.input_features",
                "policy.output_features",
                "policy.max_action_dim",
                "policy.max_state_dim",
                "dataset.episodes",
                "dataset.streaming",
                "dataset.video_backend",
            },
            "local-config",
            f"extra_lerobot_args cannot override local admission: {key}",
        )
    effective_imagenet_stats(cfg)


def effective_imagenet_stats(cfg: Any) -> bool:
    """Keep LeRobot's default; record any explicit existing escape-hatch value."""
    value = cfg.extra_lerobot_args.get("dataset.use_imagenet_stats", True)
    if value is True or (isinstance(value, str) and value in ("true", "True")):
        return True
    if value is False or (isinstance(value, str) and value in ("false", "False")):
        return False
    _fail("local-config", "dataset.use_imagenet_stats must be an explicit boolean.")


__all__ = ["LocalDatasetError", "verify_local_export"]
