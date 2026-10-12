"""Local LeRobot v3 schema, row integrity and content identity, without Hub access."""

from __future__ import annotations

import math
from bisect import bisect_left
from pathlib import Path
import stat
from typing import Any

from tether.finetune.local_dataset import (
    LocalDatasetError,
    _file_hash,
    _hash,
    _integer,
    _json,
    _object,
    _require,
)


def _check(condition: bool, field: str, detail: str) -> None:
    _require(condition, "local-v3-invalid", f"{field}: {detail}")


def validate_features(info: dict, expected: dict | None = None) -> None:
    _check(
        expected is None or isinstance(expected, dict),
        "dataset_features",
        "expected a feature object",
    )
    features = _object(info.get("features"), "meta/info.json.features")
    for key, feature in features.items():
        feature = _object(feature, f"features.{key}")
        shape = feature.get("shape")
        _check(
            isinstance(shape, list) and bool(shape) and all(_integer(n, 1) for n in shape),
            f"features.{key}.shape",
            "expected positive integer dimensions",
        )
        _check(
            feature.get("dtype")
            in {
                "float16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "bool",
                "string",
                "image",
                "video",
            },
            f"features.{key}.dtype",
            "unsupported LeRobot dtype",
        )
        names = feature.get("names")
        _check(
            names is None or isinstance(names, (list, dict)),
            f"features.{key}.names",
            "expected dimension names or null",
        )
    for key in ("action", "observation.state"):
        feature = features.get(key, {})
        _check(
            feature.get("dtype") == "float32" and len(feature.get("shape", [])) == 1,
            f"features.{key}",
            "expected a float32 vector",
        )
    cameras = {k for k, f in features.items() if f["dtype"] in {"image", "video"}}
    _check(bool(cameras), "features.observation.images", "the recipe requires at least one camera")
    for key in cameras:
        f = features[key]
        _check(
            key.startswith("observation.images.") and len(f["shape"]) == 3 and f["shape"][-1] == 3,
            f"features.{key}.shape",
            "expected HWC RGB camera",
        )
    for key in ("index", "episode_index", "frame_index", "task_index", "timestamp"):
        f = features.get(key, {})
        _check(
            f.get("shape") == [1]
            and f.get("dtype") == ("float32" if key == "timestamp" else "int64"),
            f"features.{key}",
            "invalid index/timestamp schema",
        )
    for key, wanted in (expected or {}).items():
        _check(key in features, f"features.{key}", "required by the recipe/base checkpoint")
        shape = wanted.get("shape") if isinstance(wanted, dict) else wanted
        # Policy configs store visual shapes CHW, dataset info stores HWC.
        if isinstance(wanted, dict) and wanted.get("type") == "VISUAL":
            shape = [shape[1], shape[2], shape[0]]
        _check(
            features[key]["shape"] == shape,
            f"features.{key}.shape",
            f"expected {shape}, got {features[key]['shape']}",
        )
    expected_cameras = {k for k in (expected or {}) if k.startswith("observation.images.")}
    if expected_cameras:
        _check(
            cameras == expected_cameras,
            "features.observation.images",
            f"expected camera keys {sorted(expected_cameras)}, got {sorted(cameras)}",
        )


def _rows(paths: list[Path], label: str):
    _check(bool(paths), label, "missing local parquet files")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise LocalDatasetError(
            "local-v3-reader",
            "episode_index: pyarrow from the LeRobot training runtime is required",
        ) from exc
    for path in paths:
        try:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=32):
                yield from batch.to_pylist()
        except Exception as exc:
            raise LocalDatasetError("local-v3-invalid", f"{label} ({path.name}): {exc}") from exc


def _inspect_lerobot_v3(
    root: Path, *, expected_features: dict | None = None, expected_sha256: str | None = None,
    metadata_only: bool = False,
) -> dict:
    """Validate local metadata and episode/frame indexes, hash every retained byte.

    Includes videos in identity, since camera training inputs must also be bound.
    Callers must keep the admitted directory immutable until training completes.
    """
    try:
        root = Path(root).expanduser().absolute()
        for path in [*root.parents, root]:
            _check(not path.is_symlink(), "dataset_root", f"symlink component {path}")
        _check(root.is_dir(), "dataset_root", "directory does not exist")
        manifest = []
        for directory in (("meta",) if metadata_only else ("meta", "data", "videos", "images")):
            base = root / directory
            if not base.exists():
                continue
            _check(not base.is_symlink(), directory, "symlinks are forbidden")
            for path in sorted(base.rglob("*")):
                mode = path.lstat().st_mode
                _check(
                    stat.S_ISDIR(mode) or stat.S_ISREG(mode),
                    str(path.relative_to(root)),
                    "only regular files/directories are allowed",
                )
                if stat.S_ISREG(mode) and not metadata_only:
                    manifest.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "size_bytes": path.stat().st_size,
                            "sha256": _file_hash(path),
                        }
                    )
        manifest.sort(key=lambda row: row["path"])
        content = None if metadata_only else _hash(manifest)
        _check(
            expected_sha256 is None or content == expected_sha256,
            "dataset_content_sha256",
            "local bytes changed since review",
        )
        info = _object(_json(root / "meta/info.json"), "meta/info.json")
        _check(info.get("codebase_version") == "v3.0", "codebase_version", "expected LeRobot v3.0")
        for key in ("total_frames", "total_episodes", "total_tasks", "chunks_size"):
            _check(_integer(info.get(key), 1), key, "expected a positive integer")
        _check(
            info["total_episodes"] <= info["total_frames"], "total_episodes", "exceeds total_frames"
        )
        fps = info.get("fps")
        _check(
            type(fps) in (float, int) and math.isfinite(fps) and fps > 0,
            "fps",
            "expected a finite positive rate",
        )
        _check(
            isinstance(info.get("robot_type"), str) or info.get("robot_type") is None,
            "robot_type",
            "expected string or null",
        )
        _check(
            info.get("data_path") == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "data_path",
            "expected the v3 shard layout",
        )
        splits = info.get("splits")
        _check(isinstance(splits, dict) and bool(splits), "splits", "expected named episode ranges")
        for key, span in splits.items():
            try:
                start, end = map(int, span.split(":"))
            except (AttributeError, ValueError):
                _check(False, f"splits.{key}", "expected start:end")
            _check(
                0 <= start < end <= info["total_episodes"],
                f"splits.{key}",
                "episode range out of bounds",
            )
        validate_features(info, expected_features)
        stats = _object(_json(root / "meta/stats.json"), "meta/stats.json")
        from tether.finetune.local_dataset import _shape

        for key in ("action", "observation.state"):
            moments = _object(stats.get(key), f"meta/stats.json.{key}")
            for moment in ("min", "max", "mean", "std"):
                _shape(
                    moments.get(moment),
                    info["features"][key]["shape"],
                    f"meta/stats.json.{key}.{moment}",
                )
        tasks = list(_rows([root / "meta/tasks.parquet"], "task_index"))
        _check(len(tasks) == info["total_tasks"], "total_tasks", "differs from tasks.parquet")
        episodes = list(_rows(sorted((root / "meta/episodes").rglob("*.parquet")), "episode_index"))
        _check(
            len(episodes) == info["total_episodes"], "total_episodes", "differs from episode index"
        )
        offset = 0
        video_timestamps: dict[str, tuple[list[float], list[float]]] = {}
        video_queries: list[tuple[int, str, float, list[float]]] = []
        for i, episode in enumerate(episodes):
            _check(
                type(episode.get("episode_index")) is int and episode["episode_index"] == i,
                "episode_index",
                f"expected contiguous episode {i}",
            )
            length = episode.get("length")
            _check(
                _integer(length, 1), f"episode_index[{i}].length", "expected positive frame count"
            )
            _check(
                episode.get("dataset_from_index") == offset
                and episode.get("dataset_to_index") == offset + length,
                f"episode_index[{i}].dataset_from_index/dataset_to_index",
                "gapped or inconsistent frame range",
            )
            for field in ("data/chunk_index", "data/file_index"):
                _check(
                    _integer(episode.get(field)),
                    f"episode_index[{i}].{field}",
                    "expected shard index",
                )
            shard = info["data_path"].format(
                chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
            )
            _check(
                metadata_only or (root / shard).is_file(),
                f"episode_index[{i}].data/file_index",
                "missing local shard",
            )
            for camera, feature in info["features"].items():
                if feature["dtype"] != "video":
                    continue
                _check(
                    info.get("video_path")
                    == "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                    "video_path",
                    "expected the v3 video layout",
                )
                values = {
                    suffix: episode.get(f"videos/{camera}/{suffix}")
                    for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
                }
                _check(
                    _integer(values["chunk_index"]) and _integer(values["file_index"]),
                    f"episode_index[{i}].videos/{camera}",
                    "missing video shard indexes",
                )
                _check(
                    all(
                        type(values[k]) in (int, float) and math.isfinite(values[k])
                        for k in ("from_timestamp", "to_timestamp")
                    )
                    and 0 <= values["from_timestamp"] < values["to_timestamp"],
                    f"episode_index[{i}].videos/{camera}/from_timestamp",
                    "invalid video range",
                )
                video = info["video_path"].format(
                    video_key=camera,
                    chunk_index=values["chunk_index"],
                    file_index=values["file_index"],
                )
                _check(
                    metadata_only or (root / video).is_file(),
                    f"episode_index[{i}].videos/{camera}/file_index",
                    "missing local video",
                )
                if not metadata_only:
                    field = f"episode_index[{i}].videos/{camera}/to_timestamp"
                    tolerance_s = LEROBOT_TOLERANCE_S
                    start, end = values["from_timestamp"], values["to_timestamp"]
                    _check(abs(end - start - length / fps) < tolerance_s, field,
                           "video duration differs from episode length/fps")
                    if video not in video_timestamps:
                        video_timestamps[video] = _video_timestamps(root / video, field)
                    timestamps, frame_ends = video_timestamps[video]
                    lo = bisect_left(timestamps, start - tolerance_s)
                    hi = bisect_left(timestamps, end - tolerance_s)
                    _check(hi - lo == length, field,
                           "decoded video frame count differs from episode length")
                    _check(abs(frame_ends[hi - 1] - end) < tolerance_s, field,
                           "decoded video duration differs from episode length/fps")
                    # LeRobot queries from_timestamp + the row's own timestamp, so the
                    # per-frame check runs after the row scan below.
                    video_queries.append((i, camera, start, timestamps))
            offset += length
        _check(offset == info["total_frames"], "total_frames", "differs from episode lengths")
        if metadata_only:
            return {"kind": "local-lerobot-v3", "dataset_root": str(root), "info": info,
                    "metadata_checked": True}
        retained_paths = {r["path"] for r in manifest}
        count, episode_id, frame = 0, 0, 0
        row_timestamps: list[list[float]] = [[] for _ in episodes]
        for path in sorted((root / "data").rglob("*.parquet")):
            for row in _rows([path], "data"):
                _check(episode_id < len(episodes), "index", "unexpected extra frame")
                episode = episodes[episode_id]
                shard = info["data_path"].format(
                    chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
                )
                _check(
                    path.relative_to(root).as_posix() == shard,
                    "data/file_index",
                    "frame is in a different shard from its episode index",
                )
                for key, value in (
                    ("index", count),
                    ("episode_index", episode_id),
                    ("frame_index", frame),
                ):
                    _check(
                        type(row.get(key)) is int and row[key] == value,
                        key,
                        f"expected {value} at frame {count}",
                    )
                _check(
                    _integer(row.get("task_index")) and row["task_index"] < info["total_tasks"],
                    "task_index",
                    f"out of bounds at frame {count}",
                )
                timestamp = row.get("timestamp")
                _check(
                    type(timestamp) in (int, float)
                    and math.isfinite(timestamp)
                    and abs(timestamp - frame / fps) <= 1e-4,
                    "timestamp",
                    f"inconsistent with fps at frame {count}",
                )
                row_timestamps[episode_id].append(timestamp)
                for key in ("action", "observation.state"):
                    value = row.get(key)
                    # LeRobot 0.5.1 stores shape [1] as a scalar Value, not a list
                    # (lerobot/datasets/feature_utils.py:54).
                    if info["features"][key]["shape"] == [1] and type(value) in (int, float):
                        value = [value]
                    _check(
                        isinstance(value, list)
                        and len(value) == info["features"][key]["shape"][0]
                        and all(type(n) in (int, float) and math.isfinite(n) for n in value),
                        f"features.{key}.shape",
                        f"invalid vector at frame {count}",
                    )
                for key, feature in info["features"].items():
                    if feature["dtype"] == "image":
                        image = row.get(key)
                        _check(
                            isinstance(image, dict)
                            and bool(image.get("bytes") or image.get("path")),
                            f"features.{key}",
                            f"missing image at frame {count}",
                        )
                        if image.get("path") and not image.get("bytes"):
                            image_path = Path(image["path"])
                            _check(
                                not image_path.is_absolute()
                                and ".." not in image_path.parts
                                and image_path.as_posix() in retained_paths
                                and (root / image_path).is_file()
                                and not (root / image_path).is_symlink(),
                                f"features.{key}.path",
                                "missing/unsafe local image",
                            )
                count += 1
                frame += 1
                if frame == episode["length"]:
                    episode_id += 1
                    frame = 0
        _check(count == info["total_frames"], "total_frames", "differs from local data rows")
        for i, camera, start, timestamps in video_queries:
            for frame, row_ts in enumerate(row_timestamps[i]):
                query = start + row_ts
                index = bisect_left(timestamps, query)
                neighbors = timestamps[max(0, index - 1):index + 1]
                _check(
                    any(_within_lerobot_tolerance(query, t) for t in neighbors),
                    f"episode_index[{i}].videos/{camera}/from_timestamp",
                    f"decoded frame {frame} violates LeRobot tolerance_s={LEROBOT_TOLERANCE_S}",
                )
        return {
            "kind": "local-lerobot-v3",
            "dataset_root": str(root),
            "info": info,
            "content_sha256": content,
            "artifacts": manifest,
            "size_bytes": sum(r["size_bytes"] for r in manifest),
        }
    except LocalDatasetError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise LocalDatasetError(
            "local-v3-invalid", f"meta/info.json or episode_index: {exc}"
        ) from exc


# LeRobot's TrainPipelineConfig.tolerance_s default (configs/train.py). Local
# configs refuse a tolerance_s override (local_dataset.validate_local_config).
LEROBOT_TOLERANCE_S = 1e-4


def _within_lerobot_tolerance(query: float, pts: float) -> bool:
    """LeRobot's decode rule: torch.cdist(p=1) on default float32 tensors, then
    `min_ < tolerance_s` (video_utils.py). The float32 rounding matters for long
    concatenated shards, where float64 gives the opposite answer near the edge."""
    import numpy as np

    return bool(np.abs(np.float32(query) - np.float32(pts)) < np.float32(LEROBOT_TOLERANCE_S))


def _video_timestamps(path: Path, field: str) -> tuple[list[float], list[float]]:
    """Decode each referenced shard once, including PTS, during full admission."""
    try:
        import av

        with av.open(str(path)) as container:
            timestamps, frame_ends = [], []
            for frame in container.decode(video=0):
                _check(frame.pts is not None and frame.duration > 0, field,
                       "video frame has no timestamp/duration")
                timestamps.append(float(frame.pts * frame.time_base))
                frame_ends.append(float((frame.pts + frame.duration) * frame.time_base))
        _check(bool(timestamps) and all(math.isfinite(t) for t in timestamps)
               and all(a < b for a, b in zip(timestamps, timestamps[1:])),
               field, "video has no frames or invalid decoded timestamps")
        return timestamps, frame_ends
    except LocalDatasetError:
        raise
    except Exception as exc:
        raise LocalDatasetError("local-v3-invalid", f"{field}: cannot decode local video: {exc}") from exc


def describe_lerobot_v3(root: Path, *, expected_features: dict | None = None) -> dict:
    """Check only meta/ for listing; this is not content admission or identity."""
    return _inspect_lerobot_v3(root, expected_features=expected_features, metadata_only=True)


def verify_lerobot_v3(
    root: Path, *, expected_features: dict | None = None, expected_sha256: str | None = None
) -> dict:
    """Validate metadata, all frames and video timestamps, and hash retained bytes."""
    return _inspect_lerobot_v3(root, expected_features=expected_features, expected_sha256=expected_sha256)


def verify_local_dataset(cfg: Any) -> dict:
    """Explicit receipt flag retains the old export contract; raw v3 never downgrades it."""
    from tether.finetune.local_dataset import verify_local_export

    if cfg.dataset_manifest_sha256:
        verified = verify_local_export(cfg.dataset_root, cfg.dataset_manifest_sha256)
        manifest = [
            {**row, "path": row["path"].removeprefix("dataset/")}
            for row in verified["receipt"]["artifacts"]
            if row["path"].startswith("dataset/")
        ]
        manifest.sort(key=lambda row: row["path"])
        return {
            **verified,
            "kind": "local-lerobot-export",
            "content_sha256": _hash(manifest),
            "artifacts": manifest,
            "size_bytes": sum(row["size_bytes"] for row in manifest),
        }
    return verify_lerobot_v3(
        cfg.dataset_root,
        expected_features=cfg.dataset_features,
        expected_sha256=cfg.dataset_content_sha256,
    )
