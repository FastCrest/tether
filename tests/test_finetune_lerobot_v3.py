"""Small real-parquet v3 inputs, with all dataset networking forbidden."""

import io
import json
import socket
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import pytest

from tether.finetune.config import FinetuneConfig
from tether.finetune.lerobot_v3 import describe_lerobot_v3, verify_lerobot_v3
from tether.finetune.local_dataset import LocalDatasetError
from tether.finetune.preflight import run_preflight
from tether.finetune.run import _build_lerobot_command, run_finetune


def write_v3(root, action_dim=6):
    root = Path(root)
    features = {
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
        "observation.state": {"dtype": "float32", "shape": [6], "names": None},
        "observation.images.wrist": {
            "dtype": "image",
            "shape": [32, 32, 3],
            "names": ["height", "width", "channels"],
        },
    }
    for key in ("timestamp", "index", "frame_index", "episode_index", "task_index"):
        features[key] = {
            "dtype": "float32" if key == "timestamp" else "int64",
            "shape": [1],
            "names": None,
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so101",
        "total_frames": 9,
        "total_episodes": 3,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": 10,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": features,
    }
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info))
    stats = {}
    for key, feature in features.items():
        shape = [3, 1, 1] if feature["dtype"] == "image" else feature["shape"]

        def full(value, shape):
            return [full(value, shape[1:]) for _ in range(shape[0])] if shape else value

        stats[key] = {
            "min": full(0.0, shape),
            "max": full(1.0, shape),
            "mean": full(0.5, shape),
            "std": full(0.25, shape),
            "count": [9],
        }
    (root / "meta/stats.json").write_text(json.dumps(stats))
    pq.write_table(
        pa.table({"task_index": [0], "task": ["move cube"]}), root / "meta/tasks.parquet"
    )
    episodes = [
        {
            "episode_index": i,
            "length": 3,
            "tasks": ["move cube"],
            "dataset_from_index": i * 3,
            "dataset_to_index": (i + 1) * 3,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for i in range(3)
    ]
    pq.write_table(
        pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet"
    )
    image = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 40, 60)).save(image, format="PNG")
    data = [
        {
            "index": i,
            "episode_index": i // 3,
            "frame_index": i % 3,
            "task_index": 0,
            "timestamp": (i % 3) / 10,
            # LeRobot 0.5.1 stores shape [1] as a scalar (feature_utils.py:54).
            "action": 0.1 if action_dim == 1 else [0.1] * action_dim,
            "observation.state": [0.2] * 6,
            "observation.images.wrist": {"bytes": image.getvalue(), "path": None},
        }
        for i in range(9)
    ]
    table = pa.Table.from_pylist(data)
    for key in ("action", "observation.state", "timestamp"):
        table = table.set_column(
            table.schema.get_field_index(key),
            key,
            table[key].cast(
                pa.float32()
                if key == "timestamp" or (key == "action" and action_dim == 1)
                else pa.list_(pa.float32())
            ),
        )
    pq.write_table(table, root / "data/chunk-000/file-000.parquet")
    return root


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **kw: pytest.fail("dataset network access")
    )
    return FinetuneConfig(
        base="",
        dataset="studio-local/fixture",
        output=tmp_path / "run",
        policy="act",
        mode="full",
        dataset_root=write_v3(tmp_path / "local"),
        dataset_features={"action": {"shape": [6]}, "observation.state": {"shape": [6]}},
        num_steps=2,
        skip_export=True,
    )


def test_offline_schema_command_and_identity(cfg):
    verified = verify_lerobot_v3(cfg.dataset_root, expected_features=cfg.dataset_features)
    assert len(verified["content_sha256"]) == 64
    assert not run_preflight(cfg).has_failures
    assert f"--dataset.root={cfg.dataset_root}" in _build_lerobot_command(cfg)
    result = run_finetune(replace(cfg, dry_run=True))
    assert result.status == "ok"
    record = json.loads((cfg.output / "training-dataset.json").read_text())
    assert record["content_sha256"] == verified["content_sha256"]
    assert {r["path"] for r in record["artifacts"]} >= {
        "meta/info.json",
        "data/chunk-000/file-000.parquet",
    }


def test_seven_actions_refused(cfg):
    root = write_v3(cfg.output.parent / "seven", 7)
    cfg = replace(cfg, dataset_root=root)
    report = run_preflight(cfg)
    assert report.has_failures
    assert "features.action.shape" in report.error_message()
    assert run_finetune(cfg).status == "aborted"


@pytest.mark.parametrize(
    "field", ["episode_index", "frame_index", "index", "task_index", "timestamp"]
)
def test_corrupt_row_index_refused(cfg, field):
    path = cfg.dataset_root / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[1][field] = 999
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(LocalDatasetError, match=field):
        verify_lerobot_v3(cfg.dataset_root)


def test_content_changes_refused(cfg):
    identity = verify_lerobot_v3(cfg.dataset_root)["content_sha256"]
    (cfg.dataset_root / "meta/stats.json").write_text("{}")
    with pytest.raises(LocalDatasetError, match="dataset_content_sha256"):
        verify_lerobot_v3(cfg.dataset_root, expected_sha256=identity)


def test_camera_contract_refused(cfg):
    with pytest.raises(LocalDatasetError, match="features.observation.images.front"):
        verify_lerobot_v3(
            cfg.dataset_root, expected_features={"observation.images.front": {"shape": [32, 32, 3]}}
        )


def test_smolvla_local_features_and_padding(cfg):
    base = cfg.output.parent / "smolvla-base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            {
                "type": "smolvla",
                "max_state_dim": 32,
                "max_action_dim": 32,
                "input_features": {},
                "output_features": {},
            }
        )
    )
    cfg = replace(cfg, base=str(base), policy="auto", mode="lora", dataset_features={})
    assert not run_preflight(cfg).has_failures
    assert f"--dataset.root={cfg.dataset_root}" in _build_lerobot_command(cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("codebase_version", "v2.1"),
        ("fps", 0),
        ("total_episodes", 4),
        ("total_frames", 10),
        ("data_path", "../outside.parquet"),
    ],
)
def test_info_fields_named_in_refusal(cfg, field, value):
    path = cfg.dataset_root / "meta/info.json"
    info = json.loads(path.read_text())
    info[field] = value
    path.write_text(json.dumps(info))
    with pytest.raises(LocalDatasetError, match=field):
        verify_lerobot_v3(cfg.dataset_root)


@pytest.mark.parametrize(
    "field,value",
    [("episode_index", 0), ("length", 4), ("dataset_from_index", 2), ("data/file_index", 9)],
)
def test_episode_index_integrity_refused(cfg, field, value):
    path = cfg.dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[1][field] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(LocalDatasetError, match=field if field != "length" else "dataset_to_index"):
        verify_lerobot_v3(cfg.dataset_root)


def test_public_cli_admits_raw_v3_and_binds_reviewed_identity(cfg):
    import typer
    from typer.testing import CliRunner
    from tether.finetune.cli import finetune_command

    app = typer.Typer()
    app.command()(finetune_command)
    identity = verify_lerobot_v3(cfg.dataset_root)["content_sha256"]
    result = CliRunner().invoke(app, ["--policy", "act", "--mode", "full", "--dataset", cfg.dataset,
        "--dataset-root", str(cfg.dataset_root), "--dataset-content-sha256", identity,
        "--dataset-features", json.dumps(cfg.dataset_features), "--output", str(cfg.output), "--steps", "2", "--dry-run"])
    assert result.exit_code == 0, result.output
    record = json.loads((cfg.output/"training-dataset.json").read_text())
    assert record["content_sha256"] == identity


def test_width_one_action_stored_as_scalar_admitted(cfg):
    root = write_v3(cfg.output.parent / "width-one", 1)
    verified = verify_lerobot_v3(root, expected_features={"action": {"shape": [1]}})
    assert verified["info"]["features"]["action"]["shape"] == [1]


def test_smolvla_base_embodiment_does_not_bind_local_features(cfg):
    """Real lerobot/smolvla_base features (camera1-3 at 256, 6-D) are cleared by
    --policy.input_features=null; only max_*_dim padding binds."""
    base = cfg.output.parent / "smolvla-real-base"
    base.mkdir()
    real = {
        "type": "smolvla",
        "max_state_dim": 32,
        "max_action_dim": 32,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            **{f"observation.images.camera{i}": {"type": "VISUAL", "shape": [3, 256, 256]} for i in (1, 2, 3)},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }
    (base / "config.json").write_text(json.dumps(real))
    seven = write_v3(cfg.output.parent / "seven-smolvla", 7)
    cfg = replace(cfg, base=str(base), policy="auto", mode="lora", dataset_features={}, dataset_root=seven)
    assert not run_preflight(cfg).has_failures
    cmd = _build_lerobot_command(cfg)
    assert "--policy.input_features=null" in cmd and f"--dataset.root={seven}" in cmd
    real["max_action_dim"] = 6
    (base / "config.json").write_text(json.dumps(real))
    report = run_preflight(cfg)
    assert report.has_failures and "features.action.shape" in report.error_message()


def write_video_v3(root, *, frames=9, timestamp_shift=0.0, long_last_frame=False, rate=10):
    """One shared, real mp4 shard; episodes refer to disjoint timestamp ranges."""
    av = pytest.importorskip("av")
    import numpy as np

    root = write_v3(root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.wrist"]["dtype"] = "video"
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info_path.write_text(json.dumps(info))
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episode_path).to_pylist()
    for i, episode in enumerate(episodes):
        prefix = "videos/observation.images.wrist/"
        episode.update({prefix + "chunk_index": 0, prefix + "file_index": 0,
                        prefix + "from_timestamp": i * .3 + timestamp_shift,
                        prefix + "to_timestamp": (i + 1) * .3 + timestamp_shift})
    pq.write_table(pa.Table.from_pylist(episodes), episode_path)
    video = root / "videos/observation.images.wrist/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        packets = []
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 10, np.uint8), format="rgb24")
            packets.extend(stream.encode(frame))
        packets.extend(stream.encode())
        if long_last_frame:
            packets[-1].duration = 2  # 0.2s at the encoder's 1/10 time base.
        for packet in packets:
            container.mux(packet)
    return root


def test_metadata_listing_never_reads_content(cfg, monkeypatch):
    import tether.finetune.lerobot_v3 as v3

    original = v3._rows
    def meta_rows(paths, label):
        assert all(path.is_relative_to(cfg.dataset_root / "meta") for path in paths)
        return original(paths, label)
    monkeypatch.setattr(v3, "_rows", meta_rows)
    monkeypatch.setattr(v3, "_file_hash", lambda *a: pytest.fail("listing hashes a file"))
    monkeypatch.setattr(v3, "_video_timestamps", lambda *a: pytest.fail("listing reads video"))
    described = describe_lerobot_v3(cfg.dataset_root)
    assert described["metadata_checked"]
    assert described["info"]["total_episodes"] == 3
    assert "content_sha256" not in described


@pytest.mark.parametrize("frames", [1, 8])
def test_short_video_refused_with_field(tmp_path, frames):
    root = write_video_v3(tmp_path / "short", frames=frames)
    with pytest.raises(LocalDatasetError, match=r"episode_index\[.*\].videos/observation.images.wrist/to_timestamp"):
        verify_lerobot_v3(root)


@pytest.mark.parametrize("shift,valid", [(0.0, True), (0.00009, True), (0.0001, False), (0.00011, False)])
def test_video_uses_lerobot_timestamp_tolerance(tmp_path, shift, valid):
    root = write_video_v3(tmp_path / "video", timestamp_shift=shift)
    if valid:
        assert verify_lerobot_v3(root)["content_sha256"]
    else:
        with pytest.raises(LocalDatasetError, match="videos/observation.images.wrist"):
            verify_lerobot_v3(root)


def test_video_metadata_duration_refused(tmp_path):
    root = write_video_v3(tmp_path / "video")
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["videos/observation.images.wrist/to_timestamp"] = .2
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(LocalDatasetError, match="videos/observation.images.wrist/to_timestamp"):
        verify_lerobot_v3(root)


def test_metadata_listing_refuses_symlink_into_data(cfg, monkeypatch):
    import tether.finetune.lerobot_v3 as v3
    path = cfg.dataset_root / "meta/info.json"
    path.unlink()
    path.symlink_to(cfg.dataset_root / "data/chunk-000/file-000.parquet")
    monkeypatch.setattr(v3, "_json", lambda *a: pytest.fail("listing read symlink target"))
    with pytest.raises(LocalDatasetError, match="meta/info.json"):
        describe_lerobot_v3(cfg.dataset_root)


def test_decoded_video_duration_refused(tmp_path):
    root = write_video_v3(tmp_path / "video", long_last_frame=True)
    with pytest.raises(LocalDatasetError, match="videos/observation.images.wrist/to_timestamp.*duration"):
        verify_lerobot_v3(root)


def test_video_tolerance_uses_lerobot_float32_rule():
    """LeRobot compares torch float32 distances; float64 disagrees on long shards."""
    from tether.finetune.lerobot_v3 import _within_lerobot_tolerance

    # Pairs checked against torch.cdist in LeRobot's runtime. Near 2000 s one
    # float32 step is ~1.2e-4, so float64 gives the opposite answer for both.
    assert abs(2000.00001 - 2000.000105) < 1e-4
    assert not _within_lerobot_tolerance(2000.00001, 2000.000105)
    assert abs(2000.00007 - 2000.000175) >= 1e-4
    assert _within_lerobot_tolerance(2000.00007, 2000.000175)
    assert _within_lerobot_tolerance(0.4, 0.40009)
    assert not _within_lerobot_tolerance(0.4, 0.40011)


def test_video_query_uses_row_timestamp(tmp_path):
    """LeRobot queries from_timestamp + the row timestamp, not frame/fps."""
    root = write_video_v3(tmp_path / "video", timestamp_shift=0.00006)
    assert verify_lerobot_v3(root)["content_sha256"]
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    # Each row stays within the row-vs-fps bound (1e-4), but the sum with the
    # from_timestamp shift breaks LeRobot's video tolerance.
    shifted = pa.array([t + 0.00006 for t in table["timestamp"].to_pylist()], pa.float32())
    pq.write_table(table.set_column(table.schema.get_field_index("timestamp"), "timestamp", shifted), path)
    with pytest.raises(LocalDatasetError, match=r"episode_index\[0\].videos/observation.images.wrist/from_timestamp"):
        verify_lerobot_v3(root)


def test_extra_decoded_frames_refused(tmp_path):
    """Every query has a frame within tolerance, but the shard holds 2x the frames."""
    root = write_video_v3(tmp_path / "video", frames=18, rate=20)
    with pytest.raises(LocalDatasetError, match="to_timestamp: decoded video frame count"):
        verify_lerobot_v3(root)


def test_metadata_duration_refused_when_decode_agrees(tmp_path):
    """A long last frame makes decode end match a stretched to_timestamp."""
    root = write_video_v3(tmp_path / "video", long_last_frame=True)
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[2]["videos/observation.images.wrist/to_timestamp"] = 1.0
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(LocalDatasetError, match=r"episode_index\[2\].*to_timestamp: video duration differs"):
        verify_lerobot_v3(root)
