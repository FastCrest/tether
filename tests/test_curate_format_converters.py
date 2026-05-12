"""Tests for src/reflex/curate/format_converters/ — 4 converters."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflex.curate.format_converters import (
    CONVERTER_REGISTRY,
    EMBODIMENT_OXE_MAP,
    HDF5Converter,
    LeRobotV3Converter,
    OpenXEmbodimentConverter,
    RLDSConverter,
    get_converter,
)


def _seed_jsonl(path: Path, *, episodes: int = 2, steps_per_chunk: int = 5,
                rows_per_episode: int = 20) -> None:
    with open(path, "w") as f:
        for ep in range(episodes):
            for step in range(rows_per_episode):
                row = {
                    "kind": "request",
                    "schema_version": 1,
                    "seq": step,
                    "chunk_id": step,
                    "timestamp": f"2026-05-05T{step:02d}:00:00Z",
                    "episode_id": f"ep_{ep:02d}",
                    "instruction_raw": f"Pick up block {ep}",
                    "state_vec": [0.1 * step, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
                    "action_chunk": [
                        [float(i) * 0.01 + step, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
                        for i in range(steps_per_chunk)
                    ],
                    "metadata": {
                        "contributor_id": "test_001",
                        "quality_score": 0.85 if ep == 0 else 0.45,
                        "is_canonical": True if ep == 0 else False,
                    },
                }
                f.write(json.dumps(row) + "\n")


# ── registry + factory ─────────────────────────────────────────────────────


def test_registry_has_all_4_formats() -> None:
    expected = {"lerobot-v3", "hdf5", "rlds", "openx-embodiment"}
    assert set(CONVERTER_REGISTRY.keys()) == expected


def test_get_converter_lerobot_v3() -> None:
    c = get_converter("lerobot-v3")
    assert isinstance(c, LeRobotV3Converter)


def test_get_converter_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown format"):
        get_converter("not-a-format")


# ── LeRobot v3 ─────────────────────────────────────────────────────────────


def test_lerobot_v3_basic_convert(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_lerobot"

    converter = LeRobotV3Converter(robot_type="franka", fps=30)
    result = converter.convert(input_jsonl=jsonl, output_dir=out)
    assert result.episode_count == 2
    assert result.step_count == 200  # 2 ep × 20 rows × 5 steps_per_chunk
    assert (out / "meta" / "info.json").exists()
    assert (out / "meta" / "tasks.jsonl").exists()
    assert (out / "meta" / "episodes.jsonl").exists()
    assert (out / "README.md").exists()
    assert (out / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert (out / "data" / "chunk-000" / "episode_000001.parquet").exists()


def test_lerobot_v3_info_json_shape(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_info"

    converter = LeRobotV3Converter(robot_type="franka", fps=30)
    converter.convert(input_jsonl=jsonl, output_dir=out)
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["robot_type"] == "franka"
    assert info["fps"] == 30
    assert "action" in info["features"]
    assert "observation.state" in info["features"]


def test_lerobot_v3_min_quality_filters(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_quality"

    converter = LeRobotV3Converter()
    result = converter.convert(input_jsonl=jsonl, output_dir=out, min_quality=0.6)
    # ep_00 has quality 0.85; ep_01 has 0.45 → only 1 ep passes
    assert result.episode_count == 1
    assert result.skipped_episodes == 1


def test_lerobot_v3_canonical_only_filters(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_canonical"

    converter = LeRobotV3Converter()
    result = converter.convert(input_jsonl=jsonl, output_dir=out, canonical_only=True)
    # ep_00 is_canonical=True; ep_01 is_canonical=False
    assert result.episode_count == 1


def test_lerobot_v3_emits_videos_skipped_warning_for_hash_only(tmp_path: Path) -> None:
    """JSONL with hash-only image_b64 → converter skips video encoding + warns."""
    pytest.importorskip("pyarrow")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_warn"

    result = LeRobotV3Converter().convert(input_jsonl=jsonl, output_dir=out)
    # Either warning shape is acceptable: missing image bytes (hash-only seed)
    # OR the [curate-video] extra not installed in the test env.
    has_skip_warn = any(
        "videos_skipped" in w or "video_encoder_unavailable" in w
        for w in result.warnings
    )
    assert has_skip_warn


def test_lerobot_v3_disable_videos_no_warning(tmp_path: Path) -> None:
    """encode_videos=False → no videos warning at all."""
    pytest.importorskip("pyarrow")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_no_video"

    result = LeRobotV3Converter(encode_videos=False).convert(
        input_jsonl=jsonl, output_dir=out,
    )
    assert not any("videos_skipped" in w for w in result.warnings)


# ── HDF5 ───────────────────────────────────────────────────────────────────


def test_hdf5_basic_convert(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_hdf5"

    converter = HDF5Converter(dataset_id="test_ds")
    result = converter.convert(input_jsonl=jsonl, output_dir=out)
    assert result.episode_count == 2
    h5_path = out / "test_ds.h5"
    assert h5_path.exists()
    with h5py.File(h5_path, "r") as h5:
        assert "meta" in h5
        assert "episodes" in h5
        assert h5["meta"].attrs["num_episodes"] == 2
        assert "episode_000000" in h5["episodes"]
        assert "episode_000001" in h5["episodes"]
        ep0 = h5["episodes/episode_000000"]
        assert "action" in ep0
        assert "observation/state" in ep0


def test_hdf5_split_episodes(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_split"

    converter = HDF5Converter(split_episodes=True)
    result = converter.convert(input_jsonl=jsonl, output_dir=out)
    assert result.episode_count == 2
    h5_files = sorted(out.glob("*.h5"))
    assert len(h5_files) == 2


def test_hdf5_compression_options() -> None:
    HDF5Converter(compression="lzf")
    HDF5Converter(compression="gzip")
    HDF5Converter(compression=None)
    with pytest.raises(ValueError, match="compression"):
        HDF5Converter(compression="bogus")


# ── RLDS skeleton ──────────────────────────────────────────────────────────


def test_rlds_skeleton_raises_clear_install_message(tmp_path: Path) -> None:
    """Without tensorflow_datasets, RLDSConverter raises ImportError with hint."""
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)

    converter = RLDSConverter()
    try:
        import tensorflow_datasets  # noqa: F401
        # If installed, expect NotImplementedError instead (Phase 1 skeleton).
        with pytest.raises(NotImplementedError, match="skeleton"):
            converter.convert(input_jsonl=jsonl, output_dir=tmp_path / "out")
    except ImportError:
        with pytest.raises(ImportError, match="curate-rlds"):
            converter.convert(input_jsonl=jsonl, output_dir=tmp_path / "out")


# ── OpenX-Embodiment ───────────────────────────────────────────────────────


def test_oxe_embodiment_map_has_canonical_entries() -> None:
    assert EMBODIMENT_OXE_MAP["franka"] == "franka_emika_panda"
    assert "ur5" in EMBODIMENT_OXE_MAP


def test_oxe_skeleton_raises_install_message(tmp_path: Path) -> None:
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)

    converter = OpenXEmbodimentConverter(embodiment="franka")
    assert converter.oxe_embodiment == "franka_emika_panda"
    try:
        import tensorflow_datasets  # noqa: F401
        with pytest.raises(NotImplementedError):
            converter.convert(input_jsonl=jsonl, output_dir=tmp_path / "out")
    except ImportError:
        with pytest.raises(ImportError, match="curate-rlds"):
            converter.convert(input_jsonl=jsonl, output_dir=tmp_path / "out")


def test_oxe_unknown_embodiment_falls_back() -> None:
    converter = OpenXEmbodimentConverter(embodiment="ufo_arm")
    assert converter.oxe_embodiment == "unknown"


# ── RLDS full impl (gated on [curate-rlds] extra) ──────────────────────────


def test_rlds_full_round_trip(tmp_path: Path) -> None:
    """End-to-end RLDS → tfrecord write + parse-back via tf.data."""
    tf = pytest.importorskip("tensorflow")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_rlds"

    converter = RLDSConverter(dataset_name="reflex_test", shard_size=10)
    result = converter.convert(input_jsonl=jsonl, output_dir=out)
    assert result.episode_count == 2
    assert (out / "dataset_info.json").exists()
    assert (out / "features.json").exists()
    # At least one TFRecord shard
    tfrecord_files = list(out.glob("reflex_test-train.tfrecord-*"))
    assert len(tfrecord_files) >= 1

    # Parse back: count examples in the shard.
    ds = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])
    total = sum(1 for _ in ds)
    assert total == 2  # 2 episodes


def test_rlds_dataset_info_schema(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_rlds_info"

    RLDSConverter().convert(input_jsonl=jsonl, output_dir=out)
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["name"] == "reflex_curate_dataset"
    assert info["splits"][0]["num_examples"] == 2
    assert "steps" in info["features"]
    assert "action" in info["features"]["steps"]["feature_spec"]


def test_oxe_full_emits_embodiment_id(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    jsonl = tmp_path / "input.jsonl"
    _seed_jsonl(jsonl)
    out = tmp_path / "out_oxe"

    converter = OpenXEmbodimentConverter(embodiment="franka")
    result = converter.convert(input_jsonl=jsonl, output_dir=out)
    assert result.episode_count == 2

    info = json.loads((out / "dataset_info.json").read_text())
    assert info["oxe_embodiment"] == "franka_emika_panda"
    assert info["format_version"] == "openx-embodiment-1.0"
    assert "embodiment_id" in info["features"]["episode_metadata"]["feature_spec"]


# ── Video encoder (gated on [curate-video] extra) ──────────────────────────


def test_video_encoder_handles_no_frames() -> None:
    from reflex.curate.format_converters.shared.video_encoder import (
        encode_frames_to_mp4,
    )
    with pytest.raises(ValueError, match="no frames"):
        encode_frames_to_mp4(frames=[], output_path="/tmp/x.mp4")


def test_video_encoder_decode_image_field_hash_only() -> None:
    """64-char SHA-256 hex shouldn't be misinterpreted as base64 image."""
    from reflex.curate.format_converters.shared.video_encoder import decode_image_field
    sha256_hex = "0" * 64
    assert decode_image_field(sha256_hex) is None
    assert decode_image_field(None) is None
    assert decode_image_field("") is None


def test_video_encoder_collect_frames_skips_undecodable() -> None:
    from reflex.curate.format_converters.shared.video_encoder import collect_frames_from_rows
    rows = [
        {"image_b64": None},
        {"image_b64": "abc"},  # too short
        {"image_b64": "0" * 64},  # hash-only
    ]
    assert collect_frames_from_rows(rows) == []


def test_video_encoder_writes_mp4(tmp_path: Path) -> None:
    """End-to-end: PNG frames in → mp4 out (requires [curate-video] + Pillow)."""
    pytest.importorskip("imageio_ffmpeg")
    Image = pytest.importorskip("PIL.Image")
    from reflex.curate.format_converters.shared.video_encoder import (
        encode_frames_to_mp4,
    )
    import io as _io

    # Generate 5 RGB png frames
    frames = []
    for shade in (50, 80, 110, 140, 170):
        img = Image.new("RGB", (32, 32), (shade, shade, shade))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        frames.append(buf.getvalue())

    out = tmp_path / "test.mp4"
    n_bytes = encode_frames_to_mp4(frames=frames, output_path=out, fps=10)
    assert n_bytes > 0
    assert out.exists()
    assert out.stat().st_size == n_bytes


# ── Empty / edge cases ─────────────────────────────────────────────────────


def test_lerobot_v3_no_episodes_warning(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    jsonl = tmp_path / "input.jsonl"
    jsonl.write_text("")  # empty
    out = tmp_path / "out_empty"

    result = LeRobotV3Converter().convert(input_jsonl=jsonl, output_dir=out)
    assert result.episode_count == 0
    assert any("no_episodes_passed_filter" in w for w in result.warnings)


def test_directory_input_resolution(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _seed_jsonl(in_dir / "a.jsonl", episodes=1)
    _seed_jsonl(in_dir / "b.jsonl", episodes=1)
    out = tmp_path / "out_dir"

    result = LeRobotV3Converter().convert(input_jsonl=in_dir, output_dir=out)
    # Both files have episode_id "ep_00" → they merge into one episode.
    # But the second seed creates separate ep_00; they'll concat under same id.
    assert result.episode_count == 1
