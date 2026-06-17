"""Tests for the LeRobot dataset validator (Phase 0.5 dataset-validator).

Verifies:
- Each registered check fires correctly on a synthetic LeRobot v3.0 dataset
- Each check returns BLOCKER on the corresponding intentional defect
- overall_decision reduces correctly (any blocker -> blocker; warn-vs-strict)
- format_human + format_json both contain the right markers
- run_all_checks composes cleanly without raising on edge inputs

Builds tiny synthetic datasets in tmp_path. Skips parquet-dependent checks
when pyarrow is missing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq

from tether.validation import (
    Decision,
    REGISTERED_CHECKS,
    format_human,
    format_json,
    overall_decision,
    run_all_checks,
)


# ---------- fixtures ----------

def _write_info(
    root: Path,
    *,
    action_dim: int = 7,
    state_dim: int = 8,
    fps: int = 30,
    total_episodes: int = 2,
    total_frames: int = 20,
    image_keys: list[str] | None = None,
    skip_keys: list[str] | None = None,
) -> None:
    skip_keys = skip_keys or []
    features: dict = {}
    if "action" not in skip_keys:
        features["action"] = {"dtype": "float32", "shape": [action_dim], "names": None}
    if "observation.state" not in skip_keys:
        features["observation.state"] = {"dtype": "float32", "shape": [state_dim], "names": None}
    if image_keys:
        for k in image_keys:
            features[f"observation.images.{k}"] = {"dtype": "video", "shape": [224, 224, 3]}
    features["timestamp"] = {"dtype": "float32", "shape": [1]}
    features["frame_index"] = {"dtype": "int64", "shape": [1]}
    features["episode_index"] = {"dtype": "int64", "shape": [1]}
    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "features": features,
    }
    if "codebase_version" in skip_keys:
        info.pop("codebase_version")
    if "fps" in skip_keys:
        info.pop("fps")
    if "features" in skip_keys:
        info.pop("features")
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))


def _write_data_parquet(
    root: Path,
    *,
    action_dim: int = 7,
    state_dim: int = 8,
    n_episodes: int = 2,
    n_frames_per_episode: int = 10,
    fps: int = 30,
    nan_in_action_idx: int | None = None,
    skip_action_col: bool = False,
    non_monotonic_at: int | None = None,
) -> None:
    rows = []
    base_dt = 1.0 / fps
    for ep in range(n_episodes):
        for f in range(n_frames_per_episode):
            row = {
                "observation.state": [0.0] * state_dim,
                "timestamp": float(f) * base_dt,
                "frame_index": f,
                "episode_index": ep,
            }
            if not skip_action_col:
                action = [0.0] * action_dim
                if nan_in_action_idx is not None and len(rows) == nan_in_action_idx:
                    action[0] = float("nan")
                row["action"] = action
            rows.append(row)
    if non_monotonic_at is not None and 0 < non_monotonic_at < len(rows):
        rows[non_monotonic_at]["timestamp"] = -1.0
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


def _write_episodes_parquet(root: Path, *, n_episodes: int = 2, length: int = 10) -> None:
    rows = [
        {"episode_index": i, "length": length, "task_index": 0}
        for i in range(n_episodes)
    ]
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


def _make_valid_dataset(tmp_path: Path, **kw) -> Path:
    root = tmp_path / "ds_valid"
    root.mkdir()
    _write_info(root, **kw)
    action_dim = kw.get("action_dim", 7)
    state_dim = kw.get("state_dim", 8)
    n_eps = kw.get("total_episodes", 2)
    fps = kw.get("fps", 30)
    n_frames_per = kw.get("total_frames", 20) // max(n_eps, 1)
    _write_data_parquet(
        root, action_dim=action_dim, state_dim=state_dim,
        n_episodes=n_eps, n_frames_per_episode=n_frames_per, fps=fps,
    )
    _write_episodes_parquet(root, n_episodes=n_eps, length=n_frames_per)
    return root


# ---------- happy path ----------

class TestValidDataset:
    def test_all_checks_pass_on_valid_dataset(self, tmp_path):
        root = _make_valid_dataset(tmp_path)
        results = run_all_checks(root)
        decision = overall_decision(results)
        assert decision == Decision.OK, f"got {decision}; results: {[(r.check_id, r.decision.value) for r in results]}"
        assert all(r.decision in (Decision.OK, Decision.SKIPPED) for r in results)

    def test_no_blockers_on_valid_dataset(self, tmp_path):
        root = _make_valid_dataset(tmp_path)
        results = run_all_checks(root)
        blockers = [r for r in results if r.decision == Decision.BLOCKER]
        assert blockers == [], f"unexpected blockers: {[(b.check_id, b.summary) for b in blockers]}"

    def test_run_all_checks_returns_one_per_registered(self, tmp_path):
        root = _make_valid_dataset(tmp_path)
        results = run_all_checks(root)
        assert len(results) == len(REGISTERED_CHECKS)


# ---------- broken paths (one defect per test) ----------

class TestBlockers:
    def test_missing_info_json(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        # data files but no meta/info.json
        _write_data_parquet(root)
        results = run_all_checks(root)
        info_check = next(r for r in results if r.check_id == "dataset.info-json-parseable")
        assert info_check.decision == Decision.BLOCKER
        assert overall_decision(results) == Decision.BLOCKER

    def test_corrupt_info_json(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        (root / "meta").mkdir()
        (root / "meta" / "info.json").write_text("{not valid json")
        results = run_all_checks(root)
        info_check = next(r for r in results if r.check_id == "dataset.info-json-parseable")
        assert info_check.decision == Decision.BLOCKER
        assert "JSONDecodeError" in info_check.details[0] or "Decode" in info_check.details[0]

    def test_info_json_missing_features_key(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        _write_info(root, skip_keys=["features"])
        _write_data_parquet(root)
        results = run_all_checks(root)
        info_check = next(r for r in results if r.check_id == "dataset.info-json-parseable")
        assert info_check.decision == Decision.BLOCKER

    def test_schema_missing_action_feature(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        _write_info(root, skip_keys=["action"])
        _write_data_parquet(root, skip_action_col=True)
        results = run_all_checks(root)
        schema_check = next(r for r in results if r.check_id == "dataset.schema-completeness")
        assert schema_check.decision == Decision.BLOCKER
        assert "action" in schema_check.summary

    def test_no_data_files_present(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        _write_info(root)
        # no data/ at all
        results = run_all_checks(root)
        df = next(r for r in results if r.check_id == "dataset.data-files-present")
        assert df.decision == Decision.BLOCKER

    def test_action_dim_mismatch_between_info_and_parquet(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        _write_info(root, action_dim=7)  # info declares 7
        _write_data_parquet(root, action_dim=6, n_episodes=1, n_frames_per_episode=5)  # parquet has 6
        # Re-write episodes to match
        _write_episodes_parquet(root, n_episodes=1, length=5)
        # Patch info to report 1 episode 5 frames so other checks don't bark
        info = json.loads((root / "meta" / "info.json").read_text())
        info["total_episodes"] = 1
        info["total_frames"] = 5
        (root / "meta" / "info.json").write_text(json.dumps(info))
        results = run_all_checks(root)
        shape_check = next(r for r in results if r.check_id == "dataset.shape-consistency")
        assert shape_check.decision == Decision.BLOCKER
        assert "7" in shape_check.summary and "6" in shape_check.summary

    def test_nan_in_action_blocks(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        _write_info(root, total_episodes=1, total_frames=5, action_dim=7)
        _write_data_parquet(
            root, action_dim=7, n_episodes=1, n_frames_per_episode=5,
            nan_in_action_idx=2,
        )
        _write_episodes_parquet(root, n_episodes=1, length=5)
        results = run_all_checks(root)
        nan_check = next(r for r in results if r.check_id == "dataset.action-finite")
        assert nan_check.decision == Decision.BLOCKER
        assert "NaN" in nan_check.summary or "non-finite" in nan_check.summary or "Inf" in nan_check.summary or "None" in nan_check.summary


class TestWarnings:
    def test_episode_count_mismatch_warns(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        _write_info(root, total_episodes=5, total_frames=50)  # info says 5
        _write_data_parquet(root, n_episodes=2, n_frames_per_episode=10)
        _write_episodes_parquet(root, n_episodes=2, length=10)  # only 2 actually present
        results = run_all_checks(root)
        ep_check = next(r for r in results if r.check_id == "dataset.episode-count-matches")
        assert ep_check.decision == Decision.WARN
        # And overall is WARN (not BLOCKER) since shape/finite checks should pass
        decision = overall_decision(results)
        assert decision == Decision.WARN

    def test_non_monotonic_timestamps_warns(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        _write_info(root, total_episodes=1, total_frames=10)
        _write_data_parquet(
            root, n_episodes=1, n_frames_per_episode=10, non_monotonic_at=5,
        )
        _write_episodes_parquet(root, n_episodes=1, length=10)
        results = run_all_checks(root)
        ts_check = next(r for r in results if r.check_id == "dataset.timing-monotonic")
        assert ts_check.decision == Decision.WARN

    def test_strict_mode_escalates_warn_to_blocker_overall(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        _write_info(root, total_episodes=5, total_frames=50)
        _write_data_parquet(root, n_episodes=2, n_frames_per_episode=10)
        _write_episodes_parquet(root, n_episodes=2, length=10)
        results = run_all_checks(root)
        # overall: WARN under default, BLOCKER under strict
        assert overall_decision(results, strict=False) == Decision.WARN
        assert overall_decision(results, strict=True) == Decision.BLOCKER


# ---------- embodiment cross-checks ----------

class TestEmbodimentCrossCheck:
    def test_embodiment_match_skipped_when_not_provided(self, tmp_path):
        root = _make_valid_dataset(tmp_path, action_dim=7)
        results = run_all_checks(root, embodiment_config=None)
        emb_check = next(r for r in results if r.check_id == "dataset.embodiment-action-dim-match")
        assert emb_check.decision == Decision.SKIPPED

    def test_embodiment_match_passes_when_dims_align(self, tmp_path):
        root = _make_valid_dataset(tmp_path, action_dim=7)

        class _StubEmbodiment:
            embodiment = "franka"
            action_dim = 7

        results = run_all_checks(root, embodiment_config=_StubEmbodiment())
        emb_check = next(r for r in results if r.check_id == "dataset.embodiment-action-dim-match")
        assert emb_check.decision == Decision.OK

    def test_embodiment_match_blocks_when_dims_mismatch(self, tmp_path):
        root = _make_valid_dataset(tmp_path, action_dim=7)

        class _StubEmbodiment:
            embodiment = "so100"
            action_dim = 6  # mismatched

        results = run_all_checks(root, embodiment_config=_StubEmbodiment())
        emb_check = next(r for r in results if r.check_id == "dataset.embodiment-action-dim-match")
        assert emb_check.decision == Decision.BLOCKER
        assert "so100" in emb_check.summary


# ---------- output format tests ----------

class TestOutputFormats:
    def test_format_human_lists_each_check_id(self, tmp_path):
        root = _make_valid_dataset(tmp_path)
        results = run_all_checks(root)
        text = format_human(results, dataset_root=root)
        for r in results:
            assert r.check_id in text

    def test_format_human_includes_decision_markers(self, tmp_path):
        root = _make_valid_dataset(tmp_path)
        results = run_all_checks(root)
        text = format_human(results)
        # Every result has a marker prefix
        assert "[OK]" in text or "[SKIPPED]" in text

    def test_format_json_is_valid_json(self, tmp_path):
        root = _make_valid_dataset(tmp_path)
        results = run_all_checks(root)
        report = format_json(results, dataset_root=root)
        parsed = json.loads(report)
        assert parsed["decision"] in ("ok", "warn", "blocker")
        assert parsed["n_checks"] == len(results)
        assert len(parsed["results"]) == len(results)
        # Each result has the canonical fields
        for r in parsed["results"]:
            assert {"check_id", "decision", "summary", "details", "fix_hint"} <= set(r.keys())

    def test_format_json_decision_matches_overall(self, tmp_path):
        # Construct one with a blocker
        root = tmp_path / "ds"
        root.mkdir()
        results = run_all_checks(root)  # missing everything → blockers
        report = json.loads(format_json(results, dataset_root=root))
        assert report["decision"] == "blocker"


# ---------- runner robustness ----------

class TestRunnerRobustness:
    def test_run_all_checks_does_not_raise_on_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        results = run_all_checks(empty)
        assert len(results) == len(REGISTERED_CHECKS)
        # info.json missing → blocker; rest skipped
        assert overall_decision(results) == Decision.BLOCKER

    def test_run_all_checks_does_not_raise_on_nonexistent_dir(self, tmp_path):
        # Path object passed for a path that doesn't exist
        results = run_all_checks(tmp_path / "nope")
        assert len(results) == len(REGISTERED_CHECKS)

    def test_check_results_have_falsifiable_check_ids(self, tmp_path):
        root = _make_valid_dataset(tmp_path)
        results = run_all_checks(root)
        for r in results:
            assert r.check_id.startswith("dataset.") or r.check_id.startswith("dataset.")
            assert r.check_id != ""
            assert isinstance(r.decision, Decision)
