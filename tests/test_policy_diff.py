from __future__ import annotations

from pathlib import Path

import pytest

from tether.policy_diff import diff_policy_traces, should_fail
from tether.runtime.record import RecordWriter


def _write_trace(
    record_dir: Path,
    *,
    actions: list[list[float]],
    latency_ms: float = 100.0,
    guard: dict | None = None,
    routing: dict | None = None,
    instruction: str = "pick",
) -> Path:
    writer = RecordWriter(
        record_dir=record_dir,
        model_hash="deadbeefcafe0000",
        config_hash="0011223344556677",
        export_dir=str(record_dir / "fake_export"),
        model_type="pi0.5",
        export_kind="monolithic",
        providers=["CPUExecutionProvider"],
        gpu="cpu",
        cuda_version="",
        ort_version="1.20.1",
        embodiment="franka",
        image_redaction="hash_only",
        tether_version="0.0.0-test",
        gzip_output=False,
    )
    writer.write_request(
        chunk_id=0,
        image_b64="aGVsbG8=",
        instruction=instruction,
        state=[0.1, 0.2],
        episode_id="ep-1",
        request_id="req-1",
        actions=actions,
        action_dim=len(actions[0]) if actions else 0,
        latency_total_ms=latency_ms,
        mode="onnx_cpu",
        guard=guard,
        routing=routing,
    )
    writer.write_footer({"total_requests": 1})
    writer.close()
    return writer.filepath


def test_policy_diff_trace_pair_passes_identical_actions(tmp_path: Path) -> None:
    baseline = _write_trace(tmp_path / "base", actions=[[0.1, 0.2], [0.3, 0.4]])
    candidate = _write_trace(tmp_path / "cand", actions=[[0.1, 0.2], [0.3, 0.4]])

    report = diff_policy_traces(baseline_trace=baseline, candidate_trace=candidate)

    assert report["kind"] == "tether.policy_diff"
    assert report["summary"]["verdict"] == "pass"
    assert report["summary"]["compared"] == 1
    assert report["summary"]["action_failures"] == 0
    assert should_fail(report, "any") is False


def test_policy_diff_detects_action_failure(tmp_path: Path) -> None:
    baseline = _write_trace(tmp_path / "base", actions=[[0.1, 0.2]])
    candidate = _write_trace(tmp_path / "cand", actions=[[1.0, 2.0]])

    report = diff_policy_traces(
        baseline_trace=baseline,
        candidate_trace=candidate,
        max_action_delta=0.05,
    )

    assert report["summary"]["verdict"] == "fail"
    assert report["summary"]["action_failures"] == 1
    assert should_fail(report, "actions") is True


def test_policy_diff_detects_latency_and_guard_regression(tmp_path: Path) -> None:
    baseline = _write_trace(tmp_path / "base", actions=[[0.1, 0.2]], latency_ms=100.0)
    candidate = _write_trace(
        tmp_path / "cand",
        actions=[[0.1, 0.2]],
        latency_ms=150.0,
        guard={"clamped": True, "clamp_count": 1, "violations": ["joint_0"]},
    )

    report = diff_policy_traces(
        baseline_trace=baseline,
        candidate_trace=candidate,
        max_latency_regression_pct=0.10,
    )

    assert report["summary"]["latency_regressions"] == 1
    assert report["summary"]["guard_regressions"] == 1
    assert should_fail(report, "latency") is True
    assert should_fail(report, "guard") is True


def test_policy_diff_shadow_actions(tmp_path: Path) -> None:
    shadow_trace = _write_trace(
        tmp_path / "shadow",
        actions=[[0.1, 0.2]],
        routing={"shadow_actions": [[0.11, 0.21]]},
    )

    report = diff_policy_traces(
        baseline_trace=shadow_trace,
        shadow=True,
        max_action_delta=0.05,
    )

    assert report["mode"] == "shadow_trace"
    assert report["candidate"] == {"source": "shadow_result or routing.shadow_actions"}
    assert report["summary"]["verdict"] == "pass"
    assert report["summary"]["compared"] == 1


def test_policy_diff_shadow_result_records(tmp_path: Path) -> None:
    writer = RecordWriter(
        record_dir=tmp_path / "background_shadow",
        model_hash="deadbeefcafe0000",
        config_hash="0011223344556677",
        export_dir=str(tmp_path / "fake_export"),
        model_type="pi0.5",
        export_kind="monolithic",
        providers=["CPUExecutionProvider"],
        gzip_output=False,
    )
    seq = writer.write_request(
        chunk_id=0,
        image_b64="aGVsbG8=",
        instruction="sampled",
        state=[0.1],
        actions=[[0.1, 0.2]],
        action_dim=2,
        latency_total_ms=100.0,
        routing={"shadow_sampled": True, "shadow_pending": True},
    )
    writer.write_shadow_result(
        seq=seq,
        actions=[[0.11, 0.21]],
        action_dim=2,
        latency_total_ms=12.0,
        routing={
            "shadow_sampled": True,
            "shadow_mode": "background",
            "shadow_actions": [[0.11, 0.21]],
            "shadow_latency_ms": 12.0,
        },
    )
    writer.write_footer({"total_requests": 1})
    writer.close()

    report = diff_policy_traces(
        baseline_trace=writer.filepath,
        shadow=True,
        max_action_delta=0.05,
    )

    assert report["summary"]["verdict"] == "pass"
    assert report["summary"]["compared"] == 1
    assert report["summary"]["shadow_pending"] == 0


def test_policy_diff_shadow_skips_unsampled_rows(tmp_path: Path) -> None:
    writer = RecordWriter(
        record_dir=tmp_path / "mixed",
        model_hash="deadbeefcafe0000",
        config_hash="0011223344556677",
        export_dir=str(tmp_path / "fake_export"),
        model_type="pi0.5",
        export_kind="monolithic",
        providers=["CPUExecutionProvider"],
        gzip_output=False,
    )
    writer.write_request(
        chunk_id=0,
        image_b64="aGVsbG8=",
        instruction="sampled",
        state=[0.1],
        actions=[[0.1, 0.2]],
        action_dim=2,
        latency_total_ms=100.0,
        routing={"shadow_sampled": True, "shadow_actions": [[0.11, 0.21]]},
    )
    writer.write_request(
        chunk_id=1,
        image_b64="aGVsbG8=",
        instruction="unsampled",
        state=[0.1],
        actions=[[0.1, 0.2]],
        action_dim=2,
        latency_total_ms=100.0,
        routing={"shadow_sampled": False, "shadow_sample_rate": 0.5},
    )
    writer.write_footer({"total_requests": 2})
    writer.close()

    report = diff_policy_traces(
        baseline_trace=writer.filepath,
        shadow=True,
        max_action_delta=0.05,
    )

    assert report["summary"]["verdict"] == "pass"
    assert report["summary"]["compared"] == 1
    assert report["summary"]["shadow_skipped"] == 1
    assert report["summary"]["missing_candidate"] == 0


def test_policy_diff_requires_candidate_unless_shadow(tmp_path: Path) -> None:
    baseline = _write_trace(tmp_path / "base", actions=[[0.1, 0.2]])

    with pytest.raises(ValueError, match="candidate_trace is required"):
        diff_policy_traces(baseline_trace=baseline)
