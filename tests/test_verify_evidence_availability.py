from __future__ import annotations

import copy
import itertools

import pytest

from tether.parity_cert import build_parity_cert
from tether.parity_report import render_parity_report
from tether.verification_evidence import EvidenceValue, UnavailableReason
from tether.verify import run_verify


_MEMORY_PID = itertools.count(30_000)


def _memory(value: int = 100_000_000) -> dict:
    pid = next(_MEMORY_PID)
    samples = [
        {
            "scheduled_monotonic_s": timestamp,
            "captured_monotonic_s": timestamp,
            "process_rss_bytes": value,
            "device_allocated_bytes": 0,
        }
        for timestamp in (1.0, 1.1, 1.2)
    ]
    summary = {
        "sample_hz": 10.0,
        "backend": "cpu",
        "process_identity": {"pid": pid, "capture_id": f"capture-{pid}"},
        "window": {
            "started_monotonic_s": 1.0,
            "ended_monotonic_s": 1.2,
            "duration_s": 0.2,
            "expected_samples": 3,
            "captured_samples": 3,
            "max_gap_s": 0.1,
        },
        "samples": samples,
        "process_rss": {"peak_bytes": value, "p95_bytes": value},
        "device_allocated": {"peak_bytes": 0, "p95_bytes": 0},
        "combined": {"peak_bytes": value, "p95_bytes": value},
    }
    return {"status": "available", "value": summary}


def _complete_results() -> dict:
    episodes = []
    for episode in range(30):
        episodes.append(
            {
                "ep": episode,
                "success": True,
                "steps": 2,
                "actions": [[0.1] * 7, [0.2] * 7],
                "action_timestamps_s": [1.0, 2.0],
                "inference_latency_ms": [5.0, 6.0],
                "safety_clamp_count": 0,
            }
        )
    return {
        "per_task": [{"task_idx": 0, "episodes": episodes}],
        "verification_device": "cpu",
        "safety_evidence": {
            "status": "available",
            "value": {"config": "test-safety.json"},
        },
        "memory_evidence": _memory(),
    }


def _verdict(original: dict, optimized: dict):
    original_identity = original.get("memory_evidence", {}).get("value", {}).get("process_identity")
    optimized_identity = (
        optimized.get("memory_evidence", {}).get("value", {}).get("process_identity")
    )
    if original_identity and original_identity == optimized_identity:
        optimized_identity["pid"] += 1
        optimized_identity["capture_id"] += "-candidate"

    def gather(**_kwargs):
        return original, optimized

    return run_verify(optimized_ref="export", gather_fn=gather, num_episodes=30)


@pytest.mark.parametrize(
    ("channel", "expected_gate", "expected_reason"),
    [
        ("safety", "S1", "capture_disabled"),
        ("actions", "S1", "missing_field"),
        ("velocity", "S2", "missing_field"),
        ("success", "S3", "missing_field"),
        ("latency", "P2", "missing_field"),
        ("memory", "P3", "measurement_failed"),
        ("teacher", "P4", "missing_field"),
    ],
)
def test_each_required_channel_fails_first_affected_gate(channel, expected_gate, expected_reason):
    original = _complete_results()
    optimized = copy.deepcopy(original)
    episodes = optimized["per_task"][0]["episodes"]

    if channel == "safety":
        for episode in episodes:
            episode.pop("safety_clamp_count")
        optimized["safety_evidence"] = {
            "status": "unavailable",
            "reason": "capture_disabled",
        }
    elif channel == "actions":
        for episode in episodes:
            episode.pop("actions")
    elif channel == "velocity":
        for episode in episodes:
            episode.pop("action_timestamps_s")
    elif channel == "success":
        for episode in episodes:
            episode.pop("success")
    elif channel == "latency":
        for episode in episodes:
            episode.pop("inference_latency_ms")
    elif channel == "memory":
        optimized["memory_evidence"] = {
            "status": "unavailable",
            "reason": "measurement_failed",
        }
    elif channel == "teacher":
        for episode in original["per_task"][0]["episodes"]:
            episode["actions"].append([0.3] * 7)
            episode["action_timestamps_s"].append(3.0)

    verdict = _verdict(original, optimized)
    assert verdict.passed is False
    assert verdict.first_failing_gate_id == expected_gate

    gate = verdict.eval_report.first_failing_gate
    assert gate is not None
    assert isinstance(gate.measured, EvidenceValue)
    assert gate.measured.reason is UnavailableReason(expected_reason)

    machine = verdict.to_dict()
    assert machine["eval_report"]["first_failing_gate"]["measured"] == {
        "status": "unavailable",
        "reason": expected_reason,
    }
    markdown = render_parity_report(verdict)
    assert f"UNAVAILABLE ({expected_reason})" in markdown
    cert = build_parity_cert(verdict)
    first = next(g for g in cert["gates"] if g["gate_id"] == expected_gate)
    assert first["measured"]["reason"] == expected_reason


def test_optional_diagnostics_are_not_evaluated_never_pass():
    bare = _complete_results()
    for episode in bare["per_task"][0]["episodes"]:
        episode.pop("actions")
        episode.pop("action_timestamps_s")
    verdict = _verdict(bare, copy.deepcopy(bare))
    diagnostics = verdict.to_dict()["diagnostics"]
    assert diagnostics["two_sample"]["status"] == "NOT_EVALUATED"
    assert diagnostics["embodied"]["status"] == "NOT_EVALUATED"
    assert "NOT_EVALUATED" in render_parity_report(verdict)


def test_partial_memory_payload_fails_p3_instead_of_using_partial_numbers():
    original = _complete_results()
    optimized = copy.deepcopy(original)
    optimized["memory_evidence"]["value"].pop("device_allocated")
    verdict = _verdict(original, optimized)
    p3 = next(gate for gate in verdict.eval_report.all_gates if gate.gate_id == "P3")
    assert p3.passed is False
    assert isinstance(p3.measured, EvidenceValue)
    assert p3.measured.reason is UnavailableReason.MISSING_FIELD


@pytest.mark.parametrize("mutation", ["one_sample", "sparse", "wrong_count"])
def test_inadequate_memory_coverage_fails_p3(mutation):
    original = _complete_results()
    optimized = copy.deepcopy(original)
    memory = optimized["memory_evidence"]["value"]
    if mutation == "one_sample":
        memory["samples"] = memory["samples"][:1]
        memory["window"]["captured_samples"] = 1
    elif mutation == "sparse":
        memory["samples"][1]["captured_monotonic_s"] = 1.19
        memory["window"]["max_gap_s"] = 0.19
    else:
        extra = copy.deepcopy(optimized["per_task"][0]["episodes"][-1])
        extra["ep"] = 30
        optimized["per_task"][0]["episodes"].append(extra)

    verdict = _verdict(original, optimized)
    p3 = next(gate for gate in verdict.eval_report.all_gates if gate.gate_id == "P3")
    assert p3.passed is False
    assert isinstance(p3.measured, EvidenceValue)
    assert p3.measured.reason is UnavailableReason.MEASUREMENT_FAILED


@pytest.mark.parametrize(
    ("timestamps", "duration"),
    [
        ((1.0, 1.125, 1.25), 0.25),
        ((1.0, 1.15, 1.30), 0.30),
    ],
    ids=("eight_hz", "one_hundred_fifty_ms"),
)
def test_non_ten_hz_memory_payloads_fail_p3(timestamps, duration):
    original = _complete_results()
    optimized = copy.deepcopy(original)
    memory = optimized["memory_evidence"]["value"]
    for sample, timestamp in zip(memory["samples"], timestamps):
        sample["scheduled_monotonic_s"] = timestamp
        sample["captured_monotonic_s"] = timestamp
    memory["window"].update(
        ended_monotonic_s=1.0 + duration,
        duration_s=duration,
        expected_samples=3,
        max_gap_s=timestamps[1] - timestamps[0],
    )
    verdict = _verdict(original, optimized)
    p3 = next(gate for gate in verdict.eval_report.all_gates if gate.gate_id == "P3")
    assert p3.passed is False
    assert isinstance(p3.measured, EvidenceValue)
    assert p3.measured.reason is UnavailableReason.MEASUREMENT_FAILED


def test_evidence_wire_shape_and_closed_reason_enum():
    assert EvidenceValue.available(0).to_dict() == {
        "status": "available",
        "value": 0,
    }
    assert EvidenceValue.unavailable("artifact_unavailable").to_dict() == {
        "status": "unavailable",
        "reason": "artifact_unavailable",
    }
    assert {reason.value for reason in UnavailableReason} == {
        "missing_field",
        "capture_disabled",
        "backend_unsupported",
        "measurement_failed",
        "artifact_unavailable",
    }
    with pytest.raises(ValueError):
        EvidenceValue.unavailable("not_a_reason")
    with pytest.raises(ValueError):
        EvidenceValue.available([])
