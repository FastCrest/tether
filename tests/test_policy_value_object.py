"""Tests for src/tether/runtime/policy.py — Days 3-4 + Day 5 substrate.

Per ADR 2026-04-25-policy-versioning-architecture: per-policy state
bundle for 2-policy A/B mode. The Policy dataclass is a frozen
value-object; validation helpers cover the Day 5 CLI flag combos.
"""
from __future__ import annotations

import pytest

from tether.runtime.policy import (
    DEFAULT_SINGLE_POLICY_SLOT,
    Policy,
    make_single_policy,
    validate_memory_for_two_policies,
    validate_split_and_no_rtc,
)


# ---------------------------------------------------------------------------
# Policy dataclass
# ---------------------------------------------------------------------------


def _valid_kwargs(**overrides):
    base = dict(
        slot="prod", model_id="pi0-libero-v1",
        model_hash="abc123def456",
        export_dir="/exports/pi0",
        runtime=None, action_guard=None, rtc_adapter=None,
    )
    base.update(overrides)
    return base


def test_policy_is_frozen():
    p = Policy(**_valid_kwargs())
    with pytest.raises(AttributeError):
        p.slot = "different"  # type: ignore[misc]


def test_policy_rejects_empty_slot():
    with pytest.raises(ValueError, match="slot"):
        Policy(**_valid_kwargs(slot=""))


def test_policy_rejects_empty_model_id():
    with pytest.raises(ValueError, match="model_id"):
        Policy(**_valid_kwargs(model_id=""))


def test_policy_rejects_empty_model_hash():
    with pytest.raises(ValueError, match="model_hash"):
        Policy(**_valid_kwargs(model_hash=""))


def test_policy_rejects_non_hex_model_hash():
    with pytest.raises(ValueError, match="hex"):
        Policy(**_valid_kwargs(model_hash="not-hex-zzzz"))


def test_policy_accepts_uppercase_hex():
    """SHA-256 hex strings are case-insensitive; both forms should validate."""
    p = Policy(**_valid_kwargs(model_hash="ABC123DEF456"))
    assert p.model_hash == "ABC123DEF456"


def test_policy_rejects_negative_crash_count():
    with pytest.raises(ValueError, match="consecutive_crash_count"):
        Policy(**_valid_kwargs(consecutive_crash_count=-1))


def test_policy_default_crash_count_is_zero():
    p = Policy(**_valid_kwargs())
    assert p.consecutive_crash_count == 0


def test_model_version_combines_id_and_hash():
    """X-Tether-Model-Version header format -- stable across releases."""
    p = Policy(**_valid_kwargs(model_id="pi0-libero", model_hash="abc12345"))
    assert p.model_version == "pi0-libero@abc12345"


def test_with_crash_count_returns_new_policy():
    """Frozen dataclass -- must mutate via copy, not in-place."""
    p1 = Policy(**_valid_kwargs())
    p2 = p1.with_crash_count(5)
    assert p2 is not p1
    assert p2.consecutive_crash_count == 5
    assert p1.consecutive_crash_count == 0  # original untouched
    # All other fields preserved
    assert p2.slot == p1.slot
    assert p2.model_id == p1.model_id
    assert p2.runtime is p1.runtime  # passed by reference


def test_with_crash_count_validates_negative():
    p = Policy(**_valid_kwargs())
    with pytest.raises(ValueError, match="consecutive_crash_count"):
        p.with_crash_count(-1)


# ---------------------------------------------------------------------------
# make_single_policy factory
# ---------------------------------------------------------------------------


def test_make_single_policy_uses_default_slot():
    p = make_single_policy(
        model_id="m", model_hash="abc", export_dir="/tmp",
    )
    assert p.slot == DEFAULT_SINGLE_POLICY_SLOT
    assert p.slot == "prod"


def test_make_single_policy_passes_through_state_objects():
    sentinel_runtime = object()
    sentinel_guard = object()
    sentinel_rtc = object()
    p = make_single_policy(
        model_id="m", model_hash="abc", export_dir="/tmp",
        runtime=sentinel_runtime, action_guard=sentinel_guard,
        rtc_adapter=sentinel_rtc,
    )
    assert p.runtime is sentinel_runtime
    assert p.action_guard is sentinel_guard
    assert p.rtc_adapter is sentinel_rtc


def test_make_single_policy_accepts_path_for_export_dir():
    from pathlib import Path
    p = make_single_policy(
        model_id="m", model_hash="abc",
        export_dir=Path("/exports/m"),
    )
    assert p.export_dir == "/exports/m"


# ---------------------------------------------------------------------------
# validate_split_and_no_rtc -- Day 5 CLI flag combo
# ---------------------------------------------------------------------------


def test_validate_split_accepts_zero():
    """split=0 = all traffic to slot B (effectively single-policy on B)."""
    validate_split_and_no_rtc(split_a_percent=0, no_rtc=True)


def test_validate_split_accepts_hundred():
    """split=100 = all traffic to slot A (effectively single-policy on A)."""
    validate_split_and_no_rtc(split_a_percent=100, no_rtc=True)


def test_validate_split_rejects_negative():
    with pytest.raises(ValueError, match="split_a_percent"):
        validate_split_and_no_rtc(split_a_percent=-1, no_rtc=True)


def test_validate_split_rejects_over_hundred():
    with pytest.raises(ValueError, match="split_a_percent"):
        validate_split_and_no_rtc(split_a_percent=101, no_rtc=True)


def test_validate_2_policy_requires_no_rtc():
    """RTC carry-over is per-policy; cross-policy carry-over produces OOD."""
    with pytest.raises(ValueError, match="--no-rtc"):
        validate_split_and_no_rtc(split_a_percent=50, no_rtc=False)


def test_validate_split_zero_still_requires_no_rtc():
    """Even when split=0/100, the operator has 2 policies loaded -- RTC
    must be off so the inactive slot's RTC state doesn't leak."""
    with pytest.raises(ValueError, match="--no-rtc"):
        validate_split_and_no_rtc(split_a_percent=0, no_rtc=False)


# ---------------------------------------------------------------------------
# validate_memory_for_two_policies -- Day 5 refuse-to-load check
# ---------------------------------------------------------------------------


def test_memory_check_passes_when_2x_fits_within_safety_factor():
    """2 × 5GB = 10GB; budget = 0.7 × 16GB = 11.2GB. Passes."""
    validate_memory_for_two_policies(
        model_size_bytes=5 * 10**9,
        total_gpu_bytes=16 * 10**9,
    )


def test_memory_check_fails_when_2x_exceeds_safety_factor():
    """2 × 8GB = 16GB; budget = 0.7 × 16GB = 11.2GB. Exceeds."""
    with pytest.raises(ValueError, match="VRAM"):
        validate_memory_for_two_policies(
            model_size_bytes=8 * 10**9,
            total_gpu_bytes=16 * 10**9,
        )


def test_memory_check_passes_at_exact_safety_factor_boundary():
    """2 × 5.6GB = 11.2GB; budget = 0.7 × 16GB = 11.2GB. Equal -> pass."""
    validate_memory_for_two_policies(
        model_size_bytes=int(5.6 * 10**9),
        total_gpu_bytes=16 * 10**9,
    )


def test_memory_check_custom_safety_factor():
    """Tighter 0.5 safety factor: 2 × 5GB = 10GB > 8GB budget. Fails."""
    with pytest.raises(ValueError, match="VRAM"):
        validate_memory_for_two_policies(
            model_size_bytes=5 * 10**9,
            total_gpu_bytes=16 * 10**9,
            safety_factor=0.5,
        )


def test_memory_check_rejects_zero_model_size():
    with pytest.raises(ValueError, match="model_size_bytes"):
        validate_memory_for_two_policies(
            model_size_bytes=0, total_gpu_bytes=16 * 10**9,
        )


def test_memory_check_rejects_zero_total_gpu():
    with pytest.raises(ValueError, match="total_gpu_bytes"):
        validate_memory_for_two_policies(
            model_size_bytes=5 * 10**9, total_gpu_bytes=0,
        )


def test_memory_check_rejects_invalid_safety_factor():
    with pytest.raises(ValueError, match="safety_factor"):
        validate_memory_for_two_policies(
            model_size_bytes=5 * 10**9,
            total_gpu_bytes=16 * 10**9,
            safety_factor=1.5,  # > 1
        )


def test_memory_check_error_message_includes_remediation():
    """Operator-readable error: must mention what to do."""
    with pytest.raises(ValueError) as exc:
        validate_memory_for_two_policies(
            model_size_bytes=10 * 10**9,
            total_gpu_bytes=16 * 10**9,
        )
    msg = str(exc.value)
    # Mentions VRAM amounts + how to proceed
    assert "VRAM" in msg
    assert "single-policy" in msg or "smaller models" in msg or "larger GPU" in msg
