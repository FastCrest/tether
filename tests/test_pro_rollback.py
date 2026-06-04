"""Tests for src/tether/pro/rollback.py — Phase 1 Day 7.

Per ADR 2026-04-25-self-distilling-serve-architecture decisions #4 + #7:
≤60s warm rollback via policy-versioning secondary slot. Both auto
(post-swap-monitor trip) AND manual (CLI / endpoint) paths share one
RollbackHandler.
"""
from __future__ import annotations

import pytest

from tether.pro.rollback import (
    RollbackHandler,
    RollbackOutcome,
)


class _StubRouter:
    """Minimal router stub matching the policy-versioning surface
    RollbackHandler depends on."""

    def __init__(self, initial_slot: str = "a"):
        self.active_slot = initial_slot
        self.swap_calls: list[str] = []
        self.fail_swap = False

    def set_active(self, slot: str) -> None:
        self.swap_calls.append(slot)
        if self.fail_swap:
            raise RuntimeError("router refused swap")
        self.active_slot = slot


def _mk_handler(
    *,
    initial_slot: str = "a",
    audit: list | None = None,
    metrics: list | None = None,
) -> tuple[RollbackHandler, _StubRouter]:
    router = _StubRouter(initial_slot=initial_slot)
    handler = RollbackHandler(
        router_swap_fn=router.set_active,
        active_slot_getter=lambda: router.active_slot,
        audit_writer=(audit.append if audit is not None else None),
        metric_emitter=(metrics.append if metrics is not None else None),
    )
    return handler, router


# ---------------------------------------------------------------------------
# Auto-trigger paths
# ---------------------------------------------------------------------------


def test_auto_rollback_flips_to_other_slot():
    handler, router = _mk_handler(initial_slot="a")
    outcome = handler.execute(trigger="auto", reason="T1")
    assert outcome.succeeded
    assert outcome.from_slot == "a"
    assert outcome.to_slot == "b"
    assert router.active_slot == "b"
    assert router.swap_calls == ["b"]


def test_auto_rollback_from_slot_b_flips_to_a():
    handler, router = _mk_handler(initial_slot="b")
    outcome = handler.execute(trigger="auto", reason="T2")
    assert outcome.to_slot == "a"
    assert router.active_slot == "a"


def test_auto_rollback_omits_operator():
    """trigger=auto doesn't require operator field."""
    handler, _ = _mk_handler()
    outcome = handler.execute(trigger="auto", reason="T1")
    assert outcome.succeeded


# ---------------------------------------------------------------------------
# Manual triggers (cli / endpoint) require operator
# ---------------------------------------------------------------------------


def test_cli_rollback_requires_operator():
    handler, _ = _mk_handler()
    with pytest.raises(ValueError, match="operator"):
        handler.execute(trigger="cli", reason="operator-cli")


def test_endpoint_rollback_requires_operator():
    handler, _ = _mk_handler()
    with pytest.raises(ValueError, match="operator"):
        handler.execute(trigger="endpoint", reason="operator-endpoint")


def test_cli_rollback_with_operator_succeeds():
    handler, _ = _mk_handler()
    outcome = handler.execute(
        trigger="cli", reason="operator-cli", operator="ops_engineer_42",
    )
    assert outcome.succeeded


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def test_audit_record_written_on_success():
    audit = []
    handler, _ = _mk_handler(audit=audit)
    handler.execute(trigger="auto", reason="T1")
    assert len(audit) == 1
    record = audit[0]
    assert record["trigger"] == "auto"
    assert record["reason"] == "T1"
    assert record["from_slot"] == "a"
    assert record["to_slot"] == "b"
    assert record["succeeded"] is True
    assert "audit_id" in record
    assert "timestamp" in record


def test_audit_record_written_on_failure():
    """Failed rollbacks must also be audited so operators can see them."""
    audit = []
    handler, router = _mk_handler(audit=audit)
    router.fail_swap = True
    outcome = handler.execute(trigger="auto", reason="T1")
    assert not outcome.succeeded
    assert len(audit) == 1
    assert audit[0]["succeeded"] is False
    assert audit[0]["error"] is not None


def test_audit_writer_optional():
    """No audit writer = silent (but logged at WARN level by handler).
    For tests we just verify it doesn't crash."""
    handler, _ = _mk_handler(audit=None)
    outcome = handler.execute(trigger="auto", reason="T1")
    assert outcome.succeeded


# ---------------------------------------------------------------------------
# Metric emission
# ---------------------------------------------------------------------------


def test_metric_emitted_on_success_only():
    metrics = []
    handler, router = _mk_handler(metrics=metrics)
    handler.execute(trigger="auto", reason="T1")
    assert metrics == ["T1"]
    # Failure → no metric
    metrics.clear()
    router.fail_swap = True
    handler.execute(trigger="auto", reason="T2")
    assert metrics == []


# ---------------------------------------------------------------------------
# Same-slot no-op
# ---------------------------------------------------------------------------


def test_rollback_to_same_slot_returns_failure_no_op():
    handler, _ = _mk_handler(initial_slot="a")
    outcome = handler.execute(
        trigger="auto", reason="T1", target_slot="a",  # same as active
    )
    assert not outcome.succeeded
    assert "already active" in (outcome.error or "")


# ---------------------------------------------------------------------------
# rollback_count
# ---------------------------------------------------------------------------


def test_rollback_count_increments_on_success():
    handler, _ = _mk_handler()
    assert handler.rollback_count == 0
    handler.execute(trigger="auto", reason="T1")
    assert handler.rollback_count == 1


def test_rollback_count_does_not_increment_on_failure():
    handler, router = _mk_handler()
    router.fail_swap = True
    handler.execute(trigger="auto", reason="T1")
    assert handler.rollback_count == 0


# ---------------------------------------------------------------------------
# Outcome shape
# ---------------------------------------------------------------------------


def test_outcome_is_frozen():
    handler, _ = _mk_handler()
    outcome = handler.execute(trigger="auto", reason="T1")
    with pytest.raises(AttributeError):
        outcome.succeeded = False  # type: ignore[misc]


def test_outcome_includes_elapsed_time():
    handler, _ = _mk_handler()
    outcome = handler.execute(trigger="auto", reason="T1")
    assert outcome.elapsed_s >= 0
    assert outcome.elapsed_s < 1.0  # stub router → near-zero


def test_outcome_audit_id_unique_across_calls():
    handler, _ = _mk_handler()
    o1 = handler.execute(trigger="auto", reason="T1")
    o2 = handler.execute(trigger="auto", reason="T2")
    assert o1.audit_id != o2.audit_id
