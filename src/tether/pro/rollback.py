"""Pro-tier rollback handler — composes with policy-versioning's secondary slot.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #4 + #7:
auto-rollback fires when the post-swap monitor (post_swap_monitor.py)
trips, OR manual rollback fires via CLI / endpoint. Both paths share
~40 LOC behind a single RollbackHandler.

≤60s warm rollback SLA per the ADR. Achieved by routing 100% of /act
traffic to the secondary slot (which already holds the previous model)
via the policy-versioning router. No model re-load, no warmup.

The handler is PURE — caller (Day 8+ wiring) provides a router
instance + supplies callbacks for audit logging / metric emission.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)


RollbackTrigger = Literal["auto", "cli", "endpoint"]


@dataclass(frozen=True)
class RollbackOutcome:
    """Frozen output of RollbackHandler.execute(). Caller logs/emits."""

    succeeded: bool
    trigger: RollbackTrigger
    from_slot: str  # the policy slot we're rolling FROM
    to_slot: str    # the slot we're rolling TO
    reason: str     # bounded enum: T1|T2|T3|operator-cli|operator-endpoint|forced
    audit_id: str   # unique id for log correlation
    elapsed_s: float
    error: str | None = None


class RollbackHandler:
    """Stateful rollback executor. One per server (lives on
    server.rollback_handler when Pro is enabled).

    Composition:
    - The policy-versioning router holds slots `a` (live) + `b` (warm
      secondary). On rollback, this handler atomically flips the active
      slot.
    - Audit log entry written for every rollback (auto or manual).
    - Time elapsed measured from execute() entry to slot-flip completion.
    """

    __slots__ = (
        "_router_swap_fn",
        "_audit_writer",
        "_metric_emitter",
        "_active_slot_getter",
        "_rollback_count",
    )

    def __init__(
        self,
        *,
        router_swap_fn: Callable[[str], None],
        active_slot_getter: Callable[[], str],
        audit_writer: Callable[[dict[str, Any]], None] | None = None,
        metric_emitter: Callable[[str], None] | None = None,
    ):
        """Args:
            router_swap_fn: callable(target_slot) → None. Atomically flips
                the policy-versioning router's active slot. The caller
                wires this from policy_router.PolicyRouter.set_active().
            active_slot_getter: callable() → str. Returns the currently-
                active slot name (e.g. "a" / "b").
            audit_writer: optional callable(record) → None. Writes one
                JSON record per rollback to durable storage. None = no
                audit log (NOT recommended for production).
            metric_emitter: optional callable(reason) → None. Increments
                a Prometheus counter per rollback reason. None = no
                metric emission.
        """
        self._router_swap_fn = router_swap_fn
        self._active_slot_getter = active_slot_getter
        self._audit_writer = audit_writer
        self._metric_emitter = metric_emitter
        self._rollback_count = 0

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    def execute(
        self,
        *,
        trigger: RollbackTrigger,
        reason: str,
        operator: str | None = None,
        target_slot: str | None = None,
    ) -> RollbackOutcome:
        """Perform the rollback. Returns RollbackOutcome with timing +
        success flag.

        Args:
            trigger: where the rollback came from (auto / cli / endpoint).
            reason: bounded — T1 / T2 / T3 (auto) | operator-cli /
                operator-endpoint / forced (manual).
            operator: when manual (cli/endpoint), the operator id /
                bearer-token-subject. Required for non-auto triggers.
            target_slot: which slot to roll BACK to. Default: the
                non-active slot (the warm secondary).
        """
        import time

        if trigger != "auto" and not operator:
            raise ValueError(
                f"trigger={trigger} requires operator field; auto-only "
                f"rollbacks may omit it"
            )

        from_slot = self._active_slot_getter()
        if target_slot is None:
            # Default: pick the OTHER slot. Per policy-versioning, the
            # router holds two slots {a, b}; rolling back means flipping.
            target_slot = "b" if from_slot == "a" else "a"

        if target_slot == from_slot:
            return RollbackOutcome(
                succeeded=False, trigger=trigger,
                from_slot=from_slot, to_slot=target_slot,
                reason=reason, audit_id=_make_audit_id(),
                elapsed_s=0.0,
                error=f"target_slot {target_slot} is already active; no-op",
            )

        audit_id = _make_audit_id()
        t0 = time.perf_counter()
        error: str | None = None
        try:
            self._router_swap_fn(target_slot)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "rollback_handler.swap_failed audit_id=%s from=%s to=%s "
                "trigger=%s reason=%s exc=%s",
                audit_id, from_slot, target_slot, trigger, reason, error,
            )
        elapsed = time.perf_counter() - t0
        succeeded = error is None
        if succeeded:
            self._rollback_count += 1
            logger.warning(
                "rollback_handler.executed audit_id=%s from=%s to=%s "
                "trigger=%s reason=%s elapsed_ms=%.1f operator=%s",
                audit_id, from_slot, target_slot, trigger, reason,
                elapsed * 1000, operator or "(auto)",
            )
        outcome = RollbackOutcome(
            succeeded=succeeded, trigger=trigger,
            from_slot=from_slot, to_slot=target_slot,
            reason=reason, audit_id=audit_id, elapsed_s=elapsed,
            error=error,
        )
        # Always write the audit record, even on failure (operators need
        # to see attempted-but-failed rollbacks too).
        if self._audit_writer is not None:
            try:
                self._audit_writer({
                    "audit_id": audit_id,
                    "trigger": trigger,
                    "reason": reason,
                    "operator": operator,
                    "from_slot": from_slot,
                    "to_slot": target_slot,
                    "elapsed_s": elapsed,
                    "succeeded": succeeded,
                    "error": error,
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ"
                    ),
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "rollback_handler.audit_write_failed audit_id=%s: %s",
                    audit_id, exc,
                )
        if self._metric_emitter is not None and succeeded:
            try:
                self._metric_emitter(reason)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "rollback_handler.metric_emit_failed audit_id=%s: %s",
                    audit_id, exc,
                )
        return outcome


def _make_audit_id() -> str:
    """Short audit id for log correlation. Phase 1 uses a millisecond
    timestamp + 4-hex-char random suffix; Phase 2 may switch to UUIDs."""
    import secrets
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    suffix = secrets.token_hex(2)
    return f"rb-{ts}-{suffix}"


__all__ = [
    "RollbackHandler",
    "RollbackOutcome",
    "RollbackTrigger",
]
