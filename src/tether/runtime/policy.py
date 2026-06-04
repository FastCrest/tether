"""Per-policy state bundle for 2-policy A/B serve mode.

Per ADR 2026-04-25-policy-versioning-architecture Days 3-4: bundles
the per-policy serving state (`ActionGuard`, `RtcAdapter`, `EpisodeCache`,
PolicyRuntime, model identity) into one `Policy` value-object.

Single-policy mode: `server.policies = {"prod": Policy(...)}`.
2-policy mode:      `server.policies = {"a": Policy(...), "b": Policy(...)}`.

The /act handler dispatches via PolicyRouter.route() -> server.policies[slot]
in 2-policy mode; in single-policy mode, all traffic goes to "prod" with
no router invocation.

The Policy dataclass is a VALUE-OBJECT — frozen, no methods that mutate
state. The wrapped state objects (ActionGuard, RtcAdapter, ...) hold
their own runtime state and are passed by reference; the Policy bundle
just composes their identities.

Memory safety check: holding two Policy instances simultaneously
requires ~2x model_size_bytes of GPU VRAM. Per ADR, the CLI's
--policy-b validation refuses to load when
`2 × model_size_bytes > 0.7 × total_bytes`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Single-policy mode uses this slot name. Stable contract -- record-replay
# JSONL traces + Prometheus labels both ground out on this default value
# in pre-2-policy deployments. Bumping = breaking customer dashboards.
DEFAULT_SINGLE_POLICY_SLOT = "prod"


@dataclass(frozen=True)
class Policy:
    """Frozen per-policy state bundle.

    Fields:
        slot: bounded enum -- "prod" (single-policy mode) or "a" / "b"
            (2-policy mode). Surfaced in record-replay traces +
            Prometheus labels + X-Tether-Policy-Slot header.
        model_id: human-readable identifier (e.g. "pi0-libero-v1").
            Combined with model_hash for X-Tether-Model-Version header.
        model_hash: 16-hex SHA-256 prefix of the model files.
            Stable identity for record-replay correlation.
        export_dir: filesystem path the model was loaded from. Used
            for diagnostic reporting + reload paths.
        runtime: the PolicyRuntime (queue + scheduler + worker) bound
            to this policy. Lifetime managed by the server (start/stop).
            None when the policy backend doesn't implement run_batch
            (legacy backends take the direct predict_from_base64_async
            path).
        action_guard: per-policy ActionGuard instance. None when no
            safety_config was provided. ActionGuards may have different
            limits per policy (e.g. policy A has wider limits during
            an A/B test of conservative-vs-aggressive student).
        rtc_adapter: per-policy RtcAdapter. None when --no-rtc.
            2-policy mode REQUIRES --no-rtc per ADR (RTC carry-over
            state is per-policy and cross-policy carry-over produces
            out-of-distribution actions).
        consecutive_crash_count: per-policy circuit-breaker counter.
            Day 8 wiring uses this to route 100% to the surviving
            policy when one exceeds threshold.
    """

    slot: str
    model_id: str
    model_hash: str
    export_dir: str
    runtime: Any | None  # PolicyRuntime; typed as Any to avoid import cycle
    action_guard: Any | None  # ActionGuard
    rtc_adapter: Any | None  # RtcAdapter
    consecutive_crash_count: int = 0

    def __post_init__(self) -> None:
        if not self.slot:
            raise ValueError("Policy.slot must be non-empty")
        if not self.model_id:
            raise ValueError("Policy.model_id must be non-empty")
        if not self.model_hash:
            raise ValueError("Policy.model_hash must be non-empty")
        # 16-hex prefix is the convention; tests pass shorter values
        # so we don't strictly validate length, but we do enforce hex.
        if not all(c in "0123456789abcdef" for c in self.model_hash.lower()):
            raise ValueError(
                f"Policy.model_hash must be hex (0-9 a-f), got "
                f"{self.model_hash!r}"
            )
        if self.consecutive_crash_count < 0:
            raise ValueError(
                f"Policy.consecutive_crash_count must be >= 0, got "
                f"{self.consecutive_crash_count}"
            )

    @property
    def model_version(self) -> str:
        """Combined identifier for X-Tether-Model-Version header.
        Stable across release boundaries -- customers grep on this."""
        return f"{self.model_id}@{self.model_hash}"

    def with_crash_count(self, count: int) -> "Policy":
        """Return a new Policy with consecutive_crash_count updated.
        Frozen-friendly: callers replace the dict entry rather than
        mutate. Day 8 circuit-breaker wiring uses this."""
        return Policy(
            slot=self.slot, model_id=self.model_id,
            model_hash=self.model_hash, export_dir=self.export_dir,
            runtime=self.runtime, action_guard=self.action_guard,
            rtc_adapter=self.rtc_adapter,
            consecutive_crash_count=count,
        )


def make_single_policy(
    *,
    model_id: str,
    model_hash: str,
    export_dir: str | Path,
    runtime: Any | None = None,
    action_guard: Any | None = None,
    rtc_adapter: Any | None = None,
) -> Policy:
    """Factory: build a single-policy Policy bound to the DEFAULT_SINGLE_POLICY_SLOT.

    Used by the server's load() in single-policy mode (the default deploy
    config). Wraps the existing serving state without changing it.
    """
    return Policy(
        slot=DEFAULT_SINGLE_POLICY_SLOT,
        model_id=model_id, model_hash=model_hash,
        export_dir=str(export_dir),
        runtime=runtime, action_guard=action_guard, rtc_adapter=rtc_adapter,
    )


def validate_split_and_no_rtc(
    *,
    split_a_percent: int,
    no_rtc: bool,
) -> None:
    """Validate the --policy-a / --policy-b / --split / --no-rtc flag
    combination per ADR Day 5 spec. Raises ValueError with operator-
    readable message on any violation.

    The 2-policy mode requires --no-rtc because RTC carry-over state is
    per-policy and cross-policy carry-over produces out-of-distribution
    actions.
    """
    if not (0 <= split_a_percent <= 100):
        raise ValueError(
            f"split_a_percent must be in [0, 100], got {split_a_percent}"
        )
    if not no_rtc:
        raise ValueError(
            "2-policy mode requires --no-rtc. RTC carry-over state is "
            "per-policy; cross-policy carry-over produces out-of-"
            "distribution actions. Pass --no-rtc to proceed."
        )


def validate_memory_for_two_policies(
    *,
    model_size_bytes: int,
    total_gpu_bytes: int,
    safety_factor: float = 0.7,
) -> None:
    """Refuse-to-load check: 2-policy mode requires ~2x VRAM. Per ADR
    Day 5: if `2 × model_size_bytes > safety_factor × total_bytes`,
    raise.

    Args:
        model_size_bytes: per-model VRAM footprint (sum of ONNX weights
            + optimizer state + activations buffer; the server's load()
            computes this).
        total_gpu_bytes: total VRAM available on the device.
        safety_factor: fraction of total_bytes we're willing to use
            for the two policies combined. 0.7 default leaves 30% for
            cuDNN workspace, IO buffers, OS, etc.

    Raises:
        ValueError: when 2-policy mode would exceed the safety factor.
    """
    if model_size_bytes <= 0:
        raise ValueError(
            f"model_size_bytes must be > 0, got {model_size_bytes}"
        )
    if total_gpu_bytes <= 0:
        raise ValueError(
            f"total_gpu_bytes must be > 0, got {total_gpu_bytes}"
        )
    if not (0 < safety_factor < 1):
        raise ValueError(
            f"safety_factor must be in (0, 1), got {safety_factor}"
        )
    needed = 2 * model_size_bytes
    budget = int(total_gpu_bytes * safety_factor)
    if needed > budget:
        raise ValueError(
            f"2-policy mode requires {needed / 1e9:.1f}GB VRAM but only "
            f"{budget / 1e9:.1f}GB ({safety_factor * 100:.0f}% of "
            f"{total_gpu_bytes / 1e9:.1f}GB) is available. "
            f"Either pick smaller models, run on a larger GPU, OR drop "
            f"to single-policy mode."
        )


__all__ = [
    "DEFAULT_SINGLE_POLICY_SLOT",
    "Policy",
    "make_single_policy",
    "validate_memory_for_two_policies",
    "validate_split_and_no_rtc",
]
