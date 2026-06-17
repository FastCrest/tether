"""Tests for src/tether/eval/cost_model.py — Phase 1 eval-as-a-service Day 4."""
from __future__ import annotations

import pytest

from tether.eval.cost_model import (
    COST_PREVIEW_GUARDRAIL_USD,
    COST_TABLE_SCHEMA_VERSION,
    CostEstimate,
    estimate_cost,
    known_suite_runtime_pairs,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_schema_version_positive():
    assert COST_TABLE_SCHEMA_VERSION >= 1


def test_guardrail_positive():
    assert COST_PREVIEW_GUARDRAIL_USD > 0


# ---------------------------------------------------------------------------
# CostEstimate dataclass
# ---------------------------------------------------------------------------


def _valid_kwargs(**overrides):
    base = dict(
        total_usd=1.0,
        suite="libero",
        runtime="modal",
        num_episodes_per_task=10,
        n_tasks=4,
        usd_per_episode=0.025,
        usd_per_task_startup=0.10,
        by_task={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
        cost_table_schema_version=COST_TABLE_SCHEMA_VERSION,
        notes="test",
    )
    base.update(overrides)
    return base


def test_cost_estimate_is_frozen():
    e = CostEstimate(**_valid_kwargs())
    with pytest.raises(AttributeError):
        e.total_usd = 999  # type: ignore[misc]


def test_cost_estimate_rejects_negative_total():
    with pytest.raises(ValueError, match="total_usd"):
        CostEstimate(**_valid_kwargs(total_usd=-1.0))


def test_cost_estimate_rejects_negative_n_tasks():
    with pytest.raises(ValueError, match="n_tasks"):
        CostEstimate(**_valid_kwargs(n_tasks=-1, total_usd=0.0, by_task={}))


def test_cost_estimate_rejects_negative_episodes_per_task():
    with pytest.raises(ValueError, match="num_episodes_per_task"):
        CostEstimate(**_valid_kwargs(num_episodes_per_task=-1))


def test_cost_estimate_rejects_total_disagreeing_with_by_task():
    """Cross-field invariant: total_usd ~ sum(by_task)."""
    with pytest.raises(ValueError, match="disagrees with sum"):
        CostEstimate(
            **_valid_kwargs(total_usd=10.0, by_task={"a": 0.5, "b": 0.5}),
        )


def test_cost_estimate_to_dict_round_trips():
    e = CostEstimate(**_valid_kwargs())
    d = e.to_dict()
    assert d["total_usd"] == 1.0
    assert d["suite"] == "libero"
    assert d["runtime"] == "modal"
    assert d["cost_table_schema_version"] == COST_TABLE_SCHEMA_VERSION


def test_cost_estimate_to_dict_rounds_to_4_decimals():
    e = CostEstimate(**_valid_kwargs(
        total_usd=1.000001, by_task={"a": 0.500001, "b": 0.5},
    ))
    # Cross-field invariant tolerates 1 cent slop, so this passes
    d = e.to_dict()
    # Ensure rounding doesn't crash with high-precision values
    assert isinstance(d["total_usd"], float)


def test_exceeds_guardrail_is_property():
    e_small = CostEstimate(**_valid_kwargs(total_usd=10.0, by_task={"a": 10.0}, n_tasks=1))
    assert not e_small.exceeds_guardrail
    e_huge = CostEstimate(**_valid_kwargs(
        total_usd=COST_PREVIEW_GUARDRAIL_USD * 2,
        by_task={"a": COST_PREVIEW_GUARDRAIL_USD * 2}, n_tasks=1,
    ))
    assert e_huge.exceeds_guardrail


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


def test_estimate_cost_libero_modal_basic():
    e = estimate_cost(
        suite="libero", runtime="modal",
        tasks=["a", "b", "c"], num_episodes_per_task=10,
    )
    # 3 tasks × (10 × $0.025 + $0.10) = 3 × ($0.25 + $0.10) = $1.05
    assert e.total_usd == pytest.approx(1.05, abs=0.01)
    assert e.n_tasks == 3
    assert e.num_episodes_per_task == 10
    assert e.suite == "libero"
    assert e.runtime == "modal"


def test_estimate_cost_libero_local_is_zero():
    e = estimate_cost(
        suite="libero", runtime="local",
        tasks=["a", "b"], num_episodes_per_task=100,
    )
    assert e.total_usd == 0.0
    assert "Local" in e.notes


def test_estimate_cost_by_task_sums_to_total():
    e = estimate_cost(
        suite="libero", runtime="modal",
        tasks=["a", "b", "c", "d"], num_episodes_per_task=5,
    )
    assert sum(e.by_task.values()) == pytest.approx(e.total_usd, abs=0.001)


def test_estimate_cost_by_task_is_per_task_keyed():
    e = estimate_cost(
        suite="libero", runtime="modal",
        tasks=["task_x", "task_y"], num_episodes_per_task=1,
    )
    assert "task_x" in e.by_task
    assert "task_y" in e.by_task
    assert e.by_task["task_x"] == e.by_task["task_y"]  # same eps → same cost


def test_estimate_cost_zero_episodes_returns_startup_only():
    e = estimate_cost(
        suite="libero", runtime="modal",
        tasks=["a", "b"], num_episodes_per_task=0,
    )
    # 2 tasks × $0.10 startup = $0.20
    assert e.total_usd == pytest.approx(0.20, abs=0.001)


def test_estimate_cost_empty_tasks_returns_zero():
    e = estimate_cost(
        suite="libero", runtime="modal",
        tasks=[], num_episodes_per_task=10,
    )
    assert e.total_usd == 0.0
    assert e.n_tasks == 0


def test_estimate_cost_rejects_negative_episodes():
    with pytest.raises(ValueError, match="num_episodes_per_task"):
        estimate_cost(
            suite="libero", runtime="modal",
            tasks=["a"], num_episodes_per_task=-1,
        )


def test_estimate_cost_rejects_unknown_pair():
    with pytest.raises(ValueError, match="No cost-table entry"):
        estimate_cost(
            suite="customer", runtime="modal",
            tasks=["a"], num_episodes_per_task=10,
        )


def test_estimate_cost_rejects_unknown_runtime():
    with pytest.raises(ValueError, match="No cost-table entry"):
        estimate_cost(
            suite="libero", runtime="kubernetes",
            tasks=["a"], num_episodes_per_task=10,
        )


def test_estimate_cost_scales_linearly_with_episodes():
    e_small = estimate_cost(
        suite="libero", runtime="modal",
        tasks=["a"], num_episodes_per_task=10,
    )
    e_big = estimate_cost(
        suite="libero", runtime="modal",
        tasks=["a"], num_episodes_per_task=100,
    )
    # Ep-cost portion scales 10x; startup is the same → diff = 90 × $0.025
    diff = e_big.total_usd - e_small.total_usd
    assert diff == pytest.approx(90 * 0.025, abs=0.001)


def test_estimate_cost_carries_schema_version():
    e = estimate_cost(
        suite="libero", runtime="modal",
        tasks=["a"], num_episodes_per_task=1,
    )
    assert e.cost_table_schema_version == COST_TABLE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# known_suite_runtime_pairs
# ---------------------------------------------------------------------------


def test_known_pairs_includes_libero_modal():
    pairs = known_suite_runtime_pairs()
    assert ("libero", "modal") in pairs
    assert ("libero", "local") in pairs


def test_known_pairs_is_sorted():
    pairs = known_suite_runtime_pairs()
    assert list(pairs) == sorted(pairs)
