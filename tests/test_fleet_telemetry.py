"""Tests for fleet-telemetry wiring: --robot-id → tether_robot_info gauge.

The fleet-telemetry feature publishes a single info-style gauge per process so
Grafana can join hot metrics against robot_id via `instance`. No extra label
cardinality on histograms — backward-compat for single-robot deploys.
"""
from __future__ import annotations

import pytest

from tether.observability import REGISTRY, set_robot_info
from tether.observability.prometheus import tether_robot_info


def _read_info_samples():
    """Collect all label-sets present on tether_robot_info from the REGISTRY."""
    samples = []
    for metric in REGISTRY.collect():
        if metric.name != "tether_robot_info":
            continue
        for sample in metric.samples:
            samples.append(sample)
    return samples


def test_robot_info_publishes_gauge_value_one():
    set_robot_info(robot_id="test-robot-A", embodiment="franka", model_id="pi0-libero")
    samples = _read_info_samples()
    matching = [
        s for s in samples
        if s.labels.get("robot_id") == "test-robot-A"
    ]
    assert matching, "expected a tether_robot_info sample for test-robot-A"
    assert matching[0].value == 1.0
    assert matching[0].labels["embodiment"] == "franka"
    assert matching[0].labels["model_id"] == "pi0-libero"


def test_robot_info_is_idempotent_for_same_labels():
    """Repeated set_robot_info with the same labels does not duplicate series."""
    set_robot_info(robot_id="robot-B", embodiment="so100", model_id="smolvla")
    before = [
        s for s in _read_info_samples()
        if s.labels.get("robot_id") == "robot-B"
    ]
    set_robot_info(robot_id="robot-B", embodiment="so100", model_id="smolvla")
    after = [
        s for s in _read_info_samples()
        if s.labels.get("robot_id") == "robot-B"
    ]
    assert len(before) == len(after) == 1


def test_distinct_robot_ids_produce_distinct_series():
    set_robot_info(robot_id="robot-C1", embodiment="franka", model_id="pi0")
    set_robot_info(robot_id="robot-C2", embodiment="franka", model_id="pi0")
    matching = [
        s for s in _read_info_samples()
        if s.labels.get("robot_id") in {"robot-C1", "robot-C2"}
    ]
    assert len(matching) == 2
    assert {s.labels["robot_id"] for s in matching} == {"robot-C1", "robot-C2"}


def test_robot_info_metric_is_registered_with_correct_label_keys():
    """Guards against accidental label-key edits that would break Grafana joins."""
    assert set(tether_robot_info._labelnames) == {"robot_id", "embodiment", "model_id"}
