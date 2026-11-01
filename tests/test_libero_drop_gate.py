"""LIBERO drop-gate must not silently green-light an unverified distill.

The rollout harness (tether.libero_harness) isn't shipped today, so the gate
always hits the "unavailable" path. These tests pin that the outcome is
*recorded* (skipped_unavailable, never mistaken for a pass) and that
libero_gate_require=True makes it fail closed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tether.finetune.hooks import libero_drop_gate as gate_mod
from tether.finetune.hooks.libero_drop_gate import _LiberoUnavailable, libero_drop_gate
from tether.finetune.postprocess import PostprocessReport


def _ctx(**extra_args):
    cfg = SimpleNamespace(
        extra_lerobot_args=dict(extra_args),
        phase="distill",
        teacher_export="teacher.onnx",
    )
    return SimpleNamespace(config=cfg, extra={})


def _payload():
    return {"final_checkpoint_path": "student.ckpt", "report": PostprocessReport()}


def test_unavailable_harness_is_recorded_not_silent(monkeypatch):
    monkeypatch.setattr(
        gate_mod, "_run_teacher_student_rollouts",
        lambda **k: (_ for _ in ()).throw(_LiberoUnavailable("no harness")),
    )
    ctx, payload = _ctx(), _payload()
    libero_drop_gate(ctx, **payload)
    assert payload["report"].libero_gate_status == "skipped_unavailable"
    assert "force_abort" not in ctx.extra  # default: permissive, ships


def test_require_makes_unavailable_fail_closed(monkeypatch):
    monkeypatch.setattr(
        gate_mod, "_run_teacher_student_rollouts",
        lambda **k: (_ for _ in ()).throw(_LiberoUnavailable("no harness")),
    )
    ctx, payload = _ctx(libero_gate_require=True), _payload()
    libero_drop_gate(ctx, **payload)
    assert ctx.extra.get("force_abort") is True
    assert "harness unavailable" in ctx.extra["abort_reason"]
    assert payload["report"].libero_gate_status == "skipped_unavailable"


def test_pass_records_status_and_drop(monkeypatch):
    monkeypatch.setattr(
        gate_mod, "_run_teacher_student_rollouts",
        lambda **k: (0.90, 0.89),  # 1pp drop, under 5pp threshold
    )
    ctx, payload = _ctx(), _payload()
    libero_drop_gate(ctx, **payload)
    assert payload["report"].libero_gate_status == "passed"
    assert payload["report"].libero_drop_pp == pytest.approx(1.0, abs=1e-6)
    assert "force_abort" not in ctx.extra


def test_fail_aborts(monkeypatch):
    monkeypatch.setattr(
        gate_mod, "_run_teacher_student_rollouts",
        lambda **k: (0.90, 0.80),  # 10pp drop, over 5pp threshold
    )
    ctx, payload = _ctx(), _payload()
    libero_drop_gate(ctx, **payload)
    assert payload["report"].libero_gate_status == "failed"
    assert ctx.extra.get("force_abort") is True


def test_skip_flag_recorded(monkeypatch):
    ctx, payload = _ctx(libero_gate_skip=True), _payload()
    libero_drop_gate(ctx, **payload)
    assert payload["report"].libero_gate_status == "skipped_disabled"
    assert "force_abort" not in ctx.extra


def test_non_distill_phase_recorded():
    ctx, payload = _ctx(), _payload()
    ctx.config.phase = "train"
    libero_drop_gate(ctx, **payload)
    assert payload["report"].libero_gate_status == "skipped_phase"
