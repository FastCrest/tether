"""Tests for Phase A shared finetune infra — prerequisite for `tether distill`.

3 new modules under test:
  - src/tether/finetune/backends/base.py   (Backend Protocol + context types)
  - src/tether/finetune/hooks/__init__.py  (HookRegistry)
  - src/tether/finetune/postprocess.py     (finalize() chain)

These are the three Phase-A files the architecture doc said must land
before distill can plug in. Tests here pin their contracts so future
backends can trust the interface.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tether.finetune.backends import Backend, CheckpointResult, TrainerContext, resolve_backend
from tether.finetune.config import FinetuneConfig
from tether.finetune.hooks import HookRegistry, LIFECYCLE_HOOKS


class TestHookRegistry:
    def test_empty_registry_runs_ok(self):
        r = HookRegistry()
        r.run("on_start", ctx=None)  # no handlers → no-op

    def test_handler_fires_with_payload(self):
        r = HookRegistry()
        calls = []
        r.register("on_step", lambda ctx, **kw: calls.append(kw))
        r.run("on_step", ctx=None, step=7, loss=0.42)
        assert calls == [{"step": 7, "loss": 0.42}]

    def test_multiple_handlers_fire_in_order(self):
        r = HookRegistry()
        order = []
        r.register("on_step", lambda ctx, **_: order.append("a"))
        r.register("on_step", lambda ctx, **_: order.append("b"))
        r.register("on_step", lambda ctx, **_: order.append("c"))
        r.run("on_step", ctx=None)
        assert order == ["a", "b", "c"]

    def test_handler_exception_propagates(self):
        """Architecture contract: hook crashes should NOT be swallowed —
        training needs to surface them."""
        r = HookRegistry()
        def _boom(ctx, **_):
            raise RuntimeError("intentional")
        r.register("on_step", _boom)
        with pytest.raises(RuntimeError, match="intentional"):
            r.run("on_step", ctx=None)

    def test_unknown_hook_name_is_silent(self):
        """Soft contract: unknown hook names don't reject registration
        (lets new hook points land without registry-coupling)."""
        r = HookRegistry()
        r.register("some_future_hook", lambda ctx, **_: None)
        r.run("some_future_hook", ctx=None)  # no error

    def test_lifecycle_hooks_documented(self):
        """Guardrail: if someone drops a hook from LIFECYCLE_HOOKS, this
        test catches it. Helps prevent silent API breakage."""
        expected = {"on_start", "on_step", "on_checkpoint", "on_end", "on_postprocess"}
        assert set(LIFECYCLE_HOOKS) == expected

    def test_clear_removes_handlers(self):
        r = HookRegistry()
        r.register("on_step", lambda ctx, **_: None)
        assert "on_step" in r
        r.clear("on_step")
        assert "on_step" not in r

    def test_handlers_view_is_read_only(self):
        r = HookRegistry()
        r.register("on_step", lambda ctx, **_: None)
        lst = r.handlers("on_step")
        lst.clear()  # external mutation shouldn't affect registry
        assert len(r.handlers("on_step")) == 1


class TestTrainerContext:
    def test_fields_populate(self, tmp_path):
        ctx = TrainerContext(
            config=MagicMock(),
            hooks=HookRegistry(),
            training_log_path=tmp_path / "log.jsonl",
        )
        assert ctx.teacher_path is None
        assert ctx.extra == {}

    def test_teacher_path_for_distill(self, tmp_path):
        teacher = tmp_path / "teacher_export"
        ctx = TrainerContext(
            config=MagicMock(),
            hooks=HookRegistry(),
            training_log_path=tmp_path / "log.jsonl",
            teacher_path=teacher,
        )
        assert ctx.teacher_path == teacher


class TestCheckpointResult:
    def test_default_status_ok(self, tmp_path):
        r = CheckpointResult(
            final_checkpoint_path=tmp_path,
            training_steps_completed=1000,
        )
        assert r.status == "ok"
        assert r.error is None

    def test_failed_status_carries_error(self, tmp_path):
        r = CheckpointResult(
            final_checkpoint_path=tmp_path,
            training_steps_completed=0,
            status="training_failed",
            error="subprocess rc=42",
        )
        assert r.status == "training_failed"
        assert "42" in r.error


class TestBackendProtocol:
    def test_lerobot_backend_conforms(self):
        from tether.finetune.backends.lerobot_backend import LerobotBackend
        b = LerobotBackend()
        assert isinstance(b, Backend)  # runtime_checkable protocol

    def test_non_backend_rejected(self):
        class NotABackend:
            pass
        assert not isinstance(NotABackend(), Backend)


class TestResolveBackend:
    def test_default_phase_is_train(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )
        assert cfg.phase == "train"

    def test_train_resolves_to_lerobot_backend(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )
        from tether.finetune.backends.lerobot_backend import LerobotBackend
        b = resolve_backend(cfg)
        assert isinstance(b, LerobotBackend)

    def test_distill_with_unknown_method_rejected(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/pi0_base",
            dataset="lerobot/libero",
            output=tmp_path,
            phase="distill",
            distillation_method="consistency",  # v0.5+ only
        )
        with pytest.raises(NotImplementedError, match="v0.5"):
            resolve_backend(cfg)

    def test_distill_snapflow_returns_snapflow_backend(self, tmp_path):
        """After Phase B 2/3, SnapFlowBackend exists — resolve_backend
        should return an instance for phase='distill', method='snapflow'."""
        from tether.finetune.backends.snapflow_backend import SnapFlowBackend
        cfg = FinetuneConfig(
            base="lerobot/pi0_base",
            dataset="lerobot/libero",
            output=tmp_path,
            phase="distill",
            distillation_method="snapflow",
        )
        backend = resolve_backend(cfg)
        assert isinstance(backend, SnapFlowBackend)

    def test_distill_unsupported_method_raises(self, tmp_path):
        """'consistency' is reserved for v0.5+ (GR00T DDPM)."""
        cfg = FinetuneConfig(
            base="lerobot/pi0_base",
            dataset="lerobot/libero",
            output=tmp_path,
            phase="distill",
            distillation_method="consistency",
        )
        with pytest.raises(NotImplementedError, match="v0.5"):
            resolve_backend(cfg)

    def test_unknown_phase_rejected(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            phase="pretrain",  # not in the enum
        )
        with pytest.raises(ValueError, match="Unknown phase"):
            resolve_backend(cfg)


class TestPostprocess:
    def test_finalize_export_failure_returns_export_failed(self, tmp_path):
        from tether.finetune.postprocess import finalize

        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            skip_preflight=True,
        )
        ctx = TrainerContext(
            config=cfg,
            hooks=HookRegistry(),
            training_log_path=tmp_path / "log.jsonl",
        )
        # Fake checkpoint path exists (required for the _auto_export stub)
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        ckpt_result = CheckpointResult(
            final_checkpoint_path=ckpt,
            training_steps_completed=100,
        )

        with patch(
            "tether.finetune.run._auto_export",
            return_value=(None, "torch.onnx.export raised: OOM"),
        ):
            result = finalize(ctx, ckpt_result)

        assert result.status == "export_failed"
        assert "OOM" in (result.error or "")

    def test_finalize_fires_on_postprocess_hook(self, tmp_path):
        from tether.finetune.postprocess import finalize

        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            skip_preflight=True,
            skip_export=True,  # skip export to isolate the hook test
        )
        ctx = TrainerContext(
            config=cfg,
            hooks=HookRegistry(),
            training_log_path=tmp_path / "log.jsonl",
        )
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        hook_calls = []
        ctx.hooks.register(
            "on_postprocess",
            lambda ctx, **kw: hook_calls.append(kw),
        )
        ckpt_result = CheckpointResult(
            final_checkpoint_path=ckpt,
            training_steps_completed=100,
        )

        result = finalize(ctx, ckpt_result)

        assert result.status == "ok"
        assert len(hook_calls) == 1
        assert "onnx_path" in hook_calls[0]
        assert "report" in hook_calls[0]

    def test_hook_can_veto_ship(self, tmp_path):
        """libero_drop_gate pattern: a handler sets ctx.extra
        ["force_abort"]=True and finalize returns status='aborted'."""
        from tether.finetune.postprocess import finalize

        cfg = FinetuneConfig(
            base="lerobot/pi0_base",
            dataset="lerobot/libero",
            output=tmp_path,
            skip_preflight=True,
            skip_export=True,
        )
        ctx = TrainerContext(
            config=cfg,
            hooks=HookRegistry(),
            training_log_path=tmp_path / "log.jsonl",
        )
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        ckpt_result = CheckpointResult(
            final_checkpoint_path=ckpt,
            training_steps_completed=2000,
        )

        # Register a libero-drop-gate-style handler that vetoes
        def _drop_gate(ctx, **_):
            ctx.extra["force_abort"] = True
            ctx.extra["abort_reason"] = "student task success dropped 12pp"
        ctx.hooks.register("on_postprocess", _drop_gate)

        result = finalize(ctx, ckpt_result)

        assert result.status == "aborted"
        assert "12pp" in (result.error or "")


class TestFinetuneConfigPhaseField:
    """The new fields on FinetuneConfig that distill needs."""

    def test_phase_defaults_to_train(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )
        assert cfg.phase == "train"
        assert cfg.teacher_export is None
        assert cfg.distillation_method == "snapflow"

    def test_phase_distill_carries_teacher(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/pi0_base",
            dataset="lerobot/libero",
            output=tmp_path,
            phase="distill",
            teacher_export="./teacher_dir",
            distillation_method="snapflow",
        )
        assert cfg.phase == "distill"
        assert cfg.teacher_export == "./teacher_dir"
