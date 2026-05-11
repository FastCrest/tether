"""Unit tests for src/reflex/exporters/_export_mode.py.

Mock-based — verifies auto-detection logic without needing real GPU.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import reflex.exporters.decomposed as decomposed
from reflex.exporters._export_mode import (
    ExportMode,
    ExportModeDecision,
    InsufficientVRAMError,
    estimate_model_vram_from_onnx,
    select_mode,
)


# 1.6 GB ONNX (typical SmolVLA monolithic) → 6.4 GB estimated VRAM per export
SMOLVLA_ONNX_SIZE = int(1.6 * 1024 ** 3)
SMOLVLA_VRAM = estimate_model_vram_from_onnx(SMOLVLA_ONNX_SIZE)


def test_estimate_uses_4x_multiplier():
    assert estimate_model_vram_from_onnx(1_000_000_000) == 4_000_000_000


# ─── Sequential mode (always works) ──────────────────────────────────────────


def test_sequential_explicit_returns_sequential():
    decision = select_mode(ExportMode.SEQUENTIAL, SMOLVLA_VRAM)
    assert decision.mode == ExportMode.SEQUENTIAL
    assert "explicit" in decision.reason.lower()


def test_sequential_explicit_works_on_cpu():
    """Sequential should work even with no GPU."""
    with patch("reflex.exporters._export_mode.probe_free_vram", return_value=None):
        decision = select_mode(ExportMode.SEQUENTIAL, SMOLVLA_VRAM)
    assert decision.mode == ExportMode.SEQUENTIAL


# ─── Parallel mode (fails loudly per CLAUDE.md) ──────────────────────────────


def test_parallel_explicit_fits_returns_parallel():
    """24 GB free, model needs ~14 GB combined → parallel."""
    with patch("reflex.exporters._export_mode.probe_free_vram",
               return_value=24 * 1024 ** 3):
        decision = select_mode(ExportMode.PARALLEL, SMOLVLA_VRAM)
    assert decision.mode == ExportMode.PARALLEL


def test_parallel_explicit_doesnt_fit_raises():
    """8 GB free (Orin Nano-ish), model needs ~14 GB combined → raise loudly."""
    with patch("reflex.exporters._export_mode.probe_free_vram",
               return_value=8 * 1024 ** 3):
        with pytest.raises(InsufficientVRAMError, match="parallel needs"):
            select_mode(ExportMode.PARALLEL, SMOLVLA_VRAM)


def test_parallel_explicit_no_gpu_raises():
    """No GPU at all → raise loudly, don't silently fall back."""
    with patch("reflex.exporters._export_mode.probe_free_vram", return_value=None):
        with pytest.raises(InsufficientVRAMError, match="No GPU detected"):
            select_mode(ExportMode.PARALLEL, SMOLVLA_VRAM)


# ─── Auto mode (the smart picker) ────────────────────────────────────────────


def test_auto_no_gpu_picks_sequential():
    with patch("reflex.exporters._export_mode.probe_free_vram", return_value=None):
        decision = select_mode(ExportMode.AUTO, SMOLVLA_VRAM)
    assert decision.mode == ExportMode.SEQUENTIAL
    assert "no gpu" in decision.reason.lower()


def test_auto_low_vram_picks_sequential():
    """Orin Nano 8 GB shared → sequential."""
    with patch("reflex.exporters._export_mode.probe_free_vram",
               return_value=8 * 1024 ** 3):
        decision = select_mode(ExportMode.AUTO, SMOLVLA_VRAM)
    assert decision.mode == ExportMode.SEQUENTIAL
    assert "would need" in decision.reason.lower() or "8" in decision.reason


def test_auto_high_vram_picks_parallel():
    """RTX 5090 32 GB → parallel."""
    with patch("reflex.exporters._export_mode.probe_free_vram",
               return_value=32 * 1024 ** 3):
        decision = select_mode(ExportMode.AUTO, SMOLVLA_VRAM)
    assert decision.mode == ExportMode.PARALLEL


def test_auto_a10g_24gb_picks_parallel():
    """A10G 24 GB → parallel for SmolVLA (needs ~14 GB combined)."""
    with patch("reflex.exporters._export_mode.probe_free_vram",
               return_value=24 * 1024 ** 3):
        decision = select_mode(ExportMode.AUTO, SMOLVLA_VRAM)
    assert decision.mode == ExportMode.PARALLEL


def test_auto_t4_16gb_borderline():
    """T4 16 GB free is borderline for SmolVLA (~14 GB needed). Should go parallel."""
    with patch("reflex.exporters._export_mode.probe_free_vram",
               return_value=16 * 1024 ** 3):
        decision = select_mode(ExportMode.AUTO, SMOLVLA_VRAM)
    # 16 > 14.6 (2 * 6.4 + 1.0), so parallel
    assert decision.mode == ExportMode.PARALLEL


# ─── pi05 (much larger model) ────────────────────────────────────────────────


PI05_ONNX_SIZE = int(13.0 * 1024 ** 3)  # ~13 GB monolithic
PI05_VRAM = estimate_model_vram_from_onnx(PI05_ONNX_SIZE)


def test_auto_t4_picks_sequential_for_pi05():
    """T4 16 GB can't fit 2x pi05 (~104 GB needed) → sequential."""
    with patch("reflex.exporters._export_mode.probe_free_vram",
               return_value=16 * 1024 ** 3):
        decision = select_mode(ExportMode.AUTO, PI05_VRAM)
    assert decision.mode == ExportMode.SEQUENTIAL


def test_auto_a100_80gb_picks_sequential_for_pi05():
    """Even A100 80 GB can't fit 2x pi05 (104 GB needed) → sequential."""
    with patch("reflex.exporters._export_mode.probe_free_vram",
               return_value=80 * 1024 ** 3):
        decision = select_mode(ExportMode.AUTO, PI05_VRAM)
    assert decision.mode == ExportMode.SEQUENTIAL


# ─── pi05 decomposed exporter dispatch ──────────────────────────────────────


def test_pi05_decomposed_dispatches_to_sequential(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        decomposed,
        "select_mode",
        lambda requested, estimated: ExportModeDecision(ExportMode.SEQUENTIAL, "test seq"),
    )

    def fake_sequential(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "export_mode": "sequential"}

    monkeypatch.setattr(decomposed, "_export_pi05_decomposed_sequential", fake_sequential)
    monkeypatch.setattr(
        decomposed,
        "_export_pi05_decomposed_parallel",
        lambda **kwargs: pytest.fail("parallel path should not run"),
    )

    result = decomposed.export_pi05_decomposed(
        "lerobot/pi05_libero_finetuned_v044",
        tmp_path,
        export_mode=ExportMode.AUTO,
    )

    assert result["export_mode"] == "sequential"
    assert calls[0]["output_dir"] == tmp_path
    assert calls[0]["export_mode_reason"] == "test seq"


def test_pi05_decomposed_dispatches_to_parallel(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        decomposed,
        "select_mode",
        lambda requested, estimated: ExportModeDecision(ExportMode.PARALLEL, "test parallel"),
    )

    def fake_parallel(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "export_mode": "parallel"}

    monkeypatch.setattr(decomposed, "_export_pi05_decomposed_parallel", fake_parallel)
    monkeypatch.setattr(
        decomposed,
        "_export_pi05_decomposed_sequential",
        lambda **kwargs: pytest.fail("sequential path should not run"),
    )

    result = decomposed.export_pi05_decomposed(
        "lerobot/pi05_libero_finetuned_v044",
        tmp_path,
        export_mode="parallel",
    )

    assert result["export_mode"] == "parallel"
    assert calls[0]["output_dir"] == tmp_path
    assert calls[0]["export_mode_reason"] == "test parallel"


def test_pi05_decomposed_sequential_reuses_single_policy(monkeypatch, tmp_path):
    policy = object()
    load_calls = []
    pass_order = []

    def fake_load_policy(model_id, num_steps, student_checkpoint, variant):
        load_calls.append((model_id, num_steps, student_checkpoint, variant))
        return policy

    def fake_prefix_pass(policy_arg, output_dir, past_kv_names):
        assert policy_arg is policy
        pass_order.append("prefix")
        (output_dir / "vlm_prefix.onnx").write_bytes(b"prefix")
        return {"chunk_size": 50, "action_dim": 32, "prefix_seq_len": 968}

    def fake_expert_pass(**kwargs):
        assert kwargs["policy"] is policy
        pass_order.append("expert")
        (kwargs["output_dir"] / "expert_denoise.onnx").write_bytes(b"expert")
        return {"chunk_size": 50, "action_dim": 32, "prefix_seq_len": 968}

    monkeypatch.setattr(decomposed, "_load_pi05_policy", fake_load_policy)
    monkeypatch.setattr(decomposed, "_export_pi05_prefix_pass", fake_prefix_pass)
    monkeypatch.setattr(decomposed, "_export_pi05_expert_pass", fake_expert_pass)

    result = decomposed._export_pi05_decomposed_sequential(
        model_id="lerobot/pi05_libero_finetuned_v044",
        output_dir=tmp_path,
        num_steps=10,
        target="desktop",
        student_checkpoint=None,
        variant="default",
        export_mode_reason="forced sequential",
    )

    cfg = json.loads((tmp_path / "reflex_config.json").read_text())
    assert load_calls == [("lerobot/pi05_libero_finetuned_v044", 10, None, "default")]
    assert pass_order == ["prefix", "expert"]
    assert result["export_mode"] == "sequential"
    assert cfg["export_mode"] == "sequential"


def test_run_parallel_pi05_exports_submits_both_workers(monkeypatch, tmp_path):
    calls = []

    class ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class ImmediatePool:
        def __init__(self, max_workers, mp_context):
            assert max_workers == 2
            assert mp_context is not None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            return ImmediateFuture(fn(*args))

    def fake_prefix_worker(*args):
        calls.append(("prefix", args))
        return {"chunk_size": 50, "action_dim": 32, "prefix_seq_len": 968}

    def fake_expert_worker(*args):
        calls.append(("expert", args))
        return {"chunk_size": 50, "action_dim": 32, "prefix_seq_len": 968}

    monkeypatch.setattr(decomposed, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(decomposed, "_export_pi05_prefix_worker", fake_prefix_worker)
    monkeypatch.setattr(decomposed, "_export_pi05_expert_worker", fake_expert_worker)

    prefix_meta, expert_meta = decomposed._run_parallel_pi05_exports(
        model_id="lerobot/pi05_libero_finetuned_v044",
        output_dir=tmp_path,
        num_steps=10,
        student_checkpoint=None,
        variant="default",
        past_kv_names=["past_k_0", "past_v_0"],
        prefix_seq_len=968,
    )

    assert prefix_meta["prefix_seq_len"] == 968
    assert expert_meta["prefix_seq_len"] == 968
    assert [call[0] for call in calls] == ["prefix", "expert"]
    assert calls[0][1][1] == str(tmp_path)
    # _export_pi05_expert_worker signature:
    # (model_id, output_dir, num_steps, student_checkpoint, variant,
    #  past_kv_names, prefix_seq_len, per_step_expert)
    # prefix_seq_len is positional arg 6 (0-indexed).
    assert calls[1][1][6] == 968


def test_write_decomposed_result_records_export_mode(tmp_path):
    (tmp_path / "vlm_prefix.onnx").write_bytes(b"prefix")
    (tmp_path / "expert_denoise.onnx").write_bytes(b"expert")

    result = decomposed._write_decomposed_export_result(
        model_id="lerobot/pi05_libero_finetuned_v044",
        output_dir=tmp_path,
        num_steps=10,
        target="desktop",
        student_checkpoint=None,
        variant="default",
        past_kv_names=["past_k_0", "past_v_0"],
        chunk_size=50,
        action_dim=32,
        export_mode=ExportMode.PARALLEL,
        export_mode_reason="test reason",
    )

    cfg = json.loads((tmp_path / "reflex_config.json").read_text())
    assert result["export_mode"] == "parallel"
    assert cfg["export_mode"] == "parallel"
    assert cfg["export_mode_reason"] == "test reason"
