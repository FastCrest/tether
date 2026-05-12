"""Tests for Phase B 2/3 — SnapFlow backend + teacher_loader + libero_drop_gate.

These are the three NEW modules the distill architecture needs before
CLI wiring (Phase B 3/3). The tests here pin:
  - teacher_loader.resolve_policy_type (config-field inference)
  - teacher_loader.load_teacher (allowlist + error paths; mock lerobot)
  - SnapFlowBackend.fit (happy path + missing teacher_export + checkpoint format)
  - libero_drop_gate (skip, abort, pass, LIBERO-missing)

No GPU required. lerobot + real LIBERO are mocked throughout.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

import importlib.util
_LEROBOT_INSTALLED = importlib.util.find_spec("lerobot") is not None
requires_lerobot = pytest.mark.skipif(
    not _LEROBOT_INSTALLED,
    reason="requires [native]/[monolithic]/[rtc] extras (lerobot)",
)

from reflex.distill.teacher_loader import (
    V03_TEACHER_ALLOWLIST,
    LoadedTeacher,
    _resolve_teacher_path,
    load_teacher,
    resolve_policy_type,
)
from reflex.finetune.backends import TrainerContext
from reflex.finetune.backends.base import CheckpointResult
from reflex.finetune.backends.snapflow_backend import (
    DEFAULT_CONSISTENCY_ALPHA,
    SnapFlowBackend,
    _build_preprocessor,
    _build_velocity_adapters,
    _prepare_batch,
    _save_student_checkpoint,
    _write_provenance,
)
from reflex.finetune.config import FinetuneConfig
from reflex.finetune.hooks import HookRegistry
from reflex.finetune.hooks.libero_drop_gate import (
    DEFAULT_GATE_THRESHOLD_PP,
    _LiberoUnavailable,
    attach_to,
    libero_drop_gate,
)


# ---------------------------------------------------------------------------
# teacher_loader
# ---------------------------------------------------------------------------


class TestResolvePolicyType:
    def test_explicit_type_field_wins(self):
        assert resolve_policy_type({"type": "pi0"}) == "pi0"
        assert resolve_policy_type({"type": "pi05"}) == "pi05"
        assert resolve_policy_type({"type": "smolvla"}) == "smolvla"

    def test_smolvla_inference_from_keys(self):
        cfg = {"load_vlm_weights": True, "chunk_size": 50}
        assert resolve_policy_type(cfg) == "smolvla"

    def test_pi05_inferred_from_num_expert_layers(self):
        cfg = {"prefix_length": 48, "chunk_size": 50, "num_steps": 10, "num_expert_layers": 12}
        assert resolve_policy_type(cfg) == "pi05"

    def test_pi0_default_fallback(self):
        cfg = {"prefix_length": 48, "chunk_size": 50, "num_steps": 10}
        assert resolve_policy_type(cfg) == "pi0"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="could not infer"):
            resolve_policy_type({"random_field": 42})


class TestLoadTeacher:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="teacher export not found"):
            load_teacher(tmp_path / "nope")

    def test_missing_config_json_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing config.json"):
            load_teacher(tmp_path)

    def test_disallowed_policy_type_raises(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"type": "smolvla"}))
        with pytest.raises(ValueError, match="v0.3"):
            load_teacher(tmp_path, allowlist=frozenset({"pi0", "pi05"}))

    def test_happy_path_returns_loaded_teacher(self, tmp_path):
        """Mocks the lerobot import and verifies eval() + frozen params."""
        (tmp_path / "config.json").write_text(json.dumps({"type": "pi0"}))

        fake_policy = MagicMock()
        fake_policy.parameters.return_value = [
            MagicMock(requires_grad=True) for _ in range(3)
        ]

        with patch(
            "reflex.distill.teacher_loader._load_policy_for_type",
            return_value=fake_policy,
        ):
            loaded = load_teacher(tmp_path, device="cpu", dtype="fp32")
        assert isinstance(loaded, LoadedTeacher)
        assert loaded.policy_type == "pi0"
        assert loaded.checkpoint_dir == tmp_path
        fake_policy.eval.assert_called_once()
        # Every param had requires_grad=False set after load:
        for p in fake_policy.parameters.return_value:
            assert p.requires_grad is False

    def test_v03_allowlist_is_pi_family_only(self):
        assert V03_TEACHER_ALLOWLIST == frozenset({"pi0", "pi05"})


class TestResolveTeacherPath:
    def test_local_path_passes_through(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert _resolve_teacher_path(tmp_path) == tmp_path

    def test_hf_id_triggers_snapshot_download(self, tmp_path):
        """'lerobot/pi0_base' should resolve via huggingface_hub."""
        cached_dir = tmp_path / "cache" / "lerobot_pi0_base"
        cached_dir.mkdir(parents=True)
        with patch(
            "huggingface_hub.snapshot_download",
            return_value=str(cached_dir),
        ) as mock_dl:
            result = _resolve_teacher_path("lerobot/pi0_base")
        assert result == cached_dir
        mock_dl.assert_called_once_with(
            repo_id="lerobot/pi0_base", repo_type="model",
        )

    def test_nonexistent_local_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="HF repo id"):
            _resolve_teacher_path(tmp_path / "does_not_exist")

    def test_path_like_strings_dont_trigger_hf(self):
        """'./my_teacher' or '/abs/path' shouldn't hit the HF hub."""
        with pytest.raises(FileNotFoundError, match="HF repo id"):
            _resolve_teacher_path("./not_real")
        with pytest.raises(FileNotFoundError, match="HF repo id"):
            _resolve_teacher_path("/abs/nonexistent/path")


# ---------------------------------------------------------------------------
# snapflow_backend utilities
# ---------------------------------------------------------------------------


class TestPrepareBatch:
    def test_splits_action_from_obs(self):
        batch = {
            "action": torch.randn(4, 50, 7),
            "observation.images.top": torch.randn(4, 3, 224, 224),
            "observation.state": torch.randn(4, 7),
            "language.task": ["pick up block"] * 4,
        }
        action, noise, t, obs = _prepare_batch(batch, device="cpu")
        assert action.shape == (4, 50, 7)
        assert noise.shape == action.shape
        assert t.shape == (4,)
        assert (t >= 0).all() and (t <= 1).all()
        assert "action" not in obs
        assert "observation.images.top" in obs
        assert "language.task" in obs  # non-tensor passed through

    def test_noise_is_fresh_per_call(self):
        batch = {"action": torch.randn(2, 3, 4)}
        _, n1, _, _ = _prepare_batch(batch, device="cpu")
        _, n2, _, _ = _prepare_batch(batch, device="cpu")
        assert not torch.allclose(n1, n2)


@requires_lerobot
class TestBuildPreprocessor:
    def test_overrides_rename_map_and_device(self, tmp_path):
        """_build_preprocessor should inject the image_key_map into the
        rename_observations_processor step and set device_processor's
        device from the caller."""
        captured = {}

        class _Fake:
            @classmethod
            def from_pretrained(cls, path, config_filename, overrides):
                captured["path"] = path
                captured["config_filename"] = config_filename
                captured["overrides"] = overrides
                return "pipeline_stub"

        with patch(
            "lerobot.processor.pipeline.DataProcessorPipeline",
            _Fake,
        ):
            result = _build_preprocessor(
                teacher_path=tmp_path,
                image_key_map={"obs.img.image": "obs.img.base_0_rgb"},
                device="cuda",
            )
        assert result == "pipeline_stub"
        assert captured["config_filename"] == "policy_preprocessor.json"
        assert captured["overrides"]["rename_observations_processor"] == {
            "rename_map": {"obs.img.image": "obs.img.base_0_rgb"},
        }
        assert captured["overrides"]["device_processor"]["device"] == "cuda"

    def test_no_image_key_map_still_sets_device(self, tmp_path):
        captured = {}

        class _Fake:
            @classmethod
            def from_pretrained(cls, path, config_filename, overrides):
                captured["overrides"] = overrides
                return None

        with patch(
            "lerobot.processor.pipeline.DataProcessorPipeline", _Fake,
        ):
            _build_preprocessor(teacher_path=tmp_path, image_key_map=None, device="cpu")
        assert "rename_observations_processor" not in captured["overrides"]
        assert captured["overrides"]["device_processor"]["device"] == "cpu"


class TestBuildVelocityAdapters:
    def test_unknown_policy_type_raises(self):
        with pytest.raises(NotImplementedError, match="No velocity adapter"):
            _build_velocity_adapters(
                teacher=MagicMock(), student=MagicMock(), policy_type="gr00t_n1_5",
            )

    @requires_lerobot
    def test_pi_family_adapter_calls_denoise_step_with_prefix_cache(self):
        """The adapter should call policy.model.denoise_step with the
        lerobot signature (state, prefix_pad_masks, past_key_values, x_t,
        timestep). We mock the prefix-cache build path to isolate the
        behavior of the velocity fn itself.
        """
        from unittest.mock import patch as _patch

        teacher_mock = MagicMock()
        student_mock = MagicMock()
        teacher_mock.model.denoise_step.return_value = torch.full((1, 3, 4), 0.1)
        student_mock.model.denoise_step.return_value = torch.full((1, 3, 4), 0.2)

        canned_state = torch.zeros(1, 32)
        canned_past_kv = [("fake_cache",)]
        canned_prefix_masks = torch.ones(1, 8, dtype=torch.bool)

        def fake_build_prefix(policy, obs):
            return canned_past_kv, canned_prefix_masks, canned_state

        # Patch the helper via direct module attribute override since it's
        # a closure inside _build_velocity_adapters — easier to stub by
        # replacing the whole adapter construction path.
        x = torch.zeros(1, 3, 4)
        t = torch.rand(1)
        obs = {
            "observation.state": torch.zeros(1, 7),
            "observation.language.tokens": torch.zeros(1, 8, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(1, 8, dtype=torch.long),
            "observation.images.front": torch.zeros(1, 3, 224, 224),
        }

        # Stub the lerobot helpers on the policy mocks so the prefix-cache
        # path runs to completion without needing a real PI0Policy graph.
        for p_mock in (teacher_mock, student_mock):
            p_mock._preprocess_images.return_value = (
                [torch.zeros(1, 3, 224, 224)], [torch.ones(1, dtype=torch.bool)],
            )
            p_mock.prepare_state.return_value = canned_state
            p_mock.model.embed_prefix.return_value = (
                torch.zeros(1, 8, 256), canned_prefix_masks,
                torch.ones(1, 8, dtype=torch.long),
            )
            p_mock.model._prepare_attention_masks_4d.return_value = torch.ones(1, 1, 8, 8)
            p_mock.model.paligemma_with_expert.forward.return_value = (None, canned_past_kv)
            # state_proj.weight.dtype is read for the state-dtype cast; set it
            # to match canned_state so the cast is a no-op under mocks.
            p_mock.model.state_proj.weight.dtype = canned_state.dtype
            p_mock.model.action_in_proj.weight.dtype = x.dtype
            # config.max_state_dim gets compared with < — return int, not MagicMock
            p_mock.config.max_state_dim = canned_state.shape[-1]
            # Also stub _attn_implementation writeback (setattr on the nested
            # language_model.config / gemma_expert.model.config paths).
            p_mock.model.paligemma_with_expert.paligemma.model.language_model.config = MagicMock()
            p_mock.model.paligemma_with_expert.gemma_expert.model.config = MagicMock()

        # make_att_2d_masks is imported from lerobot inside the adapter.
        with _patch(
            "lerobot.policies.pi0.modeling_pi0.make_att_2d_masks",
            return_value=torch.ones(1, 8, 8, dtype=torch.bool),
        ):
            teacher_fn, student_fn = _build_velocity_adapters(
                teacher_mock, student_mock, "pi0",
            )
            vt = teacher_fn(x, t, **obs)
            vs = student_fn(x, t, **obs)

        # denoise_step called with the right kwargs (state + x_t + timestep +
        # prefix_pad_masks + past_key_values + target_time). Teacher path
        # always passes target_time=None (inference-like semantics).
        teacher_mock.model.denoise_step.assert_called_once()
        t_call = teacher_mock.model.denoise_step.call_args.kwargs
        assert set(t_call.keys()) == {
            "state", "prefix_pad_masks", "past_key_values",
            "x_t", "timestep", "target_time",
        }
        assert torch.allclose(t_call["x_t"], x)
        assert torch.allclose(t_call["timestep"], t)
        assert t_call["target_time"] is None

        assert torch.allclose(vt, torch.full_like(x, 0.1))
        assert torch.allclose(vs, torch.full_like(x, 0.2))

    @requires_lerobot
    def test_student_passes_target_time_through_denoise_step(self):
        """v0.3.1 SnapFlowPI0Pytorch: target_time is now routed through
        denoise_step → embed_suffix → target_time_embed_mlp. The backend
        adapter just forwards the kwarg; the semantic effect lives in
        SnapFlowPI0Pytorch itself (covered by test_snapflow_pi0_model)."""
        teacher_mock = MagicMock()
        student_mock = MagicMock()
        expected = torch.full((1, 3, 4), 0.5)
        student_mock.model.denoise_step.return_value = expected
        student_mock._preprocess_images.return_value = (
            [torch.zeros(1, 3, 224, 224)], [torch.ones(1, dtype=torch.bool)],
        )
        canned_state = torch.zeros(1, 32)
        student_mock.prepare_state.return_value = canned_state
        student_mock.model.state_proj.weight.dtype = canned_state.dtype
        student_mock.model.action_in_proj.weight.dtype = torch.float32
        student_mock.config.max_state_dim = canned_state.shape[-1]
        student_mock.model.embed_prefix.return_value = (
            torch.zeros(1, 8, 256),
            torch.ones(1, 8, dtype=torch.bool),
            torch.ones(1, 8, dtype=torch.long),
        )
        student_mock.model._prepare_attention_masks_4d.return_value = torch.ones(1, 1, 8, 8)
        student_mock.model.paligemma_with_expert.forward.return_value = (None, [("fake_cache",)])
        student_mock.model.paligemma_with_expert.paligemma.model.language_model.config = MagicMock()
        student_mock.model.paligemma_with_expert.gemma_expert.model.config = MagicMock()

        obs = {
            "observation.language.tokens": torch.zeros(1, 8, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(1, 8, dtype=torch.long),
        }

        from unittest.mock import patch as _patch
        with _patch(
            "lerobot.policies.pi0.modeling_pi0.make_att_2d_masks",
            return_value=torch.ones(1, 8, 8, dtype=torch.bool),
        ):
            _, student_fn = _build_velocity_adapters(teacher_mock, student_mock, "pi0")
            tt = torch.ones(1)
            student_fn(torch.zeros(1, 3, 4), torch.rand(1), target_time=tt, **obs)
            student_fn(torch.zeros(1, 3, 4), torch.rand(1), **obs)

        # First call: target_time=tt. Second call: target_time=None (default).
        calls = student_mock.model.denoise_step.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["target_time"] is tt
        assert calls[1].kwargs["target_time"] is None


class TestSaveCheckpoint:
    def test_writes_pretrained_model_dir(self, tmp_path):
        """config.json must stay canonical (no `_reflex_*` pollution).
        Caught by 2026-04-25 distill smoke v3 + v5: PI05Config strict-
        decode rejects unknown fields. Reflex distill metadata lives in
        distill_provenance.json (separate test below)."""
        fake = MagicMock()
        def save(path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text(json.dumps({"type": "pi0"}))
            (Path(path) / "model.safetensors").write_bytes(b"fake-weights")
        fake.save_pretrained.side_effect = save

        ckpt = _save_student_checkpoint(
            fake, tmp_path / "training" / "checkpoints", step=100,
        )
        assert ckpt == tmp_path / "training" / "checkpoints" / "00000100" / "pretrained_model"
        assert (ckpt / "model.safetensors").exists()
        cfg = json.loads((ckpt / "config.json").read_text())
        # Canonical type field preserved
        assert cfg["type"] == "pi0"
        # No reflex-distill pollution -- they would break PI05Config strict-decode
        assert "_reflex_distill_method" not in cfg
        assert "_reflex_distill_teacher_type" not in cfg


class TestWriteProvenance:
    def test_writes_expected_fields(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        _write_provenance(
            tmp_path, teacher_dir=Path("/teachers/pi0"),
            policy_type="pi0", steps=5000, consistency_alpha=1.0,
        )
        prov = json.loads((tmp_path / "distill_provenance.json").read_text())
        assert prov["method"] == "snapflow"
        assert prov["policy_type"] == "pi0"
        assert prov["steps"] == 5000
        assert "arxiv.org" in prov["paper"]


class TestSnapFlowBackendFit:
    def test_missing_teacher_export_returns_training_failed(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/pi0", dataset="lerobot/libero",
            output=tmp_path, num_steps=10, phase="distill", teacher_export=None,
        )
        log_path = tmp_path / "training_log.jsonl"
        log_path.touch()
        ctx = TrainerContext(config=cfg, hooks=HookRegistry(), training_log_path=log_path)

        result = SnapFlowBackend().fit(ctx)
        assert result.status == "training_failed"
        assert "teacher_export" in result.error

    def test_dataloader_failure_returns_training_failed(self, tmp_path):
        """Missing lerobot → dataloader build raises → backend returns error."""
        cfg = FinetuneConfig(
            base="lerobot/pi0", dataset="lerobot/libero",
            output=tmp_path, num_steps=2, phase="distill",
            teacher_export=str(tmp_path / "teacher"),
        )
        (tmp_path / "teacher").mkdir()
        (tmp_path / "teacher" / "config.json").write_text(json.dumps({"type": "pi0"}))
        log_path = tmp_path / "training_log.jsonl"
        log_path.touch()
        ctx = TrainerContext(config=cfg, hooks=HookRegistry(), training_log_path=log_path)

        # Mock load_teacher to succeed, but let dataloader build blow up
        # (simulates lerobot.datasets import failure).
        fake_policy = MagicMock()
        fake_policy.parameters.return_value = [
            MagicMock(requires_grad=False) for _ in range(2)
        ]
        fake_loaded = LoadedTeacher(
            policy=fake_policy, config={"type": "pi0"},
            policy_type="pi0", checkpoint_dir=tmp_path / "teacher",
        )
        # load_teacher + _build_dataloader are LAZY-imported inside fit(),
        # so we patch them at their source modules (not at snapflow_backend).
        # enable_snapflow has a real isinstance check that rejects MagicMock —
        # patch it to no-op since this test doesn't exercise that path.
        with patch(
            "reflex.distill.teacher_loader.load_teacher",
            return_value=fake_loaded,
        ), patch(
            "reflex.distill.snapflow_pi0_model.enable_snapflow",
            return_value=None,
        ), patch(
            "reflex.finetune.backends.snapflow_backend._build_velocity_adapters",
            return_value=(lambda *a, **kw: None, lambda *a, **kw: None),
        ), patch(
            "reflex.finetune.backends.snapflow_backend._build_dataloader",
            side_effect=ImportError("lerobot.datasets not available"),
        ):
            result = SnapFlowBackend().fit(ctx)
        assert result.status == "training_failed"
        assert "dataloader" in result.error.lower()


# ---------------------------------------------------------------------------
# libero_drop_gate
# ---------------------------------------------------------------------------


def _make_distill_ctx(tmp_path, threshold_pp: float = DEFAULT_GATE_THRESHOLD_PP,
                     skip: bool = False, phase: str = "distill"):
    extra = {"libero_gate_threshold_pp": threshold_pp}
    if skip:
        extra["libero_gate_skip"] = True
    cfg = FinetuneConfig(
        base="lerobot/pi0", dataset="lerobot/libero",
        output=tmp_path, num_steps=10, phase=phase,
        teacher_export=str(tmp_path / "teacher"),
        extra_lerobot_args=extra,
    )
    log = tmp_path / "training_log.jsonl"
    log.touch()
    return TrainerContext(config=cfg, hooks=HookRegistry(), training_log_path=log)


class TestLiberoDropGate:
    def test_skip_via_config(self, tmp_path):
        ctx = _make_distill_ctx(tmp_path, skip=True)
        libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=MagicMock())
        assert "force_abort" not in ctx.extra

    def test_non_distill_phase_skips(self, tmp_path):
        ctx = _make_distill_ctx(tmp_path, phase="train")
        libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=MagicMock())
        assert "force_abort" not in ctx.extra

    def test_libero_unavailable_silently_passes(self, tmp_path):
        ctx = _make_distill_ctx(tmp_path)
        with patch(
            "reflex.finetune.hooks.libero_drop_gate._run_teacher_student_rollouts",
            side_effect=_LiberoUnavailable("no libero"),
        ):
            libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=MagicMock())
        assert "force_abort" not in ctx.extra

    def test_gate_passes_when_within_threshold(self, tmp_path):
        ctx = _make_distill_ctx(tmp_path, threshold_pp=5.0)
        report = MagicMock()
        with patch(
            "reflex.finetune.hooks.libero_drop_gate._run_teacher_student_rollouts",
            return_value=(0.90, 0.87),  # 3 pp drop — within threshold
        ):
            libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=report)
        assert "force_abort" not in ctx.extra
        assert report.libero_drop_pp == pytest.approx(3.0, abs=0.01)

    def test_gate_vetoes_when_drop_exceeds_threshold(self, tmp_path):
        ctx = _make_distill_ctx(tmp_path, threshold_pp=5.0)
        report = MagicMock()
        with patch(
            "reflex.finetune.hooks.libero_drop_gate._run_teacher_student_rollouts",
            return_value=(0.92, 0.80),  # 12 pp drop — over threshold
        ):
            libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=report)
        assert ctx.extra["force_abort"] is True
        assert "12.0" in ctx.extra["abort_reason"] or "12" in ctx.extra["abort_reason"]
        assert report.libero_drop_pp == pytest.approx(12.0, abs=0.01)

    def test_rollout_crash_aborts(self, tmp_path):
        """Unexpected errors abort — it's safer than silently shipping."""
        ctx = _make_distill_ctx(tmp_path)
        with patch(
            "reflex.finetune.hooks.libero_drop_gate._run_teacher_student_rollouts",
            side_effect=RuntimeError("sim crashed"),
        ):
            libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=MagicMock())
        assert ctx.extra["force_abort"] is True
        assert "crashed" in ctx.extra["abort_reason"]


class TestAttachTo:
    def test_registers_default_handler(self, tmp_path):
        reg = HookRegistry()
        attach_to(reg)
        assert "on_postprocess" in reg

    def test_threshold_override_reaches_handler(self, tmp_path):
        reg = HookRegistry()
        attach_to(reg, threshold_pp=2.0)
        ctx = _make_distill_ctx(tmp_path)
        report = MagicMock()
        # With threshold=2.0 and 3pp drop, we expect veto
        with patch(
            "reflex.finetune.hooks.libero_drop_gate._run_teacher_student_rollouts",
            return_value=(0.90, 0.87),
        ):
            reg.run("on_postprocess", ctx, final_checkpoint_path=tmp_path, report=report)
        assert ctx.extra["force_abort"] is True
