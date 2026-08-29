from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tether.export_config import build_producer_config, write_tether_config
from tether.runtime.verify_inference import (
    MonolithicOnnxVerificationAdapter,
    UnsupportedVerificationBackend,
    _require_session_device,
    load_verification_inference,
)


def _value_info(onnx, name: str, dtype: int, shape: list[int]):
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def _save_monolithic(root: Path) -> dict:
    onnx = pytest.importorskip("onnx")
    inputs = [
        _value_info(onnx, name, onnx.TensorProto.FLOAT, [1, 3, 8, 8])
        for name in ("img_base", "img_wrist_l", "img_wrist_r")
    ]
    inputs += [
        _value_info(onnx, name, onnx.TensorProto.BOOL, [1])
        for name in ("mask_base", "mask_wrist_l", "mask_wrist_r")
    ]
    inputs += [
        _value_info(onnx, "lang_tokens", onnx.TensorProto.INT64, [1, 4]),
        _value_info(onnx, "lang_masks", onnx.TensorProto.BOOL, [1, 4]),
        _value_info(onnx, "state", onnx.TensorProto.FLOAT, [1, 2]),
        _value_info(onnx, "noise", onnx.TensorProto.FLOAT, [1, 2, 2]),
    ]
    output = _value_info(onnx, "actions", onnx.TensorProto.FLOAT, [1, 2, 2])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["noise"], ["actions"])],
        "verify-monolithic",
        inputs,
        [output],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.save(model, root / "model.onnx")
    config = build_producer_config(
        root,
        producer="monolithic",
        model_id="org/native-reference",
        model_type="pi0",
        action_dim=2,
        num_denoising_steps=1,
        opset=17,
        metadata={"chunk_size": 2, "action_chunk_size": 2, "max_state_dim": 2},
    )
    write_tether_config(root, config)
    return config


def _save_pi05_split(root: Path) -> dict:
    onnx = pytest.importorskip("onnx")
    image_inputs = [
        _value_info(onnx, name, onnx.TensorProto.FLOAT, [1, 3, 8, 8])
        for name in ("img_base", "img_wrist_l", "img_wrist_r")
    ]
    mask_inputs = [
        _value_info(onnx, name, onnx.TensorProto.BOOL, [1])
        for name in ("mask_base", "mask_wrist_l", "mask_wrist_r")
    ]
    prefix_inputs = (
        image_inputs
        + mask_inputs
        + [
            _value_info(onnx, "lang_tokens", onnx.TensorProto.INT64, [1, 4]),
            _value_info(onnx, "lang_masks", onnx.TensorProto.BOOL, [1, 4]),
        ]
    )
    prefix_outputs = [
        _value_info(onnx, "past0", onnx.TensorProto.FLOAT, [1, 3, 8, 8]),
        _value_info(onnx, "prefix_pad_masks", onnx.TensorProto.BOOL, [1]),
    ]
    prefix_graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Identity", ["img_base"], ["past0"]),
            onnx.helper.make_node("Identity", ["mask_base"], ["prefix_pad_masks"]),
        ],
        "verify-prefix",
        prefix_inputs,
        prefix_outputs,
    )
    prefix_model = onnx.helper.make_model(
        prefix_graph,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    prefix_model.ir_version = 8
    onnx.save(prefix_model, root / "vlm_prefix.onnx")

    expert_inputs = [
        _value_info(onnx, "past0", onnx.TensorProto.FLOAT, [1, 3, 8, 8]),
        _value_info(onnx, "prefix_pad_masks", onnx.TensorProto.BOOL, [1]),
        _value_info(onnx, "noise", onnx.TensorProto.FLOAT, [1, 2, 2]),
    ]
    expert_output = _value_info(onnx, "actions", onnx.TensorProto.FLOAT, [1, 2, 2])
    expert_graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["noise"], ["actions"])],
        "verify-expert",
        expert_inputs,
        [expert_output],
    )
    expert_model = onnx.helper.make_model(
        expert_graph,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    expert_model.ir_version = 8
    onnx.save(expert_model, root / "expert_denoise.onnx")

    config = build_producer_config(
        root,
        producer="pi05_split",
        model_id="org/pi05-native",
        model_type="pi05",
        action_dim=2,
        num_denoising_steps=1,
        opset=17,
        metadata={
            "action_chunk_size": 2,
            "decomposed": {
                "vlm_prefix_onnx": "vlm_prefix.onnx",
                "expert_denoise_onnx": "expert_denoise.onnx",
                "past_kv_tensor_names": ["past0"],
                "paligemma_layers": 1,
                "per_step_expert": False,
                "expert_takes_state": False,
            },
        },
    )
    write_tether_config(root, config)
    return config


def _protocol_inputs(noise: np.ndarray) -> dict:
    image = np.zeros((1, 3, 8, 8), dtype=np.float32)
    mask = np.ones((1,), dtype=np.bool_)
    return {
        "img_base": image,
        "img_wrist_l": image,
        "img_wrist_r": image,
        "mask_base": mask,
        "mask_wrist_l": mask,
        "mask_wrist_r": mask,
        "lang_tokens": np.zeros((1, 4), dtype=np.int64),
        "lang_masks": np.ones((1, 4), dtype=np.bool_),
        "noise": noise,
        "state": np.zeros((1, 2), dtype=np.float32),
        "episode_id": "episode-1",
    }


def test_monolithic_loader_executes_real_onnx_artifact(tmp_path):
    pytest.importorskip("onnxruntime")
    root = tmp_path / "monolithic"
    root.mkdir()
    _save_monolithic(root)
    noise = np.arange(4, dtype=np.float32).reshape(1, 2, 2)

    inference = load_verification_inference(root, device="cpu")
    actions = inference.predict_action_chunk(**_protocol_inputs(noise))

    np.testing.assert_array_equal(actions, noise)
    assert inference.get_stats()["backend"] == "pi0_onnx_monolithic"


def test_pi05_split_loader_executes_both_real_onnx_artifacts(tmp_path):
    pytest.importorskip("onnxruntime")
    root = tmp_path / "split"
    root.mkdir()
    _save_pi05_split(root)
    noise = np.arange(4, dtype=np.float32).reshape(1, 2, 2)

    inference = load_verification_inference(root, device="cpu")
    actions = inference.predict_action_chunk(**_protocol_inputs(noise))

    np.testing.assert_array_equal(actions, noise)
    assert inference.get_stats()["misses"] == 1


def test_monolithic_adapter_preserves_false_camera_masks():
    calls = {}

    class Server:
        def predict(self, **kwargs):
            calls.update(kwargs)
            return {"actions": np.zeros((1, 2, 2), dtype=np.float32)}

    inputs = _protocol_inputs(np.zeros((1, 2, 2), dtype=np.float32))
    inputs["mask_wrist_r"] = np.zeros((1,), dtype=np.bool_)
    MonolithicOnnxVerificationAdapter(Server()).predict_action_chunk(**inputs)

    assert bool(calls["image_masks"][0][0]) is True
    assert bool(calls["image_masks"][2][0]) is False


def test_pi05_secondary_paths_must_match_hashed_manifest(tmp_path):
    root = tmp_path / "split-path-swap"
    root.mkdir()
    config = _save_pi05_split(root)
    (root / "undeclared.onnx").write_bytes((root / "expert_denoise.onnx").read_bytes())
    config["decomposed"]["expert_denoise_onnx"] = "undeclared.onnx"
    write_tether_config(root, config)

    with pytest.raises(UnsupportedVerificationBackend, match="secondary paths"):
        load_verification_inference(root)


def test_verification_session_rejects_cpu_cuda_backend_mismatch():
    cpu = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    cuda = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    _require_session_device(cpu, device="cpu", name="test")
    _require_session_device(cuda, device="cuda", name="test")
    with pytest.raises(UnsupportedVerificationBackend, match="device mismatch"):
        _require_session_device(cpu, device="cuda", name="test")
    with pytest.raises(UnsupportedVerificationBackend, match="device mismatch"):
        _require_session_device(cuda, device="cpu", name="test")


def test_expert_stack_is_rejected_until_conditioning_is_preserved(tmp_path):
    root = tmp_path / "expert-stack"
    root.mkdir()
    (root / "expert_stack.onnx").write_bytes(b"placeholder")
    write_tether_config(
        root,
        {
            "schema_version": 1,
            "model_id": "org/native",
            "model_type": "smolvla",
            "action_dim": 2,
            "num_denoising_steps": 1,
            "opset": 17,
            "export_kind": "decomposed_onnx",
            "artifacts": [{"path": "expert_stack.onnx", "role": "model"}],
            "io_contract": {"inputs": [], "outputs": []},
        },
    )
    with pytest.raises(UnsupportedVerificationBackend, match="image and language"):
        load_verification_inference(root)


def test_missing_and_digest_mismatched_artifacts_fail_before_runtime(tmp_path):
    root = tmp_path / "missing"
    root.mkdir()
    config = _save_monolithic(root)
    (root / "model.onnx").unlink()
    with pytest.raises(ValueError, match="is missing"):
        load_verification_inference(root)

    mismatch = tmp_path / "mismatch"
    mismatch.mkdir()
    config = _save_monolithic(mismatch)
    config["artifacts"][0]["sha256"] = hashlib.sha256(
        (mismatch / "model.onnx").read_bytes()
    ).hexdigest()
    write_tether_config(mismatch, config)
    with (mismatch / "model.onnx").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_verification_inference(mismatch)


def test_corrupt_and_unsupported_artifacts_fail_loudly(tmp_path):
    pytest.importorskip("onnxruntime")
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "model.onnx").write_bytes(b"not-onnx")
    config = {
        "schema_version": 1,
        "model_id": "org/model",
        "model_type": "pi0",
        "action_dim": 2,
        "num_denoising_steps": 1,
        "opset": 17,
        "export_kind": "monolithic_onnx",
        "artifacts": [{"path": "model.onnx", "role": "model"}],
        "io_contract": {"inputs": [], "outputs": []},
    }
    write_tether_config(corrupt, config)
    with pytest.raises(Exception, match="INVALID_PROTOBUF|protobuf|Protobuf|model"):
        load_verification_inference(corrupt)

    unsupported = tmp_path / "unsupported"
    unsupported.mkdir()
    config.update(export_kind="config_only", artifacts=[])
    write_tether_config(unsupported, config)
    with pytest.raises(UnsupportedVerificationBackend, match="config_only"):
        load_verification_inference(unsupported)

    for export_kind in ("trt_engine", "triton_bundle"):
        root = tmp_path / export_kind
        root.mkdir()
        artifact = root / "model.bin"
        artifact.write_bytes(b"engine")
        config.update(
            export_kind=export_kind,
            artifacts=[{"path": artifact.name, "role": "engine"}],
        )
        write_tether_config(root, config)
        with pytest.raises(UnsupportedVerificationBackend, match=export_kind):
            load_verification_inference(root)


@pytest.mark.parametrize(
    ("model_type", "module_name", "class_name"),
    [
        ("pi05", "lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
        ("pi05_decomposed", "lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
        ("pi0", "lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
        ("smolvla", "lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
    ],
)
def test_exact_native_loader_uses_declared_family_and_ref(
    monkeypatch, model_type, module_name, class_name
):
    import importlib
    from tether.eval.libero_rollout import _load_exact_reference_policy

    seen = {}

    class Policy:
        @classmethod
        def from_pretrained(cls, checkpoint):
            seen["checkpoint"] = checkpoint
            return cls()

    def fake_import(name):
        seen["module"] = name
        return SimpleNamespace(**{class_name: Policy})

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert isinstance(_load_exact_reference_policy(model_type, "org/exact@sha"), Policy)
    assert seen == {"module": module_name, "checkpoint": "org/exact@sha"}


def test_exact_native_loader_preserves_snapflow_student_format(monkeypatch):
    import importlib
    from tether.eval.libero_rollout import _load_exact_reference_policy

    seen = {}

    def load_student(checkpoint):
        seen["checkpoint"] = checkpoint
        return "student"

    def fake_import(name):
        seen["module"] = name
        return SimpleNamespace(load_snapflow_student=load_student)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert _load_exact_reference_policy("pi05_decomposed_student", "/exact/student") == "student"
    assert seen == {
        "module": "tether.distill.snapflow_pi0_model",
        "checkpoint": "/exact/student",
    }


def test_receipt_smoke_action_validation_fails_closed():
    from scripts.verify_artifact_receipt import _validate_smoke_actions

    noise = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    _validate_smoke_actions(np.zeros_like(noise), noise)
    with pytest.raises(RuntimeError, match="constant graph"):
        _validate_smoke_actions(np.ones_like(noise), noise)


def test_gather_uses_config_model_id_and_never_native_optimized_factory(tmp_path, monkeypatch):
    root = tmp_path / "export"
    root.mkdir()
    _save_monolithic(root)
    import tether.eval.libero_rollout as rollout
    from tether.runtime.fast_inference.libero_adapter import TritonLIBEROAdapter
    from tether.verify import gather_paired_samples

    calls: dict[str, object] = {}
    policy = SimpleNamespace()

    def fake_load(**kwargs):
        calls["student_checkpoint"] = kwargs["student_checkpoint"]
        calls["model_type"] = kwargs["model_type"]
        calls["exact"] = kwargs["require_exact_checkpoint"]
        return policy, "pre", "post"

    def fake_rollout(*, use_native, inference, **kwargs):
        calls["optimized_inference" if not use_native else "native_inference"] = inference
        if use_native:
            calls["native_actions"] = np.full((1, 2, 2), 7.0, dtype=np.float32)
        else:
            noise = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
            calls["optimized_actions"] = inference.predict_action_chunk(**_protocol_inputs(noise))
        return {
            "per_task": [],
            "seed": kwargs["seed"],
            "seed_protocol": "sha256-v1",
        }

    monkeypatch.setattr(rollout, "load_pi05_policy_and_processors", fake_load)
    monkeypatch.setattr(rollout, "run_libero_rollout", fake_rollout)
    monkeypatch.setattr(
        TritonLIBEROAdapter,
        "from_policy",
        classmethod(lambda cls, policy: pytest.fail("native optimized fallback used")),
    )

    gather_paired_samples(
        optimized_ref=str(root),
        original_ref=None,
        suite="libero",
        task_suite_name="libero_10",
        num_episodes=1,
        task_indices=[0],
        seed=1,
        verification_device="cpu",
        isolate_processes=False,
    )

    assert calls["student_checkpoint"] == "org/native-reference"
    assert calls["model_type"] == "pi0"
    assert calls["exact"] is True
    assert calls["native_inference"] is None
    np.testing.assert_array_equal(
        calls["optimized_actions"], np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    )
    assert not np.array_equal(calls["native_actions"], calls["optimized_actions"])


@pytest.mark.parametrize(
    "producer_model_type",
    ["pi05_decomposed", "pi05_decomposed_student"],
)
def test_gather_accepts_real_pi05_producer_model_types(tmp_path, monkeypatch, producer_model_type):
    pytest.importorskip("onnxruntime")
    root = tmp_path / producer_model_type
    root.mkdir()
    config = _save_pi05_split(root)
    config["model_type"] = producer_model_type
    config["model_id"] = (
        "/exact/student" if producer_model_type.endswith("_student") else "org/pi05@sha"
    )
    write_tether_config(root, config)

    import tether.eval.libero_rollout as rollout
    from tether.verify import gather_paired_samples

    calls = {}

    def fake_load(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(), "pre", "post"

    def fake_rollout(*, use_native, inference, **kwargs):
        if not use_native:
            inference.predict_action_chunk(
                **_protocol_inputs(np.zeros((1, 2, 2), dtype=np.float32))
            )
        return {
            "per_task": [],
            "seed": kwargs["seed"],
            "seed_protocol": "sha256-v1",
        }

    monkeypatch.setattr(rollout, "load_pi05_policy_and_processors", fake_load)
    monkeypatch.setattr(rollout, "run_libero_rollout", fake_rollout)
    gather_paired_samples(
        optimized_ref=str(root),
        original_ref=None,
        suite="libero",
        task_suite_name="libero_10",
        num_episodes=1,
        task_indices=[0],
        seed=1,
        verification_device="cpu",
        isolate_processes=False,
    )

    assert calls["student_checkpoint"] == config["model_id"]
    assert calls["model_type"] == producer_model_type
    assert calls["require_exact_checkpoint"] is True


def test_isolated_optimized_arm_loads_no_native_policy(monkeypatch):
    import tether.eval.libero_rollout as rollout
    import tether.export_config as export_config
    import tether.runtime.verify_inference as verify_inference
    from tether.verify import _execute_verification_arm

    calls = []
    context = SimpleNamespace(config=SimpleNamespace())

    monkeypatch.setattr(
        export_config,
        "load_tether_config",
        lambda *_args, **_kwargs: {"model_type": "pi05"},
    )
    monkeypatch.setattr(
        rollout,
        "load_pi05_policy_and_processors",
        lambda **_kwargs: pytest.fail("optimized arm loaded baseline weights"),
    )

    def load_context(**kwargs):
        calls.append(("context", kwargs["device"]))
        return context, "pre", "post"

    monkeypatch.setattr(rollout, "load_verification_policy_context_and_processors", load_context)
    monkeypatch.setattr(
        verify_inference,
        "load_verification_inference",
        lambda _ref, device: calls.append(("artifact", device)) or "inference",
    )

    def fake_rollout(**kwargs):
        calls.append(("rollout", kwargs["verification_device"]))
        assert kwargs["policy"] is context
        assert kwargs["use_native"] is False
        return {"verification_device": kwargs["verification_device"]}

    monkeypatch.setattr(rollout, "run_libero_rollout", fake_rollout)
    result = _execute_verification_arm(
        arm="optimized",
        optimized_ref="export",
        original_checkpoint="native",
        task_suite_name="libero_10",
        num_episodes=1,
        task_indices=[0],
        seed=7,
        preprocessor_ref=None,
        verification_device="cpu",
        safety_limits=None,
        safety_config_sha256=None,
    )
    assert result["verification_device"] == "cpu"
    assert calls == [("context", "cpu"), ("artifact", "cpu"), ("rollout", "cpu")]


def test_gather_dispatches_arms_to_distinct_isolated_runs(monkeypatch):
    import tether.export_config as export_config
    import tether.verify as verify

    monkeypatch.setattr(
        export_config,
        "load_tether_config",
        lambda *_args, **_kwargs: {"model_id": "native", "model_type": "pi05"},
    )
    calls = []

    def isolated(**kwargs):
        calls.append(kwargs)
        return {
            "arm": kwargs["arm"],
            "verification_device": "cpu",
            "seed": kwargs["seed"],
            "seed_protocol": "sha256-v1",
        }

    monkeypatch.setattr(verify, "_run_isolated_verification_arm", isolated)
    original, optimized = verify.gather_paired_samples(
        optimized_ref="export",
        original_ref=None,
        suite="libero",
        task_suite_name="libero_10",
        num_episodes=30,
        task_indices=[0],
        seed=7,
        verification_device="cpu",
    )
    assert [call["arm"] for call in calls] == ["original", "optimized"]
    assert all(call["original_checkpoint"] == "native" for call in calls)
    assert original["arm"] == "original"
    assert optimized["arm"] == "optimized"


def test_gather_reads_safety_once_and_passes_one_canonical_value(monkeypatch):
    import tether.eval.evidence_capture as capture
    import tether.export_config as export_config
    import tether.verify as verify
    from tether.safety import SafetyLimits

    canonical = capture.canonicalize_safety_limits(SafetyLimits.default(7))
    reads = []
    calls = []
    monkeypatch.setattr(
        export_config,
        "load_tether_config",
        lambda *_args, **_kwargs: {"model_id": "native", "model_type": "pi05"},
    )
    monkeypatch.setattr(
        capture,
        "load_canonical_safety_limits",
        lambda path: reads.append(path) or canonical,
    )

    def isolated(**kwargs):
        calls.append(kwargs)
        return {
            "seed": kwargs["seed"],
            "seed_protocol": "sha256-v1",
            "safety_evidence": {
                "status": "available",
                "value": {
                    "sha256": kwargs["safety_config_sha256"],
                    "limits": kwargs["safety_limits"].to_dict(),
                },
            },
        }

    monkeypatch.setattr(verify, "_run_isolated_verification_arm", isolated)
    verify.gather_paired_samples(
        optimized_ref="export",
        original_ref=None,
        suite="libero",
        task_suite_name="libero_10",
        num_episodes=30,
        task_indices=[0],
        seed=7,
        verification_device="cpu",
        safety_config="limits.json",
    )
    assert reads == ["limits.json"]
    assert calls[0]["safety_limits"] is canonical
    assert calls[1]["safety_limits"] is canonical
    assert calls[0]["safety_config_sha256"] == canonical.sha256


def test_gather_rejects_child_safety_identity_mutation(monkeypatch):
    import tether.eval.evidence_capture as capture
    import tether.export_config as export_config
    import tether.verify as verify
    from tether.safety import SafetyLimits

    canonical = capture.canonicalize_safety_limits(SafetyLimits.default(7))
    monkeypatch.setattr(
        export_config,
        "load_tether_config",
        lambda *_args, **_kwargs: {"model_id": "native", "model_type": "pi05"},
    )
    monkeypatch.setattr(capture, "load_canonical_safety_limits", lambda _path: canonical)

    def isolated(**kwargs):
        limits = kwargs["safety_limits"].to_dict()
        if kwargs["arm"] == "optimized":
            limits["position_max"][0] = 99.0
        return {
            "seed": kwargs["seed"],
            "seed_protocol": "sha256-v1",
            "safety_evidence": {
                "status": "available",
                "value": {"sha256": canonical.sha256, "limits": limits},
            },
        }

    monkeypatch.setattr(verify, "_run_isolated_verification_arm", isolated)
    with pytest.raises(ValueError, match="optimized rollout safety identity mismatch"):
        verify.gather_paired_samples(
            optimized_ref="export",
            original_ref=None,
            suite="libero",
            task_suite_name="libero_10",
            num_episodes=30,
            task_indices=[0],
            seed=7,
            verification_device="cpu",
            safety_config="limits.json",
        )
