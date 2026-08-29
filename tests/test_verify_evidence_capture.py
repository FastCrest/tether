from __future__ import annotations

import random
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from tether.eval.evidence_capture import (
    CanonicalSafetyLimits,
    ProcessDeviceMemorySampler,
    _read_cuda_process_bytes,
    apply_action_guard,
    canonicalize_safety_limits,
    nearest_rank_percentile,
    validate_safety_limits,
)
from tether.safety import ActionGuard, SafetyLimits
from tether.verification_evidence import (
    UnavailableReason,
    normalize_verification_device,
)
from tether.verify import _rollout_results_to_samples
from tether.eval.libero_rollout import (
    VerificationPolicyContext,
    paired_policy_call_seed,
    run_libero_rollout,
    seed_verification_random_sources,
)


def test_memory_sampler_records_10hz_process_and_device_series():
    rss_values = iter([100, 200, 300])
    device_values = iter([10, 20, 30])
    now = [0.0]
    sampler = ProcessDeviceMemorySampler(
        device="cuda",
        rss_probe=lambda _pid: next(rss_values),
        device_probe=lambda _pid: next(device_values),
        clock=lambda: now[0],
    )
    sampler.start(background=False)
    now[0] = 0.1
    sampler.sample(scheduled_at=0.1)
    now[0] = 0.2
    sampler.sample(scheduled_at=0.2)
    evidence = sampler.stop()

    assert evidence.is_available
    summary = evidence.value
    assert summary is not None
    assert summary["sample_hz"] == 10.0
    assert summary["backend"] == "cuda"
    assert summary["process_identity"]["pid"] > 0
    assert summary["window"]["captured_samples"] == 3
    assert len(summary["samples"]) == 3
    assert summary["process_rss"] == {"peak_bytes": 300, "p95_bytes": 300}
    assert summary["device_allocated"] == {"peak_bytes": 30, "p95_bytes": 30}
    assert summary["combined"] == {"peak_bytes": 330, "p95_bytes": 330}


def test_cpu_sampler_proves_zero_device_allocation_with_real_rss_probe():
    now = [0.0]
    sampler = ProcessDeviceMemorySampler(device="cpu", clock=lambda: now[0])
    sampler.start(background=False)
    now[0] = 0.1
    sampler.sample(scheduled_at=0.1)
    evidence = sampler.stop()
    assert evidence.is_available
    summary = evidence.value
    assert summary is not None
    assert summary["backend"] == "cpu"
    assert summary["device_allocated"]["peak_bytes"] == 0
    assert summary["process_rss"]["peak_bytes"] > 0


def test_memory_sampler_types_unsupported_and_failed_measurements():
    unsupported = ProcessDeviceMemorySampler(
        device="cuda",
        device_probe=lambda _pid: (_ for _ in ()).throw(FileNotFoundError()),
    )
    unsupported.start(background=False)
    assert unsupported.stop().reason is UnavailableReason.BACKEND_UNSUPPORTED

    failed = ProcessDeviceMemorySampler(
        device="cpu",
        rss_probe=lambda _pid: (_ for _ in ()).throw(OSError("probe failed")),
    )
    failed.start(background=False)
    assert failed.stop().reason is UnavailableReason.MEASUREMENT_FAILED


def test_cuda_pid_absence_is_measurement_failure_not_zero(monkeypatch):
    monkeypatch.setattr(
        "tether.eval.evidence_capture.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="999, 42\n"),
    )
    with pytest.raises(ProcessLookupError):
        _read_cuda_process_bytes(123)

    sampler = ProcessDeviceMemorySampler(device="cuda", pid=123)
    sampler.start(background=False)
    assert sampler.stop().reason is UnavailableReason.MEASUREMENT_FAILED


def test_cuda_authoritative_matched_pid_zero_is_real_zero(monkeypatch):
    monkeypatch.setattr(
        "tether.eval.evidence_capture.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="123, 0\n"),
    )
    assert _read_cuda_process_bytes(123) == 0


def test_slow_probe_and_one_sample_windows_fail_closed():
    now = [0.0]

    def slow_device_probe(_pid):
        now[0] += 0.06
        return 0

    slow = ProcessDeviceMemorySampler(
        device="cpu",
        rss_probe=lambda _pid: 100,
        device_probe=slow_device_probe,
        clock=lambda: now[0],
    )
    slow.start(background=False)
    now[0] = 0.1
    slow.sample(scheduled_at=0.1)
    assert slow.stop().reason is UnavailableReason.MEASUREMENT_FAILED

    now[0] = 0.0
    one = ProcessDeviceMemorySampler(device="cpu", rss_probe=lambda _pid: 100, clock=lambda: now[0])
    one.start(background=False)
    now[0] = 0.04
    assert one.stop().reason is UnavailableReason.MEASUREMENT_FAILED


def test_sampler_uses_absolute_deadlines_after_probe_time():
    now = [0.0]

    def probe(_pid):
        now[0] += 0.04
        return 100

    sampler = ProcessDeviceMemorySampler(
        device="cpu",
        rss_probe=probe,
        clock=lambda: now[0],
    )
    sampler.start(background=False)
    waits = []

    class StopAfterThreeWaits:
        def wait(self, delay):
            waits.append(delay)
            now[0] += delay
            return len(waits) == 3

        def set(self):
            return None

    sampler._stop_event = StopAfterThreeWaits()
    sampler._run()
    assert waits[0] == pytest.approx(0.06)
    assert waits[1] == pytest.approx(0.06)
    assert [sample.scheduled_monotonic_s for sample in sampler._samples] == pytest.approx(
        [0.0, 0.1, 0.2]
    )


def test_verification_device_is_exact_and_public():
    assert normalize_verification_device("cpu") == "cpu"
    assert normalize_verification_device("CUDA") == "cuda"
    with pytest.raises(ValueError):
        normalize_verification_device("gpu")
    with pytest.raises(ValueError):
        normalize_verification_device("mps")


def test_paired_policy_calls_use_identical_native_and_protocol_noise():
    torch = pytest.importorskip("torch")
    call_seed = paired_policy_call_seed(7, 3, 2, 4)
    seed_verification_random_sources(call_seed, torch_module=torch, device="cpu")
    native_noise = torch.randn(1, 5, 7)
    native_python = random.random()
    native_numpy = np.random.random()

    seed_verification_random_sources(call_seed, torch_module=torch, device="cpu")
    optimized_protocol_noise = torch.randn(1, 5, 7)
    assert torch.equal(native_noise, optimized_protocol_noise)
    assert random.random() == native_python
    assert np.random.random() == native_numpy


def test_cuda_seed_path_labels_and_seeds_all_cuda_generators():
    calls = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("cpu", seed)),
        cuda=SimpleNamespace(manual_seed_all=lambda seed: calls.append(("cuda", seed))),
    )
    seed_verification_random_sources(123, torch_module=fake_torch, device="cuda")
    assert calls == [("cpu", 123), ("cuda", 123)]


@pytest.mark.parametrize(
    ("model_type", "module_name", "class_name", "method_name"),
    [
        (
            "pi05",
            "lerobot.policies.pi05.modeling_pi05",
            "PI05Policy",
            "_preprocess_images",
        ),
        (
            "pi05_decomposed",
            "lerobot.policies.pi05.modeling_pi05",
            "PI05Policy",
            "_preprocess_images",
        ),
        (
            "pi05_decomposed_student",
            "lerobot.policies.pi05.modeling_pi05",
            "PI05Policy",
            "_preprocess_images",
        ),
        (
            "pi0",
            "lerobot.policies.pi0.modeling_pi0",
            "PI0Policy",
            "_preprocess_images",
        ),
        (
            "smolvla",
            "lerobot.policies.smolvla.modeling_smolvla",
            "SmolVLAPolicy",
            "prepare_images",
        ),
    ],
)
def test_weight_free_context_matches_native_image_preprocessing(
    model_type, module_name, class_name, method_name
):
    torch = pytest.importorskip("torch")
    module = pytest.importorskip(module_name, exc_type=ImportError)
    policy_class = getattr(module, class_name)
    config = SimpleNamespace(
        image_features={"camera0": object(), "camera1": object()},
        image_resolution=(8, 8),
        resize_imgs_with_padding=None,
        empty_cameras=1,
    )
    batch = {"camera0": torch.linspace(0, 1, 3 * 8 * 8).reshape(1, 3, 8, 8)}
    native_surface = SimpleNamespace(
        config=config,
        parameters=lambda: iter([torch.nn.Parameter(torch.empty(0))]),
    )
    native_images, native_masks = getattr(policy_class, method_name)(native_surface, batch)
    context = VerificationPolicyContext(
        model_type=model_type,
        config=config,
        device="cpu",
    )
    context_images, context_masks = context._preprocess_images(batch)
    assert len(context_images) == len(native_images)
    assert len(context_masks) == len(native_masks)
    for actual, expected in zip(context_images, native_images):
        assert torch.equal(actual, expected)
    for actual, expected in zip(context_masks, native_masks):
        assert torch.equal(actual, expected)


def test_native_and_artifact_use_same_explicit_chunk_noise_at_same_env_steps(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    action_key = "action"
    state_key = "observation.state"
    token_key = "observation.language.tokens"
    mask_key = "observation.language.attention_mask"
    constants = SimpleNamespace(
        ACTION=action_key,
        OBS_STATE=state_key,
        OBS_LANGUAGE_TOKENS=token_key,
        OBS_LANGUAGE_ATTENTION_MASK=mask_key,
    )
    monkeypatch.setitem(sys.modules, "lerobot.utils.constants", constants)
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            INTER_AREA=0,
            BORDER_CONSTANT=0,
            resize=lambda image, _size, interpolation: image.copy(),
            copyMakeBorder=lambda image, *_args, **_kwargs: image.copy(),
        ),
    )

    class Task:
        problem_folder = "task"
        bddl_file = "task.bddl"
        language = "move"

    class Suite:
        n_tasks = 1

        def get_task(self, _task_idx):
            return Task()

        def get_task_init_states(self, _task_idx):
            return [0]

    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_10": Suite})

    def observation():
        return {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(1, dtype=np.float32),
        }

    class FakeEnv:
        def __init__(self, **_kwargs):
            self.steps = 0

        def reset(self):
            self.steps = 0

        def set_init_state(self, _state):
            return observation()

        def step(self, _action):
            self.steps += 1
            return observation(), 0.0, self.steps >= 6, {}

        def close(self):
            return None

    libero_module = SimpleNamespace(
        benchmark=benchmark,
        get_libero_path=lambda _name: "/tmp",
    )
    monkeypatch.setitem(sys.modules, "libero", SimpleNamespace(libero=libero_module))
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)
    monkeypatch.setitem(
        sys.modules,
        "libero.libero.envs",
        SimpleNamespace(OffScreenRenderEnv=FakeEnv),
    )

    config = SimpleNamespace(
        chunk_size=4,
        max_action_dim=7,
        output_features={action_key: SimpleNamespace(shape=(7,))},
    )
    native_noises = []

    class NativePolicy:
        def __init__(self):
            self.config = config

        def reset(self):
            return None

        def select_action(self, _batch):
            pytest.fail("private native select_action queue was used")

        def predict_action_chunk(self, _batch, *, noise):
            native_noises.append(noise.detach().cpu().numpy().copy())
            return noise

    artifact_noises = []

    class ArtifactPolicyContext:
        def __init__(self):
            self.config = config

        def reset(self):
            return None

        def _preprocess_images(self, batch):
            image = batch["observation.images.image"]
            images = [image, image, image]
            masks = [torch.ones(1, dtype=torch.bool)] * 3
            return images, masks

    class ArtifactInference:
        def reset_cache(self):
            return None

        def predict_action_chunk(self, **kwargs):
            artifact_noises.append(np.asarray(kwargs["noise"]).copy())
            return np.asarray(kwargs["noise"])

        def get_stats(self):
            return {}

    def preprocess(batch):
        batch[token_key] = torch.zeros((1, 2), dtype=torch.int64)
        batch[mask_key] = torch.ones((1, 2), dtype=torch.bool)
        return batch

    common = dict(
        preprocessor=preprocess,
        postprocessor=lambda actions: actions,
        task_suite_name="libero_10",
        num_episodes=1,
        task_indices=[0],
        resize_size=8,
        replan_steps=2,
        num_steps_wait=0,
        seed=19,
        capture_trajectories=True,
        verification_device="cpu",
    )
    native = run_libero_rollout(
        inference=None,
        policy=NativePolicy(),
        use_native=True,
        label="ORIGINAL",
        **common,
    )
    artifact = run_libero_rollout(
        inference=ArtifactInference(),
        policy=ArtifactPolicyContext(),
        use_native=False,
        label="OPTIMIZED",
        **common,
    )
    assert len(native_noises) == len(artifact_noises) == 3
    for native_noise, artifact_noise in zip(native_noises, artifact_noises):
        np.testing.assert_array_equal(native_noise, artifact_noise)
    native_episode = native["per_task"][0]["episodes"][0]
    artifact_episode = artifact["per_task"][0]["episodes"][0]
    assert native_episode["inference_call_steps"] == [0, 2, 4]
    assert artifact_episode["inference_call_steps"] == [0, 2, 4]
    assert native_episode["inference_call_seeds"] == artifact_episode["inference_call_seeds"]
    assert len(native_episode["inference_latency_ms"]) == 3
    assert len(artifact_episode["inference_latency_ms"]) == 3


def test_action_guard_returns_the_exact_clamped_action_and_count():
    limits = SafetyLimits(
        joint_names=["j0", "j1"],
        position_min=[-1.0, -1.0],
        position_max=[1.0, 1.0],
        velocity_max=[10.0, 10.0],
        effort_max=[0.0, 0.0],
    )
    guard = ActionGuard(limits=limits, mode="clamp", max_consecutive_clamps=0)
    executed, clamps = apply_action_guard(np.array([5.0, 0.5]), guard)
    assert executed.tolist() == [1.0, 0.5]
    assert clamps == 1

    rejected, nonfinite_clamps = apply_action_guard(
        np.array([np.nan, 0.5]), guard, previous_action=executed
    )
    assert rejected.tolist() == [0.0, 0.0]
    assert nonfinite_clamps == 1


def test_safety_limits_must_cover_exact_seven_dimensional_action():
    validate_safety_limits(SafetyLimits.default(7))
    with pytest.raises(ValueError, match="exactly 7"):
        validate_safety_limits(SafetyLimits.default(6))

    partial = SafetyLimits.default(7)
    partial.velocity_max.pop()
    with pytest.raises(ValueError, match="exactly 7"):
        validate_safety_limits(partial)

    invalid = SafetyLimits.default(7)
    invalid.position_max[3] = invalid.position_min[3]
    with pytest.raises(ValueError, match="invalid at index 3"):
        validate_safety_limits(invalid)


def test_canonical_safety_limits_are_immutable_and_hash_addressed():
    source = SafetyLimits.default(7)
    canonical = canonicalize_safety_limits(source)
    assert isinstance(canonical, CanonicalSafetyLimits)
    assert len(canonical.sha256) == 64
    assert canonical == canonicalize_safety_limits(source)
    source.position_max[0] = 99.0
    assert canonical.position_max[0] != 99.0
    with pytest.raises(FrozenInstanceError):
        canonical.position_max = (99.0,) * 7


def test_rollout_adapter_populates_real_captured_channels_and_teacher_pair():
    teacher = {
        "per_task": [
            {
                "task_idx": 1,
                "episodes": [
                    {
                        "ep": 0,
                        "success": True,
                        "actions": [[0.0, 0.0], [0.2, 0.4]],
                        "action_timestamps_s": [10.0, 10.5],
                        "inference_latency_ms": [3.0, 7.0, 5.0],
                        "safety_clamp_count": 0,
                    }
                ],
            }
        ],
    }
    candidate = {
        "per_task": [
            {
                "task_idx": 1,
                "episodes": [
                    {
                        "ep": 0,
                        "success": True,
                        "actions": [[0.0, 0.0], [0.2, 0.4]],
                        "action_timestamps_s": [20.0, 20.5],
                        "inference_latency_ms": [4.0, 8.0, 6.0],
                        "safety_clamp_count": 1,
                    }
                ],
            }
        ],
    }
    sample = _rollout_results_to_samples(candidate, teacher_results=teacher)[0]
    assert sample.success is True
    assert sample.safety_clamp_count == 1
    assert sample.inference_latency_p99_ms == 8.0
    assert sample.per_joint_velocity == [0.4, 0.8]
    assert sample.action_trajectory == [[0.0, 0.0], [0.2, 0.4]]
    assert sample.teacher_action_trajectory == [[0.0, 0.0], [0.2, 0.4]]


def test_nearest_rank_percentile_is_not_interpolated():
    assert nearest_rank_percentile([1, 2, 100], 95.0) == 100.0
