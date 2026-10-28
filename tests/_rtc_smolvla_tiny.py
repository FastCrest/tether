"""A tiny randomly initialised SmolVLA installed on SmolVLANativeServer.

Hermetic (no Hub access): the SmolVLM backbone comes from
``tests/_smolvla_monolithic_tiny.tiny_vlm`` and ``load()`` is replaced so the
real LeRobot policy, a token-only preprocessor and a real LeRobot action
unnormalizer are wired in. The RTC path under test is LeRobot's own
(``euler_integrate`` + ``RTCProcessor``).
"""
from __future__ import annotations

import numpy as np

CHUNK = 10
ACTION_DIM = 6
NUM_STEPS = 5
ACTION_MEAN = np.array([0.5, -0.2, 0.1, 0.0, 1.0, -1.0], np.float32)
ACTION_STD = np.array([2.0, 0.5, 1.5, 1.0, 0.25, 3.0], np.float32)


def build_policy(seed: int = 0):
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from tests._smolvla_monolithic_tiny import tiny_vlm

    torch.manual_seed(seed)
    cfg = SmolVLAConfig(
        device="cpu",
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        max_state_dim=8,
        max_action_dim=ACTION_DIM,
        num_vlm_layers=2,
        num_steps=NUM_STEPS,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )
    with tiny_vlm():
        policy = SmolVLAPolicy(cfg)
    with torch.no_grad():
        for p in policy.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    return policy.eval()


def _preprocessor(batch):
    import torch

    out = dict(batch)
    out["observation.language.tokens"] = torch.tensor([[1, 5, 9, 13]], dtype=torch.long)
    out["observation.language.attention_mask"] = torch.ones((1, 4), dtype=torch.bool)
    return out


def postprocessor():
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import PolicyProcessorPipeline, UnnormalizerProcessorStep
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    step = UnnormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={"action": {"mean": ACTION_MEAN, "std": ACTION_STD}},
    )
    return PolicyProcessorPipeline(
        steps=[step],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )


def install(server, policy=None) -> None:
    """Make ``server.load()`` wire the tiny policy instead of the Hub model."""

    def load():
        server.policy = policy if policy is not None else build_policy()
        server._freeze_policy()
        server.preprocessor = _preprocessor
        server.postprocessor = postprocessor()
        server._device = "cpu"
        server.action_dim = ACTION_DIM
        server.chunk_size = CHUNK
        server.num_denoising_steps = NUM_STEPS
        if server._rtc_required:
            server._action_unnormalizer()
        server._ready = True

    server.load = load
