"""Tiny randomly initialised SmolVLA for the monolithic export + parity test.

The SmolVLM backbone config normally comes from the Hub
(HuggingFaceTB/SmolVLM2-500M-Video-Instruct); here it is built in-process at
toy size so the test is hermetic. Each mode runs in its own interpreter
because the exporter installs process-wide patches and the parity reference
must be native LeRobot:

    python tests/_smolvla_monolithic_tiny.py build    SRC
    python tests/_smolvla_monolithic_tiny.py export   SRC OUT
    python tests/_smolvla_monolithic_tiny.py validate OUT
"""

from __future__ import annotations

import contextlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

NUM_STEPS = 10


@contextlib.contextmanager
def tiny_vlm():
    from lerobot.policies.smolvla import smolvlm_with_expert as swe
    from transformers import SmolVLMConfig

    config = SmolVLMConfig(
        # Vocabulary >= 49152: the exporter's trace feeds token ids up to that.
        text_config=dict(
            model_type="llama", vocab_size=49280, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        ),
        # 512 px like the exporter's trace; patch 64 -> 8x8 patches -> 16 tokens.
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, image_size=512, patch_size=64,
        ),
        scale_factor=2,
    )
    # Only the two special-token ids are read from the processor.
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(fake_image_token_id=49189, global_image_token_id=49190)
    )
    with (
        patch.object(swe.AutoConfig, "from_pretrained", return_value=config),
        patch.object(swe.AutoProcessor, "from_pretrained", return_value=processor),
    ):
        yield


def build(src):
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.manual_seed(0)
    cfg = SmolVLAConfig(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=6,
        max_action_dim=6,
        num_vlm_layers=2,
        num_steps=NUM_STEPS,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    with tiny_vlm():
        policy = SmolVLAPolicy(cfg)
    # Random init leaves some projections near zero; make every weight carry
    # signal so a dropped or mis-wired branch moves the actions.
    with torch.no_grad():
        for p in policy.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    policy.save_pretrained(src)
    print("built", sum(p.numel() for p in policy.parameters()))


def export(src, out):
    from tether.exporters.monolithic import export_monolithic

    with tiny_vlm():
        result = export_monolithic(src, out, num_steps=NUM_STEPS, precision="fp32")
    print("exported", json.dumps(result))


def validate(out):
    from pathlib import Path

    from tether.validate_roundtrip import ValidateRoundTrip

    with tiny_vlm():
        result = ValidateRoundTrip(Path(out), num_test_cases=4, seed=0).run()
    print("RESULT=" + json.dumps(result))


if __name__ == "__main__":
    mode = sys.argv[1]
    {"build": build, "export": export, "validate": validate}[mode](*sys.argv[2:])
