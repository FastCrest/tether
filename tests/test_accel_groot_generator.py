# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in src/tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28.
"""Preserve native masks/embodiment and RTC behavior across the accel action-head handoff (ported upstream test)."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import pytest

torch = pytest.importorskip("torch")
from tether.accel.groot_n17.frontend import AccelGrootActionGenerator  # noqa: E402


class Batch(dict):
    __getattr__ = dict.__getitem__


class GeneratorTests(unittest.TestCase):
    def make(self):
        config = SimpleNamespace(
            action_horizon=40,
            max_action_dim=132,
            state_history_length=1,
            add_pos_embed=True,
            use_alternate_vl_dit=True,
        )
        self.rtc = []
        self.infer = []
        head = SimpleNamespace(
            config=config,
            num_inference_timesteps=4,
            num_timestep_buckets=1000,
            dtype=torch.bfloat16,
            state_encoder=lambda s, e: torch.zeros(1, 1, 1536),
            vlln=lambda x: x,
            vl_self_attention=lambda x: x,
            get_action_with_features=lambda *args: self.rtc.append(args) or "native RTC result",
        )
        engine = SimpleNamespace(
            infer=lambda *args: self.infer.append(args) or torch.zeros(40, 132),
            vx=torch.zeros(8, 2048),
        )
        return AccelGrootActionGenerator(head, engine, 24)

    def test_rtc_preserves_options_without_engine_denoise(self):
        generator = self.make()
        backbone = Batch(
            backbone_features=torch.ones(1, 3, 2048),
            image_mask=torch.tensor([[False, True, False]]),
            backbone_attention_mask=torch.tensor([[True, True, False]]),
        )
        action = Batch(
            state=torch.ones(1, 1, 132),
            embodiment_id=torch.tensor([24]),
            action=torch.zeros(1, 40, 132),
        )
        options = {
            "rtc_overlap_steps": 3,
            "rtc_frozen_steps": 1,
            "rtc_ramp_rate": 2,
            "action_horizon": 40,
        }
        module = SimpleNamespace(BatchFeature=lambda data: Batch(data))
        with patch.dict(sys.modules, {"transformers.feature_extraction_utils": module}):
            result = generator.get_action(backbone, action, options)
        self.assertEqual(result, "native RTC result")
        self.assertFalse(self.infer)
        self.assertIs(self.rtc[0][-1], options)
        self.assertIs(self.rtc[0][-2], action)

    def test_engine_path_passes_masks_and_returns_action_pred(self):
        generator = self.make()
        backbone = Batch(
            backbone_features=torch.ones(1, 3, 2048),
            image_mask=torch.tensor([[False, True, False]]),
            backbone_attention_mask=torch.tensor([[True, True, False]]),
        )
        action = Batch(state=torch.ones(1, 1, 132), embodiment_id=torch.tensor([24]))
        module = SimpleNamespace(BatchFeature=lambda data: Batch(data))
        with patch.dict(sys.modules, {"transformers.feature_extraction_utils": module}):
            result = generator.get_action(backbone, action)
        self.assertEqual(tuple(result["action_pred"].shape), (1, 40, 132))
        feats, image_mask, attn_mask, state, noise = self.infer[0]
        self.assertTrue(torch.equal(image_mask, torch.tensor([False, True, False])))
        self.assertTrue(torch.equal(attn_mask, torch.tensor([True, True, False])))
        self.assertEqual(tuple(noise.shape), (40, 132))

    def test_wrong_embodiment_is_rejected_before_the_engine(self):
        generator = self.make()
        action = Batch(state=torch.ones(1, 1, 132), embodiment_id=torch.tensor([2]))
        with patch.dict(
            sys.modules,
            {"transformers.feature_extraction_utils": SimpleNamespace(BatchFeature=Batch)},
        ):
            with self.assertRaises(ValueError):
                generator.get_action(Batch(backbone_features=torch.ones(1, 3, 2048)), action)
        self.assertFalse(self.infer)


if __name__ == "__main__":
    unittest.main()
