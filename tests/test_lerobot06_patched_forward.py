"""Real, tiny LeRobot forwards; process isolation keeps export patches out of other tests."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "path",
    [
        "pi0",
        "pi05",
        "decomposed",
        "snapflow_pi0",
        "snapflow_pi05",
        "snapflow_state_out",
        "private_api",
    ],
)
def test_patched_forward_matches_lerobot06(path):
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("requires the LeRobot 0.6.1 native extra")
    if version != "0.6.1":
        pytest.skip("this regression exercises LeRobot 0.6.1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), path],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check(path):
    import copy
    import torch
    from unittest.mock import patch
    from lerobot.policies.pi0 import modeling_pi0 as pi0
    from lerobot.policies.pi05 import modeling_pi05 as pi05
    from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
    from tether.exporters import monolithic, decomposed
    from tether.distill import snapflow_pi0_model as snapflow

    if path == "private_api":
        import ast
        import inspect

        for owner in (pi0.PI0Pytorch, pi05.PI05Pytorch):
            assert not hasattr(owner, "_prepare_attention_masks_4d")
            assert (
                str(inspect.signature(owner._apply_checkpoint)) == "(self, func, *args, **kwargs)"
            )
        for owner in (pi0.PI0Policy, pi05.PI05Policy):
            assert list(inspect.signature(owner._preprocess_images).parameters) == ["self", "batch"]
        root = Path(__file__).resolve().parents[1]
        for directory in (root / "src", root / "scripts"):
            for source in directory.rglob("*.py"):
                tree = ast.parse(source.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        assert node.func.attr != "_prepare_attention_masks_4d", str(source)
                    if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                        "lerobot"
                    ):
                        assert all(not name.name.startswith("_") for name in node.names), str(
                            source
                        )
        return

    torch.set_num_threads(1)
    torch.manual_seed(123)
    module = pi0 if path.endswith("pi0") or path == "pi0" else pi05
    cls = module.PI0Pytorch if module is pi0 else module.PI05Pytorch
    config_cls = module.PI0Config if module is pi0 else module.PI05Config
    # Shrink constructor configurations only. Every embedding, attention, norm,
    # velocity, cache and Euler computation remains the real LeRobot code.
    original_pali = module.PaliGemmaForConditionalGenerationWithPiGemma
    original_expert = module.PiGemmaForCausalLM

    def tiny_pali(config):
        config.text_config.vocab_size = 64
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_hidden_layers = 2
        config.vision_config.num_attention_heads = 4
        config.vision_config.patch_size = 4
        config.vision_config.projection_dim = 32
        return original_pali(config=config)

    def tiny_expert(config):
        config.vocab_size = 64
        return original_expert(config=config)

    config = config_cls(
        device="cpu",
        dtype="float32",
        image_resolution=(16, 16),
        max_action_dim=4,
        max_state_dim=4,
        chunk_size=3,
        n_action_steps=3,
        compile_model=False,
    )
    with (
        patch.object(
            module, "get_gemma_config", return_value=module.GemmaConfig(32, 2, 64, 8, 1, 4)
        ),
        patch.object(module, "PaliGemmaForConditionalGenerationWithPiGemma", tiny_pali),
        patch.object(module, "PiGemmaForCausalLM", tiny_expert),
    ):
        model = cls(config).eval()
    images = [torch.randn(2, 3, 16, 16) for _ in range(3)]
    image_masks = [
        torch.tensor([True, True]),
        torch.tensor([True, False]),
        torch.tensor([False, True]),
    ]
    tokens = torch.randint(0, 64, (2, 4))
    masks = torch.tensor([[True, True, False, False], [True, True, True, False]])
    state = torch.randn(2, 4)
    noise = torch.randn(2, 3, 4)
    actions = torch.randn_like(noise)
    time = torch.tensor([0.7, 0.3])
    obs = [images, image_masks, tokens, masks] + ([state] if module is pi0 else [])
    native = copy.deepcopy(model)
    native_step = cls.denoise_step
    with torch.no_grad():
        expected_loss = native(*obs, actions, noise, time)
        expected_actions = native.sample_actions(*obs, noise=noise.clone(), num_steps=3)
        expected_one_step = native.sample_actions(*obs, noise=noise.clone(), num_steps=1)
        prefix, pad, att = native.embed_prefix(images, image_masks, tokens, masks)
        _, cache = native.paligemma_with_expert(
            attention_mask=prepare_attention_masks_4d(make_att_2d_masks(pad, att)),
            position_ids=pad.cumsum(1) - 1,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        step_kw = dict(prefix_pad_masks=pad, past_key_values=cache, x_t=x_t, timestep=time)
        if module is pi0:
            step_kw["state"] = state
        expected_velocity = native_step(native, **step_kw)

        if path.startswith("snapflow"):
            snapflow.enable_snapflow(model)
            if path == "snapflow_state_out":
                snapflow.enable_snapflow_state_out(model)
            # A zero target-time contribution must reproduce the teacher.
            step_kw["target_time"] = torch.ones_like(time)
            if path == "snapflow_state_out":
                step_kw["state"] = state
                # The state-out student's embedding is its intentional extension.
                # Execute the UNPATCHED LeRobot denoise forward with that embedding.
                ref = copy.deepcopy(model)
                embed = ref.embed_suffix
                ref.embed_suffix = lambda x, t: embed(
                    x, t, target_time=torch.ones_like(t), state=state
                )
                expected_velocity = native_step(
                    ref, **{k: v for k, v in step_kw.items() if k not in ("state", "target_time")}
                )
            torch.testing.assert_close(
                model.denoise_step(**step_kw), expected_velocity, rtol=1e-6, atol=1e-6
            )
            import types

            if path == "snapflow_state_out":
                ref.denoise_step = types.MethodType(native_step, ref)
                expected_one_step = cls.sample_actions(ref, *obs, noise=noise.clone(), num_steps=1)
            sample_obs = obs + ([state] if path == "snapflow_state_out" else [])
            torch.testing.assert_close(
                model.sample_actions_1step(*sample_obs, noise=noise.clone()),
                expected_one_step,
                rtol=1e-6,
                atol=1e-6,
            )
            # Exercise both training-prefix adapters with real model forwards.
            from types import SimpleNamespace
            from tether.finetune.backends.snapflow_backend import _build_velocity_adapters
            from lerobot.utils.constants import (
                OBS_LANGUAGE_TOKENS,
                OBS_LANGUAGE_ATTENTION_MASK,
                OBS_STATE,
            )

            policy = SimpleNamespace(
                model=model,
                config=config,
                _preprocess_images=lambda batch: (images, image_masks),
                prepare_state=lambda batch: state,
            )
            _, student_velocity = _build_velocity_adapters(
                policy, policy, "pi0" if module is pi0 else "pi05"
            )
            torch.testing.assert_close(
                student_velocity(
                    x_t,
                    time,
                    target_time=torch.ones_like(time),
                    **{
                        OBS_LANGUAGE_TOKENS: tokens,
                        OBS_LANGUAGE_ATTENTION_MASK: masks,
                        OBS_STATE: state,
                    },
                ),
                expected_velocity,
                rtol=1e-6,
                atol=1e-6,
            )
            # Also test a learned target-time correction. A zero-init-only
            # test would miss an exporter silently dropping target_time.
            model.target_time_embed_mlp[-1].weight.normal_(std=0.01)
            model.target_time_embed_mlp[-1].bias.normal_(std=0.01)
            ref = copy.deepcopy(model)
            embed = ref.embed_suffix
            state_kwargs = {"state": state} if path == "snapflow_state_out" else {}
            ref.embed_suffix = lambda *args: embed(
                *args, target_time=torch.ones_like(args[-1]), **state_kwargs
            )
            ref.denoise_step = types.MethodType(native_step, ref)
            native_kwargs = {k: v for k, v in step_kw.items() if k != "target_time"}
            if path == "snapflow_state_out":
                native_kwargs.pop("state")
            expected_velocity = native_step(ref, **native_kwargs)
            expected_loss = cls.forward(ref, *obs, actions, noise, time)
            expected_one_step = cls.sample_actions(ref, *obs, noise=noise.clone(), num_steps=1)
            torch.testing.assert_close(
                model.denoise_step(**step_kw), expected_velocity, rtol=1e-6, atol=1e-6
            )
        monolithic.apply_export_patches()
        if module is pi05:
            monolithic._apply_pi05_denoise_step_patch()
        if path.startswith("snapflow"):
            monolithic._install_snapflow_export_denoise_step(model, is_pi05=module is pi05)
        step_kw["past_key_values"] = copy.deepcopy(cache)
        actual_velocity = model.denoise_step(**step_kw)
        torch.testing.assert_close(actual_velocity, expected_velocity, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            (noise - actions - actual_velocity).square(), expected_loss, rtol=1e-6, atol=1e-6
        )
        if not path.startswith("snapflow"):
            torch.testing.assert_close(
                model.sample_actions(*obs, noise=noise.clone(), num_steps=3),
                expected_actions,
                rtol=1e-6,
                atol=1e-6,
            )
        if module is pi05 and path != "pi05":
            decomposed.PI05_PALIGEMMA_LAYERS = 2
            prefix_wrapper = decomposed._build_prefix_class()(model)
            flat = prefix_wrapper(*images, *image_masks, tokens, masks)
            for actual, layer_value in zip(
                flat[:-1],
                [v for layer in cache.layers for v in (layer.keys, layer.values)],
                strict=True,
            ):
                torch.testing.assert_close(actual, layer_value, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(flat[-1], pad, rtol=0, atol=0)
            per_step = decomposed._build_expert_per_step_class()(model)
            torch.testing.assert_close(
                per_step(*flat, x_t, time, *([state] if path == "snapflow_state_out" else [])),
                expected_velocity,
                rtol=1e-6,
                atol=1e-6,
            )
            n_steps = 1 if path.startswith("snapflow") else 3
            baked = decomposed._build_expert_class()(model, n_steps)
            state_args = (state,) if path == "snapflow_state_out" else ()
            step_args = (*flat, x_t, time, *state_args)
            baked_args = (*flat, noise, *state_args)
            expected_baked = expected_one_step if path.startswith("snapflow") else expected_actions
            torch.testing.assert_close(baked(*baked_args), expected_baked, rtol=1e-6, atol=1e-6)
            prefix_graph = torch.export.export(
                prefix_wrapper, (*images, *image_masks, tokens, masks), strict=False
            ).module()
            for actual, expected in zip(
                prefix_graph(*images, *image_masks, tokens, masks), flat, strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            step_graph = torch.export.export(per_step, step_args, strict=False).module()
            torch.testing.assert_close(
                step_graph(*step_args), expected_velocity, rtol=1e-6, atol=1e-6
            )
            baked_graph = torch.export.export(baked, baked_args, strict=False).module()
            torch.testing.assert_close(
                baked_graph(*baked_args), expected_baked, rtol=1e-6, atol=1e-6
            )

        class ExportStep(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, *args):
                from transformers.cache_utils import DynamicCache

                kv = DynamicCache()
                for i in range(2):
                    kv.update(args[2 * i], args[2 * i + 1], i)
                kwargs = dict(
                    prefix_pad_masks=args[4], past_key_values=kv, x_t=args[5], timestep=args[6]
                )
                if module is pi0 or path == "snapflow_state_out":
                    kwargs["state"] = args[7]
                if path.startswith("snapflow"):
                    kwargs["target_time"] = torch.ones_like(args[6])
                return self.model.denoise_step(**kwargs)

        class ExportActions(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, *args):
                if path.startswith("snapflow"):
                    return self.model.sample_actions_1step(*args[:-1], noise=args[-1])
                return self.model.sample_actions(*args[:-1], noise=args[-1], num_steps=3)

        action_args = (*(sample_obs if path.startswith("snapflow") else obs), noise)
        exported_actions = torch.export.export(ExportActions(), action_args, strict=False).module()
        torch.testing.assert_close(
            exported_actions(*action_args),
            expected_one_step if path.startswith("snapflow") else expected_actions,
            rtol=1e-6,
            atol=1e-6,
        )

        flat_cache = tuple(v for layer in cache.layers for v in (layer.keys, layer.values))
        export_args = (*flat_cache, pad, x_t, time, state)
        exported = torch.export.export(ExportStep(), export_args, strict=False).module()
        torch.testing.assert_close(exported(*export_args), expected_velocity, rtol=1e-6, atol=1e-6)
    print(path, "native forward parity passed")


if __name__ == "__main__":
    _check(sys.argv[1])
