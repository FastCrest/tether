"""Golden pi0/pi0.5 reference from the qualified OLD stack (independent of LeRobot 0.6.1).

Why: the LeRobot 0.6.1 port tests compared Tether's patched forward with
native 0.6.1 in the same interpreter. Under transformers 5.3 native 0.6.1 is
itself mis-scaled (image and language embeddings ~1/sqrt(2048) of openpi), so
"matches native" passed while GPU parity failed at cosine 0.47. This fixture
pins outputs from the stack that was GPU-qualified before the migration:
LeRobot 0.5.1 + transformers 5.3.0 + Tether cdb94dc. Any stack that drifts
from openpi scale fails against it.

    capture (old stack):  python tests/_lerobot_pi_golden.py capture {pi0,pi05} OUT.pt
    compare (new stack):  python tests/_lerobot_pi_golden.py compare {pi0,pi05} GOLDEN.pt
    export  (new stack):  python tests/_lerobot_pi_golden.py export  {pi0,pi05} GOLDEN.pt DIR

The tiny model shrinks constructor sizes only (two layers, width 32, two-layer
SigLIP, vocabulary 64, 16x16 images, three-action chunk). Weights are saved in
the fixture, so initialisation order differences between versions cannot leak in.
"""

from __future__ import annotations

import sys

NUM_STEPS = 10
# float32 tolerance: the old stack divides image features by sqrt(2048) inside
# transformers 5.3 and LeRobot 0.5.1 multiplies them back; 5.4+ skips both,
# and language scaling moved into GemmaTextScaledWordEmbedding. Equal maths,
# different rounding, so allow a few float32 ulps at these magnitudes.
RTOL = 1e-5
ATOL = 1e-5


def _modules(family):
    if family == "pi0":
        from lerobot.policies.pi0 import modeling_pi0 as module

        return module, module.PI0Pytorch, module.PI0Config
    from lerobot.policies.pi05 import modeling_pi05 as module

    return module, module.PI05Pytorch, module.PI05Config


def build(family):
    import torch
    from unittest.mock import patch

    torch.set_num_threads(1)
    torch.manual_seed(123)
    module, cls, config_cls = _modules(family)
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
    return model


def make_inputs():
    import torch

    g = torch.Generator().manual_seed(7)
    # Images in [-1, 1] like LeRobot's preprocessed SigLIP input.
    return {
        "img0": torch.rand(2, 3, 16, 16, generator=g) * 2 - 1,
        "img1": torch.rand(2, 3, 16, 16, generator=g) * 2 - 1,
        "img2": torch.rand(2, 3, 16, 16, generator=g) * 2 - 1,
        "imask0": torch.tensor([True, True]),
        "imask1": torch.tensor([True, False]),
        "imask2": torch.tensor([False, True]),
        "tokens": torch.randint(0, 64, (2, 4), generator=g),
        "lmask": torch.tensor([[True, True, False, False], [True, True, True, False]]),
        "state": torch.randn(2, 4, generator=g),
        "noise": torch.randn(2, 3, 4, generator=g),
        "actions": torch.randn(2, 3, 4, generator=g),
        "time": torch.tensor([0.7, 0.3]),
    }


def _obs(family, x):
    obs = [
        [x["img0"], x["img1"], x["img2"]],
        [x["imask0"], x["imask1"], x["imask2"]],
        x["tokens"],
        x["lmask"],
    ]
    return obs + ([x["state"]] if family == "pi0" else [])


def outputs(family, model, x):
    """Native outputs, then Tether's patched export forward (installs patches)."""
    import copy

    import torch

    obs = _obs(family, x)
    native = copy.deepcopy(model)
    out = {}
    with torch.no_grad():
        prefix, pad, _ = native.embed_prefix(*obs[:4])
        out["prefix_embs"] = prefix
        out["prefix_pad"] = pad.to(torch.int64)
        out["loss"] = native(*obs, x["actions"], x["noise"], x["time"])
        out["actions_1"] = native.sample_actions(*obs, noise=x["noise"].clone(), num_steps=1)
        out["actions_10"] = native.sample_actions(
            *obs, noise=x["noise"].clone(), num_steps=NUM_STEPS
        )
        from tether.exporters import monolithic

        monolithic.apply_export_patches()
        if family == "pi05":
            monolithic._apply_pi05_denoise_step_patch()
        out["patched_actions_10"] = model.sample_actions(
            *obs, noise=x["noise"].clone(), num_steps=NUM_STEPS
        )
    return out


def capture(family, path):
    import importlib.metadata as md

    import torch

    model = build(family)
    x = make_inputs()
    state = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
    out = outputs(family, model, x)
    meta = {p: md.version(p) for p in ("lerobot", "transformers", "torch")}
    torch.save({"state_dict": state, "inputs": x, "outputs": out, "meta": meta}, path)
    print(family, "captured", meta)


def _load(family, path):
    import torch

    golden = torch.load(path, weights_only=True)
    model = build(family)
    missing, unexpected = model.load_state_dict(golden["state_dict"], strict=False)
    # Tied lm_head weights may be keyed differently across versions; anything
    # else missing means the architectures diverged and must fail loudly.
    assert not unexpected, unexpected
    assert all("lm_head" in k for k in missing), missing
    return golden, model


def _report(name, actual, expected):
    import torch

    diff = (actual.double() - expected.double()).abs()
    cos = torch.nn.functional.cosine_similarity(
        actual.double().flatten(), expected.double().flatten(), dim=0
    )
    scale = actual.double().norm() / expected.double().norm()
    print(
        f"{name}: max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e} "
        f"cos={cos.item():.9f} norm_ratio={scale.item():.6f}"
    )


def compare(family, path):
    import torch

    golden, model = _load(family, path)
    out = outputs(family, model, golden["inputs"])
    failures = []
    for name, expected in golden["outputs"].items():
        actual = out[name]
        if actual.dtype.is_floating_point:
            _report(name, actual, expected)
        try:
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
    print("golden from", golden["meta"])
    assert not failures, "\n".join(failures)
    print(family, "golden parity passed")


def export(family, path, out_dir):
    """Real monolithic toolchain (onnx-diagnostic patches, torch.export,
    torch.onnx.export, Where-dtype fix) on the tiny model, run in ONNX Runtime
    and compared with the OLD-stack golden. Mirrors export_{pi0,pi05}_monolithic
    minus checkpoint loading and tokenizer probing; B=1 like the exporter."""
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    import torch
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    from tether.exporters import monolithic

    golden, model = _load(family, path)
    x = golden["inputs"]
    monolithic.apply_export_patches()
    if family == "pi05":
        monolithic._apply_pi05_denoise_step_patch()
    monolithic._force_eager_attn(model)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, i0, i1, i2, m0, m1, m2, tokens, lmask, *rest):
            kwargs = {"noise": rest[-1], "num_steps": NUM_STEPS}
            args = [[i0, i1, i2], [m0, m1, m2], tokens, lmask, *rest[:-1]]
            return self.model.sample_actions(*args, **kwargs)

    names = ["img0", "img1", "img2", "imask0", "imask1", "imask2", "tokens", "lmask"]
    names += (["state"] if family == "pi0" else []) + ["noise"]
    # Row 0 of the golden batch: all three cameras present for row 0 except
    # camera 2; B=1 matches the production exporter's trace shape.
    feed = {n: x[n][:1].clone() for n in names}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    wrapper = Wrapper().eval()
    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(wrapper, tuple(feed.values()), dynamic_shapes=None, strict=False)
    torch.onnx.export(
        ep,
        tuple(feed.values()),
        str(onnx_path),
        input_names=names,
        output_names=["actions"],
        opset_version=19,
    )
    monolithic._fix_onnx_where_dtype_mismatches(onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (actions,) = session.run(None, {n: v.numpy() for n, v in feed.items()})
    actual = torch.from_numpy(np.asarray(actions))
    expected = golden["outputs"]["actions_10"][:1]
    _report("onnx_actions_10", actual, expected)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print(family, "onnx export golden parity passed")


if __name__ == "__main__":
    mode, family = sys.argv[1], sys.argv[2]
    {"capture": capture, "compare": compare, "export": export}[mode](family, *sys.argv[3:])
