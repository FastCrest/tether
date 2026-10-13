"""export_patches(): onnx-diagnostic's export patches with its eager_mask patch
made to match transformers 5.4+.

onnx-diagnostic 0.9.3 patches ``masking_utils.eager_mask`` with the pre-5.4
``(batch_size, cache_position, kv_length, ...)`` signature, so any eager mask
built under ``torch_export_patches`` on transformers 5.5.4 raised
"missing 1 required positional argument: 'cache_position'" (the SmolVLA
monolithic export failure in the Studio 0.1.26 release check).
"""

from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if importlib.util.find_spec("onnx_diagnostic") is None:
    pytest.skip("requires the monolithic extra (onnx-diagnostic)", allow_module_level=True)
if tuple(int(p) for p in torch.__version__.split(".")[:2]) < (2, 8):
    pytest.skip("onnx-diagnostic export patches need torch>=2.8", allow_module_level=True)
if tuple(int(p) for p in transformers.__version__.split(".")[:2]) < (5, 4):
    pytest.skip("covers the transformers>=5.4 eager_mask signature", allow_module_level=True)

from transformers import PretrainedConfig  # noqa: E402
from transformers import masking_utils as mu  # noqa: E402

from tether.exporters.monolithic import export_patches  # noqa: E402


def _cases():
    cfg = PretrainedConfig()
    cfg._attn_implementation = "eager"
    x = torch.zeros(2, 6, 4)
    pad = torch.ones(2, 6, dtype=torch.bool)
    pad[1, -2:] = False
    return {
        "bidirectional_padded": lambda: mu.create_bidirectional_mask(
            config=cfg, inputs_embeds=x, attention_mask=pad
        ),
        "bidirectional_unpadded": lambda: mu.create_bidirectional_mask(
            config=cfg, inputs_embeds=x, attention_mask=torch.ones(2, 6, dtype=torch.bool)
        ),
        "causal_with_offset": lambda: mu.eager_mask(
            batch_size=2, q_length=3, kv_length=6, q_offset=3, attention_mask=pad
        ),
        "bf16": lambda: mu.eager_mask(
            batch_size=2, q_length=6, kv_length=6, attention_mask=pad, dtype=torch.bfloat16
        ),
    }


def test_eager_masks_match_native_transformers_inside_export_patches():
    cases = _cases()
    native = {name: build() for name, build in cases.items()}
    with export_patches():
        patched = {name: build() for name, build in cases.items()}
    for name in cases:
        if native[name] is None:
            assert patched[name] is None, name
            continue
        assert patched[name].dtype == native[name].dtype, name
        assert torch.equal(patched[name], native[name]), name


def test_native_eager_mask_restored_on_exit():
    native = mu.eager_mask
    with export_patches():
        assert mu.ALL_MASK_ATTENTION_FUNCTIONS["eager"] is not native
    assert mu.eager_mask is native
    assert mu.ALL_MASK_ATTENTION_FUNCTIONS["eager"] is native


def test_unknown_transformers_eager_mask_signature_is_refused(monkeypatch):
    def eager_mask(batch_size, query_length, kv_length, **kwargs):  # a future API
        raise AssertionError("must not be called")

    monkeypatch.setattr(mu, "eager_mask", eager_mask)
    with pytest.raises(RuntimeError, match="not the qualified 5.4\\+ signature"):
        with export_patches():
            pass
