"""OpenVLA — non-spine model, ships as a shim (per lift #1 decision S-4).

OpenVLA is **intentionally NOT on the BaseVLA spine.** It's an autoregressive
Llama-2-7B with an argmax-over-bins action head — doesn't fit the
flow-matching component pattern that the spine's 6-slot taxonomy is
built around. Forcing it onto the spine would require either a fake
"argmax head" that doesn't share any abstraction with FlowMatchingHead
/ DITHead, or contorting the spine to fit. Neither pulls its weight.

Per decision S-4 (in
``reflex_context/01_decisions/2026-05-19-fluxvla-lift-program-decisions.md``):

  - OpenVLA stays a shim with the existing ``optimum-cli export onnx``
    + bin-to-continuous postprocess flow.
  - ``ModelEntry`` declares ``vla_type="_openvla_shim"`` to mark this
    non-spine status.
  - The spine's ABC enforcement (REQUIRED_SLOTS / OPTIONAL_SLOTS) is
    not violated — there's no ``OpenVLAVLA(BaseVLA)`` class to misuse.

OpenVLA is architecturally very different from the flow-matching VLAs
that Tether's custom exporters target (SmolVLA, pi0, pi0.5, GR00T).
Its "action head" is literally `argmax(lm_logits[:, -action_dim:])`
followed by a bin-to-continuous lookup:

    bin_idx = vocab_size - token_id - 1
    action_normalized = bin_centers[bin_idx]
    action = unnormalize(action_normalized, norm_stats[dataset])

where ``vocab_size`` is the effective text vocab
(``text_config.vocab_size - pad_to_multiple_of``, 32000 for openvla-7b)
and ``bin_centers`` are the centers between 256 edges over [-1, 1].

There is no dedicated action expert to reconstruct. The full model is
Llama-2-7B + DINOv2 + SigLIP + 3-layer projector — ~7.5B params of
standard transformers architecture that HuggingFace's optimum-onnx
already knows how to export.

## The recommended workflow

Rather than duplicate optimum-onnx for no architectural insight,
Tether points users at the existing path and helps with the only
OpenVLA-specific bit — the bin-to-action postprocessing:

    pip install 'optimum[onnxruntime]'
    optimum-cli export onnx --model openvla/openvla-7b ./openvla_onnx/

    # Then at inference time:
    from tether.postprocess.openvla import decode_actions
    logits = ort_session.run(None, {...})[0]  # [b, seq, vocab]
    actions = decode_actions(
        logits=logits,
        action_dim=7,
        dataset_name="bridge_orig",  # or whatever norm_stats key
        norm_stats=config["norm_stats"],
    )

## Why Tether's value-add is low here

Tether exists to unlock VLAs that HF can't ship — those with custom
action experts (flow matching over action chunks, AdaRMSNorm/AdaLN
time conditioning, alternating cross/self-attn on VLM KV caches).
OpenVLA has none of these. It is a vanilla VLM with a post-processing
trick.

## Monolithic export (M4 #54)

`optimum-cli export onnx` cannot export openvla-7b: optimum dispatches on
``config.model_type`` and ``"openvla"`` (remote-code ``auto_map``) has no
``OnnxConfig`` mapping, a gap the CLI cannot express
(``trust_remote_code`` is a Python-API argument). ``export_openvla``
below therefore keeps raising its honest ``NotImplementedError``, and the
parity-capable exporter is ``export_openvla_monolithic`` instead: plain
``torch.onnx.export`` of the pinned reference's multimodal forward
``(input_ids, attention_mask, pixel_values) -> lm_logits``, with the
fused-vision channel count read off ``use_fused_vision_backbone``
(``[1, 6, 224, 224]`` for openvla-7b's ``torch.split(pixel_values, [3, 3],
dim=1)`` backbone). Tether owns only the export + the bin-to-continuous
decode in ``tether.postprocess.openvla``; OpenVLA stays off the spine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tether.config import ExportConfig


_OPENVLA_HINT = """\
OpenVLA (openvla/openvla-7b) is a vanilla Llama-2-7B VLM — its "action
head" is argmax(lm_logits[:, -7:]) + bin lookup, not a custom expert
stack. Tether's exporters reconstruct flow-matching action experts that
HuggingFace can't ship; OpenVLA has no such expert, so there's nothing
Tether-specific to build.

Use the normal HuggingFace path instead:
    pip install 'optimum[onnxruntime]'
    optimum-cli export onnx --model openvla/openvla-7b ./openvla_onnx/

For the bin-to-action postprocessing, use:
    from tether.postprocess.openvla import decode_actions
"""


def build_openvla_expert_stack(
    state_dict: dict[str, torch.Tensor],
    **_: Any,
) -> Any:
    raise NotImplementedError(_OPENVLA_HINT)


def export_openvla(
    config: ExportConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    raise NotImplementedError(_OPENVLA_HINT)


def _openvla_pixel_channels(config: Any) -> int:
    """Channel count ``pixel_values`` must carry for this checkpoint.

    ``openvla-7b`` sets ``use_fused_vision_backbone: true`` and its
    ``PrismaticVisionBackbone.forward`` runs ``torch.split(pixel_values,
    [3, 3], dim=1)`` (one 3-channel image per tower), so the tensor is
    ``[bsz, 6, resolution, resolution]``. A non-fused fine-tune takes 3.
    Mirrors the harness's ``_pixel_channels`` idiom; refuses to guess.
    """
    fused = getattr(config, "use_fused_vision_backbone", None)
    if not isinstance(fused, bool):
        raise ValueError(
            "Could not read use_fused_vision_backbone from the OpenVLA config; "
            "refusing to guess the pixel_values channel count."
        )
    return 6 if fused else 3


class _OpenVLALogitsWrapper(torch.nn.Module):
    """Traceable multimodal forward: (ids, mask, pixels) -> lm_logits."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits


def export_openvla_monolithic(
    model_id: str,
    output_dir: str | Path,
    *,
    prompt_tokens: int = 32,
    image_size: int = 224,
    opset: int = 18,
) -> dict[str, Any]:
    """Export OpenVLA as a single monolithic ONNX (M4 #54).

    Traces the pinned reference's multimodal forward
    ``(input_ids, attention_mask, pixel_values) -> lm_logits`` with plain
    ``torch.onnx.export`` — no optimum dispatch (there is no ``openvla``
    ``OnnxConfig`` mapping). The Llama backbone is forced to ``"eager"``
    attention for traceability (same manual softmax math the reference's
    SDPA-math backend computes; the parity gate decides).

    Args:
        model_id: local snapshot dir of the exact pinned revision (so the
            hashed artifact and the reference revision agree), or a HF id.
        output_dir: directory receiving ``model.onnx`` (+ external data).
        prompt_tokens: text length to trace (receipt harness uses 32).
        image_size: square pixel edge (openvla-7b towers are 224).
        opset: ONNX opset (18 = torch 2.4 default, ORT-CUDA clean).

    Returns a status dict mirroring ``export_gr00t_monolithic``.
    """
    import logging

    from transformers import AutoConfig, AutoModelForVision2Seq

    logger = logging.getLogger(__name__)

    logger.info("[openvla] Loading config from %s", model_id)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    pixel_channels = _openvla_pixel_channels(config)
    vocab_size = int(config.text_config.vocab_size)
    logger.info(
        "[openvla] pixel_channels=%d image_size=%d prompt_tokens=%d",
        pixel_channels, image_size, prompt_tokens,
    )

    # Eager attention BEFORE construction propagates into the inner
    # LlamaForCausalLM built by modeling_prismatic (it reads
    # config._attn_implementation in from_config). SDPA's fused kernel has
    # no torchscript-export symbolic; eager is the same softmax math.
    config._attn_implementation = "eager"
    try:
        config.text_config._attn_implementation = "eager"
    except AttributeError:
        pass

    logger.info("[openvla] Loading reference weights (fp32) ...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        config=config,
    )
    model.eval()
    try:
        got = model.language_model.config._attn_implementation
    except AttributeError:
        got = None
    logger.info("[openvla] resolved LLM attn_implementation=%r", got)

    wrapper = _OpenVLALogitsWrapper(model).eval()

    dummy_ids = torch.zeros(1, prompt_tokens, dtype=torch.long)
    dummy_mask = torch.ones(1, prompt_tokens, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    dummy_pixels = torch.randn(
        1, pixel_channels, image_size, image_size,
        dtype=torch.float32, generator=gen,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    # Traceability patch (pi0 `apply_export_patches` pattern): the pinned
    # modeling code builds its patch attention mask with
    # ``torch.full((B, N), True, dtype=..., device=...)``, whose overload
    # the torchscript tracer cannot alias-analyze (``aten::full`` "isn't a
    # special case" — first-hand 2026-09-18). ``torch.ones(size, dtype,
    # device) * fill`` is value-identical (x * 1.0 is exact per IEEE;
    # True -> 1 in integer dtypes) and traces to Ones/Mul. Scoped to the
    # export call only; restored after.
    _real_ones = torch.ones

    def _traceable_full(size: Any, fill_value: Any, *args: Any, **kwargs: Any) -> Any:
        # The tracer rejects a bool Scalar overload (aten::mul(Tensor, bool));
        # normalize True/False to 1/0 first — value-identical in every dtype.
        if isinstance(fill_value, bool):
            fill_value = 1 if fill_value else 0
        return _real_ones(size, *args, **kwargs) * fill_value

    # Static shapes (no dynamic_axes): upstream supports batch size 1 only
    # (``prepare_inputs_for_generation`` raises for batch > 1), and static
    # tracing keeps every factory size concrete — no dynamic Shape/Concat
    # subgraphs for ORT's CUDA EP to force-move (cf. M4 #53).
    logger.info("[openvla] torch.onnx.export (opset %d, static shapes) ...", opset)
    _orig_full = torch.full
    torch.full = _traceable_full  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask, dummy_pixels),
                str(onnx_path),
                input_names=["input_ids", "attention_mask", "pixel_values"],
                output_names=["logits"],
                opset_version=opset,
            )
    finally:
        torch.full = _orig_full
    logger.info("[openvla] ONNX export ok")

    size_mb = onnx_path.stat().st_size / 1e6
    data_files = list(output_dir.glob("*.data")) + list(output_dir.glob("*.bin"))
    total_mb = size_mb + sum(f.stat().st_size for f in data_files) / 1e6
    return {
        "status": "ok",
        "onnx_path": str(onnx_path),
        "size_mb": total_mb,
        "pixel_channels": pixel_channels,
        "prompt_tokens": prompt_tokens,
        "image_size": image_size,
        "vocab_size": vocab_size,
        "opset": opset,
    }
