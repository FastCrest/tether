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
trick. The right abstraction is a 30-line helper, not a 600-line
exporter.
"""

from __future__ import annotations

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
