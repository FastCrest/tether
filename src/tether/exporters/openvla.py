"""OpenVLA — non-spine model, ships as a shim (per lift #1 decision S-4).

OpenVLA is **intentionally NOT on the BaseVLA spine.** It's an autoregressive
Llama-2-7B with a tokenized action head — unlike the flow-matching component
pattern used by Tether's custom pi0/pi0.5/SmolVLA/GR00T exporters.

Per decision S-4:

  - OpenVLA stays a shim around a standard HuggingFace/Optimum ONNX export.
  - ``ModelEntry`` declares ``vla_type="_openvla_shim"``.
  - Tether owns the OpenVLA-specific token-to-continuous postprocess and parity
    evidence rather than duplicating Optimum's generic transformer exporter.

## Action semantics

OpenVLA generates one action token autoregressively for each action dimension.
Those exact generated token IDs are decoded as:

    discretized = tokenizer_vocab_size - action_token_ids
    bin_idx = clip(discretized - 1, 0, len(bin_centers) - 1)
    action_normalized = bin_centers[bin_idx]
    action = unnormalize(action_normalized, norm_stats[dataset])

``bin_centers`` are the 255 centers between 256 uniformly spaced edges over
[-1, 1]. For openvla-7b the tokenizer vocabulary is 32000 even though the LM
logit width may be padded to 32064.

A single prompt-forward logits tensor is **not** the OpenVLA action output. The
model must run the normal greedy autoregressive generation loop and retain the
seven generated token IDs (or one next-token logit row per generation step).

## Recommended workflow

Use the normal HuggingFace/Optimum export path rather than a Tether-specific
7B transformer exporter:

    pip install 'optimum[onnxruntime]'
    optimum-cli export onnx --model openvla/openvla-7b ./openvla_onnx/

Then run greedy autoregressive generation with the exported model. Once the
exact generated action token IDs are available:

    from tether.postprocess.openvla import decode_token_ids

    actions = decode_token_ids(
        generated_action_token_ids,
        vocab_size=processor.tokenizer.vocab_size,
        norm_stats=norm_stats,
        dataset_name="bridge_orig",
    )

If a runtime records the next-token logits for each generation step instead,
``decode_actions`` accepts exactly ``[batch, action_dim, padded_vocab]`` — one
row per autoregressive step. It deliberately rejects arbitrary prompt-sequence
logits so a single forward pass cannot be mistaken for generated actions.

For Studio qualification, use the exact tokenized-action/export parity harness
and receipt contract rather than treating exporter existence as parity.
"""

from __future__ import annotations

from typing import Any

import torch

from tether.config import ExportConfig


_OPENVLA_HINT = """\
OpenVLA (openvla/openvla-7b) is a vanilla autoregressive VLM whose action output
is generated token IDs followed by OpenVLA's action-token decoder. Tether does
not duplicate Optimum's standard transformer export path.

Use:
    pip install 'optimum[onnxruntime]'
    optimum-cli export onnx --model openvla/openvla-7b ./openvla_onnx/

Run the exported model autoregressively to generate the action token sequence,
then decode those exact IDs with:
    from tether.postprocess.openvla import decode_token_ids

A single prompt-forward ONNX logits tensor is not equivalent to OpenVLA action
generation and must not be decoded as though its last sequence positions were
generated actions.
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
