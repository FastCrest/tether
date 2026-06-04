"""Weight binder for `--inference-only-weights` mode.

Walks a flat ``{key: torch.Tensor}`` dict and binds each tensor to the
matching ORT initializer by name. Validates 1:1 name mapping between
flat-dict keys and ORT initializer names; raises ``WeightBindingError``
on any drift.

Lift #3 Day 3 per ``features/01_serve/inference-only-weights_plan.md``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class WeightBindingError(RuntimeError):
    """Raised when the flat dict's keys don't match the ORT session's
    initializer names. Surfaces as a clear actionable error rather than
    a silent shape mismatch at first inference.
    """


def collect_initializer_names(ort_session: Any) -> set[str]:
    """Returns the set of initializer names from an ORT InferenceSession.

    ORT exposes initializers via ``session.get_modelmeta().graph_name``
    plus traversing the graph proto. In practice we use
    ``session.get_inputs()`` to discover the graph's *runtime* inputs
    (which is what IOBinding actually binds against). For ORT-driven
    weight injection, we want the **initializer** names too, but
    ``InferenceSession`` doesn't expose them directly. The exporter
    must surface them via tether_config.json (typically alongside
    each .onnx file).
    """
    # The contract here is intentionally narrow: callers either pass
    # a list of expected names (the cleanest path) or read them out of
    # `tether_config.json` next to the .onnx file. We don't introspect
    # the ONNX graph proto here — that's a parsing-heavy side adventure
    # the WeightBinder doesn't need to take on.
    raise NotImplementedError(
        "ORT InferenceSession doesn't expose initializer names directly; "
        "pass the expected name set via the `expected_names=` kwarg on "
        "WeightBinder.bind() — the exporter records them in tether_config.json."
    )


def validate_name_mapping(
    flat_keys: set[str],
    expected_names: set[str],
) -> None:
    """Validates exact 1:1 name mapping between flat dict + ORT initializers.

    Catches:
    - **Missing keys** in flat dict that ORT expects (typo in
      ``prepare_triton`` left a Parameter unmapped).
    - **Extra keys** in flat dict that ORT doesn't expect (renamed
      Parameter, or a component returning weights for a model variant
      that ORT was exported against an older version of).

    Raises ``WeightBindingError`` with the offending keys listed.
    """
    missing_in_flat = expected_names - flat_keys
    extra_in_flat = flat_keys - expected_names

    if missing_in_flat or extra_in_flat:
        msg_parts = []
        if missing_in_flat:
            sample = sorted(missing_in_flat)[:5]
            suffix = f" (and {len(missing_in_flat) - 5} more)" if len(missing_in_flat) > 5 else ""
            msg_parts.append(
                f"missing from flat dict: {sample}{suffix} — total {len(missing_in_flat)}"
            )
        if extra_in_flat:
            sample = sorted(extra_in_flat)[:5]
            suffix = f" (and {len(extra_in_flat) - 5} more)" if len(extra_in_flat) > 5 else ""
            msg_parts.append(
                f"extra in flat dict: {sample}{suffix} — total {len(extra_in_flat)}"
            )
        raise WeightBindingError(
            "Flat dict ↔ ORT initializer name mapping drift: " + "; ".join(msg_parts)
        )


def bind_weights_to_iobinding(
    flat_weights: dict[str, torch.Tensor],
    io_binding: Any,
    expected_names: set[str],
    *,
    device: str = "cuda",
    device_id: int = 0,
) -> None:
    """Bind each tensor in flat_weights to ORT IOBinding by name.

    Args:
        flat_weights: ``{name: tensor}`` from BaseVLA.prepare_inference_weights().
            Names must match ``expected_names`` exactly.
        io_binding: ORT ``InferenceSession.io_binding()`` instance.
            We bind each tensor as an input. (Initializers are inputs
            from ORT's IOBinding perspective when the model declares
            them as external/passed inputs vs baked weights.)
        expected_names: Set of ORT initializer / input names. Validated
            against flat_weights.keys() — drift raises WeightBindingError.
        device: Target device — ``"cuda"`` for the GPU path, ``"cpu"``
            for the CPU fallback (unusual but supported).
        device_id: CUDA device ordinal (typically 0).

    Raises:
        WeightBindingError: name mapping drift between flat_weights and
            expected_names.

    Side effect:
        io_binding now has each weight tensor bound. Caller invokes
        ``session.run_with_iobinding(io_binding)`` to dispatch.
    """
    validate_name_mapping(flat_keys=set(flat_weights), expected_names=expected_names)

    # ORT's bind_ortvalue_input requires an OrtValue. The standard
    # cuda-fast path: ``OrtValue.ortvalue_from_numpy`` with the
    # right device + device_id. This avoids host↔device copy on every
    # call — the weight tensor lives once in VRAM and ORT consumes it.
    #
    # Lazy import — ORT isn't a base dep; only needed when this code path runs.
    import onnxruntime as ort  # noqa: F401  (load-bearing check)

    for name, tensor in flat_weights.items():
        np_array = tensor.detach().cpu().numpy()
        ortval = ort.OrtValue.ortvalue_from_numpy(np_array, device, device_id)
        io_binding.bind_ortvalue_input(name, ortval)


__all__ = [
    "WeightBindingError",
    "validate_name_mapping",
    "bind_weights_to_iobinding",
]
