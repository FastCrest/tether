"""Inference-only-weights runtime — bind a flat weight dict to an
ORT session via IOBinding, never instantiating the nn.Module graph.

Lift #3 per ``features/01_serve/inference-only-weights.md``.
"""
from __future__ import annotations

from reflex.runtime.inference_weights.runtime import (
    InferenceWeightsRuntime,
)
from reflex.runtime.inference_weights.safetensors_direct import (
    apply_prefix_mapping,
    load_flat_dict_from_safetensors,
    load_flat_dict_from_safetensors_dir,
)
from reflex.runtime.inference_weights.weight_binder import (
    WeightBindingError,
    bind_weights_to_iobinding,
    validate_name_mapping,
)

__all__ = [
    "InferenceWeightsRuntime",
    "WeightBindingError",
    "apply_prefix_mapping",
    "bind_weights_to_iobinding",
    "load_flat_dict_from_safetensors",
    "load_flat_dict_from_safetensors_dir",
    "validate_name_mapping",
]
