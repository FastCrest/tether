"""InferenceWeightsRuntime — the consumer of the flat-dict.

Wraps an existing ORT InferenceSession (typically pi0.5's decomposed
``vlm_prefix.onnx`` + ``expert_denoise.onnx`` pair, or the monolithic
``model.onnx``) and binds weights from a flat ``{name: tensor}`` dict
to the session via ``IOBinding`` — never instantiating the source
``nn.Module`` graph at inference time.

Lift #3 Day 3 per ``features/01_serve/inference-only-weights_plan.md``.

Memory benefit: 30-40% peak RSS reduction on Pi0.5 + GR00T (the Day 5
Modal benchmark validates this number). Mechanism: the model's
``nn.Parameter`` objects aren't allocated; only the flat tensors live
in VRAM, and only the ORT session keeps them resident.

V1 scope: synthetic-input dispatch. Real Modal end-to-end parity vs
the standard runtime is Day 4's gate.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable

import torch

from tether.runtime.inference_weights.weight_binder import (
    WeightBindingError,
    bind_weights_to_iobinding,
    validate_name_mapping,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InferenceWeightsRuntime:
    """Functional inference path that consumes a flat weights dict.

    Args:
        flat_weights: ``{name: tensor}`` from
            ``BaseVLA.prepare_inference_weights()``. Names must match the
            ORT session's expected initializer names exactly.
        ort_session: An ``onnxruntime.InferenceSession`` (or a duck-typed
            mock for testing). The session is constructed from an ONNX
            export of the same VLA whose flat dict is passed.
        expected_names: The set of initializer / input names the ORT
            session expects to receive via IOBinding. Typically read
            from ``tether_config.json`` next to the ONNX file (the
            exporter records them at export time).
        device: ``"cuda"`` (default) or ``"cpu"``. CUDA is the perf path;
            CPU exists for portability / CI runs without GPU.
        device_id: CUDA device ordinal. Default 0.

    Raises:
        WeightBindingError: ``flat_weights`` keys don't match
            ``expected_names``. Raised at construction so misuse fails
            loud, not silently at first ``predict_action``.
    """

    def __init__(
        self,
        *,
        flat_weights: dict[str, torch.Tensor],
        ort_session: Any,
        expected_names: Iterable[str],
        device: str = "cuda",
        device_id: int = 0,
    ) -> None:
        self._session = ort_session
        self._device = device
        self._device_id = device_id
        self._expected_names = set(expected_names)

        # Validate at construction — fail loud at startup, not later.
        validate_name_mapping(
            flat_keys=set(flat_weights),
            expected_names=self._expected_names,
        )

        self._flat_weights = flat_weights
        self._io_binding = None  # Created lazily on first predict_action

        logger.info(
            "InferenceWeightsRuntime ready: %d weight tensors, device=%s:%d",
            len(flat_weights), device, device_id,
        )

    def _ensure_io_binding(self) -> Any:
        """Lazily create IOBinding + bind all weights. One-shot setup
        per session — IOBinding persists across predict_action calls.
        """
        if self._io_binding is None:
            self._io_binding = self._session.io_binding()
            bind_weights_to_iobinding(
                flat_weights=self._flat_weights,
                io_binding=self._io_binding,
                expected_names=self._expected_names,
                device=self._device,
                device_id=self._device_id,
            )
            logger.debug("IOBinding wired with %d weight tensors", len(self._flat_weights))
        return self._io_binding

    def predict_action(
        self,
        *,
        runtime_inputs: dict[str, torch.Tensor],
        output_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Dispatch a single inference step.

        Args:
            runtime_inputs: ``{input_name: tensor}`` for the
                per-request inputs (observation, state, etc.). Distinct
                from the weights bound at construction.
            output_names: Optional list of output names to fetch. None
                → all outputs.

        Returns:
            ``{output_name: tensor}`` — outputs as torch.Tensor (host
            side, copied from device).
        """
        import onnxruntime as ort  # noqa: F401

        io_binding = self._ensure_io_binding()

        # Bind per-request inputs (separate from weights — these change every call).
        for name, tensor in runtime_inputs.items():
            np_array = tensor.detach().cpu().numpy()
            ortval = ort.OrtValue.ortvalue_from_numpy(
                np_array, self._device, self._device_id,
            )
            io_binding.bind_ortvalue_input(name, ortval)

        # Bind outputs — let ORT allocate.
        if output_names is None:
            output_names = [o.name for o in self._session.get_outputs()]
        for name in output_names:
            io_binding.bind_output(name, self._device, self._device_id)

        self._session.run_with_iobinding(io_binding)

        outputs: dict[str, torch.Tensor] = {}
        for ortval, name in zip(io_binding.get_outputs(), output_names):
            outputs[name] = torch.from_numpy(ortval.numpy())
        return outputs

    @property
    def num_weight_tensors(self) -> int:
        return len(self._flat_weights)


__all__ = ["InferenceWeightsRuntime", "WeightBindingError"]
