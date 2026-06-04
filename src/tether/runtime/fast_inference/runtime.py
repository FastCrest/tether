"""FastKernelsPolicyRuntime — dispatch class for `tether serve --fast-kernels`.

Wraps either ``Pi05FastKernelsInference`` (Triton + CUDA Graph) or falls back
to the existing ORT ``PolicyRuntime`` silently. The fallback is **not a band-aid**
(per CLAUDE.md "no silent fallbacks that paper over errors") — it's a
**user-facing feature** that lets ``--fast-kernels`` work on ANY host: CUDA
hosts get the Triton path, Mac/CPU/unsupported-sm hosts get ORT with an INFO
log explaining why.

Fallback telemetry (``fast_kernels_active: bool``) is load-bearing per the
kill-trigger ADR (``2026-05-20-fast-kernels-kill-triggers.md``, Trigger 2:
fallback-rate ceiling). Without it we can't measure if ``--fast-kernels`` is
reaching customers.

V1 scope (T-2): Pi0.5 only. ``--fast-kernels`` on SmolVLA / GR00T / Pi0 raises
a clear error with the deferral timeline.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FastKernelsPolicyRuntime:
    """Dispatch: Triton fast-kernels path with silent ORT fallback.

    Attempts to build ``Pi05FastKernelsInference`` in order:
    1. Hardware gate (sm check)
    2. Shape whitelist (PaliGemma SigLIP-base check)
    3. Triton smoke import
    4. Runtime construction + prepare_triton_inference + graph capture

    If ANY step fails, falls back to ORT with an INFO log + sets
    ``self.fast_kernels_active = False`` for telemetry.

    Args:
        vla: A built ``Pi05VLA`` instance (from ``from_lerobot_policy`` or
            any other spine builder) with components already on CUDA.
        fallback_factory: Callable that returns the ORT fallback runtime
            when fast-kernels can't be used. Signature: ``() -> Any``
            (returns something with a ``predict_action`` method).
            If None, fallback raises RuntimeError instead of silently
            degrading (useful for testing).
        capture: Whether to enable CUDA Graph capture. Default True for
            production; False for diagnostic/parity testing.
        **kwargs: Forwarded to ``Pi05FastKernelsInference``.
    """

    def __init__(
        self,
        vla: Any,
        *,
        fallback_factory: Any | None = None,
        capture: bool = True,
        **kwargs: Any,
    ) -> None:
        self.fast_kernels_active: bool = False
        self._fallback_factory = fallback_factory
        self._inner: Any = None

        # Step 1: Hardware gate
        try:
            from tether.kernels._hardware_gate import is_fast_kernels_hardware_compatible
            ok, msg = is_fast_kernels_hardware_compatible()
            if not ok:
                self._fallback("hardware", msg)
                return
        except Exception as e:
            self._fallback("hardware-import", str(e))
            return

        # Step 2: Shape whitelist
        try:
            from tether.kernels._shape_whitelist import validate_shape_signature
            vit_cfg = vla.vision_backbone.model.vision_model.config
            shape_config = {
                "vit_hidden": vit_cfg.hidden_size,
                "vit_intermediate": vit_cfg.intermediate_size,
                "vit_num_heads": vit_cfg.num_attention_heads,
                "image_size": vit_cfg.image_size,
                "patch_size": vit_cfg.patch_size,
            }
            ok, msg = validate_shape_signature(shape_config)
            if not ok:
                self._fallback("shape", msg)
                return
        except Exception as e:
            self._fallback("shape-check", str(e))
            return

        # Step 3: Triton smoke import
        try:
            import triton  # noqa: F401
        except ImportError as e:
            self._fallback("triton-import", str(e))
            return

        # Step 4: Build runtime
        try:
            from tether.runtime.fast_inference.pi05 import Pi05FastKernelsInference
            runtime = Pi05FastKernelsInference(
                vla,
                capture=capture,
                _skip_hardware_gate=True,
                _skip_shape_whitelist=True,
                **kwargs,
            )
            runtime.prepare_triton_inference()
            self._inner = runtime
            self.fast_kernels_active = True
            logger.info("--fast-kernels: Triton + CUDA Graph path active (Pi0.5, %s)",
                        "captured" if capture else "eager")
        except Exception as e:
            self._fallback("runtime-build", str(e))
            return

    def _fallback(self, reason: str, msg: str) -> None:
        logger.info("--fast-kernels: falling back to ORT (%s: %s)", reason, msg)
        self.fast_kernels_active = False
        if self._fallback_factory is not None:
            self._inner = self._fallback_factory()
        else:
            self._inner = None

    def predict_action(self, **kwargs: Any) -> Any:
        if self._inner is None:
            raise RuntimeError(
                "--fast-kernels requested but both Triton and ORT fallback are "
                "unavailable. Pass a fallback_factory to enable ORT fallback, or "
                "drop --fast-kernels."
            )
        return self._inner.predict_action(**kwargs)

    @property
    def is_active(self) -> bool:
        return self.fast_kernels_active


__all__ = ["FastKernelsPolicyRuntime"]
