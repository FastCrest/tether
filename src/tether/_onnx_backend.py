"""ONNX inference backend for `tether validate` round-trip parity.

This module is the ONNX half of the cross-runtime parity harness used by
:mod:`tether.validate_roundtrip`. It loads the exported ONNX expert stack from
an export directory, runs the same Euler flow-matching denoise loop the
PyTorch reference uses, and returns the resulting action chunk as a numpy
array.

Provider policy (v1)
--------------------
v1 forces ``CPUExecutionProvider`` only. Validation is meant to be
deterministic and CI-portable; CUDA / TensorRT EPs introduce non-determinism
(kernel selection, reduction order, FP16 fallback) that would muddy the
parity signal we want to measure. The CUDA story for `tether validate` is
explicitly deferred (see plan boundaries → "Always" / "Never").

Seed bridge
-----------
``forward()`` accepts ``initial_noise`` as a numpy array supplied by the
orchestrator. The backend MUST consume that exact array (cast to float32 if
needed) — it must NOT call ``np.random`` or ``torch.randn`` internally. This
is what guarantees the PyTorch and ONNX paths see byte-identical noise even
though ``torch.manual_seed`` does not seed numpy.

GR00T note
----------
The exported GR00T ONNX graph has ``embodiment_id=0`` baked in at export time
(see ``gr00t_exporter.py``) and exposes only ``noisy_actions``, ``timestep``,
``position_ids`` as inputs. We still detect ``embodiment_id`` in the session
input list defensively: if a future exporter version surfaces it as a real
graph input, we'll feed ``np.array([0], dtype=np.int64)`` automatically.

Denoise schedule
----------------
We use the same canonical Euler scheme as ``runtime/server.py``:
``dt = -1.0 / num_steps``, ``t = 1.0 + step * dt``, integrating from t=1
toward t=0. Step count comes from ``tether_config.json["num_denoising_steps"]``
(typically 10 for SmolVLA / pi0, 4 for GR00T). Issue 4's PyTorch backend is
expected to mirror this scheme; Issue 6 will add explicit parity tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("tether.validate.onnx")


def _detect_model_type(config: dict[str, Any], onnx_path: Path) -> str:
    """Return the model type from config, falling back to filename heuristics."""
    mt = config.get("model_type")
    if mt:
        return str(mt).lower()
    name = onnx_path.name.lower()
    for candidate in ("gr00t", "pi0", "smolvla"):
        if candidate in name:
            return candidate
    parent = onnx_path.parent.name.lower()
    for candidate in ("gr00t", "pi0", "smolvla"):
        if candidate in parent:
            return candidate
    return "unknown"


def _read_opset(onnx_path: Path) -> int | None:
    """Read the default ONNX opset version from the graph, or None on failure."""
    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("onnx package not installed; skipping opset readout")
        return None
    try:
        model = onnx.load(str(onnx_path))
    except Exception as e:  # pragma: no cover - opset readout is best-effort
        logger.debug("Failed to load ONNX for opset detection: %s", e)
        return None
    for opset in model.opset_import:
        # The default ONNX domain has an empty domain string.
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    if model.opset_import:
        return int(model.opset_import[0].version)
    return None


class ONNXBackend:
    """Run the exported ONNX expert stack to produce action chunks.

    Mirrors the Euler flow-matching denoise loop used by the runtime server,
    pinned to ``CPUExecutionProvider`` for v1 reproducibility. Built by
    :func:`load_onnx_backend` — do not instantiate directly unless you have
    already created an :class:`onnxruntime.InferenceSession`.
    """

    def __init__(self, session: Any, config: dict[str, Any], model_type: str) -> None:
        self.session = session
        self.config: dict[str, Any] = config
        self.model_type: str = model_type

        expert_meta = config.get("expert", {}) or {}
        self.action_dim: int = int(
            config.get("action_dim", expert_meta.get("action_dim", 0))
        )
        self.chunk_size: int = int(config.get("action_chunk_size", 50))
        # Default step counts: 4 for gr00t (per gr00t_exporter), 10 elsewhere.
        default_steps = 4 if model_type == "gr00t" else 10
        self.num_steps: int = int(config.get("num_denoising_steps", default_steps))

        # Cache session input names so we can populate optional inputs (e.g.
        # embodiment_id) only when the graph actually exposes them.
        try:
            self._input_names: set[str] = {i.name for i in session.get_inputs()}
        except Exception:  # pragma: no cover - defensive for mock sessions
            self._input_names = set()

        if self.action_dim <= 0:
            logger.warning(
                "ONNXBackend: action_dim missing/zero in config; "
                "shape assertion will rely on initial_noise's last dim",
            )

    def forward(
        self,
        image: np.ndarray,  # noqa: ARG002 - reserved for v2 full-stack path
        prompt: str,        # noqa: ARG002 - reserved for v2 full-stack path
        state: np.ndarray,  # noqa: ARG002 - reserved for v2 full-stack path
        initial_noise: np.ndarray,
    ) -> np.ndarray:
        """Run the denoise loop on the ONNX graph and return ``[chunk, action_dim]``.

        ``image``, ``prompt``, ``state`` are accepted for API parity with the
        PyTorch backend; v1 validates only the decomposed expert stack (which
        is the only thing the exporter actually emits as ONNX). Conditioning
        is reserved for the v2 full-stack validation path.

        ``initial_noise`` MUST be the same numpy array consumed by the PyTorch
        backend; this is the seed bridge.
        """
        if not isinstance(initial_noise, np.ndarray):
            raise TypeError(
                f"initial_noise must be a numpy array; got {type(initial_noise)!r}"
            )

        # Normalize to float32 + add batch dim if missing.
        noise = initial_noise
        if noise.dtype != np.float32:
            noise = noise.astype(np.float32)
        if noise.ndim == 2:
            noise = noise[None, ...]
        if noise.ndim != 3:
            raise ValueError(
                f"initial_noise must be [chunk, action_dim] or "
                f"[1, chunk, action_dim]; got shape {initial_noise.shape}"
            )
        # Make sure we don't mutate the caller's array.
        current = np.ascontiguousarray(noise)

        b, chunk, action_dim = current.shape
        position_ids = np.arange(chunk, dtype=np.int64).reshape(1, -1)
        if b != 1:
            position_ids = np.tile(position_ids, (b, 1))

        # Optional embodiment_id (only fed if the graph actually accepts it).
        feed_embodiment = "embodiment_id" in self._input_names
        if self.model_type == "gr00t" and feed_embodiment:
            embodiment_arr = np.array([0], dtype=np.int64)
        else:
            embodiment_arr = None
            if self.model_type == "gr00t":
                logger.debug(
                    "GR00T graph has embodiment_id baked in (not a graph input); "
                    "pinned to 0 at export time.",
                )

        # Optional vlm_kv (v0.2+ exports include this as an input).
        feed_vlm_kv = "vlm_kv" in self._input_names
        vlm_kv = None
        if feed_vlm_kv:
            # Read the actual dim from the ONNX input shape (expert.vlm_kv_dim),
            # NOT the VLM hidden_size — they differ (320 vs 960 for SmolVLA).
            vlm_kv_shape = [
                inp.shape for inp in self.session.get_inputs()
                if inp.name == "vlm_kv"
            ][0]
            vlm_kv_dim = vlm_kv_shape[-1] if isinstance(vlm_kv_shape[-1], int) else 320
            vlm_kv = np.zeros((b, 1, vlm_kv_dim), dtype=np.float32)
            logger.debug("Expert expects vlm_kv input (dim=%d); feeding zeros for validation", vlm_kv_dim)

        dt = -1.0 / float(self.num_steps)
        for step in range(self.num_steps):
            t = 1.0 + step * dt
            timestep = np.array([t] * b, dtype=np.float32)
            inputs: dict[str, np.ndarray] = {
                "noisy_actions": current,
                "timestep": timestep,
                "position_ids": position_ids,
            }
            if embodiment_arr is not None:
                inputs["embodiment_id"] = embodiment_arr
            if feed_vlm_kv:
                inputs["vlm_kv"] = vlm_kv
            velocity = self.session.run(None, inputs)[0]
            current = current + velocity * dt

        # Drop the batch dim for the orchestrator-facing shape contract.
        out = current[0] if current.shape[0] == 1 else current

        # Output shape assertion (per acceptance criteria).
        expected_action_dim = self.action_dim or action_dim
        expected_shape = (self.chunk_size, expected_action_dim)
        if out.shape != expected_shape:
            raise AssertionError(
                f"ONNXBackend.forward: output shape {out.shape} != "
                f"expected {expected_shape} (chunk_size, action_dim)"
            )
        return out


def load_onnx_backend(export_dir: Path, device: str = "cpu") -> ONNXBackend:
    """Load the ONNX expert-stack session from an export directory.

    Parameters
    ----------
    export_dir:
        Path to a `tether export` output directory. Must contain
        ``tether_config.json`` and ``expert_stack.onnx`` (or an ONNX path
        recorded under ``files.expert_onnx`` in the config).
    device:
        Accepted for API symmetry with the PyTorch backend; ignored in v1
        because validation always runs on ``CPUExecutionProvider``. A non-cpu
        value is logged at INFO so callers know the request was honored as
        CPU intentionally.
    """
    export_dir = Path(export_dir)
    if not export_dir.exists() or not export_dir.is_dir():
        raise FileNotFoundError(f"Export directory does not exist: {export_dir}")

    config_path = export_dir / "tether_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing tether_config.json in export dir: {export_dir}"
        )
    config: dict[str, Any] = json.loads(config_path.read_text())

    # Resolve ONNX path: prefer config, fall back to canonical filename.
    onnx_path: Path | None = None
    files = config.get("files") or {}
    if isinstance(files, dict):
        candidate = files.get("expert_onnx")
        if candidate:
            cand = Path(candidate)
            if not cand.is_absolute():
                cand = export_dir / cand
            if cand.exists():
                onnx_path = cand
    if onnx_path is None:
        default = export_dir / "expert_stack.onnx"
        if default.exists():
            onnx_path = default
    if onnx_path is None:
        raise FileNotFoundError(
            f"No ONNX graph found in {export_dir}: looked for "
            f"`expert_stack.onnx` and config.files.expert_onnx",
        )

    if device.lower() != "cpu":
        logger.info(
            "load_onnx_backend: device=%s requested, but v1 validation pins "
            "CPUExecutionProvider for determinism. Honoring CPU.",
            device,
        )

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for `tether validate`. "
            "Install with: pip install onnxruntime"
        ) from e

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    active_providers = session.get_providers()
    active = active_providers[0] if active_providers else "<none>"
    if active != "CPUExecutionProvider":  # pragma: no cover - defensive
        logger.warning(
            "Requested CPUExecutionProvider but ORT activated %s; "
            "validation determinism may be affected.",
            active,
        )

    opset = _read_opset(onnx_path)
    model_type = _detect_model_type(config, onnx_path)

    logger.info(
        "Loaded ONNX backend: path=%s provider=%s opset=%s model_type=%s "
        "chunk_size=%s action_dim=%s steps=%s",
        onnx_path.name,
        active,
        opset if opset is not None else "unknown",
        model_type,
        config.get("action_chunk_size"),
        config.get("action_dim"),
        config.get("num_denoising_steps"),
    )

    return ONNXBackend(session=session, config=config, model_type=model_type)


__all__ = ["ONNXBackend", "load_onnx_backend"]
