"""PyTorch reference backend for ``reflex validate`` round-trip parity.

Reference semantics (v1)
------------------------
The "PyTorch reference" used by :mod:`reflex.validate_roundtrip` is **not** the
upstream HuggingFace / LeRobot model code. It is the *exporter's decomposed
PyTorch surrogate* — the same module graph that produced the ONNX file we are
validating. Comparing ONNX output against this surrogate proves that the export
itself is numerically faithful, which is the primary goal of the v1 round-trip
harness. End-to-end parity against the upstream policy stack is a v2 concern.

Concretely, this module reuses the same ``build_*_expert_stack`` helpers that
``reflex export`` uses to materialize the action-expert / DiT graph for ONNX
serialization. We load the safetensors into that decomposed stack, then run the
flow-matching denoising loop (Euler steps over the predicted velocity field)
starting from a caller-supplied ``initial_noise`` array. Sharing the exact same
noise array with the ONNX path is the seed bridge: ``torch.manual_seed`` does
not seed numpy, so noise must be generated once and threaded through both
runtimes byte-identically.

Per-model dispatch:

* ``smolvla`` → :func:`reflex.exporters.smolvla.build_expert_stack`
* ``pi0``     → :func:`reflex.exporters.pi0.build_pi0_expert_stack`
* ``gr00t``   → :func:`reflex.exporters.gr00t.build_gr00t_full_stack`
  pinned to ``embodiment_id=0`` (matches the ONNX export convention).
* ``pi05`` / ``openvla`` raise :class:`NotImplementedError` with the v1
  unsupported-model message.

CPU-only by default. pi0 / GR00T are 3B+ param models; bumping to CUDA is the
caller's explicit choice.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reflex.checkpoint import detect_model_type, load_checkpoint

logger = logging.getLogger("reflex.validate.pytorch")

UNSUPPORTED_MODEL_MESSAGE = (
    "reflex validate v1 supports smolvla, pi0, gr00t. "
    "For pi0.5 / openvla see roadmap."
)

# Per-model default denoise step counts. Overridden by reflex_config.json
# `num_denoising_steps` when present.
_DEFAULT_STEPS: dict[str, int] = {
    "smolvla": 10,
    "pi0": 10,
    "gr00t": 4,
}

_DEFAULT_CHUNK_SIZE: int = 50


def load_pytorch_backend(
    export_dir: Path,
    model_id: str | None,
    device: str = "cpu",
) -> "PyTorchBackend":
    """Load the exporter's decomposed PyTorch model for numerical reference.

    Reads ``reflex_config.json`` from ``export_dir`` to determine the model
    type and (optionally) the denoise step count, downloads / loads the
    checkpoint via :func:`reflex.checkpoint.load_checkpoint`, and reconstructs
    the same decomposed stack the exporter would build.

    Parameters
    ----------
    export_dir:
        Directory produced by ``reflex export`` containing
        ``reflex_config.json``.
    model_id:
        Optional HuggingFace ID for the upstream checkpoint. If ``None``, the
        ``model_id`` field from ``reflex_config.json`` is used.
    device:
        Device for the surrogate model (default ``"cpu"``).

    Returns
    -------
    PyTorchBackend
        Wrapper exposing :meth:`PyTorchBackend.forward`.
    """
    export_dir = Path(export_dir)
    config_path = export_dir / "reflex_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing reflex_config.json in export dir: {export_dir}"
        )
    import json

    config: dict[str, Any] = json.loads(config_path.read_text())
    resolved_id = model_id or config.get("model_id")
    if resolved_id is None:
        raise ValueError(
            "Cannot resolve checkpoint: pass model_id or include 'model_id' "
            "in reflex_config.json."
        )

    logger.info(
        "Loading PyTorch reference checkpoint: %s (device=%s)",
        resolved_id,
        device,
    )
    state_dict, _ckpt_config = load_checkpoint(resolved_id, device=device)

    model_type = config.get("model_type") or detect_model_type(state_dict)
    if model_type in ("pi05", "openvla"):
        raise NotImplementedError(UNSUPPORTED_MODEL_MESSAGE)
    if model_type not in ("smolvla", "pi0", "gr00t"):
        raise NotImplementedError(UNSUPPORTED_MODEL_MESSAGE)

    model = _build_decomposed_model(model_type, state_dict, device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Reconstructed %s decomposed surrogate: %.1fM params, device=%s",
        model_type,
        param_count / 1e6,
        device,
    )

    return PyTorchBackend(
        model_type=model_type,
        model=model,
        config=config,
        device=device,
    )


def _build_decomposed_model(
    model_type: str,
    state_dict: dict[str, torch.Tensor],
    device: str,
) -> torch.nn.Module:
    """Dispatch to the appropriate exporter helper and move to ``device``."""
    if model_type == "smolvla":
        from reflex.exporters.smolvla import build_expert_stack

        # Match the exporter's head_dim heuristic: SmolVLA defaults to 64 and the
        # exporter only refines via AutoConfig when transformers is importable.
        head_dim = 64
        try:  # pragma: no cover - defensive, mirrors smolvla_exporter
            from transformers import AutoConfig

            vlm_cfg = AutoConfig.from_pretrained(
                "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            )
            head_dim = (
                vlm_cfg.text_config.hidden_size
                // vlm_cfg.text_config.num_attention_heads
            )
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger(__name__).warning(
                "Could not fetch SmolVLM2 head_dim from HF (%s); "
                "falling back to default head_dim=%d",
                exc,
                head_dim,
            )
        stack, _meta = build_expert_stack(state_dict, head_dim=head_dim)
        return stack.to(device).eval()

    if model_type == "pi0":
        from reflex.exporters.pi0 import build_pi0_expert_stack

        stack, _meta = build_pi0_expert_stack(state_dict, head_dim=128)
        return stack.to(device).eval()

    if model_type == "gr00t":
        from reflex.exporters.gr00t import build_gr00t_full_stack

        # Pin embodiment_id=0 to match the ONNX export convention.
        stack, _meta = build_gr00t_full_stack(state_dict, embodiment_id=0)
        return stack.to(device).eval()

    raise NotImplementedError(UNSUPPORTED_MODEL_MESSAGE)


class PyTorchBackend:
    """Calls into the exporter-reconstructed PyTorch model to produce action chunks.

    Holds a reference to the decomposed expert stack plus the ``reflex_config``
    metadata (used to resolve denoise step count and chunk size). The
    :meth:`forward` method runs the same flow-matching Euler loop the ONNX path
    will run, starting from a caller-supplied ``initial_noise`` array so both
    runtimes consume byte-identical noise.

    Note: in v1 the surrogate consumes only the action tokens, timestep, and
    position ids (matching what ``reflex export`` actually emits). The
    ``image``, ``prompt``, and ``state`` arguments to :meth:`forward` are
    accepted for API symmetry with future image/prompt-conditioned exports but
    are not threaded through the expert in this version. This intentionally
    mirrors the ONNX graph: cross-attention KV is a zero placeholder for
    SmolVLA / GR00T and pi0's expert is self-attention only.
    """

    def __init__(
        self,
        model_type: str,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: str,
    ) -> None:
        self.model_type: str = model_type
        self.model: torch.nn.Module = model
        self.config: dict[str, Any] = dict(config)
        self.device: str = device

        self.num_steps: int = int(
            self.config.get("num_denoising_steps", _DEFAULT_STEPS[model_type])
        )
        self.chunk_size: int = int(
            self.config.get("action_chunk_size", _DEFAULT_CHUNK_SIZE)
        )
        # action_dim is the *input/output* feature dim the ONNX accepts. For
        # GR00T full-stack it is the raw DoF dim; for SmolVLA / pi0 it is the
        # native action dim. The exporter writes this into reflex_config.json.
        action_dim = self.config.get("action_dim")
        if action_dim is None:
            expert_meta = self.config.get("expert", {})
            action_dim = expert_meta.get("action_dim")
        if action_dim is None:
            raise ValueError(
                "reflex_config.json missing 'action_dim'; cannot shape noise."
            )
        self.action_dim: int = int(action_dim)

    def forward(
        self,
        image: np.ndarray,
        prompt: str,
        state: np.ndarray,
        initial_noise: np.ndarray,
    ) -> np.ndarray:
        """Run the flow-matching denoising loop and return the action chunk.

        Parameters
        ----------
        image, prompt, state:
            Conditioning inputs. Accepted for API symmetry with future
            image/prompt-conditioned exports; not consumed by the v1 surrogate
            (see class docstring).
        initial_noise:
            Starting noise of shape ``[chunk_size, action_dim]`` (or
            ``[1, chunk_size, action_dim]``). Generated once by the caller and
            shared with the ONNX backend so both runtimes start from
            byte-identical bytes.

        Returns
        -------
        np.ndarray
            Final action chunk of shape ``[chunk_size, action_dim]``.
        """
        del image, prompt, state  # unused in v1 — see class docstring

        if initial_noise.ndim == 2:
            noise = initial_noise[None, ...]
        elif initial_noise.ndim == 3:
            noise = initial_noise
        else:
            raise ValueError(
                f"initial_noise must be 2D or 3D, got shape {initial_noise.shape}"
            )

        if noise.shape[-2] != self.chunk_size or noise.shape[-1] != self.action_dim:
            raise ValueError(
                f"initial_noise shape {noise.shape} does not match expected "
                f"[*, {self.chunk_size}, {self.action_dim}]"
            )

        actions = torch.from_numpy(noise.astype(np.float32, copy=True)).to(self.device)
        position_ids = (
            torch.arange(self.chunk_size, device=self.device)
            .unsqueeze(0)
            .expand(actions.shape[0], -1)
        )

        # Flow-matching Euler integration: t goes from 1 → 0 in `num_steps` steps.
        # Canonical scheme used by reflex.inference.flow_matching_denoise +
        # runtime/server.py:_run_denoise (and mirrored in _onnx_backend.forward):
        #   dt = -1.0 / num_steps
        #   t  = 1.0 + step * dt
        #   actions += velocity * dt
        dt = -1.0 / float(self.num_steps)

        with torch.no_grad():
            for step in range(self.num_steps):
                t = 1.0 + step * dt
                timestep = torch.full(
                    (actions.shape[0],),
                    t,
                    dtype=torch.float32,
                    device=self.device,
                )
                velocity = self.model(actions, timestep, position_ids, vlm_kv=None)
                if velocity.shape != actions.shape:
                    raise RuntimeError(
                        f"velocity shape {tuple(velocity.shape)} does not match "
                        f"actions shape {tuple(actions.shape)}; the v1 backend "
                        f"requires the surrogate to emit raw-action velocities. "
                        f"For GR00T this means using build_gr00t_full_stack "
                        f"(embodiment-pinned), not the bare DiT expert."
                    )
                actions = actions + velocity * dt

        return actions[0].detach().cpu().numpy().astype(np.float32, copy=False)


__all__ = [
    "PyTorchBackend",
    "UNSUPPORTED_MODEL_MESSAGE",
    "load_pytorch_backend",
]
