"""pi05 accel engine behind Tether's ``InferenceProtocol`` (``predict_action_chunk``).

Inputs are what the rollout and server already produce for the decomposed ONNX
path: three processed camera images in [-1, 1] with presence masks, the
state-containing language token IDs with their mask, and the shared flow noise.
The output is the padded, *normalized* action chunk ``[1, chunk, 32]``; the
checkpoint's postprocessor still owns robot-space decoding, exactly as for the
ONNX backends. One engine (and its CUDA graphs) is kept per active-camera
count, all sharing one prepared FP8 weight set.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tether.accel import require_supported_device

MAX_ACTION_DIM = 32


def active_views(images, masks) -> list[int]:
    """Indices of present cameras. Batch size must be 1: the engine serves one observation."""
    active = []
    for i, (image, mask) in enumerate(zip(images, masks)):
        image, mask = np.asarray(image), np.asarray(mask).reshape(-1)
        if image.shape[0] != 1 or mask.size != 1:
            raise ValueError("accel pi05 serves one observation at a time (batch size 1)")
        if bool(mask[0]):
            active.append(i)
    if not 1 <= len(active) <= 3:
        raise ValueError("accel pi05 requires one to three active cameras")
    return active


def prompt_ids(lang_tokens, lang_masks) -> np.ndarray:
    """The valid token IDs of a ``[1, length]`` token/mask pair."""
    tokens, mask = np.asarray(lang_tokens), np.asarray(lang_masks).astype(bool)
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape != mask.shape:
        raise ValueError("pi05 token IDs and masks must have matching [1, length] shapes")
    return tokens[mask].astype(np.int64)


class AccelPi05Inference:
    """FP8 Triton pi05 runtime for Thor / RTX 4090 / RTX 5090."""

    def __init__(self, weights: Any, device: Any, *, chunk: int = 50, target: str = "unknown"):
        import torch

        self._torch = torch
        self._weights = weights
        self._device = torch.device(device)
        self._chunk = int(chunk)
        self._engines: dict[int, Any] = {}
        self._calls = 0
        self._target = target

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, *, device="cuda", steps: int | None = None):
        """Prepare ``model.safetensors`` + ``config.json`` of a gemma_2b/gemma_300m pi05 checkpoint."""
        import json
        from pathlib import Path

        import torch

        from tether.accel.pi05.weights import load_pi05_weights

        dev = torch.device(device)
        target = require_supported_device(dev)
        cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
        if (
            cfg.get("max_action_dim", MAX_ACTION_DIM) != MAX_ACTION_DIM
            or cfg.get("paligemma_variant", "gemma_2b") != "gemma_2b"
            or cfg.get("action_expert_variant", "gemma_300m") != "gemma_300m"
        ):
            raise ValueError("accel pi05 requires gemma_2b/gemma_300m with max_action_dim=32")
        chunk = int(cfg.get("chunk_size", 50))
        steps = int(steps if steps is not None else cfg.get("num_inference_steps", 10))
        weights = load_pi05_weights(checkpoint_dir, dev, steps=steps)
        return cls(weights, dev, chunk=chunk, target=target)

    def _engine(self, views: int):
        from tether.accel.pi05.engine import Pi05Engine

        eng = self._engines.get(views)
        if eng is None:
            eng = self._engines[views] = Pi05Engine(
                self._weights, views, self._device, chunk=self._chunk
            )
        return eng

    def reset_cache(self) -> None:
        # The prefix K/V is recomputed on every call; graphs depend only on geometry.
        return None

    def get_stats(self) -> dict[str, Any]:
        return {
            "backend": "accel_pi05_fp8",
            "precision": "fp8",
            "target": self._target,
            "activation_scales": "per-token dynamic",
            "weight_scales": "per-channel",
            "graphs": sum(len(e._graphs) for e in self._engines.values()),
            "calls": self._calls,
        }

    def predict_action_chunk(
        self,
        *,
        img_base,
        img_wrist_l,
        img_wrist_r,
        mask_base,
        mask_wrist_l,
        mask_wrist_r,
        lang_tokens,
        lang_masks,
        noise,
        state=None,
        episode_id=None,
    ) -> np.ndarray:
        del state, episode_id  # pi05 carries state inside the language tokens
        torch = self._torch
        images = (img_base, img_wrist_l, img_wrist_r)
        views = active_views(images, (mask_base, mask_wrist_l, mask_wrist_r))
        ids = prompt_ids(lang_tokens, lang_masks)
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != (1, self._chunk, MAX_ACTION_DIM):
            raise ValueError(
                f"noise must be [1, {self._chunk}, {MAX_ACTION_DIM}], got {noise.shape}"
            )
        stacked = torch.from_numpy(
            np.concatenate([np.asarray(images[i], dtype=np.float32) for i in views])
        )
        eng = self._engine(len(views))
        with torch.cuda.device(self._device):
            eng.set_prompt(ids)
            raw = eng.infer(stacked, noise=noise[0])
            actions = raw.detach().float().cpu().numpy().copy()
        if not np.isfinite(actions).all():
            raise RuntimeError("accel pi05 engine returned nonfinite actions")
        self._calls += 1
        return actions[None, ...]


__all__ = ["AccelPi05Inference", "active_views", "prompt_ids"]
