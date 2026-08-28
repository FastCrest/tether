"""Load the exported artifact used by ``tether verify``.

This module is intentionally fail-closed: verification never rebuilds an
"optimized" runtime from native policy weights and never guesses that an
unknown export layout is compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tether.export_config import decomposed_layout, load_tether_config
from tether.verification_evidence import normalize_verification_device


class UnsupportedVerificationBackend(RuntimeError):
    """The export is valid, but no artifact-backed verifier supports it."""


def _providers(device: str) -> list[str]:
    device = normalize_verification_device(device)
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise AssertionError("unreachable verification device")


def _require_session_device(session: Any, *, device: str, name: str) -> None:
    """Reject an ORT session whose primary active backend mismatches the CLI."""

    raw_session = getattr(session, "session", session)
    active = list(raw_session.get_providers())
    expected = "CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"
    if not active or active[0] != expected:
        raise UnsupportedVerificationBackend(
            f"{name} verification device mismatch: requested {device!r}, "
            f"active providers are {active!r}"
        )


class MonolithicOnnxVerificationAdapter:
    """Bridge a model-specific monolithic ONNX server to LIBERO inference."""

    def __init__(self, server: Any) -> None:
        self._server = server

    def reset_cache(self) -> None:
        return None

    def get_stats(self) -> dict[str, Any]:
        return {"backend": getattr(self._server, "_inference_mode", "monolithic_onnx")}

    def predict_action_chunk(
        self,
        *,
        img_base: Any,
        img_wrist_l: Any,
        img_wrist_r: Any,
        mask_base: Any,
        mask_wrist_l: Any,
        mask_wrist_r: Any,
        lang_tokens: Any,
        lang_masks: Any,
        noise: Any,
        state: Any,
        episode_id: str,
    ) -> np.ndarray:
        del episode_id
        result = self._server.predict(
            image=[img_base, img_wrist_l, img_wrist_r],
            image_masks=[mask_base, mask_wrist_l, mask_wrist_r],
            state=state,
            noise=noise,
            lang_tokens=np.asarray(lang_tokens),
            lang_masks=np.asarray(lang_masks),
        )
        if "error" in result:
            raise RuntimeError(f"exported monolithic inference failed: {result['error']}")
        actions = np.asarray(result["actions"], dtype=np.float32)
        return actions[None, ...] if actions.ndim == 2 else actions


def _primary_model_artifact(config: dict[str, Any], root: Path) -> Path:
    models = [
        root / str(item["path"])
        for item in config["artifacts"]
        if item["role"] == "model" and str(item["path"]).endswith(".onnx")
    ]
    if len(models) != 1:
        raise UnsupportedVerificationBackend(
            "monolithic_onnx verification requires exactly one declared model ONNX artifact"
        )
    return models[0]


def load_verification_inference(
    export_dir: str | Path,
    device: str = "cpu",
) -> Any:
    """Construct an inference object from the actual declared export artifacts."""
    device = normalize_verification_device(device)
    root = Path(export_dir)
    config = load_tether_config(root, inspect_artifacts=True)
    kind = config["export_kind"]
    providers = _providers(device)

    if kind == "monolithic_onnx":
        onnx_path = _primary_model_artifact(config, root)
        model_type = config["model_type"]
        server: Any
        if model_type == "pi0":
            from tether.runtime.pi0_onnx_server import Pi0OnnxServer

            server = Pi0OnnxServer(
                root,
                onnx_path=onnx_path,
                providers=providers,
                device=device,
                strict_providers=True,
            )
        elif model_type == "smolvla":
            from tether.runtime.smolvla_onnx_server import SmolVLAOnnxServer

            server = SmolVLAOnnxServer(
                root,
                onnx_path=onnx_path,
                providers=providers,
                device=device,
                strict_providers=True,
            )
        else:
            raise UnsupportedVerificationBackend(
                f"monolithic verifier does not support model_type={model_type!r}"
            )
        server.load()
        _require_session_device(server._session, device=device, name="monolithic")
        return MonolithicOnnxVerificationAdapter(server)

    if kind == "decomposed_onnx":
        layout = decomposed_layout(config)
        if layout == "pi05_split":
            declared_models = {
                str(item["path"]) for item in config["artifacts"] if item["role"] == "model"
            }
            decomposed = config.get("decomposed", {})
            referenced_models = {
                str(decomposed.get("vlm_prefix_onnx", "")),
                str(decomposed.get("expert_denoise_onnx", "")),
            }
            if "" in referenced_models or referenced_models != declared_models:
                raise UnsupportedVerificationBackend(
                    "pi05_split secondary paths must exactly match the declared, "
                    "integrity-checked model artifacts"
                )
            from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference

            inference = Pi05DecomposedInference(root, providers=providers)
            _require_session_device(
                inference._sess_prefix,
                device=device,
                name="vlm_prefix",
            )
            _require_session_device(
                inference._sess_expert,
                device=device,
                name="expert_denoise",
            )
            return inference
        if layout == "expert_stack":
            raise UnsupportedVerificationBackend(
                "expert_stack verification is disabled because that runtime does "
                "not preserve image and language conditioning"
            )
        raise UnsupportedVerificationBackend(
            f"decomposed verifier does not support layout={layout!r}"
        )

    raise UnsupportedVerificationBackend(
        f"local verification does not support export_kind={kind!r}"
    )


__all__ = [
    "MonolithicOnnxVerificationAdapter",
    "UnsupportedVerificationBackend",
    "load_verification_inference",
]
