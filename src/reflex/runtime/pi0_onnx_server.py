"""Pi0OnnxServer — serves pi0 via the monolithic ONNX exported by
scripts/modal_pi0_monolithic_export.py.

The ONNX is a single-graph trace of PI0Pytorch.sample_actions with a
fixed num_steps baked in at export time. Current ship: num_steps=1
(cos=+1.0000000 vs PyTorch verified on 2026-04-18, Modal run 11).

Design: matches SmolVLANativeServer's interface (.load(), .predict())
so the runtime is uniform across VLA families. Customers use
`reflex serve <export_dir>` and get the same API regardless of model.

Known limitation: num_steps is baked into the ONNX at export time.
Changing num_steps requires re-export. Multi-step parity (num_steps=10)
is tracked as a follow-up goal — the onnx-diagnostic + torch.export
path hits a 835→886 shape tracer bug at num_steps>1 which is unsolved
as of 2026-04. See
https://github.com/FastCrest/reflex-vault/blob/main/reflex_vla/01_architecture/pi0_monolithic_wrap_pattern.md
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Pi0OnnxServer:
    """Monolithic pi0 ONNX runtime.

    Loads the single-file ONNX from `export_dir/model.onnx` (or explicit
    onnx_path). Runs one inference per /act request — the ONNX internally
    computes prefix embeddings + runs the flow-matching Euler loop for
    `num_steps` as baked-in at export time.
    """

    def __init__(
        self,
        export_dir: str | Path,
        onnx_path: str | Path | None = None,
        providers: list[Any] | None = None,
        device: str = "cpu",
        max_batch: int = 1,
        strict_providers: bool = True,
    ):
        self.export_dir = Path(export_dir)
        self._explicit_onnx_path = Path(onnx_path) if onnx_path else None
        self._requested_providers = providers
        self._requested_device = device
        self._max_batch = max_batch
        self.providers: list[Any] = providers or ["CPUExecutionProvider"]
        self._session: Any = None
        self.config: dict[str, Any] = {}
        self._ready = False
        self._inference_mode = "pi0_onnx_monolithic"
        self._provider_mode = "onnx_cpu"
        self._active_providers: list[str] = []
        self._provider_plan: Any = None
        self._strict_providers = strict_providers
        self._tokenizer: Any = None
        self._tokenizer_unavailable = False

    def _find_onnx_path(self) -> Path:
        if self._explicit_onnx_path is not None:
            return self._explicit_onnx_path
        # Default layouts we accept
        candidates = [
            self.export_dir / "model.onnx",
            self.export_dir / "monolithic" / "model.onnx",
            self.export_dir / "pi0_monolithic.onnx",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No pi0 monolithic ONNX found in {self.export_dir}. Tried: {candidates}"
        )

    def load(self) -> None:
        import onnxruntime as ort
        from reflex.runtime.ort_providers import (
            build_ort_provider_plan,
            gpu_provider_active,
            gpu_provider_requested,
            make_ort_session_options,
        )

        onnx_path = self._find_onnx_path()
        logger.info("Loading pi0 monolithic ONNX: %s", onnx_path)
        self._provider_plan = build_ort_provider_plan(
            self.export_dir,
            device=self._requested_device,
            requested_providers=self._requested_providers,
            available_providers=list(ort.get_available_providers()),
            max_batch=self._max_batch,
            onnx_path=onnx_path,
        )
        self.providers = self._provider_plan.providers
        logger.info(
            "Requested providers: %s; available: %s; trt=%s reason=%s",
            self.providers,
            self._provider_plan.available_providers,
            self._provider_plan.used_trt,
            self._provider_plan.trt_disabled_reason,
        )
        start = time.perf_counter()
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=make_ort_session_options(onnx_path),
            providers=self.providers,
        )
        elapsed = time.perf_counter() - start
        self._active_providers = self._session.get_providers()
        if gpu_provider_requested(self.providers) and not gpu_provider_active(self._active_providers):
            msg = (
                "CUDA/TensorRT provider requested for pi0 monolithic runtime, but "
                f"ONNX Runtime activated {self._active_providers}."
            )
            if self._strict_providers:
                raise RuntimeError(msg)
            logger.warning(msg)
        if "TensorrtExecutionProvider" in self._active_providers:
            self._provider_mode = "onnx_trt_fp16"
        elif gpu_provider_active(self._active_providers):
            self._provider_mode = "onnx_gpu"
        else:
            self._provider_mode = "onnx_cpu"

        # Load config if present (optional — may not exist for ad-hoc ONNX)
        cfg_path = self.export_dir / "reflex_config.json"
        if cfg_path.exists():
            self.config = json.loads(cfg_path.read_text())

        # Extract expected inputs from the ONNX graph for validation
        self._input_names = [i.name for i in self._session.get_inputs()]
        logger.info(
            "Pi0OnnxServer ready in %.2fs (inputs: %s, active_providers=%s)",
            elapsed, self._input_names, self._active_providers,
        )
        self._ready = True

    def _get_tokenizer(self):
        if self._tokenizer_unavailable:
            return None
        if self._tokenizer is None:
            from reflex.runtime.tokenizers import load_export_tokenizer
            self._tokenizer = load_export_tokenizer(
                self.export_dir,
                self.config,
                default_ref="google/paligemma-3b-pt-224",
            )
            if self._tokenizer is None:
                self._tokenizer_unavailable = True
                logger.warning("Tokenizer unavailable; using dummy tokens")
                return None
        return self._tokenizer

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(
        self,
        image: np.ndarray | list[np.ndarray] | None = None,
        instruction: str = "",
        state: list[float] | np.ndarray | None = None,
        noise: np.ndarray | None = None,
        lang_tokens: np.ndarray | None = None,
        lang_masks: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run one pi0 forward pass.

        Inputs:
          image: single image array [H, W, 3] uint8, or list of 3 (base + wrist_l + wrist_r)
          instruction: task text (used to produce lang_tokens if not provided)
          state: robot state [max_state_dim]
          noise: fixed noise [1, chunk, action_dim] for deterministic output
          lang_tokens: optional pre-tokenized input (skips internal tokenization)
          lang_masks: optional

        Returns dict with 'actions' + metadata.
        """
        if not self._ready:
            return {"error": "Model not loaded. Call load() first."}

        t0 = time.perf_counter()

        # Image prep: accept single image or list of 3; replicate if needed
        images_list = (
            image if isinstance(image, list)
            else [image] if image is not None
            else []
        )
        while len(images_list) < 3:
            images_list.append(
                images_list[-1] if images_list else np.zeros((224, 224, 3), np.uint8)
            )
        images_list = images_list[:3]

        def _prep_img(img: np.ndarray) -> np.ndarray:
            arr = np.asarray(img, dtype=np.float32)
            if arr.max() > 1.5:  # uint8-ish range → scale to [-1, 1]
                arr = arr / 255.0
                arr = arr * 2.0 - 1.0
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)[None, :]  # [1, 3, H, W]
            return arr.astype(np.float32)

        img_base = _prep_img(images_list[0])
        img_wrist_l = _prep_img(images_list[1])
        img_wrist_r = _prep_img(images_list[2])

        mask = np.ones((1,), dtype=np.bool_)

        # Lang: take externally-supplied tokens or tokenize the instruction
        if lang_tokens is None:
            tok = self._get_tokenizer()
            if tok is not None:
                enc = tok(instruction or " ", return_tensors="np", padding="max_length",
                          max_length=16, truncation=True)
                lang_tokens = enc["input_ids"].astype(np.int64)
                lang_masks = enc["attention_mask"].astype(np.bool_)
            else:
                lang_tokens = np.zeros((1, 16), dtype=np.int64)
                lang_masks = np.ones((1, 16), dtype=np.bool_)
        if lang_masks is None:
            lang_masks = np.ones_like(lang_tokens, dtype=np.bool_)

        # State: pad to 32 if needed
        state_dim = int(self.config.get("max_state_dim", 32))
        if state is None:
            state_arr = np.zeros((1, state_dim), dtype=np.float32)
        else:
            state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
            if state_arr.shape[0] < state_dim:
                state_arr = np.concatenate([state_arr, np.zeros(state_dim - state_arr.shape[0])])
            elif state_arr.shape[0] > state_dim:
                state_arr = state_arr[:state_dim]
            state_arr = state_arr[None, :].astype(np.float32)

        # Noise
        if noise is None:
            chunk = int(self.config.get("chunk_size", 50))
            action_dim = int(self.config.get("action_dim", 32))
            noise = np.random.RandomState(0).randn(1, chunk, action_dim).astype(np.float32)
        noise = np.asarray(noise, dtype=np.float32)

        # Run the ONNX
        ort_inputs = {
            "img_base": img_base,
            "img_wrist_l": img_wrist_l,
            "img_wrist_r": img_wrist_r,
            "mask_base": mask,
            "mask_wrist_l": mask,
            "mask_wrist_r": mask,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "state": state_arr,
            "noise": noise,
        }
        # Drop any extra inputs the ONNX doesn't have (defensive)
        ort_inputs = {k: v for k, v in ort_inputs.items() if k in self._input_names}

        actions = self._session.run(None, ort_inputs)[0]  # [B, chunk, action_dim]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        actions_out = actions[0]  # drop batch dim for single-request response

        # `denoising_steps` is the README-documented field; `num_denoising_steps`
        # is the internal config key name. Emit both for backwards compat.
        steps = int(self.config.get("num_denoising_steps", 1))
        return {
            "actions": actions_out.tolist(),
            "num_actions": int(actions_out.shape[0]),
            "action_dim": int(actions_out.shape[1]),
            "latency_ms": round(elapsed_ms, 1),
            "hz": round(1000.0 / elapsed_ms, 1) if elapsed_ms > 0 else 0,
            "inference_mode": self._inference_mode,
            "provider_mode": self._provider_mode,
            "active_providers": list(self._active_providers),
            "denoising_steps": steps,
            "num_denoising_steps": steps,
        }

    # --- create_app lifespan compat ---------------------------------------
    # ReflexServer has these methods; we mirror them so create_app's lifespan
    # can treat monolithic servers as drop-in replacements.

    async def predict_from_base64_async(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        import base64
        import io
        image = None
        if image_b64:
            try:
                from PIL import Image
                img_bytes = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image = np.array(img)
            except Exception as e:
                return {"error": f"Failed to decode image: {e}"}
        return self.predict(image=image, instruction=instruction, state=state)

    async def start_batch_worker(self) -> None:
        # Monolithic runtime is single-request; batching is ReflexServer's job.
        return None

    async def stop_batch_worker(self) -> None:
        return None

    async def run_batch(self, requests: list) -> list[dict[str, Any]]:
        """PolicyRuntime entry point. Sequential dispatch per request; the
        queue + scheduler primitive applies regardless of whether ORT can
        batch (monolithic exports are static-shape; dynamic-batch is a
        future feature). Per chunk-budget-batching ADR 2026-04-24."""
        results: list[dict[str, Any]] = []
        for req in requests:
            res = await self.predict_from_base64_async(
                image_b64=getattr(req, "image", None),
                instruction=getattr(req, "instruction", "") or "",
                state=getattr(req, "state", None),
            )
            results.append(res)
        return results
