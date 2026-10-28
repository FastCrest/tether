"""Native-PyTorch SmolVLA serving path.

Imports lerobot's `SmolVLAPolicy` directly and wraps it behind the same
`predict(image, instruction, state)` interface that `TetherServer` exposes.
Uses the real model's preprocessor / postprocessor pipelines so the 79
silent-failure bugs catalogued at
https://github.com/FastCrest/reflex-vault/tree/main/reflex_vla/02_bugs_fixed don't
apply — correctness is delegated to upstream code.

The only op we decompose is ``nn.RMSNorm`` (replaced with
``tether.decompose.DecomposedRMSNorm``) because TensorRT's ONNX parser
does not yet support the opset-23 ``RMSNormalization`` op (NVIDIA issue
#4639). Everything else — attention, RoPE, GQA, MLP, the whole cross-
attention composition that burned this project's Apr-17 — runs upstream
PyTorch kernels.

Routing
-------
Set ``TETHER_NATIVE=1`` when running ``tether serve`` or when constructing
``TetherServer`` directly. The regular decomposed ONNX path is still
reachable (and still used by ``tether export`` for Jetson-side TRT engine
builds) when the env var is unset.

Latency trade-off
-----------------
Native mode: ~2-3x slower than TRT-FP16 (no kernel fusion). Target
hardware is A10G / H100 / consumer GPU. For Jetson, still use the
decomposed ONNX + TRT path.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SmolVLANativeServer:
    """PyTorch-native replacement for the decomposed-ONNX TetherServer.

    Compatible ``.load()`` and ``.predict(image, instruction, state)`` so it
    can be dropped into ``create_app()``. Takes an export directory for
    config access (``tether_config.json``) but loads the actual model via
    ``SmolVLAPolicy.from_pretrained`` from the Hub-cached copy.
    """

    def __init__(
        self,
        export_dir: str | Path,
        device: str = "cuda",
        providers: list[str] | None = None,
        strict_providers: bool = True,
        safety_config: str | Path | None = None,
        adaptive_steps: bool = False,
        cloud_fallback_url: str = "",
        deadline_ms: float | None = None,
        max_batch: int = 1,
        batch_timeout_ms: float = 5.0,
        num_denoising_steps: int | None = None,
    ):
        if providers:
            raise ValueError(
                "ORT providers cannot be honored by the native PyTorch runtime. "
                "Use the ONNX server for --providers, or omit providers for native execution."
            )
        self.export_dir = Path(export_dir)
        self._requested_device = device
        self._safety_config_path = Path(safety_config) if safety_config else None
        self._cloud_fallback_url = cloud_fallback_url
        self._deadline_ms = deadline_ms
        self.config = self._load_config()
        self.model_id = self.config.get(
            "model_id",
            self.config.get("vlm_model_id", "lerobot/smolvla_libero"),
        )

        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self._ready = False
        self._inference_mode = "pytorch_native"
        # Wedge telemetry (kept for compat with TetherServer's response shape)
        self._action_guard = None
        self._split_orchestrator = None
        self._last_good_actions: np.ndarray | None = None
        self._deadline_misses = 0
        self._adaptive_steps = adaptive_steps
        # Batching: native path is single-request for now
        self._max_batch = 1
        self._batch_timeout_s = max(0.0, batch_timeout_ms) / 1000.0
        self._batch_queue = None
        self._batch_worker_task = None
        self._batches_run = 0
        self._batched_requests = 0
        self._rmsnorm_patches = 0
        # Set by create_app when --rtc is on; load() then checks that the
        # postprocessor can be inverted for the leftover conversion.
        self._rtc_required = False

    # ---- RTC (LeRobot's own RTCProcessor) -------------------------------

    def rtc_guidance_support(self) -> str | None:
        """None: LeRobot's SmolVLA denoise loop is RTC-aware (euler_integrate)."""
        return None

    def _action_unnormalizer(self) -> Any:
        """The postprocessor's single action UnnormalizerProcessorStep.

        Refuses when the postprocessor does anything else to action values
        (only device moves are allowed), because then processed actions cannot
        be mapped back into the space the denoiser guides in.
        """
        from lerobot.processor import DeviceProcessorStep, UnnormalizerProcessorStep

        steps = list(getattr(self.postprocessor, "steps", []))
        unnorm = [st for st in steps if isinstance(st, UnnormalizerProcessorStep)]
        other = [
            type(st).__name__ for st in steps
            if not isinstance(st, (UnnormalizerProcessorStep, DeviceProcessorStep))
        ]
        if len(unnorm) != 1 or other:
            raise RuntimeError(
                "RTC needs the SmolVLA postprocessor to be one action unnormalizer "
                f"(plus device moves) to convert the leftover; got {[type(st).__name__ for st in steps]}"
            )
        return unnorm[0]

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Processed (unnormalized) actions -> the policy's normalized space,
        using the postprocessor's own stats (LeRobot's forward transform)."""
        import torch

        step = self._action_unnormalizer()
        t = torch.as_tensor(np.asarray(actions, dtype=np.float32))
        return step._normalize_action(t, inverse=False).detach().cpu().numpy()

    def _ensure_rtc_processor(self, rtc: Any) -> None:
        """Point LeRobot's policy at an RTCConfig matching this guidance."""
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        cur = self.policy.config.rtc_config
        want = (RTCAttentionSchedule(rtc.schedule), float(rtc.max_guidance_weight))
        if (
            cur is None or not cur.enabled
            or (cur.prefix_attention_schedule, float(cur.max_guidance_weight)) != want
        ):
            self.policy.config.rtc_config = RTCConfig(
                enabled=True,
                prefix_attention_schedule=want[0],
                max_guidance_weight=want[1],
                execution_horizon=int(rtc.execution_horizon),
            )
            self.policy.init_rtc_processor()

    def _load_config(self) -> dict[str, Any]:
        p = self.export_dir / "tether_config.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def load(self) -> None:
        """Load SmolVLAPolicy + its preprocessor/postprocessor."""
        import torch

        logger.info("Loading native SmolVLAPolicy: %s", self.model_id)
        start = time.perf_counter()

        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            transition_to_batch,
            policy_action_to_transition,
            transition_to_policy_action,
        )

        # Resolve device. A CUDA request on a host without CUDA still runs on
        # CPU (existing behaviour) but says so; MPS must be available if asked.
        requested = self._requested_device
        if requested == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif requested == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("device='mps' requested but torch reports MPS unavailable")
            device = "mps"
        else:
            if requested not in ("cpu", "cuda"):
                raise ValueError(f"SmolVLA native supports cuda, mps or cpu; got {requested!r}")
            if requested == "cuda":
                logger.warning(
                    "SmolVLA native: device='cuda' requested but CUDA is unavailable; "
                    "running on CPU"
                )
            device = "cpu"
        self._device = device

        self.policy = SmolVLAPolicy.from_pretrained(self.model_id)
        self.policy.eval()
        self.policy.to(dtype=torch.float32).to(device)

        # Swap RMSNorm for DecomposedRMSNorm (TRT-compat for downstream export)
        self._swap_rmsnorm()

        # Build processors — force device override so CUDA-default configs
        # don't blow up on CPU-only machines.
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(self.model_id)

        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=repo_dir,
            config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
            overrides={"device_processor": {"device": device}},
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=repo_dir,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        self.action_dim = int(self.policy.config.max_action_dim)
        self.chunk_size = int(self.policy.config.chunk_size)
        self.num_denoising_steps = int(self.policy.config.num_steps)
        if self._rtc_required:
            self._action_unnormalizer()  # refuse now if the leftover can't be converted

        # Safety guard (tether guard) composition — re-uses TetherServer logic
        # Fail closed: a requested safety config that cannot be armed stops
        # load instead of serving unguarded actions.
        if self._safety_config_path is not None:
            from tether.safety import ActionGuard, SafetyLimits
            limits = SafetyLimits.from_json(self._safety_config_path)
            self._action_guard = ActionGuard(limits=limits, mode="clamp")
            logger.info(
                "tether guard loaded (native): %d joints, space=%s, units=%s",
                len(limits.joint_names), limits.action_space, limits.units,
            )

        elapsed = time.perf_counter() - start
        self._ready = True
        logger.info(
            "Native SmolVLAPolicy ready in %.1fs (device=%s, rmsnorm_patches=%d, "
            "action_dim=%d, chunk=%d, steps=%d)",
            elapsed, device, self._rmsnorm_patches,
            self.action_dim, self.chunk_size, self.num_denoising_steps,
        )

    def _swap_rmsnorm(self) -> None:
        """Replace every nn.RMSNorm with DecomposedRMSNorm in-place.

        TensorRT's ONNX parser doesn't yet support opset-23
        ``RMSNormalization`` (NVIDIA TRT issue #4639). Keeping the
        decomposed variant means our Jetson-facing ``tether export`` path
        produces TRT-compilable graphs. The native path tolerates it because
        decomposed form is numerically equivalent.
        """
        import torch.nn as nn
        from tether.decompose import DecomposedRMSNorm

        count = 0
        # Walk every module; replace by setattr on the parent.
        for parent_name, parent in list(self.policy.named_modules()):
            for child_name, child in list(parent.named_children()):
                # Match by class name (covers nn.RMSNorm + any subclass the
                # loaded policy brings in from transformers).
                tn = type(child).__name__
                if tn in ("RMSNorm", "LlamaRMSNorm", "SmolLM2RMSNorm"):
                    # Extract the weight tensor (exact attr varies; try common names)
                    w = getattr(child, "weight", None)
                    if w is None:
                        continue
                    eps = getattr(child, "variance_epsilon", None)
                    if eps is None:
                        eps = getattr(child, "eps", 1e-6)
                    new_norm = DecomposedRMSNorm(w.detach().clone(), eps=eps)
                    new_norm.to(dtype=w.dtype, device=w.device)
                    setattr(parent, child_name, new_norm)
                    count += 1
        self._rmsnorm_patches = count

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(
        self,
        image: np.ndarray | list[np.ndarray] | None = None,
        instruction: str = "",
        state: list[float] | np.ndarray | None = None,
        noise: np.ndarray | None = None,
        rtc: Any = None,
    ) -> dict[str, Any]:
        """Run inference via real SmolVLAPolicy.

        ``rtc`` (``tether.runtime.rtc_guidance.RtcGuidance``) routes LeRobot's
        RTC kwargs into its RTC-aware Euler loop.

        Returns the same dict shape as TetherServer.predict so the FastAPI
        `/act` endpoint works unchanged.
        """
        import torch

        if not self._ready:
            return {"error": "Model not loaded. Call load() first."}
        if self._action_guard is not None and self._action_guard.tripped:
            return {
                "error": "guard_tripped",
                "reason": self._action_guard.trip_reason,
                "hint": "Investigate upstream (inputs / sensors / model) and "
                        "call POST /guard/reset to resume.",
            }

        start = time.perf_counter()

        # 1) Build raw batch in the schema lerobot's preprocessor expects.
        # 2026-04-19 lerobot-conformance rewrite:
        #  - Use 2 cameras with keys observation.images.image + image2 (matches
        #    HuggingFaceVLA/smolvla_libero's canonical config; the community
        #    lerobot/smolvla_libero config has a rename_map image->camera1,
        #    image2->camera2 that handles this automatically).
        #  - Apply 180-degree H+W flip per lerobot's LiberoProcessorStep
        #    ("HuggingFaceVLA/libero camera orientation convention"). Missing
        #    this flip alone can drop LIBERO success to near-zero.
        #  - Do NOT replicate to a phantom 3rd camera — training only saw 2.
        # Opt out: set TETHER_LIBERO_LEGACY=1 to restore the camera1/2/3
        # replicated + unflipped pattern (for comparison / decomposed path).
        images_list = (
            image if isinstance(image, list)
            else [image] if image is not None
            else []
        )
        legacy_mode = os.environ.get("TETHER_LIBERO_LEGACY", "0") == "1"
        if legacy_mode:
            # Old behavior: replicate to 3 cameras, no flip, camera1/2/3 keys.
            while len(images_list) < 3:
                images_list.append(
                    images_list[-1] if images_list
                    else np.zeros((256, 256, 3), np.uint8)
                )
            images_list = images_list[:3]
            key_prefix_pairs = [
                ("observation.images.camera1", 0),
                ("observation.images.camera2", 1),
                ("observation.images.camera3", 2),
            ]
        else:
            # New canonical behavior: 2 cameras, flipped H+W, image/image2 keys.
            if len(images_list) == 0:
                images_list = [np.zeros((256, 256, 3), np.uint8)] * 2
            elif len(images_list) == 1:
                images_list = [images_list[0], images_list[0]]
            images_list = images_list[:2]
            key_prefix_pairs = [
                ("observation.images.image", 0),
                ("observation.images.image2", 1),
            ]

        batch: dict[str, Any] = {}
        for key, idx in key_prefix_pairs:
            img = images_list[idx]
            img_np = np.asarray(img)
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            t = (
                torch.from_numpy(img_np)
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
                .unsqueeze(0)
            )
            # 180° H+W flip for LIBERO camera orientation. Skip in legacy mode
            # OR when TETHER_LIBERO_NO_FLIP=1 (A/B test for whether vla-eval
            # already applied the flip upstream — if it did, we double-flip).
            no_flip = os.environ.get("TETHER_LIBERO_NO_FLIP", "0") == "1"
            if not legacy_mode and not no_flip:
                t = torch.flip(t, dims=[2, 3])
            batch[key] = t

        # State: pad/truncate to 8D (LIBERO training dim)
        if state is None:
            state_np = np.zeros(8, dtype=np.float32)
        else:
            state_np = np.asarray(state, dtype=np.float32)
            if state_np.shape[-1] < 8:
                pad = np.zeros(8 - state_np.shape[-1], dtype=np.float32)
                state_np = np.concatenate([state_np, pad])
            elif state_np.shape[-1] > 8:
                state_np = state_np[:8]
        batch["observation.state"] = torch.from_numpy(state_np).unsqueeze(0)
        batch["task"] = [instruction or ""]

        # 2) Preprocess (tokenize + normalize)
        batch_pp = self.preprocessor(batch)
        batch_pp = {
            k: (v.to(self._device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch_pp.items()
        }
        # One-shot diagnostic — what did we feed the preprocessor vs what did
        # we get back? If our batch keys don't match the preprocessor's
        # expected feature schema, tensors silently pass through as dummy
        # zeros and the model operates with no visual/state conditioning.
        # This is the likely root cause of the 2026-04-19 LIBERO-10 0%
        # result — we pass observation.images.camera{1,2,3} but lerobot's
        # SmolVLA LIBERO preprocessor probably expects something else.
        if not getattr(self, "_batch_logged", False):
            self._batch_logged = True
            batch_keys = sorted(batch.keys())
            batch_pp_keys = sorted(batch_pp.keys())
            logger.info(
                "First predict batch keys (before preprocessor): %s", batch_keys
            )
            logger.info(
                "First predict batch_pp keys (after preprocessor): %s", batch_pp_keys
            )
            # Look for image tensors and dump their norms — if norms are
            # suspiciously uniform or zero, images were dropped.
            for k, v in batch_pp.items():
                if "image" in k.lower() and hasattr(v, "shape"):
                    v_np = v.detach().cpu().numpy() if hasattr(v, "detach") else v
                    logger.info(
                        "  batch_pp[%s]: shape=%s  min=%.3f  max=%.3f  mean=%.3f",
                        k, v.shape,
                        float(v_np.min()) if v_np.size else -1,
                        float(v_np.max()) if v_np.size else -1,
                        float(v_np.mean()) if v_np.size else -1,
                    )

        # 3) Run policy
        noise_t = None
        if noise is not None:
            noise_t = torch.as_tensor(noise, dtype=torch.float32, device=self._device)
            if noise_t.ndim == 2:
                noise_t = noise_t.unsqueeze(0)

        rtc_kwargs: dict[str, Any] = {}
        if rtc is not None:
            self._ensure_rtc_processor(rtc)
            left = rtc.prev_chunk_left_over
            rtc_kwargs = {
                "inference_delay": int(rtc.inference_delay),
                "execution_horizon": int(rtc.execution_horizon),
                "prev_chunk_left_over": (
                    None if left is None
                    else torch.as_tensor(left, dtype=torch.float32, device=self._device)
                ),
            }
        with torch.no_grad():
            actions = self.policy.predict_action_chunk(
                batch_pp, noise=noise_t, **rtc_kwargs
            )  # [1, chunk, action_dim_trimmed]

        # 4) Unnormalize via postprocessor
        post = self.postprocessor(actions.detach().cpu())
        actions_unnorm = (
            post.detach().cpu().numpy() if hasattr(post, "detach") else np.asarray(post)
        )
        if actions_unnorm.ndim == 3:
            actions_unnorm = actions_unnorm[0]  # -> [chunk, action_dim]

        # 5) Safety guard (optional) — same interface as TetherServer
        safety_violations = 0
        guard_detail: list[str] = []
        guard_hold = False
        if self._action_guard is not None:
            try:
                safe, guard_results = self._action_guard.check(actions_unnorm)
            except Exception as e:  # noqa: BLE001 — GuardStop/GuardUnitsError/bug: fail closed
                from tether.safety import guard_stop_response
                logger.error("tether guard refused chunk, no actions returned: %s", e)
                return guard_stop_response(e)
            actions_unnorm = safe
            guard_hold = any(r.hold for r in guard_results)
            safety_violations = sum(len(r.violations) for r in guard_results)
            if safety_violations:
                guard_detail = [
                    f"action {i}: {len(r.violations)} violations"
                    for i, r in enumerate(guard_results[:3]) if r.violations
                ]

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 6) Deadline enforcement (optional)
        deadline_exceeded = False
        if self._deadline_ms is not None and elapsed_ms > self._deadline_ms:
            deadline_exceeded = True
            self._deadline_misses += 1
            if self._last_good_actions is not None:
                actions_unnorm = self._last_good_actions
                logger.warning(
                    "deadline miss (%d): %.1fms > %.1fms — reusing last action",
                    self._deadline_misses, elapsed_ms, self._deadline_ms,
                )
            else:
                # Never synthesize zeros: on a position-controlled arm that
                # commands the zero pose.
                return {
                    "error": "deadline_exceeded_no_safe_action",
                    "reason": (
                        f"inference took {elapsed_ms:.1f}ms > deadline "
                        f"{self._deadline_ms:.1f}ms and no prior action exists"
                    ),
                    "hint": "No action was produced; do not move the robot.",
                    "deadline_exceeded": True,
                }

        if not deadline_exceeded:
            self._last_good_actions = actions_unnorm.copy()

        result: dict[str, Any] = {
            "actions": actions_unnorm.tolist(),
            "num_actions": len(actions_unnorm),
            "action_dim": int(actions_unnorm.shape[-1]) if actions_unnorm.ndim >= 1 else 0,
            "latency_ms": round(elapsed_ms, 1),
            "hz": round(1000.0 / elapsed_ms, 1) if elapsed_ms > 0 else 0,
            "denoising_steps": self.num_denoising_steps,
            "inference_mode": self._inference_mode,
            "vlm_conditioning": "real",
        }
        if self._action_guard is not None:
            result["safety_violations"] = safety_violations
            if guard_hold:
                result["guard_hold"] = True
            if guard_detail:
                result["safety_detail"] = guard_detail
        if self._deadline_ms is not None:
            result["deadline_exceeded"] = deadline_exceeded
            if self._deadline_misses:
                result["deadline_misses_total"] = self._deadline_misses
        return result

    def predict_from_base64(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
        rtc: Any = None,
    ) -> dict[str, Any]:
        """Base64 entry point matching TetherServer for FastAPI reuse."""
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
        return self.predict(image=image, instruction=instruction, state=state, rtc=rtc)

    # Batching stubs (no batching in native mode; predict is sync)

    async def start_batch_worker(self) -> None:
        return None

    async def stop_batch_worker(self) -> None:
        return None

    async def predict_async(
        self,
        image: np.ndarray | None = None,
        instruction: str = "",
        state: list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        return self.predict(image=image, instruction=instruction, state=state)

    async def predict_from_base64_async(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
        rtc: Any = None,
    ) -> dict[str, Any]:
        return self.predict_from_base64(
            image_b64=image_b64, instruction=instruction, state=state, rtc=rtc
        )

    async def run_batch(self, requests: list) -> list[dict[str, Any]]:
        """PolicyRuntime entry point. Sequential dispatch — native PyTorch
        path doesn't batch today. Per chunk-budget-batching ADR 2026-04-24."""
        results: list[dict[str, Any]] = []
        for req in requests:
            res = await self.predict_from_base64_async(
                image_b64=getattr(req, "image", None),
                instruction=getattr(req, "instruction", "") or "",
                state=getattr(req, "state", None),
                rtc=getattr(req, "rtc", None),
            )
            results.append(res)
        return results
