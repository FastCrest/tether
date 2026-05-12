"""Pi05DecomposedServer -- wraps Pi05DecomposedInference behind the reflex
serve interface.

Per ADR 2026-04-25-decomposed-dispatch-via-reflex-serve: closes the
B.4/B.5 measurement gap. ReflexServer's create_app router falls through
to the legacy ReflexServer for `export_kind=decomposed` exports, which
looks for `expert_stack.onnx` and fails with "No ONNX model found".

This wrapper lets the router add a 4th branch:
    if reflex_config.export_kind == "decomposed":
        server = Pi05DecomposedServer(...)

The class exposes the ReflexServer interface (predict_from_base64_async,
run_batch, _action_guard, etc.) so all the FastAPI plumbing in
create_app composes without per-wedge updates.

Scope per the ADR:
- Phase 1 substrate: this file (load + dispatch + minimal prep pipeline)
- Modal validation: filed as follow-up experiments (see ADR §"Validation
  experiments to file")
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Default camera resolution for pi05 (matches the export's expected input).
# Customers running a non-default export should set this via the export's
# reflex_config.json `camera_resolution` field; we read it at load time.
DEFAULT_CAMERA_RESOLUTION = 224

# Default chunk size + max action dim for pi05 (matches the export). Read
# from reflex_config.json at load time; these are the conservative defaults
# when the config doesn't carry them.
DEFAULT_CHUNK_SIZE = 50
DEFAULT_MAX_ACTION_DIM = 32

# Lang sequence length default. The decomposed export traces with a fixed
# seq length (typically 16 or 48); the exact value comes from the ONNX
# session's lang_tokens input shape.
DEFAULT_LANG_SEQ_LEN = 16


class Pi05DecomposedServer:
    """Server-interface wrapper around Pi05DecomposedInference.

    Mirrors the ReflexServer attributes that create_app + the /act handler
    + the wedges (action_guard, RTC, A2C2, record-replay, prometheus,
    chunk-budget-batching, policy-versioning) read.

    Lifecycle:
        server = Pi05DecomposedServer(export_dir, ...)
        server.load()  # builds the underlying Pi05DecomposedInference
        result = await server.predict_from_base64_async(
            image_b64=..., instruction="pick up the cup", state=[...],
        )
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
        action_similarity_threshold: float = 0.0,
        max_similar_skips: int = 3,
    ):
        self.export_dir = Path(export_dir)
        self.device = device
        self._requested_device = device
        self._requested_providers = providers
        self._strict_providers = strict_providers
        self._safety_config_path = (
            Path(safety_config) if safety_config else None
        )
        self._adaptive_steps = adaptive_steps
        self._cloud_fallback_url = cloud_fallback_url
        self._deadline_ms = deadline_ms
        self._max_batch = max(1, max_batch)
        self._batch_timeout_s = max(0.0, batch_timeout_ms) / 1000.0
        self._action_similarity_threshold = float(action_similarity_threshold)
        self._max_similar_skips = int(max_similar_skips)

        # ReflexServer-interface mirror attributes (populated during load
        # OR after lifespan wedge composition).
        self._ready = False
        self._inference: Any = None  # Pi05DecomposedInference
        self._tokenizer: Any = None  # HF tokenizer
        self._action_guard: Any = None
        # A2C2 hook applied INTERNALLY in _predict (between inference output
        # and action denormalization), so the head sees the same normalized
        # action distribution it was trained on. Set via set_a2c2_hook() from
        # create_app at lifespan time. None = no A2C2 (default).
        self._a2c2_hook: Any = None
        # BID (Bidirectional Decoding) chunk-selection config — alternative to
        # A2C2 head. Set via set_bid_config() from create_app at lifespan time.
        # None = no BID (default; single-sample inference path). Mutually
        # exclusive with _a2c2_hook in Phase 1; create_app validates the
        # mutex at startup.
        self._bid_config: Any = None
        # Per-server previous-chunk buffer for BID coherence scoring. Holds
        # the last emitted (post-denorm? no — pre-denorm) chunk so the next
        # call can score candidates against it. Reset to None on cold-start.
        self._bid_previous_chunk: np.ndarray | None = None
        self._split_orchestrator = None
        self._last_good_actions: np.ndarray | None = None
        self._deadline_misses = 0
        self._batches_run = 0
        self._batched_requests = 0
        self._latency_history: deque[float] = deque(maxlen=1024)
        self._model_hash: str | None = None
        self._config_hash: str | None = None

        # Normalizer stats — applied to state INPUT (before inference) +
        # action OUTPUT (after inference). Loaded in load() from the export
        # dir's policy_(pre|post)processor_*.safetensors OR fetched from
        # the teacher HF repo as fallback.
        # Without these, the env executes raw normalized actions and the
        # robot flails — caught 2026-04-26 LIBERO N=50 = 0% across all
        # tasks before this fix landed.
        self._action_mean: np.ndarray | None = None
        self._action_std: np.ndarray | None = None
        self._state_mean: np.ndarray | None = None
        self._state_std: np.ndarray | None = None

        # Loaded from reflex_config.json at .load() time.
        self.config: dict[str, Any] = {}
        self.action_dim: int = 7  # set from config.action_dim during load
        self.chunk_size: int = DEFAULT_CHUNK_SIZE
        self.max_action_dim: int = DEFAULT_MAX_ACTION_DIM
        self.camera_resolution: int = DEFAULT_CAMERA_RESOLUTION
        self.lang_seq_len: int = DEFAULT_LANG_SEQ_LEN

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def _inference_mode(self) -> str:
        """Mirrors ReflexServer._inference_mode for /act response building.
        Decomposed exports always use ORT; expose the actual provider."""
        if self._inference is None:
            return "uninitialized"
        # Pi05DecomposedInference doesn't expose providers directly; report
        # the requested one for now.
        if self._requested_device == "cuda":
            return "onnx_cuda_decomposed"
        return "onnx_cpu_decomposed"

    def _load_config(self) -> dict[str, Any]:
        """Read reflex_config.json from the export dir."""
        config_path = self.export_dir / "reflex_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"reflex_config.json not found in {self.export_dir}. "
                f"Decomposed exports must include the config sibling."
            )
        return json.loads(config_path.read_text())

    def load(self) -> None:
        """Load the decomposed inference + tokenizer + reflex_config."""
        # Lazy import keeps the heavy onnxruntime-gpu import out of module
        # scope (so importing this file doesn't pull torch/ort).
        from reflex.runtime.pi05_decomposed_server import (
            Pi05DecomposedInference,
        )

        self.config = self._load_config()

        # Pull canonical fields from config; validate export_kind.
        export_kind = self.config.get("export_kind", "")
        if export_kind != "decomposed":
            raise ValueError(
                f"Pi05DecomposedServer requires export_kind='decomposed', "
                f"got {export_kind!r}. Use the matching server class for "
                f"this export type (monolithic -> Pi0OnnxServer / "
                f"SmolVLAOnnxServer; legacy decomposed -> ReflexServer)."
            )

        self.action_dim = int(self.config.get("action_dim", 7))
        self.chunk_size = int(
            self.config.get("chunk_size") or self.config.get("action_chunk_size", DEFAULT_CHUNK_SIZE)
        )
        # max_action_dim is the export's padded action dim (usually 32 for pi05).
        # Read from `decomposed.max_action_dim` if present; default to 32.
        decomposed_block = self.config.get("decomposed", {})
        self.max_action_dim = int(
            decomposed_block.get("max_action_dim", DEFAULT_MAX_ACTION_DIM)
        )

        # Build providers list.
        providers = self._requested_providers or (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._requested_device == "cuda"
            else ["CPUExecutionProvider"]
        )

        logger.info(
            "Pi05DecomposedServer loading from %s (action_dim=%d, "
            "chunk_size=%d, max_action_dim=%d, providers=%s)",
            self.export_dir, self.action_dim, self.chunk_size,
            self.max_action_dim, providers,
        )

        # Instantiate the inference primitive. Default cache config keeps
        # the production "episode" cache off until callers explicitly pass
        # episode_id; falls back to the safe "prefix" mode.
        self._inference = Pi05DecomposedInference(
            export_dir=self.export_dir,
            providers=providers,
            enable_cache=True,
            cache_level="prefix",  # safe default; episode mode requires per-call episode_id
            action_similarity_threshold=self._action_similarity_threshold,
            max_similar_skips=self._max_similar_skips,
        )

        # Probe the lang_tokens input shape from the ONNX session to
        # determine the expected sequence length for tokenization.
        # Pi05DecomposedInference exposes both `_sess_prefix` (vlm_prefix.onnx)
        # and `_sess_expert` (expert_denoise.onnx). lang_tokens is an input to
        # the prefix session; the expert may also have it but with a different
        # shape. Probe both to find the right value (caught by 2026-04-25
        # b4 gate v5: 16 default failed against an export expecting 200).
        try:
            for sess_attr in ("_sess_prefix", "_sess_expert", "_expert_session"):
                sess = getattr(self._inference, sess_attr, None)
                if sess is None:
                    continue
                for inp in sess.get_inputs():
                    if inp.name == "lang_tokens":
                        shape = inp.shape
                        if len(shape) >= 2 and isinstance(shape[1], int):
                            self.lang_seq_len = int(shape[1])
                            logger.info(
                                "Pi05DecomposedServer lang_seq_len=%d "
                                "(probed from %s)",
                                self.lang_seq_len, sess_attr,
                            )
                            break
                if self.lang_seq_len != DEFAULT_LANG_SEQ_LEN:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "lang_seq_len probe failed (%s); using default %d",
                exc, self.lang_seq_len,
            )

        # Try to load the HF tokenizer from the export. For decomposed
        # pi05 exports the tokenizer config typically lives alongside the
        # ONNX. Lazy-loaded on first /act call to avoid import cost when
        # serving without text instructions.
        self._tokenizer = None  # populated lazily by _ensure_tokenizer

        # Load normalizer stats: state-in normalization + action-out
        # denormalization. Without these the env receives raw normalized
        # actions and the robot flails (LIBERO success-rate ~0%; caught
        # 2026-04-26 N=50 fire before fix shipped).
        self._load_normalizer_stats()

        # Apply ActionGuard if a safety config was provided.
        if self._safety_config_path is not None:
            try:
                from reflex.safety import ActionGuard, SafetyLimits
                limits = SafetyLimits.from_json(self._safety_config_path)
                self._action_guard = ActionGuard(limits=limits, mode="clamp")
                logger.info(
                    "Pi05DecomposedServer ActionGuard loaded: %d joints, mode=clamp",
                    len(limits.joint_names),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Pi05DecomposedServer safety load failed: %s", exc)

        self._ready = True
        logger.info("Pi05DecomposedServer ready")

    def _load_normalizer_stats(self) -> None:
        """Load state/action MEAN_STD normalizer stats. Needed because the
        decomposed model emits NORMALIZED actions; the env expects denormalized.

        Lookup order:
        1. Local: policy_(pre|post)processor_*.safetensors in self.export_dir
        2. HF fallback: snapshot_download from teacher repo (config.model_id
           or distill_provenance.json::teacher_export). Distill exports usually
           don't ship normalizer files; teacher repo always does.

        On miss, leaves stats=None — callers fall back to identity transforms.
        Logs the stats-source so the customer-facing /health debug surface
        shows whether normalization is on.
        """
        from reflex.runtime.adapters.vla_eval import load_normalizer_stats

        # 1. Try local export dir
        stats = load_normalizer_stats(self.export_dir)
        source = "local-export" if stats else None

        # 2. HF teacher fallback if local was empty
        if not stats:
            teacher_ref = self._infer_teacher_ref()
            if teacher_ref:
                try:
                    from huggingface_hub import snapshot_download
                    teacher_path = snapshot_download(
                        teacher_ref,
                        allow_patterns=[
                            "policy_preprocessor*",
                            "policy_postprocessor*",
                        ],
                    )
                    stats = load_normalizer_stats(teacher_path)
                    if stats:
                        source = f"hf-teacher:{teacher_ref}"
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Pi05DecomposedServer normalizer fetch from teacher %r "
                        "failed: %s",
                        teacher_ref, exc,
                    )

        if "action_mean" in stats and "action_std" in stats:
            self._action_mean = stats["action_mean"].astype(np.float32)
            self._action_std = stats["action_std"].astype(np.float32)
        if "state_mean" in stats and "state_std" in stats:
            self._state_mean = stats["state_mean"].astype(np.float32)
            self._state_std = stats["state_std"].astype(np.float32)

        if self._action_mean is not None or self._state_mean is not None:
            logger.info(
                "Pi05DecomposedServer normalizer loaded from %s "
                "(action=%s, state=%s)",
                source,
                "on" if self._action_mean is not None else "off",
                "on" if self._state_mean is not None else "off",
            )
        else:
            logger.warning(
                "Pi05DecomposedServer normalizer NOT FOUND -- actions will be "
                "in normalized space, robot will flail. "
                "Looked in: %s + HF teacher fallback. "
                "Fix: re-export with policy_(pre|post)processor_*.safetensors "
                "alongside the ONNX, OR provide a teacher HF repo via "
                "config.model_id or distill_provenance.json::teacher_export.",
                self.export_dir,
            )

    def _infer_teacher_ref(self) -> str | None:
        """Infer the teacher HF repo id for normalizer fallback. Returns the
        first valid HF id from config.model_id or distill_provenance.json's
        teacher_export, or None if neither is available."""
        # 1. config.model_id when it looks like an HF repo id
        model_id = self.config.get("model_id", "")
        if (
            isinstance(model_id, str)
            and "/" in model_id
            and not model_id.startswith("/")
            and "::" not in model_id
        ):
            return model_id

        # 2. distill_provenance.json's teacher_export
        prov_path = self.export_dir / "distill_provenance.json"
        if prov_path.exists():
            try:
                prov = json.loads(prov_path.read_text())
                teacher = prov.get("teacher_export", "")
                if (
                    isinstance(teacher, str)
                    and "/" in teacher
                    and not teacher.startswith("/")
                ):
                    return teacher
            except Exception:  # noqa: BLE001
                pass

        # 3. Hardcoded fallback for pi05_libero distill exports (the most
        # common case at this stage). Safe because all Reflex pi05 distills
        # share the v044 teacher's normalizer stats.
        return "lerobot/pi05_libero_finetuned_v044"

    def _ensure_tokenizer(self) -> None:
        """Lazy-load the HF tokenizer from the export dir on first use.

        Fallback chain (per 2026-04-25 b4 gate v4 finding -- distill outputs
        don't ship a tokenizer + model_id is a local path):
        1. tokenizer.json sibling in the export dir
        2. config.model_id when it looks like an HF repo id (org/name)
        3. distill_provenance.json -> teacher_export when it's an HF id
        4. paligemma-3b-pt-224 (the canonical VLM tokenizer all pi05 students
           use; safe default since SnapFlow distill preserves the input
           tokenization contract from the teacher)
        """
        if self._tokenizer is not None:
            return

        from transformers import AutoTokenizer
        tried: list[str] = []
        last_exc: Exception | None = None

        candidates: list[str] = []

        # 1. Sibling tokenizer in the export
        if (self.export_dir / "tokenizer.json").exists() or (
            self.export_dir / "tokenizer_config.json"
        ).exists():
            candidates.append(str(self.export_dir))

        # 2. config.model_id if it looks like an HF repo id (org/name) and
        # NOT a local path
        model_id = self.config.get("model_id", "")
        if (
            isinstance(model_id, str)
            and "/" in model_id
            and not model_id.startswith("/")
            and "::" not in model_id
        ):
            candidates.append(model_id)

        # 3. distill_provenance.json's teacher_export, if present + HF id
        prov_path = self.export_dir / "distill_provenance.json"
        if prov_path.exists():
            try:
                prov = json.loads(prov_path.read_text())
                teacher = prov.get("teacher_export", "")
                if (
                    isinstance(teacher, str)
                    and "/" in teacher
                    and not teacher.startswith("/")
                ):
                    candidates.append(teacher)
            except Exception:  # noqa: BLE001
                pass

        # 4. Paligemma fallback -- the VLM tokenizer all pi05 students use.
        # Safe because SnapFlow distill preserves the input contract from
        # the teacher (which is paligemma-based for pi0/pi05).
        candidates.append("google/paligemma-3b-pt-224")

        for cand in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(cand)
                self._tokenizer = tok
                logger.info(
                    "Pi05DecomposedServer tokenizer loaded from %r "
                    "(vocab_size=%d)", cand, tok.vocab_size,
                )
                return
            except Exception as exc:  # noqa: BLE001
                tried.append(cand)
                last_exc = exc
                continue

        logger.error(
            "Pi05DecomposedServer tokenizer load failed; tried %d sources: "
            "%r. Last error: %s", len(tried), tried, last_exc,
        )
        raise RuntimeError(
            f"No tokenizer found for {self.export_dir}. Tried {len(tried)} "
            f"sources ({tried!r}). Pi05DecomposedServer needs a tokenizer "
            f"to encode the instruction. Last error: {last_exc}"
        )

    # -- Server-interface contract -----------------------------------------

    def predict_from_base64(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
        image_wrist_b64: str | None = None,
    ) -> dict[str, Any]:
        """Sync entry point. Mirrors ReflexServer.predict_from_base64."""
        # Decode primary image
        image: np.ndarray | None = self._decode_b64_image(image_b64)
        if isinstance(image, dict):
            return image  # error envelope from decoder

        # Decode optional wrist image (multi-camera VLAs like pi05)
        image_wrist: np.ndarray | None = None
        if image_wrist_b64:
            image_wrist = self._decode_b64_image(image_wrist_b64)
            if isinstance(image_wrist, dict):
                return image_wrist

        return self._predict(
            image=image, instruction=instruction or "", state=state,
            image_wrist=image_wrist,
        )

    def _decode_b64_image(self, b64: str | None) -> "np.ndarray | dict[str, Any] | None":
        """Decode a base64 PNG/JPEG to HxWx3 uint8 ndarray. Returns None on
        empty input, an error envelope dict on decode failure."""
        if not b64:
            return None
        try:
            from PIL import Image
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return np.array(img)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Failed to decode image: {exc}"}

    async def predict_from_base64_async(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
        image_wrist_b64: str | None = None,
    ) -> dict[str, Any]:
        """Async entry point matching ReflexServer's contract."""
        # Pi05DecomposedInference is sync-only; offload to thread to avoid
        # blocking the FastAPI event loop.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.predict_from_base64(
                image_b64=image_b64, instruction=instruction,
                state=state, image_wrist_b64=image_wrist_b64,
            ),
        )

    async def run_batch(self, requests: list) -> list[dict[str, Any]]:
        """PolicyRuntime's run_batch_callback. Sequential dispatch (no
        true batching for decomposed; future dynamic-shape exporter
        enables N>1)."""
        results: list[dict[str, Any]] = []
        for req in requests:
            res = await self.predict_from_base64_async(
                image_b64=getattr(req, "image", None),
                instruction=getattr(req, "instruction", "") or "",
                state=getattr(req, "state", None),
                image_wrist_b64=getattr(req, "image_wrist", None),
            )
            results.append(res)
        return results

    # -- Internal: prep + inference + postprocess --------------------------

    def _predict(
        self,
        *,
        image: np.ndarray | None,
        instruction: str,
        state: list[float] | None,
        image_wrist: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Pure-numpy prep + inference + postprocess. Catches + envelopes
        any exception in the standard error dict (so /act returns a clean
        envelope; circuit breaker fires on the dict-with-error key)."""
        import time
        if not self._ready or self._inference is None:
            return {"error": "server not ready -- load() not called"}

        t0 = time.perf_counter()
        try:
            # 1. Image prep. pi05 was trained on 2 real cameras (base/agentview
            # + wrist/eye-in-hand). When the customer provides image_wrist, use
            # it as cam2; otherwise fall back to the -1 padding (model handles
            # missing camera via mask=0). Caught 2026-04-26: feeding 1 real
            # camera + padded cam2 produces OOD VLM prefix tokens -> 0% LIBERO.
            img_base, mask_base = self._prep_image(image)
            if image_wrist is not None:
                img_wrist_l, mask_wrist_l = self._prep_image(image_wrist)
            else:
                img_wrist_l = np.full_like(img_base, -1.0)
                mask_wrist_l = np.zeros_like(mask_base)
            # 3rd camera (wrist_r) stays padded -- pi05 was trained with
            # exactly 2 real cams + 1 explicit empty per the preprocessor JSON.
            img_pad = np.full_like(img_base, -1.0)
            mask_pad = np.zeros_like(mask_base)

            # 2. Tokenize instruction
            self._ensure_tokenizer()
            lang_tokens, lang_masks = self._tokenize(instruction)

            # 3. State pad
            state_arr = self._prep_state(state)

            # 4. Inference path — single-sample (default) or BID multi-sample.
            bid_telemetry: dict[str, Any] | None = None
            if self._bid_config is not None:
                # BID: sample N candidate chunks (varying noise per call),
                # score each by backward coherence with the previously-emitted
                # chunk, pick the highest-scoring one. Per
                # 2026-04-29-a2c2-correction_research_revisit Lens 4 (BID is
                # the architectural pivot after Phases 1-3 of head-correction
                # hit the magnitude/derailment ceiling).
                from reflex.correction.bid import predict_chunk_bid

                rng = np.random.default_rng()  # nondeterministic per call

                def _sample_one(_i: int) -> np.ndarray:
                    noise_i = rng.standard_normal(
                        (1, self.chunk_size, self.max_action_dim)
                    ).astype(np.float32)
                    out = self._inference.predict_action_chunk(
                        img_base=img_base, img_wrist_l=img_wrist_l, img_wrist_r=img_pad,
                        mask_base=mask_base, mask_wrist_l=mask_wrist_l, mask_wrist_r=mask_pad,
                        lang_tokens=lang_tokens, lang_masks=lang_masks,
                        noise=noise_i, state=state_arr,
                    )
                    if out.ndim == 3:
                        out = out[0]
                    return out

                chosen, bid_telemetry = predict_chunk_bid(
                    _sample_one,
                    previous_chunk=self._bid_previous_chunk,
                    config=self._bid_config,
                )
                # chosen is shape (chunk_size, max_action_dim); restore the
                # batch dim so downstream postprocess code sees same shape.
                actions_padded = chosen[np.newaxis, ...]
                # Save for next call's coherence scoring.
                self._bid_previous_chunk = chosen.copy()
            else:
                # Single-sample (default).
                noise = np.random.randn(
                    1, self.chunk_size, self.max_action_dim,
                ).astype(np.float32)
                actions_padded = self._inference.predict_action_chunk(
                    img_base=img_base, img_wrist_l=img_wrist_l, img_wrist_r=img_pad,
                    mask_base=mask_base, mask_wrist_l=mask_wrist_l, mask_wrist_r=mask_pad,
                    lang_tokens=lang_tokens, lang_masks=lang_masks,
                    noise=noise, state=state_arr,
                )

            # 5b. A2C2 hook -- applied in NORMALIZED action space (between
            # inference output and denorm step), because the head was trained
            # on stale-fresh gaps of normalized actions. Caught 2026-04-26:
            # applying the hook AFTER denorm corrupted actions because the
            # head's residuals are in normalized units (~[-1,1]) while
            # denormalized actions have very different magnitudes (~[0.5m]).
            a2c2_decision_meta: dict[str, Any] | None = None
            if self._a2c2_hook is not None:
                a2c2_decision_meta = self._apply_a2c2_normalized(actions_padded)

            # 6. Postprocess -- denormalize THEN slice to action_dim.
            # Denorm formula: action_real = action_norm * std + mean.
            # The normalizer applies to the FULL max_action_dim padded
            # vector (not to the sliced 7-D), since policy_postprocessor.json
            # was fit on padded actions during distill prep.
            if actions_padded.ndim == 3:
                actions_padded = actions_padded[0]
            if self._action_mean is not None and self._action_std is not None:
                # Stats are typically (action_dim,) for the real action dim.
                # Pad them to max_action_dim for the broadcast.
                a_mean = self._action_mean
                a_std = self._action_std
                ad = actions_padded.shape[-1]
                if a_mean.shape[0] < ad:
                    a_mean = np.concatenate([
                        a_mean, np.zeros(ad - a_mean.shape[0], dtype=np.float32)
                    ])
                    a_std = np.concatenate([
                        a_std, np.ones(ad - a_std.shape[0], dtype=np.float32)
                    ])
                elif a_mean.shape[0] > ad:
                    a_mean = a_mean[:ad]
                    a_std = a_std[:ad]
                actions_padded = actions_padded * a_std + a_mean
            actions_out = actions_padded[:, : self.action_dim].astype(np.float32)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._latency_history.append(elapsed_ms)

            result_dict: dict[str, Any] = {
                "actions": actions_out.tolist(),
                "num_actions": int(actions_out.shape[0]),
                "action_dim": self.action_dim,
                "latency_ms": elapsed_ms,
                "inference_mode": self._inference_mode,
            }
            if a2c2_decision_meta is not None:
                result_dict.update(a2c2_decision_meta)
            if bid_telemetry is not None:
                # Lift BID telemetry into the response with prefixed keys so
                # downstream JSONL recorders can grep + plot per-call.
                result_dict["bid_selected_idx"] = bid_telemetry["selected_idx"]
                result_dict["bid_n_candidates"] = bid_telemetry["n_candidates"]
                # Don't include full scores list per-call — too verbose.
                # Operators can pull from logs if needed.
                if bid_telemetry["scores"]:
                    result_dict["bid_top_score"] = float(max(bid_telemetry["scores"]))
                    result_dict["bid_score_spread"] = float(
                        max(bid_telemetry["scores"]) - min(bid_telemetry["scores"])
                    )
            return result_dict
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pi05DecomposedServer.predict failed")
            return {"error": f"{type(exc).__name__}: {exc}"}

    def set_a2c2_hook(self, hook: Any) -> None:
        """Bind an A2C2Hook instance to be applied internally during _predict
        (in normalized action space, before denorm). Called from create_app at
        lifespan time after the hook is loaded.

        Returns nothing. Setting None disables internal a2c2 application.
        """
        self._a2c2_hook = hook
        if hook is not None:
            logger.info(
                "Pi05DecomposedServer A2C2 hook bound INTERNALLY (applied in "
                "normalized action space before denorm). action_dim=%d, "
                "obs_dim=%d", hook.head.config.action_dim, hook.head.config.obs_dim,
            )

    def set_bid_config(self, config: Any) -> None:
        """Bind a BIDConfig to use Bidirectional Decoding for chunk selection
        during _predict. Per arxiv 2408.17355 + the 2026-04-29 a2c2-correction
        research-revisit Lens 4. Sample N candidates per /act, score by
        backward coherence with the previously-emitted chunk, pick best.

        Mutually exclusive with set_a2c2_hook (Phase 1). Setting None disables
        BID and reverts to the single-sample inference path.
        """
        self._bid_config = config
        # Reset previous_chunk on (re)config so we don't score against a stale
        # chunk from a different episode/embodiment.
        self._bid_previous_chunk = None
        if config is not None:
            if self._a2c2_hook is not None:
                logger.warning(
                    "BID config set while A2C2 hook is also bound — Phase 1 "
                    "treats these as mutually exclusive; BID will run, A2C2 "
                    "hook output will be ignored on this server."
                )
            logger.info(
                "Pi05DecomposedServer BID enabled: n_candidates=%d, "
                "coherence_window=%d, metric=%s",
                config.n_candidates, config.coherence_window, config.coherence_metric,
            )

    def reset_bid_state(self) -> None:
        """Reset the previous-chunk buffer (e.g., on episode boundary)."""
        self._bid_previous_chunk = None

    def _apply_a2c2_normalized(self, actions_padded: np.ndarray) -> dict[str, Any]:
        """Apply A2C2 hook on the chunk in NORMALIZED space.

        actions_padded shape on entry: (1, chunk_size, max_action_dim) OR
        (chunk_size, max_action_dim). Mutates the array in-place to write the
        corrected leading hook_dim slice back. Returns a dict of result
        metadata (a2c2_applied / a2c2_reason / a2c2_correction_magnitude) for
        merging into the /act response.

        The hook returns corrections in the SAME normalized space the head
        was trained on. Splice the corrected leading slice back into the
        full padded chunk; downstream denorm + slice-to-action_dim run
        unchanged.
        """
        try:
            # Strip the batch dim if present so the hook sees (chunk_size, ad).
            if actions_padded.ndim == 3:
                chunk = actions_padded[0]
                has_batch = True
            else:
                chunk = actions_padded
                has_batch = False
            hook_dim = self._a2c2_hook.head.config.action_dim
            actions_for_hook = chunk[:, :hook_dim].copy()
            corrected, decision, magnitude = (
                self._a2c2_hook.maybe_apply_to_chunk(actions=actions_for_hook)
            )
            if decision.apply:
                if has_batch:
                    actions_padded[0, :, :hook_dim] = corrected
                else:
                    actions_padded[:, :hook_dim] = corrected
            return {
                "a2c2_applied": bool(decision.apply),
                "a2c2_reason": str(decision.reason),
                "a2c2_correction_magnitude": round(float(magnitude), 6),
            }
        except Exception as exc:  # noqa: BLE001 -- A2C2 must never break /act
            logger.warning("Pi05DecomposedServer.a2c2_apply_failed: %s", exc)
            return {
                "a2c2_applied": False,
                "a2c2_reason": f"error:{type(exc).__name__}",
                "a2c2_correction_magnitude": 0.0,
            }

    def _prep_image(
        self, image: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize/pad an HxWx3 uint8 image to (1, 3, R, R) float32 [0, 1] +
        a (1,) bool mask. When image is None, returns a -1 padding image +
        zero mask (signals "missing camera" to the model)."""
        R = self.camera_resolution
        if image is None:
            img = np.full((1, 3, R, R), -1.0, dtype=np.float32)
            mask = np.zeros((1,), dtype=bool)
            return img, mask

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must be HxWx3 uint8, got shape {image.shape}"
            )

        # Resize-with-pad to RxR (preserves aspect; pads short side).
        from PIL import Image
        h, w = image.shape[:2]
        if h > w:
            pad = (h - w) // 2
            image = np.pad(
                image, [(0, 0), (pad, h - w - pad), (0, 0)], mode="constant",
            )
        elif w > h:
            pad = (w - h) // 2
            image = np.pad(
                image, [(pad, w - h - pad), (0, 0), (0, 0)], mode="constant",
            )
        pil = Image.fromarray(image)
        pil = pil.resize((R, R), Image.BILINEAR)
        resized = np.asarray(pil)

        # CHW + add batch dim + normalize to [0, 1]
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        return chw[np.newaxis, ...], np.ones((1,), dtype=bool)

    def _tokenize(self, instruction: str) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize the instruction to lang_seq_len; returns (tokens, masks).

        For pi05 state-out exports the model expects the prompt formatted as
        ``"Task: <instruction>;\\nAction: "`` (per Pi05PrepareTokenizerStateOutStep
        in src/reflex/distill/pi05_state_out_processor.py). Without this
        wrapper the lang_tokens distribution diverges from training and the
        model emits garbage actions even with correct state-norm + action-denorm.
        Caught 2026-04-26 LIBERO N=50 = 0% with denorm-only fix.

        For non-state-out exports, the same format is harmless (the teacher
        uses the same Task/Action wrapper; state-out's only difference is
        omitting ", State: ..." from the prompt — already absent here since
        we don't append state to lang).

        Pads with 0 tokens + zero mask when the formatted prompt is shorter
        than lang_seq_len; truncates when longer.
        """
        cleaned = (instruction or "").strip().replace("_", " ").replace("\n", " ")
        prompt = f"Task: {cleaned};\nAction: "
        encoded = self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self.lang_seq_len,
            truncation=True,
            return_tensors="np",
        )
        tokens = encoded["input_ids"].astype(np.int64)
        masks = encoded["attention_mask"].astype(bool)
        return tokens, masks

    def _prep_state(self, state: list[float] | None) -> np.ndarray:
        """Normalize + pad the state vector to max_action_dim (typically 32 for pi05).

        Apply order: normalize the REAL-DIM portion (state - state_mean) /
        (state_std + eps), then zero-pad up to max_action_dim. The normalizer
        was fit on the real state dim (7 for LIBERO franka, etc.); padding
        with zeros AFTER normalization keeps zeros zero-mean, which the
        model treats as "no signal" for the unused DOFs.

        Returns (1, max_action_dim) float32.
        """
        if state is None:
            return np.zeros((1, self.max_action_dim), dtype=np.float32)
        arr = np.asarray(state, dtype=np.float32).flatten()

        # Normalize the leading real-dim portion. Without this the model
        # sees raw eef coordinates (e.g., 0.5 m) instead of zero-mean
        # standardized inputs and predicts garbage actions.
        if self._state_mean is not None and self._state_std is not None:
            n = min(arr.shape[0], self._state_mean.shape[0])
            arr_norm = arr.copy()
            arr_norm[:n] = (arr[:n] - self._state_mean[:n]) / (self._state_std[:n] + 1e-8)
            arr = arr_norm

        if arr.shape[0] > self.max_action_dim:
            arr = arr[: self.max_action_dim]
        elif arr.shape[0] < self.max_action_dim:
            arr = np.concatenate([
                arr, np.zeros(self.max_action_dim - arr.shape[0], dtype=np.float32),
            ])
        return arr[np.newaxis, ...]


__all__ = [
    "DEFAULT_CAMERA_RESOLUTION",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_LANG_SEQ_LEN",
    "DEFAULT_MAX_ACTION_DIM",
    "Pi05DecomposedServer",
]
