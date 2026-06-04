"""VLM prefix orchestrator: chains the 4-file ONNX pipeline.

Loads vision_encoder.onnx, text_embedder.onnx, and decoder_prefill.onnx
(all on CPUExecutionProvider), caches AutoTokenizer + AutoProcessor at
init, and provides a single ``run()`` method that takes an image, instruction,
and robot state and returns a prefix_kv tensor suitable for expert
cross-attention.

Pipeline:
    image  -> AutoProcessor -> vision_encoder.onnx  -> image_embeds  [1, 64, 960]
    text   -> AutoTokenizer -> text_embedder.onnx   -> text_embeds   [1, T, 960]
    state  -> pad + linear  -> StateEncoder          -> state_embed   [1, 1, 960]
    concat -> assemble_prefix -> prefix_embeds [1, 64+T+1, 960]
    prefix_embeds -> decoder_prefill.onnx -> prefix_kv [1, seq, 960]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Architecture constants (SmolVLA / SmolVLM2-500M)
HIDDEN_SIZE = 960
MAX_STATE_DIM = 32


class VLMPrefixOrchestrator:
    """Loads and chains the 4-file VLM ONNX pipeline."""

    def __init__(self, export_dir: Path, config: dict[str, Any]):
        self.export_dir = Path(export_dir)
        self.config = config
        # State encoder + image embeds + text embeds all live at VLM-internal
        # hidden dim (960 for SmolLM2). The separate vlm_kv_dim (320) is the
        # POST-projection dim that the expert's cross-attn expects — produced
        # by decoder_prefill.onnx via its baked-in k_proj layer.
        self._hidden_size = config.get(
            "vlm_hidden_size",
            config.get("vlm_kv_dim", HIDDEN_SIZE),  # legacy fallback
        )
        self._image_size = config.get("vlm_image_size", [512, 512])

        # ONNX sessions (loaded lazily below)
        self._vision_session = None
        self._text_session = None
        self._prefill_session = None
        self._state_session = None  # reserved for future ONNX state encoder

        # Cached tokenizer / processor
        self._tokenizer = None
        self._processor = None

        # State encoder weights (inline linear, no ONNX needed for 32->960)
        self._state_weight = None  # [960, 32]
        self._state_bias = None  # [960]

        self._pipeline_complete = False

        self._load_sessions()
        self._load_tokenizer_and_processor()
        self._load_state_encoder()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _load_sessions(self) -> None:
        """Load ONNX sessions for each pipeline stage."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning(
                "onnxruntime not installed -- VLM orchestrator cannot load ONNX sessions"
            )
            return

        providers = ["CPUExecutionProvider"]

        # 1. Vision encoder (required)
        vision_path = self.export_dir / "vision_encoder.onnx"
        if vision_path.exists():
            self._vision_session = ort.InferenceSession(
                str(vision_path), providers=providers
            )
            logger.info("Loaded vision_encoder.onnx (CPU)")
        else:
            logger.warning(
                "vision_encoder.onnx not found at %s -- VLM pipeline incomplete",
                vision_path,
            )

        # 2. Text embedder (optional -- can fall back to ordinal encoding)
        text_path = self.export_dir / "text_embedder.onnx"
        if text_path.exists():
            self._text_session = ort.InferenceSession(
                str(text_path), providers=providers
            )
            logger.info("Loaded text_embedder.onnx (CPU)")
        else:
            logger.info(
                "text_embedder.onnx not found -- will use fallback text embedding"
            )

        # 3. Decoder prefill (may not exist yet -- being built in parallel)
        prefill_path = self.export_dir / "decoder_prefill.onnx"
        if prefill_path.exists():
            self._prefill_session = ort.InferenceSession(
                str(prefill_path), providers=providers
            )
            logger.info("Loaded decoder_prefill.onnx (CPU)")
            self._pipeline_complete = True
        else:
            logger.warning(
                "decoder_prefill.onnx not found -- VLM pipeline incomplete. "
                "Will return assembled prefix embeddings without decoder pass."
            )

    def _load_tokenizer_and_processor(self) -> None:
        """Cache AutoTokenizer and AutoProcessor at init time."""
        # Use vlm_model_id (the base VLM) for tokenizer/processor, NOT model_id
        # (which is the SmolVLA policy checkpoint and doesn't have a tokenizer).
        model_id = self.config.get(
            "vlm_model_id", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            logger.info("Cached AutoTokenizer from %s", model_id)
        except Exception as e:
            logger.warning(
                "Failed to load AutoTokenizer from %s: %s -- "
                "will use fallback tokenization",
                model_id,
                e,
            )

        try:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Cached AutoProcessor from %s", model_id)
        except Exception as e:
            logger.info(
                "AutoProcessor not available from %s: %s -- "
                "will use manual image preprocessing",
                model_id,
                e,
            )

    def _load_state_encoder(self) -> None:
        """Load state projection weights from config or init random."""
        # Check for exported state encoder ONNX first
        state_onnx = self.export_dir / "state_encoder.onnx"
        if state_onnx.exists():
            try:
                import onnxruntime as ort

                self._state_session = ort.InferenceSession(
                    str(state_onnx), providers=["CPUExecutionProvider"]
                )
                logger.info("Loaded state_encoder.onnx (CPU)")
                return
            except Exception as e:
                logger.warning("Failed to load state_encoder.onnx: %s", e)

        self._state_session = None

        # Fall back to inline linear with random weights
        # Real weights would come from the SmolVLAPolicy checkpoint
        state_weight_path = self.export_dir / "state_proj_weight.npy"
        state_bias_path = self.export_dir / "state_proj_bias.npy"

        if state_weight_path.exists():
            self._state_weight = np.load(str(state_weight_path))
            logger.info("Loaded state_proj_weight.npy: %s", self._state_weight.shape)
        else:
            self._state_weight = np.random.randn(
                self._hidden_size, MAX_STATE_DIM
            ).astype(np.float32) * 0.02
            logger.info(
                "No state_proj_weight.npy found -- using random init [%d, %d]",
                self._hidden_size,
                MAX_STATE_DIM,
            )

        if state_bias_path.exists():
            self._state_bias = np.load(str(state_bias_path))
        else:
            self._state_bias = np.zeros(self._hidden_size, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True if at least the vision encoder is available."""
        return self._vision_session is not None

    @property
    def is_complete(self) -> bool:
        """True if the full pipeline (including decoder prefill) is available."""
        return self._pipeline_complete

    def run(
        self,
        image: np.ndarray | list[np.ndarray],
        instruction: str,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full VLM forward: image(s) + instruction + state → (vlm_k, vlm_v).

        ``image`` can be a single RGB array [H, W, 3] OR a list of such
        arrays. Each image is encoded separately by vision_encoder.onnx,
        producing [B, 64, hidden] per image; all results are concatenated
        along the sequence axis so the model sees the full multi-view
        prefix (64 * num_images image tokens). SmolVLA-LIBERO was trained on
        three cameras — passing only one is a distribution shift that
        correlates with task-success collapse.

        Both outputs shape ``[num_layers, B, seq, kv_dim]``. ``vlm_k`` has
        RoPE applied (matches real SmolVLA); ``vlm_v`` does not. Expert's
        cross-attention uses them as separate k-source and v-source per layer.

        When decoder_prefill.onnx is missing, returns zeros of the correct
        shape so the expert can still run with dummy conditioning.
        """
        from tether.runtime.vlm_components import assemble_prefix, pad_state

        # 1. Image(s) -> image_embeds [1, 64*N, 960]. Encode each camera
        # view separately; SmolVLA-LIBERO's decoder sees N*64 image tokens.
        if isinstance(image, list):
            images = image
        elif image is None:
            images = []
        else:
            images = [image]

        if not images:
            image_embeds = np.zeros(
                (1, 64, self._hidden_size), dtype=np.float32
            )
        else:
            per_view = [self._encode_image(img) for img in images]
            image_embeds = np.concatenate(per_view, axis=1)  # [1, 64*N, hidden]

        # 2. Instruction -> text_embeds [1, T, 960]
        text_embeds = self._encode_text(instruction)

        # 3. State -> state_embed [1, 1, 960]
        state_embed = self._encode_state(state)

        # 4. SCALE image + text embeds by sqrt(hidden_size). This matches
        # SmolVLA's embed_prefix (modeling_smolvla.py L663 and L690) which
        # does `emb * sqrt(emb_dim)`. Without this scaling our embeds are
        # ~31× smaller in magnitude than training, silently distorting every
        # downstream attention pattern — the subtle cause of our 0% LIBERO
        # task success despite each component individually matching PyTorch.
        scale = float(self._hidden_size) ** 0.5
        image_embeds = image_embeds * scale
        text_embeds = text_embeds * scale

        # 4. Assemble prefix [1, 64+T+1, 960]
        prefix_embeds, attention_mask = assemble_prefix(
            image_embeds, text_embeds, state_embed
        )

        # 5. Decoder prefill (if available) — two outputs: vlm_k (RoPE-applied)
        # and vlm_v, both shape [L, B, seq, kv_dim].
        if self._prefill_session is not None:
            try:
                k_out, v_out = self._prefill_session.run(
                    ["vlm_k", "vlm_v"],
                    {
                        "inputs_embeds": prefix_embeds.astype(np.float32),
                        "attention_mask": attention_mask.astype(np.int64),
                    },
                )
                return k_out, v_out
            except Exception as e:
                logger.warning(
                    "decoder_prefill inference failed: %s -- "
                    "falling back to zero-initialised per-layer K/V",
                    e,
                )

        # Fallback when decoder_prefill is missing/broken: return zeros shaped
        # [L, B, seq, kv_dim] so the expert's inputs have the right rank.
        num_layers = self.config.get("vlm_num_layers", 16)
        vlm_kv_dim = self.config.get(
            "vlm_kv_dim", self._hidden_size  # back-compat for v0.3 exports
        )
        batch, seq = prefix_embeds.shape[:2]
        zeros = np.zeros((num_layers, batch, seq, vlm_kv_dim), dtype=np.float32)
        return zeros, zeros.copy()

    def close(self) -> None:
        """Release ONNX session resources."""
        for attr in ("_vision_session", "_text_session", "_prefill_session", "_state_session"):
            session = getattr(self, attr, None)
            if session is not None:
                del session
                setattr(self, attr, None)
        self._pipeline_complete = False
        logger.debug("VLMPrefixOrchestrator sessions released")

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal pipeline stages
    # ------------------------------------------------------------------

    def _encode_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image and run vision encoder -> [1, 64, 960]."""
        if self._vision_session is None:
            # No vision encoder -- return zeros
            return np.zeros(
                (1, 64, self._hidden_size), dtype=np.float32
            )

        pixel_values = self._preprocess_image(image)

        image_embeds = self._vision_session.run(
            None, {"pixel_values": pixel_values}
        )[0]

        return image_embeds

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to pixel_values [1, 3, H, W] float32.

        Uses AutoProcessor if available, otherwise manual resize + normalize.
        """
        target_h, target_w = self._image_size[0], self._image_size[1]

        # Use our manual preprocess (bilinear resize + /255 → [B, 3, H, W])
        # rather than SmolVLM's AutoProcessor, because the latter does image
        # splitting: a single image comes out as [B, N_patches, 3, H, W]
        # (5D with N_patches typically >=1 depending on source resolution).
        # Our vision_encoder.onnx expects a plain 4D [B, 3, H, W] because
        # that's what we export. Feeding 5D causes ORT to reject the call
        # and the server silently falls back to dummy zero conditioning —
        # the root-cause bug hiding behind months of 0% LIBERO.
        return self._manual_preprocess(image, target_h, target_w)

    def _manual_preprocess(
        self, image: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        """Manual image preprocessing: resize + normalize -> [1, 3, H, W].

        Produces SigLIP-range pixel values [-1, +1]. Matches lerobot's
        ``SmolVLAPolicy.prepare_images`` which does ``img * 2 - 1`` after
        normalising to [0, 1]. SigLIP was trained on this range — feeding
        [0, 1] silently shifts the visual embedding distribution and
        degrades every downstream prefix/kv computation.
        """
        try:
            from PIL import Image as PILImage

            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            img = PILImage.fromarray(image).convert("RGB")
            img = img.resize((target_w, target_h), PILImage.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
        except ImportError:
            h, w = image.shape[:2]
            row_idx = (np.arange(target_h) * h // target_h).astype(int)
            col_idx = (np.arange(target_w) * w // target_w).astype(int)
            arr = image[np.ix_(row_idx, col_idx)].astype(np.float32) / 255.0

        # SigLIP range: [-1, 1]
        arr = arr * 2.0 - 1.0

        # HWC -> CHW, add batch dim
        arr = np.transpose(arr, (2, 0, 1))
        return arr[np.newaxis, ...]

    def _encode_text(self, instruction: str) -> np.ndarray:
        """Tokenize + embed instruction -> [1, T, 960]."""
        max_seq = 32

        if self._text_session is not None and self._tokenizer is not None:
            # Real path: tokenize then embed via ONNX
            encoded = self._tokenizer(
                instruction,
                max_length=max_seq,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            input_ids = encoded["input_ids"].astype(np.int64)
            text_embeds = self._text_session.run(
                None, {"input_ids": input_ids}
            )[0]
            return text_embeds

        if self._tokenizer is not None:
            # Have tokenizer but no text_embedder ONNX — produce a
            # deterministic embedding seeded by the token IDs so the same
            # instruction always maps to the same embedding.
            encoded = self._tokenizer(
                instruction,
                max_length=max_seq,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            input_ids = encoded["input_ids"].astype(np.int64)
            seq_len = input_ids.shape[1]
            seed = int(input_ids.sum()) % (2**31)
            rng = np.random.RandomState(seed)
            return rng.randn(1, seq_len, self._hidden_size).astype(
                np.float32
            ) * 0.02

        # Fallback: ordinal encoding → deterministic embedding seeded by text
        ids = [ord(c) % 50257 for c in instruction[:max_seq]]
        ids = ids + [0] * (max_seq - len(ids))
        seed = sum(ids) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(1, max_seq, self._hidden_size).astype(
            np.float32
        ) * 0.02

    def _encode_state(self, state: np.ndarray) -> np.ndarray:
        """Project robot state to VLM hidden space -> [1, 1, 960]."""
        from tether.runtime.vlm_components import pad_state

        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[np.newaxis, :]  # [1, D]

        # Pad to MAX_STATE_DIM
        state_padded = pad_state(state, MAX_STATE_DIM)  # [1, 32]

        # If we have a state_encoder ONNX session, use it
        if hasattr(self, "_state_session") and self._state_session is not None:
            state_embed = self._state_session.run(
                None, {"state": state_padded}
            )[0]
            return state_embed  # [1, 1, 960]

        # Inline linear: state_padded @ weight^T + bias -> [1, 960]
        projected = state_padded @ self._state_weight.T + self._state_bias
        return projected[:, np.newaxis, :]  # [1, 1, 960]
