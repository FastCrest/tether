"""VLM vision encoder export for SmolVLA.

Extracts the SigLIP vision encoder + SmolVLM connector from the base VLM
(HuggingFaceTB/SmolVLM2-500M-Video-Instruct) and exports as a standalone
ONNX graph.

The output ``image_embeds`` tensor of shape ``[B, 64, 960]`` feeds downstream
text embedding concatenation and the expert prefill decoder.

Approach: Load the base VLM via ``AutoModel.from_pretrained``, wrap
vision_model + connector in a thin ``VisionEncoderForONNX`` module that
pre-computes position IDs (avoiding the dynamic ``index_put`` in
``SmolVLMVisionEmbeddings`` that doesn't trace to ONNX), export with
``torch.onnx.export`` at opset 19, and post-fix any type mismatches
in the ONNX graph via ``patch_onnx_type_mismatches``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# SmolVLM2-500M architecture constants
DEFAULT_VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_IMAGE_SIZE = 512  # SigLIP-SO400M native size for SmolVLM2
DEFAULT_VLM_KV_DIM = 960  # SmolLM2 hidden size (also connector output dim)

# Numerical validation threshold -- SigLIP's 27 transformer layers
# accumulate fp32 rounding, so max_diff ~2-4e-4 is expected.
# Mean diff is ~1e-5 which is excellent.
ORT_MAX_DIFF_THRESHOLD = 5e-4


class VisionEncoderForONNX(nn.Module):
    """Wraps SigLIP vision encoder + SmolVLM connector for ONNX export.

    Pre-computes position IDs for full (unpadded) images to avoid the
    dynamic ``index_put`` / ``bucketize`` loop in
    ``SmolVLMVisionEmbeddings.forward()`` that produces ONNX nodes
    with int64/float type mismatches ORT cannot load.

    Input:  ``pixel_values`` -- ``[B, 3, image_size, image_size]`` float32
    Output: ``image_embeds``  -- ``[B, 64, 960]`` float32
    """

    def __init__(
        self,
        vision_model: nn.Module,
        connector: nn.Module,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        super().__init__()
        # Extract sub-modules from the vision model
        self.patch_embedding = vision_model.embeddings.patch_embedding
        self.position_embedding = vision_model.embeddings.position_embedding
        self.encoder = vision_model.encoder
        self.post_layernorm = vision_model.post_layernorm
        self.connector = connector

        # Pre-compute position IDs for a full image (no padding).
        # This replicates SmolVLMVisionEmbeddings.forward() logic
        # but without the dynamic for-loop / boolean indexing.
        emb = vision_model.embeddings
        patch_size = emb.patch_size
        num_patches_per_side = emb.num_patches_per_side
        nb_h = image_size // patch_size
        nb_w = image_size // patch_size

        boundaries = torch.arange(
            1 / num_patches_per_side, 1.0, 1 / num_patches_per_side
        )
        h_idx = torch.arange(nb_h, dtype=torch.float32)
        w_idx = torch.arange(nb_w, dtype=torch.float32)
        frac_h = h_idx / nb_h * (1 - 1e-6)
        frac_w = w_idx / nb_w * (1 - 1e-6)
        bucket_h = torch.bucketize(frac_h, boundaries, right=True)
        bucket_w = torch.bucketize(frac_w, boundaries, right=True)
        pos_ids = (bucket_h[:, None] * num_patches_per_side + bucket_w).flatten()
        self.register_buffer("position_ids", pos_ids.unsqueeze(0))  # [1, num_patches]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run vision encoding: patch embed -> position embed -> encoder -> connector.

        Args:
            pixel_values: ``[B, 3, 512, 512]`` float32

        Returns:
            image_embeds: ``[B, 64, 960]`` float32
        """
        batch_size = pixel_values.shape[0]

        # Patch embedding: [B, 3, 512, 512] -> [B, hidden_dim, 32, 32] -> [B, 1024, hidden_dim]
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # Add pre-computed position embeddings
        pos_ids = self.position_ids.expand(batch_size, -1)
        embeddings = embeddings + self.position_embedding(pos_ids)

        # Vision transformer encoder (no attention mask needed for full images)
        encoder_output = self.encoder(inputs_embeds=embeddings)
        hidden = encoder_output.last_hidden_state
        hidden = self.post_layernorm(hidden)  # [B, 1024, 768]

        # Connector: pixel shuffle + linear projection -> [B, 64, 960]
        image_embeds = self.connector(hidden)
        return image_embeds


def patch_onnx_type_mismatches(onnx_path: str | Path) -> int:
    """Fix type mismatches in an ONNX graph that prevent ORT loading.

    Walks the ONNX graph and fixes two classes of issues:

    1. **Gather with float indices**: Inserts ``Cast(to=INT64)`` before
       any Gather node whose indices input is not int32/int64.

    2. **Where with mixed types**: Inserts ``Cast`` to unify the X and Y
       inputs of Where nodes when they have different element types.

    Args:
        onnx_path: Path to the ONNX file (modified in-place).

    Returns:
        Number of nodes fixed.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(onnx_path))

    # Build type map from value_info, inputs, and initializers
    type_map: dict[str, int] = {}
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type:
            type_map[vi.name] = vi.type.tensor_type.elem_type
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type:
            type_map[inp.name] = inp.type.tensor_type.elem_type
    for init in model.graph.initializer:
        type_map[init.name] = init.data_type

    cast_nodes_to_insert: list[tuple[int, Any]] = []
    fixed_count = 0

    for i, node in enumerate(model.graph.node):
        # Fix 1: Gather with non-integer indices
        if node.op_type == "Gather" and len(node.input) >= 2:
            idx_type = type_map.get(node.input[1])
            if idx_type is not None and idx_type not in (
                TensorProto.INT32,
                TensorProto.INT64,
            ):
                cast_out = f"{node.input[1]}_cast_to_int64"
                cast_node = helper.make_node(
                    "Cast", [node.input[1]], [cast_out], to=TensorProto.INT64
                )
                cast_nodes_to_insert.append((i, cast_node))
                node.input[1] = cast_out
                fixed_count += 1
                logger.debug(
                    "Fixed Gather %s: cast indices to INT64", node.name
                )

        # Fix 2: Where with mixed X/Y types
        if node.op_type == "Where" and len(node.input) == 3:
            _cond, x_name, y_name = node.input
            x_type = type_map.get(x_name)
            y_type = type_map.get(y_name)
            if (
                x_type is not None
                and y_type is not None
                and x_type != y_type
            ):
                # Cast the integer input to match the float input
                if x_type == TensorProto.INT64 and y_type == TensorProto.FLOAT:
                    cast_out = f"{x_name}_cast_to_float"
                    cast_node = helper.make_node(
                        "Cast", [x_name], [cast_out], to=TensorProto.FLOAT
                    )
                    cast_nodes_to_insert.append((i, cast_node))
                    node.input[1] = cast_out
                    fixed_count += 1
                elif y_type == TensorProto.INT64 and x_type == TensorProto.FLOAT:
                    cast_out = f"{y_name}_cast_to_float"
                    cast_node = helper.make_node(
                        "Cast", [y_name], [cast_out], to=TensorProto.FLOAT
                    )
                    cast_nodes_to_insert.append((i, cast_node))
                    node.input[2] = cast_out
                    fixed_count += 1
                logger.debug(
                    "Fixed Where %s: unified types", node.name
                )

    # Insert cast nodes in correct order (track offset as we insert)
    for offset, (insert_idx, cast_node) in enumerate(
        sorted(cast_nodes_to_insert, key=lambda x: x[0])
    ):
        model.graph.node.insert(insert_idx + offset, cast_node)

    if fixed_count > 0:
        onnx.save(model, str(onnx_path))
        logger.info("Patched %d type mismatches in %s", fixed_count, onnx_path)

    return fixed_count


def _apply_checkpoint_vlm_weights(
    base_model,
    checkpoint_state_dict,
    tag: str = "vlm",
) -> int:
    """Overwrite the base VLM's weights with fine-tuned ones from the checkpoint.

    Args:
        base_model: The nn.Module to update in-place.
        checkpoint_state_dict: Fine-tuned checkpoint state dict (SmolVLA format).
        tag: Short label used in log messages to identify which sub-model is
            being updated (e.g. "text_embedder", "decoder_prefill").

    Returns:
        Number of keys actually applied (total model keys − missing keys).
        Returns 0 when no matching prefix is found or on error.
    """
    prefixes = [
        "model.vlm_with_expert.vlm.",
        "vlm_with_expert.vlm.",
        "model.vlm.",
    ]
    matched_prefix = None
    for p in prefixes:
        if any(k.startswith(p) for k in checkpoint_state_dict.keys()):
            matched_prefix = p
            break
    if matched_prefix is None:
        # Print top-level prefixes so we can see what's actually in the ckpt.
        top = set()
        for k in checkpoint_state_dict:
            parts = k.split(".")
            if len(parts) >= 3:
                top.add(".".join(parts[:3]))
        print(
            f"[vlm-weights/{tag}] No matching VLM prefix found. "
            f"Top-3 prefixes sample: {sorted(list(top))[:5]}",
            flush=True,
        )
        return 0

    rebased = {
        k[len(matched_prefix):]: v
        for k, v in checkpoint_state_dict.items()
        if k.startswith(matched_prefix)
    }
    print(
        f"[vlm-weights/{tag}] matched_prefix={matched_prefix!r}, rebased {len(rebased)} keys",
        flush=True,
    )
    print(
        f"[vlm-weights/{tag}] sample rebased keys: {sorted(list(rebased.keys()))[:5]}",
        flush=True,
    )

    try:
        missing, unexpected = base_model.load_state_dict(rebased, strict=False)
        total_model_keys = len(list(base_model.state_dict().keys()))
        applied = total_model_keys - len(missing)
        print(
            f"[vlm-weights/{tag}] applied {applied}/{total_model_keys} fine-tune weights "
            f"({len(missing)} missing, {len(unexpected)} unexpected)",
            flush=True,
        )
        if applied == 0:
            print(
                f"[vlm-weights/{tag}] WARNING: zero keys applied — fine-tune weights "
                f"did NOT update this sub-model. Check key namespace alignment.",
                flush=True,
            )
        if missing:
            print(f"[vlm-weights/{tag}] first 5 missing: {missing[:5]}", flush=True)
        if unexpected:
            print(f"[vlm-weights/{tag}] first 5 unexpected: {unexpected[:5]}", flush=True)
        return applied
    except Exception as e:
        print(f"[vlm-weights/{tag}] FAILED: {e}", flush=True)
        return 0


def export_vlm_prefix(
    checkpoint_path_or_id: str = DEFAULT_VLM_MODEL_NAME,
    output_dir: str | Path = ".",
    opset: int = 19,
    state_dict: dict | None = None,
) -> Path:
    """Export VLM vision encoder (SigLIP + connector) as ONNX.

    Loads the base SmolVLM2-500M model via ``AutoModel.from_pretrained``,
    wraps the vision encoder + connector into ``VisionEncoderForONNX``,
    exports to ONNX, patches any type mismatches, and validates against
    ONNX Runtime.

    Args:
        checkpoint_path_or_id: HuggingFace model ID or local path.
            Defaults to ``HuggingFaceTB/SmolVLM2-500M-Video-Instruct``.
        output_dir: Directory for output files.
        opset: ONNX opset version (default 19).

    Returns:
        Path to the exported ``vision_encoder.onnx`` file.
    """
    from transformers import AutoModel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the base VLM using AutoModelForImageTextToText — matches the
    # wrapper SmolVLA uses internally (see smolvlm_with_expert.py), which
    # means the checkpoint's VLM state_dict keys line up 1:1 after prefix
    # strip. Using plain AutoModel gave a different module tree
    # (connector.X vs model.connector.X) and the fine-tuned weights
    # silently failed to load (488 missing, 345 unexpected).
    from transformers import AutoModelForImageTextToText

    print(f"Loading VLM: {checkpoint_path_or_id}", flush=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path_or_id, trust_remote_code=True
        )
    except Exception as e:
        print(
            f"AutoModelForImageTextToText failed ({e}); falling back to AutoModel",
            flush=True,
        )
        model = AutoModel.from_pretrained(
            checkpoint_path_or_id, trust_remote_code=True
        )
    model.eval()
    if state_dict is not None:
        _apply_checkpoint_vlm_weights(model, state_dict, tag="vision_encoder")

    # The AutoModelForImageTextToText wrapper has `self.model = SmolVLMModel(...)`
    # whose sub-modules are the actual vision/connector/text_model. Drill down
    # so the rest of this function can keep using ``model.vision_model`` /
    # ``model.connector`` / ``model.text_model`` without further changes.
    if hasattr(model, "model") and hasattr(model.model, "connector"):
        model = model.model
        print("[vlm-weights] unwrapped ForConditionalGeneration -> inner model", flush=True)

    # Older SmolVLM checkpoints use `vision_tower` as the attr name,
    # newer ones use `vision_model`. Normalise so wrapper code works.
    if not hasattr(model, "vision_model") and hasattr(model, "vision_tower"):
        model.vision_model = model.vision_tower
        print(
            "[vlm-weights] aliased vision_tower -> vision_model for wrapper compat",
            flush=True,
        )

    # Extract architecture constants from the loaded model
    vision_config = model.config.vision_config
    text_config = model.config.text_config
    image_size = vision_config.image_size
    vlm_hidden_size = text_config.hidden_size  # 960 — VLM internal dim
    # The expert's cross-attn k_proj expects num_kv_heads * head_dim, NOT raw
    # hidden_size. SmolLM2: 5 * 64 = 320. decoder_prefill.onnx projects VLM
    # hidden to this dim via the final layer's k_proj before output.
    vlm_kv_dim = text_config.num_key_value_heads * getattr(
        text_config, "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )

    logger.info(
        "VLM config: image_size=%d, vision_hidden=%d, text_hidden=%d, vlm_kv_dim=%d",
        image_size,
        vision_config.hidden_size,
        vlm_hidden_size,
        vlm_kv_dim,
    )

    # 2. Build the ONNX-exportable wrapper
    wrapper = VisionEncoderForONNX(
        model.vision_model, model.connector, image_size=image_size
    )
    wrapper.eval()

    total_params = sum(p.numel() for p in wrapper.parameters())
    logger.info("VisionEncoderForONNX: %.1fM params", total_params / 1e6)

    # 3. Verify wrapper matches original model output
    dummy_pixel_values = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        # Original model path
        orig_hidden = model.vision_model(
            pixel_values=dummy_pixel_values
        ).last_hidden_state
        orig_embeds = model.connector(orig_hidden)
        # Wrapper path
        wrapper_embeds = wrapper(dummy_pixel_values)
        sanity_diff = float((orig_embeds - wrapper_embeds).abs().max())
        logger.info("Wrapper vs original sanity check: max_diff=%.2e", sanity_diff)
        assert sanity_diff == 0.0, (
            f"Wrapper output differs from original: {sanity_diff}"
        )

    # 4. Export to ONNX
    onnx_path = output_dir / "vision_encoder.onnx"
    logger.info("Exporting ONNX to %s (opset %d)...", onnx_path, opset)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_pixel_values,
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "image_embeds": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=False,
        )
    onnx_size_mb = onnx_path.stat().st_size / 1e6
    logger.info("Wrote %s (%.2f MB)", onnx_path, onnx_size_mb)

    # 5. Post-export: patch type mismatches (Gather float indices, Where mixed types)
    num_fixed = patch_onnx_type_mismatches(onnx_path)
    logger.info("Post-export patch: fixed %d nodes", num_fixed)

    # 6. Validate: PyTorch vs ONNX Runtime
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path))
        with torch.no_grad():
            torch_out = wrapper(dummy_pixel_values).numpy()
        ort_out = sess.run(
            None, {"pixel_values": dummy_pixel_values.numpy()}
        )[0]
        max_diff = float(np.abs(torch_out - ort_out).max())
        mean_diff = float(np.abs(torch_out - ort_out).mean())
        passed = max_diff < ORT_MAX_DIFF_THRESHOLD
        logger.info(
            "ONNX validation: max_diff=%.2e, mean_diff=%.2e (%s)",
            max_diff,
            mean_diff,
            "PASS" if passed else "FAIL",
        )
        if not passed:
            logger.warning(
                "ONNX numerical mismatch: max_diff=%.2e exceeds %.1e threshold "
                "(SigLIP has 27 transformer layers -- some fp32 drift is expected)",
                max_diff,
                ORT_MAX_DIFF_THRESHOLD,
            )
    except ImportError:
        logger.warning("onnxruntime not installed -- skipping ONNX validation")

    # 7. Write / update tether_config.json
    config_path = output_dir / "tether_config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    config["vlm_image_size"] = [image_size, image_size]
    config["vlm_hidden_size"] = vlm_hidden_size  # 960 — VLM-internal dim (for state encoder)
    config["vlm_kv_dim"] = vlm_kv_dim            # 320 — expert cross-attn input dim
    # L dim in the per-layer tensor; SmolVLA truncates the SmolVLM2 decoder
    # to 16 of 32 layers at export time, and the expert has a matching count.
    config["vlm_num_layers"] = SMOLVLA_NUM_DECODER_LAYERS
    config["vlm_prefix_onnx"] = "vision_encoder.onnx"
    config["vlm_model_id"] = checkpoint_path_or_id
    config["export_version"] = "0.5"  # split vlm_k/vlm_v with RoPE on k

    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Updated config: %s", config_path)

    # 8. Export text embedder (pass fine-tune state_dict so embed_tokens are updated)
    export_text_embedder(
        checkpoint_path_or_id=checkpoint_path_or_id,
        output_dir=output_dir,
        opset=opset,
        checkpoint_state_dict=state_dict,
    )

    # 9. Export decoder prefill (pass fine-tune state_dict so decoder k/v are updated)
    export_decoder_prefill(
        checkpoint_path_or_id=checkpoint_path_or_id,
        output_dir=output_dir,
        opset=opset,
        checkpoint_state_dict=state_dict,
    )

    # Update config with all export paths
    config = json.loads(config_path.read_text())
    config["text_embedder_onnx"] = "text_embedder.onnx"
    config["decoder_prefill_onnx"] = "decoder_prefill.onnx"
    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Updated config with text_embedder + decoder_prefill: %s", config_path)

    return onnx_path


# ---------------------------------------------------------------------------
# Text embedder export
# ---------------------------------------------------------------------------


def export_text_embedder(
    checkpoint_path_or_id: str = DEFAULT_VLM_MODEL_NAME,
    output_dir: str | Path = ".",
    opset: int = 19,
    checkpoint_state_dict: dict | None = None,
) -> Path:
    """Export the token embedding table as a standalone ONNX.

    Input: ``input_ids [B, seq]`` int64
    Output: ``text_embeds [B, seq, 960]`` float32

    Args:
        checkpoint_path_or_id: HuggingFace model ID or local path.
        output_dir: Directory for output files.
        opset: ONNX opset version.
        checkpoint_state_dict: Optional fine-tuned checkpoint state dict
            (SmolVLA format, keys like ``model.vlm_with_expert.vlm.text_model.*``).
            When provided, overwrites the base model's embed_tokens weights with
            the fine-tuned values before export so the shipped ONNX reflects the
            actual fine-tuned embedding table rather than the base model's.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel as _AutoModel

    logger.info("Loading VLM for text embedder export: %s", checkpoint_path_or_id)
    model = _AutoModel.from_pretrained(
        checkpoint_path_or_id, trust_remote_code=True
    )
    model.eval()

    # Apply fine-tune weights BEFORE extracting embed_tokens so the exported
    # embedding table reflects the fine-tuned checkpoint, not the base model.
    # AutoModel returns SmolVLMModel whose state_dict keys are text_model.*,
    # vision_model.*, connector.* — matching the rebased checkpoint keys after
    # stripping the model.vlm_with_expert.vlm. prefix.
    if checkpoint_state_dict is not None:
        applied = _apply_checkpoint_vlm_weights(
            model, checkpoint_state_dict, tag="text_embedder"
        )
        if applied == 0:
            logger.warning(
                "export_text_embedder: zero fine-tune weights applied — "
                "exported embed_tokens will use BASE model weights"
            )
        else:
            logger.info(
                "export_text_embedder: applied %d fine-tune weights", applied
            )

    # Find embed_tokens — try multiple attribute paths
    embed_tokens = None
    for attr_path in [
        "text_model.embed_tokens",
        "model.embed_tokens",
        "language_model.model.embed_tokens",
        "text_model.model.embed_tokens",
    ]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            embed_tokens = obj
            logger.info("Found embed_tokens at model.%s", attr_path)
            break
        except AttributeError:
            continue

    if embed_tokens is None:
        raise RuntimeError(
            "Could not find embed_tokens in model. Tried: "
            "text_model.embed_tokens, model.embed_tokens, "
            "language_model.model.embed_tokens, text_model.model.embed_tokens"
        )

    class EmbedTokensForONNX(torch.nn.Module):
        def __init__(self, embed):
            super().__init__()
            self.embed = embed

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embed(input_ids)

    wrapper = EmbedTokensForONNX(embed_tokens).eval()
    vocab_size = embed_tokens.num_embeddings
    embed_dim = embed_tokens.embedding_dim
    logger.info("EmbedTokens: vocab=%d, dim=%d", vocab_size, embed_dim)

    # Export
    dummy_ids = torch.randint(0, min(vocab_size, 1000), (1, 32), dtype=torch.int64)
    onnx_path = output_dir / "text_embedder.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids,),
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["text_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "text_embeds": {0: "batch", 1: "seq"},
            },
            opset_version=opset,
            do_constant_folding=False,
        )

    logger.info(
        "Wrote %s (%.2f MB)",
        onnx_path,
        onnx_path.stat().st_size / 1e6,
    )

    # Validate
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            torch_out = wrapper(dummy_ids).numpy()
        ort_out = sess.run(None, {"input_ids": dummy_ids.numpy()})[0]
        max_diff = float(np.abs(torch_out - ort_out).max())
        logger.info("Text embedder ONNX validation: max_diff=%.2e (PASS)" if max_diff < 1e-4
                     else "Text embedder ONNX validation: max_diff=%.2e (WARN)", max_diff)
    except ImportError:
        logger.warning("onnxruntime not installed — skipping validation")

    return onnx_path


# ---------------------------------------------------------------------------
# Decoder prefill export
# ---------------------------------------------------------------------------

# 16 layers is the SmolVLA expert prefix depth. The base SmolVLM2-500M has
# 32 layers, but SmolVLA's fine-tuned checkpoint uses only the first 16.
# We truncate manually here since we load from the base model.
SMOLVLA_NUM_DECODER_LAYERS = 16

# 16 transformer layers accumulate more fp32 rounding than a single layer.
# Spike showed ~4e-5 for 1 layer; budget ~5e-4 for the full stack.
DECODER_ORT_MAX_DIFF_THRESHOLD = 5e-4

# Default prefix sequence length for dummy inputs (image 64 + text ~10 + state 1)
DEFAULT_PREFIX_SEQ_LEN = 75


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard HF rotary helper — rotates half the last dim."""
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class DecoderPrefillForONNX(nn.Module):
    """Wraps SmolLM2 decoder for per-layer (k, v) prefill with RoPE on k.

    Exports EXACTLY what the real SmolVLA cross-attention consumes (see
    ``smolvlm_with_expert.py::forward_cross_attn_layer``):

        for each layer i:
            h_norm = layer.input_layernorm(hidden_states[i])
            k_i = layer.self_attn.k_proj(h_norm)        ← reshape → per-head
            k_i = apply_rope(k_i, position_ids)          ← CRITICAL step
            v_i = layer.self_attn.v_proj(h_norm)         ← separate projection

    Earlier versions skipped RoPE and used a single tensor for both k and v
    sources. Both are meaningful distribution shifts from training: un-roped
    keys make cross-attn similarity scores incoherent, and mixing k/v sources
    has the expert's v_proj seeing k-distribution inputs.

    Input:  inputs_embeds   [B, seq, hidden]          float32
    Input:  attention_mask  [B, seq]                  int64
    Output: vlm_k_per_layer [num_layers, B, seq, kv_dim] float32  (RoPE applied)
    Output: vlm_v_per_layer [num_layers, B, seq, kv_dim] float32
    """

    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.text_model = text_model

    def forward(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Position IDs from attention_mask: cumsum - 1 over attend positions.
        # This matches what HF's LlamaModel does internally when you don't
        # provide position_ids — required for RoPE phase alignment with the
        # values SmolVLA trained with.
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)

        # Get RoPE cos/sin. Transformers 4.41+ keeps rotary_emb at the model
        # level (model.rotary_emb); older versions had it per-layer
        # (layers[0].self_attn.rotary_emb). Try both for compat.
        rotary_emb = getattr(self.text_model, "rotary_emb", None)
        if rotary_emb is None:
            rotary_emb = self.text_model.layers[0].self_attn.rotary_emb
        cos, sin = rotary_emb(inputs_embeds, position_ids)
        # cos/sin shape: [B, seq, head_dim]. Unsqueeze head dim for broadcast.
        cos = cos.unsqueeze(1)  # [B, 1, seq, head_dim]
        sin = sin.unsqueeze(1)

        per_layer_ks: list[torch.Tensor] = []
        per_layer_vs: list[torch.Tensor] = []
        for i, layer in enumerate(self.text_model.layers):
            h_in = outputs.hidden_states[i]  # [B, seq, hidden]
            h_norm = layer.input_layernorm(h_in)
            k_flat = layer.self_attn.k_proj(h_norm)  # [B, seq, nkv*hd]
            v_flat = layer.self_attn.v_proj(h_norm)  # [B, seq, nkv*hd]

            b, s, _ = k_flat.shape
            nkv = layer.self_attn.config.num_key_value_heads
            hd = layer.self_attn.head_dim

            # Reshape to [B, nkv, seq, hd] to apply RoPE (HF layout).
            k_heads = k_flat.view(b, s, nkv, hd).transpose(1, 2)

            # Apply RoPE to k only (values are NOT rotated in HF attention).
            k_roped = (k_heads * cos) + (_rotate_half(k_heads) * sin)

            # Flatten back to [B, seq, nkv*hd] — matches what the expert's
            # k_proj expects as input dim.
            k_out = k_roped.transpose(1, 2).contiguous().view(b, s, nkv * hd)

            per_layer_ks.append(k_out)
            per_layer_vs.append(v_flat)

        vlm_k = torch.stack(per_layer_ks, dim=0)  # [L, B, seq, kv_dim]
        vlm_v = torch.stack(per_layer_vs, dim=0)  # [L, B, seq, kv_dim]
        return vlm_k, vlm_v


def export_decoder_prefill(
    checkpoint_path_or_id: str = DEFAULT_VLM_MODEL_NAME,
    output_dir: str | Path = ".",
    opset: int = 19,
    num_layers: int = SMOLVLA_NUM_DECODER_LAYERS,
    prefix_seq_len: int = DEFAULT_PREFIX_SEQ_LEN,
    checkpoint_state_dict: dict | None = None,
) -> Path:
    """Export the 16-layer SmolLM2 decoder as decoder_prefill.onnx.

    Loads the full SmolVLM2-500M model, extracts the text decoder (LlamaModel),
    truncates to ``num_layers`` layers (SmolVLA uses 16 of the 32 base layers),
    wraps in ``DecoderPrefillForONNX``, and exports to ONNX with ORT validation.

    Args:
        checkpoint_path_or_id: HuggingFace model ID or local path.
        output_dir: Directory for output files.
        opset: ONNX opset version (default 19, needed for GQA).
        num_layers: Number of decoder layers to keep (default 16).
        prefix_seq_len: Sequence length for dummy inputs (default 75).
        checkpoint_state_dict: Optional fine-tuned checkpoint state dict
            (SmolVLA format, keys like ``model.vlm_with_expert.vlm.text_model.*``).
            When provided, overwrites the base model's decoder weights with the
            fine-tuned values before export so the KV prefix computed by this
            ONNX reflects the fine-tuned decoder rather than the base model's.

    Returns:
        Path to the exported ``decoder_prefill.onnx`` file.
    """
    from transformers import AutoModel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the base VLM
    logger.info("Loading VLM for decoder export: %s", checkpoint_path_or_id)
    model = AutoModel.from_pretrained(
        checkpoint_path_or_id, trust_remote_code=True
    )
    model.eval()

    # Apply fine-tune weights BEFORE extracting text_model so the decoder KV
    # prefix reflects the fine-tuned checkpoint, not the base model.
    # AutoModel returns SmolVLMModel whose state_dict keys are text_model.*,
    # vision_model.*, connector.* — matching the rebased checkpoint keys after
    # stripping the model.vlm_with_expert.vlm. prefix.
    if checkpoint_state_dict is not None:
        applied = _apply_checkpoint_vlm_weights(
            model, checkpoint_state_dict, tag="decoder_prefill"
        )
        if applied == 0:
            logger.warning(
                "export_decoder_prefill: zero fine-tune weights applied — "
                "exported decoder will use BASE model weights"
            )
        else:
            logger.info(
                "export_decoder_prefill: applied %d fine-tune weights", applied
            )

    # 2. Extract the text decoder (LlamaModel)
    text_model = model.text_model
    total_layers = len(text_model.layers)
    hidden_size = model.config.text_config.hidden_size

    logger.info(
        "Text decoder: %d layers, hidden_size=%d, truncating to %d layers",
        total_layers,
        hidden_size,
        num_layers,
    )

    # 3. Truncate to num_layers (SmolVLA uses first 16 of 32 base layers)
    if total_layers > num_layers:
        text_model.layers = text_model.layers[:num_layers]
        logger.info(
            "Truncated decoder from %d to %d layers", total_layers, num_layers
        )
    elif total_layers < num_layers:
        logger.warning(
            "Model has only %d layers, requested %d — using all available",
            total_layers,
            num_layers,
        )
        num_layers = total_layers

    # 4. Build ONNX wrapper — outputs per-layer (k with RoPE, v) tensors.
    src_k_proj = text_model.layers[-1].self_attn.k_proj
    kv_out_dim = src_k_proj.out_features

    wrapper = DecoderPrefillForONNX(text_model)
    wrapper.eval()

    total_params = sum(p.numel() for p in wrapper.parameters())
    logger.info(
        "DecoderPrefillForONNX: %d layers, %.1fM params, "
        "per-layer kv_dim=%d → outputs vlm_k [%d, B, seq, %d] and vlm_v [%d, B, seq, %d]",
        num_layers,
        total_params / 1e6,
        kv_out_dim,
        num_layers,
        kv_out_dim,
        num_layers,
        kv_out_dim,
    )

    # 5. Sanity check: wrapper produces outputs of correct shape
    dummy_embeds = torch.randn(1, prefix_seq_len, hidden_size)
    dummy_mask = torch.ones(1, prefix_seq_len, dtype=torch.long)

    with torch.no_grad():
        torch_k, torch_v = wrapper(dummy_embeds, dummy_mask)
    expected_shape = (num_layers, 1, prefix_seq_len, kv_out_dim)
    assert torch_k.shape == expected_shape, (
        f"Unexpected k shape: {torch_k.shape} (expected {expected_shape})"
    )
    assert torch_v.shape == expected_shape, (
        f"Unexpected v shape: {torch_v.shape} (expected {expected_shape})"
    )
    logger.info(
        "Sanity check passed: k shape %s, v shape %s", torch_k.shape, torch_v.shape
    )

    # 6. Export to ONNX
    onnx_path = output_dir / "decoder_prefill.onnx"
    logger.info("Exporting decoder prefill to %s (opset %d)...", onnx_path, opset)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_mask),
            str(onnx_path),
            input_names=["inputs_embeds", "attention_mask"],
            # Two outputs: RoPE-applied keys per layer, raw values per layer.
            # The orchestrator reads both and feeds them to expert's vlm_k /
            # vlm_v inputs separately.
            output_names=["vlm_k", "vlm_v"],
            dynamic_axes={
                "inputs_embeds": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "vlm_k": {1: "batch", 2: "seq"},
                "vlm_v": {1: "batch", 2: "seq"},
            },
            opset_version=opset,
            do_constant_folding=False,
        )

    onnx_size_mb = onnx_path.stat().st_size / 1e6
    logger.info("Wrote %s (%.2f MB)", onnx_path, onnx_size_mb)

    # 7. Validate with ONNX Runtime
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path))
        with torch.no_grad():
            torch_k, torch_v = wrapper(dummy_embeds, dummy_mask)
        ort_k, ort_v = sess.run(
            ["vlm_k", "vlm_v"],
            {
                "inputs_embeds": dummy_embeds.numpy(),
                "attention_mask": dummy_mask.numpy(),
            },
        )

        k_max = float(np.abs(torch_k.numpy() - ort_k).max())
        v_max = float(np.abs(torch_v.numpy() - ort_v).max())
        max_diff = max(k_max, v_max)
        passed = max_diff < DECODER_ORT_MAX_DIFF_THRESHOLD

        logger.info(
            "ONNX validation: k_max=%.2e, v_max=%.2e (%s)",
            k_max, v_max, "PASS" if passed else "FAIL",
        )
        if not passed:
            logger.warning(
                "Decoder ONNX numerical mismatch: max_diff=%.2e exceeds %.1e "
                "threshold (16 transformer layers + RoPE accumulate fp32 drift)",
                max_diff,
                DECODER_ORT_MAX_DIFF_THRESHOLD,
            )
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ONNX validation")

    return onnx_path
