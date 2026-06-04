"""VLM component exporters: text embedder, state encoder, prefix assembly.

Complements the vision encoder (Issue 1) by providing the remaining pieces
needed to build a complete VLM prefix for expert cross-attention:

- **Text embedder export**: token embedding table as standalone ONNX
- **StateEncoder**: projects robot state vector → VLM hidden space
- **assemble_prefix**: concatenates [image_embeds, text_embeds, state_embed]
- **pad_state**: zero-pads robot state to max_state_dim

Architecture constants match SmolVLA / SmolVLM2-500M:
    hidden_size = 960, max_state_dim = 32, image tokens = 64.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# SmolVLA / SmolVLM2-500M architecture constants
HIDDEN_SIZE = 960
MAX_STATE_DIM = 32
IMAGE_SEQ_LEN = 64  # number of image tokens after SigLIP + pooling


# ---------------------------------------------------------------------------
# Text embedder export
# ---------------------------------------------------------------------------

class _TextEmbedderWrapper(nn.Module):
    """Thin wrapper around nn.Embedding for clean ONNX export."""

    def __init__(self, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, seq] int64
        Returns:
            text_embeds: [B, seq, hidden_size] float32
        """
        return self.embed_tokens(input_ids)


def export_text_embedder(
    model: nn.Module,
    output_dir: str | Path,
    opset: int = 19,
) -> Path:
    """Export the token embedding table as a standalone ONNX.

    Extracts ``embed_tokens`` from the model's text backbone and exports it
    so that tokenized instruction IDs can be mapped to embedding vectors
    without loading the full VLM at inference time.

    Args:
        model: A SmolVLM2 or SmolVLAPolicy model instance. The function tries
            several attribute paths to find the embedding table.
        output_dir: Directory for the exported ONNX file.
        opset: ONNX opset version.

    Returns:
        Path to the exported ``text_embedder.onnx``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate embed_tokens from various model layouts
    embed_tokens: Optional[nn.Embedding] = None
    search_paths = [
        "vlm_with_expert.text_model.embed_tokens",
        "model.vlm_with_expert.text_model.embed_tokens",
        "text_model.embed_tokens",
        "model.text_model.embed_tokens",
        "model.embed_tokens",
    ]
    for attr_path in search_paths:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.Embedding):
                embed_tokens = obj
                logger.info("Found embed_tokens at %s", attr_path)
                break
        except AttributeError:
            continue

    if embed_tokens is None:
        raise ValueError(
            "Could not locate embed_tokens on the model. "
            f"Searched: {search_paths}"
        )

    wrapper = _TextEmbedderWrapper(embed_tokens)
    wrapper.eval()

    vocab_size, hidden = embed_tokens.weight.shape
    logger.info(
        "Text embedder: vocab_size=%d, hidden_size=%d (%.2fM params)",
        vocab_size, hidden, embed_tokens.weight.numel() / 1e6,
    )

    # Dummy input
    dummy_ids = torch.randint(0, vocab_size, (1, 16), dtype=torch.int64)

    onnx_path = output_dir / "text_embedder.onnx"
    logger.info("Exporting text embedder to %s (opset %d)", onnx_path, opset)

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

    logger.info("Wrote %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """Projects robot state vector to VLM hidden space.

    SmolVLA uses ``nn.Linear(max_state_dim=32, hidden_size=960)`` to produce
    a single "state token" that is concatenated with image and text embeddings
    before the expert prefill pass.

    For v1 the weights are randomly initialised. Real weights are loaded from
    the SmolVLAPolicy checkpoint in the orchestrator step (Issue 3/5).

    Args:
        state_proj_weight: Optional weight tensor [hidden_size, max_state_dim].
        state_proj_bias: Optional bias tensor [hidden_size].
        max_state_dim: Input state dimensionality (default 32).
        hidden_size: Output embedding dimensionality (default 960).
    """

    def __init__(
        self,
        state_proj_weight: torch.Tensor | None = None,
        state_proj_bias: torch.Tensor | None = None,
        max_state_dim: int = MAX_STATE_DIM,
        hidden_size: int = HIDDEN_SIZE,
    ):
        super().__init__()
        self.max_state_dim = max_state_dim
        self.hidden_size = hidden_size
        self.proj = nn.Linear(max_state_dim, hidden_size)

        # Optionally load real weights
        if state_proj_weight is not None:
            assert state_proj_weight.shape == (hidden_size, max_state_dim), (
                f"Expected weight shape ({hidden_size}, {max_state_dim}), "
                f"got {state_proj_weight.shape}"
            )
            self.proj.weight.data.copy_(state_proj_weight)
        if state_proj_bias is not None:
            assert state_proj_bias.shape == (hidden_size,), (
                f"Expected bias shape ({hidden_size},), got {state_proj_bias.shape}"
            )
            self.proj.bias.data.copy_(state_proj_bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Project state vector into a single VLM-space token.

        Args:
            state: [B, max_state_dim] float32 (zero-padded from actual DoF).

        Returns:
            state_embed: [B, 1, hidden_size] float32
        """
        # state: [B, 32] → proj → [B, 960] → unsqueeze → [B, 1, 960]
        return self.proj(state).unsqueeze(1)


def export_state_encoder(
    encoder: StateEncoder,
    output_dir: str | Path,
    opset: int = 19,
) -> Path:
    """Export StateEncoder as a standalone ONNX (optional, small enough to inline).

    Args:
        encoder: A StateEncoder instance.
        output_dir: Directory for output file.
        opset: ONNX opset version.

    Returns:
        Path to exported ``state_encoder.onnx``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder.eval()
    dummy_state = torch.randn(1, encoder.max_state_dim)

    onnx_path = output_dir / "state_encoder.onnx"
    logger.info("Exporting state encoder to %s", onnx_path)

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (dummy_state,),
            str(onnx_path),
            input_names=["state"],
            output_names=["state_embed"],
            dynamic_axes={
                "state": {0: "batch"},
                "state_embed": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=False,
        )

    logger.info("Wrote %s (%.2f KB)", onnx_path, onnx_path.stat().st_size / 1e3)
    return onnx_path


# ---------------------------------------------------------------------------
# State padding utility
# ---------------------------------------------------------------------------

def pad_state(
    state: np.ndarray,
    max_state_dim: int = MAX_STATE_DIM,
) -> np.ndarray:
    """Zero-pad robot state to ``max_state_dim``.

    Args:
        state: Array of shape ``[B, D]`` or ``[D]`` where ``D <= max_state_dim``.

    Returns:
        Padded array of shape ``[B, max_state_dim]`` or ``[max_state_dim]``.

    Raises:
        ValueError: If the state dimension exceeds ``max_state_dim``.
    """
    state = np.asarray(state, dtype=np.float32)

    if state.ndim == 1:
        d = state.shape[0]
        if d > max_state_dim:
            raise ValueError(
                f"State dim {d} exceeds max_state_dim {max_state_dim}"
            )
        if d == max_state_dim:
            return state
        padded = np.zeros(max_state_dim, dtype=np.float32)
        padded[:d] = state
        return padded

    if state.ndim == 2:
        b, d = state.shape
        if d > max_state_dim:
            raise ValueError(
                f"State dim {d} exceeds max_state_dim {max_state_dim}"
            )
        if d == max_state_dim:
            return state
        padded = np.zeros((b, max_state_dim), dtype=np.float32)
        padded[:, :d] = state
        return padded

    raise ValueError(f"Expected 1D or 2D state array, got {state.ndim}D")


# ---------------------------------------------------------------------------
# Prefix assembly
# ---------------------------------------------------------------------------

def assemble_prefix(
    image_embeds: np.ndarray,
    text_embeds: np.ndarray,
    state_embed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate [image_embeds, text_embeds, state_embed] into prefix sequence.

    Builds the full prefix embedding that feeds into the expert prefill pass,
    along with the attention mask indicating bidirectional (image + text) vs
    causal (state) regions.

    Args:
        image_embeds: ``[B, N_img, hidden]`` from vision encoder (typically [B, 64, 960]).
        text_embeds: ``[B, T, hidden]`` from text embedder.
        state_embed: ``[B, 1, hidden]`` from state encoder.

    Returns:
        prefix_embeds: ``[B, N_img + T + 1, hidden]`` float32
        attention_mask: ``[B, N_img + T + 1]`` int64
            - 0 for image + text tokens (bidirectional attention)
            - 1 for state token (causal attention)
    """
    image_embeds = np.asarray(image_embeds, dtype=np.float32)
    text_embeds = np.asarray(text_embeds, dtype=np.float32)
    state_embed = np.asarray(state_embed, dtype=np.float32)

    # Validate shapes
    assert image_embeds.ndim == 3, f"image_embeds must be 3D, got {image_embeds.ndim}D"
    assert text_embeds.ndim == 3, f"text_embeds must be 3D, got {text_embeds.ndim}D"
    assert state_embed.ndim == 3, f"state_embed must be 3D, got {state_embed.ndim}D"

    b = image_embeds.shape[0]
    hidden = image_embeds.shape[2]

    assert text_embeds.shape[0] == b, "Batch size mismatch"
    assert state_embed.shape[0] == b, "Batch size mismatch"
    assert text_embeds.shape[2] == hidden, (
        f"Hidden dim mismatch: image={hidden}, text={text_embeds.shape[2]}"
    )
    assert state_embed.shape[2] == hidden, (
        f"Hidden dim mismatch: image={hidden}, state={state_embed.shape[2]}"
    )

    # Concatenate along sequence dimension
    prefix_embeds = np.concatenate(
        [image_embeds, text_embeds, state_embed], axis=1
    )  # [B, N_img + T + 1, hidden]

    # Build attention mask:
    #   0 = bidirectional (image + text tokens)
    #   1 = causal (state token)
    n_img = image_embeds.shape[1]
    n_txt = text_embeds.shape[1]
    n_state = state_embed.shape[1]
    total_len = n_img + n_txt + n_state

    attention_mask = np.zeros((b, total_len), dtype=np.int64)
    # State tokens get causal mask (last n_state positions)
    attention_mask[:, n_img + n_txt:] = 1

    return prefix_embeds, attention_mask
