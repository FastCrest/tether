"""pi0 prefix-KV export pipeline — orchestrates Optimum + our custom expert.

Goal: take `lerobot/pi0_base` (PaliGemma + Gemma expert + flow matching)
and produce ONNX files that TensorRT can compile on Jetson, preserving
cos >= 0.999 end-to-end parity vs the reference PyTorch policy.

Architecture (pi0 forward):
    image -> SigLIP vision_tower -> 256 vision features (1152-dim)
                                        |
                                        v
                               multi_modal_projector
                                     (1152->2048)
                                        |
    text -> embed_tokens (257152x2048)  |
                  |                     |
                  v                     v
                  +---------------------+
                            |
                            v
                  Gemma language_model (18 layers)
                  -> per-layer past_key_values (prefix KV)
                            |
                            v
                  gemma_expert (18 layers, separate)
                  + flow matching 10-step Euler
                            |
                            v
                        actions

Export strategy (after 2026-04-17 empirical de-risk):
    1. vision_encoder.onnx     -- Optimum SigLIP (feature-extraction)
    2. multi_modal_projector.onnx -- tiny Linear (1152->2048)
    3. text_embedder.onnx      -- extracted embed_tokens (tied w/ lm_head)
    4. decoder_prefill.onnx    -- Optimum Gemma (text-generation-with-past)
    5. expert_stack.onnx       -- reuse pi0_exporter.py

Each stage verified independently at cos >= 0.9999:
- vision_encoder: Verified 2026-04-17 on pi0_base subset (cos=+0.99999994)
- decoder_prefill: Verified 2026-04-17 on pi0_base subset (cos=+0.99999994)

See https://github.com/FastCrest/reflex-vault/blob/main/reflex_vla/03_research/pi0_empirical_derisk_findings.md
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

logger = logging.getLogger(__name__)


# Pi0 state-dict prefixes (confirmed 2026-04-17 on lerobot/pi0_base)
PI0_VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower."
PI0_LANGUAGE_PREFIX = "paligemma_with_expert.paligemma.model.language_model."
PI0_PROJECTOR_KEYS = {
    "weight": "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight",
    "bias": "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias",
}
PI0_LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"  # tied with embed_tokens


# Default architecture params (confirmed for pi0_base, matches google/paligemma-3b-pt-224)
DEFAULT_SIGLIP_CFG = dict(
    hidden_size=1152,
    intermediate_size=4304,
    num_hidden_layers=27,
    num_attention_heads=16,
    num_channels=3,
    image_size=224,
    patch_size=14,
    hidden_act="gelu_pytorch_tanh",
    layer_norm_eps=1e-6,
    attention_dropout=0.0,
)
DEFAULT_GEMMA_CFG = dict(
    vocab_size=257152,  # PaliGemma tokenizer
    hidden_size=2048,
    intermediate_size=16384,
    num_hidden_layers=18,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
    max_position_embeddings=8192,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
)

# Expert is gemma-300m (smaller variant) — pi0's action expert config
DEFAULT_GEMMA_EXPERT_CFG = dict(
    vocab_size=257152,  # same tokenizer as backbone
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=18,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
    max_position_embeddings=8192,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
)


# Pi0 expert state-dict prefix
PI0_EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."


def load_pi0_state_dict(model_id: str = "lerobot/pi0_base") -> dict[str, torch.Tensor]:
    """Load the full pi0 state dict from HF cache (or download)."""
    repo_dir = Path(snapshot_download(model_id))
    sf_files = sorted(repo_dir.glob("*.safetensors"))
    if not sf_files:
        raise FileNotFoundError(f"No safetensors in {repo_dir}")

    state = {}
    for f in sf_files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                state[k] = sf.get_tensor(k)
    logger.info("Loaded pi0 state dict: %d tensors from %s", len(state), model_id)
    return state


def build_siglip_dir(state: dict[str, torch.Tensor], out_dir: Path) -> Path:
    """Extract SigLIP vision tower → standard HF SiglipVisionModel dir.

    Verified: produces ONNX with cos=+0.99999994 via Optimum (see
    scripts/local_pi0_siglip_parity.py).
    """
    from transformers import SiglipVisionConfig, SiglipVisionModel

    cfg = SiglipVisionConfig(**DEFAULT_SIGLIP_CFG)
    model = SiglipVisionModel(cfg).eval()

    sd = {}
    for k, v in state.items():
        if k.startswith(PI0_VISION_PREFIX):
            sd[k[len(PI0_VISION_PREFIX):]] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(
        "SigLIP: loaded %d keys, missing=%d unexpected=%d",
        len(sd), len(missing), len(unexpected),
    )
    # Missing keys expected: vision_model.head.* (attention-pool head,
    # unused by PaliGemma)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    (out_dir / "preprocessor_config.json").write_text(
        json.dumps({
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 1 / 255.0,
            "resample": 3,
            "size": {"height": 224, "width": 224},
            "image_processor_type": "SiglipImageProcessor",
        })
    )
    return out_dir


def build_expert_dir(state: dict[str, torch.Tensor], out_dir: Path) -> Path:
    """Extract pi0's action expert (gemma_expert.model) -> HF GemmaForCausalLM dir.

    Pi0's expert is _PiGemmaDecoderLayerBase with PiGemmaRMSNorm. For non-pi0.5
    (non-adaptive mode), it's numerically equivalent to standard HF
    GemmaForCausalLM since HF's GemmaRMSNorm also uses (1+w) convention.
    """
    from transformers import GemmaConfig, GemmaForCausalLM

    # Count actual expert layers
    layer_ids = set()
    for k in state:
        if k.startswith(PI0_EXPERT_PREFIX + "layers."):
            try:
                idx = int(k[len(PI0_EXPERT_PREFIX + "layers."):].split(".")[0])
                layer_ids.add(idx)
            except ValueError:
                continue
    num_layers = max(layer_ids) + 1 if layer_ids else 18
    logger.info("Pi0 expert has %d layers", num_layers)

    cfg_kwargs = dict(DEFAULT_GEMMA_EXPERT_CFG)
    cfg_kwargs["num_hidden_layers"] = num_layers
    cfg = GemmaConfig(**cfg_kwargs)
    model = GemmaForCausalLM(cfg).eval()

    sd = {}
    for k, v in state.items():
        if k.startswith(PI0_EXPERT_PREFIX):
            sd[k[len(PI0_EXPERT_PREFIX):]] = v
    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    logger.info("Expert: loaded %d keys, missing=%d unexpected=%d",
                len(sd), len(missing), len(unexpected))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps({
            "tokenizer_class": "GemmaTokenizer",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        })
    )
    return out_dir


def build_gemma_dir(state: dict[str, torch.Tensor], out_dir: Path) -> Path:
    """Extract Gemma language_model backbone → standard HF GemmaForCausalLM dir.

    Verified: produces ONNX with cos=+0.99999994 via Optimum (see
    scripts/local_pi0_gemma_parity.py).
    """
    from transformers import GemmaConfig, GemmaForCausalLM

    cfg = GemmaConfig(**DEFAULT_GEMMA_CFG)
    model = GemmaForCausalLM(cfg).eval()

    sd = {}
    for k, v in state.items():
        if k.startswith(PI0_LANGUAGE_PREFIX):
            sd[k[len(PI0_LANGUAGE_PREFIX):]] = v
    # Also load embed_tokens from lm_head (tied weights in Gemma)
    if PI0_LM_HEAD_KEY in state:
        sd["embed_tokens.weight"] = state[PI0_LM_HEAD_KEY].clone()
        logger.info(
            "Using tied lm_head (%s) as embed_tokens, shape %s",
            PI0_LM_HEAD_KEY, tuple(state[PI0_LM_HEAD_KEY].shape),
        )

    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    # Also tie lm_head explicitly
    if PI0_LM_HEAD_KEY in state:
        model.lm_head.weight = nn.Parameter(state[PI0_LM_HEAD_KEY].clone())
    logger.info(
        "Gemma: loaded %d keys, missing=%d unexpected=%d",
        len(sd), len(missing), len(unexpected),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps({
            "tokenizer_class": "GemmaTokenizer",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        })
    )
    return out_dir


class GemmaFromEmbeds(nn.Module):
    """Thin wrapper around GemmaModel that takes inputs_embeds (skips embed_tokens).

    pi0 fuses vision + text embeddings BEFORE passing to the decoder. Optimum's
    Gemma export via `text-generation-with-past` takes `input_ids` (and internally
    applies embed_tokens). We need the embeddings-in variant.

    Output: `last_hidden_state` + a flat list of per-layer `present.{i}.key/value`
    tensors, matching the ONNX shape our Pi0OnnxServer expects.
    """

    def __init__(self, gemma_model: nn.Module):
        super().__init__()
        # gemma_model is transformers.GemmaModel (NOT GemmaForCausalLM).
        self.model = gemma_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Forward through Gemma's transformer layers with no past_key_values
        # (fresh prefill). Request use_cache=True so present K/V are returned.
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state
        # DynamicCache -> flatten to per-layer tensors
        kv = out.past_key_values
        outs = [last_hidden]
        num_layers = self.model.config.num_hidden_layers
        if hasattr(kv, "layers"):
            for i in range(num_layers):
                outs.append(kv.layers[i].keys)
                outs.append(kv.layers[i].values)
        else:
            # Legacy tuple format
            for i in range(num_layers):
                outs.append(kv[i][0])
                outs.append(kv[i][1])
        return tuple(outs)


def build_and_export_gemma_from_embeds(
    gemma_pt_dir: Path,
    out_path: Path,
) -> Path:
    """Build GemmaFromEmbeds wrapper from saved Gemma dir, export to ONNX."""
    from transformers import GemmaForCausalLM

    full = GemmaForCausalLM.from_pretrained(gemma_pt_dir).eval()
    wrapper = GemmaFromEmbeds(full.model).eval()
    num_layers = full.config.num_hidden_layers

    # Dummy inputs matching expected Gemma dims (pi0: hidden=2048)
    B, seq = 1, 16
    hidden = full.config.hidden_size
    dummy_embeds = torch.randn(B, seq, hidden, dtype=torch.float32)
    dummy_mask = torch.ones(B, seq, dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_names = ["last_hidden_state"]
    for i in range(num_layers):
        output_names += [f"present.{i}.key", f"present.{i}.value"]
    torch.onnx.export(
        wrapper,
        (dummy_embeds, dummy_mask),
        out_path,
        input_names=["inputs_embeds", "attention_mask"],
        output_names=output_names,
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
            **{f"present.{i}.{kv}": {0: "batch", 2: "seq"}
               for i in range(num_layers) for kv in ("key", "value")},
        },
        opset_version=19,
    )
    logger.info("Exported GemmaFromEmbeds: %s", out_path)
    return out_path


class GemmaExpertFromEmbeds(nn.Module):
    """Like GemmaFromEmbeds but ALSO takes past_key_values as inputs.

    Used for pi0's action expert: suffix (state + action_time) embeddings
    + prefix_kv from the backbone = full attention context.

    Export inputs:
        inputs_embeds:          [B, suffix_len, hidden]
        attention_mask:         [B, prefix_len + suffix_len]
        past_key_values.N.key:  [B, nkv, prefix_len, head_dim] per layer
        past_key_values.N.value: same
    Export outputs:
        last_hidden_state:      [B, suffix_len, hidden]
    """

    def __init__(self, gemma_model: nn.Module):
        super().__init__()
        self.model = gemma_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values_flat: torch.Tensor,
    ) -> torch.Tensor:
        # Reconstruct DynamicCache from flat tensor inputs
        from transformers.cache_utils import DynamicCache

        num_layers = self.model.config.num_hidden_layers
        assert len(past_key_values_flat) == 2 * num_layers, \
            f"expected {2*num_layers} past_kv tensors, got {len(past_key_values_flat)}"

        cache = DynamicCache()
        for i in range(num_layers):
            k = past_key_values_flat[2 * i]
            v = past_key_values_flat[2 * i + 1]
            cache.update(k, v, i)

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=False,  # don't return updated cache; only need hidden
            return_dict=True,
        )
        return out.last_hidden_state


def build_and_export_gemma_expert_from_embeds(
    expert_pt_dir: Path,
    out_path: Path,
) -> Path:
    """Export pi0's expert (gemma-300m) with inputs_embeds + past_kv inputs."""
    from transformers import GemmaForCausalLM

    full = GemmaForCausalLM.from_pretrained(expert_pt_dir).eval()
    wrapper = GemmaExpertFromEmbeds(full.model).eval()
    num_layers = full.config.num_hidden_layers
    hidden = full.config.hidden_size
    nkv = full.config.num_key_value_heads
    hd = full.config.head_dim

    # Dummy inputs: suffix of length 51 (state=1 + chunk=50), prefix_len=16
    B, suffix_len, prefix_len = 1, 51, 16
    dummy_embeds = torch.randn(B, suffix_len, hidden, dtype=torch.float32)
    dummy_mask = torch.ones(B, prefix_len + suffix_len, dtype=torch.long)
    past_inputs = []
    past_names = []
    for i in range(num_layers):
        pk = torch.randn(B, nkv, prefix_len, hd, dtype=torch.float32)
        pv = torch.randn(B, nkv, prefix_len, hd, dtype=torch.float32)
        past_inputs.extend([pk, pv])
        past_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_embeds, dummy_mask, *past_inputs),
        out_path,
        input_names=["inputs_embeds", "attention_mask", *past_names],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "suffix_len"},
            "attention_mask": {0: "batch", 1: "total_len"},
            "last_hidden_state": {0: "batch", 1: "suffix_len"},
            **{name: {0: "batch", 2: "prefix_len"} for name in past_names},
        },
        opset_version=19,
        dynamo=False,  # Gemma-family is broken under dynamo in 2026
    )
    logger.info("Exported GemmaExpertFromEmbeds: %s", out_path)
    return out_path


class Pi0ExpertStackWithPrefix(nn.Module):
    """Pi0 expert stack that consumes per-layer VLM prefix-KV.

    Unlike SmolVLA's ExpertStack which uses a small number of cross-attn
    layers, pi0's expert applies block-causal attention on EVERY layer:
    per-layer action tokens attend to (prefix_kv_i + action_kv_i).

    Inputs to `forward`:
        noisy_actions: [B, chunk_size, action_dim]
        timestep:      [B]
        position_ids:  [B, chunk_size+1] if state_emb given, else [B, chunk_size]
                       (absolute positions OFFSET by prefix_len)
        prefix_k:      [L, B, prefix_len, nkv, hd]  per-layer, RoPE-applied
        prefix_v:      [L, B, prefix_len, nkv, hd]  per-layer, no RoPE
        state_emb:     [B, 1, expert_hidden]  optional — pi0's state token,
                       embedded externally via state_proj. When provided, the
                       suffix becomes [state_token, action_token_1..chunk_size]
                       and the output is sliced to drop the state token. Matches
                       lerobot's embed_suffix at modeling_pi0.py:687-748.

    Output:
        velocity: [B, chunk_size, action_dim]
    """

    def __init__(
        self,
        layers: list,
        expert_hidden: int,
        action_dim: int,
        suffix_weights: dict,
        action_proj_weights: dict,
        final_norm_weight: torch.Tensor,
    ):
        super().__init__()
        from reflex.decompose import DecomposedRMSNorm
        self.layers = nn.ModuleList(layers)
        self.expert_hidden = expert_hidden

        self.action_in_proj = nn.Linear(action_dim, expert_hidden)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
        self.action_in_proj.weight = nn.Parameter(suffix_weights["in_w"])
        self.action_in_proj.bias = nn.Parameter(suffix_weights["in_b"])
        self.action_time_mlp_in.weight = nn.Parameter(suffix_weights["t_in_w"])
        self.action_time_mlp_in.bias = nn.Parameter(suffix_weights["t_in_b"])
        self.action_time_mlp_out.weight = nn.Parameter(suffix_weights["t_out_w"])
        self.action_time_mlp_out.bias = nn.Parameter(suffix_weights["t_out_b"])

        self.action_out_proj = nn.Linear(expert_hidden, action_dim)
        self.action_out_proj.weight = nn.Parameter(action_proj_weights["w"])
        self.action_out_proj.bias = nn.Parameter(action_proj_weights["b"])

        self.final_norm = DecomposedRMSNorm(final_norm_weight)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        state_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from reflex.models.heads.expert_stack import _sinusoidal_pos_embedding

        b, c, _ = noisy_actions.shape
        act = self.action_in_proj(noisy_actions)
        t_emb = _sinusoidal_pos_embedding(timestep, self.expert_hidden)
        t_emb = t_emb.unsqueeze(1).expand(-1, c, -1)
        action_time_emb = self.action_time_mlp_out(
            F.silu(self.action_time_mlp_in(torch.cat([act, t_emb], dim=-1)))
        )

        # When state_emb is provided, prepend it as the first suffix token
        # (matches lerobot embed_suffix at modeling_pi0.py:687-748). Caller is
        # responsible for the corresponding position_ids extension.
        if state_emb is not None:
            x = torch.cat([state_emb, action_time_emb], dim=1)
        else:
            x = action_time_emb

        for i, layer in enumerate(self.layers):
            # prefix_k[i], prefix_v[i]: [B, prefix_len, nkv, hd]
            x = layer(
                x,
                position_ids,
                prefix_k_concat=prefix_k[i],
                prefix_v_concat=prefix_v[i],
                attn_mask=attn_mask,
            )

        x = self.final_norm(x)
        # Drop the state token from the output when it was prepended; matches
        # lerobot denoise_step's `suffix_out = outputs[:, -chunk_size:]` slice.
        if state_emb is not None:
            x = x[:, -c:]
        return self.action_out_proj(x)


def build_pi0_expert_with_prefix(state_dict: dict[str, torch.Tensor]) -> tuple[Pi0ExpertStackWithPrefix, dict]:
    """Build pi0's expert stack wired for per-layer prefix-KV concat.

    Reuses the existing pi0_exporter.build_pi0_expert_stack to create individual
    layers (with correct weights), then wraps them in Pi0ExpertStackWithPrefix.

    Critical: head_dim=256 (Gemma standard, confirmed via PI0Policy layer inspection
    2026-04-17). pi0_exporter.py's default of head_dim=128 was a silent bug that
    gave nq=16/nkv=2 instead of the correct nq=8/nkv=1 per Gemma config.
    """
    from reflex.exporters.pi0_exporter import build_pi0_expert_stack

    base_stack, meta = build_pi0_expert_stack(state_dict, head_dim=256)
    # Pull layer list out of the base stack; rebuild with prefix-aware wrapper.
    layers = list(base_stack.layers)

    # Weights for suffix/time MLP + action projections from PI0_ACTION_KEYS
    from reflex.exporters.pi0_exporter import PI0_ACTION_KEYS
    suffix = {
        "in_w": state_dict[PI0_ACTION_KEYS["in_w"]],
        "in_b": state_dict[PI0_ACTION_KEYS["in_b"]],
        "t_in_w": state_dict[PI0_ACTION_KEYS["t_in_w"]],
        "t_in_b": state_dict[PI0_ACTION_KEYS["t_in_b"]],
        "t_out_w": state_dict[PI0_ACTION_KEYS["t_out_w"]],
        "t_out_b": state_dict[PI0_ACTION_KEYS["t_out_b"]],
    }
    action_proj = {
        "w": state_dict[PI0_ACTION_KEYS["out_w"]],
        "b": state_dict[PI0_ACTION_KEYS["out_b"]],
    }
    # Final norm — pi0 uses Gemma-style RMSNorm `y = x_normed * (1 + w)`,
    # so pre-transform stored weight: `w_decomposed = 1 + w_gemma`. Matches
    # the per-layer norm convention in pi0_exporter.py:148-150.
    base_prefix = "paligemma_with_expert.gemma_expert.model."
    final_norm_w_raw = state_dict.get(f"{base_prefix}norm.weight", torch.zeros(meta["expert_hidden"]))
    final_norm_w = final_norm_w_raw + 1.0

    stack = Pi0ExpertStackWithPrefix(
        layers=layers,
        expert_hidden=meta["expert_hidden"],
        action_dim=meta["action_dim"],
        suffix_weights=suffix,
        action_proj_weights=action_proj,
        final_norm_weight=final_norm_w,
    )
    stack.eval()
    return stack, meta


class MultiModalProjector(nn.Module):
    """PaliGemma's SigLIP→Gemma linear projection (1152→2048)."""

    def __init__(self, in_dim: int = 1152, out_dim: int = 2048):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_projector(state: dict[str, torch.Tensor]) -> MultiModalProjector:
    """Extract PaliGemma's multi_modal_projector (Linear layer)."""
    w = state[PI0_PROJECTOR_KEYS["weight"]]
    b = state[PI0_PROJECTOR_KEYS["bias"]]
    in_dim, out_dim = w.shape[1], w.shape[0]
    proj = MultiModalProjector(in_dim=in_dim, out_dim=out_dim)
    proj.linear.weight.data.copy_(w)
    proj.linear.bias.data.copy_(b)
    proj.eval()
    logger.info("Projector: %d -> %d", in_dim, out_dim)
    return proj


def export_projector_onnx(projector: MultiModalProjector, out_path: Path) -> Path:
    """Export multi_modal_projector to ONNX (trivial — single Linear)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 256, projector.linear.in_features, dtype=torch.float32)
    torch.onnx.export(
        projector,
        dummy,
        out_path,
        input_names=["vision_features"],
        output_names=["projected_features"],
        dynamic_axes={"vision_features": {0: "batch", 1: "seq"}, "projected_features": {0: "batch", 1: "seq"}},
        opset_version=19,
    )
    logger.info("Exported projector ONNX: %s", out_path)
    return out_path


class EmbedTokens(nn.Module):
    """Standalone text embedding layer (tied with lm_head, vocab × hidden)."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


def build_embed_tokens(state: dict[str, torch.Tensor]) -> EmbedTokens:
    """Extract PaliGemma's tied lm_head as standalone embed_tokens."""
    w = state[PI0_LM_HEAD_KEY]
    vocab_size, hidden_size = w.shape
    emb = EmbedTokens(vocab_size, hidden_size)
    emb.embed.weight.data.copy_(w)
    emb.eval()
    logger.info("EmbedTokens: vocab=%d, hidden=%d", vocab_size, hidden_size)
    return emb


def export_embed_tokens_onnx(emb: EmbedTokens, out_path: Path) -> Path:
    """Export embed_tokens as ONNX."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randint(0, emb.embed.num_embeddings, (1, 16), dtype=torch.long)
    torch.onnx.export(
        emb,
        dummy,
        out_path,
        input_names=["input_ids"],
        output_names=["text_features"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "text_features": {0: "batch", 1: "seq"}},
        opset_version=19,
    )
    logger.info("Exported embed_tokens ONNX: %s", out_path)
    return out_path


def optimum_export_onnx(model_dir: Path, task: str, out_dir: Path) -> Path:
    """Run `optimum-cli export onnx` on a pre-built HF model dir.

    Uses legacy tracer (dynamo=False internally, which is Optimum's default)
    per the 2026 Gemma ONNX gotchas (see
    https://github.com/FastCrest/reflex-vault/blob/main/reflex_vla/03_research/pi0_onnx_importable_sources.md
    Critical risk #2).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Find the venv optimum-cli (prefer repo-local venv)
    venv_cli = Path(".venv/bin/optimum-cli")
    cli = str(venv_cli) if venv_cli.exists() else "optimum-cli"
    cmd = [
        cli, "export", "onnx",
        "--model", str(model_dir),
        "--task", task,
        "--framework", "pt",
        str(out_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    onnx_path = out_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Optimum export did not produce {onnx_path}")
    return onnx_path


def export_pi0_prefix(
    output_dir: Path,
    model_id: str = "lerobot/pi0_base",
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    skip_gemma: bool = False,
    skip_siglip: bool = False,
) -> dict[str, Any]:
    """Full pi0 prefix-KV export pipeline.

    Args:
        output_dir: where to write the ONNX bundle.
        model_id: HF model id for pi0 weights.
        state_dict: optionally pre-loaded state (saves re-reading 14GB).
        skip_gemma / skip_siglip: skip the Optimum exports (useful during
            iteration when they're already cached).

    Returns:
        Dict with 'files' (mapping of component → ONNX path) and 'metadata'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if state_dict is None:
        state_dict = load_pi0_state_dict(model_id)

    result: dict[str, Any] = {"status": "ok", "files": {}, "metadata": {}}

    # 1. SigLIP vision encoder
    siglip_pt_dir = output_dir / "_pt" / "siglip"
    if not skip_siglip or not siglip_pt_dir.exists():
        logger.info("Building SigLIP vision tower dir...")
        build_siglip_dir(state_dict, siglip_pt_dir)
    siglip_onnx_dir = output_dir / "vision_encoder"
    if not (siglip_onnx_dir / "model.onnx").exists():
        logger.info("Exporting SigLIP via Optimum (feature-extraction)...")
        optimum_export_onnx(siglip_pt_dir, "feature-extraction", siglip_onnx_dir)
    result["files"]["vision_encoder"] = str(siglip_onnx_dir / "model.onnx")

    # 2. Multi-modal projector (SigLIP features → Gemma hidden)
    logger.info("Extracting multi_modal_projector...")
    projector = build_projector(state_dict)
    projector_onnx = output_dir / "multi_modal_projector.onnx"
    export_projector_onnx(projector, projector_onnx)
    result["files"]["multi_modal_projector"] = str(projector_onnx)
    result["metadata"]["projector"] = {
        "in_dim": projector.linear.in_features,
        "out_dim": projector.linear.out_features,
    }

    # 3. Text embed_tokens (tied with lm_head, shared vocab)
    logger.info("Extracting embed_tokens...")
    emb = build_embed_tokens(state_dict)
    emb_onnx = output_dir / "text_embedder.onnx"
    export_embed_tokens_onnx(emb, emb_onnx)
    result["files"]["text_embedder"] = str(emb_onnx)
    result["metadata"]["text_embedder"] = {
        "vocab_size": emb.embed.num_embeddings,
        "hidden_size": emb.embed.embedding_dim,
    }

    # 4. Gemma backbone decoder WITH inputs_embeds API (pi0 fuses vision+text
    #    BEFORE the decoder; Optimum's Gemma export takes input_ids which
    #    doesn't fit. Use our GemmaFromEmbeds wrapper instead.)
    gemma_pt_dir = output_dir / "_pt" / "gemma"
    if not skip_gemma or not gemma_pt_dir.exists():
        logger.info("Building Gemma backbone dir...")
        build_gemma_dir(state_dict, gemma_pt_dir)
    gemma_onnx_dir = output_dir / "decoder_prefill"
    gemma_onnx_path = gemma_onnx_dir / "model.onnx"
    if not gemma_onnx_path.exists():
        logger.info("Exporting Gemma backbone via GemmaFromEmbeds wrapper...")
        build_and_export_gemma_from_embeds(gemma_pt_dir, gemma_onnx_path)
    result["files"]["decoder_prefill"] = str(gemma_onnx_path)

    # 5. Expert stack WITH prefix-KV concat (block-causal attention)
    #
    # pi0's expert uses block-causal attention where every layer's action
    # tokens attend to (prefix_kv_layer_i + action_kv). Our
    # Pi0ExpertStackWithPrefix handles this via ExpertGQALayer's new
    # prefix_k_concat/prefix_v_concat path.
    logger.info("Exporting expert stack (with per-layer prefix-KV concat)...")
    try:
        from reflex.exporters.onnx_export import export_module_to_onnx, optimize_onnx

        expert_stack, expert_meta = build_pi0_expert_with_prefix(state_dict)
        result["metadata"]["expert"] = expert_meta

        chunk_size = 50
        action_dim = expert_meta["action_dim"]
        n_layers = expert_meta["num_layers"]
        nkv = expert_meta["n_kv_heads"]
        hd = expert_meta["head_dim"]
        # Use a fixed prefix_len for tracing; dynamic axis exposes it at inference
        dummy_prefix_len = 16

        dummy_actions = torch.randn(1, chunk_size, action_dim)
        dummy_time = torch.tensor([0.5])
        dummy_pos = torch.arange(chunk_size).unsqueeze(0)
        dummy_prefix_k = torch.randn(n_layers, 1, dummy_prefix_len, nkv, hd)
        dummy_prefix_v = torch.randn(n_layers, 1, dummy_prefix_len, nkv, hd)

        expert_onnx = output_dir / "expert_stack.onnx"
        export_module_to_onnx(
            expert_stack,
            (dummy_actions, dummy_time, dummy_pos, dummy_prefix_k, dummy_prefix_v),
            expert_onnx,
            input_names=[
                "noisy_actions", "timestep", "position_ids",
                "prefix_k", "prefix_v",
            ],
            output_names=["velocity"],
            dynamic_axes={
                "noisy_actions": {0: "batch", 1: "chunk"},
                "timestep": {0: "batch"},
                "position_ids": {0: "batch", 1: "chunk"},
                "prefix_k": {1: "batch", 2: "prefix_len"},
                "prefix_v": {1: "batch", 2: "prefix_len"},
            },
            opset_version=19,
        )
        optimize_onnx(expert_onnx)
        result["files"]["expert_stack"] = str(expert_onnx)
        logger.info("Expert stack exported (with prefix-KV): %s", expert_onnx)
    except Exception as e:
        logger.warning("Expert export failed: %s", e)
        result["metadata"]["expert_error"] = str(e)

    # 6. Save config manifest
    config = {
        "model_id": model_id,
        "model_type": "pi0",
        "pipeline": "prefix_optimum + expert_custom",
        "components": result["files"],
        "metadata": result["metadata"],
    }
    (output_dir / "reflex_config.json").write_text(json.dumps(config, indent=2))
    result["files"]["config"] = str(output_dir / "reflex_config.json")

    logger.info("pi0 prefix export complete: %d components", len(result["files"]))
    return result


class Pi05ExpertStackWithPrefix(nn.Module):
    """Pi0.5 expert stack: AdaRMSNorm time-conditioned + per-layer VLM
    prefix-KV concat for self-attention.

    Mirrors `Pi0ExpertStackWithPrefix` but for pi0.5's specifics:

    - Suffix is **action-only** (chunk_size tokens) — NO state token. State
      info is encoded in lerobot's lang_tokens via knowledge insulation.
    - Time embedding feeds AdaRMSNorm per layer (NOT concatenated into
      suffix sequence like pi0 / SmolVLA).
    - Final norm is plain RMSNorm with `(1+w)` Gemma convention.

    Inputs to `forward`:
        noisy_actions: [B, chunk_size, action_dim]
        timestep:      [B]
        position_ids:  [B, chunk_size] (absolute, OFFSET by prefix_len)
        prefix_k:      [L, B, prefix_len, nkv, hd] per-layer, RoPE-applied
        prefix_v:      [L, B, prefix_len, nkv, hd] per-layer, no RoPE
        attn_mask:     optional bool [B, 1, chunk_size, prefix_len + chunk_size]
                       for lerobot's block attention pattern (all suffix
                       attends prefix bidirectionally; all suffix attends
                       all other suffix tokens).

    Output:
        velocity: [B, chunk_size, action_dim]
    """

    def __init__(
        self,
        layers: list,
        expert_hidden: int,
        action_dim: int,
        time_mlp_weights: dict,
        action_proj_weights: dict,
        in_proj_weights: dict,
        final_norm_weight: torch.Tensor | None = None,
        final_norm_dense_weights: dict | None = None,
    ):
        super().__init__()
        from reflex.decompose import DecomposedAdaRMSNorm, DecomposedRMSNorm
        self.layers = nn.ModuleList(layers)
        self.expert_hidden = expert_hidden

        # action_in_proj — same as pi0
        self.action_in_proj = nn.Linear(action_dim, expert_hidden)
        self.action_in_proj.weight = nn.Parameter(in_proj_weights["w"])
        self.action_in_proj.bias = nn.Parameter(in_proj_weights["b"])

        # Time MLP — separate from action (NOT concat-then-MLP like pi0)
        time_in_dim = time_mlp_weights["in_w"].shape[1]
        time_out_dim = time_mlp_weights["out_w"].shape[0]
        self.time_mlp_in = nn.Linear(time_in_dim, time_mlp_weights["in_w"].shape[0])
        self.time_mlp_in.weight = nn.Parameter(time_mlp_weights["in_w"])
        self.time_mlp_in.bias = nn.Parameter(time_mlp_weights["in_b"])
        self.time_mlp_out = nn.Linear(time_mlp_weights["out_w"].shape[1], time_out_dim)
        self.time_mlp_out.weight = nn.Parameter(time_mlp_weights["out_w"])
        self.time_mlp_out.bias = nn.Parameter(time_mlp_weights["out_b"])

        self.action_out_proj = nn.Linear(expert_hidden, action_dim)
        self.action_out_proj.weight = nn.Parameter(action_proj_weights["w"])
        self.action_out_proj.bias = nn.Parameter(action_proj_weights["b"])

        # Final norm — pi0.5 expert uses AdaRMSNorm (time-conditioned) per
        # lerobot modeling_pi05.py:524-540 (`compute_final_norms` passes
        # adarms_cond[1] to models[1].norm). Plain RMSNorm here gives bit-wrong
        # final output even though all 18 expert layers match bit-identically.
        if final_norm_dense_weights is not None:
            self.final_norm = DecomposedAdaRMSNorm(expert_hidden, time_dim=expert_hidden)
            self.final_norm.dense.weight = nn.Parameter(final_norm_dense_weights["w"])
            self.final_norm.dense.bias = nn.Parameter(final_norm_dense_weights["b"])
            self._final_norm_is_adarms = True
        else:
            # Legacy pi0 path — plain RMSNorm.
            assert final_norm_weight is not None
            self.final_norm = DecomposedRMSNorm(final_norm_weight)
            self._final_norm_is_adarms = False

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from reflex.models.heads.expert_stack import _sinusoidal_pos_embedding

        # action_in_proj — NO state token (pi0.5 uses state-in-language)
        x = self.action_in_proj(noisy_actions)

        # Time embedding: sinusoidal → time_mlp_in → silu → time_mlp_out → silu
        # Per lerobot pi0.5 modeling_pi05.py:706-710 (note: silu AFTER time_mlp_out too).
        t_emb_raw = _sinusoidal_pos_embedding(timestep, self.expert_hidden)
        t_emb = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(t_emb_raw))))

        from reflex.models.heads.expert_stack import Pi05ExpertGQALayer as _Pi05Layer
        for i, layer in enumerate(self.layers):
            _Pi05Layer.debug_layer_id = i
            x = layer(
                x, position_ids, t_emb,
                prefix_k_concat=prefix_k[i],
                prefix_v_concat=prefix_v[i],
                attn_mask=attn_mask,
            )

        # Final norm — AdaRMSNorm path for pi0.5 expert (uses time conditioning).
        if getattr(self, "_final_norm_is_adarms", False):
            x = self.final_norm(x, t_emb)
        else:
            x = self.final_norm(x)
        return self.action_out_proj(x)


def build_pi05_expert_with_prefix(state_dict: dict[str, torch.Tensor]) -> tuple[Pi05ExpertStackWithPrefix, dict]:
    """Build pi0.5's expert stack wired for per-layer prefix-KV concat.

    Reuses the existing pi0_exporter.build_pi05_expert_stack to validate
    keys + extract per-layer weights, then rebuilds with
    Pi05ExpertStackWithPrefix (prefix-aware Pi05ExpertGQALayer).
    """
    from reflex.exporters.pi0_exporter import build_pi05_expert_stack, PI05_ACTION_KEYS, PI0_EXPERT_PREFIX
    from reflex.models.heads.expert_stack import Pi05ExpertGQALayer

    # pi0.5 expert: SAME dim config as pi0 (num_heads=8, num_kv_heads=1,
    # head_dim=256) per lerobot modeling_pi05.py:320,329 (gemma_300m and
    # gemma_2b action_expert variants both use head_dim=256). The old
    # build_pi05_expert_stack default of head_dim=128 produced wrong
    # nq=16/nkv=2 split but worked in the non-prefix-concat path because
    # expert KV wasn't shape-compatible with paligemma's K (cross-attn
    # only). Prefix-concat path REQUIRES matching paligemma's [B, 1, prefix_len, 256] K.
    base_stack, meta = build_pi05_expert_stack(state_dict, head_dim=256)
    expert_hidden = meta["expert_hidden"]
    action_dim = meta["action_dim"]
    num_layers = meta["num_layers"]
    nq = meta["n_q_heads"]
    nkv = meta["n_kv_heads"]
    head_dim = 256
    inter = state_dict[f"{PI0_EXPERT_PREFIX}layers.0.mlp.gate_proj.weight"].shape[0]

    # Build the prefix-aware layers (Pi05ExpertGQALayer = AdaRMSNorm + prefix + gelu + gating)
    layers: list[Pi05ExpertGQALayer] = []
    base_prefix = PI0_EXPERT_PREFIX
    for i in range(num_layers):
        prefix = f"{base_prefix}layers.{i}"
        layer = Pi05ExpertGQALayer(expert_hidden, nq, nkv, head_dim, inter, rope_theta=10000.0)
        layer_sd = {
            "input_layernorm.dense.weight": state_dict[f"{prefix}.input_layernorm.dense.weight"],
            "input_layernorm.dense.bias": state_dict[f"{prefix}.input_layernorm.dense.bias"],
            "post_attention_layernorm.dense.weight": state_dict[f"{prefix}.post_attention_layernorm.dense.weight"],
            "post_attention_layernorm.dense.bias": state_dict[f"{prefix}.post_attention_layernorm.dense.bias"],
            "q_proj.weight": state_dict[f"{prefix}.self_attn.q_proj.weight"],
            "k_proj.weight": state_dict[f"{prefix}.self_attn.k_proj.weight"],
            "v_proj.weight": state_dict[f"{prefix}.self_attn.v_proj.weight"],
            "o_proj.weight": state_dict[f"{prefix}.self_attn.o_proj.weight"],
            "gate_proj.weight": state_dict[f"{prefix}.mlp.gate_proj.weight"],
            "up_proj.weight": state_dict[f"{prefix}.mlp.up_proj.weight"],
            "down_proj.weight": state_dict[f"{prefix}.mlp.down_proj.weight"],
        }
        layer.load_state_dict(layer_sd, strict=False)
        layers.append(layer)

    # Final norm — pi0.5 expert uses AdaRMSNorm (time-conditioned),
    # NOT plain RMSNorm. State dict has `norm.dense.weight` + `norm.dense.bias`.
    final_norm_dense_w_key = f"{base_prefix}norm.dense.weight"
    final_norm_dense_b_key = f"{base_prefix}norm.dense.bias"
    if final_norm_dense_w_key in state_dict:
        final_norm_dense_weights = {
            "w": state_dict[final_norm_dense_w_key],
            "b": state_dict[final_norm_dense_b_key],
        }
        final_norm_w = None
    else:
        # Fallback: plain RMSNorm (legacy / pi0 path).
        final_norm_key = f"{base_prefix}norm.weight"
        final_norm_w_raw = state_dict.get(final_norm_key, torch.zeros(expert_hidden))
        final_norm_w = final_norm_w_raw + 1.0
        final_norm_dense_weights = None

    stack = Pi05ExpertStackWithPrefix(
        layers=layers,
        expert_hidden=expert_hidden,
        action_dim=action_dim,
        time_mlp_weights={
            "in_w": state_dict[PI05_ACTION_KEYS["t_in_w"]],
            "in_b": state_dict[PI05_ACTION_KEYS["t_in_b"]],
            "out_w": state_dict[PI05_ACTION_KEYS["t_out_w"]],
            "out_b": state_dict[PI05_ACTION_KEYS["t_out_b"]],
        },
        action_proj_weights={
            "w": state_dict[PI05_ACTION_KEYS["out_w"]],
            "b": state_dict[PI05_ACTION_KEYS["out_b"]],
        },
        in_proj_weights={
            "w": state_dict[PI05_ACTION_KEYS["in_w"]],
            "b": state_dict[PI05_ACTION_KEYS["in_b"]],
        },
        final_norm_weight=final_norm_w,
        final_norm_dense_weights=final_norm_dense_weights,
    )
    stack.eval()
    return stack, meta


# TODO(v0.3): compose with pi0_exporter.build_pi0_expert_stack + flow matching
# host-side loop to produce end-to-end parity test.
# TODO(v0.3): write scripts/local_full_diff_pi0.py (analog of
# scripts/local_full_diff.py) that runs shared-noise PyTorch pi0 vs this
# composed ONNX pipeline end-to-end and reports cos >= 0.9999.
