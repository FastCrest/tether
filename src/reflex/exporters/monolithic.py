"""Monolithic ONNX export — the cos=1.0 verified production path.

Extracted from `scripts/modal_{smolvla,pi0}_monolithic_export.py` so it can
run locally (no Modal account required). Same set of transformers 5.x +
onnx-diagnostic patches; same torch.export + torch.onnx.export pipeline;
same post-export Where-type fix.

Usage (local):
    pip install 'reflex-vla[monolithic]'
    reflex export lerobot/smolvla_base --monolithic --output ./smol

Requires (pinned in the ``monolithic`` extra):
    lerobot==0.5.1
    transformers==5.3.0   (5.4+ has a q_length scalar regression)
    onnx-diagnostic>=0.9
    onnxscript>=0.1
    optree, scipy, num2words

The verified artifacts produced by this module match what the Modal
scripts produce. See
https://github.com/FastCrest/reflex-vault/blob/main/reflex_vla/measured_numbers.md
for the parity rows this export is responsible for.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Module-level "denoise phase" flag. apply_export_patches() installs a
# wrapped DynamicLayer.update that returns cat(past, new) WITHOUT mutating
# self.keys/values when this flag is True. The denoise_step wrappers for
# both PI0Pytorch (apply_export_patches) and PI05Pytorch
# (_apply_pi05_denoise_step_patch) toggle this flag for the duration of
# the expert pass, so the cache stays at prefix size across unrolled Euler
# iterations under torch.export. Pre-2026-04-30 this lived as a closure
# variable inside apply_export_patches, accessible only to the pi0 wrapper —
# pi0.5 fell through to the original mutating update, growing the cache
# 968 -> 1018 -> 1068 -> ... across 10 iterations and pinning baked /
# per-step parity at cos = 0.018 (per-step doesn't grow cache between
# Python-loop calls). See 03_experiments/2026-04-30-per-step-parity-modal-a100.md.
_denoise_phase: list[bool] = [False]


_CCM_NONE_RATIONALE = (
    "The `create_causal_mask -> None` shim is load-bearing for num_steps>1. "
    "Transformers 5.3 rebuilds a prefix-only [1,1,Q,past_len] mask that fails "
    "to broadcast against the actual [1,H,Q,past_len+suffix_len] attention "
    "scores under torch.export FakeTensor tracing (the `835 -> 886` expand "
    "error). Skipping the rebuild unblocks export. For SmolVLA (SmolLM2 "
    "attention path) this has NO semantic effect — cos=1.0 preserved. For "
    "pi0 (PaliGemma + Gemma) this skips prefix-pad masking -> cos=0.977 at "
    "num_steps=10. See 01_architecture/pi0_monolithic_wrap_pattern.md."
)

_TOKENIZER_REFS = {
    "pi0": "google/paligemma-3b-pt-224",
    "pi05": "google/paligemma-3b-pt-224",
    "smolvla": "HuggingFaceTB/SmolLM2-135M",
}


def _quiet_noisy_export_loggers() -> None:
    """Keep streamed export logs readable while preserving warnings/errors."""
    for name in ("onnx_ir", "onnxscript"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _bundle_tokenizer(output_dir: Path, model_type: str) -> dict[str, Any]:
    """Best-effort tokenizer bundle for offline serve/bench.

    Tokenizer failure should not fail ONNX export: pi0's canonical PaliGemma
    tokenizer may be gated on HF, while the model graph itself can still export
    and serve pre-tokenized requests. When bundling succeeds, runtimes load from
    ``export_dir/tokenizer`` with ``local_files_only=True``.
    """
    tokenizer_ref = _TOKENIZER_REFS.get(model_type)
    meta: dict[str, Any] = {
        "tokenizer_ref": tokenizer_ref,
        "tokenizer_path": None,
        "tokenizer_bundled": False,
        "tokenizer_error": None,
    }
    if not tokenizer_ref:
        return meta
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_ref)
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        tok_dir = output_dir / "tokenizer"
        tok.save_pretrained(tok_dir)
        meta.update(
            {
                "tokenizer_path": "tokenizer",
                "tokenizer_bundled": True,
            }
        )
    except Exception as exc:
        meta["tokenizer_error"] = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "[export] tokenizer bundle skipped for %s (%s): %s",
            model_type,
            tokenizer_ref,
            exc,
        )
    return meta


def _require_monolithic_deps() -> None:
    """Check that the ``[monolithic]`` optional dep group is installed.

    Raises ImportError with a clean message if anything's missing or a
    transformers version mismatch is detected (5.4+ has the q_length bug).
    """
    _quiet_noisy_export_loggers()
    missing = []
    try:
        import transformers
        if transformers.__version__ != "5.3.0":
            raise ImportError(
                f"transformers {transformers.__version__} detected; the "
                f"monolithic export requires exactly 5.3.0 (5.4+ has a "
                f"q_length regression in masking_utils.sdpa_mask). Install "
                f"with: pip install transformers==5.3.0"
            )
    except ImportError as e:
        missing.append(f"transformers==5.3.0 ({e})")

    for mod_name, pip_name in [
        ("lerobot", "lerobot==0.5.1"),
        ("onnx_diagnostic", "onnx-diagnostic>=0.9"),
        ("onnxscript", "onnxscript>=0.1"),
        ("optree", "optree"),
        ("scipy", "scipy"),
    ]:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        raise ImportError(
            "Missing dependencies for monolithic export:\n  - "
            + "\n  - ".join(missing)
            + "\n\nInstall with: pip install 'reflex-vla[monolithic]'"
        )


def apply_export_patches() -> None:
    """Install the full set of transformers + lerobot patches required for
    the monolithic `torch.export` path. Safe to call multiple times; later
    calls stack on earlier ones since we read the previously-monkey-patched
    function each time.
    """
    import copy
    import sys
    import types

    import torch

    # Stub GR00T imports to avoid Python 3.13 dataclass issue (harmless on 3.12 too)
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        if _mod not in sys.modules:
            stub = types.ModuleType(_mod)
            stub.GrootPolicy = None
            stub.GR00TN15 = None
            sys.modules[_mod] = stub

    # MambaCache stub for onnx-diagnostic compat with transformers 5.x
    try:
        from transformers.models.mamba.modeling_mamba import MambaCache  # noqa
    except ImportError:
        import transformers.cache_utils as _cu
        if not hasattr(_cu, "MambaCache"):
            class _MambaCacheStub:
                pass
            _cu.MambaCache = _MambaCacheStub
            try:
                import transformers.models.mamba.modeling_mamba as _mm
                if not hasattr(_mm, "MambaCache"):
                    _mm.MambaCache = _MambaCacheStub
            except ImportError:
                pass

    # create_causal_mask -> None. Bypasses the mask rebuild that would
    # trigger the 835->886 broadcast error under torch.export FakeTensor
    # tracing with num_steps>1.
    #
    # Semantic impact is backbone-specific:
    # - SmolVLA (SmolLM2 path): free, cos=1.0 preserved at machine precision.
    # - pi0 (PaliGemma + Gemma): skips prefix-pad masking, cos drops to
    #   ~0.977 at num_steps=10. v0.3 fix requires patching Gemma's inner
    #   attention (not create_causal_mask) — a 2026-04-19 investigation
    #   confirmed the 4D mask pi0 builds ([1,1,51,886]) is already
    #   [1,1,51,835] by the time it reaches create_causal_mask under
    #   tracing. The torch.cat of prefix+suffix masks doesn't survive
    #   FakeTensor propagation. See 01_architecture/pi0_monolithic_wrap_pattern.md
    from transformers import masking_utils

    def _ccm_shim(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return None

    masking_utils.create_causal_mask = _ccm_shim
    try:
        from lerobot.policies import pi_gemma as _pg
        if hasattr(_pg, "create_causal_mask"):
            _pg.create_causal_mask = _ccm_shim
    except ImportError:
        pass

    # SmolVLM's vision tower builds patch_attention_mask internally when it is
    # omitted. Under torch.export + onnx-diagnostic FakeTensor tracing that
    # internal path can produce a zero-length attention mask which later tries
    # to broadcast against the real 1024-token vision sequence. LeRobot's
    # SmolVLA always feeds dense, square images in this export path, so passing
    # the equivalent all-true patch mask explicitly preserves semantics and
    # keeps the mask shape concrete for export.
    try:
        from lerobot.policies.smolvla import modeling_smolvla as _smolvla

        if not getattr(_smolvla.SmolVLMWithExpertModel.embed_image, "_reflex_patched", False):
            def _embed_image_with_explicit_patch_mask(self, image):
                patch_size = self.get_vlm_model().vision_model.patch_size
                patch_attention_mask = torch.ones(
                    (
                        image.shape[0],
                        image.shape[-2] // patch_size,
                        image.shape[-1] // patch_size,
                    ),
                    dtype=torch.bool,
                    device=image.device,
                )
                image_hidden_states = (
                    self.get_vlm_model()
                    .vision_model(
                        pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                        patch_attention_mask=patch_attention_mask,
                    )
                    .last_hidden_state
                )
                return self.get_vlm_model().connector(image_hidden_states)

            _embed_image_with_explicit_patch_mask._reflex_patched = True
            _smolvla.SmolVLMWithExpertModel.embed_image = _embed_image_with_explicit_patch_mask
    except Exception as exc:
        logger.debug("SmolVLA explicit patch-mask export patch not installed: %s", exc)

    try:
        from transformers.models.smolvlm import modeling_smolvlm as _smolvlm

        if not getattr(_smolvlm.SmolVLMVisionAttention.forward, "_reflex_patched", False):
            def _vision_attention_forward_no_dense_mask(self, hidden_states, attention_mask=None, **kwargs):
                batch_size, seq_length, embed_dim = hidden_states.shape

                queries = self.q_proj(hidden_states)
                keys = self.k_proj(hidden_states)
                values = self.v_proj(hidden_states)

                queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
                keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
                values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

                attention_interface = _smolvlm.ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation,
                    _smolvlm.eager_attention_forward,
                )
                attn_output, attn_weights = attention_interface(
                    self,
                    queries,
                    keys,
                    values,
                    None,
                    is_causal=False,
                    scaling=self.scale,
                    dropout=0.0 if not self.training else self.dropout,
                )

                attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
                attn_output = self.out_proj(attn_output)
                return attn_output, attn_weights

            _vision_attention_forward_no_dense_mask._reflex_patched = True
            _smolvlm.SmolVLMVisionAttention.forward = _vision_attention_forward_no_dense_mask
    except Exception as exc:
        logger.debug("SmolVLM vision attention export patch not installed: %s", exc)

    # DynamicCache deepcopy bypass (FakeTensor can't be deepcopied)
    _orig_deepcopy = copy.deepcopy

    def _safe_deepcopy(obj, *args, **kwargs):
        from transformers.cache_utils import DynamicCache
        if isinstance(obj, DynamicCache):
            return obj
        return _orig_deepcopy(obj, *args, **kwargs)
    copy.deepcopy = _safe_deepcopy

    # SmolVLA pad_tensor/pad_vector: replace slice-assign with torch.cat so
    # aten::index_put_ doesn't lower to ONNX Where(bool, int64, float).
    try:
        from lerobot.policies.smolvla import modeling_smolvla as _smv

        def _safe_pad_tensor(tensor, max_len, pad_value=0):
            b, d = tensor.shape[:2]
            if d >= max_len:
                return tensor[:, :max_len]
            pad_shape = (b, max_len - d, *tensor.shape[2:])
            pad = torch.full(
                pad_shape, float(pad_value),
                dtype=tensor.dtype, device=tensor.device,
            )
            return torch.cat([tensor, pad], dim=1)
        _smv.pad_tensor = _safe_pad_tensor

        def _safe_pad_vector(vector, new_dim):
            current_dim = vector.shape[-1]
            if current_dim >= new_dim:
                return vector[..., :new_dim]
            pad_shape = (*vector.shape[:-1], new_dim - current_dim)
            pad = torch.zeros(pad_shape, dtype=vector.dtype, device=vector.device)
            return torch.cat([vector, pad], dim=-1)
        _smv.pad_vector = _safe_pad_vector
    except ImportError:
        pass

    # torch.where dtype coercion (SmolVLM2 vision embed does
    # torch.where(bool, int64_full, float32) which ONNX rejects)
    _orig_where = torch.where

    def _safe_where(condition, x=None, y=None, *args, **kwargs):
        if (
            x is not None and y is not None
            and hasattr(x, "dtype") and hasattr(y, "dtype")
            and x.dtype != y.dtype
        ):
            common = torch.promote_types(x.dtype, y.dtype)
            if x.dtype != common:
                x = x.to(common)
            if y.dtype != common:
                y = y.to(common)
        return _orig_where(condition, x, y, *args, **kwargs)
    torch.where = _safe_where

    # pi0-specific: PaliGemmaWithExpertModel.embed_image image_outputs extraction
    try:
        from lerobot.policies.pi0 import modeling_pi0
        import torch as _t

        def _patched_embed_image(self, image):
            out_dtype = image.dtype
            if image.dtype != _t.float32:
                image = image.to(_t.float32)
            out = self.paligemma.model.get_image_features(image)
            features = out.pooler_output if hasattr(out, "pooler_output") else out
            features = features * self.paligemma.config.text_config.hidden_size ** 0.5
            if features.dtype != out_dtype:
                features = features.to(out_dtype)
            return features
        modeling_pi0.PaliGemmaWithExpertModel.embed_image = _patched_embed_image
    except ImportError:
        pass

    # ===============================================================
    # pi0 cos=1.0 at num_steps=10 — the full fix stack (2026-04-19)
    # ===============================================================
    # Three interacting issues, three surgical patches:
    #
    # 1. `torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)` inside
    #    denoise_step loses its suffix dim under FakeTensor tracing — the
    #    cat'd output arrives at attention as [1,1,51,835] instead of the
    #    intended [1,1,51,886]. Replace with F.pad + logical AND, which has
    #    concrete output sizes at every step.
    #
    # 2. `DynamicLayer.update` (transformers 5.3) unconditionally APPENDS
    #    new K/V to self.keys/self.values regardless of use_cache=False.
    #    Under eager PyTorch this mutation is inert between iterations
    #    (the cache object is re-passed but not re-used). Under
    #    torch.export tracing the SAME object is seen across unrolled
    #    iterations → cache grows (784 → 835 → 886 → ...), each iter's
    #    attention sees a different K dim than the canonical eager path.
    #    Fix: wrap DynamicLayer.update to return `cat(past, new)` WITHOUT
    #    appending when a "denoise-phase" flag is set. Prefix forward runs
    #    WITHOUT the flag (cache populates normally); Euler loop runs WITH
    #    the flag (cache frozen at prefix size). Matches canonical semantics.
    #
    # 3. Use `past_key_values.get_seq_length()` (not `prefix_pad_masks.shape[1]`)
    #    for mask construction — the two can diverge under tracing.
    #
    # Result: pi0 num_steps=10 monolithic ONNX matches canonical PyTorch
    # sample_actions(num_steps=10) at cos=1.0, max_abs ~2e-07 (float32
    # precision floor).
    try:
        import torch.nn.functional as _F
        from transformers.cache_utils import DynamicLayer as _DL
        from lerobot.policies.pi0 import modeling_pi0 as _mp0
        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks as _make_att_2d_masks

        _orig_layer_update = _DL.update

        def _frozen_layer_update(self, key_states, value_states, cache_kwargs=None):
            if _denoise_phase[0] and getattr(self, "is_initialized", False):
                past_k = self.keys
                past_v = self.values
                if past_k is not None and past_v is not None:
                    new_k = torch.cat([past_k, key_states], dim=-2)
                    new_v = torch.cat([past_v, value_states], dim=-2)
                    return new_k, new_v
            return _orig_layer_update(self, key_states, value_states, cache_kwargs)
        _DL.update = _frozen_layer_update

        def _patched_denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, x_t, timestep,
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            cached_prefix_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            prefix_len = max(cached_prefix_len, prefix_pad_masks.shape[1])

            pad_deficit = prefix_len - prefix_pad_masks.shape[1]
            prefix_pad_masks_extended = prefix_pad_masks
            if pad_deficit > 0:
                prefix_pad_masks_extended = _F.pad(prefix_pad_masks, (0, pad_deficit), value=True)

            prefix_pad_2d_masks = prefix_pad_masks_extended[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

            prefix_allowed = _F.pad(prefix_pad_2d_masks, (0, suffix_len), value=True)
            suffix_allowed = _F.pad(suffix_att_2d_masks, (prefix_len, 0), value=True)
            full_att_2d_masks = prefix_allowed & suffix_allowed

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            return self.action_out_proj(suffix_out)

        _dstep_inner = _patched_denoise_step

        def _denoise_step_with_flag(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
            _denoise_phase[0] = True
            try:
                return _dstep_inner(self, state, prefix_pad_masks, past_key_values, x_t, timestep)
            finally:
                _denoise_phase[0] = False

        _mp0.PI0Pytorch.denoise_step = _denoise_step_with_flag
    except ImportError:
        pass


def _force_eager_attn(model: Any) -> None:
    """Force every module's `_attn_implementation` to 'eager' — onnx-diagnostic's
    sdpa_mask patch crashes with None cache_position under transformers 5.x."""
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "eager"
    if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"


def _fix_onnx_where_dtype_mismatches(onnx_path: Path) -> int:
    """Post-export pass: torch.onnx.export sometimes lowers aten::index_put to
    ONNX Where(bool, int64, float32). onnxruntime rejects that at load time.
    Walk the graph, find such Where ops, and insert Cast nodes that coerce
    the mismatched branch to the declared output dtype.

    Returns count of fixes applied.
    """
    import onnx
    from onnx import helper, TensorProto

    model = onnx.load(str(onnx_path), load_external_data=False)
    shape_info = onnx.shape_inference.infer_shapes(
        model, check_type=False, strict_mode=False
    )

    name_dtype: dict[str, int] = {}
    for vi in list(shape_info.graph.value_info) + list(shape_info.graph.input) + list(shape_info.graph.output):
        if vi.type.tensor_type.elem_type:
            name_dtype[vi.name] = vi.type.tensor_type.elem_type
    for init in shape_info.graph.initializer:
        name_dtype[init.name] = init.data_type

    FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE}
    INT_TYPES = {TensorProto.INT64, TensorProto.INT32, TensorProto.INT16, TensorProto.INT8}

    fixes = 0
    new_nodes: list = []
    for node in model.graph.node:
        if node.op_type != "Where":
            new_nodes.append(node)
            continue
        x_name, y_name = node.input[1], node.input[2]
        out_name = node.output[0]
        x_dt = name_dtype.get(x_name)
        y_dt = name_dtype.get(y_name)
        out_dt = name_dtype.get(out_name)
        if not x_dt or not y_dt or x_dt == y_dt:
            new_nodes.append(node)
            continue
        if out_dt and out_dt in (x_dt, y_dt):
            target_dt = out_dt
        else:
            target_dt = x_dt if x_dt in INT_TYPES else y_dt
        fixes_this = []
        if x_dt != target_dt:
            cast_out = f"{node.name}__cast_x"
            cast_node = helper.make_node(
                "Cast", [x_name], [cast_out],
                name=f"{node.name}__cast_x_node", to=target_dt,
            )
            fixes_this.append(cast_node)
            node.input[1] = cast_out
        if y_dt != target_dt:
            cast_out = f"{node.name}__cast_y"
            cast_node = helper.make_node(
                "Cast", [y_name], [cast_out],
                name=f"{node.name}__cast_y_node", to=target_dt,
            )
            fixes_this.append(cast_node)
            node.input[2] = cast_out
        new_nodes.extend(fixes_this)
        new_nodes.append(node)
        fixes += 1
        logger.info(
            "fixed Where node %s (X=%s, Y=%s, out_decl=%s -> target=%s)",
            node.name, x_dt, y_dt, out_dt, target_dt,
        )

    if fixes > 0:
        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)
        with open(onnx_path, "wb") as f:
            f.write(model.SerializeToString())

    return fixes


def export_smolvla_monolithic(
    model_id: str,
    output_dir: str | Path,
    *,
    num_steps: int = 10,
    target: str = "desktop",
) -> dict[str, Any]:
    """Export SmolVLA as a single monolithic ONNX.

    Parity: cos=1.0 at num_steps=10 vs PyTorch sample_actions(num_steps=10).

    Args:
        model_id: HuggingFace model ID (e.g. ``"lerobot/smolvla_base"``)
        output_dir: where to write ``model.onnx`` + external data + config
        num_steps: number of Euler integration steps baked into the graph.
            Canonical flow-matching = 10. Use 1 only if you want the exact
            one-shot Euler contract.

    Returns:
        ``{"status": "ok", "onnx_path": str, "size_mb": float}``
    """
    _require_monolithic_deps()

    import torch
    import torch.nn as nn
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    apply_export_patches()

    logger.info("[smolvla] Loading %s", model_id)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.eval().to("cpu").to(torch.float32)
    policy.model.config.num_steps = num_steps
    _force_eager_attn(policy.model)
    logger.info("[smolvla] Loaded in %.1fs; num_steps=%d", time.time() - t0, num_steps)

    class SmolVLAMonolithicWrapper(nn.Module):
        def __init__(self, smolvla_model):
            super().__init__()
            self.model = smolvla_model

        def forward(
            self,
            img_cam1, img_cam2, img_cam3,
            mask_cam1, mask_cam2, mask_cam3,
            lang_tokens, lang_masks,
            state, noise,
        ):
            images = [img_cam1, img_cam2, img_cam3]
            img_masks = [mask_cam1, mask_cam2, mask_cam3]
            return self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise,
            )

    wrapper = SmolVLAMonolithicWrapper(policy.model).eval()
    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    dummy = dict(
        img_cam1=torch.randn(B, 3, 512, 512, dtype=torch.float32),
        img_cam2=torch.randn(B, 3, 512, 512, dtype=torch.float32),
        img_cam3=torch.randn(B, 3, 512, 512, dtype=torch.float32),
        mask_cam1=torch.ones(B, dtype=torch.bool),
        mask_cam2=torch.ones(B, dtype=torch.bool),
        mask_cam3=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 49152, (B, 16), dtype=torch.long),
        lang_masks=torch.ones(B, 16, dtype=torch.bool),
        state=torch.randn(B, state_dim, dtype=torch.float32),
        noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info("[smolvla] torch.export.export ...")
    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(
            wrapper, tuple(dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[smolvla] torch.export: %.1fs", time.time() - t0)

    logger.info("[smolvla] torch.onnx.export ...")
    t0 = time.time()
    torch.onnx.export(
        ep, tuple(dummy.values()), str(onnx_path),
        input_names=list(dummy.keys()), output_names=["actions"],
        opset_version=19,
    )
    logger.info("[smolvla] ONNX conversion: %.1fs", time.time() - t0)

    fixes = _fix_onnx_where_dtype_mismatches(onnx_path)
    logger.info("[smolvla] post-export Cast fixes: %d", fixes)

    _write_reflex_config(
        output_dir, policy.config, num_steps=num_steps,
        model_id=model_id, model_type="smolvla", target=target,
    )

    size_mb = onnx_path.stat().st_size / 1e6
    data_files = list(output_dir.glob("*.data"))
    total_mb = sum(f.stat().st_size for f in data_files) / 1e6 + size_mb
    return {
        "status": "ok",
        "onnx_path": str(onnx_path),
        "size_mb": total_mb,
        "num_steps": num_steps,
    }


def export_pi0_monolithic(
    model_id: str,
    output_dir: str | Path,
    *,
    num_steps: int = 10,
    target: str = "desktop",
) -> dict[str, Any]:
    """Export pi0 as a single monolithic ONNX.

    Parity: cos=1.0 at num_steps=1, cos=0.977 at num_steps=10 (the ccm=None
    shim skips prefix-pad masking on pi0's PaliGemma path — v0.3 fix tracked).

    Args, returns: same shape as ``export_smolvla_monolithic``.
    """
    _require_monolithic_deps()

    import torch
    import torch.nn as nn
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    apply_export_patches()

    logger.info("[pi0] Loading %s", model_id)
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    t0 = time.time()
    # Block torch.compile during from_pretrained — see pi05 branch for the
    # "Guard failed on same frame" bug. pi0_base defaults to False so this
    # is a no-op for canonical exports, but any fine-tuned pi0 config with
    # compile_model=True (common pattern) would otherwise break the trace.
    _orig_compile = torch.compile
    torch.compile = lambda fn=None, *a, **kw: (fn if fn is not None else (lambda f: f))
    try:
        policy = PI0Policy.from_pretrained(model_id)
    finally:
        torch.compile = _orig_compile
    policy.eval().to("cpu").to(torch.float32)
    _gc_disable = getattr(policy.model, "gradient_checkpointing_disable", None)
    if callable(_gc_disable):
        _gc_disable()
    _force_eager_attn(policy.model)
    logger.info("[pi0] Loaded in %.1fs", time.time() - t0)

    class Pi0MonolithicWrapper(nn.Module):
        def __init__(self, pi0_model, n_steps):
            super().__init__()
            self.model = pi0_model
            self.n_steps = n_steps

        def forward(
            self,
            img_base, img_wrist_l, img_wrist_r,
            mask_base, mask_wrist_l, mask_wrist_r,
            lang_tokens, lang_masks,
            state, noise,
        ):
            images = [img_base, img_wrist_l, img_wrist_r]
            img_masks = [mask_base, mask_wrist_l, mask_wrist_r]
            return self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, num_steps=self.n_steps,
            )

    wrapper = Pi0MonolithicWrapper(policy.model, num_steps).eval()
    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    dummy = dict(
        img_base=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_l=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_r=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        mask_base=torch.ones(B, dtype=torch.bool),
        mask_wrist_l=torch.ones(B, dtype=torch.bool),
        mask_wrist_r=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 257152, (B, 16), dtype=torch.long),
        lang_masks=torch.ones(B, 16, dtype=torch.bool),
        state=torch.randn(B, state_dim, dtype=torch.float32),
        noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info("[pi0] torch.export.export ...")
    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(
            wrapper, tuple(dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[pi0] torch.export: %.1fs", time.time() - t0)

    logger.info("[pi0] torch.onnx.export ...")
    t0 = time.time()
    torch.onnx.export(
        ep, tuple(dummy.values()), str(onnx_path),
        input_names=list(dummy.keys()), output_names=["actions"],
        opset_version=19,
    )
    logger.info("[pi0] ONNX conversion: %.1fs", time.time() - t0)

    fixes = _fix_onnx_where_dtype_mismatches(onnx_path)
    logger.info("[pi0] post-export Cast fixes: %d", fixes)

    _write_reflex_config(
        output_dir, policy.config, num_steps=num_steps,
        model_id=model_id, model_type="pi0", target=target,
    )

    size_mb = onnx_path.stat().st_size / 1e6
    data_files = list(output_dir.glob("*.data"))
    total_mb = sum(f.stat().st_size for f in data_files) / 1e6 + size_mb
    return {
        "status": "ok",
        "onnx_path": str(onnx_path),
        "size_mb": total_mb,
        "num_steps": num_steps,
    }


def _write_reflex_config(
    output_dir: Path,
    policy_config: Any,
    *,
    num_steps: int,
    model_id: str,
    model_type: str,
    target: str = "desktop",
) -> None:
    """Write `reflex_config.json` + seed a VERIFICATION.md receipt."""
    tokenizer_meta = _bundle_tokenizer(output_dir, model_type)
    cfg_dict = {
        "model_id": model_id,
        "model_type": model_type,
        "target": target,
        "num_denoising_steps": num_steps,
        "chunk_size": getattr(policy_config, "chunk_size", 50),
        "action_chunk_size": getattr(policy_config, "chunk_size", 50),
        "action_dim": getattr(policy_config, "max_action_dim", 32),
        "max_state_dim": getattr(policy_config, "max_state_dim", 32),
        "opset": 19,
        "export_kind": "monolithic",
        "notes": _CCM_NONE_RATIONALE if num_steps > 1 else None,
        **tokenizer_meta,
    }
    (output_dir / "reflex_config.json").write_text(
        json.dumps(cfg_dict, indent=2, default=str)
    )

    # Seed VERIFICATION.md at export time — every export ships a receipt,
    # even before `reflex validate` fills in the parity numbers. Callers
    # that run validate afterward will overwrite with verified max_abs.
    # Wrapped in try/except so a report-writing error can't fail the
    # actual export (the ONNX is already on disk by this point).
    try:
        from reflex.verification_report import write_verification_report
        write_verification_report(output_dir, parity=None)
    except Exception:
        pass


def export_pi05_monolithic(
    model_id: str,
    output_dir: str | Path,
    *,
    num_steps: int = 10,
    target: str = "desktop",
) -> dict[str, Any]:
    """Export pi0.5 as a single monolithic ONNX.

    Structurally identical wrap to pi0 — the only deltas are: (1) no
    state arg (state is tokenized into language), (2) PI05Pytorch class.
    The three-patch stack (F.pad mask, frozen DynamicLayer.update,
    past_kv.seq_length for mask) is applied by apply_export_patches(),
    so pi0.5 inherits the cos=1.0 fix for free.

    Parity: cos=1.0, max_abs ~2.38e-07 at num_steps=10 vs PyTorch
    sample_actions(num_steps=10).
    """
    _require_monolithic_deps()

    import torch
    import torch.nn as nn
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    apply_export_patches()

    logger.info("[pi05] Loading %s", model_id)
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    t0 = time.time()
    # Block torch.compile during from_pretrained: LIBERO-finetuned configs
    # (e.g. lerobot/pi05_libero_finetuned_v044) ship with compile_model=True,
    # which wraps sample_actions + forward in torch.compile at __init__ time.
    # torch.compile wrappers collide with torch.export's dynamo tracer and
    # raise "Guard failed on same frame it was created" during export. Disable
    # the compile step for the duration of the load.
    _orig_compile = torch.compile
    torch.compile = lambda fn=None, *a, **kw: (fn if fn is not None else (lambda f: f))
    try:
        policy = PI05Policy.from_pretrained(model_id)
    finally:
        torch.compile = _orig_compile
    policy.eval().to("cpu").to(torch.float32)
    # Same gradient_checkpointing-on-by-default issue we hit in SnapFlow
    # training on pi05 LIBERO teacher: leaves the gemma stack in a state
    # where use_cache=True gets flipped, breaking past_kv construction.
    _gc_disable = getattr(policy.model, "gradient_checkpointing_disable", None)
    if callable(_gc_disable):
        _gc_disable()
    _force_eager_attn(policy.model)

    # Apply the same PI05Pytorch-specific denoise_step patch (monolithic.py's
    # shared patcher patches PI0Pytorch; pi0.5 needs the equivalent on
    # PI05Pytorch). Reuses the frozen-cache DynamicLayer.update from the
    # shared patcher.
    _apply_pi05_denoise_step_patch()

    logger.info("[pi05] Loaded in %.1fs", time.time() - t0)

    class Pi05MonolithicWrapper(nn.Module):
        def __init__(self, pi05_model, n_steps):
            super().__init__()
            self.model = pi05_model
            self.n_steps = n_steps

        def forward(
            self,
            img_base, img_wrist_l, img_wrist_r,
            mask_base, mask_wrist_l, mask_wrist_r,
            lang_tokens, lang_masks,
            noise,
        ):
            # pi0.5: no state arg (state is in lang_tokens).
            images = [img_base, img_wrist_l, img_wrist_r]
            img_masks = [mask_base, mask_wrist_l, mask_wrist_r]
            return self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks,
                noise=noise, num_steps=self.n_steps,
            )

    wrapper = Pi05MonolithicWrapper(policy.model, num_steps).eval()
    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim

    dummy = dict(
        img_base=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_l=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_r=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        mask_base=torch.ones(B, dtype=torch.bool),
        mask_wrist_l=torch.ones(B, dtype=torch.bool),
        mask_wrist_r=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 257152, (B, 16), dtype=torch.long),
        lang_masks=torch.ones(B, 16, dtype=torch.bool),
        noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info("[pi05] torch.export.export ...")
    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(
            wrapper, tuple(dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[pi05] torch.export: %.1fs", time.time() - t0)

    logger.info("[pi05] torch.onnx.export ...")
    t0 = time.time()
    torch.onnx.export(
        ep, tuple(dummy.values()), str(onnx_path),
        input_names=list(dummy.keys()), output_names=["actions"],
        opset_version=19,
    )
    logger.info("[pi05] ONNX conversion: %.1fs", time.time() - t0)

    fixes = _fix_onnx_where_dtype_mismatches(onnx_path)
    logger.info("[pi05] post-export Cast fixes: %d", fixes)

    _write_reflex_config(
        output_dir, policy.config, num_steps=num_steps,
        model_id=model_id, model_type="pi05", target=target,
    )

    size_mb = onnx_path.stat().st_size / 1e6
    data_files = list(output_dir.glob("*.data"))
    total_mb = sum(f.stat().st_size for f in data_files) / 1e6 + size_mb
    return {
        "status": "ok",
        "onnx_path": str(onnx_path),
        "size_mb": total_mb,
        "num_steps": num_steps,
    }


def _apply_pi05_denoise_step_patch() -> None:
    """Patch PI05Pytorch.denoise_step with the F.pad mask + frozen-cache
    flag. Called from export_pi05_monolithic after apply_export_patches(),
    which already installed the DynamicLayer.update freeze.

    The wrapper toggles the module-level _denoise_phase flag for the
    duration of the expert pass — same pattern as apply_export_patches'
    PI0Pytorch wrapper. Without this, pi0.5's traced cache mutates each
    Euler iteration (cache grows 968->1018->...->1418 across n=10), and
    the resulting baked ONNX no longer matches a per-step ONNX driven N
    times in Python (per-step rebuilds cache per call -> stays at 968+50).
    Pre-2026-04-30 the flag was set only for pi0; the per-step parity gate
    surfaced this asymmetry at cos=0.018 / max_abs=1.336.
    """
    try:
        import torch
        import torch.nn.functional as _F
        from lerobot.policies.pi05 import modeling_pi05 as _mp05
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks as _make

        def _patched(self, prefix_pad_masks, past_key_values, x_t, timestep):
            # pi0.5: NO state arg (tokenized into language upstream).
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                x_t, timestep,
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            cached_prefix_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            prefix_len = max(cached_prefix_len, prefix_pad_masks.shape[1])

            pad_deficit = prefix_len - prefix_pad_masks.shape[1]
            prefix_pad_masks_extended = prefix_pad_masks
            if pad_deficit > 0:
                prefix_pad_masks_extended = _F.pad(prefix_pad_masks, (0, pad_deficit), value=True)

            prefix_pad_2d_masks = prefix_pad_masks_extended[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = _make(suffix_pad_masks, suffix_att_masks)

            prefix_allowed = _F.pad(prefix_pad_2d_masks, (0, suffix_len), value=True)
            suffix_allowed = _F.pad(suffix_att_2d_masks, (prefix_len, 0), value=True)
            full_att_2d_masks = prefix_allowed & suffix_allowed

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            return self.action_out_proj(suffix_out)

        _dstep_inner_pi05 = _patched

        def _denoise_step_with_flag_pi05(self, prefix_pad_masks, past_key_values, x_t, timestep):
            _denoise_phase[0] = True
            try:
                return _dstep_inner_pi05(self, prefix_pad_masks, past_key_values, x_t, timestep)
            finally:
                _denoise_phase[0] = False

        _mp05.PI05Pytorch.denoise_step = _denoise_step_with_flag_pi05
    except ImportError:
        pass


def export_snapflow_student_monolithic(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    target: str = "desktop",
) -> dict[str, Any]:
    """Export a SnapFlow-distilled student as monolithic ONNX at 1-NFE.

    Takes a reflex-saved distill checkpoint dir (contains
    ``model.safetensors`` with the base pi0/pi0.5 weights plus the
    ``target_time_embed_mlp.*`` SnapFlow keys, config.json, processor
    files), loads it via ``load_snapflow_student``, and produces a single
    monolithic ONNX where the denoising loop is BAKED to one step with
    ``target_time=1``.

    The output ONNX has the same input/output signature as a regular
    pi0/pi0.5 monolithic ONNX (per model family), so ``reflex serve``
    can load it through the existing path without special-casing
    distilled students. At runtime, one forward pass = one full action
    chunk (no host-side Euler loop needed).

    Parity claim: the ONNX replicates ``student.model.sample_actions_1step``
    bit-exact up to the usual ONNX float32 precision floor. Student
    task-success separately measured at 29/30 on LIBERO (vs teacher
    28/30) — see the 2026-04-21 session log.
    """
    _require_monolithic_deps()

    import torch
    import torch.nn as nn
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    from reflex.distill.snapflow_pi0_model import load_snapflow_student

    apply_export_patches()

    logger.info("[snapflow-export] Loading student from %s", checkpoint_path)
    t0 = time.time()
    policy = load_snapflow_student(checkpoint_path)
    policy.eval().to("cpu").to(torch.float32)
    _force_eager_attn(policy.model)
    logger.info("[snapflow-export] Student loaded in %.1fs", time.time() - t0)

    # Dispatch by model family (pi0 vs pi0.5). SnapFlowPI05Pytorch is a
    # dynamic subclass of PI05Pytorch; isinstance catches it.
    from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    is_pi05 = isinstance(policy.model, PI05Pytorch)
    is_pi0 = isinstance(policy.model, PI0Pytorch)

    if not (is_pi05 or is_pi0):
        raise TypeError(
            f"SnapFlow export supports pi0 / pi0.5 students only; got "
            f"{type(policy.model).__name__}"
        )

    # Monkey-patch the SnapFlow denoise_step to drop detach+deepcopy for
    # export. During training we needed them for gradient isolation, but
    # at export time we trace a single forward pass with no gradients;
    # deepcopy of DynamicCache doesn't trace cleanly through torch.export.
    # Replace with a minimal version that just passes past_kv through.
    _install_snapflow_export_denoise_step(policy.model, is_pi05=is_pi05)

    family = "pi05" if is_pi05 else "pi0"
    logger.info("[snapflow-export] Family=%s, wrapping for export", family)

    if is_pi05:
        class SnapFlowStudentWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(
                self,
                img_base, img_wrist_l, img_wrist_r,
                mask_base, mask_wrist_l, mask_wrist_r,
                lang_tokens, lang_masks,
                noise,
            ):
                return self.model.sample_actions_1step(
                    [img_base, img_wrist_l, img_wrist_r],
                    [mask_base, mask_wrist_l, mask_wrist_r],
                    lang_tokens, lang_masks, noise=noise,
                )
    else:
        class SnapFlowStudentWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(
                self,
                img_cam1, img_cam2, img_cam3,
                mask_cam1, mask_cam2, mask_cam3,
                lang_tokens, lang_masks,
                state, noise,
            ):
                return self.model.sample_actions_1step(
                    [img_cam1, img_cam2, img_cam3],
                    [mask_cam1, mask_cam2, mask_cam3],
                    lang_tokens, lang_masks, state, noise=noise,
                )

    wrapper = SnapFlowStudentWrapper(policy.model).eval()
    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim

    if is_pi05:
        dummy = dict(
            img_base=torch.randn(B, 3, 224, 224, dtype=torch.float32),
            img_wrist_l=torch.randn(B, 3, 224, 224, dtype=torch.float32),
            img_wrist_r=torch.randn(B, 3, 224, 224, dtype=torch.float32),
            mask_base=torch.ones(B, dtype=torch.bool),
            mask_wrist_l=torch.ones(B, dtype=torch.bool),
            mask_wrist_r=torch.ones(B, dtype=torch.bool),
            lang_tokens=torch.randint(0, 257152, (B, 16), dtype=torch.long),
            lang_masks=torch.ones(B, 16, dtype=torch.bool),
            noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
        )
    else:
        state_dim = getattr(cfg, "max_state_dim", 32)
        dummy = dict(
            img_cam1=torch.randn(B, 3, 224, 224, dtype=torch.float32),
            img_cam2=torch.randn(B, 3, 224, 224, dtype=torch.float32),
            img_cam3=torch.randn(B, 3, 224, 224, dtype=torch.float32),
            mask_cam1=torch.ones(B, dtype=torch.bool),
            mask_cam2=torch.ones(B, dtype=torch.bool),
            mask_cam3=torch.ones(B, dtype=torch.bool),
            lang_tokens=torch.randint(0, 257152, (B, 16), dtype=torch.long),
            lang_masks=torch.ones(B, 16, dtype=torch.bool),
            state=torch.randn(B, state_dim, dtype=torch.float32),
            noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info("[snapflow-export] torch.export.export ...")
    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(
            wrapper, tuple(dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[snapflow-export] torch.export: %.1fs", time.time() - t0)

    logger.info("[snapflow-export] torch.onnx.export ...")
    t0 = time.time()
    torch.onnx.export(
        ep, tuple(dummy.values()), str(onnx_path),
        input_names=list(dummy.keys()), output_names=["actions"],
        opset_version=19,
    )
    logger.info("[snapflow-export] ONNX conversion: %.1fs", time.time() - t0)

    fixes = _fix_onnx_where_dtype_mismatches(onnx_path)
    logger.info("[snapflow-export] post-export Cast fixes: %d", fixes)

    _write_reflex_config(
        output_dir, policy.config, num_steps=1,
        model_id=str(checkpoint_path), model_type=f"{family}_snapflow",
        target=target,
    )

    size_mb = onnx_path.stat().st_size / 1e6
    data_files = list(output_dir.glob("*.data"))
    total_mb = sum(f.stat().st_size for f in data_files) / 1e6 + size_mb
    return {
        "status": "ok",
        "onnx_path": str(onnx_path),
        "size_mb": total_mb,
        "model_type": f"{family}_snapflow",
        "num_steps": 1,
        "target_time": 1.0,
    }


def _install_snapflow_export_denoise_step(model: Any, *, is_pi05: bool) -> None:
    """Replace SnapFlowPI*Pytorch.denoise_step with an export-friendly variant
    that drops the training-only detach+deepcopy (which doesn't trace
    through torch.export.export). The 1-step export path calls denoise_step
    exactly once per forward, so past_kv isn't reused across calls — the
    defensive copy that matters for multi-step inference isn't needed here.
    """
    import torch
    import torch.nn.functional as _F

    if is_pi05:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    else:
        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

    if is_pi05:
        def _export_denoise_step(self, prefix_pad_masks, past_key_values, x_t,
                                 timestep, target_time=None, state=None):
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                x_t, timestep, target_time=target_time,
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            prefix_allowed = _F.pad(prefix_pad_2d_masks, (0, suffix_len), value=True)
            suffix_allowed = _F.pad(suffix_att_2d_masks, (prefix_len, 0), value=True)
            full_att_2d_masks = prefix_allowed & suffix_allowed

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1][:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(suffix_out)
    else:
        def _export_denoise_step(self, state, prefix_pad_masks, past_key_values,
                                 x_t, timestep, target_time=None):
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, x_t, timestep, target_time=target_time,
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            prefix_allowed = _F.pad(prefix_pad_2d_masks, (0, suffix_len), value=True)
            suffix_allowed = _F.pad(suffix_att_2d_masks, (prefix_len, 0), value=True)
            full_att_2d_masks = prefix_allowed & suffix_allowed

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1][:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(suffix_out)

    # Bind onto the class (the SnapFlow subclass) so the instance picks
    # it up. This overrides the training-time detach+deepcopy variant
    # for the duration of the export process.
    type(model).denoise_step = _export_denoise_step


def export_gr00t_monolithic(
    model_id: str,
    output_dir: str | Path,
    *,
    num_steps: int = 4,
    target: str = "desktop",
    embodiment_id: int = 0,
) -> dict[str, Any]:
    """Export GR00T N1.6 as a single monolithic ONNX.

    GR00T differs from pi0/pi0.5/SmolVLA: it's a DDPM-style DiT (not
    flow matching), so the ONNX exports the *per-step* velocity function
    (noisy_actions, timestep, position_ids) → velocity. `reflex serve`
    wraps it in the canonical 4-step denoise loop at inference time.

    `num_steps` is informational — it's recorded in reflex_config.json
    for runtime but NOT baked into the ONNX (the graph is one step).

    Parity target: cos=1.0 at machine precision on single-step velocity
    vs PyTorch `GR00TFullStack.forward`. Measured 2026-04-19 on Modal
    A100-40GB: first-action max_abs=8.34e-07, full-chunk max_abs=3.70e-06
    (single-step); 4-step denoise loop max_abs=1.91e-06.
    """
    _require_monolithic_deps()

    import torch
    from reflex.checkpoint import load_checkpoint
    from reflex.exporters.gr00t import build_gr00t_full_stack

    logger.info("[gr00t] Loading %s", model_id)
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    logger.info("[gr00t] Checkpoint loaded in %.1fs", time.time() - t0)

    logger.info("[gr00t] Building full stack (embodiment=%d)", embodiment_id)
    t0 = time.time()
    full, meta = build_gr00t_full_stack(state_dict, embodiment_id=embodiment_id)
    full.eval()
    logger.info("[gr00t] Built in %.1fs — raw_action_dim=%d, params=%.1fM",
                time.time() - t0, meta["raw_action_dim"],
                meta["full_stack_params_m"])

    raw_action_dim = meta["raw_action_dim"]
    chunk = 50
    B = 1
    dummy_actions = torch.randn(B, chunk, raw_action_dim, dtype=torch.float32)
    dummy_time = torch.tensor([0.5], dtype=torch.float32)
    dummy_pos = torch.arange(chunk, dtype=torch.long).unsqueeze(0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info("[gr00t] torch.onnx.export (opset 19)")
    t0 = time.time()
    torch.onnx.export(
        full,
        (dummy_actions, dummy_time, dummy_pos),
        str(onnx_path),
        input_names=["noisy_actions", "timestep", "position_ids"],
        output_names=["velocity"],
        dynamic_axes={
            "noisy_actions": {0: "batch"},
            "timestep": {0: "batch"},
            "position_ids": {0: "batch"},
            "velocity": {0: "batch"},
        },
        opset_version=19,
    )
    logger.info("[gr00t] ONNX export: %.1fs", time.time() - t0)

    _write_reflex_config(
        output_dir,
        type("GR00TConfig", (), {
            "chunk_size": chunk,
            "max_action_dim": raw_action_dim,
            "num_steps": num_steps,
        })(),
        num_steps=num_steps,
        model_id=model_id,
        model_type="gr00t",
        target=target,
    )

    size_mb = onnx_path.stat().st_size / 1e6
    data_files = list(output_dir.glob("*.data")) + list(output_dir.glob("*.bin"))
    total_mb = size_mb + sum(f.stat().st_size for f in data_files) / 1e6
    return {
        "status": "ok",
        "onnx_path": str(onnx_path),
        "size_mb": total_mb,
        "raw_action_dim": raw_action_dim,
        "embodiment_id": embodiment_id,
        "num_steps": num_steps,
    }


def _model_type_from_local_config(model_id: str) -> str | None:
    """When model_id is a local path, try to read config.json's
    `model_type` (HF) or `policy_type` (lerobot) field. Returns None
    when the path doesn't exist OR config.json is absent OR neither
    field is present.

    Used by export_monolithic as a fallback when the model_id substring
    doesn't carry the family name (e.g., distill output checkpoint
    paths). Per 2026-04-25 self-distilling-serve distill smoke
    (reflex_context/03_experiments/...-distill-smoke.md): the SnapFlow
    distill output dir was rejected because the substring match failed
    on the path, even though the checkpoint's config.json had the
    correct model_type field.
    """
    import json
    from pathlib import Path

    p = Path(model_id)
    if not p.exists():
        return None

    config_path = p / "config.json"
    if not config_path.exists():
        return None

    try:
        with config_path.open() as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # HF transformers convention: 'model_type' field.
    # lerobot policy convention: 'policy_type' field.
    # SnapFlow checkpoint convention: 'type' field
    # (e.g., 'pi05', 'smolvla'). Caught by 2026-04-25 distill smoke v2.
    raw = (
        cfg.get("model_type")
        or cfg.get("policy_type")
        or cfg.get("type")
    )
    if not isinstance(raw, str):
        return None
    raw = raw.lower()

    # Map raw value to our 4-family bounded enum.
    if "smolvla" in raw:
        return "smolvla"
    if "pi05" in raw or "pi_05" in raw or "pi0_5" in raw:
        return "pi05"
    if "pi0" in raw or "pi_0" in raw:
        return "pi0"
    if "gr00t" in raw or "groot" in raw:
        return "gr00t"
    return None


def export_monolithic(
    model_id: str,
    output_dir: str | Path,
    *,
    num_steps: int = 10,
    model_type: str | None = None,
    target: str = "desktop",
) -> dict[str, Any]:
    """Dispatch to the right model-specific exporter.

    If ``model_type`` is None, infer from ``model_id`` (substring match).
    Currently supported: smolvla, pi0, pi05, gr00t.
    """
    if model_type is None:
        # Substring match on model_id first (works for HF ids like
        # 'lerobot/pi05_libero_finetuned_v044'). Per CLAUDE.md no-band-aid
        # principle: when the substring path fails for local-checkpoint
        # paths (e.g., distill output dirs that don't carry the family
        # name), fall back to reading the checkpoint's config.json.
        # Caught by 2026-04-25 self-distilling-serve distill smoke
        # (reflex_context/03_experiments/2026-04-25-self-distilling-serve-distill-smoke.md).
        mid = model_id.lower()
        if "smolvla" in mid:
            model_type = "smolvla"
        elif "pi05" in mid or "pi_05" in mid or "pi0_5" in mid:
            model_type = "pi05"
        elif "pi0" in mid or "pi_0" in mid:
            model_type = "pi0"
        elif "gr00t" in mid or "groot" in mid:
            model_type = "gr00t"
        else:
            # Fallback: read config.json from local path if it exists
            model_type = _model_type_from_local_config(model_id)
            if model_type is None:
                raise ValueError(
                    f"Cannot infer model_type from '{model_id}' "
                    f"(no family substring matched + no readable "
                    f"config.json with policy_type). Pass "
                    f"model_type='smolvla', 'pi0', 'pi05', or 'gr00t' "
                    f"explicitly."
                )

    if model_type == "smolvla":
        result = export_smolvla_monolithic(model_id, output_dir, num_steps=num_steps, target=target)
    elif model_type == "pi0":
        result = export_pi0_monolithic(model_id, output_dir, num_steps=num_steps, target=target)
    elif model_type == "pi05":
        result = export_pi05_monolithic(model_id, output_dir, num_steps=num_steps, target=target)
    elif model_type == "gr00t":
        gr00t_steps = 4 if num_steps == 10 else num_steps
        result = export_gr00t_monolithic(model_id, output_dir, num_steps=gr00t_steps, target=target)
    else:
        raise ValueError(
            f"Monolithic export for model_type={model_type!r} not yet supported."
        )

    # Post-export weight fusion pass (lifted from dexmal/realtime-vla MIT).
    # Fuses RMSNorm scales into MatMul weights, folds Euler dt into output
    # projections. Algebraically equivalent — zero accuracy cost.
    onnx_path = result.get("onnx_path")
    if onnx_path:
        try:
            from reflex.exporters.weight_fusion import fuse_weights
            fused_path = fuse_weights(onnx_path, num_steps=num_steps)
            logger.info("Weight fusion pass complete: %s", fused_path)
        except Exception as e:
            logger.warning("Weight fusion pass failed (non-fatal): %s", e)

    return result


__all__ = [
    "export_monolithic",
    "export_pi0_monolithic",
    "export_smolvla_monolithic",
    "export_gr00t_monolithic",
    "apply_export_patches",
]
