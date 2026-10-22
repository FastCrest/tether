"""Monolithic ONNX export — the cos=1.0 verified production path.

Extracted from `scripts/modal_{smolvla,pi0}_monolithic_export.py` so it can
run locally (no Modal account required). Same set of transformers 5.x +
onnx-diagnostic patches; same torch.export + torch.onnx.export pipeline;
same post-export Where-type fix.

Usage (local):
    pip install 'fastcrest-tether[monolithic]'
    tether export lerobot/smolvla_base --monolithic --output ./smol

Requires (pinned in the ``monolithic`` extra):
    lerobot==0.6.1
    transformers==5.5.4   (LeRobot 0.6.1 requires >=5.4,<5.6)
    onnx-diagnostic>=0.9
    onnxscript>=0.1
    optree, scipy, num2words

The verified artifacts produced by this module match what the Modal
scripts produce. See
https://github.com/FastCrest/reflex-vault/blob/main/reflex_vla/measured_numbers.md
for the parity rows this export is responsible for.
"""
from __future__ import annotations

from tether._lerobot_version import QUALIFIED_LEROBOT_VERSION, QUALIFIED_TRANSFORMERS_VERSION

import contextlib
import logging
import time
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
from typing import Any

MIN_EXPORT_TORCH = (2, 8)

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


_CCM_DISPATCH_RATIONALE = (
    "`create_causal_mask` is dispatched for num_steps>1 export: an "
    "already-prepared 4D mask is returned unchanged, everything else returns "
    "None. Transformers 5.3 rebuilds a prefix-only [1,1,Q,past_len] mask that "
    "fails to broadcast against the actual [1,H,Q,past_len+suffix_len] "
    "attention scores under torch.export FakeTensor tracing (the `835 -> 886` "
    "expand error); returning None for that case unblocks export. SmolVLA "
    "(SmolLM2 path) applies its own 3D mask inside its attention "
    "implementation and never reaches the 4D branch — cos=1.0 either way. "
    "pi0/pi0.5 (PaliGemma + Gemma) pass the 4D block-causal mask built by the "
    "export denoise_step patch, and returning it preserves prefix-pad "
    "masking: measured cos=1.000000 at num_steps=10 (2026-04-19) vs cos=0.977 "
    "when the mask was dropped. See 01_architecture/pi0_monolithic_wrap_pattern.md."
)


def _causal_mask_export_shim(*args, **kwargs):
    """Stand-in for ``transformers.masking_utils.create_causal_mask`` under
    ``torch.export``.

    Returns the caller's mask unchanged when it is already a prepared 4D mask
    (pi0/pi0.5 with the F.pad-based ``denoise_step`` patch installed by
    :func:`apply_export_patches`), otherwise ``None`` to skip the rebuild that
    cannot be traced. Module level so it is importable and testable without a
    transformers/lerobot install.
    """
    if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
        kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
    attn = kwargs.get("attention_mask")
    if attn is None and len(args) >= 3:
        attn = args[2]
    if attn is not None and getattr(attn, "ndim", None) == 4:
        return attn
    return None


# transformers>=5.4 ``masking_utils.eager_mask`` signature (verified on 5.5.4).
# onnx-diagnostic <=0.9.3 (latest as of 2026-10-13, identical on its GitHub
# main) replaces it during ``torch_export_patches`` with a function written
# for the pre-5.4 signature ``(batch_size, cache_position, kv_length, ...)``,
# so every eager-attention mask built through ``create_bidirectional_mask`` or
# ``create_causal_mask`` dies with "missing 1 required positional argument:
# 'cache_position'". SmolVLA reaches it through the SmolVLM vision tower;
# pi0/pi0.5 never do (SigLIP's vision tower builds no mask and Gemma's
# create_causal_mask is replaced by _causal_mask_export_shim above).
_EAGER_MASK_PARAMS_TF54 = (
    "batch_size", "q_length", "kv_length", "q_offset", "kv_offset",
    "mask_function", "attention_mask", "dtype", "allow_is_bidirectional_skip",
    "use_vmap", "device", "kwargs",
)


def _eager_mask_tf54(
    batch_size,
    q_length,
    kv_length,
    q_offset=0,
    kv_offset=0,
    mask_function=None,
    attention_mask=None,
    dtype=None,
    allow_is_bidirectional_skip=False,
    use_vmap=False,
    device="cpu",
    **kwargs,
):
    """transformers 5.5.4 ``eager_mask`` semantics, built on onnx-diagnostic's
    export-safe ``patched_sdpa_mask_recent_torch`` (which already accepts the
    5.4+ ``q_length``/``q_offset`` arguments) and its ``where``-free bool->float
    conversion."""
    import torch
    from onnx_diagnostic.torch_export_patches.patches import (
        _patch_transformers_masking_utils as od_masking,
    )
    from transformers import masking_utils

    kwargs.pop("allow_is_causal_skip", None)
    kwargs.pop("allow_torch_fix", None)
    mask = od_masking.patched_sdpa_mask_recent_torch(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_function or masking_utils.causal_mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        use_vmap=use_vmap,
        device=device,
        **kwargs,
    )
    if mask is None:  # only a skippable bidirectional mask returns None
        return None
    dtype = dtype or torch.float32
    return (~mask).to(dtype) * torch.finfo(dtype).min


@contextlib.contextmanager
def export_patches():
    """``onnx_diagnostic``'s ``torch_export_patches(patch_transformers=True)``
    with its ``eager_mask`` patch replaced when it does not match the installed
    transformers. Every Tether ``torch.export`` call goes through here.

    Refuses (RuntimeError) on any transformers ``eager_mask`` signature other
    than the qualified 5.4+ one, rather than guessing at an unknown API.
    """
    import inspect

    import transformers
    from onnx_diagnostic.torch_export_patches import torch_export_patches
    from onnx_diagnostic.torch_export_patches.patches import (
        _patch_transformers_masking_utils as od_masking,
    )
    from transformers import masking_utils

    tf_params = tuple(inspect.signature(masking_utils.eager_mask).parameters)
    od_params = tuple(inspect.signature(od_masking.patched_eager_mask).parameters)
    if tf_params != _EAGER_MASK_PARAMS_TF54:
        raise RuntimeError(
            f"transformers {transformers.__version__} eager_mask signature {tf_params} "
            f"is not the qualified 5.4+ signature {_EAGER_MASK_PARAMS_TF54}; Tether "
            "cannot vouch for onnx-diagnostic's mask patches on it. Install "
            f"transformers=={QUALIFIED_TRANSFORMERS_VERSION}."
        )
    repair = "cache_position" in od_params
    if not repair and od_params != tf_params:
        raise RuntimeError(
            f"onnx-diagnostic's patched_eager_mask signature {od_params} matches "
            f"neither the pre-5.4 form nor transformers {transformers.__version__} "
            f"{tf_params}; refusing to export with an unknown mask patch."
        )

    with torch_export_patches(patch_transformers=True):
        if not repair:
            yield
            return
        registry = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
        saved_module, saved_registry = masking_utils.eager_mask, registry["eager"]
        masking_utils.eager_mask = _eager_mask_tf54
        registry["eager"] = _eager_mask_tf54
        try:
            yield
        finally:
            # Hand back exactly what onnx-diagnostic installed so its own
            # restore logic sees the state it expects.
            masking_utils.eager_mask = saved_module
            registry["eager"] = saved_registry

_TOKENIZER_REFS = {
    "pi0": "google/paligemma-3b-pt-224",
    "pi05": "google/paligemma-3b-pt-224",
    "smolvla": "HuggingFaceTB/SmolLM2-135M",
}

_SMOLVLA_EXPORT_PATCH_FAILURES: list[str] = []


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
        from tether.runtime.tokenizers import tokenizer_offline_enabled
        tok = AutoTokenizer.from_pretrained(
            tokenizer_ref,
            local_files_only=tokenizer_offline_enabled(),
        )
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        tok_dir = output_dir / "tokenizer"
        saved_paths = tok.save_pretrained(tok_dir)
        meta.update(
            {
                "tokenizer_path": "tokenizer",
                "tokenizer_bundled": True,
                "_artifact_paths": [
                    Path(path).resolve().relative_to(output_dir.resolve()).as_posix()
                    for path in saved_paths
                ],
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
    required package version mismatch is detected.
    """
    _quiet_noisy_export_loggers()
    missing = []
    try:
        import transformers
        if transformers.__version__ != QUALIFIED_TRANSFORMERS_VERSION:
            raise ImportError(
                f"transformers {transformers.__version__} detected; the "
                f"monolithic export requires exactly {QUALIFIED_TRANSFORMERS_VERSION} "
                f"(LeRobot {QUALIFIED_LEROBOT_VERSION} needs >=5.4: on 5.3 its "
                f"pi0/pi0.5 embeddings are mis-scaled). Install with: "
                f"pip install transformers=={QUALIFIED_TRANSFORMERS_VERSION}"
            )
    except ImportError as e:
        missing.append(f"transformers=={QUALIFIED_TRANSFORMERS_VERSION} ({e})")

    # onnx-diagnostic 0.8.11 and 0.9.0-0.9.3 unconditionally patch
    # torch._export.non_strict_utils._constrain_user_specified_dimhint_range,
    # which first exists in torch 2.8.0. On older torch every pi0/pi0.5/
    # SmolVLA/GR00T export dies with an AttributeError on entering
    # torch_export_patches(); refuse here instead.
    try:
        import torch
        major_minor = tuple(int(part) for part in str(torch.__version__).split(".")[:2])
        if major_minor < MIN_EXPORT_TORCH:
            raise ImportError(
                f"torch {torch.__version__} detected; onnx-diagnostic's export "
                "patches need torch>=2.8"
            )
    except (ImportError, ValueError) as e:
        missing.append(f"torch>=2.8 ({e})")

    try:
        __import__("lerobot")
        try:
            lerobot_version = _dist_version("lerobot")
        except PackageNotFoundError as exc:
            raise ImportError(
                "lerobot imports but package metadata is missing; install with: "
                "pip install 'fastcrest-tether[monolithic]'"
            ) from exc
        if lerobot_version != QUALIFIED_LEROBOT_VERSION:
            raise ImportError(
                f"lerobot {lerobot_version} detected; monolithic export requires "
                f"exactly lerobot=={QUALIFIED_LEROBOT_VERSION}. Install with: "
                "pip install 'fastcrest-tether[monolithic]'"
            )
    except ImportError as e:
        missing.append(f"lerobot=={QUALIFIED_LEROBOT_VERSION} ({e})")

    for mod_name, pip_name in [
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
            + "\n\nInstall with: pip install 'fastcrest-tether[monolithic]'"
        )


def _record_smolvla_export_patch_failure(patch_name: str, exc: BaseException) -> None:
    message = f"{patch_name}: {type(exc).__name__}: {exc}"
    _SMOLVLA_EXPORT_PATCH_FAILURES.append(message)
    logger.warning(
        "[smolvla] export patch failed; SmolVLA monolithic export will stop "
        "before torch.export instead of failing later: %s",
        message,
    )


def _raise_if_smolvla_export_patches_failed() -> None:
    if not _SMOLVLA_EXPORT_PATCH_FAILURES:
        return
    details = "\n  - ".join(_SMOLVLA_EXPORT_PATCH_FAILURES)
    raise RuntimeError(
        "SmolVLA monolithic export patches failed to install. This usually "
        "means the lerobot/transformers export stack does not match the pinned "
        "monolithic environment; install with: pip install "
        "'fastcrest-tether[monolithic]'.\n  - "
        + details
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

    _SMOLVLA_EXPORT_PATCH_FAILURES.clear()

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

    # create_causal_mask -> dispatch (see _causal_mask_export_shim).
    #
    # A prepared 4D mask is passed through; everything else returns None to
    # bypass the mask rebuild that would trigger the 835->886 broadcast error
    # under torch.export FakeTensor tracing with num_steps>1.
    #
    # The pass-through branch is what makes pi0/pi0.5 parity cos=1.0 rather
    # than 0.977: the mask the denoise_step patch below builds with F.pad is
    # the correct [1,1,suffix_len,prefix_len+suffix_len] block-causal mask,
    # and dropping it silently removes prefix-pad masking on the PaliGemma +
    # Gemma path. Returning None unconditionally (the state this module
    # shipped in before 2026-09-16) is what the 0.977 row in
    # reflex_context/measured_numbers.md measured; the 2026-04-19 row that
    # measured cos=1.000000 was produced by scripts/modal_pi0_monolithic_export.py,
    # whose shim already dispatched on 4D. That dispatch was lost when this
    # module was extracted from the Modal scripts.
    #
    # SmolVLA is unaffected either way: its expert stack applies a 3D mask
    # inside its own attention implementation (lerobot smolvla
    # smolvlm_with_expert.eager_attention_forward), so it never takes the
    # 4D branch. See 01_architecture/pi0_monolithic_wrap_pattern.md
    from transformers import masking_utils

    masking_utils.create_causal_mask = _causal_mask_export_shim
    try:
        from lerobot.policies import pi_gemma as _pg
        if hasattr(_pg, "create_causal_mask"):
            _pg.create_causal_mask = _causal_mask_export_shim
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

        if not getattr(_smolvla.SmolVLMWithExpertModel.embed_image, "_tether_patched", False):
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

            _embed_image_with_explicit_patch_mask._tether_patched = True
            _smolvla.SmolVLMWithExpertModel.embed_image = _embed_image_with_explicit_patch_mask
    except Exception as exc:
        _record_smolvla_export_patch_failure(
            "SmolVLA explicit patch-mask export patch",
            exc,
        )

    try:
        from transformers.models.smolvlm import modeling_smolvlm as _smolvlm

        if not getattr(_smolvlm.SmolVLMVisionAttention.forward, "_tether_patched", False):
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

            _vision_attention_forward_no_dense_mask._tether_patched = True
            _smolvlm.SmolVLMVisionAttention.forward = _vision_attention_forward_no_dense_mask
    except Exception as exc:
        _record_smolvla_export_patch_failure(
            "SmolVLM vision attention export patch",
            exc,
        )

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

    # LeRobot 0.6.1 returns image features in its native embedding scale.
    # Keep embed_image unchanged: multiplying by sqrt(hidden_size), as the
    # old 0.5.1 shim did, changes every prefix cache and exported action.

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
        from transformers.cache_utils import DynamicLayer as _DL
        from lerobot.policies.pi0 import modeling_pi0 as _mp0
        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks as _make_att_2d_masks
        from lerobot.policies.common.vla_utils import prepare_attention_masks_4d

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
                prefix_pad_masks_extended = _pad_bool_true_last_dim(
                    prefix_pad_masks, 0, pad_deficit
                )

            prefix_pad_2d_masks = prefix_pad_masks_extended[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

            prefix_allowed = _pad_bool_true_last_dim(prefix_pad_2d_masks, 0, suffix_len)
            suffix_allowed = _pad_bool_true_last_dim(suffix_att_2d_masks, prefix_len, 0)
            full_att_2d_masks = prefix_allowed & suffix_allowed

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
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


def _strip_default_scatternd_reduction(graph: Any) -> int:
    """Drop ``reduction="none"`` from ScatterND nodes; return how many.

    torch.onnx.export writes the attribute explicitly on every ScatterND it
    emits for slice/index assignment (542 nodes in SmolVLA-base at torch 2.10).
    "none" is the ONNX default, so removing it changes nothing numerically, but
    TensorRT 10.3-10.7's ONNX parser rejects ScatterND whenever the attribute
    is *present* (onnx-tensorrt ``onnxOpImporters.cpp``, release/10.3-GA:
    ``ASSERT_NODE(!attrs.count("reduction"), ...)``), which is what pushed whole
    VLA graphs off the TensorRT EP on JetPack 6.2. Real reductions ("add",
    "mul", ...) are kept; they are not equivalent and stay off TensorRT.
    Subgraphs are included so If/Loop bodies are cleaned too.
    """
    from onnx import AttributeProto

    stripped = 0
    for node in graph.node:
        if node.op_type == "ScatterND" and node.domain in ("", "ai.onnx"):
            for index, attr in enumerate(node.attribute):
                if attr.name == "reduction" and attr.s == b"none":
                    del node.attribute[index]
                    stripped += 1
                    break
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                stripped += _strip_default_scatternd_reduction(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                stripped += sum(_strip_default_scatternd_reduction(g) for g in attr.graphs)
    return stripped


def _repair_exported_onnx(onnx_path: Path) -> int:
    """Post-export repairs every exporter runs on its ``torch.onnx.export`` output.

    1. torch.onnx.export sometimes lowers aten::index_put to ONNX
       Where(bool, int64, float32), which onnxruntime rejects at load time.
       Insert Cast nodes that coerce the mismatched branch to the declared
       output dtype.
    2. Remove the explicit default ``reduction="none"`` from ScatterND so
       TensorRT accepts the node (see ``_strip_default_scatternd_reduction``).

    Returns the count of fixes applied.
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
    stripped = _strip_default_scatternd_reduction(model.graph)
    for function in model.functions:
        stripped += _strip_default_scatternd_reduction(function)
    if stripped:
        logger.info("stripped default reduction from %d ScatterND nodes", stripped)
    if fixes or stripped:
        with open(onnx_path, "wb") as f:
            f.write(model.SerializeToString())

    return fixes + stripped


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

    apply_export_patches()
    _raise_if_smolvla_export_patches_failed()

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
    with export_patches():
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

    fixes = _repair_exported_onnx(onnx_path)
    logger.info("[smolvla] post-export Cast fixes: %d", fixes)

    _write_tether_config(
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

    Parity: cos=1.0 at num_steps=1 and at num_steps=10. The num_steps=10 path
    needs the full fix stack in :func:`apply_export_patches` — the F.pad mask
    assembly, the frozen ``DynamicLayer.update`` and the 4D pass-through in
    :func:`_causal_mask_export_shim`. Dropping any one of them regresses
    num_steps=10 parity (cos=0.977 when the 4D mask is discarded).

    The cos=1.0 figure is a Linux/CUDA measurement (2026-04-19,
    reflex_context/measured_numbers.md). Landing this module does not produce
    a receipt; see docs/pi0-export-parity.md.

    Trace dummy shapes (lang length, state length) are read off the pinned
    processor's real output on the canonical probe task — the receipt
    harness feeds that same tokenization, so export and harness agree by
    construction.

    Args, returns: same shape as ``export_smolvla_monolithic``.
    """
    _require_monolithic_deps()

    import torch
    import torch.nn as nn

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

    # Size the trace dummies from the pinned processor's real output, not a
    # hardcoded lang length. The receipt harness feeds the processor's
    # tokenization of the canonical probe task (48 tokens at the pinned
    # pi0_base revision + lerobot 0.5.1); the old (B, 16) dummy froze a
    # graph whose lang_tokens dim the shared inputs cannot feed
    # (first-hand: "Got 48 Expected 16" at session.run). Same model_id,
    # same pipeline, same task as the harness, so the two agree by
    # construction; values stay random, only shapes are read.
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    _probe_pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=model_id,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )
    _probe_img = torch.zeros(3, 224, 224)
    _probe_batch = _probe_pre({
        "observation.images.base_0_rgb": _probe_img.unsqueeze(0),
        "observation.images.left_wrist_0_rgb": _probe_img.unsqueeze(0),
        "observation.images.right_wrist_0_rgb": _probe_img.unsqueeze(0),
        "observation.state": torch.zeros(1, 14),
        "task": ["pick up the red bowl"],
    })
    _lang_len = int(_probe_batch["observation.language.tokens"].shape[1])
    _state_len = int(policy.prepare_state(_probe_batch).shape[1])
    logger.info("[pi0] probe shapes: lang_len=%d state_len=%d", _lang_len, _state_len)

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

    dummy = dict(
        img_base=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_l=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_r=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        mask_base=torch.ones(B, dtype=torch.bool),
        mask_wrist_l=torch.ones(B, dtype=torch.bool),
        mask_wrist_r=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 257152, (B, _lang_len), dtype=torch.long),
        lang_masks=torch.ones(B, _lang_len, dtype=torch.bool),
        state=torch.randn(B, _state_len, dtype=torch.float32),
        noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info("[pi0] torch.export.export ...")
    t0 = time.time()
    with export_patches():
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

    fixes = _repair_exported_onnx(onnx_path)
    logger.info("[pi0] post-export Cast fixes: %d", fixes)

    _write_tether_config(
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


def _write_tether_config(
    output_dir: Path,
    policy_config: Any,
    *,
    num_steps: int,
    model_id: str,
    model_type: str,
    target: str = "desktop",
) -> None:
    """Write `tether_config.json` + seed a VERIFICATION.md receipt."""
    tokenizer_meta = _bundle_tokenizer(output_dir, model_type)
    tokenizer_artifacts = [
        (path, "tokenizer") for path in tokenizer_meta.pop("_artifact_paths", [])
    ]
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
        "export_kind": "monolithic_onnx",
        "notes": _CCM_DISPATCH_RATIONALE if num_steps > 1 else None,
        **tokenizer_meta,
    }
    from tether.export_config import build_producer_config, write_tether_config

    canonical = build_producer_config(
        output_dir,
        producer="monolithic",
        model_id=model_id,
        model_type=model_type,
        action_dim=cfg_dict["action_dim"],
        num_denoising_steps=num_steps,
        opset=19,
        optional_artifacts=tokenizer_artifacts,
        metadata=cfg_dict,
    )
    write_tether_config(output_dir, canonical)

    # Seed VERIFICATION.md at export time — every export ships a receipt,
    # even before `tether validate` fills in the parity numbers. Callers
    # that run validate afterward will overwrite with verified max_abs.
    # Wrapped in try/except so a report-writing error can't fail the
    # actual export (the ONNX is already on disk by this point).
    try:
        from tether.verification_report import write_verification_report
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
    The four-patch stack (F.pad mask, frozen DynamicLayer.update,
    past_kv.seq_length for mask, and the 4D pass-through in
    _causal_mask_export_shim) is applied by apply_export_patches(),
    so pi0.5 inherits the cos=1.0 fix for free.

    Parity: cos=1.0, max_abs ~2.38e-07 at num_steps=10 vs PyTorch
    sample_actions(num_steps=10).
    """
    _require_monolithic_deps()

    import torch
    import torch.nn as nn

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
    with export_patches():
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

    fixes = _repair_exported_onnx(onnx_path)
    logger.info("[pi05] post-export Cast fixes: %d", fixes)

    _write_tether_config(
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


def _pad_bool_true_last_dim(masks: Any, pad_left: int, pad_right: int) -> Any:
    """Pad a bool mask with True columns along the last dim WITHOUT F.pad.

    ``F.pad(..., value=True)`` on a bool tensor lowers to ONNX Pad, for
    which ORT's CUDA EP has no kernel: the pi0.5 monolithic export carried
    exactly one such node (bool ``[1,50,784] -> [1,50,834]``) and the
    no-CPU-fallback gate failed the session at creation (M4 #52). A Concat
    with a True block lowers to ONNX Concat, which probes CUDA-clean, and
    is bit-identical (True-pad + AND-mask = same allowed positions).

    The True block is built as scalar-ones + expand (no factory call with
    symbolic sizes) so it traces under torch.export exactly like the
    pre-existing ``[:, None, :].expand(...)`` in this mask build.
    """
    import torch

    parts = []
    if pad_left:
        parts.append(
            torch.ones((), dtype=torch.bool, device=masks.device).expand(
                *masks.shape[:-1], pad_left
            )
        )
    parts.append(masks)
    if pad_right:
        parts.append(
            torch.ones((), dtype=torch.bool, device=masks.device).expand(
                *masks.shape[:-1], pad_right
            )
        )
    if len(parts) == 1:
        return masks
    return torch.cat(parts, dim=-1)


def _apply_pi05_denoise_step_patch() -> None:
    """Patch PI05Pytorch.denoise_step with the Pad-free mask + frozen-cache
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
        from lerobot.policies.pi05 import modeling_pi05 as _mp05
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks as _make
        from lerobot.policies.common.vla_utils import prepare_attention_masks_4d

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
                prefix_pad_masks_extended = _pad_bool_true_last_dim(
                    prefix_pad_masks, 0, pad_deficit
                )

            prefix_pad_2d_masks = prefix_pad_masks_extended[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = _make(suffix_pad_masks, suffix_att_masks)

            prefix_allowed = _pad_bool_true_last_dim(prefix_pad_2d_masks, 0, suffix_len)
            suffix_allowed = _pad_bool_true_last_dim(suffix_att_2d_masks, prefix_len, 0)
            full_att_2d_masks = prefix_allowed & suffix_allowed

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
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
    precision: str = "fp32",
) -> dict[str, Any]:
    """Export a SnapFlow-distilled student as monolithic ONNX at 1-NFE.

    Takes a tether-saved distill checkpoint dir (contains
    ``model.safetensors`` with the base pi0/pi0.5 weights plus the
    ``target_time_embed_mlp.*`` SnapFlow keys, config.json, processor
    files), loads it via ``load_snapflow_student``, and produces a single
    monolithic ONNX where the denoising loop is BAKED to one step with
    ``target_time=1``.

    The output ONNX has the same input/output signature as a regular
    pi0/pi0.5 monolithic ONNX (per model family), so ``tether serve``
    can load it through the existing path without special-casing
    distilled students. At runtime, one forward pass = one full action
    chunk (no host-side Euler loop needed).

    Parity claim: the ONNX replicates ``student.model.sample_actions_1step``
    bit-exact up to the usual ONNX float32 precision floor. Student
    task-success separately measured at 29/30 on LIBERO (vs teacher
    28/30) — see the 2026-04-21 session log.
    """
    validate_monolithic_precision(precision)
    _require_monolithic_deps()

    import torch
    import torch.nn as nn

    from tether.distill.snapflow_pi0_model import load_snapflow_student

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
    with export_patches():
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

    fixes = _repair_exported_onnx(onnx_path)
    logger.info("[snapflow-export] post-export Cast fixes: %d", fixes)

    _write_tether_config(
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
    through torch.export.export). Freeze cache updates during the expert
    pass, as for teacher export: tracing can evaluate even a single-step
    forward repeatedly, and must preserve the prefix cache each time.
    """
    import torch

    from lerobot.policies.common.vla_utils import prepare_attention_masks_4d

    if is_pi05:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    else:
        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

    if is_pi05:
        def _export_denoise_step(self, prefix_pad_masks, past_key_values, x_t,
                                 timestep, target_time=None, state=None):
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                x_t, timestep, target_time=target_time,
                **({"state": state} if hasattr(self, "state_proj") else {}),
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            prefix_allowed = _pad_bool_true_last_dim(prefix_pad_2d_masks, 0, suffix_len)
            suffix_allowed = _pad_bool_true_last_dim(suffix_att_2d_masks, prefix_len, 0)
            full_att_2d_masks = prefix_allowed & suffix_allowed

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
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
            prefix_allowed = _pad_bool_true_last_dim(prefix_pad_2d_masks, 0, suffix_len)
            suffix_allowed = _pad_bool_true_last_dim(suffix_att_2d_masks, prefix_len, 0)
            full_att_2d_masks = prefix_allowed & suffix_allowed

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
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
    def _denoise_step_with_flag(self, *args, **kwargs):
        previous_phase = _denoise_phase[0]
        _denoise_phase[0] = True
        try:
            return _export_denoise_step(self, *args, **kwargs)
        finally:
            _denoise_phase[0] = previous_phase

    type(model).denoise_step = _denoise_step_with_flag


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
    (noisy_actions, timestep, position_ids) → velocity. `tether serve`
    wraps it in the canonical 4-step denoise loop at inference time.

    `num_steps` is informational — it's recorded in tether_config.json
    for runtime but NOT baked into the ONNX (the graph is one step).

    Parity target: cos=1.0 at machine precision on single-step velocity
    vs PyTorch `GR00TFullStack.forward`. Measured 2026-04-19 on Modal
    A100-40GB: first-action max_abs=8.34e-07, full-chunk max_abs=3.70e-06
    (single-step); 4-step denoise loop max_abs=1.91e-06.
    """
    _require_monolithic_deps()

    import torch
    from tether.checkpoint import load_checkpoint
    from tether.exporters.gr00t import build_gr00t_full_stack

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

    _write_tether_config(
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


# Config values are exact aliases. ID matching uses the same aliases as a
# complete leading token of the repository name, never the namespace/path.
_FAMILY_ALIASES = {
    "smolvla": "smolvla",
    "pi0": "pi0", "pi_0": "pi0",
    "pi05": "pi05", "pi_05": "pi05", "pi0_5": "pi05", "pi0.5": "pi05",
    "gr00t": "gr00t", "groot": "gr00t", "gr00tn1d6": "gr00t",
    "pi0_fast": "pi0_fast", "pi0fast": "pi0_fast", "pi0-fast": "pi0_fast",
}


def validate_monolithic_precision(precision: str) -> None:
    """Reject precision requests the FP32 ONNX exporters cannot implement."""
    if precision != "fp32":
        raise ValueError(
            f"Monolithic export does not support precision={precision!r}. "
            "Supported precisions: fp32. No precision conversion is implemented."
        )


def _family_from_config(config_path: Path) -> str | None:
    import json

    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read model family from {config_path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"Model config must be an object: {config_path}")
    # LeRobot's policy type is authoritative; model_type may name a backbone.
    for field in ("type", "policy_type", "model_type"):
        if field not in cfg:
            continue
        raw = cfg[field]
        if not isinstance(raw, str):
            raise ValueError(f"Invalid model family {field}={raw!r} in {config_path}")
        family = _FAMILY_ALIASES.get(raw.lower())
        if family is None:
            raise ValueError(f"Unsupported model family {field}={raw!r} in {config_path}")
        return family
    return None


def _model_type_from_local_config(model_id: str) -> str | None:
    """Read local family metadata first; absent metadata permits ID matching."""
    path = Path(model_id)
    if not path.exists():
        return None
    config_path = (path if path.is_dir() else path.parent) / "config.json"
    return _family_from_config(config_path) if config_path.exists() else None


def _model_type_from_hub_config(model_id: str) -> str | None:
    """Read HF metadata before ID matching; only a missing config is optional."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
    from tether.registry.models import hub_reference

    repo_id, revision = hub_reference(model_id)
    try:
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json",
            **({"revision": revision} if revision else {}),
        )
    except EntryNotFoundError as exc:
        # Local cache misses / offline errors do not prove config is absent.
        if getattr(exc, "response", None) is not None and exc.response.status_code == 404:
            logger.info("No config.json in %s; matching repository family token", model_id)
            return None
        raise ValueError(f"Cannot read config.json for {model_id}: {exc}") from exc
    except (HfHubHTTPError, OSError) as exc:
        raise ValueError(f"Cannot read config.json for {model_id}: {exc}") from exc
    return _family_from_config(Path(config_path))


def _family_from_model_id(model_id: str) -> str | None:
    name = Path(model_id).name.lower()
    # Longest aliases first: pi0_fast must never match the shorter pi0 token.
    for alias in sorted(_FAMILY_ALIASES, key=len, reverse=True):
        if name == alias or name.startswith((alias + "_", alias + "-")):
            return _FAMILY_ALIASES[alias]
    return None


def export_monolithic(
    model_id: str,
    output_dir: str | Path,
    *,
    num_steps: int = 10,
    model_type: str | None = None,
    target: str = "desktop",
    precision: str = "fp32",
) -> dict[str, Any]:
    """Dispatch using exact config aliases first, then repository family tokens.

    The monolithic ONNX path currently supports FP32 only.
    """
    validate_monolithic_precision(precision)
    if model_type is None:
        if Path(model_id).exists():
            model_type = _model_type_from_local_config(model_id)
        elif not Path(model_id).is_absolute() and len(model_id.split("/")) == 2:
            model_type = _model_type_from_hub_config(model_id)
        if model_type is None:
            model_type = _family_from_model_id(model_id)
        if model_type is None:
            raise ValueError(
                f"Cannot infer model_type from '{model_id}'. No family metadata or "
                "exact repository family token. Pass model_type='smolvla', 'pi0', "
                "'pi05', or 'gr00t' explicitly."
            )
    else:
        model_type = _FAMILY_ALIASES.get(model_type.lower(), model_type)
    if model_type == "pi0_fast":
        raise ValueError(
            "pi0_fast (pi0-FAST) has no monolithic exporter. "
            "Supported families: smolvla, pi0, pi05, gr00t."
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
            from tether.exporters.weight_fusion import fuse_weights
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
