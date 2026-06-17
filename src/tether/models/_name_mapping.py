"""Per-VLA name_mapping walker for HF checkpoint compatibility.

Lifted from FluxVLA's `base_vla.py:201-299` pattern per decision S-1
(`01_decisions/2026-05-19-fluxvla-lift-program-decisions.md`) — per-VLA-config,
NOT per-component. Each VLA class declares its rename map; the walker applies
it to an incoming HF state_dict before loading into the spine's components.

Why per-VLA-config not per-component: keeps the spine's component classes
oblivious to HF-checkpoint naming conventions. The naming is a VLA-level
concern (each VLA has its own historical naming from upstream training); the
components just need cleanly-named weights at load time.

Pattern (in a VLA subclass):

    @VLAS.register
    class Pi05VLA(BaseVLA):
        # Strip 'module.' prefix from DDP-trained checkpoints; route
        # paligemma_with_expert.paligemma.model.language_model.* → llm_backbone.*
        NAME_MAPPING = {
            "module.": "",
            "paligemma_with_expert.paligemma.model.language_model": "llm_backbone",
            "paligemma_with_expert.gemma_expert": "llm_backbone.expert",
            "action_in_proj": "vla_head.action_in_proj",
            "action_out_proj": "vla_head.action_out_proj",
            "state_proj": "projector.state_proj",
        }
        ...

The walker is intentionally simple — first-match-wins prefix substitution.
For checkpoints that need regex-level remapping, override
`apply_name_mapping()` on the VLA subclass instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def apply_name_mapping(
    state_dict: dict[str, "torch.Tensor"],
    name_mapping: dict[str, str],
    *,
    strict: bool = False,
) -> dict[str, "torch.Tensor"]:
    """Rename state_dict keys via a per-VLA name_mapping dict.

    The mapping is iterated in declaration order; first prefix match wins.
    Keys without any matching prefix pass through unchanged (callers can
    set `strict=True` to raise on unmatched keys instead).

    Args:
        state_dict: HF checkpoint as `{old_key: tensor}`.
        name_mapping: `{src_prefix: dst_prefix}`. Empty src_prefix is invalid.
        strict: If True, raise KeyError for any state_dict key that doesn't
            match any name_mapping prefix. Useful for catching upstream HF
            checkpoint layout drift early (per CLAUDE.md "fail loudly").

    Returns:
        New dict with renamed keys. The original state_dict is not mutated.

    Raises:
        KeyError: if `strict=True` and a state_dict key matches no prefix.
        ValueError: if `name_mapping` contains an empty src_prefix (ambiguous).
    """
    # Validate mapping eagerly — easier to debug at load time than at use time.
    for src in name_mapping:
        if not src:
            raise ValueError(
                "name_mapping contains empty src_prefix — would match every key. "
                "If you intended to add a prefix to all keys, use a non-empty "
                "marker (e.g. 'model.') as the src."
            )

    # Preserve mapping declaration order for first-match-wins semantics.
    # (dict iteration is insertion-ordered as of Python 3.7+.)
    mapping_items = list(name_mapping.items())

    renamed: dict[str, "torch.Tensor"] = {}
    unmatched: list[str] = []

    for old_key, tensor in state_dict.items():
        new_key = _rename_key(old_key, mapping_items)
        if new_key is old_key and strict:
            # Pure identity → no prefix matched → unmatched in strict mode.
            unmatched.append(old_key)
        renamed[new_key] = tensor

    if strict and unmatched:
        sample = unmatched[:5]
        more = f" (+ {len(unmatched) - 5} more)" if len(unmatched) > 5 else ""
        raise KeyError(
            f"{len(unmatched)} state_dict key(s) matched no name_mapping prefix "
            f"in strict mode. First 5: {sample}{more}. "
            f"Either add a passthrough entry to name_mapping (e.g. "
            f"'kept_prefix.': 'kept_prefix.') or call with strict=False."
        )

    return renamed


def _rename_key(key: str, mapping_items: list[tuple[str, str]]) -> str:
    """First-match-wins prefix substitution. Returns key unchanged if no match."""
    for src_prefix, dst_prefix in mapping_items:
        if key.startswith(src_prefix):
            return dst_prefix + key[len(src_prefix):]
    return key


__all__ = ["apply_name_mapping"]
