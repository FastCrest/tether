"""BaseVLA abstract spine — foundation for the model-agnostic VLA refactor.

Lifted PATTERN from FluxVLA's `fluxvla/models/vlas/base_vla.py:36-299` per
the lift-program ADR + decisions (S-1 through S-5 in `01_decisions/
2026-05-19-fluxvla-lift-program-decisions.md`). Reimplemented without
mmengine / GenerationMixin / nn.Module-as-abstract — we don't need their
training-spine machinery; we need a clean composition surface that
exporters and the runtime can both walk.

Per decision S-2: **six component slots**, not the five FluxVLA spec'd.
GR00T's Eagle (fused SigLIP + Llama) doesn't decompose into the standard
vision_backbone + llm_backbone pattern, so it gets its own `vlm_backbone`
slot. DreamZero's T5 text encoder doesn't fit `llm_backbone` (no RoPE'd
attention) so it gets `text_encoder`.

| Slot | Used by |
|---|---|
| `vision_backbone` | pi0, pi0.5, smolvla (SigLIP / DinoSigLIP) |
| `llm_backbone` | pi0, pi0.5, smolvla (PaliGemma / SmolLM2) |
| `vlm_backbone` | GR00T (Eagle = SigLIP+Llama fused) |
| `projector` | most VLAs (action/state projection) |
| `vla_head` | every VLA (flow-matching / DiT / argmax) |
| `text_encoder` | DreamZero (T5) |

Concrete VLAs declare `REQUIRED_SLOTS` + `OPTIONAL_SLOTS` to enforce
which slots their composition needs. The spine raises at construction
if a required slot is missing.

Per decision S-3 (hybrid registration): subclasses register via the
`@VLAS.register` decorator from `reflex.registry.components`.

Per decision S-5: JSON-Schema validation defers to Phase 2 — V1 catches
typos at instantiation time via natural `TypeError` from missing kwargs.

Construction surface (lift #1 Day 4+):

    @VLAS.register
    class Pi05VLA(BaseVLA):
        REQUIRED_SLOTS = ("vision_backbone", "llm_backbone", "projector", "vla_head")
        OPTIONAL_SLOTS = ()
        NAME_MAPPING = {...}

        def forward(self, batch):
            ...

        def predict_action(self, images, state, instruction):
            ...

    pi05 = Pi05VLA.from_config({
        "vision_backbone": {"type": "SigLIPBackbone", "model_id": "..."},
        "llm_backbone": {"type": "PaliGemmaWithExpert", "model_id": "..."},
        "projector": {"type": "LinearProjector", ...},
        "vla_head": {"type": "FlowMatchingHead", ...},
    })
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from reflex.models._name_mapping import apply_name_mapping

if TYPE_CHECKING:
    import torch


# All component slot names recognized by the spine. Subclasses pick from
# this set via REQUIRED_SLOTS + OPTIONAL_SLOTS.
ALL_SLOTS: tuple[str, ...] = (
    "vision_backbone",
    "llm_backbone",
    "vlm_backbone",
    "projector",
    "vla_head",
    "text_encoder",
)


class BaseVLA(ABC):
    """Abstract spine for vision-language-action models.

    Concrete VLAs subclass BaseVLA + register via `@VLAS.register`. The
    constructor accepts component instances (any of the 6 slots, optional
    for the spine but required-per-VLA via REQUIRED_SLOTS) and stores them
    as attributes.

    `from_config(spec)` is the recommended construction path — it resolves
    type-tagged dicts through the Registry. Direct kwarg construction is
    supported for tests + dynamic patterns.

    Subclass contract:

    - **REQUIRED_SLOTS** (class var): slot names that MUST be provided to
      the constructor. Missing required slot raises ValueError.
    - **OPTIONAL_SLOTS** (class var): slot names that MAY be provided.
      Slots not in either set are rejected.
    - **NAME_MAPPING** (class var): per-VLA HF-checkpoint rename map
      (per decision S-1). Default empty dict = no renaming.
    - **forward(batch)** (abstract): the model's main forward pass.
    - **predict_action(...)** (abstract): inference-time action prediction.
      Signature is VLA-specific; the abstract method just declares it exists.

    Provided concretely:

    - **prepare_inference_weights()**: flatten all components' `prepare_triton`
      outputs into one dict. Used by `--inference-only-weights` mode in
      lift #3. Override only if your VLA needs custom weight aggregation.
    - **load_state_dict(state_dict, *, strict=False)**: apply NAME_MAPPING
      then load into the spine's components. Per decision S-1.
    """

    # ── Class-level contract — subclasses MUST override ────────────────
    REQUIRED_SLOTS: ClassVar[tuple[str, ...]] = ()
    OPTIONAL_SLOTS: ClassVar[tuple[str, ...]] = ()
    NAME_MAPPING: ClassVar[dict[str, str]] = {}

    # ── Construction ─────────────────────────────────────────────────────

    def __init__(
        self,
        *,
        vision_backbone: Any = None,
        llm_backbone: Any = None,
        vlm_backbone: Any = None,
        projector: Any = None,
        vla_head: Any = None,
        text_encoder: Any = None,
    ) -> None:
        # Subclasses must declare REQUIRED_SLOTS / OPTIONAL_SLOTS.
        # Validate the subclass declaration is sane.
        cls = type(self)
        declared = set(cls.REQUIRED_SLOTS) | set(cls.OPTIONAL_SLOTS)
        unknown = declared - set(ALL_SLOTS)
        if unknown:
            raise TypeError(
                f"{cls.__name__} declares unknown slot(s): {sorted(unknown)}. "
                f"Valid slots: {ALL_SLOTS}. Adding a new slot type requires "
                f"a spine extension proposal — see basevla-spine.md anti-goals."
            )
        overlap = set(cls.REQUIRED_SLOTS) & set(cls.OPTIONAL_SLOTS)
        if overlap:
            raise TypeError(
                f"{cls.__name__}: slot(s) {sorted(overlap)} appear in both "
                f"REQUIRED_SLOTS and OPTIONAL_SLOTS. Pick one."
            )

        # Bind passed components to attributes.
        passed = {
            "vision_backbone": vision_backbone,
            "llm_backbone": llm_backbone,
            "vlm_backbone": vlm_backbone,
            "projector": projector,
            "vla_head": vla_head,
            "text_encoder": text_encoder,
        }

        # Validate required slots are populated.
        missing = [
            slot for slot in cls.REQUIRED_SLOTS
            if passed.get(slot) is None
        ]
        if missing:
            raise ValueError(
                f"{cls.__name__} missing required slot(s): {missing}. "
                f"REQUIRED_SLOTS = {cls.REQUIRED_SLOTS}"
            )

        # Validate non-declared slots aren't populated (catches typos
        # where caller passes vlm_backbone to a 2-tower model).
        passed_slots = {s for s, v in passed.items() if v is not None}
        undeclared = passed_slots - declared
        if undeclared:
            raise ValueError(
                f"{cls.__name__} received component(s) for undeclared "
                f"slot(s): {sorted(undeclared)}. REQUIRED + OPTIONAL = "
                f"{sorted(declared)}"
            )

        # Bind all 6 slots (declared get the value, undeclared stay None).
        for slot in ALL_SLOTS:
            setattr(self, slot, passed.get(slot))

    # ── Construction-from-config (the spine's primary entry point) ──────

    @classmethod
    def from_config(cls, spec: dict[str, Any]) -> "BaseVLA":
        """Build a VLA instance from a type-tagged spec dict.

        Spec format:
            {
                "vision_backbone": {"type": "SigLIPBackbone", "model_id": "..."},
                "llm_backbone": {"type": "PaliGemmaWithExpert", ...},
                ...
            }

        Each slot value is either:
        - A type-tagged dict → resolved through the matching component
          Registry (VISION_BACKBONES / LLM_BACKBONES / etc.)
        - A pre-built component instance → passed through unchanged
        - None → slot stays unbound (and must be in OPTIONAL_SLOTS)

        Returns:
            A new instance of `cls`.

        Raises:
            ValueError: if a required slot is missing or an undeclared
                slot is provided.
            RegistryError: if a slot's `type` doesn't resolve in the
                matching component registry.
        """
        # Lazy import to avoid circular: components.py imports builder.py,
        # both depend on this file's __all__ exposure indirectly.
        from reflex.registry.builder import build_from_cfg
        from reflex.registry.components import (
            VISION_BACKBONES,
            LLM_BACKBONES,
            VLM_BACKBONES,
            PROJECTORS,
            VLA_HEADS,
            TEXT_ENCODERS,
        )

        slot_registries = {
            "vision_backbone": VISION_BACKBONES,
            "llm_backbone": LLM_BACKBONES,
            "vlm_backbone": VLM_BACKBONES,
            "projector": PROJECTORS,
            "vla_head": VLA_HEADS,
            "text_encoder": TEXT_ENCODERS,
        }

        kwargs: dict[str, Any] = {}
        for slot, value in spec.items():
            if slot not in ALL_SLOTS:
                raise ValueError(
                    f"Unknown slot {slot!r} in spec for {cls.__name__}. "
                    f"Valid: {ALL_SLOTS}"
                )
            if value is None:
                kwargs[slot] = None
            elif isinstance(value, dict) and "type" in value:
                kwargs[slot] = build_from_cfg(value, slot_registries[slot])
            else:
                # Pre-built instance — pass through.
                kwargs[slot] = value

        return cls(**kwargs)

    # ── Abstract methods — subclasses MUST implement ────────────────────

    @abstractmethod
    def forward(self, batch: Any) -> Any:
        """Forward pass — VLA-specific signature.

        Subclasses define the exact input/output shape. The spine just
        guarantees this method exists.
        """
        ...

    @abstractmethod
    def predict_action(
        self,
        *,
        images: Any,
        state: Any,
        instruction: str,
    ) -> Any:
        """Inference-time action prediction.

        Standard signature across VLAs to keep the runtime + chat surface
        uniform. VLA-specific details (chunk size, action_dim, denoise
        steps) are subclass concerns.
        """
        ...

    # ── Concrete methods — subclasses MAY override ───────────────────────

    def prepare_inference_weights(self, prefix: str = "") -> dict[str, "torch.Tensor"]:
        """Flatten all components' prepare_triton outputs into one dict.

        Used by `--inference-only-weights` mode (lift #3). Calls each
        bound component's `prepare_triton(prefix=<slot>.)` and merges
        the returned dicts.

        Components return empty dicts by default (the base ABCs ship a
        no-op `prepare_triton`); lift #3 fills in the real
        implementations per-component.

        Args:
            prefix: Optional prefix prepended to ALL keys. Useful when
                composing one VLA's weights into a larger dict.

        Returns:
            `{full_key: tensor}` — e.g.
            `{"vision_backbone.patch_embed.weight": <tensor>, ...}`.
            Empty if no components are bound or no components override
            `prepare_triton`.
        """
        flat: dict[str, "torch.Tensor"] = {}
        for slot in ALL_SLOTS:
            component = getattr(self, slot, None)
            if component is None:
                continue
            if not hasattr(component, "prepare_triton"):
                # Component base classes provide a default; user-supplied
                # components without it are silently skipped (caller can
                # validate themselves if they care).
                continue
            slot_prefix = f"{prefix}{slot}."
            sub_weights = component.prepare_triton(prefix=slot_prefix)
            # Defensive against accidental prefix collision (Lift #3 Day 2
            # spec). Silently letting one slot's weights overwrite another's
            # would corrupt inference. The slot_prefix design SHOULD prevent
            # this, but check anyway in case a user-supplied component
            # forgets to use the prefix.
            duplicates = set(flat) & set(sub_weights)
            if duplicates:
                raise ValueError(
                    f"prepare_inference_weights: duplicate key(s) when "
                    f"merging slot {slot!r}: {sorted(duplicates)[:5]}"
                    + (f" (and {len(duplicates) - 5} more)" if len(duplicates) > 5 else "")
                    + ". A component is not honoring the prefix kwarg."
                )
            flat.update(sub_weights)
        return flat

    def load_state_dict(
        self,
        state_dict: dict[str, "torch.Tensor"],
        *,
        strict: bool = False,
    ) -> None:
        """Apply NAME_MAPPING + route weights to components.

        Per decision S-1 — the NAME_MAPPING is per-VLA-config (class var).
        This method:

        1. Applies NAME_MAPPING to the input state_dict.
        2. Groups keys by slot prefix (the leading segment matches a slot
           name in ALL_SLOTS).
        3. For each slot, delegates `{slot-prefix}.x.y.z` → `component.x.y.z`
           via the component's own load_state_dict if present (PyTorch
           Modules support this natively; plain components must implement
           it explicitly).

        Args:
            state_dict: HF checkpoint state_dict.
            strict: If True, every key in the (renamed) state_dict must
                map to a bound component slot. False = unmatched keys
                are silently dropped (default for backward compatibility
                with HF checkpoints that ship extra metadata).
        """
        renamed = apply_name_mapping(state_dict, self.NAME_MAPPING)

        # Group by slot prefix.
        by_slot: dict[str, dict[str, "torch.Tensor"]] = {s: {} for s in ALL_SLOTS}
        unassigned: list[str] = []
        for key, tensor in renamed.items():
            slot_name, _, sub_key = key.partition(".")
            if slot_name in ALL_SLOTS and sub_key:
                by_slot[slot_name][sub_key] = tensor
            else:
                unassigned.append(key)

        if strict and unassigned:
            sample = unassigned[:5]
            more = f" (+ {len(unassigned) - 5} more)" if len(unassigned) > 5 else ""
            raise KeyError(
                f"{len(unassigned)} renamed key(s) don't start with a known "
                f"slot prefix. First 5: {sample}{more}. Valid slots: "
                f"{ALL_SLOTS}"
            )

        # Delegate per-slot loading.
        for slot, sub_state in by_slot.items():
            if not sub_state:
                continue
            component = getattr(self, slot, None)
            if component is None:
                if strict:
                    raise ValueError(
                        f"state_dict has keys for slot {slot!r} but no "
                        f"component is bound there in {type(self).__name__}."
                    )
                continue
            if hasattr(component, "load_state_dict"):
                # PyTorch nn.Module pattern: signature is
                # (state_dict, strict=True). Some custom components may
                # take only the state_dict. Try the rich signature first.
                try:
                    component.load_state_dict(sub_state, strict=strict)
                except TypeError:
                    # Fall back to single-arg signature.
                    component.load_state_dict(sub_state)


__all__ = ["BaseVLA", "ALL_SLOTS"]
