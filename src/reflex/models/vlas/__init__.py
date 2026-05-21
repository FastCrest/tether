"""Concrete VLA composition classes on the BaseVLA spine.

Each VLA family gets a class here that:

- Declares REQUIRED_SLOTS + OPTIONAL_SLOTS + NAME_MAPPING (per decision S-1)
- Implements forward() + predict_action()
- Provides from_pretrained() / from_config() construction helpers
- Registers via @VLAS.register

Day 4-9 adds:
- Day 4f: Pi0VLA — pi0 (PaliGemma + GemmaExpert + FlowMatchingHead)
- Day 5: Pi05VLA — pi0.5 (adds AdaRMSNorm)
- Day 6: SmolVLA — SmolVLA (different backbone, same head)
- Day 7: GR00TVLA — GR00T (uses vlm_backbone = Eagle + dit_head)
- Day 8: OpenVLA — STAYS in exporters/openvla.py as a shim (decision S-4)
- Phase 1.5: DreamZeroVLA (lift #7)
"""
from reflex.models.vlas.pi0 import Pi0VLA

__all__ = ["Pi0VLA"]
