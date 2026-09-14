"""Conservative runtime qualification facts for Tether model families.

These facts describe checked runtime paths. Registry presence alone does not
mean that export, training, evaluation, or a target device has been qualified.
Studio reads this file without importing heavyweight model dependencies.
"""

from __future__ import annotations


FAMILY_QUALIFICATIONS: dict[str, dict[str, object]] = {
    "smolvla": {
        "checkpoint": "qualified-pinned",
        "export": "runtime-implemented",
        "training": "qualified-lora",
        "evaluation": "qualified-libero-linux-cuda",
        "studio_recipe": "tether-smolvla-lora-v1",
        "notes": [
            "Studio has retained real SmolVLA LoRA and native LIBERO evidence.",
            "Local native LIBERO execution requires Linux and CUDA.",
        ],
    },
    "pi0": {
        "checkpoint": "runtime-implemented",
        "export": "verification-pending",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["Do not qualify from registry metadata alone."],
    },
    "pi05": {
        "checkpoint": "runtime-implemented",
        "export": "runtime-implemented",
        "training": "snapflow-hardware-gated",
        "evaluation": "runtime-evidence-only",
        "studio_recipe": None,
        "notes": ["Studio has no accepted pi0.5 training and decision journey."],
    },
    "groot": {
        "checkpoint": "runtime-implemented",
        "export": "verification-pending",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["GPU export parity does not establish task success."],
    },
    "openvla": {
        "checkpoint": "runtime-implemented",
        "export": "verification-pending",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["The tokenized action path needs its own matched evaluation contract."],
    },
    "dreamzero": {
        "checkpoint": "registry-only",
        "export": "not-qualified",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["Requires a 40 GB class GPU and a verified exporter path."],
    },
    "molmoact2": {
        "checkpoint": "registry-only",
        "export": "not-qualified",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["Registry metadata has not been accepted as execution evidence."],
    },
}


def qualification_for(family: str) -> dict[str, object]:
    """Return a copy so callers cannot mutate the process-wide contract."""

    facts = FAMILY_QUALIFICATIONS.get(family)
    if facts is None:
        return {
            "checkpoint": "unqualified",
            "export": "unqualified",
            "training": "unqualified",
            "evaluation": "unqualified",
            "studio_recipe": None,
            "notes": ["No qualification contract is recorded for this family."],
        }
    return {key: list(value) if isinstance(value, list) else value for key, value in facts.items()}


__all__ = ["FAMILY_QUALIFICATIONS", "qualification_for"]
