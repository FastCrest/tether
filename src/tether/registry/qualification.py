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
        "acceptance": ["pinned checkpoint", "LoRA training receipt", "matched LIBERO development", "matched LIBERO held-out"],
    },
    "pi0": {
        "checkpoint": "runtime-implemented",
        "export": "verification-pending",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": [
            "Do not qualify from registry metadata alone.",
            "An exact-revision/shared-input ONNX parity receipt harness exists; retained Linux-CUDA acceptance is still pending.",
        ],
        "acceptance": ["pinned checkpoint", "shared-noise export parity receipt", "matched task development", "matched task held-out"],
    },
    "pi05": {
        "checkpoint": "runtime-implemented",
        "export": "runtime-implemented",
        "training": "snapflow-hardware-gated",
        "evaluation": "runtime-evidence-only",
        "studio_recipe": None,
        "notes": ["Studio has no accepted pi0.5 training and decision journey."],
        "acceptance": ["pinned teacher and student", "export parity receipt", "matched LIBERO development", "matched LIBERO held-out"],
    },
    "groot": {
        "checkpoint": "runtime-implemented",
        "export": "verification-pending",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["GPU export parity does not establish task success."],
        "acceptance": ["pinned checkpoint", "export parity receipt", "matched task development", "matched task held-out"],
    },
    "openvla": {
        "checkpoint": "runtime-implemented",
        "export": "verification-pending",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["The tokenized action path needs its own matched evaluation contract."],
        "acceptance": ["pinned checkpoint", "tokenized-action parity receipt", "matched task development", "matched task held-out"],
    },
    "dreamzero": {
        "checkpoint": "registry-only",
        "export": "not-qualified",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["Requires a 40 GB class GPU and a verified exporter path."],
        "acceptance": ["pinned checkpoint", "verified exporter", "task adapter", "matched development and held-out evidence"],
    },
    "molmoact2": {
        "checkpoint": "registry-only",
        "export": "not-qualified",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["Registry metadata has not been accepted as execution evidence."],
        "acceptance": ["pinned checkpoint", "verified exporter", "task adapter", "matched development and held-out evidence"],
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
            "acceptance": ["pinned checkpoint", "verified execution receipt"],
        }
    return {key: list(value) if isinstance(value, list) else value for key, value in facts.items()}


def qualification_gaps(family: str) -> list[str]:
    """Return acceptance requirements when a family is not fully qualified."""

    facts = qualification_for(family)
    statuses = (
        str(facts.get("checkpoint")),
        str(facts.get("export")),
        str(facts.get("training")),
        str(facts.get("evaluation")),
    )
    if all(value.startswith("qualified") for value in statuses):
        return []
    return list(facts.get("acceptance") or [])


__all__ = ["FAMILY_QUALIFICATIONS", "qualification_for", "qualification_gaps"]