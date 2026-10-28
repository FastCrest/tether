"""Conservative runtime qualification facts for Tether model families.

These facts describe checked runtime paths. Registry presence alone does not
mean that export, training, evaluation, or a target device has been qualified.
Studio reads this file without importing heavyweight model dependencies.
"""

from __future__ import annotations

import copy

# Where each "qualified-*" status comes from. A qualified export names a passing
# Linux-CUDA parity receipt; a qualified evaluation names a matched result with a
# non-zero success count for the qualified arm. tests/test_registry_qualification.py
# fails when a status and its evidence disagree in either direction.
# Literal strings only: Studio reads this dict by AST (backend/tether_registry.py).
# Studio paths are relative to the Tether Studio repository. Studio binds each
# receipt to Tether pin 432923b (tether_private/qualification/committed_parity.py
# BOUND_PRODUCERS); a later Tether pin needs Studio's export-path equality proof
# re-run before the receipt binds there.


FAMILY_QUALIFICATIONS: dict[str, dict[str, object]] = {
    "smolvla": {
        "checkpoint": "qualified-pinned",
        "export": "qualified-linux-cuda-parity",
        "training": "qualified-lora",
        "evaluation": "harness-verified-libero-linux-cuda",
        "studio_recipe": "tether-smolvla-lora-v1",
        "notes": [
            "Local native LIBERO execution requires Linux and CUDA.",
            (
                "2026-10-28: export qualified by a Linux-CUDA receipt (A10G, ORT-CUDA 1.23.2, "
                "CPU fallback disabled) for smolvla_base@c83c3163 over 8 LIBERO-10 "
                "instructions: worst full cosine 0.99999992, max_abs 2.2e-3 (p99 4.4e-4). "
                "TF32 accounts for the error: with NVIDIA_TF32_OVERRIDE=0 the same graph "
                "gives max_abs 3.1e-6."
            ),
            (
                "2026-10-28: the monolithic graph has a static 16-token language input while "
                "the checkpoint preprocessor pads to 48; 2 of 10 LIBERO-10 instructions "
                "(20 and 22 tokens) do not fit and were refused, not compared."
            ),
            ("2026-10-28: evaluation corrected from qualified-libero-linux-cuda. Matched "
            "libero_10 n=20 per arm: smolvla_base@c83c3163 0/20 and the "
            "tether-smolvla-lora-v1 step-5000 LoRA 0/20 (all 530-step timeouts). The "
            "harness itself is verified: lerobot/smolvla_libero@a24f1f38 scored 14/20 on "
            "the same 20 cases (SmolVLA paper LIBERO-Long 71%). No Tether-produced "
            "SmolVLA arm has a non-zero matched success, so evaluation is not qualified."),
            ("2026-10-28: the LoRA training receipt shows the recipe runs; the step-5000 "
            "adapter did not learn libero_10 (0/20), so training efficacy is unproven."),
        ],
        "evidence": {
            "export": {
                "receipt": "modal volume parity-gpu-receipts:smolvla-c83c3163-20261028/smolvla/smolvla-export-parity-receipt.json",
                "receipt_sha256": "aafcb61db5c917c5bddd469a0289e0b8376532477d0664544995ef781324c87a",
                "model": "lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205",
                "tether_commit": "16b50c63aacdee37d8d4dd40fd812403a4a912ef",
                "verdict": "passed",
                "scope": "linux-cuda",
            },
            "evaluation": {
                "suite": "libero_10",
                "cases": 20,
                "arms": {
                    "lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205": [0, 20],
                    "tether-smolvla-lora-v1 step 5000 (adapter sha256 0d3dc759...574c)": [0, 20],
                },
                "control": {"lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d": [14, 20]},
                "qualified_arm": None,
                "record": "Studio docs/evidence/smolvla-qualification-20261027/README.md "
                "(feat/smolvla-policy-import @ 0969d86d)",
            },
        },
        "acceptance": [
            "pinned checkpoint",
            "LoRA training receipt",
            "matched LIBERO development",
            "matched LIBERO held-out",
        ],
    },
    "pi0": {
        "checkpoint": "runtime-implemented",
        "export": "qualified-linux-cuda-parity",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": [
            "Do not qualify from registry metadata alone.",
            ("2026-10-28: export corrected from verification-pending. Studio commits a "
            "passing Linux-CUDA receipt (A100-40GB, CUDA EP, CPU fallback disabled): "
            "full cosine 0.99999973, max_abs 6.9e-4."),
        ],
        "evidence": {
            "export": {
                "receipt": "tether_private/qualification/parity_evidence/2026-10-12/pi0/pi0-export-parity-receipt.json",
                "receipt_sha256": "9dc91d12a8df9024f761ff56d16c20abe50d00b6ee4c6068d9be0ea9d3886488",
                "model": "lerobot/pi0_base@26b99b9439acb1e352439e34ee9c67af0d76efa3",
                "tether_commit": "b7223069a4817f043f0cb15f1ba0d65ef2475763",
                "verdict": "passed",
                "scope": "linux-cuda",
            },
        },
        "acceptance": [
            "pinned checkpoint",
            "shared-noise export parity receipt",
            "matched task development",
            "matched task held-out",
        ],
    },
    "pi05": {
        "checkpoint": "runtime-implemented",
        "export": "qualified-linux-cuda-parity",
        "training": "snapflow-hardware-gated",
        "evaluation": "runtime-evidence-only",
        "studio_recipe": None,
        "notes": [
            "Studio has no accepted pi0.5 training and decision journey.",
            ("2026-10-28: export corrected from runtime-implemented. Studio commits a "
            "passing Linux-CUDA receipt: full cosine 0.99999990, max_abs 5.2e-4."),
        ],
        "evidence": {
            "export": {
                "receipt": "tether_private/qualification/parity_evidence/2026-10-12/pi05/pi05-export-parity-receipt.json",
                "receipt_sha256": "628b05ab0b00fa3073ec30a15d28a44630f83c248149b684bd613c32e8e05c66",
                "model": "lerobot/pi05_base@a538eb273274eb30f126a118f39dbc0ee212c883",
                "tether_commit": "b7223069a4817f043f0cb15f1ba0d65ef2475763",
                "verdict": "passed",
                "scope": "linux-cuda",
            },
        },
        "acceptance": [
            "pinned teacher and student",
            "export parity receipt",
            "matched LIBERO development",
            "matched LIBERO held-out",
        ],
    },
    "groot": {
        "checkpoint": "runtime-implemented",
        "export": "qualified-linux-cuda-parity-strict-fp32",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": [
            "GPU export parity does not establish task success.",
            ("2026-10-28: export corrected from verification-pending. The receipt passes "
            "only with NVIDIA_TF32_OVERRIDE=0 (full cosine 0.9999999999994, max_abs 2.0e-6); "
            "the default-TF32 receipt groot/groot-export-parity-receipt.json failed and TF32 "
            "execution is not qualified."),
        ],
        "evidence": {
            "export": {
                "receipt": "tether_private/qualification/parity_evidence/2026-09-18/groot/groot-export-parity-receipt-strict-tf32.json",
                "receipt_sha256": "a34fcf84471c7842192d35c9af8975984840aa2b97f530424ca1c9d132d21d0e",
                "model": "nvidia/GR00T-N1.6-3B@d0814e7ecb19202e7c8468b46098b0b7ef3a6d61",
                "tether_commit": "4cf9854a72ca9c97ec6a7a398f4246650b001d2a",
                "verdict": "passed",
                "scope": "linux-cuda",
                "condition": "NVIDIA_TF32_OVERRIDE=0",
            },
        },
        "acceptance": [
            "pinned checkpoint",
            "export parity receipt",
            "matched task development",
            "matched task held-out",
        ],
    },
    "openvla": {
        "checkpoint": "runtime-implemented",
        "export": "qualified-linux-cuda-parity",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": [
            "The tokenized action path needs its own matched evaluation contract.",
            ("2026-10-28: export corrected from verification-pending. Studio commits a "
            "passing Linux-CUDA receipt with 7/7 action tokens agreeing and max_abs 0.0."),
        ],
        "evidence": {
            "export": {
                "receipt": "tether_private/qualification/parity_evidence/2026-09-18/openvla/openvla-export-parity-receipt.json",
                "receipt_sha256": "99ab67d70343153c9dbec5a0f9509b7e4c22ee3a24b4262502d14ca41f8b2430",
                "model": "openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f",
                "tether_commit": "6e84101136dcaa6204983956b80418796430504a",
                "verdict": "passed",
                "scope": "linux-cuda",
            },
        },
        "acceptance": [
            "pinned checkpoint",
            "tokenized-action parity receipt",
            "matched task development",
            "matched task held-out",
        ],
    },
    "dreamzero": {
        "checkpoint": "registry-only",
        "export": "not-qualified",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["Requires a 40 GB class GPU and a verified exporter path."],
        "acceptance": [
            "pinned checkpoint",
            "verified exporter",
            "task adapter",
            "matched development and held-out evidence",
        ],
    },
    "molmoact2": {
        "checkpoint": "registry-only",
        "export": "not-qualified",
        "training": "not-qualified",
        "evaluation": "not-qualified",
        "studio_recipe": None,
        "notes": ["Registry metadata has not been accepted as execution evidence."],
        "acceptance": [
            "pinned checkpoint",
            "verified exporter",
            "task adapter",
            "matched development and held-out evidence",
        ],
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
    return copy.deepcopy(facts)


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
    acceptance = facts.get("acceptance")
    if not isinstance(acceptance, list):
        return []
    return [str(item) for item in acceptance]


__all__ = ["FAMILY_QUALIFICATIONS", "qualification_for", "qualification_gaps"]
