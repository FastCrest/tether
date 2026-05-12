"""Registry-completeness contract — every primary exporter has at least
one ModelEntry in `reflex.registry.data.REGISTRY`.

The bug this catches: customer reports 2026-05-10 found GR00T + OpenVLA
exporters had shipped without registry entries → `reflex models list`,
`reflex chat`, and `reflex doctor` all said "not supported" even though
the underlying export pipelines worked. README claimed support; CLI
discovery surface didn't know.

Without this test, every future exporter shipped to `src/reflex/exporters/`
silently regresses the discovery surface. Anyone running `reflex models
list` won't see the new model.

Internal-only exporters (helpers consumed by other exporters, not
directly customer-facing) are exempt via the EXPORTERS_INTERNAL set.
Adding to the exempt list requires a docstring justification per the
"explicit exemption" pattern.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from reflex.registry.data import REGISTRY

EXPORTERS_DIR = Path(__file__).parent.parent / "src" / "reflex" / "exporters"

# Exporters that are NOT directly customer-facing — they're shared helpers
# consumed by the primary exporters. Each must have a docstring explaining
# WHY it's internal so future contributors don't add to this set casually.
EXPORTERS_INTERNAL = {
    "decomposed.py",          # pi05 + pi05-libero use this; not standalone
    "eagle_export_stack.py",  # GR00T VLM stack; consumed by gr00t_exporter
    "fp16_convert.py",        # FP32 → FP16 weight converter utility
    "monolithic.py",          # path-spec for all monolithic exports
    "onnx_export.py",         # generic torch.onnx.export wrapper
    "pi0_prefix_exporter.py", # SigLIP/Gemma split; internal R&D path
    "vlm_components.py",      # text embedder + state encoder helpers
    "vlm_prefix_exporter.py", # vlm prefix exporter; consumed via decomposed
    "_export_mode.py",        # parallel/sequential mode selector
    "trt_build.py",           # TensorRT engine builder; consumed by all primary exporters
    "__init__.py",
}

# Primary exporters → expected registry family. Add a row here when
# shipping a new exporter; the test below enforces the registry has at
# least one entry with the matching family.
EXPECTED_FAMILIES = {
    # Validation in models.py uses 'groot' (existing convention).
    # NVIDIA's brand is "GR00T" with zeros but our family slug stays
    # consistent with the validator until/unless we bump the validator.
    "gr00t_exporter.py": "groot",
    "openvla_exporter.py": "openvla",
    "pi0_exporter.py": "pi0",
    "smolvla_exporter.py": "smolvla",
    # pi05 is exported via decomposed.py (which has _export_pi05_*
    # callsites). It's listed as "internal" above but the registry must
    # still cover the pi05 family because that's the primary user-facing
    # surface for pi05-base / pi05-libero.
    "_pi05_via_decomposed": "pi05",
}


def _list_exporter_files() -> list[str]:
    """Names (basename only) of all .py files in src/reflex/exporters/."""
    return [
        p.name for p in EXPORTERS_DIR.iterdir()
        if p.suffix == ".py"
    ]


def test_every_primary_exporter_has_registry_entry():
    """For every customer-facing exporter, at least one ModelEntry
    in REGISTRY uses the matching family. Catches the GR00T/OpenVLA
    class of bug at CI time."""
    families_in_registry = {entry.family for entry in REGISTRY}
    missing = []
    for exporter_file, expected_family in EXPECTED_FAMILIES.items():
        if expected_family not in families_in_registry:
            missing.append(
                f"{exporter_file} → no registry entry with family={expected_family!r}"
            )
    assert not missing, (
        "Primary exporter(s) missing registry entry — `reflex models list` / "
        "`reflex chat` / `reflex doctor` will say 'not supported' for these "
        "even though the export pipeline works:\n  "
        + "\n  ".join(missing)
        + "\nFix: add a ModelEntry to src/reflex/registry/data.py"
    )


def test_exporter_directory_audit_covers_all_files():
    """Every file in src/reflex/exporters/ must be either:
      (a) classified as internal (EXPORTERS_INTERNAL), OR
      (b) classified as primary with expected family (EXPECTED_FAMILIES)
    Catches the case where someone adds a NEW exporter file but
    forgets to update either set, leaving its registry-coverage
    untested."""
    actual_files = set(_list_exporter_files())
    classified_files = EXPORTERS_INTERNAL | set(EXPECTED_FAMILIES.keys())
    # Drop the synthetic _pi05_via_decomposed marker
    classified_files.discard("_pi05_via_decomposed")

    unclassified = actual_files - classified_files
    assert not unclassified, (
        f"New exporter file(s) found that aren't classified as internal "
        f"or primary:\n  {sorted(unclassified)}\n"
        f"Fix: add to either EXPORTERS_INTERNAL (with docstring why) or "
        f"EXPECTED_FAMILIES (with the registry family it should map to) "
        f"in tests/test_registry_completeness.py."
    )

    # Inverse — every entry in our maps must correspond to a real file
    # (catches typos + drift when a file is renamed/deleted).
    stale = classified_files - actual_files
    assert not stale, (
        f"Classification map references file(s) that no longer exist:\n"
        f"  {sorted(stale)}\n"
        f"Fix: remove these from EXPORTERS_INTERNAL / EXPECTED_FAMILIES."
    )


def test_registry_entries_have_required_fields():
    """Every ModelEntry must populate the fields downstream consumers
    rely on (model_id, hf_repo, family, action_dim, size_mb, etc.)."""
    for entry in REGISTRY:
        assert entry.model_id, f"empty model_id in entry {entry}"
        assert entry.hf_repo, f"empty hf_repo for {entry.model_id}"
        assert entry.family, f"empty family for {entry.model_id}"
        assert entry.action_dim > 0, (
            f"action_dim must be positive for {entry.model_id}, got {entry.action_dim}"
        )
        assert entry.size_mb > 0, (
            f"size_mb must be positive for {entry.model_id}, got {entry.size_mb}"
        )
        assert entry.supported_embodiments, (
            f"supported_embodiments empty for {entry.model_id} — at least one required"
        )
        assert entry.supported_devices, (
            f"supported_devices empty for {entry.model_id} — at least one required"
        )
        assert entry.description, f"empty description for {entry.model_id}"


def test_gr00t_n16_in_registry():
    """Specific assertion for the 2026-05-10 customer report — GR00T
    N1.6 must surface in `reflex models list`."""
    matching = [e for e in REGISTRY if e.family == "groot"]
    assert matching, (
        "GR00T family missing from registry — customer 2026-05-10 reported "
        "this exact bug. Add nvidia/GR00T-N1.6-3B entry."
    )
    assert any("N1.6" in e.hf_repo for e in matching), (
        "GR00T family present but no N1.6 variant — README claims N1.6 "
        "support with measured cos parity."
    )


def test_openvla_in_registry():
    """Specific assertion for the second gap surfaced 2026-05-10 —
    OpenVLA exporter shipped without registry entry."""
    matching = [e for e in REGISTRY if e.family == "openvla"]
    assert matching, (
        "OpenVLA family missing from registry. README claims support via "
        "optimum-cli + reflex.postprocess.openvla.decode_actions."
    )
    assert any("openvla-7b" in e.hf_repo.lower() for e in matching), (
        "OpenVLA family present but openvla-7b not in registry."
    )
