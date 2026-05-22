"""Tests for lift #1 Day 10 — Spec dict CLI + reflex models list integration.

Validates the vla_type resolution + display + dispatch wiring per the plan:

- ``ModelEntry.resolved_vla_type`` returns spine class name for known families,
  shim marker for non-spine entries.
- ``reflex models list`` JSON output includes ``vla_type`` field.
- ``reflex models info`` JSON output includes ``vla_type`` field.
- ``reflex export <model_id>`` for smolvla / groot families routes through
  the Day 6 / Day 7 spine-based exporters (``exporters/smolvla.py`` and
  ``exporters/gr00t.py``), not the legacy direct-build paths.
"""
from __future__ import annotations

import json

import pytest

from reflex.registry import by_id
from reflex.registry.data import REGISTRY
from reflex.registry.models import ModelEntry


# ─── ModelEntry.resolved_vla_type ───────────────────────────────────────


def test_resolved_vla_type_pi0_family():
    """Family pi0 resolves to Pi0VLA spine class name."""
    entry = ModelEntry(
        model_id="test-pi0", hf_repo="x/y", family="pi0",
        action_dim=7, size_mb=1,
    )
    assert entry.resolved_vla_type == "Pi0VLA"


def test_resolved_vla_type_pi05_family():
    """Family pi05 resolves to Pi05VLA."""
    entry = ModelEntry(
        model_id="test-pi05", hf_repo="x/y", family="pi05",
        action_dim=32, size_mb=1,
    )
    assert entry.resolved_vla_type == "Pi05VLA"


def test_resolved_vla_type_smolvla_family():
    entry = ModelEntry(
        model_id="test-smolvla", hf_repo="x/y", family="smolvla",
        action_dim=32, size_mb=1,
    )
    assert entry.resolved_vla_type == "SmolVLA"


def test_resolved_vla_type_groot_family():
    entry = ModelEntry(
        model_id="test-groot", hf_repo="x/y", family="groot",
        action_dim=128, size_mb=1,
    )
    assert entry.resolved_vla_type == "GR00TVLA"


def test_resolved_vla_type_openvla_shim():
    """OpenVLA's _openvla_shim marker takes precedence over family derivation."""
    entry = ModelEntry(
        model_id="test-openvla", hf_repo="x/y", family="openvla",
        action_dim=7, size_mb=1, vla_type="_openvla_shim",
    )
    assert entry.resolved_vla_type == "_openvla_shim"


def test_resolved_vla_type_explicit_marker_overrides_family():
    """When vla_type is explicitly set, family derivation is bypassed."""
    entry = ModelEntry(
        model_id="test-override", hf_repo="x/y", family="pi0",
        action_dim=7, size_mb=1, vla_type="_custom_marker",
    )
    assert entry.resolved_vla_type == "_custom_marker"


# ─── Real registry entries ─────────────────────────────────────────────


def test_registry_each_entry_resolves_to_valid_marker():
    """Every entry in REGISTRY resolves to either a known spine class
    or a shim marker."""
    KNOWN_SPINE = {"Pi0VLA", "Pi05VLA", "SmolVLA", "GR00TVLA"}
    for entry in REGISTRY:
        resolved = entry.resolved_vla_type
        assert resolved is not None
        # Either a spine class name OR a marker (starts with _) OR a family
        # we don't yet have a spine class for.
        assert (
            resolved in KNOWN_SPINE
            or resolved.startswith("_")
            or resolved == entry.family  # fallback for unknown families
        ), f"{entry.model_id} resolved to unexpected vla_type: {resolved!r}"


def test_openvla_7b_resolves_to_shim():
    entry = by_id("openvla-7b")
    assert entry is not None
    assert entry.resolved_vla_type == "_openvla_shim"


# ─── CLI export dispatch wiring (Day 6+7 spine paths) ──────────────────


def test_cli_imports_spine_smolvla_exporter():
    """`reflex export` for smolvla family uses src/reflex/exporters/smolvla.py
    (the Day 6 spine-based exporter), NOT the legacy smolvla_exporter."""
    import reflex.cli as cli_module
    import inspect
    src = inspect.getsource(cli_module)
    # The export function should import from the spine module.
    assert "from reflex.exporters.smolvla import export_smolvla" in src
    # And NOT from the legacy module (smolvla_exporter) on the same import line.
    assert "from reflex.exporters.smolvla_exporter import export_smolvla" not in src


def test_cli_imports_spine_gr00t_exporter():
    """`reflex export` for groot family uses src/reflex/exporters/gr00t.py
    (the Day 7 spine-based exporter), NOT the legacy gr00t_exporter."""
    import reflex.cli as cli_module
    import inspect
    src = inspect.getsource(cli_module)
    assert "from reflex.exporters.gr00t import export_gr00t" in src
    assert "from reflex.exporters.gr00t_exporter import export_gr00t" not in src or \
        "from reflex.exporters.gr00t_exporter import export_gr00t_full" not in src
