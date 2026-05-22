"""Tests for lift #1 Day 9 — pi0_prefix exporter rename.

Pins the rename from ``reflex.exporters.pi0_prefix_exporter`` →
``reflex.exporters.pi0_prefix``. The module's classes and builders are
spine-load-bearing (``Pi0ExpertStackWithPrefix`` / ``Pi05ExpertStackWithPrefix``
are what ``Pi0VLA.from_pretrained`` and ``Pi05VLA.from_pretrained`` construct
under the hood per Day 4h / 5b parity gates).

After this rename, the spine VLAs (pi0 + pi0.5) + scripts (modal parity)
+ runtime composition all import from the new path. Old name is
intentionally NOT kept as an alias — the rename is a clean break per
the lift plan.
"""
from __future__ import annotations

import importlib

import pytest


def test_pi0_prefix_module_at_new_path():
    """Day 9 rename: imports resolve from ``reflex.exporters.pi0_prefix``."""
    mod = importlib.import_module("reflex.exporters.pi0_prefix")
    assert mod is not None
    # Sanity: the load-bearing builders are exposed.
    assert hasattr(mod, "build_pi0_expert_with_prefix")
    assert hasattr(mod, "build_pi05_expert_with_prefix")
    assert hasattr(mod, "Pi0ExpertStackWithPrefix")
    assert hasattr(mod, "Pi05ExpertStackWithPrefix")
    assert hasattr(mod, "export_pi0_prefix")


def test_pi0_prefix_old_name_removed():
    """Day 9 cleanup: old ``pi0_prefix_exporter`` module is GONE (no alias).
    The rename is a clean break — callers must use the new name. This pins
    that we don't accidentally keep both around (would defeat the cleanup
    pre Day 11)."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("reflex.exporters.pi0_prefix_exporter")


def test_spine_vlas_import_from_new_path():
    """Pi0VLA + Pi05VLA + FlowMatchingHead use the renamed module."""
    # These imports must resolve through the spine without ImportError.
    from reflex.models.vlas.pi0 import Pi0VLA
    from reflex.models.vlas.pi05 import Pi05VLA
    from reflex.models.heads.flow_matching_head import FlowMatchingHead

    assert Pi0VLA is not None
    assert Pi05VLA is not None
    assert FlowMatchingHead is not None
