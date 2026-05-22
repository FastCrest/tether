"""Tests for OpenVLA — non-spine shim (lift #1 decision S-4, Day 8).

OpenVLA stays a shim with the optimum-cli + bin-to-continuous postprocess
flow. Spine never composes it. This test file pins S-4:

- The exporter module lives at ``reflex.exporters.openvla`` (renamed from
  ``openvla_exporter`` in Day 8)
- The exporter raises ``NotImplementedError`` with a hint pointing at
  the optimum-cli + decode_actions path
- ``ModelEntry.vla_type`` is ``_openvla_shim`` on the registry row
- No ``OpenVLAVLA`` class is registered on the BaseVLA spine
"""
from __future__ import annotations

import pytest

from reflex.registry.components import VLAS
from reflex.registry.data import REGISTRY
from reflex.registry.models import ModelEntry


# ─── S-4: OpenVLA is not on the spine ──────────────────────────────────


def test_openvla_not_registered_on_spine():
    """No OpenVLAVLA class on the BaseVLA spine — that's the S-4 decision."""
    assert "OpenVLAVLA" not in VLAS, (
        "OpenVLA must remain a shim per decision S-4; do not add a spine class."
    )


def test_openvla_modelentry_marks_shim():
    """Registry row sets vla_type='_openvla_shim' to flag non-spine status."""
    openvla = next((m for m in REGISTRY if m.model_id == "openvla-7b"), None)
    assert openvla is not None, "openvla-7b missing from REGISTRY"
    assert openvla.family == "openvla"
    assert openvla.vla_type == "_openvla_shim"


def test_spine_models_have_no_vla_type_marker():
    """Spine VLAs (pi0/pi05/smolvla/groot) leave vla_type=None.
    The BaseVLA registry name is the source of truth for those."""
    for entry in REGISTRY:
        if entry.family == "openvla":
            continue
        assert entry.vla_type is None, (
            f"{entry.model_id}: spine families must NOT set vla_type marker "
            f"(got {entry.vla_type!r}); only non-spine shims (OpenVLA) do."
        )


def test_vla_type_marker_must_start_with_underscore():
    """Convention: marker types are prefixed with `_` so they don't collide
    with real spine VLA class names (Pi0VLA, Pi05VLA, etc)."""
    with pytest.raises(ValueError, match="must start with"):
        ModelEntry(
            model_id="bogus",
            hf_repo="x/y",
            family="openvla",
            action_dim=7,
            size_mb=1,
            vla_type="NotAShimMarker",  # missing leading `_`
        )


# ─── Module rename: exporters/openvla_exporter.py → exporters/openvla.py ──


def test_openvla_module_at_new_path():
    """Day 8 renamed openvla_exporter.py → openvla.py. Verify the new
    import path resolves; the cli.py dispatch was updated to match."""
    from reflex.exporters import openvla as openvla_module
    assert hasattr(openvla_module, "export_openvla")


def test_export_openvla_raises_with_hint():
    """The shim's export_openvla raises NotImplementedError pointing
    at the optimum-cli + decode_actions path."""
    from reflex.config import ExportConfig
    from reflex.exporters.openvla import export_openvla

    cfg = ExportConfig(
        model_id="openvla/openvla-7b",
        output_dir="/tmp/openvla_test",
        target="a10g",
    )
    with pytest.raises(NotImplementedError) as excinfo:
        export_openvla(cfg)
    msg = str(excinfo.value)
    assert "optimum-cli" in msg
    assert "decode_actions" in msg
