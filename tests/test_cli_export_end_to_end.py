"""Regression test for `cli-export-end-to-end` GOALS.yaml gate.

The CLI's `tether export --monolithic` path is the cos=1.0 verified
production export. Running a real export requires a GPU + ~15 min +
the `[monolithic]` extras, so that can't live in unit tests — the
Modal harness (`scripts/modal_{smolvla,pi0}_monolithic_export.py`) is
the full-run reproducer.

This test verifies:
1. `tether.exporters.monolithic` imports cleanly (module structure ok)
2. `export_monolithic` dispatches by model_id
3. `_require_monolithic_deps()` emits a useful error when transformers
   is at the wrong version or a dep is missing

If this fails, something in the extraction broke — investigate before
releasing.
"""
from __future__ import annotations

import logging
import sys
import types

import pytest


def test_monolithic_module_importable():
    """The module must be importable even without the [monolithic] extras
    installed (it only checks at call time)."""
    from tether.exporters import monolithic
    assert hasattr(monolithic, "export_monolithic")
    assert hasattr(monolithic, "export_smolvla_monolithic")
    assert hasattr(monolithic, "export_pi0_monolithic")
    assert hasattr(monolithic, "apply_export_patches")


def test_dispatch_by_model_id():
    """`export_monolithic` picks the right backend from the model_id."""
    from tether.exporters import monolithic

    with pytest.raises(ImportError, match="Missing dependencies|monolithic"):
        # Wrong transformers version or missing deps -> ImportError
        monolithic.export_monolithic(
            "lerobot/smolvla_base", "/tmp/should_not_run",
        )


def test_unsupported_model_type_raises():
    """A model id that doesn't match any family substring AND has no
    readable local config.json must raise a clean ValueError. (GR00T,
    pi0.5, smolvla all hit the substring matcher now — pick a name
    that doesn't.)"""
    from tether.exporters import monolithic

    with pytest.raises(ValueError, match="Cannot infer model_type"):
        monolithic.export_monolithic(
            "some-org/totally-unknown-architecture", "/tmp/out",
        )


def test_dep_check_catches_wrong_transformers(monkeypatch):
    """_require_monolithic_deps() raises with a helpful message if
    transformers is the wrong version."""
    from tether.exporters import monolithic
    import transformers

    monkeypatch.setattr(transformers, "__version__", "4.99.0")
    with pytest.raises(ImportError, match="5.3.0"):
        monolithic._require_monolithic_deps()


def test_dep_check_catches_wrong_lerobot(monkeypatch):
    """_require_monolithic_deps() must fail early on the known-bad
    lerobot 0.4.x stack instead of letting torch.export fail downstream."""
    from tether.exporters import monolithic
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.3.0")
    for mod_name in ("lerobot", "onnx_diagnostic", "onnxscript", "optree", "scipy"):
        monkeypatch.setitem(sys.modules, mod_name, types.ModuleType(mod_name))

    def _fake_dist_version(dist_name: str) -> str:
        if dist_name == "lerobot":
            return "0.4.4"
        raise monolithic.PackageNotFoundError(dist_name)

    monkeypatch.setattr(monolithic, "_dist_version", _fake_dist_version)

    with pytest.raises(ImportError, match=r"lerobot 0\.4\.4.*lerobot==0\.5\.1"):
        monolithic._require_monolithic_deps()


def test_smolvla_export_patch_failure_is_warning_then_fatal(caplog):
    """SmolVLA-specific export patches should not hide at debug level.

    A failed patch is logged immediately and converted into a clear RuntimeError
    before SmolVLA torch.export can hit a cryptic FakeTensor shape error.
    """
    from tether.exporters import monolithic

    monolithic._SMOLVLA_EXPORT_PATCH_FAILURES.clear()
    try:
        with caplog.at_level(logging.WARNING, logger="tether.exporters.monolithic"):
            monolithic._record_smolvla_export_patch_failure(
                "SmolVLA explicit patch-mask export patch",
                RuntimeError("boom"),
            )

        assert any("export patch failed" in rec.message for rec in caplog.records)
        with pytest.raises(RuntimeError, match="SmolVLA monolithic export patches failed"):
            monolithic._raise_if_smolvla_export_patches_failed()
    finally:
        monolithic._SMOLVLA_EXPORT_PATCH_FAILURES.clear()
