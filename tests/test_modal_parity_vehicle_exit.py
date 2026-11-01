"""A Modal parity vehicle must exit non-zero on every outcome that is not a pass.

Loads each vehicle with a stub `modal` module, so no Modal account or GPU is needed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
VEHICLES = [
    "modal_pi0_export_parity_pinned.py",
    "modal_pi05_export_parity_pinned.py",
    "modal_groot_export_parity_pinned.py",
    "modal_openvla_export_cpu_pinned.py",
    "modal_openvla_export_parity_pinned.py",
]


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Stub()

    def __call__(self, *args, **kwargs):
        return _Stub()


class _App:
    def __init__(self, *args, **kwargs):
        pass

    def function(self, **kwargs):
        def wrap(fn):
            fn.remote = None  # set per test
            return fn

        return wrap

    def local_entrypoint(self):
        return lambda fn: fn


def _load(name, monkeypatch):
    modal = types.ModuleType("modal")
    modal.App = _App
    modal.Image = _Stub()
    modal.Volume = _Stub()
    modal.Secret = _Stub()
    monkeypatch.setitem(sys.modules, "modal", modal)
    spec = importlib.util.spec_from_file_location(name[:-3], SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _remote_fn(module):
    return next(
        getattr(module, n) for n in ("run_export_parity", "run_export_cpu") if hasattr(module, n)
    )


@pytest.mark.parametrize("name", VEHICLES)
@pytest.mark.parametrize(
    "verdict",
    ["aborted-restarted-attempt", "refused-export-provenance-mismatch", "no-receipt", "failed"],
)
def test_a_non_pass_exits_non_zero(name, verdict, monkeypatch):
    module = _load(name, monkeypatch)
    _remote_fn(module).remote = lambda **kwargs: {"verdict": verdict}
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code not in (0, None)


@pytest.mark.parametrize("name", VEHICLES)
def test_a_pass_exits_zero(name, monkeypatch):
    module = _load(name, monkeypatch)
    _remote_fn(module).remote = lambda **kwargs: {"verdict": "passed"}
    module.main()


def test_an_export_only_cpu_run_is_not_a_failure(monkeypatch):
    module = _load("modal_openvla_export_cpu_pinned.py", monkeypatch)
    module.run_export_cpu.remote = lambda **kwargs: {"export": {"status": "built"}}
    module.main(run_harness=False)


@pytest.mark.parametrize("name", VEHICLES)
def test_every_vehicle_defaults_to_the_pin_its_output_root_names(name, monkeypatch):
    module = _load(name, monkeypatch)
    assert module.TETHER_REVISION[:7] in module.PARITY_OUT_PATH
