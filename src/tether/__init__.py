"""Tether — FastCrest's edge-to-cloud robotics + edge AI deploy CLI.

Optimize, deploy, and verify any VLA model across Jetson, RTX, Apple Silicon,
and AMD edge hardware. Standalone or wired to FastCrest Cloud.

Renamed from ``reflex-vla`` in v0.12.0. The ``reflex`` Python package and CLI
binary still work via a backwards-compat shim through v0.13.x (removed in
v0.14.0). See CHANGELOG.md v0.12.0 for the migration guide.
"""

__version__ = "0.12.0"

# Mirror legacy REFLEX_* env vars to TETHER_* before any other module reads
# os.environ. Idempotent; opt out via TETHER_NO_ENV_COMPAT=1. Removed in
# v0.14.0. See _compat_env.py for the deprecation contract.
from tether._compat_env import _mirror_reflex_env_to_tether as _mirror_env

_mirror_env()
del _mirror_env

# Heavy submodules (validate_roundtrip pulls in torch) are lazy-loaded so that
# `tether --version`, `tether --help`, `tether chat`, etc. don't pay the
# 700ms+ torch-import cost on every invocation. Importers like
# `from tether import ValidateRoundTrip` still work — the __getattr__ hook
# imports on first access.
__all__ = [
    "__version__",
    "ValidateRoundTrip",
    "SUPPORTED_MODEL_TYPES",
    "UNSUPPORTED_MODEL_MESSAGE",
    "load_fixtures",
]


# GPU dependencies can live in active Python paths or system installations.
# Changing LD_LIBRARY_PATH helps children only; dlopen prepares this process.
# Mixed CUDA families are reported by doctor and never chosen automatically.
def _candidate_lib_dirs():
    from tether._gpu_loader import candidate_lib_dirs
    return candidate_lib_dirs()


def _patch_ld_library_path() -> None:
    from tether._gpu_loader import prepare_libraries
    prepare_libraries(preload=False)


def _eager_dlopen_nvidia_libs() -> None:
    from tether._gpu_loader import prepare_libraries
    prepare_libraries()


_eager_dlopen_nvidia_libs()


def __getattr__(name: str):
    if name in {"ValidateRoundTrip", "SUPPORTED_MODEL_TYPES", "UNSUPPORTED_MODEL_MESSAGE"}:
        from tether.validate_roundtrip import (
            SUPPORTED_MODEL_TYPES,
            UNSUPPORTED_MODEL_MESSAGE,
            ValidateRoundTrip,
        )
        return {
            "ValidateRoundTrip": ValidateRoundTrip,
            "SUPPORTED_MODEL_TYPES": SUPPORTED_MODEL_TYPES,
            "UNSUPPORTED_MODEL_MESSAGE": UNSUPPORTED_MODEL_MESSAGE,
        }[name]
    if name == "load_fixtures":
        from tether.fixtures import load_fixtures
        return load_fixtures
    raise AttributeError(f"module 'tether' has no attribute {name!r}")
