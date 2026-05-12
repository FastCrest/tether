"""Reflex — Deploy any VLA model to any edge hardware. One command."""

__version__ = "0.9.6"

# Heavy submodules (validate_roundtrip pulls in torch) are lazy-loaded so that
# `reflex --version`, `reflex --help`, `reflex chat`, etc. don't pay the
# 700ms+ torch-import cost on every invocation. Importers like
# `from reflex import ValidateRoundTrip` still work — the __getattr__ hook
# imports on first access.
__all__ = [
    "__version__",
    "ValidateRoundTrip",
    "SUPPORTED_MODEL_TYPES",
    "UNSUPPORTED_MODEL_MESSAGE",
    "load_fixtures",
]


# ─── ORT-TRT EP first-class support (v0.7) ──────────────────────────────────
# ORT-TRT EP needs libnvinfer.so.10 (from the `tensorrt` pip pkg) + CUDA libs
# (libcublas, libcudnn) loadable at session-create time. The pip-installed
# nvidia/tensorrt libs live under site-packages but Linux's dynamic loader
# doesn't know to look there. Without these libs findable at runtime,
# ORT-TRT EP fails to load and ORT silently falls back to CUDA EP — losing
# the 5.55× perf win measured 2026-04-29 (Modal A10G, SmolVLA monolithic).
#
# Two-part fix at import time:
#   (1) Set LD_LIBRARY_PATH so any child subprocess inherits the right paths
#   (2) eagerly dlopen libnvinfer/libcublas/libcudnn with RTLD_GLOBAL so the
#       symbols are visible to ORT's later C++ dlopen — modifying
#       LD_LIBRARY_PATH after process start does NOT update the dynamic
#       loader for the current process, so (2) is the load-bearing piece.
#
# Both are idempotent. No-op on macOS/Windows or when the paths don't exist.
# Opt out via REFLEX_NO_LD_LIBRARY_PATH_PATCH=1.
#
# Per ADR 2026-04-29-ort-trt-ep-first-class-support.md.
def _candidate_lib_dirs():
    """Return the list of pip-installed nvidia/tensorrt lib dirs to consider."""
    import sys

    py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []
    for base in (sys.prefix, "/usr/local"):
        candidates.extend([
            f"{base}/lib/{py_lib}/site-packages/tensorrt_libs",
            f"{base}/lib/{py_lib}/site-packages/tensorrt",
            f"{base}/lib/{py_lib}/site-packages/nvidia/cudnn/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/cublas/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/cuda_runtime/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/cuda_nvrtc/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/nccl/lib",
            # ORT CUDA EP also needs curand/cufft/cusparse — added 2026-04-30
            # after gate-4 per-step-overhead image (built without torch's
            # transitive curand path) failed to load CUDA EP. Fix at source
            # so all reflex consumers benefit, not just the export image.
            f"{base}/lib/{py_lib}/site-packages/nvidia/curand/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/cufft/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/cusparse/lib",
            f"{base}/lib/{py_lib}/site-packages/nvidia/nvjitlink/lib",
        ])
    return candidates


def _patch_ld_library_path() -> None:
    """Prepend pip-installed nvidia/tensorrt lib dirs to LD_LIBRARY_PATH.

    Helps SUBPROCESSES inherit the right loader paths. Does NOT affect the
    current process's dynamic loader — see _eager_dlopen_nvidia_libs() for
    that. Returns silently on macOS, Windows, or when no paths exist.
    """
    import os
    import sys

    if os.environ.get("REFLEX_NO_LD_LIBRARY_PATH_PATCH"):
        return
    if sys.platform not in ("linux", "linux2"):
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(os.pathsep) if p]

    to_prepend = []
    for path in _candidate_lib_dirs():
        if not os.path.isdir(path):
            continue
        if path in existing_parts or path in to_prepend:
            continue  # idempotent: don't re-add
        to_prepend.append(path)

    if not to_prepend:
        return  # nothing to add — silent no-op

    new_value = os.pathsep.join(to_prepend + existing_parts)
    os.environ["LD_LIBRARY_PATH"] = new_value


def _eager_dlopen_nvidia_libs() -> None:
    """dlopen libnvinfer/libcublas/libcudnn into the current process w/ RTLD_GLOBAL.

    THIS is the load-bearing piece. Modifying LD_LIBRARY_PATH after process
    start does NOT affect the dynamic loader for this process — it only
    helps SUBPROCESSES. So we explicitly resolve full paths to the pip-
    installed shared objects and dlopen them with RTLD_GLOBAL, which makes
    the symbols visible to subsequent dlopen calls (including from ORT's
    C++ TRT EP layer when a session is created later).

    Idempotent — ctypes.CDLL with the same path twice just returns the
    cached handle. No-op on macOS/Windows or when libs don't exist.
    """
    import ctypes
    import glob
    import os
    import sys

    if os.environ.get("REFLEX_NO_LD_LIBRARY_PATH_PATCH"):
        return
    if sys.platform not in ("linux", "linux2"):
        return

    # Map of libname pattern → list of candidate .so files we'll search for.
    # The libs are loaded in dependency order (cuBLAS/cuDNN first, then the
    # higher-level TensorRT runtime that depends on them).
    # Order matters: deps first (cuda runtime + blas + cudnn), then TensorRT
    # core, then TensorRT plugin/parser libs that ORT's
    # libonnxruntime_providers_tensorrt.so needs at session-create time.
    # Caught 2026-04-29 — Modal A10G validation showed even after libnvinfer
    # loaded, ORT's TRT EP still failed because libnvonnxparser.so.10 wasn't
    # loaded yet.
    targets = [
        "libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12",
        "libcudnn.so.9",
        # ORT CUDA EP needs curand/cufft/cusparse for various kernels
        # (sample, FFT-based ops, sparse attention). Caught 2026-04-30 when
        # the gate-4 per-step-overhead image had nvidia-curand-cu12 pip-
        # installed but lib unfindable by ORT's libonnxruntime_providers_cuda
        # dlopen — without these here, ORT silently fell back to CPU EP.
        "libcurand.so.10",
        "libcufft.so.11",
        "libcusparse.so.12",
        "libnvJitLink.so.12",
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
        "libnvonnxparser.so.10",
        "libnvinfer_dispatch.so.10",
        "libnvinfer_lean.so.10",
    ]

    for libname in targets:
        # Try to find the lib in any of our candidate dirs
        for libdir in _candidate_lib_dirs():
            full_path = os.path.join(libdir, libname)
            if os.path.exists(full_path):
                try:
                    ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass  # best-effort; will surface in reflex doctor
                break  # found it; move to next lib


_patch_ld_library_path()
_eager_dlopen_nvidia_libs()


def __getattr__(name: str):
    if name in {"ValidateRoundTrip", "SUPPORTED_MODEL_TYPES", "UNSUPPORTED_MODEL_MESSAGE"}:
        from reflex.validate_roundtrip import (
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
        from reflex.fixtures import load_fixtures
        return load_fixtures
    raise AttributeError(f"module 'reflex' has no attribute {name!r}")
