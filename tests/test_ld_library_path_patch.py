"""Unit tests for tether's LD_LIBRARY_PATH auto-patcher.

Per ADR 2026-04-29-ort-trt-ep-first-class-support.md: at module load,
tether prepends pip-installed nvidia/tensorrt lib dirs to LD_LIBRARY_PATH
so ORT-TRT EP can find libnvinfer.so.10 + CUDA libs. Tests verify:
- Linux + paths exist → paths get prepended
- Linux + no paths exist → no-op (no env var modification)
- Linux + paths already in LD_LIBRARY_PATH → idempotent (don't double-add)
- Linux + TETHER_NO_LD_LIBRARY_PATH_PATCH=1 → opt-out respected
- macOS / Windows → no-op regardless of paths
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

# Re-import the patch helper directly (NOT the side-effect at module load).
# This lets us test in isolation with mocked filesystem + env state.
from tether import _patch_ld_library_path


def _run_with(env=None, exists_paths=None, platform="linux"):
    """Run _patch_ld_library_path() with mocked env + filesystem + platform."""
    env = env or {}
    exists_paths = set(exists_paths or [])

    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(sys, "platform", platform),
        patch("os.path.isdir", side_effect=lambda p: p in exists_paths),
    ):
        _patch_ld_library_path()
        return os.environ.get("LD_LIBRARY_PATH", "")


def test_macos_is_no_op():
    """macOS shouldn't touch LD_LIBRARY_PATH (it doesn't use it anyway)."""
    result = _run_with(env={}, exists_paths=[], platform="darwin")
    assert result == ""


def test_windows_is_no_op():
    result = _run_with(env={}, exists_paths=[], platform="win32")
    assert result == ""


def test_linux_no_paths_exist_is_no_op():
    """Bare Linux install with no nvidia libs → don't pollute LD_LIBRARY_PATH."""
    result = _run_with(env={}, exists_paths=[], platform="linux")
    assert result == ""


def test_linux_paths_get_prepended():
    """Linux + tensorrt_libs exists → it gets prepended."""
    py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    trt_path = f"{sys.prefix}/lib/{py_lib}/site-packages/tensorrt_libs"
    result = _run_with(env={}, exists_paths=[trt_path], platform="linux")
    assert trt_path in result
    assert result.startswith(trt_path)


def test_linux_existing_ld_library_path_preserved():
    """Existing LD_LIBRARY_PATH entries are kept (appended after our prepends)."""
    py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    trt_path = f"{sys.prefix}/lib/{py_lib}/site-packages/tensorrt_libs"
    result = _run_with(
        env={"LD_LIBRARY_PATH": "/some/other/lib"},
        exists_paths=[trt_path],
        platform="linux",
    )
    parts = result.split(os.pathsep)
    assert trt_path in parts
    assert "/some/other/lib" in parts
    # Our paths come first
    assert parts.index(trt_path) < parts.index("/some/other/lib")


def test_linux_idempotent_no_double_add():
    """If a path is already in LD_LIBRARY_PATH, don't re-add it."""
    py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    trt_path = f"{sys.prefix}/lib/{py_lib}/site-packages/tensorrt_libs"
    result = _run_with(
        env={"LD_LIBRARY_PATH": trt_path},
        exists_paths=[trt_path],
        platform="linux",
    )
    parts = [p for p in result.split(os.pathsep) if p == trt_path]
    assert len(parts) == 1, f"Path appeared {len(parts)} times in {result!r}"


def test_opt_out_via_env_var():
    """TETHER_NO_LD_LIBRARY_PATH_PATCH=1 disables the patch entirely."""
    py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    trt_path = f"{sys.prefix}/lib/{py_lib}/site-packages/tensorrt_libs"
    result = _run_with(
        env={"TETHER_NO_LD_LIBRARY_PATH_PATCH": "1"},
        exists_paths=[trt_path],
        platform="linux",
    )
    # Patch was opted out — env var not modified
    assert result == ""


def test_multiple_paths_prepended_in_order():
    """When multiple lib dirs exist, all get prepended in candidate order."""
    py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    trt = f"{sys.prefix}/lib/{py_lib}/site-packages/tensorrt_libs"
    cudnn = f"{sys.prefix}/lib/{py_lib}/site-packages/nvidia/cudnn/lib"
    cublas = f"{sys.prefix}/lib/{py_lib}/site-packages/nvidia/cublas/lib"
    result = _run_with(
        env={},
        exists_paths=[trt, cudnn, cublas],
        platform="linux",
    )
    parts = result.split(os.pathsep)
    assert trt in parts
    assert cudnn in parts
    assert cublas in parts
    # tensorrt_libs comes before nvidia/cudnn/lib in the candidate list
    assert parts.index(trt) < parts.index(cudnn)
