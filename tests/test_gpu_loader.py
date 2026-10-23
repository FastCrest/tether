"""Library discovery tests. No GPU libraries or devices are opened."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tether import _gpu_loader as loader


def test_failed_candidate_does_not_hide_later_working_library(monkeypatch, tmp_path):
    dirs = [tmp_path / "stale", tmp_path / "working"]
    for directory in dirs:
        directory.mkdir()
        (directory / "libnvinfer.so.10").touch()

    def load(path, **kwargs):
        if "stale" in path:
            raise OSError("missing libcudart.so.13")
        return object()

    monkeypatch.setattr(loader.ctypes, "CDLL", load)
    monkeypatch.setattr(loader.ctypes.util, "find_library", lambda name: None)
    result = loader.load_library("libnvinfer.so.10", list(map(str, dirs)))
    assert result.path == str(dirs[1] / "libnvinfer.so.10")
    assert "libcudart.so.13" in result.errors[0]


def test_system_cache_library_does_not_require_pip_path(monkeypatch):
    monkeypatch.setattr(loader.ctypes.util, "find_library", lambda name: "/system/libnvinfer.so.10")
    load = MagicMock()
    monkeypatch.setattr(loader.ctypes, "CDLL", load)
    result = loader.load_library("libnvinfer.so.10", [])
    assert result.loaded
    assert result.path == "/system/libnvinfer.so.10"
    load.assert_called_once()


def test_all_load_failures_remain_visible(monkeypatch, tmp_path):
    (tmp_path / "libcudnn.so.9").touch()
    monkeypatch.setattr(loader.ctypes.util, "find_library", lambda name: None)
    monkeypatch.setattr(loader.ctypes, "CDLL", MagicMock(side_effect=OSError("dependency missing")))
    result = loader.load_library("libcudnn.so.9", [str(tmp_path)])
    assert not result.loaded
    assert len(result.errors) == 2
    assert all("dependency missing" in failure for failure in result.errors)


def test_active_python_and_explicit_loader_paths_are_discovered(monkeypatch, tmp_path):
    monkeypatch.setattr(loader.sys, "path", [str(tmp_path / "custom-python")])
    monkeypatch.setattr(
        loader.sysconfig, "get_path", lambda key: str(tmp_path / "configured-python")
    )
    monkeypatch.setattr(loader.site, "getsitepackages", lambda: [str(tmp_path / "site-python")])
    monkeypatch.setattr(loader.site, "getusersitepackages", lambda: str(tmp_path / "user-python"))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "manual-trt"))
    dirs = loader.candidate_lib_dirs()
    for root in ("custom-python", "configured-python", "site-python", "user-python"):
        assert str(tmp_path / root / "tensorrt_libs") in dirs
    assert str(tmp_path / "custom-python" / "tensorrt_cu13_libs") in dirs
    assert str(tmp_path / "custom-python" / "nvidia/cu13/lib") in dirs
    assert str(tmp_path / "manual-trt") in dirs
    assert len(dirs) == len(set(dirs))


@pytest.mark.parametrize(
    "names,expected",
    [
        (["nvidia-cublas-cu12", "tensorrt_cu12_libs"], {12}),
        (["nvidia-cuda-runtime", "tensorrt-cu13-libs"], {13}),
        (["nvidia-cublas-cu12", "tensorrt-cu13-libs"], {12, 13}),
    ],
)
def test_family_detection_uses_distribution_names_without_importing_gpu_modules(
    monkeypatch, names, expected
):
    monkeypatch.setattr(
        loader.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": name}, version="13.0.0") for name in names],
    )
    monkeypatch.delitem(loader.sys.modules, "torch", raising=False)
    assert loader.cuda_families([]) == expected


def test_system_soname_identifies_family_for_doctor(monkeypatch):
    monkeypatch.setattr(loader.metadata, "distributions", lambda: [])
    monkeypatch.delitem(loader.sys.modules, "torch", raising=False)
    monkeypatch.setattr(loader.ctypes.util, "find_library", lambda name: "libcublas.so.13")
    assert loader.cuda_families([], system=True) == {13}


@pytest.mark.parametrize("families", [{12, 13}, set()])
def test_ambiguous_family_does_not_poison_loader_path_or_dlopen(monkeypatch, tmp_path, families):
    monkeypatch.setattr(loader.sys, "platform", "linux")
    monkeypatch.delenv("TETHER_NO_LD_LIBRARY_PATH_PATCH", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/already-configured")
    monkeypatch.setattr(loader, "candidate_lib_dirs", lambda: [str(tmp_path)])
    monkeypatch.setattr(loader, "cuda_families", lambda dirs: families)
    load = MagicMock()
    monkeypatch.setattr(loader.ctypes, "CDLL", load)
    loader.prepare_libraries()
    assert loader.os.environ["LD_LIBRARY_PATH"] == "/already-configured"
    load.assert_not_called()
    assert loader.LAST_PRELOAD["reason"]


def test_cuda13_uses_correct_runtime_and_blas_majors(monkeypatch, tmp_path):
    monkeypatch.setattr(loader.sys, "platform", "linux")
    monkeypatch.delenv("TETHER_NO_LD_LIBRARY_PATH_PATCH", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(loader, "candidate_lib_dirs", lambda: [str(tmp_path)])
    monkeypatch.setattr(loader, "cuda_families", lambda dirs: {13})
    names = []
    monkeypatch.setattr(loader, "load_library", lambda name, dirs: names.append(name))
    loader.prepare_libraries()
    first = loader.os.environ["LD_LIBRARY_PATH"]
    loader.prepare_libraries()
    assert loader.os.environ["LD_LIBRARY_PATH"] == first
    assert "libcudart.so.13" in names
    assert "libcublas.so.13" in names
    assert not any(
        name.endswith(".12") and name.startswith(("libcudart", "libcublas", "libnvJitLink"))
        for name in names
    )


def test_opt_out_does_not_even_discover_libraries(monkeypatch):
    monkeypatch.setattr(loader.sys, "platform", "linux")
    monkeypatch.setenv("TETHER_NO_LD_LIBRARY_PATH_PATCH", "1")
    discover = MagicMock()
    monkeypatch.setattr(loader, "candidate_lib_dirs", discover)
    loader.prepare_libraries()
    discover.assert_not_called()


def test_os_cache_wrong_soname_cannot_cross_cuda_family(monkeypatch):
    monkeypatch.setattr(loader.ctypes.util, "find_library", lambda name: "libcublas.so.12")
    load = MagicMock(side_effect=OSError("CUDA13 unavailable"))
    monkeypatch.setattr(loader.ctypes, "CDLL", load)
    result = loader.load_library("libcublas.so.13", [])
    assert not result.loaded
    assert [call.args[0] for call in load.call_args_list] == ["libcublas.so.13"]


def test_default_system_paths_are_not_prepended_to_child_loader(monkeypatch):
    monkeypatch.setattr(loader.sys, "platform", "linux")
    monkeypatch.delenv("TETHER_NO_LD_LIBRARY_PATH_PATCH", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/intentional/path")
    monkeypatch.setattr(loader, "candidate_lib_dirs", lambda: ["/usr/lib/x86_64-linux-gnu"])
    monkeypatch.setattr(loader, "cuda_families", lambda dirs: {12})
    monkeypatch.setattr(loader.os.path, "isdir", lambda path: True)
    loader.prepare_libraries(preload=False)
    assert loader.os.environ["LD_LIBRARY_PATH"] == "/intentional/path"


@pytest.mark.parametrize("family,cufft_major", [(12, 11), (13, 12)])
def test_component_abi_majors_and_dependency_order(family, cufft_major):
    # CUDA13 wheel ZIP directories verified using bounded HTTP ranges:
    # nvidia-cusparse12.6.3.3 -> libcusparse.so.12
    # nvidia-curand10.4.0.35 -> libcurand.so.10
    # nvidia-cufft12.0.0.61 -> libcufft.so.12
    # These component ABI majors do not all equal the CUDA major.
    targets = loader.preload_targets(family)
    assert "libcusparse.so.12" in targets
    assert "libcurand.so.10" in targets
    assert f"libcufft.so.{cufft_major}" in targets
    assert "libcusparse.so.13" not in targets
    assert "libcurand.so.11" not in targets
    # Both wheels declare an nvidia-nvjitlink dependency. The current process
    # cannot use LD_LIBRARY_PATH changes to find an unloaded dependency.
    assert targets.index(f"libnvJitLink.so.{family}") < targets.index("libcusparse.so.12")
    assert targets.index(f"libnvJitLink.so.{family}") < targets.index(f"libcufft.so.{cufft_major}")
