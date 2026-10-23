"""Linux GPU library discovery without importing torch, TensorRT or ORT.

Library discovery is diagnostic evidence, not proof that a provider works.
Do not preload an arbitrary CUDA family into a mixed environment: shared
cuDNN/TensorRT SONAMEs do not identify their CUDA dependency family.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import site
import sys
import sysconfig
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class LibraryLoad:
    name: str
    path: str | None
    errors: tuple[str, ...]

    @property
    def loaded(self) -> bool:
        return self.path is not None


def candidate_lib_dirs() -> list[str]:
    """Include active Python paths, pip wheel layouts and explicit loader paths."""
    roots = list(sys.path)
    roots.extend(sysconfig.get_path(key) for key in ("purelib", "platlib"))
    roots.extend(site.getsitepackages())
    roots.append(site.getusersitepackages())
    suffixes = [
        "tensorrt_libs",
        "tensorrt_cu12_libs",
        "tensorrt_cu13_libs",
        "tensorrt",
        "torch/lib",
        "nvidia/cu12/lib",
        "nvidia/cu13/lib",
    ]
    suffixes += [
        f"nvidia/{name}/lib"
        for name in (
            "cuda_runtime",
            "cublas",
            "cudnn",
            "cuda_nvrtc",
            "nccl",
            "curand",
            "cufft",
            "cusparse",
            "nvjitlink",
        )
    ]
    candidates = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    candidates += [str(Path(root) / suffix) for root in roots if root for suffix in suffixes]
    # System installs, including JetPack, need not have Python distribution metadata.
    candidates += [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ]
    return list(dict.fromkeys(candidates))


def cuda_families(dirs: list[str] | None = None, *, system: bool = False) -> set[int]:
    """Return observed CUDA ABI families. A mixed result is a conflict, not a choice."""
    families: set[int] = set()
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name.startswith(("nvidia-", "tensorrt")):
            match = re.search(r"-cu(12|13)(?:-|$)", name)
            if match:
                families.add(int(match.group(1)))
            elif name in {"nvidia-cuda-runtime", "nvidia-cublas", "nvidia-cuda-nvrtc"}:
                major = dist.version.split(".", 1)[0]
                if major in {"12", "13"}:
                    families.add(int(major))
    for directory in dirs if dirs is not None else candidate_lib_dirs():
        for family in (12, 13):
            if any(
                (Path(directory) / f"{lib}.so.{family}").exists()
                for lib in ("libcudart", "libcublas")
            ):
                families.add(family)
    # Inspect an already imported torch only. Never import it for library discovery.
    torch_cuda = getattr(getattr(sys.modules.get("torch"), "version", None), "cuda", None)
    if isinstance(torch_cuda, str) and torch_cuda.split(".")[0] in {"12", "13"}:
        families.add(int(torch_cuda.split(".")[0]))
    if system:
        for lib in ("cudart", "cublas"):
            found = ctypes.util.find_library(lib) or ""
            match = re.search(r"\.so\.(12|13)(?:\D|$)", found)
            if match:
                families.add(int(match.group(1)))
    return families


def load_library(name: str, dirs: list[str] | None = None) -> LibraryLoad:
    """Try every existing candidate, then the OS loader; retain concrete failures."""
    paths = [
        str(Path(directory) / name)
        for directory in (dirs if dirs is not None else candidate_lib_dirs())
        if (Path(directory) / name).exists()
    ]
    found = ctypes.util.find_library(name.removeprefix("lib").split(".so", 1)[0])
    if found and (Path(found).name == name or Path(found).name.startswith(name + ".")):
        paths.append(found)
    paths.append(name)  # LD cache/system paths may work without pip metadata.
    errors = []
    for path in dict.fromkeys(paths):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
        else:
            return LibraryLoad(name, path, tuple(errors))
    return LibraryLoad(name, None, tuple(errors))


def preload_targets(family: int) -> tuple[str, ...]:
    # Shared major versions for cuDNN/TRT do not establish CUDA ABI compatibility.
    return (
        f"libcudart.so.{family}",
        # cuFFT and cuSPARSE depend on nvJitLink; preload it before either.
        f"libnvJitLink.so.{family}",
        f"libcublasLt.so.{family}",
        f"libcublas.so.{family}",
        "libcudnn.so.9",
        f"libnvrtc.so.{family}",
        "libcurand.so.10",
        f"libcufft.so.{11 if family == 12 else 12}",
        # cuSPARSE stays at SONAME12 in the CUDA13 wheel as well.
        "libcusparse.so.12",
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
        "libnvonnxparser.so.10",
        "libnvinfer_dispatch.so.10",
        "libnvinfer_lean.so.10",
    )


LAST_PRELOAD: dict[str, object] = {}
_SYSTEM_LIB_DIRS = {"/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"}


def prepare_libraries(*, patch_path: bool = True, preload: bool = True) -> None:
    """Best effort on Linux, with an inspectable reason when automatic loading skips."""
    if sys.platform not in {"linux", "linux2"} or os.environ.get("TETHER_NO_LD_LIBRARY_PATH_PATCH"):
        return
    dirs = candidate_lib_dirs()
    families = cuda_families(dirs)
    LAST_PRELOAD.clear()
    LAST_PRELOAD.update(families=sorted(families), loads=[])
    if len(families) != 1:
        LAST_PRELOAD["reason"] = (
            "mixed CUDA 12/13 libraries; automatic preloading disabled"
            if families
            else "CUDA family not detected; using the OS/provider loader"
        )
        return
    family = next(iter(families))
    if patch_path:
        existing = [
            part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part
        ]
        added = [
            path
            for path in dirs
            if os.path.isdir(path) and path not in existing and path not in _SYSTEM_LIB_DIRS
        ]
        if added:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(added + existing)
    if preload:
        LAST_PRELOAD["loads"] = [load_library(name, dirs) for name in preload_targets(family)]
