"""Jetson detection: the one place that knows how a Tegra board answers.

Every probe in the rest of Tether reached for `nvidia-smi`, which **does not
exist on a Jetson**. On an Orin the effect is not an error, it is a wrong answer:
`tether doctor` reported "no CUDA detected" on a board with CUDA 12.6 installed
and both GPU execution providers available. Jetson-shaped knowledge was also
duplicated in four places, each knowing a different subset, which is how that
went unnoticed.

What a Jetson answers instead:

* `/etc/nv_tegra_release` — the L4T release banner. Its presence is the
  authoritative "this is a Tegra board" signal.
* `/usr/local/cuda/version.json`, else `nvcc --version` — the CUDA toolkit
  version. JetPack installs CUDA at the OS level; there is no driver query.
* `NvInferVersion.h`, else `dpkg` — TensorRT. JetPack installs it through apt,
  so `importlib.metadata.version("tensorrt")` returns `None` on a perfectly
  healthy Jetson. pip metadata is checked last, not first.
* JetPack version is **derived** from the L4T release. No file states it.

Parsers take text and collectors read the filesystem, so the parsers are
testable off-device. Nothing here guesses: an unreadable source returns `None`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path

L4T_RELEASE_PATH = "etc/nv_tegra_release"

#: L4T release -> JetPack release, from NVIDIA's published mapping. Unlisted
#: releases resolve to None rather than to a nearby guess.
L4T_TO_JETPACK: dict[str, str] = {
    "36.4.4": "6.2.1",
    "36.4.3": "6.2",
    "36.4.0": "6.1",
    "36.3.0": "6.0",
    "36.2.0": "6.0-DP",
    "35.6.0": "5.1.4",
    "35.5.0": "5.1.3",
    "35.4.1": "5.1.2",
    "35.3.1": "5.1.1",
    "35.2.1": "5.1",
    "35.1.0": "5.0.2",
}

#: L4T R36 (JetPack 6) is the first branch shipping CUDA 12.x. ORT 1.20+ needs
#: CUDA 12, so an R35 board silently falls to CPU no matter how it is installed.
MIN_L4T_MAJOR_FOR_CUDA12 = 36

_L4T_MAJOR = re.compile(r"#\s*R(\d+)\s*\(release\)", re.IGNORECASE)
_L4T_REVISION = re.compile(r"REVISION:\s*([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)
_NVCC_RELEASE = re.compile(r"release\s+([0-9]+\.[0-9]+)")
_TRT_DEFINE = re.compile(r"#define\s+NV_TENSORRT_(MAJOR|MINOR|PATCH)\s+(\d+)")

_MAX_PROBE_BYTES = 64 * 1024
_PROBE_TIMEOUT_S = 5.0

_TRT_HEADERS = (
    "usr/include/aarch64-linux-gnu/NvInferVersion.h",
    "usr/include/x86_64-linux-gnu/NvInferVersion.h",
    "usr/include/NvInferVersion.h",
)


@dataclass(frozen=True)
class L4TRelease:
    major: int
    minor: int
    patch: int
    raw: str

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def jetpack(self) -> str | None:
        return L4T_TO_JETPACK.get(str(self))

    @property
    def ships_cuda_12(self) -> bool:
        return self.major >= MIN_L4T_MAJOR_FOR_CUDA12


def parse_l4t_release(text: str | None) -> L4TRelease | None:
    """`# R36 (release), REVISION: 4.3, GCID: ...` -> `L4TRelease(36, 4, 3)`.

    A revision with no patch component is normalised to `x.y.0`, which is how
    NVIDIA numbers the first release of a branch.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    major = _L4T_MAJOR.search(text)
    revision = _L4T_REVISION.search(text)
    if not major or not revision:
        return None
    parts = [int(part) for part in revision.group(1).split(".")]
    while len(parts) < 2:
        parts.append(0)
    return L4TRelease(int(major.group(1)), parts[0], parts[1], text.strip())


def parse_cuda_version_json(text: str | None) -> str | None:
    """`/usr/local/cuda/version.json` -> `"12.6.68"`."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    for key in ("cuda", "cuda_nvcc", "cuda_cudart"):
        entry = value.get(key)
        if isinstance(entry, dict) and isinstance(entry.get("version"), str):
            return entry["version"].strip() or None
    return None


def parse_nvcc_version(text: str | None) -> str | None:
    """`nvcc --version` -> `"12.6"`."""
    if not isinstance(text, str):
        return None
    match = _NVCC_RELEASE.search(text)
    return match.group(1) if match else None


def parse_tensorrt_header(text: str | None) -> str | None:
    """`NvInferVersion.h` -> `"10.3.0"`."""
    if not isinstance(text, str):
        return None
    found = dict(_TRT_DEFINE.findall(text))
    if "MAJOR" not in found:
        return None
    return f"{found['MAJOR']}.{found.get('MINOR', '0')}.{found.get('PATCH', '0')}"


def _read(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_PROBE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def l4t_release(root: Path | str = "/") -> L4TRelease | None:
    """The board's L4T release, or `None` when this is not a Tegra board."""
    return parse_l4t_release(_read(Path(root) / L4T_RELEASE_PATH))


def is_jetson(root: Path | str = "/") -> bool:
    """True on a Jetson-class Linux/aarch64 host."""
    return (
        platform.system().lower() == "linux"
        and platform.machine().lower() in {"aarch64", "arm64"}
        and (Path(root) / L4T_RELEASE_PATH).exists()
    )


def cuda_version(root: Path | str = "/") -> tuple[str | None, str]:
    """CUDA toolkit version and where it came from. Never uses `nvidia-smi`."""
    version = parse_cuda_version_json(_read(Path(root) / "usr/local/cuda/version.json"))
    if version:
        return version, "cuda/version.json"
    version = parse_nvcc_version(_run(["nvcc", "--version"]))
    if version:
        return version, "nvcc"
    return None, "unavailable"


def tensorrt_version(root: Path | str = "/") -> tuple[str | None, str]:
    """TensorRT version and its source, apt-first.

    pip metadata is the last resort, not the first: under JetPack, TensorRT is
    an apt package and pip knows nothing about it.
    """
    root = Path(root)
    for relative in _TRT_HEADERS:
        version = parse_tensorrt_header(_read(root / relative))
        if version:
            return version, relative
    for package in ("tensorrt", "libnvinfer10", "libnvinfer8"):
        version = _run(["dpkg-query", "-W", "-f=${Version}", package])
        if version:
            return version, f"dpkg:{package}"
    try:
        import importlib.metadata

        return importlib.metadata.version("tensorrt"), "pip"
    except Exception:
        return None, "unavailable"


def jetson_profile(root: Path | str = "/") -> dict:
    """Everything a Jetson can be asked about itself, with per-field provenance."""
    release = l4t_release(root)
    cuda, cuda_source = cuda_version(root)
    tensorrt, tensorrt_source = tensorrt_version(root)
    return {
        "is_jetson": release is not None,
        "l4t_release": str(release) if release else None,
        "l4t_raw": release.raw if release else None,
        "jetpack": release.jetpack if release else None,
        "ships_cuda_12": release.ships_cuda_12 if release else None,
        "cuda": cuda,
        "tensorrt": tensorrt,
        "sources": {
            "l4t_release": L4T_RELEASE_PATH if release else "unavailable",
            "jetpack": "derived-from-l4t" if (release and release.jetpack) else "unavailable",
            "cuda": cuda_source,
            "tensorrt": tensorrt_source,
        },
    }


__all__ = [
    "L4T_RELEASE_PATH",
    "L4T_TO_JETPACK",
    "MIN_L4T_MAJOR_FOR_CUDA12",
    "L4TRelease",
    "cuda_version",
    "is_jetson",
    "jetson_profile",
    "l4t_release",
    "parse_cuda_version_json",
    "parse_l4t_release",
    "parse_nvcc_version",
    "parse_tensorrt_header",
    "tensorrt_version",
]
