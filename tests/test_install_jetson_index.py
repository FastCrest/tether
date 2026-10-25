"""install.sh Jetson branch: L4T-based CUDA index choice and no silent CPU fallback."""

from pathlib import Path
import subprocess

import pytest

INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"

FAKE_PYTHON = """#!/bin/sh
case "$1" in
  -c) case "$2" in
        *micro*) echo 3.${TEST_PY_MINOR:-11}.12 ;;
        *minor*) echo ${TEST_PY_MINOR:-11} ;;
        *major*) echo 3 ;;
      esac ;;
  -m) shift 2; printf '%s\\n' "$*" >> "$PIP_LOG"
      case "$*" in
        download*) exit "${FAIL_PROBE:-0}" ;;
        *onnxruntime-gpu*) exit "${FAIL_ORT:-0}" ;;
      esac ;;
esac
exit 0
"""


def run(tmp_path, revision=None, args=(), fail_ort=0, fail_probe=0, python_minor=11):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("python3.13", "python3.12", "python3.11", "python3.10", "python3", "python"):
        (bin_dir / name).write_text(FAKE_PYTHON)
        (bin_dir / name).chmod(0o755)
    rootfs = tmp_path / "rootfs"
    (rootfs / "etc").mkdir(parents=True, exist_ok=True)
    if revision:
        (rootfs / "etc/nv_tegra_release").write_text(
            f"# R{revision[0]} (release), REVISION: {revision[1]}, GCID: 1, BOARD: generic, EABI: aarch64\n"
        )
    log = tmp_path / "pip.log"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "PIP_LOG": str(log),
        "TETHER_INSTALL_ROOTFS": str(rootfs),
        "FAIL_ORT": str(fail_ort),
        "FAIL_PROBE": str(fail_probe),
        "TEST_PY_MINOR": str(python_minor),
        "NO_COLOR": "1",
    }
    result = subprocess.run(
        ["bash", str(INSTALLER), "--jetson", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return result, (log.read_text() if log.exists() else "")


@pytest.mark.parametrize(
    "revision,jetpack",
    [
        (("36", "4.3"), "6.2"),
        (("36", "4.4"), "6.2.1"),
        (("36", "4.0"), "6.1"),
        (("36", "4"), "6.1"),
    ],
)
def test_l4t_36_4_uses_cu126_jetson_extra_and_pinned_ort(tmp_path, revision, jetpack):
    result, pip = run(tmp_path, revision)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"JetPack {jetpack} (L4T 36.4." in result.stdout
    assert "--index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu==1.24.0" in pip
    assert "cu129" not in pip
    assert "download --no-deps --only-binary=:all:" in pip
    assert pip.splitlines()[-1] == "install --upgrade fastcrest-tether[jetson]"


def test_explicit_jetpack_6_2_selects_cu126_not_cu129(tmp_path):
    result, pip = run(tmp_path, None, ["--jetpack", "6.2"])
    assert result.returncode == 0, result.stdout
    assert "jp6/cu126" in pip and "cu129" not in pip


@pytest.mark.parametrize(
    "revision,args,message",
    [
        (("36", "3.0"), (), "JetPack 6.0 is not supported"),
        (("35", "4.1"), (), "JetPack 5 is not supported"),
        (("38", "2.0"), (), "JetPack unknown (L4T 38.2.0) is not supported"),
        (None, (), "Could not read the L4T release"),
        (None, ("--jetpack", "6.0"), "JetPack 6.0 is not supported"),
    ],
)
def test_unsupported_or_unknown_jetpack_refuses_before_pip(tmp_path, revision, args, message):
    result, pip = run(tmp_path, revision, args)
    assert result.returncode == 1
    assert message in result.stdout
    assert pip == ""


def test_gpu_onnxruntime_failure_stops_instead_of_cpu_fallback(tmp_path):
    result, pip = run(tmp_path, ("36", "4.3"), fail_ort=1)
    assert result.returncode == 1
    assert "Not installing Tether without a GPU onnxruntime" in result.stdout
    assert "fastcrest-tether" not in pip


def test_python_310_refuses_before_any_package_operation(tmp_path):
    result, pip = run(tmp_path, ("36", "4.3"), python_minor=10)
    assert result.returncode == 1
    assert "requires Python 3.11+" in result.stdout
    assert pip == ""


def test_python_311_missing_vendor_wheel_refuses_before_install(tmp_path):
    result, pip = run(tmp_path, ("36", "4.3"), fail_probe=1)
    assert result.returncode == 1
    assert "No compatible JetPack 6 GPU onnxruntime wheel" in result.stdout
    assert "No GPU packages were installed" in result.stdout
    assert "download --no-deps --only-binary=:all:" in pip
    assert not any(line.startswith("install ") for line in pip.splitlines())
    assert "use Python 3.10" not in result.stdout
