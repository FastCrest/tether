"""Jetson stack detection, and the check that used to get it wrong.

`check_hardware_compat` gated every probe on `nvidia-smi`, which is not present
on a Tegra board, so a healthy Orin Nano running CUDA 12.6 with both GPU
execution providers available was reported as "no CUDA detected". These tests
pin the answers a Jetson actually gives and assert that verdict is gone.
"""

from __future__ import annotations

import pytest

from tether.jetson import (
    L4TRelease,
    cuda_version,
    jetson_profile,
    l4t_release,
    parse_cuda_version_json,
    parse_l4t_release,
    parse_nvcc_version,
    parse_tensorrt_header,
    tensorrt_version,
)

R36_BANNER = (
    "# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, "
    "EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025\n"
)
R35_BANNER = "# R35 (release), REVISION: 4.1, GCID: 33958178, BOARD: t186ref, EABI: aarch64\n"
TRT_HEADER = (
    "#define NV_TENSORRT_MAJOR 10\n#define NV_TENSORRT_MINOR 3\n#define NV_TENSORRT_PATCH 0\n"
)


def jetson_root(tmp_path, *, banner=R36_BANNER, cuda="12.6.68", tensorrt=TRT_HEADER):
    (tmp_path / "etc").mkdir(exist_ok=True)
    (tmp_path / "etc/nv_tegra_release").write_text(banner)
    if cuda is not None:
        (tmp_path / "usr/local/cuda").mkdir(parents=True, exist_ok=True)
        (tmp_path / "usr/local/cuda/version.json").write_text(
            '{"cuda": {"name": "CUDA SDK", "version": "%s"}}' % cuda
        )
    if tensorrt is not None:
        (tmp_path / "usr/include/aarch64-linux-gnu").mkdir(parents=True, exist_ok=True)
        (tmp_path / "usr/include/aarch64-linux-gnu/NvInferVersion.h").write_text(tensorrt)
    return tmp_path


class TestParsers:
    def test_l4t_banner_yields_release_and_jetpack(self):
        release = parse_l4t_release(R36_BANNER)
        assert (release.major, release.minor, release.patch) == (36, 4, 3)
        assert str(release) == "36.4.3"
        assert release.jetpack == "6.2"
        assert release.ships_cuda_12 is True

    def test_jetpack_5_is_recognised_as_cuda_11(self):
        release = parse_l4t_release(R35_BANNER)
        assert str(release) == "35.4.1" and release.jetpack == "5.1.2"
        assert release.ships_cuda_12 is False

    def test_an_initial_branch_release_normalises_to_patch_zero(self):
        assert str(parse_l4t_release("# R36 (release), REVISION: 4.0, GCID: 1")) == "36.4.0"
        assert parse_l4t_release("# R36 (release), REVISION: 4.0, GCID: 1").jetpack == "6.1"

    def test_an_unmapped_release_has_no_jetpack_rather_than_a_nearby_guess(self):
        assert L4TRelease(36, 9, 9, "raw").jetpack is None

    def test_non_tegra_text_is_not_a_release(self):
        assert parse_l4t_release("Ubuntu 22.04.4 LTS") is None
        assert parse_l4t_release("") is None
        assert parse_l4t_release(None) is None

    def test_cuda_and_tensorrt_parsers(self):
        assert parse_cuda_version_json('{"cuda": {"version": "12.6.68"}}') == "12.6.68"
        assert parse_cuda_version_json("not json") is None
        assert parse_nvcc_version("Cuda compilation tools, release 12.6, V12.6.68") == "12.6"
        assert parse_tensorrt_header(TRT_HEADER) == "10.3.0"
        assert parse_tensorrt_header("#define UNRELATED 3") is None


class TestCollectors:
    def test_cuda_is_read_without_nvidia_smi(self, tmp_path):
        root = jetson_root(tmp_path)
        assert cuda_version(root) == ("12.6.68", "cuda/version.json")

    def test_tensorrt_prefers_the_apt_header_over_pip_metadata(self, tmp_path):
        """JetPack installs TensorRT via apt; pip metadata is empty on a healthy board."""
        root = jetson_root(tmp_path)
        assert tensorrt_version(root) == (
            "10.3.0",
            "usr/include/aarch64-linux-gnu/NvInferVersion.h",
        )

    def test_profile_reports_provenance_and_stays_none_when_unreadable(self, tmp_path):
        profile = jetson_profile(jetson_root(tmp_path))
        assert profile["is_jetson"] is True
        assert profile["l4t_release"] == "36.4.3"
        assert profile["jetpack"] == "6.2"
        assert profile["cuda"] == "12.6.68"
        assert profile["tensorrt"] == "10.3.0"
        assert profile["sources"]["cuda"] == "cuda/version.json"

        bare = jetson_profile(tmp_path / "nonexistent")
        assert bare["is_jetson"] is False
        assert bare["l4t_release"] is None and bare["jetpack"] is None

    def test_l4t_release_is_none_off_device(self, tmp_path):
        assert l4t_release(tmp_path) is None


class TestHardwareCompatCheckOnJetson:
    """The regression that motivated this module."""

    @staticmethod
    def _patch(monkeypatch, tmp_path, *, banner=R36_BANNER, cuda="12.6.68", tensorrt=TRT_HEADER):
        from tether.diagnostics import check_hardware_compat as module

        root = jetson_root(tmp_path, banner=banner, cuda=cuda, tensorrt=tensorrt)
        monkeypatch.setattr(module, "l4t_release", lambda: l4t_release(root))
        monkeypatch.setattr(module, "jetson_profile", lambda: jetson_profile(root))
        return module

    def test_a_healthy_orin_no_longer_reports_no_cuda_detected(self, monkeypatch, tmp_path):
        module = self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(
            module,
            "_probe_cuda_version",
            lambda: pytest.fail("nvidia-smi must never be consulted on a Jetson"),
        )
        pytest.importorskip("onnxruntime")
        import onnxruntime

        monkeypatch.setattr(
            onnxruntime,
            "get_available_providers",
            lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        result = module._run()
        assert result.status == "pass"
        assert "no CUDA detected" not in result.actual
        assert "CUDA=12.6.68" in result.actual
        assert "JetPack=6.2" in result.actual
        assert "TensorRT=10.3.0" in result.actual

    def test_jetpack_5_fails_because_ort_needs_cuda_12(self, monkeypatch, tmp_path):
        module = self._patch(monkeypatch, tmp_path, banner=R35_BANNER, cuda="11.4.315")
        pytest.importorskip("onnxruntime")
        import onnxruntime

        monkeypatch.setattr(
            onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider"]
        )
        result = module._run()
        assert result.status == "fail"
        assert "CUDA 11.4" in result.remediation and "JetPack 6" in result.remediation

    def test_cpu_only_jetson_gets_jetson_specific_remediation(self, monkeypatch, tmp_path):
        """The generic `pip install onnxruntime-gpu` advice is wrong on Jetson."""
        module = self._patch(monkeypatch, tmp_path)
        pytest.importorskip("onnxruntime")
        import onnxruntime

        monkeypatch.setattr(
            onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"]
        )
        result = module._run()
        assert result.status == "fail"
        assert "jetson" in result.remediation.lower()
        assert "pip uninstall onnxruntime" not in result.remediation

    def test_jetson_without_a_cuda_toolkit_warns_rather_than_passing(self, monkeypatch, tmp_path):
        module = self._patch(monkeypatch, tmp_path, cuda=None)
        pytest.importorskip("onnxruntime")
        import onnxruntime

        monkeypatch.setattr(
            onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider"]
        )
        result = module._run()
        assert result.status == "warn" and result.remediation

    def test_the_discrete_gpu_path_is_untouched_off_tegra(self, monkeypatch):
        from tether.diagnostics import check_hardware_compat as module

        monkeypatch.setattr(module, "l4t_release", lambda: None)
        monkeypatch.setattr(module, "_probe_cuda_version", lambda: None)
        pytest.importorskip("onnxruntime")
        result = module._run()
        assert result.status == "warn" and "no CUDA detected" in result.actual


def test_agent_hardware_profile_carries_the_derived_jetpack_version(monkeypatch):
    """`JetsonInfo.jetpack_version` being declared and never written is the bug
    this field exists to avoid repeating."""
    import tether.agent.hardware as module

    monkeypatch.setattr(module, "_l4t_release", lambda: parse_l4t_release(R36_BANNER))
    monkeypatch.setattr(module, "_tensorrt_version", lambda: ("10.3.0", "header"))
    profile = module.collect_hardware_profile()
    assert profile["l4t_release"] == "36.4.3"
    assert profile["jetpack_version"] == "6.2"
    assert profile["jetpack"].startswith("# R36")
    assert profile["tensorrt"] == "10.3.0"
