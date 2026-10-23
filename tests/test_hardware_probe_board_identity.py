"""Board identity must win over telemetry that does not identify a Jetson SKU."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from tether.runtime import hardware_probe as probe


@pytest.fixture
def board_files(tmp_path, monkeypatch):
    paths = (tmp_path / "proc-model", tmp_path / "sys-model")
    monkeypatch.setattr(probe, "_DEVICE_TREE_MODEL_PATHS", paths)
    return paths


@pytest.mark.parametrize(
    "model",
    [
        "NVIDIA Jetson Orin Nano Developer Kit\x00",
        "NVIDIA Jetson Orin Nano Super Developer Kit\x00",
        "NVIDIA Jetson Orin-Nano Engineering Reference Developer Kit\x00",
    ],
)
def test_nano_device_tree_wins_over_generic_orin_smi(model, board_files):
    board_files[0].write_bytes(model.encode())
    with (
        patch.object(probe, "_try_nvidia_smi") as smi,
        patch.object(probe, "_try_tegrastats") as tegra,
    ):
        result = probe.probe_device_class()
    assert result.device_class == "orin_nano"
    assert result.detection_method == "device-tree"
    assert result.requires_device_override is False
    assert result.raw_gpu_name == model.strip("\x00")
    smi.assert_not_called()
    tegra.assert_not_called()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("NVIDIA Jetson AGX Orin Developer Kit", "agx_orin"),
        ("NVIDIA Jetson Thor Developer Kit", "thor"),
        ("NVIDIA Jetson Orin NX Developer Kit", "cpu"),
        ("NVIDIA Jetson Orin", "cpu"),
        ("NVIDIA Jetson Nano Developer Kit", "cpu"),
    ],
)
def test_other_board_models_are_not_promoted_to_agx(model, expected, board_files):
    board_files[1].write_text(model)
    result = probe.probe_device_class()
    assert result.device_class == expected
    assert result.detection_method == "device-tree"
    assert result.requires_device_override is (expected == "cpu")
    if expected == "cpu":
        assert "--device-class" in result.notes[0]
        assert "no AGX profile assumed" in result.notes[0]


def test_oversize_device_tree_is_ignored_and_desktop_probe_preserved(board_files):
    board_files[0].write_bytes(b"NVIDIA Jetson AGX Orin " + b"x" * 4096)
    board_files[1].write_text("Raspberry Pi 5")
    with patch.object(probe, "_try_nvidia_smi", return_value=("NVIDIA H100", "raw")):
        result = probe.probe_device_class()
    assert result.device_class == "h100"
    assert result.detection_method == "nvidia-smi"


def test_unreadable_device_tree_uses_second_model_file(board_files):
    board_files[0].mkdir()
    board_files[1].write_text("NVIDIA Jetson Orin Nano Developer Kit")
    assert probe.probe_device_class().device_class == "orin_nano"


def test_override_does_not_probe_board_files(board_files):
    board_files[0].write_text("NVIDIA Jetson Orin Nano Developer Kit")
    with patch.object(probe, "_try_jetson_model") as board:
        result = probe.probe_device_class(override="h100")
    assert result.device_class == "h100"
    assert result.detection_method == "override"
    assert result.requires_device_override is False
    board.assert_not_called()


@pytest.mark.parametrize(
    "sample",
    [
        "RAM 1512/7620MB SWAP 0/3810MB CPU [0%@729] GR3D_FREQ 0%@306\n",
        "RAM 1512/7620MB NVIDIA Jetson Nano\n",
    ],
)
def test_ambiguous_tegrastats_does_not_claim_agx(sample):
    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, sample, "")):
        result = probe._try_tegrastats()
    assert result is not None
    assert result[0] == "cpu"


def test_ambiguous_tegrastats_timeout_bytes_preserve_diagnostics(board_files):
    sample = b"RAM 1512/7620MB SWAP 0/3810MB CPU [0%@729]\n"
    with (
        patch.object(probe, "_try_nvidia_smi", return_value=None),
        patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("tegrastats", 0.1, output=sample),
        ),
    ):
        result = probe.probe_device_class()
    assert result.device_class == "cpu"
    assert result.requires_device_override is True
    assert "no AGX profile assumed" in result.notes[0]


def test_nvidia_smi_wsl_fallback_when_not_on_path():
    with patch(
        "subprocess.run",
        side_effect=[
            FileNotFoundError(),
            subprocess.CompletedProcess([], 0, "NVIDIA GeForce RTX 5090\n", ""),
        ],
    ) as run:
        result = probe._try_nvidia_smi()
    assert result == ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 5090")
    assert run.call_args_list[0].args[0][0] == "nvidia-smi"
    assert run.call_args_list[1].args[0][0] == "/usr/lib/wsl/lib/nvidia-smi"


@pytest.mark.parametrize(
    "outcome",
    [
        subprocess.TimeoutExpired("nvidia-smi", 5),
        PermissionError(),
        subprocess.CompletedProcess([], 1, "", "driver failure"),
    ],
)
def test_nvidia_smi_does_not_retry_driver_errors(outcome):
    with patch("subprocess.run") as run:
        if isinstance(outcome, Exception):
            run.side_effect = outcome
        else:
            run.return_value = outcome
        assert probe._try_nvidia_smi() is None
    assert run.call_count == 1
