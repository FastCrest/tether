"""Hardware profiles and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    sm_version: str
    memory_gb: int
    fp8_support: bool
    fp4_support: bool
    max_batch: int
    max_cameras: int
    max_seq_len: int
    trt_precision: str

    @property
    def supports_fp8(self) -> bool:
        return self.fp8_support

    @property
    def supports_fp4(self) -> bool:
        return self.fp4_support


HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "orin-nano": HardwareProfile(
        name="Jetson Orin Nano",
        sm_version="8.7",
        memory_gb=8,
        fp8_support=False,
        fp4_support=False,
        max_batch=1,
        max_cameras=2,
        max_seq_len=512,
        trt_precision="fp16",
    ),
    "orin": HardwareProfile(
        name="Jetson AGX Orin 32GB",
        sm_version="8.7",
        memory_gb=32,
        fp8_support=False,
        fp4_support=False,
        max_batch=1,
        max_cameras=3,
        max_seq_len=1024,
        trt_precision="fp16",
    ),
    "orin-64": HardwareProfile(
        name="Jetson AGX Orin 64GB",
        sm_version="8.7",
        memory_gb=64,
        fp8_support=False,
        fp4_support=False,
        max_batch=2,
        max_cameras=3,
        max_seq_len=2048,
        trt_precision="fp16",
    ),
    "thor": HardwareProfile(
        name="Jetson Thor",
        sm_version="10.0",
        memory_gb=128,
        fp8_support=True,
        fp4_support=True,
        max_batch=4,
        max_cameras=8,
        max_seq_len=4096,
        trt_precision="fp8",
    ),
    "desktop": HardwareProfile(
        name="Desktop GPU (RTX 4090 / A100 / H100)",
        sm_version="8.9",
        memory_gb=24,
        fp8_support=True,
        fp4_support=False,
        max_batch=4,
        max_cameras=3,
        max_seq_len=2048,
        trt_precision="fp16",
    ),
}


def get_hardware_profile(target: str) -> HardwareProfile:
    if target not in HARDWARE_PROFILES:
        available = ", ".join(HARDWARE_PROFILES.keys())
        raise ValueError(f"Unknown target '{target}'. Available: {available}")
    return HARDWARE_PROFILES[target]


@dataclass
class ExportConfig:
    model_id: str
    target: str
    output_dir: str
    precision: str = "fp16"
    opset: int = 19
    num_denoising_steps: int = 10
    action_chunk_size: int = 50
    action_dim: int = 6
    validate: bool = True
    benchmark_iterations: int = 100
