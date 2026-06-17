"""TensorRT engine building from ONNX models."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from tether.config import HardwareProfile

logger = logging.getLogger(__name__)


def check_trtexec() -> bool:
    """Check if trtexec is available."""
    try:
        result = subprocess.run(
            ["trtexec", "--help"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_tensorrt_python() -> bool:
    """Check if TensorRT Python bindings are available."""
    try:
        import tensorrt
        return True
    except ImportError:
        return False


def build_engine(
    onnx_path: Path,
    output_path: Path,
    hardware: HardwareProfile,
    min_shapes: dict[str, str] | None = None,
    opt_shapes: dict[str, str] | None = None,
    max_shapes: dict[str, str] | None = None,
) -> Path:
    """Build a TensorRT engine from an ONNX model using trtexec."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not check_trtexec():
        logger.warning(
            "trtexec not found. Install TensorRT or run on a system with TensorRT. "
            "Skipping engine build for %s",
            onnx_path.name,
        )
        return output_path

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
    ]

    # Precision
    if hardware.trt_precision == "fp8" and hardware.fp8_support:
        cmd.append("--fp8")
    else:
        cmd.append("--fp16")

    # Shape profiles
    if opt_shapes:
        for name, shape in opt_shapes.items():
            cmd.append(f"--optShapes={name}:{shape}")
    if min_shapes:
        for name, shape in min_shapes.items():
            cmd.append(f"--minShapes={name}:{shape}")
    if max_shapes:
        for name, shape in max_shapes.items():
            cmd.append(f"--maxShapes={name}:{shape}")

    # Workspace
    cmd.append(f"--memPoolSize=workspace:{min(hardware.memory_gb * 256, 4096)}MiB")

    logger.info("Building TRT engine: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error("trtexec failed:\n%s", result.stderr[-2000:] if result.stderr else "no stderr")
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path.name}")

    logger.info("Built TRT engine: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def build_all(
    onnx_dir: Path,
    output_dir: Path,
    hardware: HardwareProfile,
) -> dict[str, Path]:
    """Build TRT engines for all ONNX models in a directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engines = {}

    onnx_files = sorted(Path(onnx_dir).glob("*.onnx"))
    if not onnx_files:
        logger.warning("No ONNX files found in %s", onnx_dir)
        return engines

    for onnx_path in onnx_files:
        engine_name = onnx_path.stem + ".trt"
        engine_path = output_dir / engine_name
        try:
            engines[onnx_path.stem] = build_engine(onnx_path, engine_path, hardware)
        except RuntimeError as e:
            logger.error("Skipping %s: %s", onnx_path.name, e)

    return engines
