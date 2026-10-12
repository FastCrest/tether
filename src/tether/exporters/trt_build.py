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


def validate_precision(hardware: HardwareProfile, precision: str | None = None) -> str:
    """Resolve an omitted precision to the profile default; never replace a request."""
    precision = hardware.trt_precision if precision is None else precision
    if precision == "int8":
        raise ValueError(
            "Standalone TensorRT builder does not support int8: INT8 calibration "
            "is not implemented. Supported precisions: fp16, fp8 (FP8-capable hardware)."
        )
    if precision not in ("fp16", "fp8"):
        raise ValueError(
            f"Unsupported TensorRT precision {precision!r}. "
            "Supported precisions: fp16, fp8 (FP8-capable hardware)."
        )
    if precision == "fp8" and not hardware.fp8_support:
        raise ValueError(
            "TensorRT fp8 requires FP8-capable hardware. "
            "Supported precision on this hardware: fp16."
        )
    return precision


def build_engine(
    onnx_path: Path,
    output_path: Path,
    hardware: HardwareProfile,
    min_shapes: dict[str, str] | None = None,
    opt_shapes: dict[str, str] | None = None,
    max_shapes: dict[str, str] | None = None,
    *,
    precision: str | None = None,
) -> Path:
    """Build a TensorRT engine from an ONNX model using trtexec.

    Explicit ``precision`` overrides the hardware default and is either used
    unchanged or refused before creating output files or invoking trtexec.
    Returns the path to an engine that exists on disk. Raises ``RuntimeError``
    if trtexec is missing, fails, or exits 0 without writing the engine — a
    returned path is always a real file, so callers never need to re-check.
    """
    precision = validate_precision(hardware, precision)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not check_trtexec():
        raise RuntimeError(
            f"TensorRT engine build failed for {onnx_path.name}: trtexec not found. "
            "Install TensorRT or run on a system with TensorRT."
        )

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
    ]

    # Precision
    if precision == "fp8":
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

    if not output_path.exists():
        raise RuntimeError(
            f"TensorRT engine build failed for {onnx_path.name}: trtexec exited 0 but "
            f"wrote no engine at {output_path}"
        )

    logger.info("Built TRT engine: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def build_all(
    onnx_dir: Path,
    output_dir: Path,
    hardware: HardwareProfile,
    *,
    precision: str | None = None,
) -> dict[str, Path]:
    """Build TRT engines for all ONNX models in a directory."""
    precision = validate_precision(hardware, precision)
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
            engines[onnx_path.stem] = build_engine(
                onnx_path, engine_path, hardware, precision=precision,
            )
        except RuntimeError as e:
            logger.error("Skipping %s: %s", onnx_path.name, e)

    return engines
