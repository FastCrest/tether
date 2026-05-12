"""Format converters: Reflex JSONL traces → published-dataset formats.

Per the architecture ADR (decision #5), 4 format converters ship in Phase 1
in dependency order:

    LeRobot v3   first — highest customer overlap (HF ecosystem)
    RLDS         second — Google's standard; built on tensorflow_datasets
    Open X-Embodiment   third — built on top of RLDS
    HDF5         fourth — generic scientific format; lowest commercial pull

CLI surface:
    reflex curate convert <input.jsonl> --format <name> --output ./out/

Install extras gate the heavy deps:
    [curate-hdf5]   h5py (~5 MB)               for HDF5 output
    [curate-video]  imageio-ffmpeg + Pillow    for LeRobot v3 mp4 encoding
                                               (without it, parquet + metadata
                                               only; warning emitted)
    [curate-rlds]   tensorflow + tfds (~80 MB) for RLDS + Open X-Embodiment

Submodules:
    base                  abstract FormatConverter interface
    lerobot_v3            LeRobot v3 dataset format
    rlds                  RLDS / TFRecord (tensorflow_datasets)
    openx_embodiment      OXE published-dataset format (uses RLDS)
    hdf5                  generic HDF5 hierarchical container
    shared/               common helpers (parquet writer, etc.)
"""
from __future__ import annotations

from reflex.curate.format_converters.base import (
    ConversionResult,
    FormatConverter,
)
from reflex.curate.format_converters.hdf5 import HDF5Converter
from reflex.curate.format_converters.lerobot_v3 import LeRobotV3Converter
from reflex.curate.format_converters.openx_embodiment import (
    EMBODIMENT_OXE_MAP,
    OpenXEmbodimentConverter,
)
from reflex.curate.format_converters.rlds import RLDSConverter


CONVERTER_REGISTRY: dict[str, type[FormatConverter]] = {
    LeRobotV3Converter.FORMAT_NAME: LeRobotV3Converter,
    HDF5Converter.FORMAT_NAME: HDF5Converter,
    RLDSConverter.FORMAT_NAME: RLDSConverter,
    OpenXEmbodimentConverter.FORMAT_NAME: OpenXEmbodimentConverter,
}


def get_converter(format: str, **kwargs) -> FormatConverter:
    """Look up a converter by format name + instantiate with kwargs.

    Raises ValueError on unknown format. Format names:
        lerobot-v3 / hdf5 / rlds / openx-embodiment
    """
    cls = CONVERTER_REGISTRY.get(format)
    if cls is None:
        raise ValueError(
            f"unknown format {format!r}; available: {sorted(CONVERTER_REGISTRY.keys())}"
        )
    return cls(**kwargs)


__all__ = [
    "CONVERTER_REGISTRY",
    "ConversionResult",
    "EMBODIMENT_OXE_MAP",
    "FormatConverter",
    "HDF5Converter",
    "LeRobotV3Converter",
    "OpenXEmbodimentConverter",
    "RLDSConverter",
    "get_converter",
]
