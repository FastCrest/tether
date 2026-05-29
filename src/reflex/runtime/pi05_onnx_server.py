"""Pi05OnnxServer — pi0.5 monolithic ONNX runtime.

pi0.5's monolithic export has the same image/mask/lang/noise signature as pi0,
except proprio state is already encoded into language and therefore no
``state`` input exists. Pi0OnnxServer already filters feeds to ONNX input names,
so this class only specializes metadata and public inference mode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from reflex.runtime.pi0_onnx_server import Pi0OnnxServer


class Pi05OnnxServer(Pi0OnnxServer):
    def __init__(
        self,
        export_dir: str | Path,
        onnx_path: str | Path | None = None,
        providers: list[Any] | None = None,
        device: str = "cpu",
        max_batch: int = 1,
        strict_providers: bool = True,
    ):
        super().__init__(
            export_dir,
            onnx_path=onnx_path,
            providers=providers,
            device=device,
            max_batch=max_batch,
            strict_providers=strict_providers,
        )
        self._inference_mode = "pi05_onnx_monolithic"
