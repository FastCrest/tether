"""Reproducibility envelope + Markdown/JSON report renderer.

A `BenchReport` bundles `LatencyStats` + a `BenchEnvironment` capture
(git SHA, GPU name, ORT/TRT/CUDA versions, ONNX file hashes, seed, etc.).
Two output formats:

- **Markdown** (`bench.md`): human + PR-friendly. Includes the full envelope
  + a stats table + a "What this measures" section explaining methodology
  (matches ISB-1's reproducibility-doc convention).
- **JSON** (`bench.json`): machine-readable; CI consumers grep this. Stable
  schema; bumping the schema requires a new top-level `schema_version` key.

The envelope is what makes a bench report comparable across releases — without
it, a "9.79× speedup" claim from one machine on one commit is unverifiable on
another.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reflex.bench.methodology import LatencyStats


@dataclass(frozen=True)
class BenchEnvironment:
    """Reproducibility envelope: enough to re-run + cross-check a bench.

    Filled by `capture_environment(...)` at bench-run time. Frozen so a
    BenchReport that includes it can't drift after creation.
    """

    timestamp_utc: str
    reflex_version: str
    git_sha: str
    git_dirty: bool  # True if working tree had uncommitted changes
    python_version: str
    platform: str  # e.g. "Darwin-25.3.0-arm64"
    gpu_name: str
    cuda_version: str
    ort_version: str
    onnx_files: list[dict]  # [{name, sha256, bytes}]
    seed: int
    export_dir: str
    inference_mode: str  # e.g. "onnx_trt_fp16", "onnx_cpu"
    device: str  # "cuda" / "cpu"
    provider_mode: str = ""  # e.g. "onnx_trt_fp16", "onnx_gpu", "onnx_cpu"
    active_providers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "").strip().split("\n")[0].strip()
    except Exception:
        return ""


def _git_info(repo_dir: Path) -> tuple[str, bool]:
    """Returns (sha, is_dirty). Empty sha if not in a git repo."""
    cwd = str(repo_dir)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return sha[:12] if sha else "", bool(status)
    except Exception:
        return "", False


def _gpu_name() -> str:
    out = _safe_run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return out


def _cuda_version() -> str:
    out = _safe_run(["nvcc", "--version"])
    if "release" in out:
        # "Cuda compilation tools, release 12.6, V12.6.85" → "12.6"
        for tok in out.split(","):
            if "release" in tok:
                return tok.split("release")[-1].strip()
    out = _safe_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out


def _ort_version() -> str:
    try:
        import onnxruntime as ort
        return ort.__version__
    except ImportError:
        return ""


def _reflex_version() -> str:
    try:
        from importlib.metadata import version
        return version("reflex-vla")
    except Exception:
        return "0.1.0+dev"


def _sha256_prefix(path: Path, prefix_len: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:prefix_len]


def _onnx_file_summary(export_dir: Path) -> list[dict]:
    """Hash + size ONNX model files, including external-data weight files."""
    out: list[dict] = []
    patterns = ("*.onnx", "*.onnx.data", "*.data")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(export_dir.glob(pattern))
    for p in sorted(set(files)):
        try:
            out.append(
                {
                    "name": p.name,
                    "sha256_prefix": _sha256_prefix(p),
                    "bytes": p.stat().st_size,
                }
            )
        except Exception:
            out.append({"name": p.name, "sha256_prefix": "", "bytes": -1})
    return out


def capture_environment(
    export_dir: str | Path,
    device: str = "cuda",
    inference_mode: str = "",
    provider_mode: str = "",
    active_providers: list[str] | None = None,
    seed: int = 0,
    repo_dir: str | Path | None = None,
) -> BenchEnvironment:
    """Snapshot the reproducibility envelope at bench-run time."""
    export_path = Path(export_dir)
    repo_path = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[3]
    sha, dirty = _git_info(repo_path)
    return BenchEnvironment(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        reflex_version=_reflex_version(),
        git_sha=sha,
        git_dirty=dirty,
        python_version=sys.version.split()[0],
        platform=f"{platform.system()}-{platform.release()}-{platform.machine()}",
        gpu_name=_gpu_name(),
        cuda_version=_cuda_version(),
        ort_version=_ort_version(),
        onnx_files=_onnx_file_summary(export_path),
        seed=seed,
        export_dir=str(export_path),
        inference_mode=inference_mode,
        device=device,
        provider_mode=provider_mode,
        active_providers=list(active_providers or []),
    )


@dataclass
class BenchReport:
    """Bundle of stats + environment + optional parity result. Renders to
    Markdown or JSON; both consume the same data so they can never drift."""

    stats: LatencyStats
    environment: BenchEnvironment
    parity: dict | None = None  # {"cos": float, "passed": bool, "threshold": float}
    notes: list[str] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "stats": self.stats.to_dict(),
            "environment": self.environment.to_dict(),
            "parity": self.parity,
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        s = self.stats
        e = self.environment
        lines = [
            "# Reflex Bench Report",
            "",
            f"_Generated {e.timestamp_utc} • reflex {e.reflex_version} @ {e.git_sha}{' (dirty)' if e.git_dirty else ''}_",
            "",
            "## Per-chunk latency (post-warmup)",
            "",
            f"- **n** = {s.n} samples (warmup_discarded = {s.warmup_discarded})",
            f"- **min** = {s.min_ms:.2f} ms",
            f"- **mean** = {s.mean_ms:.2f} ms (95% CI [{s.ci95_low_ms:.2f}, {s.ci95_high_ms:.2f}])",
            f"- **p50** = {s.p50_ms:.2f} ms",
            f"- **p95** = {s.p95_ms:.2f} ms",
            f"- **p99** = {s.p99_ms:.2f} ms",
            f"- **p99.9** = {s.p99_9_ms:.2f} ms",
            f"- **max** = {s.max_ms:.2f} ms",
            f"- **std** = {s.std_ms:.2f} ms",
            f"- **jitter** (std/mean) = {s.jitter:.4f}",
            f"- **hz** (1/mean) = {s.hz_mean:.1f} Hz",
            "",
        ]
        if self.parity is not None:
            verdict = "PASS" if self.parity.get("passed") else "FAIL"
            lines += [
                "## Parity check",
                "",
                f"- cos = {self.parity.get('cos', float('nan')):.6f}",
                f"- threshold = {self.parity.get('threshold', float('nan')):.6f}",
                f"- verdict = **{verdict}**",
                "",
            ]
        lines += [
            "## Reproducibility envelope",
            "",
            f"- **timestamp**: `{e.timestamp_utc}`",
            f"- **reflex**: `{e.reflex_version}`",
            f"- **git**: `{e.git_sha}`{' (dirty)' if e.git_dirty else ''}",
            f"- **python**: `{e.python_version}`",
            f"- **platform**: `{e.platform}`",
            f"- **gpu**: `{e.gpu_name or 'n/a'}`",
            f"- **cuda**: `{e.cuda_version or 'n/a'}`",
            f"- **onnxruntime**: `{e.ort_version or 'n/a'}`",
            f"- **device**: `{e.device}`",
            f"- **inference_mode**: `{e.inference_mode or 'n/a'}`",
            f"- **provider_mode**: `{e.provider_mode or 'n/a'}`",
            f"- **active_providers**: `{', '.join(e.active_providers) or 'n/a'}`",
            f"- **seed**: `{e.seed}`",
            f"- **export_dir**: `{e.export_dir}`",
            "",
            "### ONNX files",
            "",
            "| file | sha256 (16) | bytes |",
            "|---|---|---:|",
        ]
        for f in e.onnx_files:
            lines.append(f"| `{f['name']}` | `{f['sha256_prefix']}` | {f['bytes']} |")
        lines.append("")
        if self.notes:
            lines += ["## Notes", ""]
            lines += [f"- {n}" for n in self.notes]
            lines.append("")
        lines += [
            "## What this measures",
            "",
            "Per-chunk wall-clock latency of `server.predict()` — the full denoising loop "
            "for flow-matching VLAs (10 Euler steps for pi0 / pi0.5 / SmolVLA, 4 DDIM steps "
            "for GR00T) including VLM-prefix + expert-denoise + postprocess. Warmup samples "
            "discarded so TRT engine build / ORT graph optimization does not contaminate the "
            "steady-state distribution. Methodology lifted from ISB-1 (sibling project "
            "EasyInference); see `reference/NOTES.md` for the source pattern.",
            "",
        ]
        return "\n".join(lines)

    def write_markdown(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_markdown())

    def write_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json())
