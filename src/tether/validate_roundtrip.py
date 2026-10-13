"""Round-trip validation orchestrator for `tether validate`.

This module defines :class:`ValidateRoundTrip`, the orchestrator that loads an
export directory, drives identical (image, prompt, state) fixtures through both
the exported ONNX graph and a PyTorch reference, then aggregates per-fixture
numerical differences into a single pass / fail decision.

Reference semantics (v1)
------------------------
"PyTorch reference" here means the *exporter's decomposed PyTorch model* — the
same architecture used to produce the ONNX export. This proves ONNX export
correctness, which is the primary goal of `tether validate`. Comparing against
the true upstream reference (e.g., LeRobot's `SmolVLAPolicy` end-to-end) is a
separate v2 concern and is intentionally out of scope here.

Seed bridge
-----------
`torch.manual_seed()` does not seed numpy. To guarantee that the PyTorch and
ONNX paths consume *byte-identical* initial noise, this orchestrator generates
the noise tensor exactly once via a seeded `torch.Generator`, converts it to a
numpy array, and hands the same numpy array to both backends. The backends are
responsible for re-wrapping it as needed for their runtime.

Exit code convention
--------------------
The CLI handler that wraps this class translates `run()`'s ``status`` field
into the following process exit codes:

* ``0`` — pass (every fixture's max abs diff < threshold)
* ``1`` — fail (at least one fixture exceeds threshold)
* ``2`` — error (missing ONNX, malformed export dir, unsupported model type)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tether.fixtures.vla_fixtures import load_fixtures
from tether.export_config import (
    UnsupportedExportKindError,
    UnsupportedExportPipelineError,
    decomposed_layout,
    load_tether_config,
    require_supported_export_kind,
    require_supported_pipeline,
)
from tether.validate import ValidationResult, validate_outputs

# Backends from Issues 4/5 — required at import time now.
from tether._pytorch_backend import load_pytorch_backend
from tether._onnx_backend import load_onnx_backend

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES: tuple[str, ...] = ("smolvla", "pi0", "gr00t")
MONOLITHIC_MODEL_TYPES: tuple[str, ...] = ("smolvla",)
UNSUPPORTED_MODEL_MESSAGE = (
    "tether validate v1 supports smolvla, pi0, gr00t. "
    "For pi0.5 / openvla see roadmap."
)


class ValidateRoundTrip:
    """Orchestrate ONNX-vs-PyTorch round-trip validation for an export directory.

    The class loads the export config, builds both backends, generates seeded
    fixtures + initial noise, runs every fixture through both runtimes, and
    aggregates per-fixture L2/abs-diff statistics into a single result dict.

    Parameters
    ----------
    export_dir:
        Path to a `tether export` output directory containing
        ``tether_config.json`` plus the ONNX graph(s).
    model_id:
        Optional HuggingFace ID (e.g., ``lerobot/pi0_base``) for the PyTorch
        reference checkpoint. If ``None``, the value from ``tether_config.json``
        is used.
    threshold:
        Per-element max-abs-diff threshold (default ``1e-4``). A fixture
        passes when ``max_abs_diff < threshold``.
    num_test_cases:
        Number of seeded fixtures to evaluate (default ``5``).
    seed:
        RNG seed shared by fixture generation and the initial-noise bridge
        (default ``0``).
    device:
        Device for the PyTorch reference (default ``"cpu"``). pi0 and GR00T are
        3B+ parameter models — keep CPU unless explicitly overridden.

    Raises
    ------
    FileNotFoundError
        If ``export_dir`` does not exist.
    ValueError
        If ``tether_config.json`` is missing, or the detected model type is not
        in :data:`SUPPORTED_MODEL_TYPES`.
    """

    def __init__(
        self,
        export_dir: Path,
        model_id: str | None = None,
        threshold: float = 1e-4,
        num_test_cases: int = 5,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.export_dir: Path = Path(export_dir)
        self.model_id: str | None = model_id
        self.threshold: float = float(threshold)
        self.num_test_cases: int = int(num_test_cases)
        self.seed: int = int(seed)
        self.device: str = device

        if self.num_test_cases < 1:
            raise ValueError(
                f"num_test_cases must be >= 1, got {self.num_test_cases}"
            )
        if self.threshold <= 0:
            raise ValueError(
                f"threshold must be > 0, got {self.threshold}"
            )

        if not self.export_dir.exists():
            raise FileNotFoundError(
                f"Export directory does not exist: {self.export_dir}"
            )
        if not self.export_dir.is_dir():
            raise ValueError(
                f"Export path is not a directory: {self.export_dir}"
            )

        self.config = load_tether_config(self.export_dir)
        self.export_kind: str = require_supported_export_kind(
            self.config,
            {"decomposed_onnx", "monolithic_onnx"},
            "tether validate",
        )
        require_supported_pipeline(self.config, set(), "tether validate")
        model_type = self.config["model_type"]
        if self.export_kind == "monolithic_onnx":
            if model_type not in MONOLITHIC_MODEL_TYPES:
                raise UnsupportedExportKindError(
                    f"tether validate does not support export_kind='monolithic_onnx' "
                    f"for model_type={model_type!r}; supported: {list(MONOLITHIC_MODEL_TYPES)}. "
                    "pi0/pi0.5/GR00T monolithic parity: scripts/local_*_monolithic_parity.py"
                )
        else:
            layout = decomposed_layout(self.config)
            if layout not in {"expert_stack", "smolvla_full_bundle"}:
                raise UnsupportedExportPipelineError(
                    f"tether validate supports only the expert_stack family, got {layout!r}"
                )

        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(UNSUPPORTED_MODEL_MESSAGE)

        self.model_type: str = model_type
        if self.model_id is None:
            self.model_id = self.config.get("model_id")

        logger.info(
            "ValidateRoundTrip ready: export_dir=%s model_type=%s threshold=%s "
            "num_test_cases=%d seed=%d device=%s",
            self.export_dir,
            self.model_type,
            self.threshold,
            self.num_test_cases,
            self.seed,
            self.device,
        )

    # ------------------------------------------------------------------ public

    def run(self) -> dict[str, Any]:
        """Execute the full round-trip validation and return a result dict.

        Loads both backends, generates ``num_test_cases`` seeded fixtures plus
        their shared initial-noise tensors, runs each fixture through both
        runtimes, compares outputs, and aggregates the results.
        """
        if self.export_kind == "monolithic_onnx":
            return self._run_monolithic()
        self.pytorch = self._load_pytorch()
        self.onnx = self._load_onnx()

        fixtures = load_fixtures(self.model_type, self.num_test_cases, self.seed)
        rng = torch.Generator(device="cpu").manual_seed(self.seed)

        per_fixture: list[dict[str, Any]] = []
        for idx, (image, prompt, state) in enumerate(fixtures):
            noise = self._generate_initial_noise(rng)
            pt_out = self.pytorch.forward(image, prompt, state, noise)
            onnx_out = self.onnx.forward(image, prompt, state, noise)
            result = self._compare(pt_out, onnx_out)
            result["fixture_idx"] = idx
            per_fixture.append(result)

        return self._aggregate(per_fixture)

    # ----------------------------------------------------------------- private

    def _run_monolithic(self) -> dict[str, Any]:
        """Exported ``model.onnx`` (ONNX Runtime, CPU) vs native LeRobot
        ``sample_actions`` on identical seeded inputs and noise.

        The reference is the unpatched LeRobot policy: nothing here installs
        Tether's export patches, so an export-only rewrite that changes
        numerics shows up as a failure. Pass/fail uses the committed
        export-parity thresholds (cosine >= MIN_COSINE and
        max_abs <= MAX_ABS_ERROR, tether/eval/export_parity.py); ``threshold``
        is not used on this path.
        """
        import onnxruntime as ort
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        from tether.eval.export_parity import DEFAULT_MAX_ABS_ERROR, DEFAULT_MIN_COSINE

        onnx_path = self.export_dir / "model.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(onnx_path)
        num_steps = int(self.config["num_denoising_steps"])
        policy = SmolVLAPolicy.from_pretrained(self.model_id)
        policy.eval().to(self.device).to(torch.float32)
        policy.model.config.num_steps = num_steps
        vocab = policy.model.vlm_with_expert.get_vlm_model().text_model.embed_tokens.num_embeddings
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        specs = {i.name: [int(d) for d in i.shape] for i in session.get_inputs()}

        per_fixture: list[dict[str, Any]] = []
        for idx in range(self.num_test_cases):
            feed = _smolvla_monolithic_feed(specs, vocab, self.seed + idx, idx)
            t = {k: torch.from_numpy(v).to(self.device) for k, v in feed.items()}
            with torch.no_grad():
                ref = policy.model.sample_actions(
                    [t["img_cam1"], t["img_cam2"], t["img_cam3"]],
                    [t["mask_cam1"], t["mask_cam2"], t["mask_cam3"]],
                    t["lang_tokens"],
                    t["lang_masks"],
                    t["state"],
                    noise=t["noise"],
                )
            ref_np = ref.detach().cpu().numpy().astype(np.float64)
            (out,) = session.run(None, feed)
            out_np = np.asarray(out, dtype=np.float64)
            if ref_np.shape != out_np.shape:
                raise ValueError(f"shape mismatch: reference {ref_np.shape} vs ONNX {out_np.shape}")
            diff = np.abs(ref_np - out_np)
            denom = np.linalg.norm(ref_np) * np.linalg.norm(out_np)
            cosine = float(ref_np.ravel() @ out_np.ravel() / denom) if denom else float(
                np.array_equal(ref_np, out_np)
            )
            max_abs = float(diff.max())
            per_fixture.append(
                {
                    "fixture_idx": idx,
                    "cosine": cosine,
                    "max_abs_diff": max_abs,
                    "mean_abs_diff": float(diff.mean()),
                    "passed": bool(
                        np.isfinite(out_np).all()
                        and cosine >= DEFAULT_MIN_COSINE
                        and max_abs <= DEFAULT_MAX_ABS_ERROR
                    ),
                }
            )

        result = self._aggregate(per_fixture)
        result.update(
            export_kind="monolithic_onnx",
            num_denoising_steps=num_steps,
            threshold=DEFAULT_MAX_ABS_ERROR,
            min_cosine=DEFAULT_MIN_COSINE,
        )
        result["summary"]["min_cosine_across_all"] = min(r["cosine"] for r in per_fixture)
        return result

    def _load_pytorch(self) -> Any:
        """Build and return the PyTorch reference backend for this export."""
        return load_pytorch_backend(self.export_dir, self.model_id, self.device)

    def _load_onnx(self) -> Any:
        """Build and return the ONNX runtime backend for this export."""
        return load_onnx_backend(self.export_dir, self.device)

    def _generate_initial_noise(self, rng: torch.Generator) -> np.ndarray:
        """Generate the initial-noise tensor shared by both backends."""
        chunk_size = int(
            self.config.get("action_chunk_size")
            or self.config.get("chunk_size")
            or 50
        )
        action_dim = int(self.config["action_dim"])
        noise = torch.randn((chunk_size, action_dim), generator=rng).numpy().astype(np.float32)
        return noise

    def _compare(
        self,
        pytorch_out: np.ndarray,
        onnx_out: np.ndarray,
    ) -> dict[str, Any]:
        """Compare a single fixture's PyTorch and ONNX outputs."""
        result = validate_outputs(
            pytorch_out, onnx_out, threshold=self.threshold, name="roundtrip"
        )
        return result.to_dict()

    def _aggregate(
        self,
        per_fixture_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate per-fixture results into a single run summary."""
        if per_fixture_results:
            max_abs = max(float(r["max_abs_diff"]) for r in per_fixture_results)
        else:
            max_abs = 0.0
        passed = all(bool(r["passed"]) for r in per_fixture_results) if per_fixture_results else False

        return {
            "model_type": self.model_type,
            "threshold": self.threshold,
            "num_test_cases": self.num_test_cases,
            "seed": self.seed,
            "results": per_fixture_results,
            "summary": {
                "max_abs_diff_across_all": max_abs,
                "passed": passed,
            },
        }


def _smolvla_monolithic_feed(
    specs: dict[str, list[int]], vocab: int, seed: int, idx: int
) -> dict[str, np.ndarray]:
    """Seeded inputs for a SmolVLA monolithic graph, shaped from its ONNX inputs.

    Fixture 0 is fully populated; odd fixtures drop camera 3; fixtures >= 2
    right-pad the language with ``idx`` masked tokens, so the camera and
    language masks are exercised as graph inputs, not just traced constants.
    """
    expected = {
        "img_cam1", "img_cam2", "img_cam3", "mask_cam1", "mask_cam2", "mask_cam3",
        "lang_tokens", "lang_masks", "state", "noise",
    }
    if set(specs) != expected:
        raise ValueError(f"unexpected SmolVLA monolithic inputs {sorted(specs)}; expected {sorted(expected)}")
    g = torch.Generator().manual_seed(seed)
    feed: dict[str, np.ndarray] = {}
    for cam in (1, 2, 3):
        feed[f"img_cam{cam}"] = (torch.rand(specs[f"img_cam{cam}"], generator=g) * 2 - 1).numpy()
        feed[f"mask_cam{cam}"] = np.full(specs[f"mask_cam{cam}"], not (cam == 3 and idx % 2), dtype=bool)
    tokens_shape = specs["lang_tokens"]
    feed["lang_tokens"] = torch.randint(0, vocab, tokens_shape, generator=g).numpy().astype(np.int64)
    lang_masks = np.ones(specs["lang_masks"], dtype=bool)
    if idx >= 2:
        lang_masks[:, max(1, tokens_shape[-1] - idx):] = False
    feed["lang_masks"] = lang_masks
    feed["state"] = torch.randn(specs["state"], generator=g).numpy()
    feed["noise"] = torch.randn(specs["noise"], generator=g).numpy()
    return feed


__all__ = [
    "SUPPORTED_MODEL_TYPES",
    "UNSUPPORTED_MODEL_MESSAGE",
    "ValidateRoundTrip",
    "ValidationResult",
    "validate_outputs",
]
