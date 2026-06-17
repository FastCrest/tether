"""Curated registry rows. Edit this file (NOT models.py) to add/remove entries.

PR convention when adding a row:
1. Verify the model loads + serves with `tether serve <export>` end-to-end
2. Run the bundled parity test against an existing model in the same family
3. Pin `hf_revision` to a specific commit sha (not HEAD) for reproducibility
4. Add benchmark numbers from the matching `03_experiments/*.md` if measured
5. Set `requires_export=False` ONLY if the HF repo contains
   tether_config.json + the .onnx files (a Tether-ready export, not raw weights)

Initial registry seeded 2026-04-24 with public LeRobot-distributed weights —
all require `tether export` after `pull` because they ship as raw PyTorch
checkpoints. Tether-pre-exported entries (under our own HF org) added in a
follow-up once the export-and-upload pipeline ships.
"""
from __future__ import annotations

from tether.registry.models import ModelBenchmark, ModelEntry

REGISTRY: tuple[ModelEntry, ...] = (
    ModelEntry(
        model_id="pi0-base",
        hf_repo="lerobot/pi0_base",
        family="pi0",
        action_dim=32,
        size_mb=14000,  # ~14 GB BF16
        supported_embodiments=("franka", "so100", "ur5"),
        supported_devices=("a10g", "a100", "h100", "h200"),
        benchmarks=(
            ModelBenchmark(device="a10g", p50_ms=110.0, p99_ms=180.0, vram_mb=14000,
                           measured_at="2026-04-14"),
        ),
        requires_export=True,
        description="pi0 base from Physical Intelligence (LeRobot mirror). 7B params, BF16. "
                    "Run `tether export pi0-base` after pull to produce ONNX.",
        license="apache-2.0",
        hf_revision=None,  # use HEAD until we pin
    ),
    ModelEntry(
        model_id="pi05-base",
        hf_repo="lerobot/pi05_base",
        family="pi05",
        action_dim=32,
        size_mb=14000,
        supported_embodiments=("franka", "so100", "ur5"),
        supported_devices=("a10g", "a100", "h100", "h200"),
        benchmarks=(
            ModelBenchmark(device="a10g", p50_ms=98.0, p99_ms=160.0, vram_mb=14000,
                           measured_at="2026-04-14"),
        ),
        requires_export=True,
        description="pi0.5 base — Physical Intelligence's improved pi0 (state-out variant). "
                    "Use this as the teacher for SnapFlow distillation.",
        license="apache-2.0",
        hf_revision=None,
    ),
    ModelEntry(
        model_id="pi05-libero",
        hf_repo="lerobot/pi05_libero_finetuned_v044",
        family="pi05",
        action_dim=7,  # Franka — 7 joints
        size_mb=14000,
        supported_embodiments=("franka",),
        supported_devices=("a10g", "a100", "h100"),
        benchmarks=(
            ModelBenchmark(device="a10g", p50_ms=98.0, p99_ms=160.0, vram_mb=14000,
                           measured_at="2026-04-14"),
        ),
        requires_export=True,
        description="pi0.5 finetuned on LIBERO benchmark (Franka 7-DoF). "
                    "Tether's reference SnapFlow teacher — 28/30 task-success at 10 NFE.",
        license="apache-2.0",
        hf_revision=None,
    ),
    ModelEntry(
        model_id="smolvla-base",
        hf_repo="lerobot/smolvla_base",
        family="smolvla",
        action_dim=7,
        size_mb=900,  # SmolVLA is small (~900MB FP32)
        supported_embodiments=("franka", "so100"),
        supported_devices=("orin_nano", "agx_orin", "a10g", "a100", "h100"),
        benchmarks=(
            ModelBenchmark(device="a10g", p50_ms=22.0, p99_ms=45.0, vram_mb=900,
                           measured_at="2026-04-14"),
        ),
        requires_export=True,
        description="SmolVLA — HuggingFace's small VLA. Edge-friendly (~900MB). "
                    "Best starting point for Jetson Orin Nano deployments.",
        license="apache-2.0",
        hf_revision=None,
    ),
    ModelEntry(
        model_id="smolvla-libero",
        hf_repo="lerobot/smolvla_libero",
        family="smolvla",
        action_dim=7,
        size_mb=900,
        supported_embodiments=("franka",),
        supported_devices=("orin_nano", "agx_orin", "a10g", "a100"),
        benchmarks=(),  # not measured yet
        requires_export=True,
        description="SmolVLA finetuned on LIBERO. Smaller-footprint alternative to pi0.5-libero "
                    "for edge-first deployments.",
        license="apache-2.0",
        hf_revision=None,
    ),
    # ──────────────────────────────────────────────────────────────────────
    # Added 2026-05-10 after customer report — exporters shipped but
    # registry entries missing → `tether models list`, `tether chat`, and
    # `tether doctor` all said "not supported" even though export pipeline
    # works. Contract test in tests/test_registry_completeness.py prevents
    # future drift.
    # ──────────────────────────────────────────────────────────────────────
    ModelEntry(
        model_id="gr00t-n1.6",
        hf_repo="nvidia/GR00T-N1.6-3B",
        family="groot",  # validation uses 'groot' (existing convention; NVIDIA brand is "GR00T")
        action_dim=32,  # GR00T humanoid action space (joint targets + grippers)
        size_mb=3290,  # 3.29B params per README
        # GR00T is humanoid-focused but supports embodiment-id-based routing
        # so single + dual-arm configs work via the standard preset path.
        supported_embodiments=("humanoid", "franka", "so100"),
        # Strategic signal per spec: NVIDIA Inception path; runs on Thor in
        # production. Excludes orin_nano (3.29B params @ FP16 = ~6.5GB > 8GB
        # tier comfortable budget when combined with activations + OS).
        supported_devices=("agx_orin", "thor", "a10g", "a100", "h100", "h200"),
        benchmarks=(),  # in-house numbers TBD; export validated to cos parity
        requires_export=True,
        description="NVIDIA GR00T N1.6 — humanoid-focused VLA with DiT action expert + "
                    "Eagle (SigLIP+Llama) VLM backbone. Validated max_diff=8.34e-07 vs "
                    "PyTorch reference. Run `tether export gr00t-n1.6` after pull.",
        license="nvidia-source-code-license",  # NVIDIA's bespoke OSS-adjacent license
        hf_revision=None,
    ),
    ModelEntry(
        model_id="openvla-7b",
        hf_repo="openvla/openvla-7b",
        family="openvla",
        action_dim=7,  # discrete action tokens decoded to 7-DoF continuous
        size_mb=7500,  # 7.5B params per README
        # OpenVLA was trained on Open X-Embodiment cross-embodiment dataset;
        # broad coverage via embodiment-id mapping.
        supported_embodiments=("franka", "so100", "ur5", "widowx"),
        # 7.5B model needs >=14GB VRAM in BF16. Excludes Jetson Orin Nano + AGX.
        supported_devices=("a10g", "a100", "h100", "h200"),
        benchmarks=(),  # in-house numbers TBD
        requires_export=True,
        description="OpenVLA — vanilla Llama-2-7B VLM with discrete action tokens. "
                    "Export uses optimum-cli onnx path + the bin-to-continuous postprocess "
                    "helper at `tether.postprocess.openvla.decode_actions`. Run `tether "
                    "export openvla-7b` after pull.",
        license="mit",
        hf_revision=None,
        # Decision S-4: OpenVLA stays a shim, not on the BaseVLA spine.
        # argmax-over-bins doesn't fit the flow-matching component pattern.
        vla_type="_openvla_shim",
    ),
    # ──────────────────────────────────────────────────────────────────────
    # Lift #4: FluxVLA pi0.5 LIBERO-10 fine-tuned checkpoint (Apache-2.0).
    # Published 97.85% on LIBERO-10 across 4 subsuites (FluxVLA paper).
    # Converted from limxdynamics/FluxVLAEngine to lerobot format.
    # ──────────────────────────────────────────────────────────────────────
    ModelEntry(
        model_id="pi05-libero10-fluxvla",
        hf_repo="Rylinjames/pi05-libero10-finetune-v1",
        family="pi05",
        action_dim=7,
        size_mb=8300,
        supported_embodiments=("franka",),
        supported_devices=("a10g", "a100", "h100", "h200"),
        benchmarks=(),
        requires_export=True,
        description="FluxVLA's pi0.5 finetuned on LIBERO-10 (BS=64, 24 epochs). "
                    "Published 97.85% average success. Apache-2.0 from LimX Dynamics. "
                    "Republished with attribution from limxdynamics/FluxVLAEngine.",
        license="apache-2.0",
        hf_revision=None,
    ),
    # ──────────────────────────────────────────────────────────────────────
    # Lift #7: DreamZero WAM (World-Action Model) — NVIDIA Research via FluxVLA.
    # 6th model family on the spine. Joint video + action diffusion.
    # 94.65% LIBERO average per FluxVLA benchmarks.
    # ──────────────────────────────────────────────────────────────────────
    ModelEntry(
        model_id="dreamzero-libero10",
        hf_repo="limxdynamics/FluxVLAEngine",
        family="dreamzero",
        action_dim=7,
        size_mb=28000,  # ~14B params (Wan2.1 DiT + T5 + CLIP + VAE)
        supported_embodiments=("franka",),
        supported_devices=("a100", "h100", "h200"),  # 14B model needs >=40GB VRAM
        benchmarks=(),
        requires_export=True,
        description="DreamZero — NVIDIA Research World-Action Model. Joint video + action "
                    "diffusion on Wan 2.1 DiT backbone. 94.65% LIBERO average. Apache-2.0 "
                    "via FluxVLA. Use `tether export dreamzero-libero10` after pull.",
        license="apache-2.0",
        hf_revision=None,
    ),
    # ──────────────────────────────────────────────────────────────────────
    # MolmoAct2 — Allen AI's VLA. SigLIP2 + Qwen3 Molmo2-ER + flow-matching
    # action expert. 7th model family on the spine.
    # ──────────────────────────────────────────────────────────────────────
    ModelEntry(
        model_id="molmoact2-base",
        hf_repo="allenai/MolmoAct2",
        family="molmoact2",
        action_dim=7,
        size_mb=21800,
        supported_embodiments=("franka", "so100", "yam"),
        supported_devices=("a10g", "a100", "h100", "h200"),
        benchmarks=(),
        requires_export=False,
        description="MolmoAct2 base — Allen AI's VLA. SigLIP2 + Molmo2-ER (Qwen3) "
                    "+ flow-matching action expert. 5B params. Apache-2.0.",
        license="apache-2.0",
        hf_revision=None,
    ),
    ModelEntry(
        model_id="molmoact2-libero",
        hf_repo="allenai/MolmoAct2-LIBERO",
        family="molmoact2",
        action_dim=7,
        size_mb=21800,
        supported_embodiments=("franka",),
        supported_devices=("a10g", "a100", "h100", "h200"),
        benchmarks=(),
        requires_export=False,
        description="MolmoAct2 LIBERO fine-tune — Allen AI's VLA on LIBERO benchmark. "
                    "Franka 7-DoF. Apache-2.0.",
        license="apache-2.0",
        hf_revision=None,
    ),
    ModelEntry(
        model_id="molmoact2-droid",
        hf_repo="allenai/MolmoAct2-DROID",
        family="molmoact2",
        action_dim=7,
        size_mb=21800,
        supported_embodiments=("franka",),
        supported_devices=("a10g", "a100", "h100", "h200"),
        benchmarks=(),
        requires_export=False,
        description="MolmoAct2 DROID fine-tune — Franka DROID embodiment. Apache-2.0.",
        license="apache-2.0",
        hf_revision=None,
    ),
)
