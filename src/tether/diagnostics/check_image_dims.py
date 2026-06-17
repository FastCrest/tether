"""Check 4 — Image dim mismatch (LeRobot #1700).

Cross-checks the embodiment config's `cameras[*].resolution` against the
ONNX model's image input shape. Mismatches mean the client either resizes
on its end or the export was wrong — silently sending the wrong-sized
image produces garbage actions.
"""
from __future__ import annotations

from pathlib import Path

from . import Check, CheckResult, register

CHECK_ID = "check_image_dims"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/1700"


def _run(model_path: str, embodiment_name: str = "custom", **kwargs) -> CheckResult:
    p = Path(model_path)
    if not p.exists():
        return CheckResult(
            check_id=CHECK_ID,
            name="Image dim mismatch",
            status="skip",
            expected="export dir to inspect ONNX inputs",
            actual="export dir missing (caught by check_model_load)",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    if embodiment_name == "custom":
        return CheckResult(
            check_id=CHECK_ID,
            name="Image dim mismatch",
            status="skip",
            expected="--embodiment <preset> to cross-check resolution",
            actual="embodiment=custom — no preset to compare against",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Load embodiment config
    try:
        from tether.embodiments import EmbodimentConfig
        cfg = EmbodimentConfig.load_preset(embodiment_name)
    except (ValueError, FileNotFoundError) as e:
        return CheckResult(
            check_id=CHECK_ID,
            name="Image dim mismatch",
            status="fail",
            expected=f"embodiment preset {embodiment_name!r} loads",
            actual=f"load failed: {e}",
            remediation=(
                f"Use a shipped preset (franka/so100/ur5) or pass --custom-embodiment-config "
                f"<path>. See docs/embodiment_schema.md."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    config_cameras = cfg.cameras
    if not config_cameras:
        return CheckResult(
            check_id=CHECK_ID,
            name="Image dim mismatch",
            status="skip",
            expected="embodiment.cameras with resolution",
            actual=f"{embodiment_name} has no cameras configured",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Inspect ONNX model inputs
    try:
        import onnxruntime as ort
    except ImportError:
        return CheckResult(
            check_id=CHECK_ID,
            name="Image dim mismatch",
            status="skip",
            expected="onnxruntime to inspect model inputs",
            actual="onnxruntime not installed",
            remediation="pip install fastcrest-tether[serve]",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    onnx_files = sorted(p.glob("*.onnx"))
    if not onnx_files:
        return CheckResult(
            check_id=CHECK_ID,
            name="Image dim mismatch",
            status="skip",
            expected="ONNX file in export dir",
            actual="no .onnx files (caught by check_model_load)",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Load only graph metadata, not weights — quickly via onnx package
    try:
        import onnx
        model = onnx.load(str(onnx_files[0]), load_external_data=False)
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            check_id=CHECK_ID,
            name="Image dim mismatch",
            status="warn",
            expected="ONNX graph parses for input inspection",
            actual=f"onnx.load raised {type(e).__name__}: {e}",
            remediation=(
                "Skipping image-dim check. Run `tether serve` to surface the same "
                "issue at startup if the ONNX is genuinely broken."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Find image-shaped inputs (3D or 4D tensors with 3 channels)
    image_inputs: list[tuple[str, list[int | None]]] = []
    for inp in model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(None)  # dynamic dim
        # Heuristic: image is 3D (H,W,C) or 4D (B,C,H,W) or (B,H,W,C) with C in {1,3,4}
        is_image = (
            len(shape) in (3, 4)
            and any(d == 3 for d in shape if d is not None)
        )
        if is_image:
            image_inputs.append((inp.name, shape))

    if not image_inputs:
        return CheckResult(
            check_id=CHECK_ID,
            name="Image dim mismatch",
            status="skip",
            expected="image-shaped input in ONNX graph",
            actual="no 3D/4D inputs with channel-dim 3 found",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Compare config resolution vs ONNX shape
    cfg_w, cfg_h = config_cameras[0]["resolution"]  # check first camera only
    mismatches: list[str] = []
    for name, shape in image_inputs:
        # Find spatial dims (any non-3 dim that's not 1 — heuristic)
        spatial = [d for d in shape if d is not None and d != 3 and d != 1]
        if not spatial:
            continue  # all dynamic, can't compare
        if cfg_h not in spatial or cfg_w not in spatial:
            mismatches.append(
                f"input {name!r} shape {shape} doesn't include "
                f"({cfg_h}, {cfg_w}) from {embodiment_name}.cameras[0]"
            )

    if mismatches:
        return CheckResult(
            check_id=CHECK_ID,
            name="Image dim mismatch",
            status="warn",
            expected=f"ONNX image input contains ({cfg_h}, {cfg_w})",
            actual=mismatches[0],
            remediation=(
                f"Either: (a) resize images in client before /act (most common — "
                f"OpenCV cv2.resize to model's expected H,W), (b) update "
                f"{embodiment_name}.json cameras[0].resolution to match the model, "
                f"or (c) re-export with native resolution."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="Image dim mismatch",
        status="pass",
        expected=f"ONNX image input includes ({cfg_h}, {cfg_w})",
        actual=f"{len(image_inputs)} image input(s) match",
        remediation="",
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="Image dim mismatch",
    severity="warn",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
