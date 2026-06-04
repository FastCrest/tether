"""Check 7 — State/proprio deserialization (LeRobot #2458).

Verifies the ONNX model's `state` input expects float32, and the
embodiment config's normalization.mean_state / std_state lengths match.
LeRobot #2458 reports float64 silently truncated by ORT → ~0.3 fps perf
collapse and wrong actions.
"""
from __future__ import annotations

from pathlib import Path

from . import Check, CheckResult, register

CHECK_ID = "check_state_proprio"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/2458"


def _run(model_path: str, embodiment_name: str = "custom", **kwargs) -> CheckResult:
    p = Path(model_path)
    if not p.exists():
        return CheckResult(
            check_id=CHECK_ID,
            name="State/proprio dtype",
            status="skip",
            expected="export dir exists for ONNX state-input inspection",
            actual="export dir missing (caught by check_model_load)",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    onnx_files = sorted(p.glob("*.onnx"))
    if not onnx_files:
        return CheckResult(
            check_id=CHECK_ID,
            name="State/proprio dtype",
            status="skip",
            expected="ONNX file in export dir",
            actual="no .onnx files (caught by check_model_load)",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Inspect ONNX inputs without loading weights
    try:
        import onnx
        model = onnx.load(str(onnx_files[0]), load_external_data=False)
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            check_id=CHECK_ID,
            name="State/proprio dtype",
            status="warn",
            expected="ONNX graph parses for state-input inspection",
            actual=f"onnx.load raised {type(e).__name__}: {e}",
            remediation="Skipping state-dtype check; surface this issue in check_model_load.",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Find a state-shaped input — heuristic: name contains 'state' or 'proprio'
    state_input = None
    for inp in model.graph.input:
        name_lower = inp.name.lower()
        if "state" in name_lower or "proprio" in name_lower:
            state_input = inp
            break

    if state_input is None:
        return CheckResult(
            check_id=CHECK_ID,
            name="State/proprio dtype",
            status="skip",
            expected="ONNX input named 'state' or 'proprio'",
            actual=f"no state input found in {len(list(model.graph.input))} inputs",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # ONNX TensorProto.FLOAT == 1, FLOAT16 == 10, DOUBLE == 11, INT64 == 7
    elem_type = state_input.type.tensor_type.elem_type
    elem_name = {1: "float32", 10: "float16", 11: "float64", 7: "int64"}.get(
        elem_type, f"type_{elem_type}"
    )

    if elem_type != 1:  # not float32
        return CheckResult(
            check_id=CHECK_ID,
            name="State/proprio dtype",
            status="fail",
            expected="state input dtype is float32 (ONNX TensorProto.FLOAT)",
            actual=f"state input is {elem_name} (TensorProto code {elem_type})",
            remediation=(
                f"State input is {elem_name}, not float32. Per LeRobot #2458, "
                f"sending float64 state to a float32-expecting graph silently truncates "
                f"and tanks throughput to ~0.3 fps. Re-export the model OR cast state "
                f"in client: np.asarray(state, dtype=np.float32)."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Cross-check state dim against embodiment.normalization.mean_state length
    if embodiment_name != "custom":
        try:
            from tether.embodiments import EmbodimentConfig
            cfg = EmbodimentConfig.load_preset(embodiment_name)
            cfg_state_dim = cfg.state_dim

            # Get the last dim of the ONNX state shape (state vector dim)
            shape_dims = [
                d.dim_value if d.dim_value > 0 else None
                for d in state_input.type.tensor_type.shape.dim
            ]
            onnx_state_dim = shape_dims[-1] if shape_dims and shape_dims[-1] else None

            if onnx_state_dim is not None and onnx_state_dim != cfg_state_dim:
                # Many state inputs are padded to a max — emit warn, not fail
                if onnx_state_dim > cfg_state_dim:
                    return CheckResult(
                        check_id=CHECK_ID,
                        name="State/proprio dtype",
                        status="warn",
                        expected=f"state dim matches between ONNX ({onnx_state_dim}) and {embodiment_name} ({cfg_state_dim})",
                        actual=f"ONNX expects {onnx_state_dim}, embodiment provides {cfg_state_dim} (will be padded)",
                        remediation=(
                            "ONNX state dim larger than embodiment — runtime pads with "
                            "zeros. This is intended for cross-embodiment-trained models. "
                            "Verify the pad-to-max behavior is what you want."
                        ),
                        duration_ms=0.0,
                        github_issue=GH_ISSUE,
                    )
                return CheckResult(
                    check_id=CHECK_ID,
                    name="State/proprio dtype",
                    status="fail",
                    expected=f"ONNX state dim ({onnx_state_dim}) ≥ embodiment state dim ({cfg_state_dim})",
                    actual=f"ONNX expects {onnx_state_dim}, embodiment provides {cfg_state_dim}",
                    remediation=(
                        f"State-dim mismatch: model wants {onnx_state_dim} but "
                        f"{embodiment_name} only has {cfg_state_dim} state dims. "
                        f"Either re-export with smaller state input or extend the "
                        f"embodiment's normalization arrays."
                    ),
                    duration_ms=0.0,
                    github_issue=GH_ISSUE,
                )
        except (ValueError, FileNotFoundError):
            # Embodiment load failure surfaces in other checks; don't double-report
            pass

    return CheckResult(
        check_id=CHECK_ID,
        name="State/proprio dtype",
        status="pass",
        expected="ONNX state input is float32",
        actual=f"state input {state_input.name!r} is {elem_name}",
        remediation="",
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="State/proprio dtype",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
