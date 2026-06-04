# tether doctor — frozen check list

10 falsifiable checks. Each maps to a known LeRobot GitHub issue or systemic VLA deploy failure mode. Every check has a 1-pass + 1-fail unit test in `tests/test_doctor_diagnostics.py` per the falsifiability gate.

| ID | Name | What it tests | LeRobot issue |
|---|---|---|---|
| `check_model_load` | Model load | Export dir exists + contains ONNX + fits in available RAM (×1.4 overhead, 20% headroom) | [#386](https://github.com/huggingface/lerobot/issues/386), [#414](https://github.com/huggingface/lerobot/issues/414) |
| `check_onnx_provider` | ONNX provider | onnxruntime importable + CPU EP present (always required) + GPU EP noted | [#2137](https://github.com/huggingface/lerobot/issues/2137) |
| `check_vlm_tokenization` | VLM tokenization | Tokenizer config loads + 5 probe prompts produce in-range token IDs (skips when tokens are baked into ONNX) | [#2119](https://github.com/huggingface/lerobot/issues/2119), [#683](https://github.com/huggingface/lerobot/issues/683) |
| `check_image_dims` | Image dim mismatch | `embodiment.cameras[*].resolution` appears in ONNX image input shape | [#1700](https://github.com/huggingface/lerobot/issues/1700) |
| `check_action_denorm` | Action denormalization | `embodiment.normalization.mean_action` / `std_action` length == `action_dim`, no NaN/Inf, std > 0 | [#414](https://github.com/huggingface/lerobot/issues/414), [#2210](https://github.com/huggingface/lerobot/issues/2210) |
| `check_gripper` | Gripper config | `gripper.component_idx` < `action_dim`, `close_threshold` ∈ [0, 1], `inverted` flag sanity-checked against embodiment defaults | [#2210](https://github.com/huggingface/lerobot/issues/2210), [#2531](https://github.com/huggingface/lerobot/issues/2531) |
| `check_state_proprio` | State/proprio dtype | ONNX `state` input is float32 (not float64 — silent truncation drops fps to ~0.3); state dim ≥ embodiment state dim | [#2458](https://github.com/huggingface/lerobot/issues/2458) |
| `check_gpu_memory` | GPU memory | `nvidia-smi` reports ≥ 90% headroom over estimated model footprint (×1.6 file size for KV + activations) | [#2137](https://github.com/huggingface/lerobot/issues/2137) |
| `check_rtc_chunks` | RTC chunk boundary | `chunk_size` ≥ `frequency_hz × rtc_execution_horizon` (one horizon's worth of actions); clean integer multiple preferred (warn otherwise) | [#2356](https://github.com/huggingface/lerobot/issues/2356), [#2531](https://github.com/huggingface/lerobot/issues/2531) |
| `check_hardware_compat` | Hardware compat | CUDA driver ≥ 12.x + ORT GPU EP present when CUDA detected; warns on version drift (the most-common silent CPU fallback cause per ADR 2026-04-14) | [#2137](https://github.com/huggingface/lerobot/issues/2137) |

## CheckResult contract

Every check returns a `CheckResult` (see `src/tether/diagnostics/__init__.py`):

| Field | Type | Notes |
|---|---|---|
| `check_id` | str | Stable ID (e.g. `check_model_load`); used by `--skip` |
| `name` | str | Human-readable name (e.g. "Model load") |
| `status` | enum | `pass` \| `fail` \| `warn` \| `skip` |
| `expected` | str | What the check wanted to see |
| `actual` | str | What it actually saw |
| `remediation` | str | Required when `status="fail"`. Empty otherwise. |
| `duration_ms` | float | Wall-clock for the check (auto-filled if check returns 0.0) |
| `github_issue` | str \| None | URL to the load-bearing LeRobot issue (or None) |

**Falsifiability gate**: `CheckResult.__post_init__` raises `ValueError` if `status="fail"` and `remediation` is empty. Enforced at construction time so a check with no fix-it suggestion can never ship.

## Status semantics

- **pass** — the check verified the expected condition. No action.
- **fail** — the check verified a known-broken condition. Doctor exits 1. Caller should follow `remediation`.
- **warn** — the check found a non-blocking concern (e.g. CPU-only on a system that should have GPU). Doctor still exits 0 but the warning is surfaced.
- **skip** — the check couldn't run because a precondition wasn't met (e.g. embodiment is `custom` so embodiment-dependent checks have nothing to compare against). Not an error.

## Adding check #11

1. Create `src/tether/diagnostics/check_<name>.py` with a `_run(model_path, embodiment_name, **kwargs) -> CheckResult` function.
2. At the bottom: `register(Check(check_id=..., name=..., severity=..., github_issue=..., run_fn=_run))`.
3. Import the new module in `_ensure_registry_loaded()` in `src/tether/diagnostics/__init__.py`.
4. Add at least 1 pass test + 1 fail test to `tests/test_doctor_diagnostics.py`.
5. Update this doc with the new row.

The registry is auto-loaded; no other wiring needed.

## Output format

`tether doctor --model <dir> --format json` emits an envelope validated by `docs/doctor_output_schema.json` (JSON Schema draft-07). Stable schema_version=1; additive fields don't bump version, breaking changes do.
