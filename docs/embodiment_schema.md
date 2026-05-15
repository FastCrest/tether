# Embodiment config schema (v1)

`reflex serve --embodiment <name>` reads a per-robot config from `configs/embodiments/<name>.json`. Three presets ship out of the box: **franka**, **so100**, **ur5**. You can also point at a custom config with `--custom-embodiment-config <path>`.

The authoritative schema is at `src/reflex/embodiments/schema.json` (JSON Schema draft-07). This doc is a human-readable mirror — if they drift, the JSON is canonical.

## Top-level structure

```json
{
  "schema_version": 1,
  "embodiment": "franka",
  "action_space":   { ... },
  "normalization":  { ... },
  "gripper":        { ... },
  "cameras":        [ ... ],
  "control":        { ... },
  "constraints":    { ... }
}
```

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int (= 1) | Bump only on removed/renamed fields. Additive changes don't bump. |
| `embodiment` | enum | `franka` \| `so100` \| `ur5` \| `trossen` \| `stretch` \| `quadcopter` \| `custom` |

## `action_space`

The robot's commanded action vector.

| Field | Type | Range | Notes |
|---|---|---|---|
| `type` | enum | `continuous` \| `discrete` | Reflex currently only supports `continuous`. |
| `dim` | int | 1–32 | Number of action dimensions (e.g. 7 for Franka 6-DOF arm + gripper). |
| `ranges` | array of [min, max] | length = `dim` | Per-dim hard limits (radians for joints, [0,1] for gripper width). |

**Cross-field rules** (validated in Python, not JSON schema):
- `len(ranges) == dim` (else `action-ranges-length-mismatch`)
- For each `[lo, hi]`: `lo < hi` (else `action-range-inverted`)

## `normalization`

Action/state denormalization stats. The model is trained on normalized inputs; these undo the normalization at runtime.

| Field | Type | Notes |
|---|---|---|
| `mean_action` | array of floats | Length must equal `action_space.dim`. |
| `std_action`  | array of floats | Same length. Each element must be > 0 and ≤ 100. |
| `mean_state`  | array of floats | Length defines the state vector size (no separate `state_dim` field). |
| `std_state`   | array of floats | Must equal `mean_state` length. Each element > 0, ≤ 100. |

**Cross-field rules:**
- `len(mean_action) == action_space.dim` (else `norm-mean-action-length-mismatch`)
- `len(std_action) == action_space.dim` (else `norm-std-action-length-mismatch`)
- `len(mean_state) == len(std_state)` (else `norm-state-length-mismatch`)

## `gripper`

| Field | Type | Range | Notes |
|---|---|---|---|
| `component_idx` | int | 0–`action_space.dim` − 1 | Index into the action vector that controls the gripper. |
| `close_threshold` | float | 0.0–1.0 | Action value above which gripper is "close." |
| `inverted` | bool |  | If true, low values close the gripper. |

**Cross-field rule:** `0 ≤ component_idx < action_space.dim` (else `gripper-idx-out-of-range`).

## `cameras`

List of camera streams the model expects (1–8 cameras).

| Field | Type | Range | Notes |
|---|---|---|---|
| `name` | string | non-empty | Logical name (e.g. `wrist`, `front`). Must be unique across cameras. |
| `resolution` | [int, int] | 32–4096 | Width × height in pixels. |
| `fps` | float | 0–240 (exclusive 0) | Stream frame rate. |
| `color_space` | enum | `rgb8` \| `bgr8` \| `mono8` \| `rgba8` |  |

**Cross-field rule:** camera names must be unique (else `duplicate-camera-name`).

## `control`

| Field | Type | Range | Notes |
|---|---|---|---|
| `frequency_hz` | float | 0–1000 (exclusive 0) | Robot control loop rate. |
| `chunk_size` | int | 1–200 | Actions in a single inference chunk. |
| `rtc_execution_horizon` | float | 0–5.0 (exclusive 0) | Seconds of chunk to execute before requesting next inference. |

**Cross-field rule (warning, not blocking):**
- `frequency_hz × rtc_execution_horizon ≥ 1` (else `rtc-horizon-too-short` warning — RTC degenerates if the horizon is shorter than one control step).

## `constraints`

| Field | Type | Range | Notes |
|---|---|---|---|
| `max_ee_velocity` | float | 0–10.0 (exclusive 0) | End-effector velocity cap (m/s). |
| `max_gripper_velocity` | float | 0–10.0 (exclusive 0) | Gripper open/close velocity cap. |
| `collision_check` | bool |  | Whether the runtime should run a per-step collision check. |

## Validation

Two layers run on every config load:

1. **JSON Schema** (`jsonschema.Draft7Validator`) — enforces types, enums, ranges, required fields, `additionalProperties: false`.
2. **Cross-field rules** (Python) — enforces array-length parity, gripper-idx bounds, RTC horizon sanity, camera uniqueness.

Each failure carries a stable error slug (`action-ranges-length-mismatch`, `gripper-idx-out-of-range`, etc.) so downstream tools can map to docs / GitHub issues.

```python
from reflex.embodiments import EmbodimentConfig
from reflex.embodiments.validate import validate_embodiment_config, format_errors

cfg = EmbodimentConfig.load_preset("franka")
ok, errors = validate_embodiment_config(cfg)
if not ok:
    print(format_errors(errors))
```

CI gate: `bash scripts/verify_embodiment_structure.sh && pytest tests/test_embodiments.py -v`.

## Adding a new preset

1. Add a new dict to `scripts/emit_embodiment_presets.py` — copy a similar robot's preset and adjust DOFs, gripper index, normalization stats.
2. Add the slug to the `embodiment` enum in `src/reflex/embodiments/schema.json`.
3. Add the slug to `ALL_PRESETS` in `tests/test_embodiments.py`.
4. Run `python scripts/emit_embodiment_presets.py` — emits the new JSON, validates it.
5. `pytest tests/test_embodiments.py -v` — should pass on the new preset's parametrized tests.
6. Commit JSON + script change + schema change + test change in one commit.

## Editor integration (recommended)

Add to your project `.vscode/settings.json` to get autocomplete on embodiment JSONs:

```json
{
  "json.schemas": [
    {
      "fileMatch": ["configs/embodiments/*.json"],
      "url": "./src/reflex/embodiments/schema.json"
    }
  ]
}
```

(Reflex doesn't ship `.vscode/settings.json` itself — that's user config.)

## Related

- Plan: [`features/01_serve/subfeatures/_rtc_a2c2/per-embodiment-configs_plan.md`](https://github.com/FastCrest/reflex-vla) (in reflex_context vault)
- Canonical feature page: `features/01_serve/subfeatures/_rtc_a2c2/per-embodiment-configs.md`
- Downstream consumers (Phase 0.5):
  - **B.3 RTC adapter** uses `control.frequency_hz`, `control.rtc_execution_horizon`, `control.chunk_size`
  - **B.6 action denormalization** uses `constraints.max_ee_velocity`, `gripper.*`
  - **D.1 reflex doctor** validates `gripper.inverted`, `control.chunk_size`
