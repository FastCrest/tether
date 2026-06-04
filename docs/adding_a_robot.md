# Adding a Robot — Embodiment Cookbook

> Sibling cookbook: [`adding_a_vla.md`](./adding_a_vla.md) — for adding a new VLA model on the BaseVLA spine.

Step-by-step guide to adding a new robot (arm, drone, or other manipulator) to Tether. Tether ships four shipped presets out of the box (`franka`, `so100`, `ur5`, `quadcopter`); this guide shows how to add a fifth.

The full schema lives at [`src/tether/embodiments/schema.json`](../src/tether/embodiments/schema.json) — that file is authoritative, this doc is a friendly walkthrough. The two examples below have been validated against the live schema at PR time so they parse without modification.

---

## Overview

Adding a new robot is 4 steps:

1. **Create the JSON config** — declare action space, normalization, control rates, safety constraints.
2. **Place it in the presets directory** — `src/tether/embodiments/presets/<slug>.json`.
3. **Add the slug to the schema enum** — `src/tether/embodiments/schema.json` `embodiment.enum`.
4. **Validate** — run the test suite + `tether doctor` + `tether go --dry-run`.

Two worked examples below: **MyArm-6** (6-DOF arm with gripper) and **SkyScout** (quadcopter delivery drone with payload release).

---

## Step 1: Create the JSON config

The schema has eight top-level fields. Three of them (`gripper`, `payload_release`, `constraints.max_gripper_velocity`) are *optional* — drones omit `gripper` + `max_gripper_velocity`, arms include them.

### Example A — robotic arm (MyArm-6)

Imagine a 6-axis robotic arm with a parallel-jaw gripper as the 7th component. Action space is joint positions in radians; the gripper component is normalized [0, 1].

Save to `src/tether/embodiments/presets/myarm6.json`:

```json
{
  "schema_version": 1,
  "embodiment": "myarm6",
  "action_space": {
    "type": "continuous",
    "dim": 7,
    "ranges": [
      [-3.14, 3.14],
      [-1.57, 1.57],
      [-3.14, 3.14],
      [-1.57, 1.57],
      [-3.14, 3.14],
      [-3.14, 3.14],
      [0.0, 1.0]
    ]
  },
  "normalization": {
    "mean_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    "std_action": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25],
    "mean_state": [0.0, 0.0],
    "std_state": [0.25, 0.25]
  },
  "gripper": {
    "component_idx": 6,
    "close_threshold": 0.5,
    "inverted": false
  },
  "cameras": [
    {
      "name": "wrist",
      "resolution": [640, 480],
      "fps": 30.0,
      "color_space": "rgb8"
    }
  ],
  "control": {
    "frequency_hz": 30.0,
    "chunk_size": 50,
    "rtc_execution_horizon": 10
  },
  "constraints": {
    "max_ee_velocity": 1.5,
    "max_gripper_velocity": 2.0,
    "collision_check": true
  }
}
```

Field-by-field notes:

- **`action_space.dim`** must equal the length of `ranges`, `mean_action`, and `std_action`. The cross-field validator catches mismatches with a `norm-mean-action-length-mismatch` error.
- **`mean_state` / `std_state`** are independent of action_dim — they're the shape of the state vector you POST to `/act`. Arms typically use a small representation (e.g. 2-DOF positional summary); drones use larger (see Example B).
- **`gripper.component_idx`** must be in `[0, action_dim)`. Set to `dim - 1` if the gripper is the last action component (the convention).
- **`control.frequency_hz`** is the loop rate. Arms typically run 15-30 Hz; drones 50+ Hz.
- **`constraints.max_ee_velocity`** has a hard schema ceiling of **10.0 m/s**. Realistic arm presets use 0.5-2.0 m/s; drones can use up to ~5-6 m/s for flight speed.

### Example B — quadcopter delivery drone (SkyScout)

A 5-DOF drone with body-rate control + thrust + payload release. No gripper. State is 10-DOF (position + orientation quaternion + linear velocity — matches `nav_msgs/Odometry`).

Save to `src/tether/embodiments/presets/skyscout.json`:

```json
{
  "schema_version": 1,
  "embodiment": "quadcopter",
  "action_space": {
    "type": "continuous",
    "dim": 5,
    "ranges": [
      [-3.1416, 3.1416],
      [-3.1416, 3.1416],
      [-3.1416, 3.1416],
      [0.0, 1.0],
      [0.0, 1.0]
    ]
  },
  "normalization": {
    "mean_action": [0.0, 0.0, 0.0, 0.5, 0.0],
    "std_action": [1.0, 1.0, 1.0, 0.25, 0.5],
    "mean_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "std_state": [10.0, 10.0, 10.0, 0.5, 0.5, 0.5, 0.5, 5.0, 5.0, 5.0]
  },
  "payload_release": {
    "component_idx": 4,
    "trigger_threshold": 0.5
  },
  "cameras": [
    {
      "name": "front",
      "resolution": [640, 480],
      "fps": 30.0,
      "color_space": "rgb8"
    },
    {
      "name": "downward",
      "resolution": [640, 480],
      "fps": 30.0,
      "color_space": "rgb8"
    }
  ],
  "control": {
    "frequency_hz": 50.0,
    "chunk_size": 20,
    "rtc_execution_horizon": 10
  },
  "constraints": {
    "max_ee_velocity": 5.0,
    "collision_check": true
  }
}
```

Note for the drone:

- **`"embodiment": "quadcopter"`** — the slug must match an entry in the schema enum. If you're adding a NEW slug (e.g. `"skyscout"`), you also need [Step 3](#step-3-add-the-slug-to-the-schema-enum). For variants of an existing slug, reuse the slug.
- **No `gripper` block.** Drones omit it. The cross-field validator only checks `gripper.component_idx` when the block is present.
- **No `max_gripper_velocity`** in `constraints`. Optional — required only when a `gripper` block is also present (enforced by the cross-field validator with the `gripper-missing-velocity-cap` slug).
- **`payload_release.component_idx = 4`** maps the 5th action component (zero-indexed) to the payload trigger.
- **10-DOF state** matches `nav_msgs/Odometry` (position 3 + orientation quaternion xyzw 4 + linear velocity 3). Pair with `tether ros2-serve --state-msg-type odom --state-topic /mavros/local_position/odom` at deployment.

---

## Step 2: Place it in the presets directory

The canonical location is **`src/tether/embodiments/presets/<slug>.json`**. The package's `load_preset()` reads from there; everything else is dev fallback.

```bash
# Move your config in
mv myarm6.json src/tether/embodiments/presets/myarm6.json
```

The file name (minus `.json`) must match the `"embodiment"` field inside the JSON. The loader uses the filename as the slug.

---

## Step 3: Add the slug to the schema enum

Edit [`src/tether/embodiments/schema.json`](../src/tether/embodiments/schema.json) and add your new slug to the `embodiment.enum` array:

```json
"embodiment": {
  "type": "string",
  "enum": ["franka", "so100", "ur5", "trossen", "stretch", "quadcopter", "myarm6", "custom"],
  "description": "Embodiment slug. Must match the file name minus .json for presets."
}
```

If you're reusing an existing slug (e.g. shipping a quadcopter variant), skip this step.

---

## Step 4: Validate

Three things to verify, in order:

### a. Validate the JSON programmatically

```python
from tether.embodiments import EmbodimentConfig
from tether.embodiments.validate import validate_embodiment_config

cfg = EmbodimentConfig.load_preset("myarm6")
ok, errors = validate_embodiment_config(cfg)
if ok:
    print("✓ valid")
else:
    for e in errors:
        print(f"  {e['severity']}: {e['slug']}: {e['message']}")
```

The validator runs two layers — JSON-schema (types, enums, ranges) and Python cross-field (array-length matching, gripper-index bounds, RTC horizon sanity). Warnings are non-blocking; errors are.

### b. Run the test suite

```bash
pytest tests/test_embodiments.py -v
```

Add a `test_<slug>_specifics` test if your embodiment has invariants worth pinning (e.g. action_dim, control frequency, gripper presence). See `test_quadcopter_specifics` for the pattern.

### c. End-to-end smoke

```bash
# Verify the embodiment loads + the preset table sees it
tether doctor

# Dry-run a deploy without pulling weights
tether go --model smolvla-base --embodiment myarm6 --dry-run
```

---

## Validate before opening a PR

A one-liner that catches the most common schema mistake (using wrong field names like `"mean"` instead of `"mean_action"`, or `"width"`/`"height"` instead of `"resolution"`):

```bash
python -c "
from tether.embodiments import EmbodimentConfig
from tether.embodiments.validate import validate_embodiment_config
cfg = EmbodimentConfig.load_preset('myarm6')
ok, errs = validate_embodiment_config(cfg)
blocking = [e for e in errs if e['severity']=='error']
if blocking:
    for e in blocking: print('ERROR', e['slug'], e['field'], e['message'])
    raise SystemExit(1)
print('valid')
"
```

If this exits 0 with `"valid"`, your preset round-trips through schema + cross-field. CI will catch the same things, but it's faster to fix locally.

---

## Common patterns by vertical

These align with the FastCrest customer vertical research base. Numbers are starting points — tune against your data.

### Warehouse AMR / mobile manipulator

- **Action space:** 6-8 DOF (joints) + 1 gripper
- **Control rate:** 20-30 Hz
- **Camera setup:** Wrist RGB + (optional) scene RGB
- **Hardware tier:** Jetson Orin AGX / desktop GPU
- **Reference preset:** `franka.json` (Franka Panda, 7-DOF + gripper)

### Farm / hobby manipulator (SO-100 class)

- **Action space:** 6 DOF + 1 gripper
- **Control rate:** 15 Hz (matches Orin Nano compute budget)
- **Camera setup:** Single wrist RGB
- **Hardware tier:** Jetson Orin Nano
- **Reference preset:** `so100.json` (SO-ARM 100, 5+1 DOF)

### Aerial drone (delivery, surveillance, inspection)

- **Action space:** 4-5 DOF (3 body rates + thrust ± payload release)
- **Control rate:** 50 Hz (matches PX4 outer-loop rate)
- **Camera setup:** Front RGB + downward RGB
- **State source:** `nav_msgs/Odometry` from `/mavros/local_position/odom` (full 10-DOF state — pos + quat + linear velocity)
- **Hardware tier:** Jetson Orin Nano (companion computer)
- **Reference preset:** `quadcopter.json`
- **Deploy:** `tether ros2-serve <export> --state-msg-type odom --state-topic /mavros/local_position/odom --rate-hz 50`

### Smart-camera deployment (camera-only inference)

- **Action space:** typically 0-DOF (pure perception) — Tether serves classification + bounding boxes via `/act` with whatever output channels the model produces
- **Control rate:** 10-15 Hz (frame-rate bound)
- **Camera setup:** Fixed or PTZ
- **Hardware tier:** Jetson Orin Nano / Xavier NX
- **Note:** For pure-perception use cases, `--embodiment` is optional — the preset only matters for action normalization

---

## Checklist

Before opening a PR adding a new embodiment:

- [ ] JSON config created with correct `"schema_version": 1`
- [ ] `"embodiment"` slug matches the filename (minus `.json`)
- [ ] `action_space.dim` matches the lengths of `ranges`, `mean_action`, and `std_action`
- [ ] `mean_state` and `std_state` have equal length (each other's length, not action_dim)
- [ ] If `gripper` is present, `gripper.component_idx` is in `[0, action_dim)` AND `constraints.max_gripper_velocity` is present
- [ ] If `payload_release` is present, `payload_release.component_idx` is in `[0, action_dim)`
- [ ] Config placed at `src/tether/embodiments/presets/<slug>.json` only (no duplicate in `configs/embodiments/`)
- [ ] If the slug is new, it's added to `embodiment.enum` in `src/tether/embodiments/schema.json`
- [ ] `pytest tests/test_embodiments.py` passes
- [ ] `tether doctor` reports the new preset as available
- [ ] `tether go --embodiment <slug> --dry-run` resolves cleanly

---

## See also

- [`docs/embodiment_schema.md`](./embodiment_schema.md) — full field-by-field schema reference
- [`docs/cli_reference.md`](./cli_reference.md) — every tether command and its flags
- [`docs/getting_started.md`](./getting_started.md) — step-by-step first deploy
- [`src/tether/embodiments/schema.json`](../src/tether/embodiments/schema.json) — authoritative JSON schema
- [`src/tether/embodiments/presets/`](../src/tether/embodiments/presets/) — four shipped presets as living examples
