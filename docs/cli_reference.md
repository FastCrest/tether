# Reflex CLI Command Reference

Complete reference for every visible `reflex` command in v0.9.x. The exhaustive flag list for any command is always one keystroke away — run `reflex <command> --help` for the live, source-of-truth list. This document covers the surface most users actually touch plus worked examples per vertical.

---

## Quick orientation — 14 visible verbs

After the v0.9.5 CLI cut, `reflex --help` shows these 14 top-level verbs. Each is described in its own section below.

| Verb | Purpose |
|---|---|
| [`go`](#reflex-go) | One-command deploy: probe hardware → pick model → pull → export → serve |
| [`serve`](#reflex-serve) | Start an inference server from an exported model directory |
| [`doctor`](#reflex-doctor) | Diagnose install + GPU issues + per-deploy traps |
| [`eval`](#reflex-eval) | Task-success eval (LIBERO success rate + per-task numbers + optional video) |
| [`chat`](#reflex-chat) | Natural-language agent that runs reflex commands for you |
| [`models`](#reflex-models) | Browse + download Reflex-compatible VLA models from HuggingFace |
| [`train`](#reflex-train) | Finetune checkpoints, distill teachers into 1-NFE students |
| [`validate`](#reflex-validate) | Pre-flight validation — datasets before training, exports before serving |
| [`inspect`](#reflex-inspect) | Diagnostic + forensic tools — bench, replay, targets, guard state |
| [`traces`](#reflex-traces) | Searchable + summarizable view over recorded `/act` traces |
| [`pro`](#reflex-pro) | Reflex Pro — activate, check, or deactivate your license |
| [`contribute`](#reflex-contribute) | Reflex Data Contribution — opt in / out / check / revoke |
| [`curate`](#reflex-curate) | Convert recorded traces to published dataset formats |
| [`data`](#reflex-data) | Manage episode data uploads and contributions |

Less-used + power-user verbs (hidden from `reflex --help` but fully supported) live in [Advanced commands](#advanced-commands).

---

## `reflex go`

One-command deploy. Probes hardware, resolves a model from the registry, pulls weights, exports to ONNX if needed, and starts serving on the chosen port.

```bash
# Robot arm: Franka with pi0.5, default port 8000
reflex go --model pi05 --embodiment franka

# Edge: SmolVLA on Jetson Orin Nano with explicit hardware override
reflex go --model smolvla-base --device-class orin_nano --port 8001

# Quadcopter (drone embodiment)
reflex go --model smolvla-base --embodiment quadcopter --port 8002

# Plan without pulling — useful when probing
reflex go --model pi05-libero --dry-run
```

| Key flag | Default | Purpose |
|---|---|---|
| `--model` | _(required)_ | Registry ID (`pi05-libero`) or family (`pi05`, `smolvla`, `pi0`) — see `reflex models list` |
| `--embodiment` | _(none)_ | Preset name: `franka`, `so100`, `ur5`, `quadcopter`. Cross-checks dataset / action shapes |
| `--device-class` | _(auto)_ | Override hardware probe: `h200`, `h100`, `a100`, `a10g`, `thor`, `agx_orin`, `orin_nano`, `cpu` |
| `--port` / `--host` | `8000` / `0.0.0.0` | HTTP listener for `/act` + `/health` |
| `--api-key` | _(none)_ | If set, `/act` requires `X-Reflex-Key` header (or `Authorization: Bearer`) |
| `--dry-run` | `false` | Probe + resolve + print plan; do not pull or serve |

Full flag list: `reflex go --help`. Note: models that ship as raw PyTorch require the `[monolithic]` extra (`pip install 'reflex-vla[monolithic]'`) for the inline export step.

---

## `reflex serve`

Production inference server. Serves an already-exported model directory via HTTP. Use this when you've separated the export step (CI builds the export, deployment serves it).

```bash
# Basic serve
reflex serve ./reflex_export/ --port 8000

# With safety limits clamping actions to joint bounds
reflex serve ./reflex_export/ --safety-config safety_limits.json

# With API auth (X-Reflex-Key or Authorization: Bearer)
reflex serve ./reflex_export/ --api-key "$REFLEX_API_KEY"

# Adaptive denoising for lower latency
reflex serve ./reflex_export/ --adaptive-steps
```

| Key flag | Default | Purpose |
|---|---|---|
| `export_dir` | _(required)_ | Path to the exported model directory |
| `--port` / `--host` | `8000` / `0.0.0.0` | Server port + bind address |
| `--device` | `cuda` | Execution device: `cuda` or `cpu` |
| `--providers` | _(auto)_ | Comma-separated ORT execution providers (e.g. `TensorrtExecutionProvider,CUDAExecutionProvider`) |
| `--no-strict-providers` | `false` | Allow silent fallback to CPU when requested providers fail to load |
| `--safety-config` | _(none)_ | Path to a SafetyLimits JSON (see `reflex inspect guard`) |
| `--adaptive-steps` | `false` | Early-stop denoising when velocity norm converges (`reflex turbo` heritage) |
| `--api-key` | _(none)_ | Require auth header on `/act` and `/config` |
| `--cloud-fallback` | _(none)_ | URL of a remote `reflex serve` for cloud-edge split-execution |
| `--ros2` | `false` | Short-circuit HTTP and run the [ROS2 bridge](#advanced-commands) instead |

Full flag list: `reflex serve --help`.

### HTTP endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/act` | Yes (if `--api-key`) | Send image + instruction + state → receive actions |
| `GET` | `/health` | No (always open for orchestrator probes) | Readiness state |
| `GET` | `/config` | Yes (if `--api-key`) | Server configuration + model metadata |

---

## `reflex doctor`

Pre-deploy + post-deploy health check. Detects the silent-failure traps that bite VLA deployments at edge — CUDA / cuDNN version skew, JetPack target mismatch, Blackwell sm_120 support, ONNX Runtime EP loadchain, multi-GPU mixed-architecture warnings, and more.

```bash
# Top-level: install + GPU sanity check
reflex doctor

# Per-deploy: validate that a specific export will actually run on this box
reflex doctor --export-dir ./reflex_export/
```

Full flag list: `reflex doctor --help`.

---

## `reflex eval`

Task-success eval against LIBERO (or any compatible benchmark suite). Reports per-task success rate, aggregate numbers, optional video rollout.

```bash
reflex eval ./reflex_export/ --task libero-spatial --episodes 50
```

Full flag list: `reflex eval --help`. See [`docs/eval.md`](./eval.md) for the methodology.

---

## `reflex chat`

Natural-language agent that wraps the rest of the CLI. Talk to your robot fleet in plain English; the agent calls `models list`, `doctor`, `serve`, etc. on your behalf. Hosted via the FastCrest proxy at `chat.fastcrest.com` (GPT-5-mini). 100 calls/day free, no signup, no API key.

```bash
reflex chat
```

Full flag list: `reflex chat --help`.

---

## `reflex models`

Browse + download Reflex-compatible VLA checkpoints from the curated registry.

```bash
# Browse — registry table with hardware tier + status
reflex models list

# Download to ~/.cache/reflex/models/<id>/
reflex models pull pi05-libero

# Inspect a single model's metadata + supported embodiments
reflex models info smolvla-base
```

Full subcommand list: `reflex models --help`.

---

## `reflex train`

Train models. Two subcommands — finetune an existing checkpoint, or distill a teacher into a 1-NFE student via SnapFlow.

```bash
# Finetune
reflex train finetune --base smolvla-base --data ./my_dataset/

# Distill (1-step student from N-step teacher)
reflex train distill --teacher ./teacher_export/ --steps 1
```

Full subcommand list: `reflex train --help`. See [`docs/self_distilling_serve.md`](./self_distilling_serve.md) for the continuous-distill loop (Pro tier).

---

## `reflex validate`

Pre-flight validation. Two subcommands:

```bash
# Validate a LeRobot v2/v3 dataset before training
reflex validate dataset ./dataset/

# Validate an exported model's round-trip parity vs PyTorch
reflex validate export ./reflex_export/
```

Full subcommand list: `reflex validate --help`.

---

## `reflex inspect`

Diagnostic + forensic tools. The visible subcommand is `traces`; related top-level commands (`bench`, `replay`, `targets`, `guard`) are hidden but supported — call them directly (e.g. `reflex replay --help`).

```bash
# View per-task trace rollups
reflex inspect traces
```

Full subcommand list: `reflex inspect --help`.

---

## `reflex traces`

Search + summarize JSONL traces written by `reflex serve --record <dir>`.

```bash
# Filter recorded /act traces
reflex traces query --task pick-up-cup --status failed --since 7d --output failures.json

# Aggregate by task / model / day
reflex traces summary --by task --since 7d
```

Full subcommand list: `reflex traces --help`.

---

## `reflex pro`

Manage Reflex Pro license. See [Pricing](https://docs.fastcrest.com/pricing/) for tier details.

```bash
reflex pro activate <license-key>
reflex pro status
reflex pro deactivate
```

Full subcommand list: `reflex pro --help`.

---

## `reflex contribute`

Reflex Data Contribution program. Opt in to share anonymized eval traces back to the registry in exchange for early access to community-curated improvements. Opt-in only, fully reversible.

```bash
reflex contribute --status      # check current state
reflex contribute --opt-in
reflex contribute --opt-out
reflex contribute --revoke      # erase previously contributed data
```

Full flag list: `reflex contribute --help`.

---

## `reflex curate`

Convert recorded traces into published dataset formats (LeRobot v3, raw JSONL, parquet).

```bash
reflex curate convert ./traces/ --format lerobot-v3 --out ./dataset/
```

Full subcommand list: `reflex curate --help`.

---

## `reflex data`

Manage episode data uploads + contributions. Server-side review, stats, revocation.

```bash
reflex data review   # open the review UI
reflex data stats    # aggregate stats over contributed episodes
reflex data revoke <episode-id>
```

Full subcommand list: `reflex data --help`.

---

## Vertical quick-start matrix

Pick a starting point that matches what you're deploying. Each row points at the canonical command + key flag; substitute your own model / embodiment / hardware as needed.

| Vertical | Use case | Canonical command |
|---|---|---|
| **Warehouse AMR** | Pick + sort + place on grid carts (Symbotic / GreyOrange / Ocado tier) | `reflex go --model pi05 --embodiment franka --port 8000` |
| **Autonomous tractors** | Row navigation + boom control on John Deere-class platforms | `reflex go --model smolvla-base --embodiment ur5 --device-class agx_orin` |
| **Mining** | Autonomous haul + drill positioning (Cat-class fleets) | `reflex go --model pi05 --embodiment ur5 --device-class a10g` |
| **Drone surveillance** | Aerial pattern-of-life ISR (defense + civilian) | `reflex go --model smolvla-base --embodiment quadcopter --port 8002` |
| **Last-mile drone delivery** | Civilian + tactical drop with payload release | `reflex go --model smolvla-base --embodiment quadcopter` |
| **Traffic management** | Adaptive signal control at edge (NoTraffic / Rekor / NVIDIA Metropolis pattern) | `reflex serve ./traffic_export/ --device-class orin_nano --deadline-ms 100` |
| **Smart-camera retail** | Loss prevention + SKU recognition at the shelf | `reflex serve ./retail_export/ --adaptive-steps --deadline-ms 100` |
| **Smart-camera warehouse** | Multi-camera AMR / forklift / worker tracking | `reflex serve ./warehouse_export/ --max-batch-cost-ms 200` |
| **ADAS / autonomous trucking** | In-vehicle perception pipeline | `reflex serve ./adas_export/ --providers TensorrtExecutionProvider --strict-providers` |
| **Maritime port inspection** | ROV / surface drone hull + container inspection | `reflex go --model smolvla-base --embodiment quadcopter` |
| **Autonomous baggage tugs** | Airside tow-vehicle pickup + drop (Stinger / Towflexx) | `reflex go --model pi05 --embodiment so100` |

Vertical research lives in the FastCrest customer research notes — these are the **P0 picks** by composite pay × fit × velocity score (top 11, all 12+/15).

---

## Advanced commands

These commands are hidden from `reflex --help` to keep the discovery surface focused, but they're production-supported. Run `reflex <command> --help` for full details.

### `reflex ros2-serve`

ROS2 bridge node. Subscribes to image + state + task topics, runs inference, publishes action chunks. Requires a ROS2 install (humble / iron / jazzy) — `rclpy` is NOT pip-installable.

```bash
# Source ROS2 first
source /opt/ros/humble/setup.bash

# Robotic arm — default state extractor reads sensor_msgs/JointState
reflex ros2-serve ./reflex_export/ \
  --image-topic /camera/image_raw \
  --state-topic /joint_states \
  --action-topic /reflex/actions \
  --rate-hz 20

# Drone with full 10-DOF state (pos + quat + linear velocity)
reflex ros2-serve ./reflex_export/ \
  --state-topic /mavros/local_position/odom \
  --state-msg-type odom \
  --rate-hz 50

# Drone with orientation-only fallback (4-DOF)
reflex ros2-serve ./reflex_export/ \
  --state-topic /mavros/imu/data \
  --state-msg-type imu \
  --rate-hz 50

# With MCP exposure for agent-driven control
reflex ros2-serve ./reflex_export/ --mcp --mcp-transport stdio
```

| Key flag | Default | Purpose |
|---|---|---|
| `export_dir` | _(required)_ | Exported model directory |
| `--state-msg-type` | `joint_state` | How to extract the state vector: `joint_state` (arms), `imu` (drone partial — 4 DOF), `odom` (drone full — 10 DOF) |
| `--state-topic` | `/joint_states` | State topic. For drones: `/mavros/local_position/odom` (with `--state-msg-type odom`) or `/mavros/imu/data` (with `--state-msg-type imu`) |
| `--image-topic` | `/camera/image_raw` | `sensor_msgs/Image` |
| `--action-topic` | `/reflex/actions` | `std_msgs/Float32MultiArray` |
| `--rate-hz` | `20.0` | Inference rate (50 Hz typical for drones, 20 Hz for arms) |
| `--mcp` | `false` | Also expose the live ROS2 node as MCP tools (Claude Desktop / Cursor) |

Full flag list: `reflex ros2-serve --help`.

### `reflex replay`

Replay a recorded trace through a fresh export — useful for regression testing a new model against a known-good rollout.

```bash
reflex replay ./traces/episode_0001.jsonl --against ./new_export/
```

Full flag list: `reflex replay --help`.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `REFLEX_NO_UPGRADE_CHECK=1` | Suppress the daily PyPI upgrade nag |
| `REFLEX_API_KEY` | Default API key for `reflex serve --api-key` (alternative to passing on the command line) |
| `CUDA_VISIBLE_DEVICES` | Restrict which GPUs reflex sees — useful on mixed-architecture multi-GPU hosts |

---

## See also

- [`README.md`](../README.md) — install + verb-surface overview
- [`docs/getting_started.md`](./getting_started.md) — step-by-step first deploy
- [`docs/embodiment_schema.md`](./embodiment_schema.md) — per-robot config reference
- [`docs/eval.md`](./eval.md) — eval methodology
- [`docs/doctor_check_list.md`](./doctor_check_list.md) — what `reflex doctor` checks
- [`docs/record_replay.md`](./record_replay.md) — record + replay traces

Hidden internal-only commands (`bench`, `bench-game`, `targets`, `guard`, `check`, `calibrate`, `status`, `config show/set`, `validate-legacy`, `validate-dataset`) are intentionally omitted from this reference. They remain callable for power-user scripts and CI hooks.

Deprecated verbs: `turbo` (replaced by `serve --adaptive-steps`), `split` (replaced by `serve --cloud-fallback`), `adapt` (folded into `reflex guard`). All three print a deprecation banner pointing at the replacement when invoked.
