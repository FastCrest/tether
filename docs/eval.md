# Eval-as-a-service (`tether eval`)

`tether eval ./my-export/ --suite libero --num-episodes 3` — one command, ~30 minutes, LIBERO success rate + per-task numbers + optional MP4 clips + cost transparency. Wraps the existing Modal image + `osmesa`/MuJoCo recipe + the `vla-eval` adapter.

Per ADR `2026-04-25-eval-as-a-service-architecture`. Phase 1 ships LIBERO only on Modal (with Linux x86_64 local fallback); Phase 2 adds SimplerEnv + `customer` suite + HF Hub video upload.

## Quick start

```bash
# 1. (Optional) Set up Modal auth — skip if you only run --runtime local
modal token new

# 2. Smoke run (~$0.20, ~3 minutes on A10G cold start)
tether eval ./my-export/ \
    --suite libero \
    --num-episodes 3 \
    --tasks libero_spatial

# 3. Full bench (~$10, ~30 minutes)
tether eval ./my-export/ \
    --suite libero \
    --num-episodes 50 \
    --video \
    --output ./eval-out/

# 4. Cost preview before kicking off something expensive
tether eval ./my-export/ \
    --num-episodes 100 \
    --cost-preview
```

Output: `./eval-out/report.json` (machine-readable, schema v1) + `./eval-out/videos/<task>_episode_<N>.mp4` (when `--video` set).

## Why this exists

Every research group evaluating a new VLA asks for the same thing: "give me task-success numbers I can put in a paper". The existing path is `clone modal_libero_*.py + figure out the auth + figure out the dep pins + handle the 5 documented failure modes + parse the output yourself`. That's 1-2 days of yak-shaving per group. `tether eval` ships the whole path in one verb.

## The 10 flags

| Flag | Default | Notes |
|---|---|---|
| `--suite` | `libero` | Phase 1 ships LIBERO only. Phase 2: `simpler`, `customer`. |
| `--num-episodes` | `3` | Per-task. 3 = smoke; 50-100 = published-paper grade. |
| `--tasks` | `(all)` | Comma-separated. Empty = the 4 LIBERO families (spatial / object / goal / 10). |
| `--runtime` | `modal` | `modal` = bundled image (turnkey). `local` = Linux x86_64 + `[eval-local]` extra. |
| `--seed` | `0` | Matches `tether bench`. Pass `--seed 7` to reproduce prior `modal_libero_*.py` published runs. |
| `--max-parallel` | `1` | Honored when the runtime supports it (Modal yes, local no). |
| `--cost-preview` | `false` | Dry-run: estimate `$` without invoking. Useful before 100-ep × 90-task runs. |
| `--video` | `false` | Per-episode MP4 to `<output>/videos/`. Cap at ~10MB / episode. |
| `--output` | `./eval_output` | Directory for JSON envelope + (optional) videos. Created if missing. |
| `--preflight-timeout` | `300` | Seconds for the LIBERO smoke test. Cold `osmesa` scene-compile can take 60-180s. |

## Pre-flight smoke test

Before invoking the expensive run, `tether eval` runs a **pre-flight** in an isolated subprocess that exercises the LIBERO init path (import + `OffScreenRenderEnv` + `env.reset()`). This catches **4 of the 5 documented LIBERO failure modes** in ~2 seconds, before you spend $$ on a doomed run.

Bounded `failure_mode` enum, surfaced in the CLI + telemetry:

| Mode | Fix |
|---|---|
| `input-hang` | Run `scripts/patch_libero.py` first (or use `--runtime modal` — the bundled image patches this in). |
| `egl-black-frames` | Force `MUJOCO_GL=osmesa` in your env. |
| `dep-version-conflict` | Pin `robosuite==1.4.1`, `bddl==1.0.1`, `mujoco==3.3.2`. Use `--runtime modal` for known-good pins. |
| `osmesa-compile-hang` | Increase `--preflight-timeout` (cold containers take 60-180s for first-scene compile). |
| `import-error` | `pip install 'fastcrest-tether[eval-local]'` for local; `--runtime modal` for the bundled image. |

The 5th failure (per-episode OOM) is per-call probabilistic; backoff + a legible error in the runner covers it.

## JSON envelope (schema v1 — LOCKED)

`<output>/report.json` is the machine-readable envelope. Schema v1 is **locked**; Phase 2 evolution is additive-only. Customers grep on these fields in CI; renaming = breakage.

```jsonc
{
  "schema_version": 1,
  "reflex_version": "0.1.0+dev",
  "suite": "libero",
  "runtime": "modal",
  "seed": 0,
  "started_at": "2026-04-25T14:30:00.000000Z",
  "finished_at": "2026-04-25T14:33:21.000000Z",
  "wall_clock_s": 201.0,
  "tasks": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
  "num_episodes_per_task": 3,
  "aggregate": {"success_rate": 0.83, "n_success": 10, "n_total": 12},
  "results": [
    {"task_id": "libero_spatial", "n_success": 3, "n_total": 3, "success_rate": 1.0},
    /* ... per-task ... */
  ],
  "episodes": [
    {"task_id": "libero_spatial", "episode_index": 0, "success": true,
     "terminal_reason": "success", "wall_clock_s": 28.4, "n_steps": 200,
     "video_path": "./eval-out/videos/libero_spatial_episode_0.mp4",
     "error_message": null},
    /* ... flat list across all tasks ... */
  ],
  "cost": {
    "total_usd": 0.50,
    "suite": "libero", "runtime": "modal",
    "num_episodes_per_task": 3, "n_tasks": 4,
    "usd_per_episode": 0.025, "usd_per_task_startup": 0.10,
    "by_task": {"libero_spatial": 0.175, /* ... */ },
    "cost_table_schema_version": 1,
    "notes": "Modal runtime: 4 tasks × (3 eps × $0.025/ep + $0.10 cold-startup). ..."
  },
  "modal": {"image_digest": "sha256:abc...", "provider": "modal.com"},
  "env": {
    "timestamp_utc": "...", "reflex_version": "...",
    "git_sha": "deadbeefcafe", "git_dirty": false,
    "python_version": "3.13.11", "platform": "Darwin-25.3.0-arm64",
    "export_dir": "/path/to/export", "onnx_files": [{"name": "model.onnx", "sha256": "...", "bytes": 12345}]
  },
  "video_paths": ["./eval-out/videos/libero_spatial_episode_0.mp4"],
  "notes": []
}
```

### `terminal_reason` enum

Bounded; stable across releases. Valid values:

- `success` — task completed successfully (`success: true` REQUIRED)
- `timeout` — episode hit `--preflight-timeout` or runner-side cap
- `bddl_failure` — task BDDL file failed to parse
- `rendering_failure` — `osmesa` / EGL render returned an error
- `adapter_error` — anything else the runner didn't classify

Cross-field invariant: `success == True` if and only if `terminal_reason == "success"`. Enforced at construction time.

## Cost transparency

`tether eval --cost-preview` prints a `$` estimate **before** invoking. The cost table is baked at ship time and refreshed quarterly against actual Modal billing logs.

Current rates (cost-table schema v1):

| Suite × Runtime | $ / episode | $ / task-startup |
|---|---|---|
| `libero` × `modal` | $0.025 (A10G) | $0.10 (cold container + image pull + osmesa scene compile) |
| `libero` × `local` | $0.00 | $0.00 |

Above-`$50` estimate triggers an extra **"are you sure?"** warning so the customer doesn't accidentally fire a 1000-episode × 90-task run.

## Local fallback (`--runtime local`)

Phase 1: **Linux x86_64 only**. Requires the `[eval-local]` extra:

```bash
pip install 'fastcrest-tether[eval-local]'
```

`--runtime local` **never silently falls back to Modal**. If the local env is broken, `tether eval` fails loud with a remediation pointer. This avoids surprise Modal bills + masks real env-config issues.

macOS local fallback is Phase 2 (the `osmesa` + MuJoCo + `lerobot` dep stack on macOS isn't ready).

## Doctor integration

`tether doctor` gains 3 additive checks (no new `--eval-ready` subflag — keeps the doctor surface flat):

- `check_modal_auth` — pass when `~/.modal.toml` exists OR `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` env are set
- `check_libero_importable` — pass when `import libero` works
- `check_vla_eval_importable` — pass when the Tether internal adapter at `src/tether/runtime/adapters/vla_eval.py` imports

All 3 are **warns** (not fails) — customers not using `tether eval` aren't blocked.

## Reproducing prior LIBERO numbers

The `modal_libero_*.py` scripts produced the published 80%+ LIBERO numbers using `seed=7`. To reproduce:

```bash
tether eval ./my-export/ \
    --suite libero \
    --num-episodes 50 \
    --seed 7 \
    --runtime modal
```

The `env` block in `report.json` captures `git_sha`, `python_version`, `platform`, and per-`*.onnx` `sha256` hashes — enough to re-run + cross-check. Treat the `cost_table_schema_version` field as the canonical pin (changes here mean cost numbers shifted).

## What's deliberately NOT shipped Phase 1

- **SimplerEnv suite** — Phase 2.
- **`customer` suite** (run on the customer's own task set) — Phase 2; needs a stable customer-task spec format.
- **HF Hub video upload** — Phase 2; composes naturally with `self_distilling_serve.md` HF token plumbing.
- **macOS local fallback** — Phase 2; `osmesa`/MuJoCo/`lerobot` stack on macOS isn't ready.
- **Concurrent task fan-out on local runtime** — Phase 1 local is sequential; Modal honors `--max-parallel`.

## Common errors

### `Pre-flight FAILED (osmesa-compile-hang, 300.1s)`

Cold containers take 60-180s for first-scene compile. Bump `--preflight-timeout 600` and retry. If reproducible, file a GitHub issue with the stderr from the smoke test.

### `Pre-flight FAILED (dep-version-conflict, ...)`

Pin: `robosuite==1.4.1`, `bddl==1.0.1`, `mujoco==3.3.2`. Easiest path: drop `--runtime local` and use `--runtime modal` (the bundled image has the known-good pins).

### `Estimate exceeds $50 guardrail`

Drop `--num-episodes` or use fewer `--tasks`. The guardrail is conservative; if you really do want a 1000-ep run, just pass `--num-episodes 1000` and hit Enter — it'll proceed.

### `All episodes returned adapter_error` (exit 5)

Either:
- The Modal subprocess crashed mid-run — check `report.json` for the `error_message` on the per-episode rows. The first ~500 chars of `modal run` stderr are surfaced there.
- You're on `--runtime local` (the local runner is Phase 1 follow-up; use `--runtime modal` for now).
- The Modal `run_libero_*` script's stdout format changed — the parser pins on `====== <suite> (ONNX monolithic) ======` headers. File a bug if the script was bumped.

### `modal CLI not found on PATH` (exit 6)

Install via `pip install modal` then run `modal token new` to authenticate. `--runtime modal` cannot proceed without it; we never silently fall back to `--runtime local` (would mask real config issues + cost surprises).

## Pricing

`tether eval` is **free** to use. You pay Modal directly for compute (per the cost table above). Tether Pro license is **not required** for `tether eval` — it's the open-source serve-side feature. (Pro license gates `--collect-data`, `--distill-schedule`, etc. on the `tether serve` side; see `self_distilling_serve.md`.)
