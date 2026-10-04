# Spine-parity GPU runs: pi0.5 (M4 #52), GR00T (M4 #53), OpenVLA (M4 #54)

Outcome: **0 of 3 harnesses pass. No acceptance weakened, no receipts fabricated.**
Each failure is root-caused to a specific, evidenced mechanism below. All runs
used the pi0 pattern (PR #343 / `fix/pin-pi0-parity-revision`): exact 40-hex
model pin, no floating HEAD, in-function self-kill watchdog, A10G except where
noted.

## Pins (all exact 40-hex)

| Family | Model | Model revision | Tether implementation |
|---|---|---|---|
| pi0.5 | `lerobot/pi05_base` | `a538eb273274eb30f126a118f39dbc0ee212c883` (2026-06-01, last before relative-action processor steps broke lerobot 0.5.1 loads — same hole as pi0's `26b99b94` pin) | `8f464ebdff9cda09d9f30fca0af5b409cfd794b6` |
| GR00T | `nvidia/GR00T-N1.6-3B` | `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61` (only main commit) | same |
| OpenVLA | `openvla/openvla-7b` | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` (HEAD; weights/code stable since 2024-09-16) | same |

Gates applied unmodified: pi0.5/OpenVLA cos >= 0.999, max-abs < 0.1;
GR00T cos >= 0.9999, max-abs < 1e-3 (its doc gate is stricter than the
0.999/0.1 summary — not relaxed). Harness scripts + `src/` untouched from
`8f464eb`; only new vehicle scripts were added (worktree
`/Users/romirjain/dev/tether-wt-parity-gpu`, branch `parity-gpu-runs`).
Shared checkouts (`tether`, `tether-pinned-18ed0100`) untouched and clean.

## 1. pi0.5 (rank 52) — FAIL, root-caused to a bool Pad node

- Export from the pinned snapshot **succeeds**: `model.onnx` + `model.onnx.data`,
  ~13 GB, retained at Modal volume `parity-gpu-receipts:pi05/export/`.
- The receipt-grade harness (`local_pi05_monolithic_parity.py`, CUDA, CPU
  fallback disabled) fails at ORT session creation: nodes assigned to CPU EP.
- Per-op bisection over the export's full 34-op census (ORT 1.23.2, one
  single-op session per type, fallback disabled): **only `Pad` fails**; all
  others pass, including `Where`/`And`/`CumSum`/`LayerNormalization`.
- The export contains exactly one Pad: `node_pad` (idx 4288), mode=constant,
  **BOOL `[1,50,784]` -> BOOL `[1,50,834]`** (Expand -> Pad -> And). ORT's CUDA
  EP has no bool-Pad kernel, so the node pins to CPU and the no-fallback gate
  (required for acceptance) fails the session.
- Source: the exporter's own F.pad mask surgery (`_apply_pi05_denoise_step_patch`
  pads bool `prefix_pad_2d_masks` with `value=True`) — the fix that restored
  cos=1.0 at num_steps=10. The numerics fix and the CUDA-EP gate are in direct
  conflict; resolving needs exporter-side surgery (e.g. Pad-free mask build via
  Concat, which probes CUDA-clean) plus re-verified parity. Not attempted here:
  implementation change with numerics risk, outside parity-runner remit.
- Inference (ASSERTED, not executed): pi0's export carries the same bool-Pad
  pattern (`monolithic.py:515-521`; smolvla/snapflow variants `:1371-1406`), so
  M4 #51 likely fails identically. Hypothesis: after the Pad is fixed, ORT's
  forced-CPU optimization (see GR00T below) may surface as the next blocker.

## 2. GR00T (rank 53) — FAIL, root-caused to one forced-CPU Concat node

- Export from the pinned snapshot **succeeds** on A100-40GB: `model.onnx`
  4.4 GB, raw_action_dim=128, retained at `parity-gpu-receipts:groot/export/`.
- The receipt-grade harness (`local_groot_monolithic_parity.py`, CUDA, no
  fallback) fails at session creation. Verbose EP-assignment capture names
  exactly one node:
  `fallback_cpu_capability.cc: Force fallback to CPU execution for node:
  node_Concat_43 because the CPU execution path is deemed faster than overhead
  involved with execution on other EPs` — followed by
  `Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1`.
- I.e. the CUDA EP *could* run it (single-op GatherND/Concat probes pass), but
  ORT's own optimizer force-moves this tiny INT64 shape-Concat to CPU, and the
  mandatory `disable_cpu_ep_fallback` makes that single node fatal. Fix is
  exporter-side (eliminate the shape-Concat or otherwise keep every node off
  the forced-fallback path) or an ORT config change in the harness — the latter
  would weaken the gate, so not done.

## 3. OpenVLA (rank 54) — BLOCKED, no exporter exists (three first-hand blockers)

- `optimum-cli export onnx --model <pinned snapshot>` with no `--task`:
  `RuntimeError: Cannot infer the task from a local directory yet` (run 1).
- With `--task image-text-to-text`: `ValueError: ... contains custom code ...
  Please pass trust_remote_code=True` — a Python-API argument the CLI cannot
  express (run 2, 13.8 s).
- Even past that: optimum 2.3.0's architecture registry (`tasks.py`, read
  first-hand) contains **zero** mentions of `openvla`; `model_type: "openvla"`
  + remote-code `auto_map` (verified in the pinned `config.json`) has no
  `OnnxConfig` mapping, so dispatch cannot succeed.
- The parity harness refuses loudly as designed: `FileNotFoundError: Missing
  OpenVLA ONNX export`, rc=1, no receipt, no vacuous pass.
- Retained: attempt record `parity-gpu-receipts:openvla/openvla-export-attempt.json`
  (explicitly `is_parity_receipt: false`; local copy at
  `/Users/romirjain/dev/parity-openvla-attempt.json`).
- Related static verification (no GPU): the pinned config sets
  `use_fused_vision_backbone: true` and `modeling_prismatic.py:120` runs
  `torch.split(pixel_values, [3, 3], dim=1)`, so the harness's `(1,3,224,224)`
  input would crash the reference even if an export existed (matches the
  `agent/openvla-54-verify` analysis; that fix is not in `8f464eb`).

## Vehicle defects found and fixed (own scripts, not the harness)

1. `transformers==5.3.0` exact pin (lib gate refuses 5.17).
2. `PYTHONPATH` passed to harness subprocesses (ModuleNotFoundError).
3. Parent GPU-cache release before parity subprocess (CUDA OOM: parent held
   22.03/22.06 GiB, child got 16 MiB).
4. Full CUDA wheel set (`cuda-runtime`, `cufft`, `curand` — `libcurand.so.10`
   was unloadable, silently zeroing CUDA placement) + fail-fast CUDA-EP smoke
   test. Note: the repo's own gpu_image stanza (cudnn+cublas only) has defect 4.

## Cost (observed GPU wall-time; ~$20 cap)

A10G ≈ 116 min total (pi0.5: 35+35+24+3.5+3+2+2+3 probe/diag minutes;
GR00T placement/GatherND probes + OpenVLA attempts ≈ 15 min; CPU inspects
negligible). A100-40GB ≈ 10 min (GR00T download + 201 s export/parity).
Single-digit dollars at Modal per-second rates — well under the cap. Watchdogs
(3300 s parity / 1500 s openvla / 600-900 s diags) never fired except as designed.

## Receipts / artifacts

- Passing parity receipts: **none** (see root causes; none weakened).
- Retained: `parity-gpu-receipts:pi05/export/` (13 GB pinned export),
  `parity-gpu-receipts:groot/export/` (4.4 GB pinned export),
  `parity-gpu-receipts:openvla/openvla-export-attempt.json`.
- Vehicles: worktree `/Users/romirjain/dev/tether-wt-parity-gpu`,
  branch `parity-gpu-runs` (12 commits, all new `scripts/modal_*` files).

## Recommendation (highest leverage next)

Fix the pi0.5 bool-Pad exporter-side (Pad-free mask build), re-run the retained
vehicle as-is; the same fix pattern likely unblocks pi0 (#51). GR00T needs the
shape-Concat eliminated or an ORT-accepted placement. OpenVLA needs a real
exporter (custom `OnnxConfig` or `export_openvla_monolithic`) before any parity
run can exist.
