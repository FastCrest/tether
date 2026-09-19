# GR00T strict-math parity receipt (M4 #53, gate-owner option exercised)

Verdict: **PASS under a permanently recorded strict-math environment.**
Receipt `parity-gpu-receipts:groot/groot-export-parity-receipt-strict-tf32.json`:
verdict `passed`, `external_acceptance: recorded`, harness rc 0, ORT 1.23.2,
Linux, CUDA with fallback disabled (`scope: linux-cuda`).

Worktree `/Users/romirjain/dev/tether-wt-groot-strict`, branch
`padfix/groot-strict-parity`, branched off trunk `18157ec` (current
`origin/main` at run time). One new file only:
`scripts/modal_groot_strictmath_parity.py` (committed `f1cdc6d`). No
threshold, gate, fallback-flag, harness, exporter, or other-family change.

## Measured numbers (unmodified harness, thresholds unmodified)

| Metric | Measured | Gate | Headroom |
|---|---|---|---|
| first_cosine | 1.00000000 (0.99999999999976) | ≥ 0.9999 | ~1e-4 above gate |
| first_max_abs | 4.17e-07 | < 1e-3 | ~2400x |
| full_cosine | 1.00000000 (0.99999999999935) | ≥ 0.9999 | ~1e-4 above gate |
| full_max_abs | 1.98e-06 | < 1e-3 | ~500x |

Thresholds in the receipt are the pinned values
(`minimum_cosine` 0.9999, `maximum_absolute_error_exclusive` 1e-3).
Shapes `[1,50,128]` == `[1,50,128]`; shared-input sha
`28ea4ecc…` (seeds 42/99, embodiment 0).

## Pins

- Model `nvidia/GR00T-N1.6-3B @ d0814e7ecb19202e7c8468b46098b0b7ef3a6d61`
  (snapshot of the exact revision; reference and export agree by construction).
- Tether implementation `4cf9854a72ca9c97ec6a7a398f4246650b001d2a`
  (container checkout; harness recorded `git rev-parse HEAD`).

## Export: hash differed, rebuilt identically per spec

The volume's `groot/export/model.onnx` no longer hashed to the retained
`573cf899…` identity (intervening round-2 profile/matrix runs had reused that
directory), so the vehicle rebuilt via the identical path
(`export_gr00t_monolithic` from the pinned snapshot, same tether commit,
embodiment 0) instead of reusing. New artifact 4.41 GB, identity
`b8ef0eb7…` (self-consistent: the receipt binds the artifact it tested).
Result matches the round-2 diagnostic order (1.67e-06 → 1.98e-06).

## Env recorded (the asterisk, auditable in-receipt)

Top-level `environment` block in the retained strict receipt:

- `NVIDIA_TF32_OVERRIDE: "0"`, `math_mode: strict-fp32 (cuBLAS/cuDNN IEEE
  fp32, TF32 disabled)`
- `raw_harness_receipt_sha256: 57777993…` — sha of the exact bytes the
  unmodified harness wrote before annotation; re-verified by stripping the
  block and re-hashing (match: true), so metrics/verdict/thresholds are
  byte-identical to harness output and only the additive block differs.
- Strict receipt sha `a34fcf84…` (verified by independent volume read-back).

Default-env behavior is as-is: `groot-export-parity-receipt.json` was opened
read-only (hash `30ab9b5b…` before AND after — untouched), and the strict run
wrote only the new `-strict-tf32` path (guard-asserted never to equal the
default path).

## Spend (~$0.4 estimated vs ~$10 cap)

One A100-40GB run, ~3.5 GPU-min wall (snapshot cached, export rebuild +
parity; watchdog armed, never fired). No further runs needed; cap untouched.

## Independent read-back, 2026-09-19

Downloaded the retained receipt from Modal without launching a GPU job.
The [receipt](receipts/groot-export-parity-receipt-strict-tf32.json) hashes to
`a34fcf84471c7842192d35c9af8975984840aa2b97f530424ca1c9d132d21d0e`.
Removing only `environment`, serializing with sorted keys, two-space indentation
and a trailing newline reproduces the recorded raw harness SHA-256
`577779938ea9761c681b2459c4ba3633710b264f457810342b10ebb56848b584`.
Thresholds remain cosine >= 0.9999 and max-abs < 0.001. Default-environment
failure remains a separate finding; this pass does not qualify default TF32 math.
