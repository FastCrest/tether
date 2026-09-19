# pi0.5 Padfix Re-run (M4 #52 refix): PASS

**Verdict: PASS** — receipt `parity-gpu-receipts:/pi05/pi05-export-parity-receipt.json`
has `verdict: "passed"`, `external_acceptance: "recorded"`, harness returncode 0.
No gate weakened; no receipt fabricated.

## Measured numbers (from the receipt, thresholds unmodified)

| Metric | Measured | Gate | Margin |
|---|---|---|---|
| first_cosine | 0.9999995922 | ≥ 0.999 | +9.996e-04 |
| first_max_abs | 1.1896e-04 | < 0.1 | ~840x headroom |
| full_cosine | 0.9999999488 | ≥ 0.999 | +1.0e-03 |
| full_max_abs | 2.4962e-04 | < 0.1 | ~400x headroom |

Execution: `requested_provider CUDAExecutionProvider`, active
`["CUDAExecutionProvider", "CPUExecutionProvider"]` (membership semantics —
same as the receipt gate), `cpu_fallback_disabled: true`, Linux, ORT 1.23.2,
num_steps 10, input/noise seeds 42/99, shapes [1,50,32] both sides.

## Pins (all exact 40-hex)

- Model: `lerobot/pi05_base` @ `a538eb273274eb30f126a118f39dbc0ee212c883` (unchanged).
- Implementation: `f43e5977260b9eac84e23f404379e0280f34ac23`
  (branch `padfix/pi05-padfree-mask` own worktree
  `/Users/romirjain/dev/tether-wt-pi05-padfix`; nothing committed into
  `exporter-parity-fixes` or `parity-gpu-runs`).

## What changed (one file, 41+/5-)

`src/tether/exporters/monolithic.py` only — new helper
`_pad_bool_true_last_dim` (Concat with a scalar-ones+expand True block)
replaces all three `F.pad(..., value=True)` bool sites in
`_apply_pi05_denoise_step_patch` (pi0.5 path only). pi0 (`apply_export_patches`)
and both snapflow variants keep `F.pad` — separate functions, out of scope.
Harness, receipt builder, thresholds, vehicle logic: byte-identical.

Pre-GPU proof (local CPU torch): 6/6 mask shapes bit-identical vs `F.pad`
(incl. the exact `[1,50,784]+50` / `[1,50,50]+784` failing shapes);
`torch.export` lowers to `cat/expand/ones` (no `aten.pad`); ONNX graph is
`And`+`Concat` only.

## Run sequence and spend (~$1.3 estimated, cap $10)

1. A10G fresh-export run (~36.5 min GPU): export rebuilt from `f43e5977`
   (13.0 GB), session creation **passed the placement gate** (the original
   bool-Pad failure is gone) but ORT then OOM'd allocating a 128 MB arena —
   the fp32 reference + 13 GB session cannot co-reside in 22 GB. Environmental,
   not numerics: failure moved from placement to capacity.
2. CPU op census on the retained export (~1 min, negligible): **19,110 nodes,
   0 Pad, 779 Concat** — the bool Pad is eliminated (was exactly 1).
3. A100-40GB parity rerun reusing that export (~4 min GPU): PASS, numbers above.
   GPU-tier change only; gates, model pin, watchdog, image unchanged.

Note: the A100 vehicle's stdout summary shows `first_cosine: None` etc. —
a key-name mismatch in the vehicle's receipt echo, not the receipt; the
`[receipt]` JSON line and the downloaded receipt carry the real metrics above.

## Follow-ups (not blockers)

- pi0 (#51) likely carries the same bool-Pad pattern (`monolithic.py:515-521`)
  and the same A10G capacity cliff — apply the identical helper there and run
  parity on A100-40GB when scheduled.
- The retained `parity-gpu-receipts:pi05/export/` is now the Pad-free artifact
  (sha `36b25828…`); the pre-fix 13 GB export it overwrote is superseded.
