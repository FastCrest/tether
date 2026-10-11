# Operations log (public Tether)

Running record of issues faced, root causes, fixes, and costs.
Standing rule: agent vehicle scripts and reports live in the repo
(`scripts/`, `docs/parity/`), never in `/tmp` or home-directory scratch.
Full per-round reports: `docs/parity/parity-agent-report.md`,
`docs/parity/parity-pi05-refix-report.md`, `docs/parity/parity-round2-report.md`.

## 2026-09-18 — Parity campaign rounds 0-2 (pi0.5, pi0, GR00T, OpenVLA)

Template (now 4-for-4): exact 40-hex model + implementation pins, no floating
HEAD, in-function self-kill watchdog, own worktree only, A100-40GB for parity
receipt runs (A10G 22 GB cannot co-reside fp32 reference + session),
cheap-CPU export/pre-check/census before every GPU run, reuse retained
exports by hash. Total ~$3 vs ~$45 of caps.

| Item | Mechanism | Fix |
|---|---|---|
| pi0.5 bool Pad | Single bool `Pad` node, no CUDA-EP kernel | Concat-based True-pad helper (`_pad_bool_true_last_dim`), path-only |
| pi0 same pattern | Same Pad + frozen (B,16) dummies vs 48-token processor | Same helper + dummies from pinned pipeline |
| GR00T INT64 Concat | ORT force-moves tiny shape-Concat to CPU | Zeros buffer broadcast (Slice/Mul/Add only); session creates on CUDA first try |
| GR00T max_abs | cuBLAS TF32 default vs fp32-strict CPU ref (proven, not export) | RESOLVED for recorded strict fp32, full max-abs 1.98e-06; default TF32 still fails (see `docs/parity/parity-groot-strict-report.md`) |
| OpenVLA no exporter | optimum has no openvla mapping; fused-vision needs 6ch input | New `export_openvla_monolithic` (torch export, fp32, opset 18, static shapes) |
| Vehicle defect | Repo gpu_image stanza lacks CUDA wheels (`libcurand` unloadable, silently zeroing CUDA placement) | Upstream fix owed before next campaign |
| Watchdog discipline | Long GPU jobs need self-kill + attempt caps | Armed everywhere; fired only as designed |

Preserved exports (Modal volume `parity-gpu-receipts`): `pi05/export`
(Pad-free, sha `36b25828…`), `groot/export` (4.41 GB, sha `573cf899…`),
plus 30 GB OpenVLA artifact (sha `2eb6e08e…`).

## Placement pattern (3-for-3)

Every dynamic-shape factory in an exporter has produced a forced-CPU node.
Default for future exporters: broadcast-or-static (documented in
`src/tether/exporters/openvla.py`).

## 2026-09-19 — GR00T strict receipt independently verified

[Strict-fp32 report](parity/parity-groot-strict-report.md) and
[receipt](parity/receipts/groot-export-parity-receipt-strict-tf32.json) are now
retained in the repository. Both the downloaded receipt SHA-256 and original
harness-output hash verify. No GPU run or threshold change was needed.
The former `573cf899…` export was superseded; the passing receipt binds artifact
`b8ef0eb784171600ac17cefd1a82c1242d8bbd4d73e53e5e59b4d3c44e942150`.
