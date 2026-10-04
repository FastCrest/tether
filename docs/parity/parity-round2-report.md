# Exporter parity round two (M4 #51/#53/#54): 2 PASS, 1 FAIL with proven root cause

Outcome: **pi0 PASS, OpenVLA PASS, GR00T FAIL as-specified** (placement blocker
eliminated; remaining gap proven to be backend matmul precision, not export
fidelity). No gate weakened, no receipt fabricated. All work on own worktree
`/Users/romirjain/dev/tether-wt-round2`, branch `padfix/round2-parity` (20
commits, pushed to origin). Shared/pinned checkouts untouched and clean.

Template reused from the pi0.5 refix: exact 40-hex model + implementation pins,
in-function self-kill watchdogs, A100-40GB for parity (A10G cannot co-reside
reference + session), cheap-CPU export/pre-check before every GPU receipt run.

## 1. pi0 (rank 51) — PASS

Receipt `parity-gpu-receipts:pi0/pi0-export-parity-receipt.json`: verdict
`passed`, `external_acceptance: recorded`, harness rc 0, ORT 1.23.2, Linux,
CUDA with fallback disabled, num_steps 10.

| Metric | Measured | Gate | Headroom |
|---|---|---|---|
| first_cosine | 0.99999983 | ≥ 0.999 | ~1e-3 |
| first_max_abs | 4.0072e-04 | < 0.1 | ~250x |
| full_cosine | 0.99999944 | ≥ 0.999 | ~1e-3 |
| full_max_abs | 1.1170e-03 | < 0.1 | ~90x |

Pins: model `lerobot/pi0_base @ 26b99b9439acb1e352439e34ee9c67af0d76efa3`;
tether `6e84101136dcaa6204983956b80418796430504a` (export built from `d414707`,
whose pi0 exporter/harness files are byte-identical at the receipt commit —
verified by diff). Artifact 12.57 GB, sha `daab820a…`.

Two fixes, both pi0-only (`apply_export_patches` path; pi0.5/snapflow untouched):
- `0e08a30`: the proven Concat-based True-pad helper applied to
  `monolithic.py:515-521`. Local proof 6/6 shapes bit-identical vs `F.pad`,
  lowering `cat/expand/ones` only. Retained-export census: 16,934 nodes,
  **0 Pad**. The first A100 run then *passed session creation* (placement
  gate fixed) but failed at `session.run`: `lang_tokens Got 48 Expected 16`.
- `d414707`: the export froze a `(B, 16)` dummy while the harness feeds the
  processor's real tokenization (48 tokens). Dummies now read lang/state
  lengths off the pinned `PolicyProcessorPipeline` on the canonical probe
  task — same pins, same pipeline as the harness, values stay random. CPU
  pre-check then passed (cos 1.0, max_abs 1.37e-06); A100 receipt above.

## 2. GR00T (rank 53) — FAIL as specified, placement fixed, root cause proven

Receipt `parity-gpu-receipts:groot/groot-export-parity-receipt.json`: verdict
`failed`, harness rc 1. Default-env numbers: first_cos 0.99999990 ✓,
first_abs 2.99e-04 ✓, full_cos 0.99999972 ✓, **full_abs 1.2579e-03 vs < 1e-3** ✗
(+26%). Pins: model `nvidia/GR00T-N1.6-3B @ d0814e7ecb19202e7c8468b46098b0b7ef3a6d61`;
tether `4cf9854a72ca9c97ec6a7a398f4246650b001d2a`; export rebuilt 4.41 GB,
sha `573cf899…`.

Placement (the delegated blocker) is **eliminated**: CPU-only census of the
retained export showed `node_Concat_43 = Concat(Shape(batch), 1, 2048)` →
Expand → MatMuls (the `torch.zeros(b, 1, 2048)` VLM-KV placeholder), the only
INT64 Concat of 3. Replaced with a `[1,1,2048]` zeros buffer broadcast against
a `[b,1,1]` slice — bit-exact zeros (proven B=1,2), lowers to Slice/Mul/Add
only. New export creates its session on CUDA with fallback disabled, first try.
Reference and export share the module, so parity holds by construction.

Numerics, measured (same artifact + inputs, three backends on one A100 box):
- torch-CPU vs ORT-CPU: **3.78e-06** — the export is faithful.
- torch-CPU vs ORT-CUDA: 1.26e-03, uniform across all 50 positions — pure
  backend-kernel delta (ORT-CPU vs ORT-CUDA alone: 1.26e-03).
- Reference matrix: strict-fp32 torch-CUDA == torch-CPU (2.1e-06, sanity ✓);
  TF32 torch-CUDA vs torch-CPU differ at 1.39e-03 (TF32 floor confirmed);
  TF32 torch-CUDA vs ORT-CUDA: 6.7e-04 (both reduced-precision, different kernels).
- **Intervention**: unmodified harness under `NVIDIA_TF32_OVERRIDE=0`
  (diagnostic path `groot/groot-tf32strict-diag.json`, never the receipt path):
  full_abs **1.67e-06** (750x collapse), harness verdict `passed`, rc 0.

Root cause: cuBLAS default math (TF32) in ORT-CUDA vs the fp32-strict CPU
reference — not the export. Corroboration: pi0's A100 full_abs is 1.1e-03 on
the same stack (its 0.1 gate absorbs the identical floor); April's 3.7e-06
matches my CPU number to 2%, i.e. it was never a CUDA-vs-CPU figure. No
exporter-side lever exists (backend math isn't ONNX-expressible; session
construction is the harness's). Options for gate owners: reference-on-CUDA
(predicted ~6.7e-4, thin pass), strict-math env (~1.7e-6, fat pass, must be
recorded), or revisit max_abs. The strict-env file stays a diagnostic; the
receipt path keeps the truthful default-env failure.

## 3. OpenVLA (rank 54) — PASS

Receipt `parity-gpu-receipts:openvla/openvla-export-parity-receipt.json`:
verdict `passed`, `external_acceptance: recorded`, harness rc 0, ORT 1.23.2,
Linux, CUDA fallback-disabled. Metrics: token agreement **7/7**, first/full
cosine **1.0**, first/full max_abs **0.0** (bit-exact). Pins: model
`openvla/openvla-7b @ 47a0ec7fc4ec123775a391911046cf33cf9ed83f`; tether
`6e84101136dcaa6204983956b80418796430504a` (export built from `83e63d1`,
exporter/builder byte-identical at receipt commit). Artifact 30 GB, sha
`2eb6e08e…`. Thresholds unmodified (0.999 / 0.1 / 1.0 agreement).

New `export_openvla_monolithic` (`src/tether/exporters/openvla.py`):
torch.onnx.export of the pinned reference's multimodal forward
`(input_ids, attention_mask, pixel_values) -> lm_logits`, fp32, opset 18,
static shapes (upstream supports batch-1 only; keeps factories concrete so no
dynamic Shape/Concat for ORT to move — the M4 #53 lesson). `export_openvla`
keeps its pinned `NotImplementedError` (existing test asserts it).
First-hand trace blockers fixed exporter-side with scoped runtime patches
(IEEE-exact): `torch.full` 7-arg overload → `ones*fill` (bool normalized, the
tracer has no Tensor-bool mul), Llama forced `"eager"` (no SDPA symbolic).
Fused-vision input handled as reported: channels read off
`use_fused_vision_backbone` (`[1,6,224,224]` for the `[3,3]` split); the
harness gained the same `_pixel_channels` fixture fix (it hard-coded 3
channels, which raises in the reference). Two provenance strings corrected
for honesty (receipt `exporter` label, artifact `source`); thresholds,
fallback flag, receipt schema untouched. Family tests: 137 passed, 13 skipped
(GPU-gated). Note: the argmax-bin gate design is TF32-immune (logit noise
can't flip 7/7 exact tokens here), which is why OpenVLA is bit-exact where
GR00T's continuous gate meets the backend floor.

## Spend (~$2 estimated vs ~$25 cap)

A100-40GB ≈ 80 GPU-min (pi0 export+parity ~40 + reuse 5.6; openvla 10.3 + 3.9;
groot rebuild+parity 3.4, profile ~6, TF32 matrix ~10, strict diag 0.8).
CPU jobs ≈ 2 hr wall (exports, pre-checks, censuses — cents). Watchdogs armed
everywhere, never fired except as designed. Stop-and-report was never needed.

## Follow-ups (gate owners, not blockers)

- GR00T max_abs gate vs ORT-CUDA default math (see §2 options with predicted
  numbers). The export itself needs no further work.
- `tether export` CLI / registry copy still advertise the optimum-cli OpenVLA
  path (cli.py:3543, registry/data.py:151) — stale, left untouched as
  out-of-scope surface; the working path is `export_openvla_monolithic`.
- The pi05→pi0→GR00T pattern is now 3-for-3 on placement: any future
  dynamic-shape factory in an exporter should be broadcast-or-static by
  default (openvla.py documents this).
