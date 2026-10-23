# Pre-retest corrections, 2026-10-23

The customer needs a Studio desktop that can retain full-policy serving evidence
and stop its managed servers reliably, plus a measured smaller SmolVLA export
before target memory qualification. No robot motion, paid compute or Actions
configuration changes are part of these checks.

## Full SmolVLA numerical evidence

[Artifact identities](full-smolvla-artifacts.json) bind the public checkpoint
`lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`, backbone/tokenizer
revisions, actual files and unchanged monolithic exporter blob. A task-local
Python 3.12.12 environment uses torch 2.10.0, torchvision 0.25.0, transformers
5.5.4, LeRobot 0.6.1, ONNX 1.23.2 and CPU ORT 1.30.0. Existing shared environments
were preserved. Previously retained full-model files were no longer available;
this is a new actual export, not a reused tiny fixture.

[Eight seeded FP32/PyTorch comparisons](full-fp32-pytorch-parity.json) pass frozen
max-absolute 1e-4 / cosine 0.99999 tolerances; observed max error is 6.02007e-6.
Inputs include three `[1,3,512,512]` cameras, varying third-camera masks, token IDs,
state and explicit noise. Outputs are `[1,50,32]`. These are generated numerical
inputs, not task success or physical execution evidence.

The first converter refused a non-SSA graph caused by repeated generated cast
names. Structural repair removes only identical generated unary casts; conflicting
producers/types refuse. The next [measured candidate failed](mixed-fp16-first-numerical-failure.json),
with max errors 0.93–3.58 and cosines 0.32–0.96. No serving directory was published.
Individual FP32 normalization operations were separated by FP16 casts, overflowing
intermediates above 65,504. Rotary phase rounding was a secondary error source.

The [corrected mixed-FP16 receipt](full-mixed-fp16-parity.json) preserves strongly
typed FP32 normalization/phase connections and every original I/O type. Original
semantic casts are not bypassed. All eight original cases pass the unchanged
0.01 / 0.999 thresholds: max error 0.00366658, min cosine 0.999998734. Model bytes
fall from 1,645,562,606 to 833,138,249. The receipt lists 1,535 removed duplicate
casts and 2,006 preserved FP32 connections. The FP32 source remains unchanged.
A diagnostic-only hard link caused one ONNX 1.23 checker refusal; removing only
that diagnostic link restored the source's link count without changing its bytes.
That refused attempt is not numerical evidence.

This is an opt-in existing-export conversion, currently exposed through the
[precision CLI](../../../precision-conversion.md). It does not add a Studio GUI
conversion control or change monolithic `export --precision` support. Studio can
serve a reviewed converted directory through its existing local-export flow.
The supplied base-model cases do not qualify a different trained export. Smaller
storage does not prove TensorRT builder peak memory or Orin fit.

## RTSM and SigLIP

The public upstream contract exists at `calabi-inc/rtsm@85b94e603b25e32fe9c3484070807081da07dca0`.
Its current config specifies SigLIP/WebLI; README's older 512-D CLIP narrative is
stale. Query fixes retain envelopes and metadata, correct inaccurate annotations,
refuse malformed collections and transmit `radius_m` rather than ignored `radius`.

Real `timm/ViT-B-16-SigLIP@41f575766f40e752fdd1383e9565b7f02388c1c4` checkpoint
SHA-256 is `81942ea8c09b9e41963357cca5d2682118bf7eb7491c2ae51a28bc4f4f95b194`.
The image-only ONNX has 371,674,073 bytes and SHA-256
`32a3b971b49791d2abcc3f4eecd6e63c28dea8ab1d7640705417a5cd8e722797`.
Four generated cases pass in each upstream transform mode, with worst absolute
errors 3.716e-7 / 4.917e-7 and minimum cosines above 0.999999999997. Fresh ORT
reload verifies all 17 artifact hashes and repeats outputs bit-identically.
[Reload evidence](siglip-independent-reload.json) and [metadata reconciliation](siglip-manifest-reconciliation.json)
retain their scope. Reconciled [pretrained](siglip-webli-manifest-reconciled.json)
and [local](siglip-rtsm-local-manifest-reconciled.json) manifests identify the actual
transform and local provenance without claiming another inference run. Root's
[actual final CLI manifest](siglip-final-cli-manifest.json) exercises the corrected
publication path with three generated cases and the same checkpoint bytes.

The paths differ materially: same RGB inputs produced preprocessing differences
up to 3.051779. Deployed version/config/checkpoint, paired customer fixtures and
policy adapter mapping remain external inputs. The user confirmed Chifeng, and
the narrow request was observed Delivered in WhatsApp. No retest request was sent.
A snapshot API or embedding encoder is not a policy/control integration.

## Verification scope

Focused results and final source identities are added alongside this report as
integration completes. An earlier 197-case pass predates final numerical-island
and SigLIP corrections and is retained as historical scope, not the final total.
The source-only final selection reports 226 passes and four absent-optional-stack
skips; SigLIP's own isolated export environment supplies its unskipped checks.
The full-project `uv lock` resolution fails on the preexisting aarch64 NumPy<2 /
LeRobot>=2 conflict, reproduced from clean `9832130d` metadata. No global runtime
cohort was changed to hide it. The selected SigLIP extra resolves 57 packages for
Python 3.12 / macOS 14+; GPU and CPU encoder cohorts require separate environments.

GPU/TensorRT, native WSL/5090, Orin build fit, customer images/policy adapter and
physical stop qualification remain unestablished. Existing FAIL certificates are
not revised. Protected Studio installer/customer-flow evidence belongs in the
Studio repository's matching receipt folder.
