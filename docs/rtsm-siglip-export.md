# Export the RTSM SigLIP image encoder

`tether export-rtsm-siglip` exports a standalone, static batch-one image encoder
for RTSM's `ViT-B-16-SigLIP` architecture. It returns a pooled, L2-normalized
768-value embedding. It is separate from the Transformers patch-token backbone
and is not a policy export or a Studio object-to-action adapter.

Use a separate Python 3.11+ export environment with `[siglip-export]`. The measured
cohort is Python 3.12, torch 2.10.0, torchvision 0.25.0, open-clip-torch 3.3.0,
timm 1.0.30, ONNX 1.23.2 and CPU ORT 1.30.0. ORT's Mac wheel needs macOS 14+.
Keep this environment separate from CUDA serving. The Studio serving runtime
does not add open-clip or a new policy adapter through this change.

```bash
# Download the public WebLI checkpoint and use its pretrained transform.
tether export-rtsm-siglip ./rtsm-image-encoder --seed 0 --samples 4

# Local weights require an explicit transform choice.
tether export-rtsm-siglip ./rtsm-local-encoder --checkpoint ./model.pt \
  --preprocess rtsm-local --seed 0 --samples 4
```

The pinned [upstream loader](https://github.com/calabi-inc/rtsm/blob/85b94e603b25e32fe9c3484070807081da07dca0/rtsm/models/clip/loader.py)
has two different preprocessing paths. `pretrained` uses bicubic squash and
mean/std 0.5. `rtsm-local` reproduces its center-crop path and OpenAI CLIP
normalization. They produce different tensors for the same RGB image. Confirm
the deployed branch before selecting one; the filename alone cannot establish it.
Arbitrary local weights retain unverified provenance even when their architecture
matches. The manifest records the actual applied transform separately from
reference WebLI settings and binds the unchanged checkpoint by SHA-256.

The output contains `image-encoder.onnx`, `manifest.json` and generated PNG/input/
PyTorch/ONNX parity files. RGB preprocessing runs outside ONNX; reproduce the
manifest's applied contract before passing FP32 `[1,3,224,224]` tensors. Each
finite, unit-normalized output must meet maximum absolute error 1e-4 and cosine
at least 0.99999 against PyTorch CPU. Invalid parity, changed weights and occupied
output directories refuse publication. Local checkpoint loading uses weights-only
mode.

The [recorded real-checkpoint evidence](research/verification/pre-retest-20261023/README.md)
covers both transform paths. Generated-image numerical parity does not establish
customer-image parity, TensorRT acceptance, target memory/latency or task accuracy.

The public RTSM API already exposes object/search envelopes. Tether's query helpers
preserve those envelopes and their vectors/pose/timestamp metadata; spatial search
now transmits `radius_m`. A policy consumer still needs its trained adapter/input
contract, coordinate frames, selection/padding/masks, freshness/failure behavior
and control authority. Do not feed arbitrary object embeddings into SmolVLA.
