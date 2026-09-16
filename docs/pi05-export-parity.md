# pi0.5 export parity qualification

Tether Studio M4 #52 requires a reference/shared-input parity receipt for the exact pi0.5 export being qualified. Existing Python-path parity and monolithic-vs-decomposed comparisons are useful diagnostics, but neither substitutes for comparing an exact upstream PyTorch checkpoint directly with the exact hashed ONNX artifact.

The qualification harness is:

```bash
python scripts/local_pi05_monolithic_parity.py \
  --model-source lerobot/pi05_base \
  --model-revision <40-hex Hugging Face commit> \
  --onnx-dir /path/to/pi05-monolithic-export \
  --provider CUDAExecutionProvider \
  --num-steps 10 \
  --receipt /path/to/pi05-export-parity-receipt.json
```

The harness mechanically:

1. downloads the exact requested LeRobot pi0.5 checkpoint revision;
2. creates deterministic model inputs using the recorded input seed;
3. creates deterministic diffusion noise using the recorded noise seed;
4. feeds the same tensors to `PI05Pytorch.sample_actions` and the ONNX artifact;
5. hashes the complete ONNX artifact directory using Tether's artifact-identity contract;
6. hashes names, dtypes, shapes and bytes for the shared inputs;
7. records the exact Tether Git commit, platform and active ONNX Runtime providers;
8. disables ONNX Runtime CPU execution-provider fallback when CUDA is requested;
9. evaluates the versioned parity thresholds; and
10. emits a durable JSON receipt plus `TETHER_PI05_EXPORT_PARITY_JSON=<receipt>`.

A passing local or CPU run remains `external_acceptance=not-run`. External acceptance is recorded only when the exact same run proves Linux + `CUDAExecutionProvider` + disabled CPU EP fallback, matching output shapes, cosine >= `0.999`, and max absolute error < `0.1` for both first action and full action chunk.

The receipt proves only numerical parity for the exact checkpoint, Tether commit, exported artifact, provider and shared input it names. It does not prove LIBERO/task success, training support, physical-device readiness, deployment correctness or safety.

## Current acceptance state

The receipt contract and harness are implemented in source. Landing them does not execute a GPU job or download a model. Until a real Linux-CUDA run produces and retains a passing receipt, Studio must keep pi0.5 export acceptance pending.
