# GR00T export parity qualification

Tether Studio M4 #53 requires a reference/shared-input export-parity receipt for the exact GR00T artifact being qualified. The existing GR00T exporter already exposes the correct comparison seam: a monolithic ONNX graph representing one per-denoise-step velocity evaluation. This qualification hardens the evidence around that seam rather than changing GR00T inference.

The qualification harness is:

```bash
python scripts/local_groot_monolithic_parity.py \
  --model-source nvidia/GR00T-N1.6-3B \
  --model-revision <40-hex Hugging Face commit> \
  --embodiment-id 0 \
  --onnx-dir /path/to/groot-monolithic-export \
  --provider CUDAExecutionProvider \
  --receipt /path/to/groot-export-parity-receipt.json
```

The harness mechanically:

1. downloads the exact requested upstream checkpoint revision;
2. builds the GR00T full stack for the exact recorded embodiment ID;
3. creates deterministic noisy actions and a deterministic timestep plus exact position IDs;
4. feeds those same tensors to `GR00TFullStack` and the exported ONNX velocity graph;
5. hashes the complete ONNX artifact directory using Tether's artifact-identity contract;
6. hashes names, dtypes, shapes and bytes for the shared tensors;
7. records the exact Tether Git commit, platform and active ONNX Runtime providers;
8. disables ONNX Runtime CPU execution-provider fallback when CUDA is requested;
9. applies the GR00T parity gate: cosine >= `0.9999` and max absolute error < `1e-3`; and
10. emits a durable JSON receipt plus `TETHER_GROOT_EXPORT_PARITY_JSON=<receipt>`.

GR00T's ONNX artifact is a per-step velocity graph; the denoise loop remains a runtime operation. The receipt records this explicitly together with the embodiment ID so a result for one embodiment cannot silently qualify another.

A passing local or CPU run remains `external_acceptance=not-run`. External acceptance is recorded only for a passing Linux + `CUDAExecutionProvider` run with CPU EP fallback disabled.

The receipt proves numerical parity only for the exact checkpoint revision, embodiment, Tether commit, ONNX artifact, provider and shared input it names. It does not prove task success, training support, device readiness, deployment correctness or physical safety.

## Current acceptance state

The receipt contract and harness are implemented in source. Landing them does not run the A100-class export/parity workload or download GR00T. Until a real Linux-CUDA run produces and retains a passing receipt, GR00T export qualification remains pending.
