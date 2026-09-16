# pi0 export parity qualification

Tether Studio M4 #51 requires more than a working pi0 exporter. The public prerequisite is a reference/shared-input parity test whose evidence is bound to the exact reference checkpoint, exact Tether implementation and exact exported ONNX bytes.

The qualification harness is:

```bash
python scripts/local_pi0_monolithic_parity.py \
  --model-source lerobot/pi0_base \
  --model-revision <40-hex Hugging Face commit> \
  --onnx-dir /path/to/pi0-monolithic-export \
  --provider CUDAExecutionProvider \
  --num-steps 10 \
  --receipt /path/to/pi0-export-parity-receipt.json
```

The harness mechanically:

1. downloads the exact requested LeRobot checkpoint revision;
2. builds one deterministic preprocessed input using the recorded input seed;
3. builds one deterministic diffusion-noise tensor using the recorded noise seed;
4. sends those exact tensors to both the PyTorch reference and the ONNX artifact;
5. hashes the complete ONNX artifact directory with the public artifact-identity contract;
6. hashes the shared input tensors including names, dtypes, shapes and bytes;
7. records the exact Tether Git commit, platform and active ONNX Runtime providers;
8. evaluates the fixed pi0 parity thresholds; and
9. emits `TETHER_PI0_EXPORT_PARITY_JSON=<receipt>` plus a durable JSON receipt.

A passing local CPU/macOS run is useful debugging evidence but remains `external_acceptance=not-run`. External acceptance is recorded only when all of the following are mechanically true in the same run:

- the model revision is an exact 40-hex commit;
- the Tether implementation is an exact 40-hex Git commit;
- `model.onnx` is present in the hashed export identity;
- the requested ONNX Runtime provider is active;
- the platform is Linux;
- the requested provider is `CUDAExecutionProvider`;
- reference and export output shapes match;
- first-action and full-chunk cosine are at least `0.999`;
- first-action and full-chunk max absolute error are below `0.1`.

The receipt proves parity only for its exact artifact, reference revision, implementation revision, runtime/provider and shared input. It does not prove task success, training support, device readiness, deployment correctness or physical safety.

## Current acceptance state

The contract and harness are implemented in source. No new GPU/provider execution is performed merely by landing them. Until a real Linux-CUDA run produces and retains a passing receipt, the pi0 registry qualification must remain `verification-pending` and Studio must not treat pi0 as export-qualified.
