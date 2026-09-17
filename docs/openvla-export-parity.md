# OpenVLA export parity qualification

Tether Studio M4 #54 requires a reference/shared-input parity receipt for OpenVLA. OpenVLA is deliberately not on Tether's flow-matching spine: `src/tether/exporters/openvla.py` ships as a shim, the ONNX graph comes from `optimum-cli export onnx`, and the only Tether-owned piece is the tokenized-action decoder in `src/tether/postprocess/openvla.py`. The subject of this receipt is that pair — the exported language-model graph and the decoder that turns its logits into actions.

OpenVLA's action head is `argmax` over the top `n_action_bins` vocabulary tokens, so parity here is partly discrete. One disagreeing action token is a whole-bin jump, which a cosine over seven dimensions can hide. The receipt therefore gates on exact action-token agreement **in addition to** the shared cosine and max-absolute-error thresholds; it does not relax either.

The export itself is the standard HuggingFace path:

```bash
pip install 'optimum[onnxruntime]'
optimum-cli export onnx --model openvla/openvla-7b /path/to/openvla-onnx/
```

The qualification harness is:

```bash
python scripts/local_openvla_monolithic_parity.py \
  --model-source openvla/openvla-7b \
  --model-revision <40-hex Hugging Face commit> \
  --onnx-dir /path/to/openvla-onnx \
  --provider CUDAExecutionProvider \
  --action-dim 7 \
  --trust-remote-code \
  --receipt /path/to/openvla-export-parity-receipt.json
```

The harness mechanically:

1. downloads the exact requested checkpoint revision;
2. reads the decode vocabulary off that config rather than hard-coding it — OpenVLA decodes against `text_config.vocab_size - pad_to_multiple_of` (32064 - 64 = 32000 for `openvla-7b`), not the padded LM size;
3. builds one deterministic `input_ids` / `attention_mask` / `pixel_values` triple from the recorded input seed;
4. refuses to run if the exported graph expects an input the harness does not build, rather than feeding it a different graph than the reference saw;
5. sends those exact tensors to both the PyTorch reference and the ONNX artifact;
6. takes `argmax` over the last `action_dim` positions on both sides and counts exact action-token agreement;
7. decodes both sides through `tether.postprocess.openvla.decode_actions` — bin index `vocab_size - token_id - 1` clipped to `[0, n_action_bins - 2]`, mapped through the centers between `n_action_bins` edges, matching `modeling_prismatic.py`;
8. hashes the complete ONNX artifact directory with the public artifact-identity contract, and hashes the shared input names, dtypes, shapes and bytes;
9. records the exact Tether Git commit, platform and active ONNX Runtime providers, and disables ONNX Runtime CPU execution-provider fallback when CUDA is requested;
10. applies the gate — action-token agreement `1.0`, cosine at least `0.999`, max absolute error below `0.1`; and
11. emits `TETHER_OPENVLA_EXPORT_PARITY_JSON=<receipt>` plus a durable JSON receipt.

`--trust-remote-code` is required and has no default. OpenVLA's reference implementation lives in the checkpoint repository, so loading it executes code from `--model-source` at the pinned revision. That is an explicit opt-in, not a flag the harness sets for you.

`--dataset-name` is optional. Omitted, both sides are compared as normalized bin centers. Supplied, both sides go through the same `norm_stats` q01/q99 affine, including OpenVLA's per-dimension `mask` — the gripper dimension is excluded from unnormalization in the shipped `openvla-7b` stats and stays in bin-center space on both sides.

A passing local CPU/macOS run is useful debugging evidence but remains `external_acceptance=not-run`. External acceptance is recorded only when all of the following are mechanically true in the same run:

- the model revision is an exact 40-hex commit;
- the Tether implementation is an exact 40-hex Git commit;
- `model.onnx` is present in the hashed export identity;
- the requested ONNX Runtime provider is active;
- the platform is Linux;
- the requested provider is `CUDAExecutionProvider`;
- CPU execution-provider fallback is disabled for that CUDA session;
- reference and export output shapes match;
- every action token agrees exactly;
- first-action and full-chunk cosine are at least `0.999`; and
- first-action and full-chunk max absolute error are below `0.1`.

The receipt proves parity only for its exact artifact, reference revision, implementation revision, runtime/provider, decode parameters and shared input. It does not prove task success, training support, device readiness, deployment correctness or physical safety.

## Current acceptance state

The contract, decoder and harness are implemented in source. Landing them does not export `openvla-7b` (~7.5B parameters) or run any GPU workload. Until a real Linux-CUDA run produces and retains a passing receipt, OpenVLA export qualification remains `verification-pending` and Studio must not treat OpenVLA as export-qualified.
