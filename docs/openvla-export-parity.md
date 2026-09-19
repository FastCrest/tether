# OpenVLA export parity qualification

Tether Studio M4 #54 requires a reference/shared-input parity receipt for OpenVLA. OpenVLA is deliberately not on Tether's flow-matching spine: `src/tether/exporters/openvla.py` ships as a shim that raises, no ONNX graph can be produced for it yet (see the next section), and the only implemented Tether-owned piece is the tokenized-action decoder in `src/tether/postprocess/openvla.py`. The subject of this receipt is that pair — the exported language-model graph and the decoder that turns its logits into actions.

OpenVLA's action head is `argmax` over the top `n_action_bins` vocabulary tokens, so parity here is partly discrete. One disagreeing action token is a whole-bin jump, which a cosine over seven dimensions can hide. The receipt therefore gates on exact action-token agreement **in addition to** the shared cosine and max-absolute-error thresholds; it does not relax either.

## There is no working export path today — VERIFIED 2026-09-16

The harness below requires a `model.onnx`. **Nothing in this repository, and no
`optimum-cli` invocation, can currently produce one for `openvla-7b.`** This was
checked, not assumed:

| Candidate path | Outcome | Evidence |
|---|---|---|
| `optimum-cli export onnx --model openvla/openvla-7b out/` | Cannot work | `openvla` is not among Optimum's supported ONNX architectures; `TasksManager` dispatches on `config.model_type`, and openvla-7b's `config.json` declares `"model_type": "openvla"` with `auto_map` pointing at remote-code `modeling_prismatic.OpenVLAForActionPrediction`. Optimum's own guide states that a `trust_remote_code` architecture requires a `custom_onnx_configs` dict passed to `main_export()` **in Python**; `optimum-cli export onnx --help` exposes no flag for it. |
| `tether export openvla-7b` | Raises | `src/tether/cli.py:490-492` routes to `exporters/openvla.py:export_openvla`, which raises `NotImplementedError` unconditionally. |
| `tether export … --monolithic` | Raises | `exporters/monolithic.py:export_monolithic` is a switch over `smolvla`/`pi0`/`pi05`/`gr00t` only, and each branch loads a LeRobot policy class and installs family-specific flow-matching monkeypatches. OpenVLA is not a LeRobot policy and has no branch; the dispatcher's `else` raises `ValueError`. |

Closing this needs an OpenVLA-specific exporter — either an `OnnxConfig` for the
fused dual-tower vision backbone plus Llama-2-7B driven through
`optimum.exporters.onnx.main_export(custom_onnx_configs=…)`, or an
`export_openvla_monolithic` wrapper in `exporters/monolithic.py` following the
`export_smolvla_monolithic` shape. Neither exists. Until one does, this document
describes a gate with no artifact to gate, and no Modal runner has been written
for it on purpose: a runner whose first command is known to fail is not evidence.

### What the run would cost, once there is something to run

Sizing, so the number is on record before anyone asks for compute:

- **GPU class: A100-80GB or H100-80GB.** `openvla-7b` is ~7.5B parameters; an
  fp32 ONNX graph is ~30 GB and ONNX Runtime's CUDA execution provider holds all
  of it resident. A 24 GB A10G or L4 — the class every other Tether export runs
  on — cannot hold it. The harness additionally loads the PyTorch reference with
  `torch_dtype=torch.float32` on the host, so the container also needs ~30 GB of
  host RAM at the same time.
- **Wall clock: 1.5-3 h**, dominated by a ~30 GB checkpoint download, an fp32
  `torch.onnx` export that must write external data, and ONNX Runtime session
  initialisation over a 30 GB graph. The two forward passes themselves are
  seconds.
- **Estimated cost: roughly $6-15** for one end-to-end attempt at Modal's
  80GB-class rates. Treat this as an estimate: the only Modal rate verified in
  this repository is the A10G's `$0.000625/s` at `src/tether/eval/cost_model.py:48`,
  and an 80GB part is several times that.
- Any runner written for this **must** carry a wall-clock self-kill budget passed
  into the function body, the way `scripts/modal_libero_lerobot_native.py` takes
  `wall_clock_budget_s`. `Function.with_options` was removed in modal 1.4, so the
  declared `timeout=` is the only outer ceiling and an in-body watchdog is the
  only thing that stops a stalled 80GB container from burning the budget.

Once such an exporter exists, the qualification harness is:

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
3. builds one deterministic `input_ids` / `attention_mask` / `pixel_values` triple from the recorded input seed, reading the `pixel_values` channel count off `use_fused_vision_backbone` — `openvla-7b` is fused (`dinosiglip-vit-so-224px`), so `PrismaticVisionBackbone.forward` runs `torch.split(pixel_values, [3, 3], dim=1)` and the tensor is `[1, 6, 224, 224]`, not `[1, 3, 224, 224]`;
4. refuses to run if the exported graph expects an input the harness does not build, rather than feeding it a different graph than the reference saw;
5. sends those exact tensors to both the PyTorch reference and the ONNX artifact;
6. takes `argmax` over the last `action_dim` positions on both sides and counts exact action-token agreement. These are the next-token argmaxes at the tail of a single prompt-only forward, not tokens produced by `generate`, so the count is a parity probe of one decode run twice, not a check that the model emitted semantically correct actions. That is the right subject for this receipt — argmax over 32064 logits is extremely sensitive to numerical drift — but the receipt should not be read as an action-correctness claim;
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

The contract, decoder and harness are implemented in source. The decoder's bin-center
arithmetic is verified numerically against an independent closed-form oracle over every
token id in the padded vocabulary (`tests/test_openvla_postprocess.py::TestBinCentersAgainstAnIndependentOracle`);
that is the only part of this pipeline with evidence behind it.

The harness itself has never been executed against a checkpoint. It cannot be, because no
exporter produces the `model.onnx` it consumes. OpenVLA export qualification is therefore
**blocked on a missing exporter**, which is a stronger statement than `verification-pending`:
it is not waiting on GPU time, it is waiting on code that has not been written. Studio must
not treat OpenVLA as export-qualified, and `external_acceptance` stays `not-run`.
