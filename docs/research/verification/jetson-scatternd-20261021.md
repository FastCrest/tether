# TensorRT declined SmolVLA on JetPack 6.2: ScatterND `reduction="none"` (2026-10-21)

Grades: VERIFIED = read in source or run here; ASSERTED = a source says so, not checked; UNKNOWN = not established.

## Root cause

- VERIFIED. A SmolVLA-base monolithic export made by this repo at `dcbcbb5` (torch 2.10.0, onnxscript 0.7.2, onnx 1.23.2; Studio's export lock `backend/export-requirements-aarch64-apple-darwin.txt`) has 12,514 nodes, 542 of them ScatterND, and **every one carries an explicit `reduction="none"`**. None carries a real reduction (`add`/`mul`/`min`/`max`). They come from slice and index assignment (`res[..., :d_half] = ...` in LeRobot's `smolvlm_with_expert.py:61-62`, SmolVLM's boolean `position_ids[...] = ...`).
- VERIFIED. TensorRT 10.3 (JetPack 6.2) rejects ScatterND whenever the attribute is *present*, whatever its value: onnx-tensorrt `onnxOpImporters.cpp` release/10.3-GA line 5156 `ASSERT_NODE(!attrs.count("reduction"), "Attribute reduction is not supported.", …)`. 10.5-GA (:5243) and 10.7-GA (:5325) are the same; onnx-tensorrt `main` (:6276-6278) accepts `"none"` only.
- VERIFIED. `runtime/ort_providers.py:onnx_has_scatternd_reduction` flags the attribute's presence, so `build_ort_provider_plan` dropped TensorRT for the whole graph with the reason "ScatterND reduction unsupported by TensorRT EP" and served on CUDA. That is the July external-tester observation (Studio `docs/research/verification/jetson-external-tester-20260703.md`, ASSERTED there).

## Fix

`exporters/monolithic.py:_strip_default_scatternd_reduction`, run by `_repair_exported_onnx` (renamed from `_fix_onnx_where_dtype_mismatches`, which every monolithic and decomposed export already calls). It deletes `reduction` only when its value is `"none"` (the ONNX default, so semantics are unchanged), recurses into If/Loop subgraphs and model-local functions, and keeps real reductions, which stay off TensorRT through the preflight.

Rejected alternatives: a TensorRT plugin (nothing to implement: the op is a plain scatter); opset change (ScatterND gained `reduction` in opset 16; staying below 16 is not available because torch's dynamo exporter targets opset 18+, ASSERTED from torch docs, and the other ops need 19); patching torch's exporter (not ours; a post-export pass covers all torch versions); relaxing the preflight to accept `"none"` (wrong for TRT 10.3-10.7, which reject it).

## Numerical evidence (VERIFIED, macOS arm64, ORT 1.30.0 CPU)

Script: `jetson-scatternd-20261021-parity.py`; output: `jetson-scatternd-20261021-parity.json`. Eight seeded input sets, output `[1, 50, 32]`, 12,800 values in total.

| Comparison | mean | p50 | p95 | p99 | max | frac > 1e-6 | frac > 1e-4 |
|---|---|---|---|---|---|---|---|
| stripped vs original ONNX (ORT CPU) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| stripped ONNX vs PyTorch `sample_actions` (CPU) | 2.22e-7 | 1.01e-7 | 9.20e-7 | 2.21e-6 | 5.72e-6 | 0.042 | 0 |
| original ONNX vs PyTorch (CPU) | identical to the row above | | | | | | |

PyTorch output |max| 3.46. `onnx.checker` passes; node list, inputs and outputs are otherwise identical; the preflight goes from True to False; the pass takes 0.2 s.

## Not verified here

TensorRT does not run on macOS. Whether TensorRT 10.3 on an Orin then accepts *every* SmolVLA node, and TRT-vs-CPU error and latency, are UNKNOWN until the Jetson run (`tether-jetson-verify` in Tether Studio's device-agent release). Pi0/pi0.5/GR00T exports pass through the same repair but their ScatterND inventory was not measured.
