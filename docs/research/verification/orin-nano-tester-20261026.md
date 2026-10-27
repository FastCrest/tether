# Orin Nano tester measurements (2026-10-21 and 2026-10-26)

A tester ran these on their own board. The raw logs are kept privately (Tether Studio,
`docs/research/verification/orin-nano-tester-20261026/`). Only the numbers are reproduced here.

**Board:** Jetson Orin Nano Engineering Reference Developer Kit Super, 8 GB (7.6 GB usable).
BIOS 36.4.3 (JetPack 6, L4T 36.4.3), kernel 5.15.148-tegra. 12.0 GB swap.

**Runtime:** onnxruntime-gpu 1.24.0 from the Jetson AI Lab index, Python 3.10, native venv (no container).

## SigLIP image encoder, TensorRT FP16 (2026-10-26)

The check script loads `image-encoder.onnx` (371,674,073 bytes, FP32) under the TensorRT EP with
`trt_fp16_enable`, then under the CUDA EP in FP32. It uses 24 real crops supplied by the tester.

Reference: the FP32 CPU outputs for the same crops.

| Path | Top-1 vocab labels | Cosine min / mean | Max abs | ms per crop |
|---|---|---|---|---|
| TensorRT FP16 | 24/24 | 0.9999507 / 0.9999716 | 0.0016 | 10.43 |
| CUDA FP32 | 24/24 | 0.9999992 / 0.9999997 | 0.0002 | 22.94 |

VERIFIED (the tester's receipt and log):
- The TensorRT session was active, with TensorRT, CUDA and CPU providers.
- The engine build took 152 s. Peak process RSS was 4.22 GB, and the board peaked at 5.4 of 7.6 GB.
- MemAvailable was 6.12 GB before the build and 2.41 GB after it. Swap stayed about 92% free.

During the build TensorRT logged:
- `NvMapMemAllocInternalTagged ... error 12`;
- a CUDA out-of-memory on a 366,812,544-byte `consts` region;
- `Skipping tactic`.

The engine still built and passed. One tactic was simply not considered.

### Memory estimate

`check_model_load.trt_build_peak_bytes` gives about 4.6 GB for this file, against 4.22 GB measured. The estimate is conservative here.

An older 2.4 GB estimate, without the builder term, was used in the check package's README. It was wrong.

## SmolVLA 1.6 GB FP32, TensorRT engine build (2026-10-21)

The model was served with `--providers TensorrtExecutionProvider,CUDAExecutionProvider` and was killed about ten minutes into the engine build.

The kernel record, from `journalctl -k`, because `dmesg` needs sudo on this board:
- `Out of memory: Killed process ... (python) total-vm:27331984kB, anon-rss:4538032kB`
- `Free swap = 0kB`, `Total swap = 12289972kB`
- `global_oom`

So the build needed at least 4.5 GB resident plus 12.3 GB of swap, about 16.8 GB. The estimate for this file is 14.0 GB, so on this board it under-predicts by at least 20%.

## What this changes

- **Swap is not enough.** On a Jetson the GPU allocates from the same RAM, and those allocations cannot be swapped. Swap helps only the host side of an engine build. The doctor's advice and the smoke SIGKILL hint now say so, and the hint points to `journalctl -k` before `dmesg`.
- **The coefficients are unchanged.** There are two Jetson points with opposite errors, and the FP32 one is only a lower bound. A refit needs a completed SmolVLA build on the board, with its peak RSS, swap and tegrastats.
