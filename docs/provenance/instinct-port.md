# Provenance: the `tether.accel` / `tether.noninferiority` port

Date: 2026-10-28. Branch `feat/accel-port-from-instinct`.
Upstream: General Instinct's **InstinctFlash** (https://github.com/General-Instinct/InstinctFlash).
Pinned commit: `34e1c6089fb6317cdfdd3a4cdd93ad92fbf0ebaa`.

Claim markers: **VERIFIED** (read the code or ran it), **ASSERTED** (a source or person says so;
not checked), **UNKNOWN**.

## Licence basis

| Item | Status | Evidence |
|---|---|---|
| Upstream is Apache-2.0 at the pinned commit | VERIFIED | `LICENSE` at `34e1c60` is the Apache 2.0 text; the commit message is "License InstinctFlash under Apache-2.0 and add a citation" (2026-10-28 20:02 UTC). It changes LICENSE, every package LICENSE, the pyproject `license` field and `release/OSS_RELEASE.rst`. |
| The previous public licence was AGPL-3.0-or-later | VERIFIED | Same commit removes the AGPL text. The earlier teardown read `b49146d`, which is AGPL. **Nothing in this port comes from an AGPL revision.** Every file was copied from `34e1c60`. |
| No root `NOTICE` file to carry forward | VERIFIED | The only NOTICE in the tree is `release/vendor/cosmos/NOTICE`, which is excluded. |
| Ported files are General Instinct's own work | VERIFIED (as far as git can show) | `git log` on every ported path lists one author, `Guanming <guanming@general-instinct.com>`. History starts at `55800cf`, "InstinctWM becomes InstinctFlash; the public repo is frozen as-is", so earlier history is not public. No ported file has a third-party copyright, SPDX tag, or "adapted from" note (grepped). |
| Attention kernel derived from Triton's MIT tutorial | UNKNOWN | It uses the tutorial's online-softmax idioms (`m_i`, `l_i`, `qk_scale = scale*log2e`, `exp2`), but its structure differs: GQA row folding, a device-resident key length, and split-KV. We include the Triton MIT notice as a precaution (THIRD_PARTY_NOTICES §2). The Triton copyright lines are ASSERTED from memory of Triton's LICENSE. |
| Verbal permission from General Instinct to copy with no AGPL obligations, provided the code is renamed | **ASSERTED**: verbal, reported by Romir on 2026-10-27 | No written grant is on file. The Apache-2.0 relicence makes the permission legally unnecessary for this port, because Apache-2.0 alone allows the copy. We renamed anyway, as Romir instructed. The public LICENSE at `b49146d` said AGPL-3.0. A future acquirer's diligence should therefore see (a) that we copied only from `34e1c60` or later, and ideally (b) a written grant that covers anything taken from AGPL-era revisions. Romir has explicitly declined to block on (b). |

Obligations we carry, under Apache-2.0 §4:
- the licence text (`src/tether/accel/LICENSE-Apache-2.0.txt`);
- a "modified by FastCrest" header on every ported file (§4(b));
- the attribution list (`src/tether/accel/THIRD_PARTY_NOTICES`).

§6 grants no trademark rights. That is one more reason the upstream names appear only in
THIRD_PARTY_NOTICES and in this file.

## What was copied

| Upstream (`34e1c60`) | Tether | Changes |
|---|---|---|
| `instinctflash/verify/certify.py` | `src/tether/noninferiority.py` | Header. The `interval`/`margin` validation is factored into `check_declaration()`, which also rejects non-finite or non-numeric margins. Docstring references to upstream files reworded. The statistics are untouched. |
| `instinctflash/engines/common.py` | `src/tether/accel/_common.py` | Import paths, docstring. |
| `instinctflash/kernels/{__init__,_triton,fp8,activation,attention,norm,rope}.py` | `src/tether/accel/kernels/` | Import paths. The package docstring drops kernels we did not port. |
| `instinctflash/engines/pi05/{engine,weights}.py` | `src/tether/accel/pi05/` | Import paths, names. |
| `instinctflash/engines/groot_n17/{engine,weights,frontend}.py` | `src/tether/accel/groot_n17/` | Import paths. `GiGrootActionGenerator` becomes `AccelGrootActionGenerator`. `HORIZON`/`ACT_DIM` move to the package `__init__` so the generator imports without Triton. `from_checkpoint` now calls the device gate. |
| `tests/test_certify.py`, `tests/test_gi_kernels.py` (minus the cuBLASLt case), `tests/test_groot_action_generator.py` | `tests/test_noninferiority_upstream.py`, `tests/test_accel_kernels_gpu.py`, `tests/test_accel_groot_generator.py` | Import paths. Pytest entry point. GPU module skip for devices below sm_89. |

Written new in Tether, so not third-party: `accel/__init__.py` (the `SUPPORTED_DEVICES` gate),
`accel/pi05/inference.py` (`AccelPi05Inference`, Tether's rollout `InferenceProtocol`), the
non-inferiority wiring in `verify.py`, `parity_cert.py` and `parity_report.py`, and the tests
`test_noninferiority_oracle.py`, `test_verify_noninferiority.py`, `test_accel_cpu.py`,
`test_accel_graphs_gpu.py`, plus `scripts/modal_accel_kernels_l4.py`.

## Excluded, and why

| Upstream path | Licence | Reason |
|---|---|---|
| `release/vendor/{cosmos,dreamzero,groot,va,vla2,vla4}/` | NVIDIA proprietary (cosmos); Apache-2.0 (the rest) | Upstream model code with its own copyright holders. Not needed, because the engines read checkpoint safetensors directly. The cosmos vendor licence is not open source. |
| `instinctflash/serving/msgpack_numpy.py` | Apache-2.0 (openpi) / BSD (msgpack-numpy) | Third-party wire codec. Tether has its own transports. |
| `instinctflash/native/*.cu`, `kernels/cublaslt.py`, `kernels/gemm_bf16.py`, `kernels/wan.py` | Apache-2.0 | Used only by WAN/Cosmos engines or by a benchmark. Not on the pi05 or GR00T path. |
| `instinctflash/engines/pi05/loop.py`, `runtime/pi05_engine.py` | Apache-2.0 | A LeRobot-processor loop. Tether's own `InferenceProtocol` adapter (`accel/pi05/inference.py`) replaces it, so the rollout and server keep owning preprocessing. |
| `instinctflash/runtime/{groot_engine,groot_fp8,torch_fp8_linear,static_tensor_graph}.py` | Apache-2.0 | The full N1.7 serving loop, which depends on the upstream `groot_n17_iwm` example adapter, plus FP8 Qwen3 text projections. Tether has no native N1.7 policy loop to put them behind, so porting them now would be dead code. **Trigger to port:** Tether gains an N1.7 runtime. |
| `instinctflash/engines/{vla2,vla4,wan_va}/` | Apache-2.0 | Tether has no model path for these families. |
| Benchmarks, results, `distill/`, `train/`, CLI | Apache-2.0 | Out of scope for a runtime/backend port. |

No CUTLASS- or FlashAttention-*code*-derived file is in the port (VERIFIED by grep; the kernels
are Triton plus `torch._scaled_mm`).

## Hardware scope

The engines target Jetson Thor (sm_110), RTX 4090 (sm_89) and RTX 5090 (sm_120)
(`SUPPORTED_DEVICES`). Anything else raises `UnsupportedAccelDevice`, and there is no silent
fallback. Orin (sm_87) has no FP8 tensor cores and keeps using the TensorRT export path. The
port is **not** wired into Orin export.

Validation status:
- Upstream claims Thor and RTX 4090 qualification, with 5090 "pending re-qualification": ASSERTED, `release/OSS_RELEASE.rst`.
- In Tether: CPU unit tests pass (VERIFIED). The 17 kernel and CUDA-graph tests pass on a Modal L40S (sm_89, the 4090 compute capability), 2026-10-28 (VERIFIED). An L4 could not be scheduled on the workspace that day.
- Not yet measured: end-to-end pi05 action parity of `AccelPi05Inference` against the native policy on real weights, i.e. a `tether verify --non-inferiority-margin` run on a 4090 or Thor.

## Naming and CLI (decided 2026-10-29)

- Module: `tether.accel`. Docs name: "Tether accelerated engines (FP8)", targets Thor, RTX 4090, RTX 5090.
- `tether verify --non-inferiority-margin FLOAT --non-inferiority-interval [wald_central95|tango_one_sided95]`,
  mirroring the `run_verify` keyword arguments. Omitting the margin leaves behaviour unchanged (`NOT_EVALUATED`).

## Still open

- Whether to request the written grant anyway (see Licence basis).
