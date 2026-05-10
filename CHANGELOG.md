# Changelog

## v0.9.6 — 2026-05-10

**Fix: GR00T N1.6 + OpenVLA registry entries (closes 2026-05-10 customer report).** Plus a contract test that prevents this exact class of bug from recurring.

### Fixed

- **GR00T N1.6 (`nvidia/GR00T-N1.6-3B`) + OpenVLA (`openvla/openvla-7b`) added to `src/reflex/registry/data.py`.** Both exporters had shipped (`gr00t_exporter.py` validated to max_diff=8.34e-07 vs PyTorch reference; `openvla_exporter.py` shipped via optimum-cli onnx path) but the curated model registry only had pi0/pi05/smolvla entries. Customer attempting to use GR00T or OpenVLA via `reflex models list` / `reflex chat` / `reflex doctor` was told "not supported" even though the underlying export pipeline worked. Now both surface correctly + are pullable via `reflex models pull gr00t-n1.6` / `reflex models pull openvla-7b`.

### Added

- **`tests/test_registry_completeness.py`** — contract test that prevents future drift between exporters and the registry. 5 tests:
  - `test_every_primary_exporter_has_registry_entry` — for each primary exporter (gr00t / openvla / pi0 / pi05 / smolvla), assert at least one `ModelEntry` uses the matching family. Catches the GR00T/OpenVLA class of bug at CI time.
  - `test_exporter_directory_audit_covers_all_files` — every file in `src/reflex/exporters/` must be classified as either internal-only (with docstring justification) or primary (with expected family). Catches the case where a contributor adds a new exporter file but forgets registry coverage.
  - `test_registry_entries_have_required_fields` — each `ModelEntry` populates `model_id` / `hf_repo` / `family` / `action_dim` / `size_mb` / `supported_embodiments` / `supported_devices` / `description`.
  - `test_gr00t_n16_in_registry` + `test_openvla_in_registry` — specific assertions that pin the 2026-05-10 fix.

Per CLAUDE.md "real fixes not band-aids" — the registry entries patch the immediate bug; the contract test prevents the next contributor from shipping a new exporter without registry entry. Closes the structural hole, not just the symptom.

## v0.9.5 — 2026-05-07

**CLI cut pass — `reflex --help` shrinks 22% (18 → 14 visible top-level verbs); `reflex inspect --help` shrinks 60% (5 → 2 visible).** No commands deleted; cluttered/redundant ones moved to `hidden=True` in their typer registration. All still callable directly for power-user scripts; just removed from the discovery surface.

### Hidden (still callable)

- **`inspect doctor`** — pure duplicate of top-level `reflex doctor` (cross-registered "for completeness" but added redundant entry to --help)
- **`inspect targets`** — lists hardware profiles. Used once during install, never after
- **`inspect guard`** — dumps shipped safety config. Niche diagnostic
- **`inspect bench`** — internal-only latency microbench (`customer_signal: internal` per spec)
- **`config show/set`** — config schema is a stub (no real config knobs surfaced via this CLI yet; verb-noun ADR scopes config-driven workflows for Phase 2)
- **`bench-game`** — SO-ARM 100 hardware-specific bench rigs (~3 customers globally)
- **`calibrate`** — SO-ARM 100 calibration (corners/surface/tap)
- **`status`** — list running serves; `ps aux | grep reflex` does the same

### Customer-facing impact

Customer's `reflex --help` now shows 11 daily-driver verbs + Pro/contribute. Discord onboarding becomes "here are the 11 verbs you actually need" instead of "good luck navigating 18." Power-user invocations (`reflex inspect doctor`, `reflex calibrate so100 corners`, etc.) still work — only the discovery surface changed.

## v0.9.4 — 2026-05-07

**`reflex doctor` expanded with 4 silent-failure guards.** Customers no longer hit silent failures at deploy time for any of these traps:

### Added

- **Multi-GPU mixed-architecture warning.** If `nvidia-smi` reports 2+ GPUs of different generations (e.g. 1× H100 + 1× RTX 5090), surface a row warning that ORT only uses `CUDA_VISIBLE_DEVICES[0]` and switching GPUs at runtime will silently fail with arch-mismatched kernels.
- **Jetson JetPack target check.** Detects Jetson via `/etc/nv_tegra_release`, parses JetPack version (R35 / R36 / etc.). R35 ships CUDA 11.4 → fails ORT 1.20+'s CUDA 12.x requirement (silent CPU fallback). R36+ passes. Loud message tells JetPack 5.x customers to upgrade or use `[serve,onnx]` CPU extra.
- **CUDA driver vs cuDNN version skew check.** cuDNN 9.5+ requires NVIDIA driver R555+; cuDNN 9.0-9.4 requires R550+. Customer pinning old driver via `apt-hold` + bundled cuDNN 9.5 silently fails at first inference call. Guard reads `nvidia-smi --query-gpu=driver_version` + `importlib.metadata.version('nvidia-cudnn-cu12')` and surfaces the gap.
- **ORT-TRT EP empirical session test.** `available_providers` says the lib loaded — does NOT confirm session-init succeeds. The v0.7 install gap (caught 2026-04-29 v07-install-validation experiment) is exactly this: TRT EP available + session falls back to CUDA EP because `libnvinfer.so.10` isn't on dlopen path. Customer silently loses ~5× perf. Guard creates a tiny stub ONNX model + forces TRT EP + checks `sess.get_providers()` — if TRT EP is missing from active list, surface the loadchain breakage.

### Tests

- 35 new unit tests in `tests/test_doctor_guards.py` covering arch detection (16 GPU SKUs across Blackwell/Hopper/Ada/Ampere/Orin/unknown), Jetson version parsing (R35 / R36 / future / malformed), CUDA driver vs cuDNN version logic (cuDNN 9.0-9.10, drivers R550/R555).
- All 4 guards gracefully skip on non-applicable systems (no NVIDIA: returns silently; no Jetson: returns silently; etc.).

Per CLAUDE.md "no silent fallbacks that paper over errors" — every customer-facing silent-failure mode in our matrix now surfaces a specific row with a concrete remediation command at `reflex doctor` time.

## v0.9.3 — 2026-05-07

**`reflex doctor` Blackwell guard.** Loud check that catches the trap rob (RTX 5090) hit for 2 weeks: customers running Blackwell hardware on `onnxruntime-gpu < 1.25.1` will see an explicit failure row in `reflex doctor` output telling them to upgrade ORT. Previously the customer had to discover this via segfault.

### Added

- **Blackwell sm_120 support row in `reflex doctor`**:
  - **Fires only on Blackwell hardware** (RTX 50-series, RTX PRO Blackwell, B200, GB200) — uses existing `reflex.runtime.server._gpu_is_blackwell()` so no new detection logic. Silent on non-Blackwell.
  - **❌ FAIL** when ORT < 1.25.1: explicit upgrade message with exact pip command (`pip install -U 'onnxruntime-gpu>=1.25.1'`) + reference to PR #27278 + reason ("predates Blackwell support; will SEGFAULT at session-init").
  - **✅ PASS** when ORT >= 1.25.1: confirms support active + flags live caveat (open ORT issue #27621 about silent threading deadlock on sm_120 with PTX JIT + GIL; reflex's single-thread inference doesn't trigger but multi-threaded customers should monitor).
- 7 unit tests in `tests/test_doctor_blackwell_guard.py` covering version-comparison logic (pre-1.25.0, 1.25.0, 1.25.1, post-1.25.1, dev/rc/post pre-release suffixes) + GPU-name pattern matching (real Blackwell SKUs match; Hopper/Ada/Ampere/Jetson don't).

Per CLAUDE.md "no silent fallbacks that paper over errors" — Blackwell users now see the upgrade path immediately at `reflex doctor` time, not 2 weeks later via debugging a segfault.

## v0.9.2 — 2026-05-07

**Blackwell (RTX 5090 / B200 / GB200) support unblocked via ORT 1.25.1 bump.**

### Fixed

- **`onnxruntime-gpu` pin bumped from `>=1.20,<1.24` to `>=1.25.1`** in both `[gpu]` and `[gpu-min]` extras. ORT 1.25.0 (2026-04-20) shipped Blackwell sm_120 kernels via PR [#27278](https://github.com/microsoft/onnxruntime/pull/27278) (CUDA arch family codes `100f`/`110f`/`120f`); 1.25.1 (2026-04-27) is current stable. Earlier 1.23/1.24 regressed sm_120 (`cudaErrorNoKernelImageForDevice` on Blackwell). The ceiling `<1.24` was blocking customers from picking up the working release.
- Bumped `nvidia-cudnn-cu12>=9.5` and `nvidia-cublas-cu12>=12.6` floors to match ORT 1.25's documented build requirements (CUDA SDK 12.8+, cuDNN 9.5+ per maintainer guidance in #26181).

### Live caveat

Open ORT issue [#27621](https://github.com/microsoft/onnxruntime/issues/27621) tracks a silent threading deadlock on sm_120 (PTX JIT + GIL interaction). Reflex doesn't yet exercise the multi-threaded `InferenceSession.run()` codepath that triggers it (single-server, single-request through PolicyRuntime). Will smoke-validate on RTX 5090 hardware before declaring Blackwell tier "production-ready" in the README. Customers using `--max-batch >1` or running multiple `reflex serve` processes on a single GPU should monitor.

### Context — why this took 2 weeks

ADR `2026-04-29-ort-trt-ep-first-class-support.md` (written 9 days AFTER ORT 1.25.0 dropped) assumed Blackwell support would arrive in some future ORT 1.21+. We didn't track ORT releases. Tester `not rob` (RTX 5090) hit segfaults from 2026-04-28 → 2026-05-07 because reflex was pinning `onnxruntime-gpu<1.24`. The fix: drop the ceiling, bump the floor.

## v0.9.1 — 2026-05-07

**Fix: stale `__version__` constant.** v0.9.0's `pyproject.toml` was bumped to 0.9.0 but `src/reflex/__init__.py` still hardcoded `__version__ = "0.8.0"`. The wheel metadata (`pip show`) correctly reported 0.9.0 — only the human-readable upgrade-check nag + `reflex --version` print misreported. No functional bug; all v0.9.0 features (action-similarity-fast-path, customer-trace-archive, uncertainty scoring) shipped working code.

## v0.9.0 — 2026-05-07

**Three Phase 1.5 perf-compound + observability features ship.** Action-similarity fast-path closes redundant expert calls (FlashVLA), customer-trace-archive lands query/summary CLI on recorded /act traces, data-labeling-pipeline gets uncertainty scoring (the orthogonal third axis to success + quality).

### Added

- **`reflex serve --action-similarity-threshold <L2>` + `--max-similar-skips <N>`** (FlashVLA, arxiv 2505.21200). When the expert produces an action chunk L2-similar to the previously-emitted one, the next `predict_action_chunk()` skips the expert and reuses the cached chunk. Default OFF (`0.0` = disabled); paper default `0.05`. Capped at `--max-similar-skips 3` consecutive cached returns to bound drift on slow-changing scenes. Wired only on Pi05DecomposedServer (decomposed pi0.5); legacy + monolithic exports ignore the flags silently.
  - **Production validated 2026-05-07** on Modal A100 with real pi0.5 decomposed export: 9 skips / 20 calls = 45% skip rate, 20/20 bit-exact actions vs disabled mode, **1.24× wall-clock speedup** (2.7s → 2.2s with deterministic identical inputs).
  - Surfaces skip events via new `reflex_action_skip_total` Prometheus counter at `/metrics`.
  - 19 unit tests + 8 mock-integration tests covering enabled/disabled, threshold gating, max_skips cap, cached_actions returns copy, episode reset, stats accounting.

- **`reflex traces query` + `reflex traces summary` subcommands** (customer-trace-archive v1). Filter and aggregate JSONL traces written by `reflex serve --record <dir>`.
  - `reflex traces query --task X --status failed --since 7d --output failures.json` — filter by time window / task substring / `success`/`failed`/`any` status / `model_hash` substring + export to JSON or CSV.
  - `reflex traces summary --by {task,model,day} --since 7d` — aggregate count + success_rate + latency p50/p95/p99/max per bucket. Output rich table (default), JSON or CSV via `--output FILE` + auto-detected from suffix.
  - Built directly on existing JSONL storage; parquet+DuckDB index migration deferred to v2.
  - 20 unit tests + manual end-to-end smoke validated.

- **`uncertainty_score()` + `classify_episode_value()` in `reflex.curate.quality`** (data-labeling-pipeline subsystem 3). N-pass inference variance for flow-matching VLAs (pi0/pi0.5/SmolVLA). Per-dim variance across N samples → normalize by per-dim observed range² → mean across dims → mean across steps → score in [0, 1]. Components dict surfaces `argmax_step` + `argmax_dim` for debugging.
  - 4-quadrant classifier per the spec's training-value framing: high-uncertainty + success → `informative_edge_case` (highest value); high-uncertainty + failure → `edge_case_to_correct`; low + success → `redundant_known_good`; low + failure → `model_blind_spot`.
  - Pure-numpy. Sample-generation is the caller's responsibility (decoupled from inference). Wiring into a `reflex traces uncertainty` CLI deferred to v2.
  - 17 unit tests covering identical-samples → 0, maximally-divergent → ~0.33, constant-dim contributes 0, monotone in variance, argmax pointers correct, parquet-ready `to_dict()`, all 4 classifier quadrants.

### Fixed

- **`ActionFastPath.observe()` stats counter** now increments `expert_calls` regardless of `enabled` state. Previously the counter early-returned at `if not self._enabled: return`, so disabled-mode runs reported `expert_calls=0` even though the expert ran on each call. Caught by 2026-05-07 production smoke. New regression test.

### Phase 1 status

**Phase 1 closed 2026-05-07** with 16/16 features shipped or explicitly killed. a2c2-correction marked `phase_1_shipped` (Phase 1 fix at OFF parity validated 2026-04-29; Phase 2 ON > OFF positive delta filed as 3 successor stubs: FASTER, TAS, Legato — research revisit committed). Phase 1.5 perf-compound queue done — language-layer-pruning + cross-request-pipelining deferred to Phase 2 with explicit re-open triggers.

## v0.8.0 — 2026-05-02

**Per-step expert ONNX export feature ships.** New `per_step_expert=True` flag on `export_pi05_decomposed` produces an `expert_denoise.onnx` that takes `(x_t, t, past_kv)` and returns `v_t` (single Euler step velocity), instead of the default baked-loop ONNX that takes `noise` and returns the fully-denoised action chunk. The per-step shape unblocks RTC + per-step caching (Dexmal 3-stage) by exposing the denoise loop in Python rather than baking it into the ONNX graph.

This release closes gates 1-5 of the per-step ship sequence (per `features/03_export/per-step-expert-export.md`). Gate 6 LIBERO RTC regression closes as PARTIAL — see CAVEATS below.

### Added

- **`per_step_expert: bool = False` flag on `export_pi05_decomposed`.** When True, builds the expert ONNX as a single-Euler-step graph (input `x_t, t, past_kv`, output `v_t`) instead of the default baked-loop. The wrapper class `Pi05ExpertPerStepWrapper` handles dtype coercion, DynamicCache reconstruction, and SnapFlow / state_out variants.
- **Runtime per-step Euler loop in `Pi05DecomposedInference._run_expert`.** When `reflex_config.json` has `decomposed.per_step_expert=true`, the runtime detects this at load and drives the Euler integration in Python: `for step in range(num_steps): v_t = sess.run(["v_t"], feed); x_t += dt * v_t`. Uses ORT IOBinding to pin past_kvs to device once per chunk (vs naive re-copy per iter — see v0.7.3 CHANGELOG entry for the IOBinding fix).
- **Gate 1+2 unit tests:** `tests/test_decomposed_per_step_smoke.py` (11 wrapper-logic tests) + `tests/test_rtc_adapter_per_step.py` (6 config-time guard tests, rejects `--rtc` with `num_steps=1`).
- **Gate 3+4+5 receipt-checker tests:** `tests/test_decomposed_per_step_parity.py` (cos≥0.99999 / max_abs≤1e-5 across pi05_teacher × num_steps={1, 10}), `tests/test_decomposed_per_step_overhead.py` (IOBinding ≤+20% median, ≤1.30x p99), `tests/test_decomposed_per_step_e2e_latency.py` (E2E pipeline overhead bounds + vlm phase symmetry check).
- 32 per-step gate tests pass locally.

### Validated

- **Gate 3 parity (Modal A100-80GB)**: pi05_teacher × num_steps={1, 10} both produce **cos=1.0000000000, max_abs=0.000e+00** for baked-loop ONNX vs per-step ONNX driven 10×/1× in Python loop. Bit-exact.
- **Gate 4 overhead (Modal A100-80GB)**: per-step expert ORT call overhead via IOBinding measured at **+13.3% median / +1.17x p99** vs baked single call. Naive (no IOBinding) was +36.4% / 1.46x — IOBinding is load-bearing for this gate.
- **Gate 5 E2E latency (Modal A100-80GB)**: full `Pi05DecomposedInference.predict_action_chunk` pipeline measured at **+7.7% median / +1.08x p99** vs baked when `cudnn_conv_algo_search=HEURISTIC` is pinned (default ORT EXHAUSTIVE picks bimodal cuDNN algos that distort the comparison).

### Caveats — `--rtc` integration with per-step is EXPERIMENTAL

`--rtc` (Real-Time Chunking) on top of per-step expert ONNX is functional but produces **mixed results at N=20 LIBERO inject-latency-100**:

| libero_10 task | baseline (baked, no RTC) | treatment (per-step + --rtc) | delta |
|---|---|---|---|
| 0 (alphabet soup + tomato sauce) | 18/19 (95%) | 15/19 (79%) | **-16pp** |
| 1 (cream cheese + butter) | 17/20 (85%) | 16/20 (80%) | -5pp |
| 2 (turn on stove + moka pot) | 14/19 (74%) | 10/17 (59%) | **-15pp** |
| 3 (black bowl in cabinet drawer) | 11/20 (55%) | 9/18 (~50%) | ~0 |
| 4 (book on shelf) | 14/20 (70%) | 15/20 (75%) | **+5pp** |
| **aggregate** | **74/98 (75.5%)** | **65/94 (69.1%)** | **-6.4pp** |

The regression is **localized to RTC's interaction with specific task structures** — per-step expert ONNX itself is bit-exact-equivalent to baked (gate 3 cos=1.0), so the success-rate difference cannot come from per-step. It comes from RTC's prefix-attention guidance influencing the policy output in ways that help on some tasks (task 4) but hurt on others (tasks 0+2).

**Recommendation:** treat `--rtc` as opt-in, measure on YOUR task distribution before relying on it for production. The default `reflex serve` does not enable RTC. Customers who explicitly want RTC's smooth-chunk-transition behavior can pass `--rtc`, with the understanding that the success-rate tradeoff varies by task.

Open follow-up investigation: ablation of WHY RTC helps task 4 but hurts tasks 0+2. Tracked in ADR follow-up note in `2026-04-24-defer-rtc-per-step-to-phase-1.md`.

### Notes

- Per-step expert export is OFF by default. Existing customers who use `reflex export` get the baked-loop ONNX they always got.
- The new IOBinding runtime (in v0.7.3) only kicks in for per-step ONNX. Baked-loop runtime is unchanged.
- Modal cost across the per-step ship gates: ~$28 (gates 3+4+5) + ~$94 (gate 6 with retries across multiple budget-capped Modal workspaces) = **~$122 total** vs $15-22 spec budget. Overrun captured 5 production-relevant bug fixes (shipped in v0.7.3) + one shipped feature (this release).

## v0.7.3 — 2026-04-30

Five production-relevant fixes surfaced during per-step expert ONNX export validation work. Each was caught by a parity / latency gate, root-caused, fixed at source. None require user action; benefits show up automatically on `pip install reflex-vla --upgrade` + re-export of any decomposed pi0.5 ONNX.

### Fixed

- **pi0.5 decomposed export precision (cache mutation in trace).** `apply_export_patches` installs a `DynamicLayer.update` freeze patch that prevents the cache from growing across unrolled Euler iterations during torch.export tracing. The flag-toggling wrapper around `denoise_step` was applied to `PI0Pytorch` only — `PI05Pytorch` fell through to the original mutating update, so pi0.5 decomposed exports traced cache growth 968 → 1418 across n=10 with stale-suffix K/V in attention. Promoted `_denoise_phase` flag to module level + wrapped `PI05Pytorch.denoise_step` symmetrically. (Caught by per-step parity gate at cos=0.018; commit `60c30b9`.)

- **Expert ONNX precision (constant-folding sin/cos in fp32).** `torch.onnx.export(dynamo=True, optimize=True)` (the default) constant-folds the time embedding's float64 sin/cos in FP32-precision arithmetic, producing a ~3e-5 max_abs deviation from a true float64 compute. Set `optimize=False` on the expert ONNX export call. Marginal node-count increase, immaterial runtime cost (sin/cos << matmul). (Caught while chasing the residual after the freeze-flag fix; minimal local repro confirmed; commit `d86dca1`.)

- **Reflex CUDA loadchain (libcurand / libcufft / libcusparse).** ORT's `libonnxruntime_providers_cuda.so` links against `libcurand`/`libcufft`/`libcusparse`/`libnvJitLink`. Reflex's `_eager_dlopen_nvidia_libs` only handled cudart/cublas/cudnn + TensorRT — the others happened to be findable on most images via standard `/usr/local/cuda` paths or torch's transitive layout. On Modal images that pinned dependencies tighter (no torch transitively pulling curand), ORT silently fell back to CPU EP and crashed on float64 Cos at the first per-step call. Added the four libs to `_candidate_lib_dirs` + eager-dlopen targets in `src/reflex/__init__.py`. Independently load-bearing for any reflex consumer running on a fresh image. (Caught in gate-4 overhead bench; commit `462c191`.)

- **`--rtc` actually works on decomposed servers.** Pre-fix `--rtc` constructed `RtcAdapter` with `action_buffer=None` whenever the user hadn't also passed `--replan-hz`/`--execute-hz`. `merge_and_update` then raised `'NoneType' object has no attribute peek_all'` on every `/act` call, logged a warning, and silently no-op'd RTC's carry-forward (the inertia mechanism that's the entire point of RTC). Worse: `Pi05DecomposedServer` doesn't have `configure_replan` (only `MonolithicServer` does), so an initial fix that auto-called `configure_replan` only worked for monolithic. Real fix: build the `ActionChunkBuffer` directly when `--rtc` is set and no buffer exists, regardless of server class. Stash on `server._action_buffer`. Decomposed pi0.5 + RTC now actually applies prefix-attention guidance per call. (Caught in gate-6 LIBERO smoke; commits `b863915` then `c059b14`.)

- **Per-step expert ONNX runtime IOBinding.** When `Pi05DecomposedInference` is loaded with a per-step expert ONNX (the new export shape from v0.8.0 per-step expert work), the Python Euler loop calls the expert ONNX N times per chunk. The naive `sess.run(feed_dict)` path forces ORT to re-copy past_kvs (~140 MB across 38 tensors) host→device on every iter — measured at +36% chunk overhead vs baked. ORT IOBinding pins past_kvs + prefix_pad_masks to device once per chunk via `OrtValue.ortvalue_from_numpy(arr, "cuda", 0)`; only x_t (~6 KB) and t (4 B) cross host→device per iter. Drops overhead to +13% (passes the spec gate). Production runtime updated in `Pi05DecomposedInference._run_expert`. (Caught in gate-4 A/B bench; commit `e4b7ca5`.)

### Tests

- Two new receipt-checker tests for the per-step gate sweep:
  - `tests/test_decomposed_per_step_overhead.py` — asserts IOBinding overhead ≤ +20% / ≤ 1.30x p99, IOBinding strictly faster than naive, CUDA EP active.
  - `tests/test_decomposed_per_step_e2e_latency.py` — asserts E2E overhead ≤ +20% / ≤ 1.30x, vlm phase wall-time matches across baked vs per-step within 10% (catches if `cudnn_conv_algo_search=HEURISTIC` wasn't pinned).
- 32/32 per-step gate tests pass locally.

### Notes

- **Existing shipped pi0.5 ONNX files behave the same.** Both the freeze-flag and the optimize=False fixes only affect FUTURE re-exports. Customers who don't re-export keep their current behavior. Re-export gets tighter precision (~10x improvement on baked-loop time-emb).
- **The freeze-flag fix removes "stale suffix K/V" from attention** in the baked decomposed pi0.5 trace. This is semantically the correct behavior (matches PyTorch eager without the cache-mutation artifact). The previous behavior worked at production because the existing `cuda_runtime_parity` gate uses cos≥0.999 (looser than the 0.99999 the per-step parity gate caught it with).
- **All 5 fixes were caught + diagnosed in a single session** by the per-step expert ONNX export ship gates (1-6). Total Modal cost across the diagnostic work: ~$28 (vs original $15-22 per-step ship budget). Each overrun was driven by a real bug discovery — not waste.
- **Per-step expert export feature itself ships in v0.8.0** pending the gate-6 LIBERO RTC regression sweep landing PASS. v0.7.3 covers only the incidental fixes that benefit existing users.

## v0.7.2 — 2026-04-29

A2C2 correction head Phase 1 fix — eliminates the magnitude-7 catastrophic regression that the 2026-04-26 N=50 LIBERO experiment surfaced. Closes the path to A2C2 production deploy with no risk of policy derailment.

### Background

The A2C2 head shipped earlier as `partial` after a 2026-04-26 N=50 LIBERO experiment showed catastrophic regression: ON tasks 0+1 = 0/10 vs OFF baseline 8/10. Root cause analysis (4 parallel research lenses, 2026-04-29) converged on two architectural issues: unbounded output layer + MSE loss with no magnitude regularization let the head learn magnitude-7 corrections that systematically saturated action clip bounds and inverted policy outputs.

### Fixed
- **Bounded head output.** `kernels/a2c2_correction.py` output layer applies `tanh(z/3.0) * 3.0` saturation. Output is now bounded to ±3.0 in normalized action space (matching pi0.5's typical 3σ action range), preserves the zero-init cold-start invariant (`tanh(0)=0`).
- **Huber loss + L2 magnitude penalty.** `correction/a2c2_training.py` switched MSE → Huber(δ=0.1) + λ=0.01 magnitude penalty. Caps gradient on outliers (no more chasing tail-of-distribution magnitudes), discourages large outputs at training time. Validation loss uses the same formula for direct comparability. Backward pass updated to backprop through the tanh saturation.
- **Runtime safety net.** `runtime/a2c2_hook.py` now refuses corrections with chunk magnitude > `sqrt(chunk_size) * 3.0` (theoretical max from the bounded per-step output). Falls back to base actions + emits `reason='magnitude_safety_skip'` metric. Catches future regressions even if a bypass head sneaks in.

### Validated on Modal A100 (commit 9f8edb5)
- Local sanity: 1000 random extreme inputs, max output L-inf = 0.365 << saturation 3.0. Saturation works as designed.
- N=10 LIBERO smoke at `--inject-latency-ms 100` (paper-fidelity setup, same as the catastrophic experiment):
  - **Phase 1 ON: 8/10 (80%)** — matches OFF baseline 8/10 exactly
  - Catastrophic ON (2026-04-26): 0/10 — eliminated
  - Per-call chunk magnitude: 0.755 (consistent across all calls) vs catastrophic ~7.0 = **9× smaller**
- Full experiment writeup: `reflex_context/03_experiments/2026-04-29-a2c2-phase1-libero-smoke-modal.md`

### Tests
- 2 new tests in `tests/test_a2c2_correction_head.py`:
  - `test_forward_output_bounded_to_saturation_scale` (20 random extreme inputs, asserts L-inf ≤ 3.0)
  - `test_forward_zero_init_still_emits_zero_correction` (cold-start invariant preserved)
- 67/67 a2c2 tests pass; 38/38 head tests pass

### Notes
- **The fix is "no regression" not "perf claim".** Phase 1 head produces bounded corrections that are non-destructive but conservative. Customer-facing claim "A2C2 helps Jetson reactivity" is NOT yet supported by this release — pending Phase 2 retrain (multi-latency training data + relaxed L2 penalty) to produce ON > OFF positive delta.
- **`--a2c2-checkpoint` is still opt-in.** No default-on. Customers who previously avoided A2C2 due to the catastrophic finding can now safely enable it without policy derailment risk.
- **Existing checkpoints work.** The bounded output is applied at inference time; previously-trained heads (including the magnitude-7 catastrophic one) load fine and just have their outputs clamped at ±3.0 — preventing the catastrophe from re-occurring even with bad weights.
- Research synthesis at `reflex_context/features/01_serve/subfeatures/_rtc_a2c2/a2c2-correction/a2c2-correction_research_revisit.md`. Modal cost across the Phase 1 work: $3.

## v0.7.1 — 2026-04-29

Surfaces the previously-silent `--cuda-graphs` flag with measured A100 + A10G perf numbers. The flag has been in main since v0.5.0 (commits `49110d2` → `7dfcab6`, April 24-25) but no CHANGELOG entry told users it existed. v0.7.1 fixes that.

### Documented (no new code, surfacing existing capability)
- **`reflex serve --cuda-graphs`** — opt-in CUDA graph capture for the decomposed pi0.5 path (`vlm_prefix.onnx` + `expert_denoise.onnx`). Captures both ONNX sessions at startup using ORT's `enable_cuda_graph=1` + replays for every subsequent request. Skips per-op kernel-launch overhead.
- **Measured speedup on A100-80GB (pi0.5 num_steps=10, n=200 paired iterations, 5 warmup discarded):**
  - Per-chunk latency: 270.85 ms → 207.74 ms = **1.30× speedup, p99 304 → 212 ms (-30%)**
  - `vlm_prefix`: 93.45 → 87.34 ms (1.07× mean, jitter 1.4× tighter)
  - `expert_denoise` (runs 10× per chunk): 17.74 → 12.04 ms (**1.47× mean, p99 -40%, jitter 4.1× tighter**)
- **Measured speedup on A10G (n=100 paired):**
  - `expert_denoise`: 12.34 → 10.47 ms (**1.18× mean, p99 -35%, jitter 14× tighter**)
  - `vlm_prefix`: capacity-bounded — capture buffer doesn't fit alongside working memory on A10G's 24 GB envelope. Wrapper gracefully degrades to eager per ADR `2026-04-24-cuda-graphs-architecture` decision #5 (tier-aware semantics). Same numerics, no perf change for that session, expert still wins.
- **`reflex doctor` cuda-graphs diagnostic** — `tests/test_check_cuda_graphs.py` (already in v0.5.0+) checks `vlm_prefix` + `expert_denoise` capture status. Surface with `reflex doctor` after running `reflex serve --cuda-graphs` to verify capture on your hardware.
- **Customer-facing docs** at `docs/cuda_graphs.md` (already in v0.5.0+) — when to enable, hardware tier matrix, what metrics to watch (`reflex_cuda_graph_captured_total` / `reflex_cuda_graph_replayed_total` / `reflex_cuda_graph_eager_fallback_total` / `reflex_cuda_graph_capture_failed_at_init_total` / `reflex_cuda_graph_capture_seconds`).

### Added (this release)
- **`tests/test_cuda_graphs_integration.py`** (Day 7 of cuda-graphs plan) — 7 mock-based + 1 CUDA-gated integration tests covering create_app flag propagation, legacy ReflexServer no-op log, Pi05DecomposedInference provider wiring, capture-failed-at-init eager fallback, /metrics endpoint counter exposure, and label cardinality bound. The CUDA-gated test runs on Modal A10G/A100; mock-based tests run anywhere.
- **`scripts/modal_cuda_graphs_ab.py`** — Day 8-9 A/B benchmark script. Loads a decomposed export from the `pi0-onnx-outputs` Modal volume, runs N iterations OFF (eager) + N iterations ON (cuda-graphs) for each session (`vlm_prefix` + `expert_denoise`), computes ISB-1 stats with 5 warmup discarded + 95% CI on means, writes JSON output to volume. 30s progress logging so degenerate runs are catchable mid-flight.

### Notes
- **The `--cuda-graphs` flag was never deprecated, hidden, or feature-flagged.** It was always live in `reflex serve --help` since v0.5.0. v0.7.1 simply documents it with measured numbers.
- **Hardware tier guidance:**
  - A100 / H100: capture both sessions (1.30× per-chunk)
  - A10G: capture expert_denoise only (1.18× expert mean, vlm_prefix gracefully degrades to eager)
  - Orin Nano / smaller: not yet measured. Try with `reflex doctor` to check if your GPU has headroom.
- **Why the win shape is "predictability > raw speed":** robotics control loops care about p99 + jitter more than mean. CUDA graph replay eliminates per-op kernel-launch jitter; the means are smaller wins because ORT already uses fused cuBLAS kernels for the heavy ops.
- **Validation experiments:** Modal app URLs in the Reflex context vault at `reflex_context/03_experiments/2026-04-29-cuda-graphs-ab-modal-a10g.md` + `2026-04-29-cuda-graphs-ab-modal-a100.md`. Total Modal cost: $3.50.

## v0.7.0 — 2026-04-29

ORT-TensorRT execution provider first-class support. Most users now get the **5.55× speedup** measured on Modal A10G (108.11 ms → 19.49 ms on SmolVLA monolithic, 2026-04-29) automatically — without manual `LD_LIBRARY_PATH` setup.

### Added
- **`tensorrt>=10.0,<11`** is now in the default `[gpu]` extras. Provides `libnvinfer.so.10` so ORT-TRT EP loads at runtime instead of silently falling back to ORT-CUDA EP. Adds ~2 GB to install footprint — that's the cost of the 5.55× win. Linux only (the `; sys_platform == 'linux'` marker keeps Mac installs untouched).
- **`[gpu-min]` extras** as escape hatch for users who don't want the `tensorrt` install — gets you ORT-CUDA EP only (~5× slower on transformer workloads). Use when bandwidth/storage matter more than perf.
- **Auto-`LD_LIBRARY_PATH` patch in `reflex/__init__.py`.** Prepends pip-installed `tensorrt_libs/`, `nvidia/cudnn/lib/`, `nvidia/cublas/lib/` paths so ORT can find the shared objects without manual env config. Idempotent. Opt-out via `REFLEX_NO_LD_LIBRARY_PATH_PATCH=1`. No-op on macOS, Windows, or when paths don't exist.
- **`reflex doctor` validates the full ORT-TRT EP load chain.** Four new checks:
  1. `libnvinfer.so.10` loadable via `ctypes.CDLL`
  2. `libcublas.so.12` loadable
  3. `libcudnn.so.9` loadable
  4. Empty `ort.InferenceSession` with `TensorrtExecutionProvider` succeeds AND active providers includes TRT EP (gold-standard end-to-end check)
  Each failure has a `pip install` remediation hint inline.
- **README "Performance" section** documents the measured 5.55× claim with the Modal hardware (A10G), the workload (SmolVLA monolithic), the method (5+20 forward passes), the date, and the reproducer command. Plus a "How to verify" section pointing at `reflex doctor`.
- **14 new unit tests** (8 LD_LIBRARY_PATH patch + 6 doctor TRT EP) covering Linux/macOS/Windows + opt-out + idempotency + lib loadable / not / fallback / session-create-throws branches.

### Changed
- `[gpu]` extras now Linux-marked for `tensorrt`. Mac users on `[gpu]` get cuDNN + cuBLAS pulls but no `tensorrt` (which has no Mac wheel anyway).
- `__version__` bumped to `0.7.0` in `src/reflex/__init__.py`.

### Notes
- **No Blackwell support changes.** v0.5.5 documents Blackwell as not yet supported (ORT bundled cuBLAS/cuDNN lacks sm_120 kernels). v0.7's TRT EP work doesn't change this — the upstream gap is at the kernel level, not the install level.
- **The 5.55× was measured on SmolVLA monolithic + A10G only.** Other model architectures (pi0.5 decomposed, GR00T DiT) and other hardware tiers (Orin Nano, T4, L4, H100) may show different ratios. Broader matrix tracked for v0.7.x.
- **The original v0.7 plan was different.** ADR `2026-04-28-tensorrt-llm-runtime-path.md` proposed adding a `--runtime tensorrt-llm` flag with a new RuntimeSession abstraction (~3-4 weeks). Three independent research subagents (Lenses 1+2+3 of `tensorrt-llm-runtime_research.md`) plus a Modal A10G spike invalidated that direction: TRT-LLM is LLM-shaped (not VLA-shaped), doesn't actually unblock Blackwell, and the perf win it was trying to deliver was already in v0.6.0 via ORT-TRT EP — users just weren't getting it because their installs were missing libs. The real v0.7 fix turned out to be 5 days of install/docs/doctor hardening, not 3-4 weeks of new architecture. Full reasoning in the ADRs at `reflex_context/01_decisions/2026-04-{28,29}-*.md`.

## v0.6.0 — 2026-04-29

Hardware-aware decomposed export + C-level crash visibility. Two real-fix surfaces from the v0.5.x first-tester debug session.

### Added
- **`--export-mode {auto, parallel, sequential}` flag** on `reflex export` for pi0.5 decomposed exports. `auto` (default) probes free GPU VRAM and picks parallel iff `2 × estimated_model_vram + 1 GB buffer < free_vram`. `parallel` forces parallel and **fails loudly with `InsufficientVRAMError` before any model load** if the GPU can't fit (no silent fallback). `sequential` always works (the safe baseline). Per CLAUDE.md "no silent degradation."
- **Multiprocessing wrapper for decomposed export** — when parallel mode is selected, `vlm_prefix` and `expert_denoise` exports run as two `multiprocessing.get_context("spawn")` worker processes that each load the policy independently from disk and produce their respective ONNX artifact in parallel. Subprocess crashes propagate to the parent with full traceback (no silent SubprocessError swallowing).
- **`faulthandler.enable()` in FastAPI lifespan** — Python's built-in fault handler now installs at server startup so any C-level crash (SIGSEGV / SIGABRT / SIGFPE) during model load prints a Python traceback to stderr **before** the process dies. Without this, signal-based deaths were silent (caught 2026-04-28 by Rob's RTX 5090 segfault). Disable via `REFLEX_NO_FAULTHANDLER=1` env var.
- **18 unit tests** for export-mode auto-detection across the hardware matrix (Mac CPU, Orin Nano 8 GB, A10G 24 GB, A100 80 GB, RTX 5090 32 GB, T4 16 GB borderline) — verify mode selection + InsufficientVRAMError behavior.
- **Modal A100-80GB validation** of the full pipeline (`scripts/modal_export_pi05_decomposed.py --export-mode auto/parallel`). Auto run produced clean 12,995 MB decomposed export. Forced-parallel raised `InsufficientVRAMError` before model load. Full reproducer + cost (~$5) in `reflex_context/03_experiments/2026-04-28-pi05-decomposed-export-mode-modal.md`.

### Notes
- **Pi0.5 parallel actual-success not yet recorded** — would need a GPU with at least the conservative free-VRAM threshold (~112.7 GB free for pi0.5, only realistic on H200 / multi-GPU setups with workload that frees memory first). Documented honestly in the experiment note. SmolVLA decomposed export (which has much smaller per-pass VRAM) is the realistic parallel-win path; will land in a v0.6.x follow-up if/when SmolVLA decomposed export ships.
- **No Blackwell support changes** in v0.6.0 — Blackwell remains documented as not yet supported (per v0.5.5 README). Real Blackwell fix is the v0.7 TensorRT-LLM runtime path (ADR `2026-04-28-tensorrt-llm-runtime-path.md` in the vault).
- Repo gitignore extended to prevent agents/IDEs from accidentally creating a sibling `reflex_context/` inside reflex-vla (caught + cleaned up 2026-04-29).

## v0.5.5 — 2026-04-28

Blackwell GPU support — TensorRT EP auto-disabled to prevent segfault.

### Fixed
- **TensorRT EP no longer segfaults on Blackwell GPUs (RTX 50-series, B200, GB200).** Caught 2026-04-28 by first-tester Rob (RTX 5090, exit code 139 + `ip 0x0` per dmesg). Root cause: ONNX Runtime's bundled TensorRT runtime predates Blackwell (sm_100), so TRT can't register kernels for the unknown architecture and leaves a NULL function pointer that gets called during model load. Auto-detects Blackwell via `nvidia-smi` and excludes `TensorrtExecutionProvider` from the providers list before session creation.
- Falls back to `CUDAExecutionProvider` on Blackwell — same numerics (cos=+1.0 vs PyTorch), ~3-5× slower per inference. Suitable for chat / dev / prototyping / low-Hz robot control. Marginal for real-time control above 20 Hz.
- **Loud, multi-line warning at every server startup on Blackwell** explaining the perf trade-off, the upstream tracking issue, and what workloads are/aren't impacted. Per CLAUDE.md "no silent fallbacks that paper over errors" — this is documented degradation, not silent. The warning auto-clears when ORT ships a Blackwell-aware TensorRT bundle.

### Notes
- Detection patterns: `rtx 50`, `rtx pro 60`, `blackwell`, `b200`, `gb200`. Add new patterns to `_BLACKWELL_GPU_PATTERNS` in `runtime/server.py` if NVIDIA ships another Blackwell SKU under a different name.
- No code changes affect non-Blackwell GPUs — TRT EP path is identical to v0.5.4 there.
- Subprocess wrapper for catching C-level crashes generically (so the next "TRT segfaults on a future architecture we don't know about" is also visible) is deferred to v0.6.0 — needs careful lifespan handling design + test matrix.

## v0.5.4 — 2026-04-28

Real fixes for the four UX issues surfaced by the first-tester debugging session.

### Added
- **Pre-startup model load logging + loud failure on lifespan crash.** Before v0.5.4, if `server.load()` raised an exception during FastAPI's lifespan startup, uvicorn swallowed the exception and the user saw "Waiting for application startup." followed by silent process death. Now we wrap `server.load()`, print a full traceback to stderr with a clear "FATAL: model load failed" banner, list the three most common causes (stale cache / ORT-TRT mismatch / GPU OOM) with concrete fixes, then re-raise so uvicorn exits non-zero. Added "Loading model from ..." phase log so users can see what's happening during the 10-60s cold start.
- **Export cache version pinning + auto-invalidation.** `reflex go` now writes `_reflex_meta.json` alongside `VERIFICATION.md` with the reflex version, model_id, export_target, export_mode, and timestamp. On cache hit, validates: (1) version matches current package, (2) target matches what we'd build for current hardware. Mismatches → loud `⚠ Cache stale` message + automatic `rm -rf` of the cache dir + rebuild. Legacy caches (no `_reflex_meta.json`, built by ≤0.5.3) are also rebuilt automatically. No more silent stale-cache server crashes.

### Fixed
- **`__version__` in `reflex/__init__.py` was stale.** Pinned to `0.5.0` since launch despite shipping v0.5.0 → v0.5.3. Now matches `pyproject.toml`. The cache versioning fix above relies on `__version__` being correct.
- **Bundled embodiment presets used fractional `rtc_execution_horizon`** (`0.5` for franka/ur5, `0.4` for so100), triggering a deprecation warning at every `reflex go` startup since v0.5.2. Converted to integer counts: franka/ur5 → `25` (0.5 × chunk_size 50), so100 → `12` (0.4 × chunk_size 30). Schema v2 will reject fractionals.

### Documentation
- **README "Upgrading" section.** Calls out `pip install --upgrade reflex-vla` and `uv add --refresh reflex-vla` (uv caches the package index aggressively and won't see new releases without `--refresh` — caught when the first tester couldn't pull v0.5.3 right after release). Documents the manual cache-clear escape hatch for users still on ≤0.5.3 caches.

### Notes
- All four bugs caught from the same first-tester (Rob, RTX 5090, Arch Linux) debugging session that spanned roughly 90 minutes from public install URL going live → v0.5.4 ship.
- Total v0.5.x velocity: 5 patch releases in 6 hours, all driven by real first-user feedback.

## v0.5.3 — 2026-04-28

Quickstart polish — README + embodiment error message.

### Changed
- **README quickstart now leads with the no-embodiment smoke test.** The headline example used to be `reflex go --model smolvla-base --embodiment franka`, which crashes the server hard if the preset isn't loadable. The first command is now `reflex go --model smolvla-base` (always works, returns raw actions). Embodiment is the second example, framed as "now make it real."
- **Embodiment-not-found error is much more actionable.** Lists bundled presets, then three workarounds in priority order: (1) drop `--embodiment`, (2) use a bundled preset, (3) drop your own JSON at `~/.cache/reflex/embodiments/<name>.json` + use `--custom-embodiment-config`. Prior message just said "run scripts/emit_embodiment_presets.py" which isn't shipped to pip-installed users.

### Notes
- No behavioral changes to embodiment loading semantics — preset-not-found still fails hard at startup. The hard-vs-soft fail decision is parked for a separate v0.5.4 discussion.

## v0.5.2 — 2026-04-28

Embodiment presets ship in the package now.

### Fixed
- **Embodiment preset JSONs (`franka`, `so100`, `ur5`) now ship inside the package** at `reflex/embodiments/presets/`. Before v0.5.2 these lived only in `<repo>/configs/embodiments/` outside the package — so `pip install`ed users running `reflex go --embodiment franka` (which is the example in the README) hit `Unknown embodiment preset 'franka'. Available: (none)`. Caught within the first hour of public install (Rob, RTX 5090 testing run).
- **`pyproject.toml` now explicitly force-includes** `src/reflex/embodiments/presets/` in the wheel via `[tool.hatch.build.targets.wheel.force-include]` so the JSONs are guaranteed to ship.
- **Dev-mode fallback**: when running from a source checkout (editable install) and the bundled presets dir is missing for any reason, falls back to `<repo>/configs/embodiments/`. Keeps the dev workflow working.

### Notes
- No behavioral changes to embodiment loading or normalization — just package bundling.
- Embodiment is still optional: `reflex go --model X` (no `--embodiment`) works for testing without normalization.

## v0.5.1 — 2026-04-28

First-tester polish — bugs caught within the first hour of public install.

### Added
- **Bootstrap installer** at `https://fastcrest.com/install`. One-liner: `curl -fsSL https://fastcrest.com/install | sh`. Detects platform (Mac / Jetson Orin / NVIDIA GPU / CPU) and picks the right `[serve,*]` extras automatically. Bails fast on unsupported hardware (original Maxwell-era Jetson Nano: JetPack 4.6 + 4 GB memory + no Tensor Cores can't run modern VLAs — redirected to Mac / Orin / cloud paths). Bootstraps `pip` via `ensurepip` when missing (caught on Arch with system Python 3.13 lacking pip module). Source: `install.sh` in the repo root.

### Fixed
- **`reflex doctor` /tmp check is no longer misleading on tmpfs systems.** Many Linux distros mount `/tmp` as `tmpfs` (RAM-backed), where the previous "Free disk in /tmp" check was actually measuring free RAM. Doctor now detects tmpfs via `/proc/mounts` and labels the check accordingly: `(tmpfs/RAM-backed — model exports use ~/.cache/reflex/exports instead)`. Also clarifies that `/tmp` only holds transient ONNX/TRT scratch — the real export cache lives at `~/.cache/reflex/exports` (or `$REFLEX_HOME/exports`). Reduced the threshold from 10 GB → 2 GB since /tmp doesn't need to hold the full model artifact.

### Notes
- No functional code changes vs v0.5.0 in the model export, serve, or chat paths.
- The bootstrap installer is independent of the PyPI package — it just calls `pip install` after pre-flight checks. Existing `pip install reflex-vla` continues to work.

## v0.5.0 — 2026-04-28

License + repo visibility milestone.

### Changed
- **License: Apache 2.0 → Business Source License 1.1** (auto-converts to Apache 2.0 in 4 years). Same source-available model HashiCorp, MongoDB, Sentry, Cockroach, and Couchbase use. Free for any non-competitive use (personal, commercial, internal); restricts only competing hosted/embedded offerings. Older releases (v0.3.x, v0.4.x) remain Apache-licensed forever — that grant cannot be retracted.
- **GitHub repo flipped from private to public.** Source now visible at https://github.com/FastCrest/reflex-vla. Earlier hidden by accident.

### Security
- **Scrubbed leaked HuggingFace token** (`hf_rfnFx...`) from git history via `git filter-repo`. Also scrubbed accidentally-committed `.agents/` editor-agent session logs (135 files across 4 commits). The token has been revoked at huggingface.co.
- Added `.agents/` to `.gitignore` to prevent recurrence.

### Notes
- This is a license/visibility release — no functional code changes vs v0.4.1.
- The closed-source-binary architecture explored mid-development was reversed: BSL 1.1 provides legal protection against commercial cloning without losing the open-source adoption story.

## v0.4.1 — 2026-04-27

UX onboarding pass — discoverability + persistence + the missing one-command tool.

### Added
- **First-run welcome card in `reflex chat`** — shown once per machine (cached in `$REFLEX_HOME/.welcomed`), explains what the assistant can do, lists slash commands, suggests starter prompts. Blank-prompt-paralysis is dead.
- **Slash commands**: `/help`, `/tools`, `/history`, `/clear`, `/reset`, `/tour`. `/tools` lists all 17 tools grouped by category (Deploy / Models / Train / Inspect / Status). `/history` shows the conversation so far. `/tour` shows 5 example prompts to copy-paste.
- **Conversation persistence** — every chat session auto-saves to `$REFLEX_HOME/chat_history/session-YYYYMMDD-HHMMSS.jsonl` after each turn. New CLI flag: `reflex chat --resume` loads the most recent session so Ctrl+C never loses context.
- **`deploy_one_command` chat tool** wrapping `reflex go`. The chat agent can now do "deploy smolvla to my mac" as a single tool call instead of manually chaining 4 tools (probe → pull → export → serve). Closes the audit gap where the headline command wasn't in chat.

### Changed
- **`reflex` (no args) shows a curated action-first summary** instead of typer's alphabetical command dump. Leads with `chat` and `go` — the two commands 90% of users want — followed by `doctor` and `models list`. Full alphabetical list still available via `reflex --help`.

### Notes
- Total chat tools is now **17** (was 16). Regression test updated.
- 47/47 tests pass.

## v0.4.0 — 2026-04-27

Polish release — bundles five small wins + one new optional surface.

### Added
- **Textual TUI for `reflex chat`** (opt-in via `pip install 'reflex-vla[tui]'`, then `reflex chat --tui`). Multi-panel layout: scrollable transcript, dedicated tool-calls panel with live status, persistent input box, status bar with token/tool count. Mouse + scroll-back + keyboard shortcuts (Ctrl+L clear, Ctrl+R reset). Falls back to the Rich REPL automatically if textual isn't installed. New module: `src/reflex/chat/tui.py`.
- **Examples directory** — `examples/01-chat-quickstart.md`, `02-deploy-smolvla-jetson.md`, `03-distill-pi05.md`, `04-record-and-replay.md`. Self-contained walkthroughs for each major workflow.
- **Once-per-day upgrade nag** — `reflex --version` (or any subcommand) now prints a one-line nag to stderr if PyPI has a newer release. Cached 24h in `$REFLEX_HOME/.upgrade_check`. Disable via `REFLEX_NO_UPGRADE_CHECK=1`. Skipped on editable installs. New module: `src/reflex/upgrade_check.py`.
- **PyPI install digest script** — `python scripts/install_digest.py` pulls download counts via pypistats and prints a Markdown summary suitable for sharing.

### Changed
- **Chat tool result truncation now keeps the tail** (`executor.py:_smart_truncate`). Long stack traces and compile errors put the actionable info at the end; the old head-only truncation lost that. New default: 1/3 head + 2/3 tail with a marker.

### Notes
- `[tui]` extra adds ~5 MB (textual). Base install footprint unchanged for users who don't want the TUI.

## v0.3.5 — 2026-04-27

### Added
- **`reflex chat` now streams tokens live.** Reply renders to the terminal as it arrives instead of showing "thinking…" then a full block. Multi-tool queries still surface tool calls between turns. Use `--no-stream` for scripts that pipe output.
- New backend method: `ChatBackend.chat_stream()` — yields parsed OpenAI delta chunks (Server-Sent Events through the existing Cloudflare Worker proxy).
- New helper: `assemble_stream(chunks, on_token, on_tool_call_progress)` — assembles streaming chunks into a final assistant message dict, with optional callbacks for live UI.
- New event: `LoopState` emits `token` events (per content fragment) and `turn_start` events (per LLM round-trip).
- New flag: `LoopState.streaming: bool = True` — set False for tests that need deterministic single-shot replies.

### Changed
- **System prompt tightened against hallucination.** Added a CRITICAL rule: "Copy verbatim values (versions, paths, IDs, sizes, error messages) exactly from tool output. Do not paraphrase, round, or 'fix' them. If you didn't run a tool that returned the value, say 'I don't have that information' instead of guessing." Closes the v0.3.0 case where chat cited "torch 2.10.0" when the actual was 2.11.0.

### Fixed
- (No regressions — 46/46 tests pass including all 16 chat-tool routes.)

## v0.3.4 — 2026-04-27

### Changed
- **`reflex doctor` now suggests the right install extras for your platform.** On Apple Silicon (no NVIDIA), recommends `'reflex-vla[serve,onnx]'` (CPU runtime). On NVIDIA boxes, still recommends `'reflex-vla[serve,gpu]'`.
- **`reflex models pull` now accepts HuggingFace repo IDs** in addition to registry aliases. `reflex models pull lerobot/smolvla_base` works just like `reflex models pull smolvla-base` — automatically resolved to the registry entry.

### Fixed
- Doctor's install hint no longer eats `[serve,gpu]` due to Rich markup interpretation (escaped properly with raw string + escaped bracket).

## v0.3.3 — 2026-04-27

### Added
- **`reflex go` now actually deploys.** When a model has `requires_export=True`, `reflex go` runs the export inline (chains pull → export → serve) instead of printing manual instructions. Closes the biggest README→reality gap.
- New CLI command: **`reflex status`** — list running `reflex serve` processes via `ps` regex (PID, uptime, port, command).
- New CLI command: **`reflex config show`** — dump effective config (paths, defaults, env vars).
- New CLI command: **`reflex inspect traces`** — scan `~/.cache/reflex/traces` and `/tmp/traces` for JSONL files written by `reflex serve --record`. Filters: `--since`, `--task`, `--status`, `--limit`.
- New env var: **`REFLEX_HOME`** — overrides `~/.cache/reflex` for export cache root + config defaults.
- New regression test: `tests/test_chat_tools_executable.py` parametrized over all 16 chat tools — asserts each routes to a real CLI command.

### Changed
- `reflex chat` proxy default URL is now `https://chat.fastcrest.com` (was `fastcrest-proxy.fastcrest.workers.dev`).
- Export cache for `reflex go` lands at `~/.cache/reflex/exports/<model_id>/` (or `$REFLEX_HOME/exports/<model_id>/`). Cache-skip on `VERIFICATION.md` marker.

### Fixed
- **Rich markup ate `[monolithic]`** in install hints. Users saw `pip install 'reflex-vla'` instead of `pip install 'reflex-vla[monolithic]'`. Fixed by escaping brackets or using `markup=False` in 5 places.
- 4 `reflex chat` tools routed to non-existent CLI commands. All four now route correctly thanks to the new `status` / `config show` / `inspect traces` commands.

## v0.3.2 — 2026-04-26

### Changed
- **24× CLI speedup.** `reflex/__init__.py` now lazy-loads `validate_roundtrip` + `fixtures` via PEP 562 `__getattr__` instead of eager-importing torch. `reflex --version` 2.4s → **0.10s**. Every fast-path command (`--help`, `chat`, `models list`, `inspect targets`, `inspect traces`, `config show`) is now sub-second on a warm cache. `reflex doctor` still imports torch on-demand (correct — it's the diagnostic).

## v0.3.0 — 2026-04-26

### Added
- **`reflex chat` ships.** Natural-language CLI agent: GPT-5 Mini routes user prompts to 16 reflex tools (export, serve, bench, eval, distill, finetune, traces, doctor, etc.) and runs them as subprocess. Hosted Cloudflare Worker proxy at `chat.fastcrest.com` — free tier 100 calls/day per machine, no signup, no API key.
- New module: `src/reflex/chat/{schema,backends,executor,loop,console}.py`
- New CLI: `reflex chat [--proxy-url URL] [--dry-run]`

### Changed
- **PyPI launch.** `pip install reflex-vla` now works without a git URL. Bumped from internal `0.1.0` to public `0.3.0`.
- README rebranded with "Reflex by [FastCrest](https://fastcrest.com)" header + chat quickstart at top.

## v0.2.x (pre-PyPI internal milestone tags — never published)

## Unreleased

### Added
- `reflex validate` command now runs a real ONNX/TRT-vs-PyTorch round-trip parity check.
  - Seeded fixtures for SmolVLA, pi0, GR00T (pi0.5 and OpenVLA defer to v2).
  - Per-fixture max/mean L2 abs-diff + summary, JSON and Rich table output.
  - `--init-ci` emits a GitHub Actions workflow template at `.github/workflows/reflex-validate.yml`.
  - Exit codes: 0 pass, 1 fail, 2 error.
- Public exports added to `reflex`: `ValidateRoundTrip`, `load_fixtures`, `SUPPORTED_MODEL_TYPES`.

### Changed
- **BREAKING (from stub):** `reflex validate` default `--threshold` changed from `0.02` (the v0.1 placeholder) to `1e-4`. The stub never performed real validation so no existing deployments depended on the old default. Pass `--threshold 0.02` explicitly to match the previous behavior.
- `reflex validate` now requires a valid `reflex_config.json` inside the export directory — the stub accepted any path.

### Fixed
- `_pytorch_backend` SmolVLA path no longer swallows `AutoConfig` fetch errors silently — now logs a warning with the exception and continues with the fallback head_dim.
- CLI handler now catches `KeyboardInterrupt` explicitly (exits 130) instead of emitting a raw traceback.

## v0.1.0 (previous)

Initial release — see README for the seven-wedge scope at that time.
