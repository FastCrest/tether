# RTC over HTTP `/act`: local CPU/MPS evidence

Date: 2026-10-27. Branch `feat/rtc-guidance-act` (base `cb586a7`). Host: Mac
arm64, macOS 26.6.2, Python 3.12.12, torch 2.11.0, LeRobot 0.6.1, worktree
`.venv` synced from `uv.lock` (`--extra serve rtc dev onnx native`). No paid
compute. Raw receipts: `rtc-smolvla-local-20261027/{mps,cpu,mps-nograd}.json`.
Script: `scripts/local_rtc_smolvla_evidence.py`.

Grades: VERIFIED = executed here; UNKNOWN = not established.

## What this does and does not show

- Model: full-size SmolVLA architecture (450.0M parameters, chunk 50, 10 denoise
  steps, 16 VLM layers) with **random seeded weights**. No checkpoint was
  downloaded. Latency is architecture-determined and real; action quality is not.
- Sim: kinematic, virtual clock, perfect tracking (state = last executed action),
  constant zero image. The robot executes 20 actions, requests the next chunk with
  `actions_executed` + `chunk_id`, keeps executing the old chunk for
  `ceil(latency × hz)` actions, then switches. `hz` maps the warm p50 latency to
  the target delay. RTC: LINEAR schedule, horizon 20, β 10.
- Arms on matched seeds: `off` (naive async), `rtc` (what Tether serves),
  `rtc_vjp` (paper/Kinetix VJP, diagnostic patch), `rtc_nograd` (LeRobot's
  arithmetic without `enable_grad`, diagnostic patch).
- **Not shown:** task success (needs a trained checkpoint and a sim it was trained
  for), GPU behaviour, pi0.5 end to end with real weights.

All requests went through the real `create_app` + `/act` path
(`SmolVLANativeServer` with LeRobot's RTC-aware `euler_integrate`). `rtc` blocks
reported `applied: true` on 100 % of non-first chunks in RTC arms. VERIFIED.

## Results (mean / p50 / p95 / p99 / max; 100 chunks per arm, MPS)

Prefix error = max |new[k] − old[idx+k]| over the actions executed during
inference (processed units). Seam jump = ‖new[d] − old[idx+d]‖ at the switch.

| delay target | arm | prefix error | seam jump L2 | latency ms |
|---|---|---|---|---|
| 3 | off | 4.613 / 4.610 / 6.618 / 7.078 / 7.459 | 4.954 / 4.824 / 7.508 / 7.992 / 8.378 | 222.5 / 209.2 / 281.1 / 289.0 / 291.7 |
| 3 | rtc | 0.0033 / 0.0032 / 0.0045 / 0.0055 / 0.0055 | 0.0051 / 0.0039 / 0.0073 / 0.0118 / 0.0125 | 263.8 / 228.0 / 345.0 / 360.3 / 371.4 |
| 3 | rtc_vjp | 0.0034 / 0.0033 / 0.0038 / 0.0063 / 0.0067 | 0.0050 / 0.0039 / 0.0083 / 0.0135 / 0.0148 | 422.5 / 451.4 / 538.8 / 568.1 / 605.1 |
| 6 | off | 5.283 / 5.165 / 7.532 / 8.042 / 8.334 | 5.105 / 4.816 / 7.983 / 10.51 / 10.62 | 232.6 / 201.6 / 299.6 / 310.2 / 321.4 |
| 6 | rtc | 0.0035 / 0.0034 / 0.0036 / 0.0061 / 0.0075 | 0.0054 / 0.0040 / 0.0088 / 0.0141 / 0.0156 | 294.9 / 314.9 / 373.7 / 394.4 / 417.5 |
| 6 | rtc_vjp | 0.0039 / 0.0035 / 0.0042 / 0.0156 / 0.0181 | 0.0067 / 0.0041 / 0.0240 / 0.0305 / 0.0892 | 418.8 / 434.6 / 534.4 / 555.3 / 561.5 |
| 10 | off | 5.841 / 5.651 / 8.156 / 10.48 / 10.72 | 4.909 / 4.730 / 7.678 / 8.767 / 9.020 | 215.7 / 186.6 / 293.1 / 306.9 / 309.3 |
| 10 | rtc | 0.0041 / 0.0035 / 0.0037 / 0.0056 / 0.0648 (n=98) | 0.0094 / 0.0041 / 0.0051 / 0.170 / 0.242 (n=98) | 269.6 / 229.6 / 361.5 / 380.5 / 441.5 |
| 10 | rtc_vjp | 0.0043 / 0.0036 / 0.0040 / 0.0191 / 0.0395 (n=78) | 0.0070 / 0.0042 / 0.0048 / 0.0724 / 0.104 (n=78) | 343.0 / 318.0 / 464.3 / 506.5 / 521.1 |

- VERIFIED: RTC cuts the prefix disagreement and seam jump by about three orders
  of magnitude (≈5 → ≈0.004) at every delay. `n < 100` rows had a delay beyond
  the remaining chunk; `rtc_vjp` at delay 10 starved for 92 actions.
- **Boundary jerk does not discriminate here.** Third-difference jerk at the
  seam equals within-chunk jerk in every arm (e.g. delay 3: off 15.22 vs 15.32,
  rtc 15.25 vs 14.43) because a random-weight policy's actions are already
  discontinuous step to step. The full jerk distributions are in the JSON. A
  trained policy is needed for this metric to mean anything (UNKNOWN).
- CPU (30 chunks, delay 6): off latency 2023 / 1789 / 3052 / 3193 / 3220 ms,
  rtc 2362 / 2300 / 3278 / 3315 / 3328 ms; prefix error off 5.79 mean, rtc
  0.0043 mean (max 0.012). VERIFIED.

## Where RTC's latency overhead comes from

RTC as served adds 17-27 % mean latency on MPS and 17 % on CPU. The
`mps-nograd` run isolates it (delay 6, 100 chunks each):

| arm | latency ms | prefix error mean |
|---|---|---|
| off | 239.1 / 241.0 / 308.1 / 318.7 / 364.5 | 5.802 |
| rtc (LeRobot as-is) | 279.5 / 308.9 / 355.1 / 380.4 / 432.6 | 0.0037 |
| rtc_nograd | 211.1 / 178.3 / 282.5 / 295.6 / 303.3 | 0.0035 |

VERIFIED: the same identity-Jacobian arithmetic without `torch.enable_grad`
removes the overhead with the same prefix error. LeRobot 0.6.1 calls the
denoiser inside `enable_grad` (`modeling_rtc.py` L212-213), recording the whole
expert forward for a gradient that is the identity. Tether serves LeRobot
unmodified per owner decision (a); removing this cost needs an owner decision
(patch/upstream fix vs. keep). Not changed in this branch.

## Identity vs VJP (the paper's correction)

`‖Jᵀe − e‖ / ‖e‖` per denoise step, `rtc_vjp` arm, 1000 steps per delay:
3.30 / 3.45 / 5.66 / 5.93 / 6.22 % (delay 3), 3.41 / 3.60 / 5.77 / 5.97 / 6.58 %
(delay 6), 3.62 / 3.85 / 5.99 / 6.25 / 6.46 % (delay 10). VERIFIED for this
random-weight model. The VJP gave no better seam metrics and cost 1.6-1.9x the
RTC latency. Its effect on a trained policy is UNKNOWN.

## Unit and oracle tests (same branch)

- `tests/test_rtc_guidance_oracle.py`: numpy port vs LeRobot over full Euler
  loops (4 schedules × delays 0-8 × NFE {2,5,10} × 3 seeds, 19,440 elements):
  |err| mean 4.4e-8, p50 0, p95 2.4e-7, p99 3.6e-7, max 9.5e-7, 0 % above 1e-6.
- `tests/test_rtc_pi05_per_step.py`: pi0.5 per-step loop on the CPU EP with a
  tiny real ONNX expert, guided, vs LeRobot `euler_integrate`: within 1e-5 for 36
  schedule × delay × NFE cases.
- `tests/test_rtc_act.py`: tiny real SmolVLA through `/act`: on matched seeds the
  RTC prefix error is below every RTC-off prefix error across 4 seeds.
