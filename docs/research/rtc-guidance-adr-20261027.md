# ADR: RTC guidance over HTTP `/act`

Date: 2026-10-27. Status: accepted by owner (decisions a-c, 2026-10-27). Branch
`feat/rtc-guidance-act` from `origin/main` `cb586a7`.

Grades: VERIFIED = code read or executed in this pass; ASSERTED = a source says
so, not re-derived; UNKNOWN = not established.

## Context

`tether serve --rtc` was refused at start-up with "the loaded policy must expose
a differentiable per-denoise-step contract" (`server.py:2097-2103` on `cb586a7`,
VERIFIED). The RTC adapter existed but `/act` never called its guidance path, the
leftover was the whole previous chunk (HTTP never pops the buffer), and the
leftover was in processed units while guidance runs in normalized space
(VERIFIED, read on `cb586a7`).

Owner decisions (2026-10-27): (a) SmolVLA through LeRobot 0.6.1's own RTC on the
PyTorch path first, then a numpy port of LeRobot's step for the pi0.5 per-step
ONNX loop, checked against LeRobot on matched seeds; reuse LeRobot (Apache-2.0)
rather than writing our own. (b) New public `/act` fields `actions_executed` and
`chunk_id`. (c) GPU qualification deferred; no paid jobs.

## The Jacobian question (settled)

Does the RTC paper's reference implementation use the full Jacobian?

- **Paper, yes.** arXiv 2506.07339 Eq. 2 is the ΠGDM correction
  `(Y - Â1)^T diag(W) ∂Â1/∂A_τ`; the text after Eq. 4 calls it a vector-Jacobian
  product computed by backpropagation, and Algorithm 1 says to compute it "via
  autodiff". β clipping is the authors' addition (App. A.2, β = 5). VERIFIED
  (PDF read).
- **Reference code, yes.** `Physical-Intelligence/real-time-chunking-kinetix`
  `src/model.py` at `9296f31d`: the network runs inside `denoiser(x_t)`
  (L236-239) and L241 calls `jax.vjp(denoiser, x_t, has_aux=True)`; L246 applies
  `vjp_fun(error)`. The Jacobian is `I + (1-t) ∂v/∂x_t`. VERIFIED (source read).
- **LeRobot 0.6.1, no.** `lerobot/policies/rtc/modeling_rtc.py` L213 computes
  `v_t` before L214 `x_t.requires_grad_(True)` (`x_t` was detached at L172), so
  `autograd.grad` at L219 returns `err` exactly: an identity Jacobian. VERIFIED
  (source read; and by execution: Tether's numpy port with correction = err
  matches LeRobot to 9.5e-7). Upstream: issue huggingface/lerobot#2511 (closed;
  maintainer kept the behaviour as it "works with real robot"); PRs #2499 and
  #4945 (adds `guidance_jacobian = "identity" | "vjp"`) are open. ASSERTED
  (GitHub API read by a research agent, not re-fetched).
- Guidance weight `min(β, (1-τ)/(τ r_τ²))` with `r_τ² = (1-τ)²/(τ²+(1-τ)²)`:
  LeRobot matches the paper and Kinetix with the time axis flipped. VERIFIED.

**They differ. Tether implements LeRobot's behaviour** (owner decision a) and
labels it: every `/act` RTC block carries `"guidance": "lerobot_identity_jacobian"`.

**Effect of the difference** (measured, `docs/research/verification/rtc-smolvla-local-20261027.md`):
on the full-size SmolVLA architecture with random weights, `‖Jᵀe − e‖ / ‖e‖`
is 3.3-3.8 % mean, 6.6 % max per denoise step. The paper's VJP gave no better
seam continuity than identity in the kinematic sim, and cost 1.6-1.9x the RTC
latency (one backward pass through the expert per step), which at a 10-action
delay starved the robot for 92 actions over 100 chunks. With identity guidance
the correction is zero wherever W = 0: the prefix is matched, but the effect of
the prefix is not propagated through the model onto later rows. Whether that
matters on a trained policy is UNKNOWN until task-success evidence exists
(needs a checkpoint and a sim the policy was trained for).

## Decisions and rejected alternatives

### D1. Where guidance runs

| Option | Verdict | Why |
|---|---|---|
| **A. Pass-through to LeRobot's RTC-aware loop (SmolVLA native)** | **Taken** for SmolVLA | Zero math to own; LeRobot is the reference; Apache-2.0. `SmolVLANativeServer` forwards `inference_delay`, `prev_chunk_left_over`, `execution_horizon` to `predict_action_chunk`. |
| **B. Numpy port of LeRobot's step in the ONNX per-step loop** | **Taken** for pi0.5 per-step | Keeps ORT speed; ~30 lines; differential oracle against LeRobot (`tests/test_rtc_guidance_oracle.py`, `tests/test_rtc_pi05_per_step.py`). |
| C. Guidance baked into the ONNX graph | Rejected | Horizon/delay/schedule become graph inputs or re-exports; hardest to debug; no oracle. |
| D. Own torch implementation with the paper's VJP | Rejected | Owner decision (a) is reuse LeRobot; needs autograd through the expert (measured 1.6-1.9x); cannot run in ONNX. |
| E. Hard prefix inpainting (GR00T `frozen_steps`) | Rejected | Different technique; discontinuity at the frozen boundary. |

### D2. HTTP state contract

| Option | Verdict | Why |
|---|---|---|
| **A. Client reports `actions_executed` + `chunk_id`** | **Taken** (owner decision b) | Exact leftover. 409 on a stale `chunk_id`, 422 on an incomplete or out-of-range report. |
| **B. Wall-clock estimate** | **Kept as labelled fallback** | Old clients keep working; the response says `leftover_source: "estimated"`. Never silent. |
| C. Server-side action queue / streaming `/act` | Rejected | Changes `/act` from request-response to a stream; larger. |
| D. RTC only on the ZMQ transport | Rejected | Leaves HTTP, the stated gap, unsolved. |

**Correction 2026-10-28 (independent review, PR #392).** As first written the
carry-forward was server-global: two clients without `episode_id` shared one
leftover, an episode-less request inherited the last episode's leftover, and
concurrent `/act` calls could both plan against the same chunk (VERIFIED by the
reviewer's spikes and by failing-first tests in `tests/test_rtc_streams.py`).
Now:
- State is per stream: with `--rtc` every `/act` must carry `episode_id`
  (422 `rtc_episode_required`), and each episode has its own adapter and
  buffer (`RtcStreams`, LRU-capped at 64 idle streams).
- Plan -> predict -> commit is serialised per stream by an `asyncio.Lock`;
  other streams run concurrently.
- An exact retry of the report that produced the last chunk returns that chunk
  (`X-Tether-RTC-Replay: true`) instead of 409. The Python client no longer
  retries non-408/429 4xx, and its episode proxy drops a chunk id that got 409.
- `actions_executed` is a strict int in [0, 4096] (and <= chunk length);
  `chunk_id` is lowercase hex, at most 64 characters.

**Correction 2026-10-28 (adversarial re-review of `b4802d8`).**
- An exact replay was answered before the license, circuit-breaker and
  guard-latch checks, so a retry could return actions while the guard was
  tripped. The gates now run first for every RTC request, replays included.
- `episode_id` became the stream key but was unbounded; under `--rtc` it is
  now 1-128 characters of `[A-Za-z0-9._:-]` and is logged only as length plus
  a short hash.
- LRU eviction let 64 unknown ids evict a live robot, and could delete a
  stream between a lock release and its queued waiter. Streams now count
  users; only a stream with no users, idle at least `--rtc-stream-min-idle-s`
  (300 s), is evicted at `--rtc-max-streams` (64), otherwise the new episode
  gets 503.
- A reused `episode_id` carried the last episode's leftover. Contract chosen:
  one `episode_id` per episode (the client's `episode()` generates one), and
  `episode_start: true` on the first request when an id is reused. That is
  explicit, per stream, and cannot be sent with a report.
- Replays now carry the original `X-Tether-*` headers.

### D3. Leftover units

| Option | Verdict | Why |
|---|---|---|
| **A. Store the returned (post-guard) chunk; convert the leftover with each runtime's `normalize_actions`** | **Taken** | The leftover is what the client actually executes, including any embodiment-guard clamp. SmolVLA uses its postprocessor's own `UnnormalizerProcessorStep` stats (forward transform); pi0.5 inverts its `x * std + mean`. SmolVLA refuses a postprocessor it cannot invert. |
| B. Store the pre-postprocessor chunk (LeRobot's `original_queue`) | Rejected | Ignores guard clamps; diverges from executed actions. Equal to A when nothing clamps (round-trip error ≤ 1e-5, tested). |
| C. Guide in processed space | Rejected | Wrong units: the bug being fixed. |

### D4. Reuse vs port vs own (licence)

| Option | Licence | Verdict |
|---|---|---|
| **A. Import `lerobot.policies.rtc` where torch exists; numpy port only for ORT** | Apache-2.0 (VERIFIED, `lerobot-0.6.1.dist-info`) | **Taken**; notice in `THIRD_PARTY_LICENSES.md`, header in `rtc_guidance.py`. |
| B. Vendor all of LeRobot RTC | Apache-2.0 | Rejected: drift, no benefit. |
| C. LeRobot `RTCInferenceEngine` wholesale | Apache-2.0 | Rejected: owns the robot loop and observation threads; does not fit HTTP. |
| D. Kinetix reference (JAX) | Apache-2.0 (ASSERTED) | Rejected: JAX, a second framework at serve time. |

### D5. Capability gate replaces the blanket refusal

`create_app` asks the server for `rtc_guidance_support()`; only SmolVLA native
and pi0.5 `per_step_expert=true` answer None. Everything else is refused with
"RTC guidance needs per-step velocity access inside the denoise loop…". Also
refused with `--rtc`: `--max-batch > 1`, `--a2c2-checkpoint`,
`--bid-n-candidates`, `--action-similarity-threshold`, `--policy-b` (each
replaces or reuses chunks outside the guided loop). Adapter construction errors
stop start-up; the old `except` that logged and served without RTC is gone.

## Known costs and open items

- **RTC latency overhead (corrected 2026-10-28).** The 2026-10-27 entry here
  said LeRobot's `enable_grad` graph costs +17-27 % and that removing it needs a
  LeRobot patch. **That was wrong.** In `modeling_rtc.py` L212-219 `v_t` is
  computed inside `enable_grad` but before `x_t.requires_grad_(True)`, so autograd
  records the denoiser only because the policy's parameters had
  `requires_grad=True`. `SmolVLANativeServer.load` now calls `eval()` and
  `requires_grad_(False)`; the correction still equals the error (x1_t depends
  on x_t directly) and RTC output is bit-identical (test). Re-measured overhead:
  MPS +2.6 % / -0.7 % mean at delays 6 / 10, CPU +0.5 %; MPS delay 3 +13 % mean
  was a single-pair artefact: an interleaved 6x ABAB run gives pooled rtc 204.0
  vs off 212.8 ms (verification doc, 2026-10-28). pi0.5 RTC runs in ONNX
  (no torch parameters); Tether has no torch pi0.5 serving path to fix.
- **pi0.5 startup refusals (2026-10-28).** `rtc_guidance_support()` now also
  refuses the similarity fast path, BID and A2C2, and the A2C2/BID setters raise
  once RTC is required; before, a misconfigured server started and failed every
  request. The served inference is always built with `cache_level="prefix"`, so
  the action cache cannot be on.
- **TensorRT IOBinding (2026-10-28).** The CPU fix bound past_kv to host memory
  unless the first provider was CUDA, so a TensorRT-first session would copy
  ~140 MB host->device per Euler step (reasoned from code; base `cb586a7`
  always bound `"cuda"`). `iobinding_device()` now pins to `"cuda"` for CUDA or
  TensorRT sessions and host memory otherwise. That TensorRT consumes CUDA
  OrtValues is ASSERTED (ORT docs); not measured on a GPU in this pass.
- `inference_delay` rounds up (`ceil(p95 latency × execute_hz)`) as LeRobot does.
- Padded action dims: like LeRobot, the leftover is zero-padded to the model's
  action width, so the guided prefix also pulls padded dims toward 0.
- GPU qualification (CUDA IOBinding overhead with guidance, LIBERO delay sweep
  with CIs) is deferred (owner decision c).

## Capability ledger row

| Component | Method in use | Best-known alternative | Surveyed | Revisit when |
|---|---|---|---|---|
| RTC guidance | LeRobot 0.6.1 identity-Jacobian correction (torch pass-through; numpy port for ORT) | Paper/Kinetix VJP (`jax.vjp`); LeRobot PR #4945 `guidance_jacobian="vjp"` | 2026-10-27 | #4945 merges, or a trained-policy task-success A/B shows the VJP matters |
