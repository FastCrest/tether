# LeRobot Real-Time Chunking (RTC) Deep Dive

**Date:** April 22, 2025
**Purpose:** Design doc for wrapping LeRobot's RTC in Reflex's serve runtime without re-deriving the math.
**Reference Paper:** arxiv 2506.07339 — Real-Time Execution of Action Chunking Flow Policies

---

## Section 1: RTC API & Lifecycle

### 1.1 Overview

RTC (Real-Time Chunking) is **not a policy**, but an **inference-time wrapper** around flow-matching policies (Pi0, Pi0.5, SmolVLA). It enables "replan-while-execute" by treating chunk generation as an inpainting problem: when the robot starts executing a new chunk, we feed the unexecuted prefix from the **previous chunk** to guide the **next chunk's denoising**, reducing divergence from the old plan.

**Key insight:** The policy's diffusion denoiser runs at constant compute cost (always N steps). RTC adds guidance at inference time (no training cost), so 2-3x throughput boost is nearly free.

**Source:** `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/modeling_rtc.py:1-35`

### 1.2 RTCProcessor: The Core Class

The main entry point is `RTCProcessor` (modeling_rtc.py:37+). It wraps the policy's denoising loop.

**Initialization:**
```python
RTCProcessor(rtc_config: RTCConfig)
```
where `RTCConfig` (configuration_rtc.py) specifies:
- `enabled: bool` — master kill-switch
- `prefix_attention_schedule: RTCAttentionSchedule` — how to weight old vs new actions
  - `LINEAR`: smooth fade from 1.0 (full weight on old prefix) to 0.0 (full weight on new)
  - `ONES`: constant 1.0 on prefix, 0.0 on rest
  - `ZEROS`: constant 1.0 on prefix, 0.0 on rest
  - `EXP`: exponential decay
- `max_guidance_weight: float = 10.0` — clamp on the guidance strength (prevents numerical instability)
- `execution_horizon: int = 10` — number of timesteps at the head of the new chunk to guide (rest is free)
- `debug: bool` — enable step-by-step tracking for diagnostics

**File:** `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/configuration_rtc.py`

### 1.3 The RTC Guidance Wrapper: `denoise_step()`

This is the key method (modeling_rtc.py:111-237). It wraps the policy's native denoiser.

**Signature:**
```python
denoise_step(
    x_t: Tensor,                              # latent action (B, T, A) or (T, A)
    prev_chunk_left_over: Tensor | None,     # unexecuted actions from prev chunk
    inference_delay: int,                    # how many steps elapsed during inference
    time: float,                              # denoise time in [0,1] (inverted in RTC)
    original_denoise_step_partial: Callable,  # the policy's native denoiser
    execution_horizon: int | None = None,    # override config horizon
) -> Tensor  # guided velocity v_t, same shape as x_t
```

**Control flow:**

1. **No guidance path** (first chunk, prev_chunk_left_over is None):
   - Call `original_denoise_step_partial(x_t)` and return v_t directly
   - Source: modeling_rtc.py:151-155

2. **Guidance path** (prev_chunk_left_over provided):
   - Squeeze/unsqueeze batch dimensions for consistency
   - Pad prev_chunk_left_over if shorter than current chunk (right-padded with zeros)
   - Compute prefix weights via `get_prefix_weights(inference_delay, execution_horizon, T)`
     - Weights are 1.0 for the first `inference_delay` timesteps (old actions to keep)
     - Weights fade to 0.0 at `execution_horizon` (new actions to replan)
     - Source: modeling_rtc.py:242-285
   
   - **Gradient-based correction** (the core math):
     ```
     v_t = original_denoise_step_partial(x_t)          # base velocity from policy
     x1_t = x_t - time * v_t                           # predicted denoised state
     err = (prev_chunk_left_over - x1_t) * weights     # weighted mismatch
     correction = ∇_{x_t} (x1_t) · err                 # backprop the error
     guidance_weight = f(time)                         # time-dependent weight
     result = v_t - guidance_weight * correction       # guided velocity
     ```
     - Source: modeling_rtc.py:157-193
   
   - **Guidance weight schedule** (modeling_rtc.py:194-199):
     ```
     tau = 1 - time                 # invert time (PI's convention)
     c = (1 - tau) / tau            # grows as we approach denoised state
     inv_r2 = ((1-tau)² + tau²) / (1-tau)²
     guidance_weight = clamp(c * inv_r2, max=max_guidance_weight)
     ```
     - Early steps (high tau): guidance is weak (policy is still adding noise, guidance would be noise)
     - Late steps (low tau): guidance is strong (policy is in the clean manifold, guidance pulls toward old actions)

**Key insight:** Guidance only acts on the **prefix**: the first `inference_delay` actions. The tail of the new chunk is free to diverge from the old plan. This allows fast re-planning without getting stuck to stale actions.

**Source files:**
- `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/modeling_rtc.py:111-237`
- `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/modeling_rtc.py:242-285` (weight schedule)

### 1.4 Integration with Policy's sample_actions()

Both Pi0 and Pi0.5 call RTC inside their main denoising loop. The policy's `sample_actions()` method:
1. Sets up the diffusion loop (noise → denoised actions)
2. For each denoising step, calls `denoise_step_partial_call(x_t)` 
3. If RTC is enabled, wraps that call via `rtc_processor.denoise_step(...)`

**Pi0 example:** `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/pi0/modeling_pi0.py:862-885`

```python
if self._rtc_enabled():
    v_t = self.rtc_processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk_left_over,
        inference_delay=inference_delay,
        time=time,
        original_denoise_step_partial=denoise_step_partial_call,
        execution_horizon=execution_horizon,
    )
else:
    v_t = denoise_step_partial_call(x_t)
```

**Pi0.5 identical:** `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/pi05/modeling_pi05.py:843-862`

**Entry point:** `predict_action_chunk(**kwargs)` accepts:
- `inference_delay: int | None` — elapsed timesteps during this inference
- `prev_chunk_left_over: Tensor | None` — unconsumed actions from the last chunk
- `execution_horizon: int | None` — override config

**Sources:**
- Pi0: `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/pi0/modeling_pi0.py:1262-1277`
- Pi0.5: `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/pi05/modeling_pi05.py:1233-1247`

---

## Section 2: Action Chunk Handling & Denormalization

### 2.1 The ActionQueue: Merge Point for RTC

LeRobot provides `ActionQueue` (action_queue.py) to manage the **original** (unprocessed) and **processed** (normalized→denormalized) action streams.

**Purpose:** RTC needs the **original policy output** to compute leftovers for guidance. But the **robot needs denormalized actions**. ActionQueue holds both.

**Constructor:**
```python
ActionQueue(cfg: RTCConfig)
```

**Key state:**
- `queue: Tensor | None` — processed actions for robot [T, action_dim]
- `original_queue: Tensor | None` — original policy output [T, action_dim]
- `last_index: int` — consumption cursor

**Main method: `merge()`** (action_queue.py:137-163):
```python
merge(
    original_actions: Tensor,      # raw policy output
    processed_actions: Tensor,     # denormalized, ready for robot
    real_delay: int,               # timesteps consumed during inference
    action_index_before_inference: int | None = None
)
```

**Two modes:**

1. **RTC enabled** (`_replace_actions_queue`, line 165-183):
   - Discard the first `real_delay` actions (robot was executing while we inferred)
   - Replace entire queue with new chunk (stale actions are killed)
   - Reset `last_index = 0`
   - This is the "fast replan" behavior: the new chunk subsumes the tail of the old

2. **RTC disabled** (`_append_actions_queue`, line 185-204):
   - Append new actions to queue (maintain continuity)
   - First consume what's already been executed (via `last_index`)
   - This is incremental planning (useful for longer-horizon tasks)

**Get leftover for next RTC call:** `get_left_over()` (line 106-118) returns `original_queue[last_index:]` — the actions **not yet executed** that will become the guidance term in the next inference.

**Source:** `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/action_queue.py`

### 2.2 Postprocessor: From Normalized → Denormalized

LeRobot's action postprocessor (processor/normalize_processor.py + policy_robot_bridge.py) converts the policy's normalized action space into robot joint space.

**Normalization pipeline** (processor_xvla.py in XVLA, but similar for Pi0/Pi0.5):
1. **Preprocessing** (normalization): action space (e.g., EE6D, Joint) defines per-component scales
   - Pi0/Pi0.5 use normalization via `NormalizerProcessorStep`
   - Each action feature has `(mean, std)` from training data
   - `normalized_action = (action - mean) / std`
   - Source: `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/processor/normalize_processor.py`

2. **Postprocessing** (denormalization): reverse the scale
   - `denormalized_action = normalized_action * std + mean`
   - Source: `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/processor/normalize_processor.py:UnnormalizerProcessorStep`

3. **Action space postprocess** (e.g., EE6D.postprocess()): handle gripper logic
   - Apply sigmoid to gripper logits (convert unbounded ℝ → [0,1])
   - Trim padding if action dim is larger than real robot dim
   - Source: `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/xvla/action_hub.py:130-145` (EE6DActionSpace)

**Full denormalization chain:**
```
policy_output [B, T, action_dim]
    ↓ (UnnormalizerProcessorStep)
denormalized [B, T, action_dim]
    ↓ (ActionSpace.postprocess())
robot_ready [B, T, real_action_dim]
```

**Key for RTC:** The policy works in **normalized space**. The prefix guidance (`prev_chunk_left_over`) must also be in **normalized space**, so it aligns with the latent `x_t` that the denoiser sees. The denormalization happens only when sending to the robot.

### 2.3 ActionInterpolator: Higher Control Rate

Optional component (action_interpolator.py) that increases the effective control frequency.

**Use case:** Policy outputs actions at 10Hz (100ms/action), but robot can handle 50Hz (20ms). Interpolator produces intermediate actions by linear interpolation between consecutive policy outputs.

**Constructor:**
```python
ActionInterpolator(multiplier: int = 1)
```
- `multiplier=1` → no interpolation (use policy rate)
- `multiplier=2` → 2x rate (interpolate 1 extra action between each policy action)
- `multiplier=3` → 3x rate, etc.

**Usage in loop:**
```python
interpolator = ActionInterpolator(multiplier=3)
while True:
    if interpolator.needs_new_action():
        new_action = queue.get()
        if new_action:
            interpolator.add(new_action)
    
    action = interpolator.get()
    if action:
        robot.send(action)
```

**Source:** `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/action_interpolator.py`

**Important:** Interpolation happens **post-denormalization** (on robot actions), so it doesn't affect RTC guidance computation.

---

## Section 3: Edge Cases Worth Stealing

### 3.1 Chunk Underrun (Buffer Starve)

**Scenario:** The robot consumes actions faster than the server can generate new chunks (inference latency > execution time between replans).

**LeRobot handling:**
- ActionQueue tracks `last_index` (cursor)
- `get()` returns None if `last_index >= len(queue)`
- The serve loop should check for underrun and either:
  - Repeat the last valid action
  - Trigger an emergency replan (higher priority inference)
  - Fail safe (stop robot)

**Reflex analogy:** Our ActionChunkBuffer has `should_replan(threshold_ratio)`. We should **trigger replan before buffer drains to zero**, not after. See `/home/user/Desktop/building projects/reflex-vla/src/reflex/runtime/buffer.py:94-102`.

### 3.2 Inference Delay Tracking

**Problem:** Between when we start inference (`t0`) and when we finish (`t1`), the robot executes `Δt * fps` actions. If we call `merge(real_delay=Δt*fps)`, we skip exactly those stale actions. But if our delay estimate is off, we skip too many (robot lag) or too few (duplicate motion).

**LeRobot solution:**
- `LatencyTracker` (latency_tracker.py) records inference latencies
- Provides `.max()`, `.percentile(q)`, `.p95()` for adaptive delay estimation
- Source: `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/latency_tracker.py`

**ActionQueue validation:**
- `_check_and_resolve_delays()` (action_queue.py:206-228) compares expected vs. actual consumed actions
- Logs a warning if they diverge (drift detection)
- This catches bugs early

### 3.3 Gripper State Continuity

**Problem:** Gripper open/close is discrete. If the previous chunk predicted "gripper open" (0.9) and the new chunk predicts "gripper closed" (0.1), the boundary action must interpolate smoothly, otherwise the robot sees a jump instruction.

**LeRobot approach:**
- Gripper logic is handled in action spaces (EE6DActionSpace, JointActionSpace, etc.)
- Apply sigmoid only **after** guidance (in postprocess), so the guidance term operates on logits (unbounded ℝ)
- Prefix weighting ensures the first few actions of the new chunk are anchored to the old gripper state via guidance
- Source: `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/xvla/action_hub.py:123-145` (postprocess logic)

### 3.4 Execution Horizon < Chunk Size

**Problem:** If execution_horizon=10 but the new chunk is only 8 actions, we have nothing to guide after step 8.

**LeRobot mitigation:**
- modeling_rtc.py:177-179: clamp horizon to chunk size
  ```python
  if execution_horizon > prev_chunk_left_over.shape[1]:
      execution_horizon = prev_chunk_left_over.shape[1]
  ```
- Prefix weights naturally become 0 after the chunk ends anyway

### 3.5 Device/Dtype Safety

**Problem:** Policy might be on CUDA float32, robot control might need CPU float64 for precision. Mismatches cause silent errors.

**LeRobot patterns:**
- `get_safe_dtype()` (pi0/modeling_pi0.py:94-102) handles device-specific quirks (MPS doesn't support float64, etc.)
- Device ProcessorStep (normalize_processor.py) explicitly moves data
- ActionQueue uses `.to(device)` before returns
- Source: `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/pi0/modeling_pi0.py:94-102`

### 3.6 Debug Tracking Without Production Overhead

**Mechanism:** Optional `Tracker` object (debug_tracker.py) records denoising steps at runtime.

**Zero-cost when disabled:**
- `RTCProcessor.is_debug_enabled()` checks `if self.tracker is not None`
- All tracking calls are no-ops if tracker is None
- `@torch._dynamo.disable` prevents graph break in compiled models
- Source: `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/debug_tracker.py:92-97`

---

## Section 4: RtcAdapter Proposal for Reflex

### 4.1 Architecture

We wrap LeRobot's RTC components inside our serve runtime's `ReflexServer` and `ActionChunkBuffer`. The adapter is responsible for:

1. **Maintaining RTC state** (which chunk are we on? what actions are unconsumed?)
2. **Computing inference delay** (how many actions were consumed since last inference start?)
3. **Calling the policy with RTC kwargs**
4. **Merging original + processed actions** into our buffer
5. **Extracting leftovers** for the next RTC call

### 4.2 Proposed RtcAdapter Class

**Location:** `src/reflex/runtime/rtc_adapter.py` (new file)

```python
"""
RTC (Real-Time Chunking) adapter for Reflex VLA serve runtime.

Bridges LeRobot's RTC/ActionQueue/LatencyTracker with Reflex's
ActionChunkBuffer and Policy.predict_action_chunk() interface.

Responsibilities:
  - Manage RTC state (current chunk, leftovers, action indices)
  - Track inference latency and compute real_delay
  - Call policy.predict_action_chunk() with RTC kwargs
  - Merge original + processed action streams
  - Extract leftovers for next inference

Usage:
    rtc_adapter = RtcAdapter(
        policy=policy,
        rtc_config=RTCConfig(enabled=True, execution_horizon=10),
        action_buffer=buffer,
        fps=100.0
    )
    
    # In inference coroutine:
    inference_start_idx = action_buffer.get_action_index()
    t_start = time.monotonic()
    
    actions = rtc_adapter.predict_chunk_with_rtc(
        batch=obs_dict,
        action_index_before_inference=inference_start_idx
    )
    
    t_elapsed = time.monotonic() - t_start
    rtc_adapter.merge_and_update(
        actions=actions,
        elapsed_time=t_elapsed,
        action_index_before_inference=inference_start_idx
    )
    
    # In execution loop:
    action = action_buffer.pop_next()
    if action is None and action_buffer.should_replan(threshold=0.5):
        # Trigger emergency replan or fallback
        ...
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.rtc import (
    ActionInterpolator,
    ActionQueue,
    LatencyTracker,
    RTCConfig,
    RTCProcessor,
)
from lerobot.policies.rtc.debug_tracker import DebugStep

# Import the Reflex buffer (adjust path as needed)
from reflex.runtime.buffer import ActionChunkBuffer


@dataclass
class RtcAdapterConfig:
    """Configuration for the RTC adapter."""
    rtc_enabled: bool = True
    rtc_config: RTCConfig | None = None
    latency_tracker_maxlen: int = 100
    interpolation_multiplier: int = 1
    debug: bool = False


class RtcAdapter:
    """Bridges LeRobot RTC with Reflex serve runtime.
    
    This adapter:
      1. Maintains RTC processor, action queues, and latency tracking
      2. Computes inference_delay (actions consumed during inference)
      3. Extracts prev_chunk_left_over (unconsumed actions for guidance)
      4. Calls policy.predict_action_chunk() with RTC kwargs
      5. Merges original + processed actions, updates buffer
      6. Provides debug info (optional)
    """

    def __init__(
        self,
        policy: Any,  # Policy with predict_action_chunk(**kwargs)
        action_buffer: ActionChunkBuffer,
        config: RtcAdapterConfig | None = None,
        fps: float = 100.0,
        device: torch.device | str = "cuda",
    ):
        """Initialize RTC adapter.
        
        Args:
            policy: Policy object with predict_action_chunk(**kwargs).
            action_buffer: Reflex's ActionChunkBuffer to manage action queue.
            config: RtcAdapterConfig with RTCConfig and other settings.
            fps: Robot control frequency (Hz) for latency→action-delay conversion.
            device: Torch device for RTC processing.
        """
        self.policy = policy
        self.action_buffer = action_buffer
        self.fps = fps
        self.device = device
        
        if config is None:
            config = RtcAdapterConfig()
        self.config = config
        
        # Initialize RTC components
        self.rtc_processor: RTCProcessor | None = None
        self.action_queue: ActionQueue | None = None
        self.latency_tracker: LatencyTracker | None = None
        self.interpolator: ActionInterpolator | None = None
        
        if self.config.rtc_enabled and self.config.rtc_config:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
            self.action_queue = ActionQueue(self.config.rtc_config)
            self.latency_tracker = LatencyTracker(maxlen=config.latency_tracker_maxlen)
        
        if self.config.interpolation_multiplier > 1:
            self.interpolator = ActionInterpolator(multiplier=config.interpolation_multiplier)
        
        # Track last extraction index for debug
        self._last_extraction_index = 0

    def predict_chunk_with_rtc(
        self,
        batch: dict[str, Tensor],
        action_index_before_inference: int | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        """Call policy.predict_action_chunk() with RTC guidance.
        
        Args:
            batch: Observation batch (images, language, state, etc.)
            action_index_before_inference: ActionBuffer's get_action_index() before inference started.
                Used to compute real_delay = current_index - action_index_before_inference.
            execution_horizon: Override RTC execution_horizon if provided.
        
        Returns:
            actions: [batch_size, chunk_size, action_dim] predicted actions (denormalized, ready for robot).
        """
        # Compute inference delay (actions consumed while we were inferring)
        real_delay = 0
        if self.rtc_processor and action_index_before_inference is not None:
            current_index = self.action_buffer.get_action_index() if hasattr(self.action_buffer, 'get_action_index') else 0
            # If action_buffer is our ActionChunkBuffer, estimate from size change
            # For now, assume real_delay is passed explicitly or use latency-based estimate
            real_delay = max(0, self._estimate_delay_from_fps())
        
        # Extract prev_chunk_left_over from the action queue (original, normalized space)
        prev_chunk_left_over = None
        if self.rtc_processor and self.action_queue:
            prev_chunk_left_over = self.action_queue.get_left_over()
        
        # Call policy with RTC kwargs
        kwargs = {}
        if self.rtc_processor:
            kwargs.update({
                "inference_delay": real_delay,
                "prev_chunk_left_over": prev_chunk_left_over,
                "execution_horizon": execution_horizon,
            })
        
        with torch.no_grad():
            actions = self.policy.predict_action_chunk(batch, **kwargs)
        
        return actions  # [batch_size, chunk_size, action_dim], denormalized

    def merge_and_update(
        self,
        actions: Tensor,  # denormalized actions [B, T, A]
        elapsed_time: float,  # seconds spent in inference
        action_index_before_inference: int | None = None,
    ) -> None:
        """Merge actions into buffer and update RTC state.
        
        Args:
            actions: Denormalized actions from policy [batch_size, chunk_size, action_dim].
                We take only the first batch (assume batch_size=1 in serve).
            elapsed_time: Wall-clock time spent in predict_chunk_with_rtc.
            action_index_before_inference: For validation (optional).
        """
        if actions.ndim == 3:
            # Squeeze batch dimension (assume batch_size=1 in serve)
            actions = actions[0]  # [chunk_size, action_dim]
        
        # Track latency for future delay estimates
        if self.latency_tracker:
            self.latency_tracker.add(elapsed_time)
        
        # For now, we directly push to ActionChunkBuffer
        # In a more integrated setup, use ActionQueue.merge() instead
        self.action_buffer.push_chunk(actions.cpu().numpy(), overwrite_stale=True)

    def get_leftovers_for_next_inference(self) -> Tensor | None:
        """Extract unconsumed actions (original, normalized) for RTC guidance in next inference.
        
        Returns:
            Tensor: Unconsumed original actions [remaining_steps, action_dim], or None if no queue.
        """
        if self.action_queue:
            return self.action_queue.get_left_over()
        return None

    def _estimate_delay_from_fps(self) -> int:
        """Estimate real_delay (consumed action count) from fps and latency tracker.
        
        Returns:
            int: Estimated number of actions consumed during last inference.
        """
        if not self.latency_tracker or self.latency_tracker.max() is None:
            return 0
        
        # Use 95th percentile latency to be conservative
        p95_latency = self.latency_tracker.p95()
        if p95_latency is None:
            return 0
        
        actions_consumed = int(p95_latency * self.fps)
        return max(0, actions_consumed)

    def reset(self) -> None:
        """Reset RTC state (e.g., on episode boundary)."""
        if self.action_queue:
            self.action_queue.clear()
        if self.latency_tracker:
            self.latency_tracker.reset()
        if self.interpolator:
            self.interpolator.reset()
        if self.rtc_processor:
            self.rtc_processor.reset_tracker()

    def debug_info(self) -> dict[str, Any]:
        """Return debug snapshot."""
        info = {
            "rtc_enabled": self.config.rtc_enabled,
            "interpolation_multiplier": self.config.interpolation_multiplier,
        }
        
        if self.latency_tracker:
            info["latency_max_ms"] = (self.latency_tracker.max() * 1000) if self.latency_tracker.max() else 0
            info["latency_p95_ms"] = (self.latency_tracker.p95() * 1000) if self.latency_tracker.p95() else 0
        
        if self.action_queue:
            info["action_queue_size"] = self.action_queue.qsize()
        
        if self.rtc_processor and self.rtc_processor.is_debug_enabled():
            info["debug_steps"] = len(self.rtc_processor.get_all_debug_steps())
        
        return info
```

### 4.3 Integration Points with Reflex

**File:** `src/reflex/runtime/server.py` (existing)

**Where RtcAdapter hooks in:**

1. **Initialization** (in `ReflexServer.__init__`):
   ```python
   from reflex.runtime.rtc_adapter import RtcAdapter, RtcAdapterConfig
   
   if self.config.rtc_enabled:
       rtc_config = RTCConfig(
           enabled=True,
           execution_horizon=self.config.rtc_execution_horizon,
           max_guidance_weight=self.config.rtc_max_guidance_weight,
           prefix_attention_schedule=self.config.rtc_attention_schedule,
           debug=self.config.debug,
       )
       adapter_config = RtcAdapterConfig(
           rtc_enabled=True,
           rtc_config=rtc_config,
           interpolation_multiplier=self.config.interpolation_multiplier,
       )
       self.rtc_adapter = RtcAdapter(
           policy=self.policy,
           action_buffer=self.action_buffer,
           config=adapter_config,
           fps=self.execute_hz,
           device=self.device,
       )
   else:
       self.rtc_adapter = None
   ```

2. **Inference coroutine** (background replan task):
   ```python
   async def _replan_loop(self):
       while not self._stop:
           await asyncio.sleep(1.0 / self.replan_hz)
           
           # Record index before inference
           action_idx_before = self.action_buffer.get_action_index() if hasattr(...) else 0
           t_start = time.monotonic()
           
           # Predict (with RTC if enabled)
           if self.rtc_adapter:
               actions = self.rtc_adapter.predict_chunk_with_rtc(
                   batch=self.current_obs,
                   action_index_before_inference=action_idx_before,
               )
               elapsed = time.monotonic() - t_start
               self.rtc_adapter.merge_and_update(
                   actions=actions,
                   elapsed_time=elapsed,
                   action_index_before_inference=action_idx_before,
               )
           else:
               # Fallback: standard push_chunk without RTC
               actions = self.policy.predict_action_chunk(batch=self.current_obs)
               self.action_buffer.push_chunk(actions[0].cpu().numpy())
   ```

3. **Action execution** (on-demand /act endpoint):
   ```python
   def predict_single_action(self, obs_dict):
       """Pop from buffer; trigger replan if needed."""
       action = self.action_buffer.pop_next()
       
       if action is None:
           # Buffer starved — emergency behavior
           logger.warning("Action buffer underrun")
           action = self.last_action  # repeat last, or fail-safe
       
       if self.action_buffer.should_replan(threshold=self.replan_threshold):
           # Trigger background replan (scheduled async)
           self.replan_event.set()
       
       # Optional interpolation
       if self.rtc_adapter and self.rtc_adapter.interpolator:
           if self.rtc_adapter.interpolator.needs_new_action():
               self.rtc_adapter.interpolator.add(torch.from_numpy(action))
           action = self.rtc_adapter.interpolator.get()
       
       return action
   ```

### 4.4 Configuration Schema

**File:** `src/reflex/config.py` (additions)

```python
@dataclass
class ReflexRtcConfig:
    """RTC-specific configuration."""
    enabled: bool = True
    execution_horizon: int = 10
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: str = "LINEAR"  # LINEAR, ONES, ZEROS, EXP
    debug: bool = False
    latency_tracker_maxlen: int = 100

@dataclass
class ReflexServeConfig:
    """Main serve config."""
    ...
    rtc: ReflexRtcConfig = field(default_factory=ReflexRtcConfig)
    interpolation_multiplier: int = 1  # 1 = no interpolation, 2 = 2x, etc.
    ...
```

### 4.5 Testing Strategy

**File:** `tests/runtime/test_rtc_adapter.py` (new)

Key tests:
1. **RTC guidance reduces divergence**: Run 2 chunks with vs. without RTC, compare leftovers
2. **Latency tracking**: Inject variable delays, verify delay estimate
3. **Chunk underrun graceful handling**: Exhaust buffer, verify fallback behavior
4. **Denormalization round-trip**: Action → normalized → denormalized → check close
5. **Interpolation monotonicity**: Verify interpolated actions are monotonic
6. **Device safety**: Test dtype/device conversion
7. **Integration with actionqueue**: Merge original + processed, verify both are synced

---

## Summary for Implementation

### Must-Read Files (in order)

1. **RTC Core Math**
   - `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/modeling_rtc.py` (denoise_step, prefix_weights)
   - `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/configuration_rtc.py`

2. **Action Management**
   - `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/action_queue.py` (merge, get_left_over)
   - `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/action_interpolator.py`
   - `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/rtc/latency_tracker.py`

3. **Policy Integration**
   - `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/pi0/modeling_pi0.py:1262-1277` (predict_action_chunk)
   - `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/pi0/modeling_pi0.py:862-885` (RTC call site in sample_actions)

4. **Denormalization**
   - `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/processor/normalize_processor.py` (UnnormalizerProcessorStep)
   - `/home/user/Desktop/building projects/reflex-vla/reference/lerobot/src/lerobot/policies/xvla/action_hub.py` (action space postprocess)

5. **Existing Reflex Pattern**
   - `/home/user/Desktop/building projects/reflex-vla/src/reflex/runtime/buffer.py` (ActionChunkBuffer, understand interface)

### Key Takeaways

1. **RTC is guidance, not replacement**: The policy still runs full diffusion. RTC just modifies the velocity by adding a correction term.

2. **Prefix weighting is everything**: The weight schedule `[1,1,1,...,0,0,0]` (ones for old, zeros for new) determines which actions are anchored to the past vs. free to replan.

3. **Original + Processed streams**: The policy outputs **normalized actions**. RTC guidance operates in normalized space. Denormalization happens only for robot execution. Keep both streams synced via ActionQueue.

4. **Latency → real_delay conversion**: The robot consumes one action per `1/fps` seconds. If inference took `t_elapsed`, approximately `t_elapsed * fps` actions were consumed. Use this to skip stale actions.

5. **Graceful degradation**: If RTC is disabled or first chunk, the guidance term is 0 and RTC becomes a pass-through. No special handling needed.

6. **Gripper continuity**: By applying sigmoid **after** guidance (in postprocess), the gripper logits can change smoothly across chunks without jumps.

---

**Generated:** 2025-04-22
**Paper Reference:** https://arxiv.org/abs/2506.07339
**LeRobot Source Base:** `/home/user/Desktop/building projects/reflex-vla/reference/lerobot`
