"""vla-eval adapter for Tether VLA.

Exposes Tether's VLA inference (real VLM conditioning via the 4-file ONNX
pipeline) as a vla-evaluation-harness model server. Use with LIBERO,
SimplerEnv, ManiSkill, or any benchmark that speaks vla-eval.

Run:
    python -m tether.runtime.adapters.vla_eval \\
        --export-dir /path/to/tether_export/ --port 8000

Or set env vars and let vla-eval's run_server() parse flags:
    TETHER_EXPORT_DIR=/path/to/tether_export/
    TETHER_ACTION_DIM_OUT=7           # LIBERO=7, SimplerEnv=7, pi0=32
    TETHER_CAMERA_KEY=agentview       # default: first camera in obs['images']
    TETHER_DEVICE=cpu                 # cpu|cuda, default: cpu (sim is CPU-heavy)
    python -m tether.runtime.adapters.vla_eval --port 8000

Why this exists:
    TetherServer already owns the full SmolVLA inference pipeline including
    real VLM prefix conditioning (text_embedder.onnx + vision_encoder.onnx +
    decoder_prefill.onnx). The previous LIBERO benchmark shipped with an
    inline adapter that reimplemented the denoising loop and fed
    ``vlm_kv=zeros``, nulling out VLM conditioning and producing 0% task
    success. This adapter delegates everything to TetherServer, so the VLM
    pipeline is actually used.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "pick_image",
    "truncate_actions",
    "load_normalizer_stats",
    "build_adapter_class",
    "main",
]


def load_normalizer_stats(export_dir: "os.PathLike | str") -> dict[str, np.ndarray]:
    """Load MEAN_STD normalizer stats from the export dir.

    Looks for ``policy_preprocessor_*.safetensors`` and
    ``policy_postprocessor_*.safetensors`` (produced by LeRobot). Returns a
    dict with any of the following keys that were found:

        ``state_mean``, ``state_std``   — apply before inference:
            state_in = (state - state_mean) / (state_std + eps)

        ``action_mean``, ``action_std`` — apply after inference:
            action_out = action * action_std + action_mean

    Returns an empty dict if no normalizer files exist or the library can't
    be loaded. The adapter treats that as a no-op (raw pass-through).

    LeRobot's safetensors keys look like ``observation.state.mean``,
    ``observation.state.std``, ``action.mean``, ``action.std`` — this function
    reshapes those into the simpler flat names above.
    """
    from pathlib import Path

    export_dir = Path(export_dir)
    stats: dict[str, np.ndarray] = {}

    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning(
            "safetensors not available — skipping normalizer load (predictions "
            "will be in normalized space, likely producing garbage trajectories)"
        )
        return stats

    candidates = list(export_dir.glob("policy_preprocessor*.safetensors")) + list(
        export_dir.glob("policy_postprocessor*.safetensors")
    )
    if not candidates:
        return stats

    # Map lerobot's canonical key → our flat name
    key_map = {
        "observation.state.mean": "state_mean",
        "observation.state.std": "state_std",
        "action.mean": "action_mean",
        "action.std": "action_std",
    }

    for fpath in candidates:
        try:
            with safe_open(str(fpath), framework="numpy") as f:
                for k in f.keys():
                    # lerobot uses keys like "buffer.observation.state.mean" or
                    # just "observation.state.mean" — strip optional prefix
                    k_clean = k
                    for canonical, short in key_map.items():
                        if k_clean.endswith(canonical):
                            stats[short] = f.get_tensor(k).astype(np.float32)
                            break
        except Exception as e:
            logger.warning("Failed to load stats from %s: %s", fpath, e)

    found = sorted(stats.keys())
    if found:
        logger.info(
            "Loaded normalizer stats from %d file(s): %s",
            len(candidates),
            found,
        )
    return stats


# ---------------------------------------------------------------------------
# Pure helpers (testable without vla-eval installed)
# ---------------------------------------------------------------------------

def pick_image(obs: dict[str, Any], camera_key: str | None = None) -> np.ndarray | None:
    """Pick a single camera image from a vla-eval observation dict.

    Observations typically carry an ``images`` dict keyed by camera name
    (e.g. ``agentview`` or ``robot0_eye_in_hand`` for LIBERO). When
    ``camera_key`` is unset, picks the first camera in iteration order —
    stable on Python 3.7+.
    """
    images = obs.get("images") or {}
    if not images:
        return None
    if camera_key and camera_key in images:
        return np.asarray(images[camera_key])
    first_key = next(iter(images))
    return np.asarray(images[first_key])


def truncate_actions(actions: np.ndarray, target_dim: int) -> np.ndarray:
    """Truncate or zero-pad per-timestep action vectors to ``target_dim``.

    SmolVLA exports 32-dim actions; LIBERO wants 7 (6 joints + gripper).
    SimplerEnv, ManiSkill etc. each define their own dim. Padding with zeros
    when the model has fewer dims than the benchmark expects is unusual but
    safe — the env just ignores the extra DOFs.
    """
    if actions.ndim == 1:
        actions = actions[np.newaxis, :]
    current = actions.shape[-1]
    if current == target_dim:
        return actions
    if current > target_dim:
        return actions[..., :target_dim]
    pad_shape = actions.shape[:-1] + (target_dim - current,)
    pad = np.zeros(pad_shape, dtype=actions.dtype)
    return np.concatenate([actions, pad], axis=-1)


# ---------------------------------------------------------------------------
# Adapter class — built lazily so import doesn't require vla-eval
# ---------------------------------------------------------------------------

def build_adapter_class():
    """Build and return the vla-eval-compatible adapter class.

    Deferred-import pattern: vla-eval's base class is only imported when the
    adapter is actually built, so importing this module (for the pure helpers
    or for introspection) does not require ``fastcrest-tether[eval]``.
    """
    try:
        from vla_eval.model_servers.predict import PredictModelServer
    except ImportError:
        try:
            from vla_eval.model_servers.base import ModelServer as PredictModelServer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "vla-eval is not installed. Install with:\n"
                "    pip install 'fastcrest-tether[eval]'\n"
                "or\n"
                "    pip install vla-eval"
            ) from e

    class TetherVlaEvalAdapter(PredictModelServer):
        """Serves tether.runtime.TetherServer via vla-eval's protocol."""

        def __init__(
            self,
            export_dir: str | None = None,
            action_dim_out: int | None = None,
            camera_key: str | None = None,
            device: str | None = None,
            num_denoising_steps: int = 10,
            **kwargs,
        ):
            # vla-eval auto-injects parent-class kwargs (chunk_size, etc.);
            # pass them through so super().__init__ doesn't blow up.
            super().__init__(**kwargs)

            from tether.runtime.server import TetherServer

            export_dir = export_dir or os.environ.get("TETHER_EXPORT_DIR")
            if not export_dir:
                raise ValueError(
                    "TetherVlaEvalAdapter needs export_dir. Pass --export-dir "
                    "or set TETHER_EXPORT_DIR."
                )

            action_dim_out = action_dim_out or int(
                os.environ.get("TETHER_ACTION_DIM_OUT", "7")
            )
            camera_key = camera_key or os.environ.get("TETHER_CAMERA_KEY") or None
            device = device or os.environ.get("TETHER_DEVICE", "cpu")

            self._action_dim_out = action_dim_out
            self._camera_key = camera_key

            # strict_providers=False: sim doesn't need GPU; silently falling
            # back to CPU is fine here (unlike production benchmarking).
            # Route to native PyTorch path when TETHER_NATIVE=1. The native
            # server handles preprocess + postprocess internally so the adapter
            # doesn't need to apply its own normalizer.
            if os.environ.get("TETHER_NATIVE", "0") == "1":
                from tether.runtime.smolvla_native import SmolVLANativeServer
                self._server = SmolVLANativeServer(
                    export_dir,
                    device=device,
                    strict_providers=False,
                    num_denoising_steps=num_denoising_steps,
                )
                self._native_mode = True
            else:
                self._server = TetherServer(
                    export_dir,
                    device=device,
                    num_denoising_steps=num_denoising_steps,
                    strict_providers=False,
                )
                self._native_mode = False
            self._server.load()

            # Load LeRobot MEAN_STD stats for state-in / action-out normalization.
            # Without these, LIBERO etc. receive actions in normalized space and
            # the robot flails. Empty dict == no-op.
            self._norm_stats = load_normalizer_stats(export_dir)

            # VLM conditioning state:
            #  - native mode: real SmolVLAPolicy runs inside the server;
            #    VLM is "real" (uses the lerobot VLM pipe internally).
            #  - decomposed mode: TetherServer's _vlm_loaded flag tracks
            #    whether the 4-file ONNX VLM chain was loaded successfully.
            # The prior "vlm=off" in native mode was a false signal — native
            # always has VLM via lerobot's internals. Fixed 2026-04-19.
            if getattr(self, "_native_mode", False):
                vlm_state = "real (lerobot)"
            elif getattr(self._server, "_vlm_loaded", False):
                vlm_state = "on (decomposed)"
            else:
                vlm_state = "off (decomposed/stubbed)"
            logger.info(
                "TetherVlaEvalAdapter ready: export=%s device=%s out_dim=%d "
                "camera=%s vlm=%s norm=%s",
                export_dir,
                device,
                action_dim_out,
                camera_key or "<first>",
                vlm_state,
                "on" if self._norm_stats else "off",
            )

        def predict(self, obs: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
            """Run one inference pass. ``obs`` is a vla-eval observation dict."""
            # One-shot obs schema dump on first call — invaluable when the
            # model silently fails because the benchmark's obs keys don't
            # match what the checkpoint was trained on (e.g. SmolVLA expects
            # camera1/2/3, LIBERO emits agentview_image + robot0_eye_in_hand).
            if not getattr(self, "_obs_logged", False):
                self._obs_logged = True
                keys = sorted(obs.keys())
                img_info = "none"
                if "images" in obs and isinstance(obs["images"], dict):
                    img_entries = []
                    for k, v in obs["images"].items():
                        arr = np.asarray(v) if not isinstance(v, np.ndarray) else v
                        img_entries.append(f"{k}:{arr.shape}/{arr.dtype}")
                    img_info = ", ".join(img_entries) or "empty"
                state_raw = obs.get("states") if "states" in obs else obs.get("state")
                state_info = "none"
                if state_raw is not None:
                    sarr = np.asarray(state_raw)
                    state_info = f"{sarr.shape}/{sarr.dtype}"
                logger.info(
                    "First predict() obs schema: top_keys=%s images=%s state=%s "
                    "task=%r",
                    keys,
                    img_info,
                    state_info,
                    (obs.get("task_description") or obs.get("instruction") or "")[:80],
                )

            # Pass ALL camera views when available. SmolVLA-LIBERO was trained
            # on camera1/2/3 (three views); feeding only one produces OOD
            # prefix token distribution. Passing whatever is in obs["images"]
            # is closer to training — falls back to a single view or None.
            images_dict = obs.get("images") or {}
            if self._camera_key and self._camera_key in images_dict:
                # Explicit single-camera override
                image = np.asarray(images_dict[self._camera_key])
            elif images_dict:
                # Multi-camera: pass list in a deterministic order
                image = [np.asarray(v) for v in images_dict.values()]
            else:
                image = None
            instruction = (
                obs.get("task_description")
                or obs.get("instruction")
                or ""
            )
            # LeRobot's NewLineTaskProcessorStep appends \n to task strings
            # before tokenization (matches how SmolVLM was pre-trained to
            # format prompts). Our adapter must do the same or token IDs
            # drift from training distribution — shifting text_embeds,
            # then VLM prefix, then per-layer k/v, then expert cross-attn.
            if instruction and not instruction.endswith("\n"):
                instruction = instruction + "\n"
            # LIBERO obs has both "states" (raw env obs) and "controller_states"
            # (from the robot controller). LeRobot SmolVLA training on
            # lerobot/libero dataset uses the controller output, so prefer that
            # when available. Set TETHER_STATE_KEY to override.
            # Note: can't use `a or b or c` because obs values are numpy arrays
            # and Python `or` evaluates truthiness (ambiguous for arrays).
            pref_key = os.environ.get("TETHER_STATE_KEY", "controller_states")
            state_raw = None
            for k in (pref_key, "states", "state"):
                v = obs.get(k)
                if v is not None:
                    state_raw = v
                    break
            state = (
                np.asarray(state_raw, dtype=np.float32)
                if state_raw is not None
                else None
            )

            # Normalize state: state_norm = (state - mean) / (std + eps).
            # Model was trained on normalized state; raw input produces garbage.
            # Skip when using native path — the server's preprocessor handles
            # normalization internally.
            if (
                not getattr(self, "_native_mode", False)
                and state is not None
                and "state_mean" in self._norm_stats
                and "state_std" in self._norm_stats
            ):
                s_mean = self._norm_stats["state_mean"]
                s_std = self._norm_stats["state_std"]
                # Pad state up to stats length if obs is narrower than training dim
                if state.shape[-1] < s_mean.shape[-1]:
                    pad = np.zeros(
                        s_mean.shape[-1] - state.shape[-1], dtype=np.float32
                    )
                    state = np.concatenate([state, pad])
                elif state.shape[-1] > s_mean.shape[-1]:
                    state = state[: s_mean.shape[-1]]
                state = (state - s_mean) / (s_std + 1e-8)

            result = self._server.predict(
                image=image,
                instruction=instruction,
                state=state,
            )
            if "error" in result:
                raise RuntimeError(f"TetherServer error: {result['error']}")

            actions = np.asarray(result["actions"], dtype=np.float32)  # [chunk, 32]
            actions = truncate_actions(actions, self._action_dim_out)  # [chunk, 7]

            # Unnormalize actions: action_real = action_norm * std + mean.
            # Skip when using native path — server's postprocessor already
            # unnormalized via the real lerobot pipeline.
            if (
                not getattr(self, "_native_mode", False)
                and "action_mean" in self._norm_stats
                and "action_std" in self._norm_stats
            ):
                a_mean = self._norm_stats["action_mean"]
                a_std = self._norm_stats["action_std"]
                # Align dims: truncate a_mean/a_std to action_dim_out if needed
                a_mean = a_mean[: self._action_dim_out]
                a_std = a_std[: self._action_dim_out]
                actions = actions * a_std + a_mean

            # One-shot action dump so we can sanity-check magnitudes against the
            # normalizer stats (if actions are way out of std*N range, something
            # upstream is wrong — wrong vlm_kv, wrong state, wrong flow dt).
            if not getattr(self, "_action_logged", False):
                self._action_logged = True
                a = np.asarray(actions)
                logger.info(
                    "First predict actions: shape=%s  chunk_mean=%+.3f  "
                    "chunk_std=%+.3f  first_action=%s  a_mean_stat=%s  "
                    "a_std_stat=%s",
                    a.shape,
                    float(a.mean()),
                    float(a.std()),
                    np.round(a[0], 3).tolist(),
                    np.round(self._norm_stats.get("action_mean", [0] * 7)[:7], 3).tolist(),
                    np.round(self._norm_stats.get("action_std", [1] * 7)[:7], 3).tolist(),
                )

            # LIBERO / most vla-eval benchmarks query once per env step and
            # expect a SINGLE action vector (not the full chunk). Return the
            # first action; caller re-queries with fresh obs next step. Set
            # TETHER_RETURN_CHUNK=1 to return the full chunk for benchmarks
            # that natively support action chunking.
            if os.environ.get("TETHER_RETURN_CHUNK", "0") == "1":
                return {"actions": actions}
            return {"actions": actions[0]}  # shape: [action_dim_out]

    return TetherVlaEvalAdapter


def main() -> None:
    """Entry point for ``python -m tether.runtime.adapters.vla_eval``."""
    try:
        from vla_eval.model_servers.serve import run_server
    except ImportError as e:
        raise ImportError(
            "vla-eval is not installed. Install with:\n"
            "    pip install 'fastcrest-tether[eval]'"
        ) from e

    adapter_cls = build_adapter_class()

    if "--port" not in sys.argv:
        sys.argv.extend(["--port", "8000"])
    run_server(adapter_cls)


if __name__ == "__main__":
    main()
