"""Pi05DecomposedInference — run pi0.5 through the decomposed
``vlm_prefix.onnx`` + ``expert_denoise.onnx`` pair, with optional
cross-query VLM prefix cache.

Design doc: ``reflex_context/reflex_vla/01_architecture/prefix_kv_cache_reuse_design.md``

The cache hashes the VLM input (images + language tokens) per call and
reuses the last ``past_kv`` output when the hash matches inside a
staleness window. During cache-hit the expensive VLM forward (90% of
compute for a 1-NFE student) is skipped; only the tiny expert-denoise
graph runs.

Callers typically use this through:

- the LIBERO-eval harness via ``--decomposed-dir`` flag, or
- ``reflex serve <export_dir>`` when ``reflex_config.json`` declares
  ``"export_kind": "decomposed"`` (wiring lives in
  ``reflex.runtime.server``).

The class exposes the same ``predict_action_chunk`` contract as
``Pi0OnnxServer.predict`` so downstream harnesses don't need to know
which export pattern is active.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """One VLM output we keep around for potential reuse."""
    past_kv: list[np.ndarray]            # flat [k_0, v_0, ..., k_17, v_17]
    prefix_pad_masks: np.ndarray
    image_phashes: tuple[bytes, ...]     # per-camera perceptual hash
    lang_hash: bytes                     # exact md5 of language tokens
    timestamp: float                     # wall-clock time (for TTL-sec path)
    step_index: int                      # predict_action_chunk call number (for step-count path)


@dataclass
class ActionCacheEntry:
    """One action chunk cached on (image_phashes, lang_hash). Keyed on
    the VLM-input side only, noise is re-sampled stochastically on
    every call so the same observation can produce different actions
    across calls — but for a SnapFlow 1-NFE student at target_time=1
    the noise dependence is minimal, so reusing a cached chunk on a
    matching obs is effectively zero-cost."""
    image_phashes: tuple[bytes, ...]
    lang_hash: bytes
    step_index: int
    actions: np.ndarray


@dataclass
class CacheStats:
    """Cumulative cache metrics — exposed via get_stats() so callers
    can log hit rate alongside LIBERO task success."""
    hits: int = 0
    misses: int = 0
    evictions_ttl: int = 0
    evictions_lang: int = 0
    evictions_phash: int = 0
    # action-chunk cache (separate layer, stats tracked independently)
    action_hits: int = 0
    action_misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    @property
    def action_total(self) -> int:
        return self.action_hits + self.action_misses

    @property
    def action_hit_rate(self) -> float:
        return self.action_hits / self.action_total if self.action_total > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": self.total,
            "hit_rate": self.hit_rate,
            "evictions_ttl": self.evictions_ttl,
            "evictions_lang": self.evictions_lang,
            "evictions_phash": self.evictions_phash,
            "action_hits": self.action_hits,
            "action_misses": self.action_misses,
            "action_total": self.action_total,
            "action_hit_rate": self.action_hit_rate,
        }


class Pi05DecomposedInference:
    """Two-ONNX pi0.5 inference with optional prefix-cache reuse.

    Parameters
    ----------
    export_dir
        Directory containing ``vlm_prefix.onnx``, ``expert_denoise.onnx``,
        and ``reflex_config.json`` (written by ``export_pi05_decomposed``).
    providers
        onnxruntime execution providers (default CPU).
    enable_cache
        When False, every call runs the VLM. When True, the cache matches
        on perceptual-image-hash + exact-language-hash + TTL.
    cache_ttl_sec
        Seconds of wall-clock after which a cache entry is considered
        stale. Default 0.2s — good for 20-50 Hz online deployment where
        0.2s = 4-10 real frames. For offline eval (LIBERO, <1 Hz) use
        ``cache_max_age_steps`` instead since wall-clock is meaningless.
    cache_max_age_steps
        Alternative staleness check: expire cache entry after this many
        ``predict_action_chunk`` calls regardless of wall-clock. 0 =
        disabled (fall back to cache_ttl_sec). Recommended: 3-5 for
        offline eval.
    phash_hamming_threshold
        Per-image perceptual-hash distance allowed for a cache hit.
        Default 6 — tuned for typical manipulation sensor noise (tune
        per-deployment via telemetry).
    """

    PHASH_SIZE: int = 8  # 8x8 phash → 64-bit hash → hamming ≤ 64

    def __init__(
        self,
        export_dir: str | Path,
        providers: list[str] | None = None,
        enable_cache: bool = True,
        cache_ttl_sec: float = 0.2,
        cache_max_age_steps: int = 0,
        phash_hamming_threshold: int = 6,
        cache_level: str = "prefix",
        action_cache_max_age_steps: int = 2,
        cache_ignore_lang: bool = False,
        episode_cache_max_episodes: int = 8,
        *,
        cuda_graphs_enabled: bool = False,
        cuda_graphs_embodiment: str = "unknown",
        cuda_graphs_model_id: str = "unknown",
        action_similarity_threshold: float = 0.0,  # 0.0 = disabled
        max_similar_skips: int = 3,
    ):
        """``cache_level`` controls which layer is cached:

        - ``"none"``: every call runs VLM + expert.
        - ``"prefix"`` (default): VLM prefix cached on (phash, lang);
          expert always runs. Works best for VLAs with stable lang
          across frames (pi0 with explicit state input, SmolVLA).
        - ``"action"``: final action chunk cached on (phash, lang).
          Skips BOTH VLM and expert on hit. Works regardless of
          state-in-language — the hash captures all VLA input state
          that affects the output. Designed for state-in-language
          VLAs like pi0.5 where prefix caching doesn't hit in
          production. ``action_cache_max_age_steps`` bounds how many
          calls a stale cached chunk can be reused across.
        - ``"episode"``: VLM prefix cached on (episode_id, lang_hash);
          expert always runs. Image is IGNORED in the key — within an
          episode the image moves but lang is stable (after the
          state-out preprocessor swap). Hits ~99%% within an episode
          → ~9x per-chunk speedup (validated by latency microbench).
          Requires ``episode_id`` arg to ``predict_action_chunk``.
          Designed for v0.5 state-out pi0.5 students. THE MOAT.

        """
        import onnxruntime as ort

        self.export_dir = Path(export_dir)
        self.enable_cache = enable_cache
        self.cache_ttl_sec = cache_ttl_sec
        self.cache_max_age_steps = cache_max_age_steps
        self.phash_hamming_threshold = phash_hamming_threshold
        if cache_level not in ("none", "prefix", "action", "episode"):
            raise ValueError(
                f"cache_level must be 'none'|'prefix'|'action'|'episode', got {cache_level!r}"
            )
        self.cache_level = cache_level
        self.action_cache_max_age_steps = action_cache_max_age_steps
        # cache_ignore_lang: bypass lang_hash check for state-in-language
        # VLAs (pi0.5). Safe when reset_cache() is called between
        # episodes (prevents cross-task cache collisions). Required
        # for pi0.5 action-chunk cache to actually hit in production.
        self.cache_ignore_lang = cache_ignore_lang
        self._call_index: int = 0  # monotonic call counter for step-count TTL
        self._action_cache: ActionCacheEntry | None = None
        # Action-similarity fast path (FlashVLA, Phase 1.5). Disabled when
        # threshold == 0; enabled when threshold > 0. Defaults to off so
        # existing behavior is unchanged unless --action-similarity-threshold
        # is set on the CLI.
        from reflex.runtime.action_fast_path import ActionFastPath
        self._fast_path = ActionFastPath(
            threshold=float(action_similarity_threshold),
            max_skips=int(max_similar_skips),
            enabled=action_similarity_threshold > 0,
        )
        # Default prefers CUDA when available, falls back to CPU if the
        # runtime doesn't have GPU providers. LIBERO eval on an A100 box
        # runs ~50× faster on GPU; only use CPU explicitly when matching
        # PyTorch reference bytes (parity tests).
        #
        # cuda_graphs_enabled=True overrides the providers list to include
        # enable_cuda_graph=1 on the CUDA Execution Provider (Phase 1
        # cuda-graphs feature; see ADR 2026-04-24-cuda-graphs-architecture).
        self._cuda_graphs_enabled = cuda_graphs_enabled
        self._cuda_graphs_embodiment = cuda_graphs_embodiment
        self._cuda_graphs_model_id = cuda_graphs_model_id
        if cuda_graphs_enabled:
            from reflex.runtime.cuda_graphs import build_cuda_graph_providers
            if providers is not None:
                logger.warning(
                    "cuda_graphs_enabled=True overrides user-provided providers=%s",
                    providers,
                )
            self._providers = build_cuda_graph_providers(enabled=True)
        else:
            self._providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]

        cfg_path = self.export_dir / "reflex_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"reflex_config.json missing in {self.export_dir}")
        self.config: dict[str, Any] = json.loads(cfg_path.read_text())
        if self.config.get("export_kind") != "decomposed":
            raise ValueError(
                f"{cfg_path} has export_kind={self.config.get('export_kind')!r}; "
                "Pi05DecomposedInference requires 'decomposed'"
            )
        self._past_kv_names: list[str] = self.config["decomposed"]["past_kv_tensor_names"]
        self._n_layers: int = self.config["decomposed"]["paligemma_layers"]

        # Per-step expert detection. When True, expert_denoise.onnx takes
        # (x_t, t, past_kv) → v_t and we drive the Euler loop in Python.
        # When False (default), it's the baked-loop shape (noise → actions,
        # single call). Spec: features/03_export/per-step-expert-export.md.
        self._per_step_expert: bool = bool(
            self.config["decomposed"].get("per_step_expert", False)
        )
        self._num_steps: int = int(self.config.get("num_denoising_steps", 1))
        if self._per_step_expert:
            logger.info(
                "Pi05DecomposedInference: per-step expert path active "
                "(num_steps=%d, Python Euler loop)", self._num_steps,
            )

        prefix_path = self.export_dir / self.config["decomposed"]["vlm_prefix_onnx"]
        expert_path = self.export_dir / self.config["decomposed"]["expert_denoise_onnx"]
        logger.info("ONNXRuntime available providers: %s", ort.get_available_providers())
        logger.info("ONNXRuntime device: %s", ort.get_device())
        logger.info("requested providers: %s", self._providers)

        # When cuda_graphs_enabled: use try_capture_or_fall_back() per session.
        # This probes capture at init time. On success → CudaGraphWrapper wrapping
        # the captured session. On failure (e.g., OOM on A10G for vlm_prefix) →
        # EagerSessionWrapper wrapping a fresh eager session, with a
        # capture_failed_at_init metric + warning log. Expert_denoise still
        # captures cleanly on both A10G + A100; vlm_prefix captures on A100+
        # only (per ADR 2026-04-24 Day-0 spike findings).
        cuda_required = self._providers[0] == "CUDAExecutionProvider" or (
            isinstance(self._providers[0], tuple) and self._providers[0][0] == "CUDAExecutionProvider"
        )

        if cuda_graphs_enabled:
            from reflex.runtime.cuda_graphs import (
                build_cuda_graph_providers,
                try_capture_or_fall_back,
            )

            def _build_prefix_session(cg_enabled: bool) -> ort.InferenceSession:
                return ort.InferenceSession(
                    str(prefix_path),
                    providers=build_cuda_graph_providers(enabled=cg_enabled),
                )

            def _build_expert_session(cg_enabled: bool) -> ort.InferenceSession:
                return ort.InferenceSession(
                    str(expert_path),
                    providers=build_cuda_graph_providers(enabled=cg_enabled),
                )

            logger.info("loading vlm_prefix with capture probe: %s", prefix_path)
            self._sess_prefix = try_capture_or_fall_back(
                _build_prefix_session,
                session_name="vlm_prefix",
                embodiment=cuda_graphs_embodiment,
                model_id=cuda_graphs_model_id,
            )
            _raw_prefix = self._sess_prefix.session
            actual_prefix = _raw_prefix.get_providers()
            logger.info("vlm_prefix actual providers: %s", actual_prefix)
            if cuda_required and "CUDAExecutionProvider" not in actual_prefix:
                raise RuntimeError(
                    f"cuda_graphs_enabled=True but CUDAExecutionProvider NOT active "
                    f"for vlm_prefix (actual={actual_prefix}). Check onnxruntime-gpu + "
                    f"cuDNN + cuBLAS libs in LD_LIBRARY_PATH."
                )

            logger.info("loading expert_denoise with capture probe: %s", expert_path)
            self._sess_expert = try_capture_or_fall_back(
                _build_expert_session,
                session_name="expert_denoise",
                embodiment=cuda_graphs_embodiment,
                model_id=cuda_graphs_model_id,
            )
            _raw_expert = self._sess_expert.session
            actual_expert = _raw_expert.get_providers()
            logger.info("expert_denoise actual providers: %s", actual_expert)
            if cuda_required and "CUDAExecutionProvider" not in actual_expert:
                raise RuntimeError(
                    f"cuda_graphs_enabled=True but CUDAExecutionProvider NOT active "
                    f"for expert_denoise (actual={actual_expert})"
                )

            logger.info(
                "cuda-graphs enabled: vlm_prefix.captured=%s, expert_denoise.captured=%s "
                "(embodiment=%s, model_id=%s)",
                self._sess_prefix.captured, self._sess_expert.captured,
                cuda_graphs_embodiment, cuda_graphs_model_id,
            )
        else:
            logger.info("loading vlm_prefix: %s", prefix_path)
            _raw_prefix = ort.InferenceSession(str(prefix_path), providers=self._providers)
            actual_prefix = _raw_prefix.get_providers()
            logger.info("vlm_prefix actual providers: %s", actual_prefix)
            if cuda_required and "CUDAExecutionProvider" not in actual_prefix:
                logger.warning(
                    "CUDAExecutionProvider requested but NOT active for vlm_prefix "
                    "(actual=%s). Check onnxruntime-gpu + cuDNN + cuBLAS libs in "
                    "LD_LIBRARY_PATH.",
                    actual_prefix,
                )
            self._sess_prefix = _raw_prefix

            logger.info("loading expert_denoise: %s", expert_path)
            _raw_expert = ort.InferenceSession(str(expert_path), providers=self._providers)
            actual_expert = _raw_expert.get_providers()
            logger.info("expert_denoise actual providers: %s", actual_expert)
            self._sess_expert = _raw_expert

        # Capture input/output name metadata from the underlying session
        # (works for both raw sessions and wrapped sessions).
        self._prefix_input_names = [i.name for i in _raw_prefix.get_inputs()]
        self._prefix_output_names = [o.name for o in _raw_prefix.get_outputs()]
        self._expert_input_names = [i.name for i in _raw_expert.get_inputs()]

        self._cache: CacheEntry | None = None
        self._stats = CacheStats()

        # Episode cache — instantiated only when cache_level='episode'.
        # Lives alongside the single-slot _cache rather than replacing it
        # so the existing phash/action modes are untouched.
        self._episode_cache = None
        if cache_level == "episode":
            from reflex.runtime.episode_cache import EpisodeCache
            self._episode_cache = EpisodeCache(max_episodes=episode_cache_max_episodes)
            logger.info(
                "[decomposed] episode cache enabled (max_episodes=%d)",
                episode_cache_max_episodes,
            )

    # ---- Public API -------------------------------------------------

    def predict_action_chunk(
        self,
        *,
        img_base: np.ndarray,
        img_wrist_l: np.ndarray,
        img_wrist_r: np.ndarray,
        mask_base: np.ndarray,
        mask_wrist_l: np.ndarray,
        mask_wrist_r: np.ndarray,
        lang_tokens: np.ndarray,
        lang_masks: np.ndarray,
        noise: np.ndarray,
        state: np.ndarray | None = None,
        episode_id: str | None = None,
    ) -> np.ndarray:
        """Run one pi0.5 forward, returning ``actions`` of shape
        ``(B, chunk_size, action_dim)``.

        Uses the prefix cache when enabled + hashes match + TTL valid.
        When ``cache_level='action'``, skips both VLM + expert when the
        (image_phashes, lang_hash) key matches a recent cached chunk.
        When ``cache_level='episode'``, caches VLM prefix per
        ``episode_id`` + lang_hash (image IGNORED); requires caller to
        pass a stable ``episode_id`` for the duration of one episode.
        Returns float32 regardless of the ONNX internal dtype."""
        self._call_index += 1

        # Episode cache path — short-circuits the existing phash lookup.
        # Image is NOT hashed here; key is (episode_id, lang_hash) only.
        if self.cache_level == "episode":
            if episode_id is None:
                raise ValueError(
                    "cache_level='episode' requires episode_id=<str> to "
                    "predict_action_chunk. Use a stable id for the duration "
                    "of one episode."
                )
            return self._predict_with_episode_cache(
                episode_id=episode_id,
                img_base=img_base, img_wrist_l=img_wrist_l, img_wrist_r=img_wrist_r,
                mask_base=mask_base, mask_wrist_l=mask_wrist_l, mask_wrist_r=mask_wrist_r,
                lang_tokens=lang_tokens, lang_masks=lang_masks,
                noise=noise, state=state,
            )

        image_phashes = (
            self._phash(img_base),
            self._phash(img_wrist_l),
            self._phash(img_wrist_r),
        )
        lang_hash = self._lang_hash(lang_tokens)

        # ---- Action-chunk cache (skip full forward on hit) --------------
        if self.cache_level == "action" and self._action_cache is not None:
            entry = self._action_cache
            steps_since = self._call_index - entry.step_index
            lang_ok = self.cache_ignore_lang or entry.lang_hash == lang_hash
            if (steps_since <= self.action_cache_max_age_steps
                and lang_ok
                and self._phashes_match(entry.image_phashes, image_phashes)):
                self._stats.action_hits += 1
                return entry.actions
        if self.cache_level == "action":
            self._stats.action_misses += 1

        past_kv, prefix_pad = self._get_or_run_prefix(
            img_base=img_base,
            img_wrist_l=img_wrist_l,
            img_wrist_r=img_wrist_r,
            mask_base=mask_base,
            mask_wrist_l=mask_wrist_l,
            mask_wrist_r=mask_wrist_r,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            image_phashes=image_phashes,
            lang_hash=lang_hash,
        )

        expert_feed_base = {name: past_kv[i] for i, name in enumerate(self._past_kv_names)}
        expert_feed_base["prefix_pad_masks"] = prefix_pad
        # v0.5 state-out: expert ONNX has a 'state' input. Pad to
        # max_state_dim if caller passed a shorter vector.
        if self.config.get("decomposed", {}).get("expert_takes_state"):
            if state is None:
                raise ValueError(
                    "decomposed export was built with expert_takes_state=True; "
                    "predict_action_chunk requires state=<np.ndarray>"
                )
            state_arr = state.astype(np.float32, copy=False)
            # state_proj input dim is max_state_dim (32 for pi0.5)
            expected_dim = next(
                i.shape[-1] for i in self._sess_expert.get_inputs() if i.name == "state"
            )
            if isinstance(expected_dim, int) and state_arr.shape[-1] < expected_dim:
                pad = np.zeros(
                    state_arr.shape[:-1] + (expected_dim - state_arr.shape[-1],),
                    dtype=state_arr.dtype,
                )
                state_arr = np.concatenate([state_arr, pad], axis=-1)
            expert_feed_base["state"] = state_arr

        # ---- Action-similarity fast path (FlashVLA, Phase 1.5) -----------
        # Skip the expert when the previous chunk was L2-similar enough +
        # we still have skip budget. PRE-A2C2 cache per spec — A2C2 hooks
        # outside this method recompute corrections on the reused actions.
        if self._fast_path.should_skip():
            cached = self._fast_path.cached_actions()
            if cached is not None:
                self._fast_path.consume_skip()
                # Surface skip to the per-flush metric. Defensive import
                # so unit tests without the prometheus extra don't fail.
                try:
                    from reflex.observability.prometheus import inc_action_skip
                    inc_action_skip()
                except Exception:  # noqa: BLE001
                    pass
                return cached

        actions = self._run_expert(expert_feed_base, noise)
        # Update fast-path tracker on each real expert call.
        self._fast_path.observe(actions)

        # ---- Populate action cache -------------------------------------
        if self.cache_level == "action":
            self._action_cache = ActionCacheEntry(
                image_phashes=image_phashes,
                lang_hash=lang_hash,
                step_index=self._call_index,
                actions=actions,
            )

        return actions

    def _run_expert(
        self,
        expert_feed_base: dict[str, np.ndarray],
        noise: np.ndarray,
    ) -> np.ndarray:
        """Dispatch to either the baked-loop expert (single ORT call returning
        actions) or the per-step expert (N ORT calls in a Python Euler loop,
        returning velocity per step). Choice is made at export time and
        recorded in ``reflex_config.json:decomposed.per_step_expert``.

        See features/03_export/per-step-expert-export.md for the contract.
        """
        if not self._per_step_expert:
            # Baked-loop path (the shipped default).
            expert_feed = dict(expert_feed_base)
            expert_feed["noise"] = noise.astype(np.float32, copy=False)
            actions = self._sess_expert.run(["actions"], expert_feed)[0]
            if actions.dtype != np.float32:
                actions = actions.astype(np.float32)
            return actions

        # Per-step path: drive the Euler loop in Python, calling the ONNX
        # once per step. Matches OpenPI's externalized loop pattern (see
        # research sidecar Lens 1).
        #
        # IOBinding is load-bearing here. The naive `sess.run(feed_dict)`
        # path forces ORT to re-copy past_kvs (~140 MB across 38 tensors)
        # host->device on every Euler iter — measured at +36.4% chunk
        # overhead vs baked, failing gate 4. With IOBinding we pin past_kvs
        # + prefix_pad_masks (and state if present) to device ONCE per chunk
        # via OrtValue.ortvalue_from_numpy(arr, "cuda", 0); only x_t (~6 KB)
        # and t (4 B) cross host->device per iter. Measured +13.3% chunk
        # overhead, passes gate 4 (≤20% median, ≤1.30x p99). See
        # 03_experiments/2026-04-30-per-step-overhead-modal-a100.md.
        import onnxruntime as ort
        n = self._num_steps
        dt = -1.0 / n
        x_t = noise.astype(np.float32, copy=False)
        B = x_t.shape[0]
        # Underlying ORT session (the cuda-graphs wrapper exposes .session).
        raw_sess = getattr(self._sess_expert, "session", self._sess_expert)
        binding = raw_sess.io_binding()
        # Pin constant inputs to device. Hold OrtValues in a list so Python
        # GC can't drop the device memory mid-loop (would segfault ORT).
        device_kept_alive = []
        for nm, arr in expert_feed_base.items():
            ortval = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", 0)
            binding.bind_ortvalue_input(nm, ortval)
            device_kept_alive.append(ortval)
        for step in range(n):
            time_val = 1.0 + step * dt
            t_tensor = np.full((B,), time_val, dtype=np.float32)
            binding.bind_cpu_input("x_t", x_t)
            binding.bind_cpu_input("t", t_tensor)
            binding.bind_output("v_t", "cpu")
            raw_sess.run_with_iobinding(binding)
            v_t = binding.get_outputs()[0].numpy()
            if v_t.dtype != np.float32:
                v_t = v_t.astype(np.float32)
            x_t = x_t + dt * v_t
        return x_t

    def reset_cache(self) -> None:
        """Drop any cached VLM output. Call between episodes so cross-task
        phash collisions can't bridge unrelated observations. For
        cache_level='episode' this also clears the episode cache."""
        self._cache = None
        self._action_cache = None
        self._call_index = 0
        if self._episode_cache is not None:
            self._episode_cache.reset()
        # Reset action-similarity fast path — last episode's action chunk
        # has no relevance to the next task's first chunk.
        self._fast_path.reset()

    def get_stats(self) -> dict[str, Any]:
        base = self._stats.as_dict()
        if self._episode_cache is not None:
            base["episode_cache"] = self._episode_cache.stats.as_dict()
        return base

    def _predict_with_episode_cache(
        self,
        *,
        episode_id: str,
        img_base, img_wrist_l, img_wrist_r,
        mask_base, mask_wrist_l, mask_wrist_r,
        lang_tokens, lang_masks,
        noise, state,
    ) -> np.ndarray:
        """Episode-keyed cache path. Looks up (episode_id, lang_hash);
        on hit reuses cached past_kv + prefix_pad_masks (skips VLM).
        On miss runs the VLM and stores the result. Expert always runs
        (needs fresh state + noise per timestep)."""
        assert self._episode_cache is not None
        hit = self._episode_cache.lookup(episode_id, lang_tokens)
        if hit is not None:
            past_kv = hit.past_kv
            prefix_pad = hit.prefix_pad_masks
        else:
            # Cache miss — run the VLM forward
            prefix_feed: dict[str, np.ndarray] = {
                "img_base": img_base, "img_wrist_l": img_wrist_l, "img_wrist_r": img_wrist_r,
                "mask_base": mask_base, "mask_wrist_l": mask_wrist_l, "mask_wrist_r": mask_wrist_r,
                "lang_tokens": lang_tokens, "lang_masks": lang_masks,
            }
            prefix_feed = {k: v for k, v in prefix_feed.items() if k in self._prefix_input_names}
            prefix_out = self._sess_prefix.run(self._prefix_output_names, prefix_feed)
            # Last output is prefix_pad_masks; earlier ones are past_kv tensors
            past_kv = [prefix_out[i] for i in range(len(self._past_kv_names))]
            prefix_pad = prefix_out[-1]
            self._episode_cache.insert(episode_id, lang_tokens, past_kv, prefix_pad)

        # Build expert feed base (everything except noise/x_t/t which the
        # per-step path injects per Euler step). Mirror non-episode path.
        expert_feed_base = {name: past_kv[i] for i, name in enumerate(self._past_kv_names)}
        expert_feed_base["prefix_pad_masks"] = prefix_pad
        if self.config.get("decomposed", {}).get("expert_takes_state"):
            if state is None:
                raise ValueError(
                    "decomposed export was built with expert_takes_state=True; "
                    "predict_action_chunk requires state=<np.ndarray>"
                )
            state_arr = state.astype(np.float32, copy=False)
            expected_dim = next(
                i.shape[-1] for i in self._sess_expert.get_inputs() if i.name == "state"
            )
            if isinstance(expected_dim, int) and state_arr.shape[-1] < expected_dim:
                pad = np.zeros(
                    state_arr.shape[:-1] + (expected_dim - state_arr.shape[-1],),
                    dtype=state_arr.dtype,
                )
                state_arr = np.concatenate([state_arr, pad], axis=-1)
            expert_feed_base["state"] = state_arr

        actions = self._run_expert(expert_feed_base, noise)
        return actions

    # ---- Cache machinery --------------------------------------------

    def _get_or_run_prefix(
        self,
        *,
        img_base, img_wrist_l, img_wrist_r,
        mask_base, mask_wrist_l, mask_wrist_r,
        lang_tokens, lang_masks,
        image_phashes: tuple[bytes, ...],
        lang_hash: bytes,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        now = time.time()
        # `self._call_index` is bumped by predict_action_chunk once per
        # public call — don't bump again here or step-count TTL breaks.

        if self.enable_cache and self._cache is not None:
            entry = self._cache
            # Staleness check: prefer step-count when the caller has
            # configured `cache_max_age_steps > 0` (offline eval); fall
            # back to wall-clock TTL otherwise (online deployment where
            # frames arrive at 20-50 Hz and wall-clock is meaningful).
            if self.cache_max_age_steps > 0:
                stale = (self._call_index - entry.step_index) > self.cache_max_age_steps
            else:
                stale = (now - entry.timestamp) > self.cache_ttl_sec
            if stale:
                self._stats.evictions_ttl += 1
                self._cache = None
            elif entry.lang_hash != lang_hash:
                self._stats.evictions_lang += 1
                self._cache = None
            elif not self._phashes_match(entry.image_phashes, image_phashes):
                self._stats.evictions_phash += 1
                self._cache = None
            else:
                self._stats.hits += 1
                return entry.past_kv, entry.prefix_pad_masks

        # Cache miss (or disabled): run VLM
        self._stats.misses += 1
        prefix_feed = {
            "img_base": img_base.astype(np.float32, copy=False),
            "img_wrist_l": img_wrist_l.astype(np.float32, copy=False),
            "img_wrist_r": img_wrist_r.astype(np.float32, copy=False),
            "mask_base": mask_base.astype(np.bool_, copy=False),
            "mask_wrist_l": mask_wrist_l.astype(np.bool_, copy=False),
            "mask_wrist_r": mask_wrist_r.astype(np.bool_, copy=False),
            "lang_tokens": lang_tokens.astype(np.int64, copy=False),
            "lang_masks": lang_masks.astype(np.bool_, copy=False),
        }
        outputs = self._sess_prefix.run(self._prefix_output_names, prefix_feed)
        out_dict = dict(zip(self._prefix_output_names, outputs))
        past_kv = [out_dict[n] for n in self._past_kv_names]
        prefix_pad = out_dict["prefix_pad_masks"]

        if self.enable_cache:
            self._cache = CacheEntry(
                past_kv=past_kv,
                prefix_pad_masks=prefix_pad,
                image_phashes=image_phashes,
                lang_hash=lang_hash,
                timestamp=now,
                step_index=self._call_index,
            )
        return past_kv, prefix_pad

    def _phashes_match(
        self,
        a: tuple[bytes, ...],
        b: tuple[bytes, ...],
    ) -> bool:
        if len(a) != len(b):
            return False
        for ha, hb in zip(a, b):
            if self._hamming(ha, hb) > self.phash_hamming_threshold:
                return False
        return True

    @staticmethod
    def _hamming(a: bytes, b: bytes) -> int:
        return sum(bin(x ^ y).count("1") for x, y in zip(a, b))

    @classmethod
    def _phash(cls, img: np.ndarray) -> bytes:
        """Average-hash perceptual hash. ``img`` is (B, 3, H, W) float
        in arbitrary range — we downsample to ``PHASH_SIZE`` and compare
        each pixel to the mean. Robust to small sensor noise and slight
        camera-motion. Pure numpy so no optional deps.

        Returns a ``PHASH_SIZE*PHASH_SIZE//8`` byte string (8 bytes for
        8×8 = 64 bits).
        """
        if img.ndim != 4:
            raise ValueError(f"expected (B,3,H,W) image, got shape {img.shape}")
        # Take first batch item + mean across channels → (H, W) gray
        gray = img[0].mean(axis=0)
        h, w = gray.shape
        step_h = max(1, h // cls.PHASH_SIZE)
        step_w = max(1, w // cls.PHASH_SIZE)
        # Integer downsample — coarse but fast + dependency-free
        small = gray[:step_h * cls.PHASH_SIZE, :step_w * cls.PHASH_SIZE]
        small = small.reshape(cls.PHASH_SIZE, step_h, cls.PHASH_SIZE, step_w).mean(axis=(1, 3))
        bits = small > small.mean()
        bits_flat = bits.flatten()
        # Pack bits into bytes
        out = bytearray()
        for byte_idx in range(0, len(bits_flat), 8):
            byte = 0
            for bit_idx, bit in enumerate(bits_flat[byte_idx : byte_idx + 8]):
                if bit:
                    byte |= 1 << bit_idx
            out.append(byte)
        return bytes(out)

    @staticmethod
    def _lang_hash(lang_tokens: np.ndarray) -> bytes:
        # Use tolist() to normalize the numpy dtype + byte layout. The
        # tensor path in the LIBERO harness (`lang_tokens.cpu().numpy()`)
        # can produce arrays that have the same integer VALUES but
        # different low-level byte patterns across calls (e.g., int64
        # padding, uninitialized trailing bytes in the alignment) which
        # breaks `tobytes()`-based hashing. Python int repr is
        # canonical, so md5 over the list-repr is stable.
        return hashlib.md5(repr(lang_tokens.tolist()).encode()).digest()


__all__ = ["Pi05DecomposedInference", "CacheStats", "CacheEntry"]
