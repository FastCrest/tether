# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Small JSONL run writer used by game collectors/evaluators.

Phase 7 (vendor-auto-soarm ADR): write_episode() also enqueues the episode
into the Curate contribution queue when the local consent receipt at
~/.reflex/consent.json is present. Idempotent + best-effort — failures
never block the bench-game run.
"""
import json
import logging

logger = logging.getLogger(__name__)


class JsonlRunWriter:
    def __init__(self, run_dir, config_name, config):
        self.run_dir = run_dir
        self.frame_dir = run_dir / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=False)
        self.records_path = run_dir / "episodes.jsonl"
        self.config_path = run_dir / config_name
        self.config_path.write_text(json.dumps(config, indent=2))
        # Curate dual-write: lazily resolve the consent receipt + free-tier
        # collector. None when the customer hasn't opted in (typical default).
        self._curate_collector = None
        self._curate_init_attempted = False

    def _ensure_curate_collector(self):
        """Lazy-init on first write. Reads ~/.reflex/consent.json once."""
        if self._curate_init_attempted:
            return self._curate_collector
        self._curate_init_attempted = True
        try:
            from reflex.curate import consent as _curate_consent
            from reflex.curate.free_collector import FreeContributorCollector
            if not _curate_consent.is_opted_in():
                return None
            collector = FreeContributorCollector.from_consent()
            collector.start()
            self._curate_collector = collector
            logger.info(
                "bench-game.curate dual-write armed: contributor_id=%s tier=%s",
                collector.contributor_id, collector.tier,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("bench-game.curate dual-write skipped: %s", exc)
            self._curate_collector = None
        return self._curate_collector

    def write_episode(self, record):
        with self.records_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        # Curate dual-write — best-effort; never block the bench-game run.
        collector = self._ensure_curate_collector()
        if collector is None:
            return
        try:
            from reflex.pro.data_collection import (
                CollectedEvent,
                QueueFull,
                hash_instruction,
            )
            actions = record.get("actions") or record.get("action_chunk") or []
            state = record.get("state") or record.get("state_vec") or []
            instruction = record.get("instruction") or record.get("task") or "circle_tap"
            event = CollectedEvent(
                timestamp=record.get("timestamp") or record.get("ended_at") or "",
                episode_id=str(record.get("episode_index", record.get("episode_id", "ep"))),
                state_vec=list(state) if isinstance(state, list) else [],
                action_chunk=actions if isinstance(actions, list) else [],
                reward_proxy=float(record.get("hit", 0.0)) if "hit" in record else 1.0,
                image_b64=None,
                instruction_hash=hash_instruction(instruction),
                instruction_raw=instruction,
                metadata={
                    "source": "bench_game.circle_lr",
                    "episode_index": record.get("episode_index"),
                },
            )
            try:
                collector.record(event)
            except QueueFull:
                pass  # drop — collector tracks via events_dropped
        except Exception as exc:  # noqa: BLE001
            logger.debug("bench-game.curate dual-write event failed: %s", exc)

    def close(self):
        """Stop the curate collector if it was started. Optional — collectors
        also stop themselves at process exit (daemon threads)."""
        if self._curate_collector is not None:
            try:
                self._curate_collector.stop()
            except Exception:  # noqa: BLE001
                pass
            self._curate_collector = None

    def write_json(self, name, payload):
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2))
        return path

    def write_demo_episode(self, episode, frames):
        path = self.run_dir / "demo_episodes.jsonl"
        rec = {"episode": episode, "frames": frames}
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
