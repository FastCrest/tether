"""Free-tier data collector — extends ProDataCollector for the Curate flywheel.

Composition (per `data-collection-free-tier.md`):
- Writes to `~/.reflex/contribute/queue/` (NOT the customer's HF Hub like Pro)
- Refuses to start without a valid CurateConsent receipt
- Tags every event with the receipt's contributor_id (so the uploader can
  attribute the contribution and so revoke can find + delete by id)
- Applies per-event sanity filters at record time (NaN/Inf actions drop;
  episode-level filters happen later in the uploader)

Episode-level quality filters (length < 30 steps, all-zero actions, etc.)
live in `uploader.py` rather than here — they need the full episode in hand,
which is only assembled when the uploader reads the JSONL files at flush time.

Pro tier continues to use `ProDataCollector` directly with the customer's
chosen `data_dir` (typically `~/.reflex/pro-data/`). The two collectors run
in parallel when a customer is on Pro AND opted into curate (Pro contributors
get both pipelines).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from reflex.curate import consent as curate_consent
from reflex.pro.data_collection import (
    DEFAULT_FLUSH_EVERY_EVENTS,
    DEFAULT_FLUSH_EVERY_SECONDS,
    DEFAULT_MAX_QUEUE,
    CollectedEvent,
    ProDataCollector,
)

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_DIR = "~/.reflex/contribute/queue"


def _has_nan_or_inf(values: list[float] | list[list[float]]) -> bool:
    """Recursive NaN/Inf check on lists or list-of-lists of floats."""
    for v in values:
        if isinstance(v, list):
            if _has_nan_or_inf(v):
                return True
        else:
            try:
                if math.isnan(v) or math.isinf(v):
                    return True
            except TypeError:
                continue
    return False


class FreeContributorCollector(ProDataCollector):
    """Free-tier extension of ProDataCollector.

    Lifecycle:
        collector = FreeContributorCollector.from_consent()
        collector.start()
        # ... in /act handler, when consent.is_opted_in() and --record:
        try:
            collector.record(event)  # auto-tags with contributor_id
        except QueueFull:
            # /act never blocks — drop on full queue
            pass
        # ... at shutdown:
        collector.stop()
    """

    __slots__ = ("_contributor_id", "_tier", "_consent_path")

    def __init__(
        self,
        *,
        contributor_id: str,
        tier: str = "free",
        data_dir: str | Path = DEFAULT_QUEUE_DIR,
        max_queue: int = DEFAULT_MAX_QUEUE,
        flush_every_events: int = DEFAULT_FLUSH_EVERY_EVENTS,
        flush_every_seconds: float = DEFAULT_FLUSH_EVERY_SECONDS,
        consent_path: str | Path = curate_consent.DEFAULT_CONSENT_PATH,
    ):
        if not contributor_id:
            raise ValueError("FreeContributorCollector requires non-empty contributor_id")
        super().__init__(
            data_dir=data_dir,
            max_queue=max_queue,
            flush_every_events=flush_every_events,
            flush_every_seconds=flush_every_seconds,
        )
        self._contributor_id = contributor_id
        self._tier = tier
        self._consent_path = Path(consent_path).expanduser()

    @property
    def contributor_id(self) -> str:
        return self._contributor_id

    @property
    def tier(self) -> str:
        return self._tier

    @classmethod
    def from_consent(
        cls,
        *,
        data_dir: str | Path = DEFAULT_QUEUE_DIR,
        consent_path: str | Path = curate_consent.DEFAULT_CONSENT_PATH,
        **kwargs: Any,
    ) -> "FreeContributorCollector":
        """Construct from the on-disk CurateConsent receipt. Refuses (raises
        ConsentNotFound) when the customer hasn't opted in — callers should
        gate on `curate_consent.is_opted_in()` before instantiating."""
        receipt = curate_consent.load(consent_path)
        return cls(
            contributor_id=receipt.contributor_id,
            tier=receipt.tier,
            data_dir=data_dir,
            consent_path=consent_path,
            **kwargs,
        )

    def record(self, event: CollectedEvent) -> None:
        """Per-event sanity filter (NaN/Inf in actions or state) + contributor
        tag. Episode-level filters happen at upload time — this just keeps
        catastrophically-broken events out of the queue.

        The frozen CollectedEvent is rebuilt with the contributor_id added
        to its metadata. Skipping (vs raising) on filter rejection keeps
        /act non-blocking — broken events become a dropped-counter increment.
        """
        if _has_nan_or_inf(event.action_chunk):
            self._events_dropped += 1
            logger.debug(
                "free_collector dropped event: NaN/Inf in action_chunk "
                "(episode_id=%s timestamp=%s)",
                event.episode_id, event.timestamp,
            )
            return
        if _has_nan_or_inf(event.state_vec):
            self._events_dropped += 1
            logger.debug(
                "free_collector dropped event: NaN/Inf in state_vec "
                "(episode_id=%s timestamp=%s)",
                event.episode_id, event.timestamp,
            )
            return

        # Inject contributor_id into metadata. The original event is frozen,
        # so build a new one — cheap (just dataclass replace).
        new_metadata = dict(event.metadata)
        new_metadata["contributor_id"] = self._contributor_id
        new_metadata["tier"] = self._tier
        tagged = CollectedEvent(
            timestamp=event.timestamp,
            episode_id=event.episode_id,
            state_vec=event.state_vec,
            action_chunk=event.action_chunk,
            reward_proxy=event.reward_proxy,
            image_b64=event.image_b64,
            instruction_hash=event.instruction_hash,
            instruction_raw=event.instruction_raw,
            metadata=new_metadata,
        )
        super().record(tagged)


__all__ = [
    "DEFAULT_QUEUE_DIR",
    "FreeContributorCollector",
]
