"""Canonical episode selection within a duplicate cluster.

Per the spec: pick the canonical by:
    1. Highest quality_score
    2. Tiebreaker: longest episode (more data)
    3. Tiebreaker: oldest contribution (incumbent advantage; first
       contributor gets canonical status when scores tie)

Returns the (episode_id, score_tuple) of the winner so callers can mark
`is_canonical=True` on that one and `is_canonical=False` on the rest.
"""
from __future__ import annotations

from typing import Any


def score_episode_for_canonical(
    episode_meta: dict[str, Any],
) -> tuple[float, int, str]:
    """Compute the canonical-selection score tuple for an episode.

    Higher is better. Tuple is sortable in descending order:
        (quality_score, step_count, -timestamp_for_oldest_wins)

    Notes:
        quality_score is read from episode_meta["quality_score"] (set by
        quality.composite). Defaults to 0.0 if absent.
        step_count is read from episode_meta["step_count"] or
        len(episode_meta.get("rows", [])).
        timestamp is read from episode_meta["first_seen_at"] (ISO 8601
        UTC string). Defaults to "" so absent timestamps lose ties.

    Returns a tuple where larger first elements (and longer episodes,
    earlier timestamps) win in `max(scores, key=...)` style sorts.
    """
    qs = float(episode_meta.get("quality_score") or 0.0)
    step_count = int(episode_meta.get("step_count") or 0)
    ts = str(episode_meta.get("first_seen_at") or "")
    # We want EARLIEST timestamp to win → invert the string comparison via
    # negation in the caller. Returning a string here keeps the tuple
    # JSON-friendly + lexicographic-comparable.
    # Caller uses: sorted(reverse=True) on (quality_score, step_count) but
    # ascending on timestamp. Simplest: pre-invert via "~" + ts so that
    # in a single descending sort, EARLIER ts wins.
    inverted_ts = _invert_ts_for_descending_sort(ts)
    return (qs, step_count, inverted_ts)


def _invert_ts_for_descending_sort(ts: str) -> str:
    """Return a string that compares to other inverted timestamps in the
    OPPOSITE order (so earlier original timestamps sort LATER in
    ascending order, which is what we want under reversed sort)."""
    # Trick: invert each character via XOR with a high byte. Sorting the
    # inverted strings descending is equivalent to sorting the originals
    # ascending. Empty string → "" still compares lowest (no win).
    if not ts:
        return ""
    return "".join(chr(255 - ord(c)) for c in ts)


def canonical_episode(
    cluster_episodes: dict[str, dict[str, Any]],
) -> str | None:
    """Select the canonical episode_id from a cluster.

    Args:
        cluster_episodes: dict of episode_id → meta dict (quality_score,
            step_count, first_seen_at fields).

    Returns:
        episode_id of the canonical, or None when the cluster is empty.
        Singleton clusters (size 1) trivially return their only member.
    """
    if not cluster_episodes:
        return None
    if len(cluster_episodes) == 1:
        return next(iter(cluster_episodes))

    # Highest score wins.
    return max(
        cluster_episodes.keys(),
        key=lambda eid: score_episode_for_canonical(cluster_episodes[eid]),
    )


__all__ = [
    "canonical_episode",
    "score_episode_for_canonical",
]
