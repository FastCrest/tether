"""Top-level dedup pipeline: phash pre-filter → DTW verification → clusters.

Composes the per-stage modules into a single function the uploader calls.
Returns a `DedupResult` that tags each episode with cluster_id + is_canonical
for downstream consumers.

Phase 1 simplification: O(N²) phash pairwise comparison + O(N²) DTW. Fine
at Phase 1 scale (≤200 episodes per uploader pass). Phase 1.5 adds an
LSH or VPTree for sub-O(N²) phash near-neighbor when corpus grows.

NEVER deletes data. Always returns ALL input episodes in the output —
canonical selection just flags one per cluster, never excludes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reflex.curate.dedup import DEDUP_VERSION
from reflex.curate.dedup.canonical import canonical_episode
from reflex.curate.dedup.cluster import assemble_clusters
from reflex.curate.dedup.phash import hamming_distance
from reflex.curate.dedup.trajectory import trajectory_similarity

# Default thresholds (per spec):
#   phash hamming distance ≤ 4 → candidate (very-similar scenes)
#   trajectory similarity ≥ 0.85 → confirmed duplicate
DEFAULT_PHASH_HAMMING_THRESHOLD = 4
DEFAULT_TRAJECTORY_SIMILARITY_THRESHOLD = 0.85


@dataclass(frozen=True)
class DedupInfo:
    """Per-episode dedup metadata. Stamped onto the parquet record alongside
    the quality_score block."""

    cluster_id: str
    is_canonical: bool
    cluster_size: int
    dedup_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dedup_cluster_id": self.cluster_id,
            "is_canonical": self.is_canonical,
            "dedup_cluster_size": self.cluster_size,
            "dedup_version": self.dedup_version,
        }


def dedup_episodes(
    episodes: dict[str, dict[str, Any]],
    *,
    phash_threshold: int = DEFAULT_PHASH_HAMMING_THRESHOLD,
    trajectory_threshold: float = DEFAULT_TRAJECTORY_SIMILARITY_THRESHOLD,
) -> dict[str, DedupInfo]:
    """Compute cluster_id + is_canonical for every input episode.

    Args:
        episodes: dict mapping episode_id → meta dict. Required keys:
            - phash (str | None): perceptual hash of the first frame.
              When None for an episode, that episode skips stage-1 entirely
              and only matches via trajectory similarity (slower but works).
            - actions (np.ndarray): (T, action_dim) action array. Required
              for stage-2 verification.
            - quality_score (float): used by canonical selection.
            - step_count (int): used by canonical selection tiebreak.
            - first_seen_at (str): ISO 8601 UTC; used by canonical
              selection final tiebreak.
        phash_threshold: max hamming distance for stage-1 candidate match.
            Default 4 catches identical / very-similar scenes (per spec).
        trajectory_threshold: min similarity for stage-2 confirmation.
            Default 0.85 (per spec).

    Returns:
        Dict mapping episode_id → DedupInfo for every input episode.
        Singleton clusters (no duplicates) get is_canonical=True with
        cluster_size=1.
    """
    if not episodes:
        return {}

    episode_ids = list(episodes.keys())

    # Stage 1: phash candidate pairs.
    phash_candidates: list[tuple[str, str]] = []
    for i, ep_a in enumerate(episode_ids):
        ph_a = episodes[ep_a].get("phash")
        if not isinstance(ph_a, str):
            continue
        for ep_b in episode_ids[i + 1:]:
            ph_b = episodes[ep_b].get("phash")
            if not isinstance(ph_b, str):
                continue
            if hamming_distance(ph_a, ph_b) <= phash_threshold:
                phash_candidates.append((ep_a, ep_b))

    # Stage 2: trajectory verification on candidates.
    confirmed: list[tuple[str, str]] = []
    for ep_a, ep_b in phash_candidates:
        actions_a = episodes[ep_a].get("actions")
        actions_b = episodes[ep_b].get("actions")
        if not isinstance(actions_a, np.ndarray) or not isinstance(actions_b, np.ndarray):
            continue
        sim = trajectory_similarity(actions_a, actions_b)
        if sim >= trajectory_threshold:
            confirmed.append((ep_a, ep_b))

    # Cluster assembly (union-find on confirmed pairs).
    clusters = assemble_clusters(
        all_episode_ids=episode_ids,
        confirmed_pairs=confirmed,
    )

    # Canonical selection per cluster + DedupInfo emission.
    out: dict[str, DedupInfo] = {}
    for cluster_id, members in clusters.items():
        member_meta = {ep_id: episodes[ep_id] for ep_id in members}
        canonical = canonical_episode(member_meta)
        for ep_id in members:
            out[ep_id] = DedupInfo(
                cluster_id=cluster_id,
                is_canonical=(ep_id == canonical),
                cluster_size=len(members),
                dedup_version=DEDUP_VERSION,
            )
    return out


__all__ = [
    "DEFAULT_PHASH_HAMMING_THRESHOLD",
    "DEFAULT_TRAJECTORY_SIMILARITY_THRESHOLD",
    "DedupInfo",
    "dedup_episodes",
]
