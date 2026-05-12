"""Episode-level near-duplicate detection for the Curate wedge.

Per `_curation/dedup.md`: 2-stage pipeline — image perceptual hash (phash)
on the first frame for cheap pre-filter + DTW trajectory similarity for
verification. Outputs a `dedup_cluster_id` + `is_canonical: bool` per episode.

NEVER deletes data. Flags clusters; downstream consumers (dataset publishers,
quality scoring) choose how to use the cluster info.

Phase 1 simplification: dedup runs per-queue-file (within-session). Cross-
session within-customer dedup is Phase 1.5; cross-customer is Phase 2.

Submodules:
    phash       — 64-bit average-hash on episode first frame (Pillow + numpy)
    trajectory  — DTW similarity on action arrays (pure numpy)
    cluster     — union-find clustering of confirmed-duplicate pairs
    canonical   — canonical episode selection within a cluster
"""
from __future__ import annotations

DEDUP_VERSION = "v1"

from reflex.curate.dedup.canonical import canonical_episode, score_episode_for_canonical
from reflex.curate.dedup.cluster import assemble_clusters
from reflex.curate.dedup.phash import (
    compute_average_hash,
    fingerprint_bytes,
    hamming_distance,
)
from reflex.curate.dedup.pipeline import (
    DEFAULT_PHASH_HAMMING_THRESHOLD,
    DEFAULT_TRAJECTORY_SIMILARITY_THRESHOLD,
    DedupInfo,
    dedup_episodes,
)
from reflex.curate.dedup.trajectory import trajectory_similarity

__all__ = [
    "DEDUP_VERSION",
    "DEFAULT_PHASH_HAMMING_THRESHOLD",
    "DEFAULT_TRAJECTORY_SIMILARITY_THRESHOLD",
    "DedupInfo",
    "assemble_clusters",
    "canonical_episode",
    "compute_average_hash",
    "dedup_episodes",
    "fingerprint_bytes",
    "hamming_distance",
    "score_episode_for_canonical",
    "trajectory_similarity",
]
