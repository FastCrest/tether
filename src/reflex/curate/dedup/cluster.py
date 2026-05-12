"""Union-find cluster assembly from confirmed-duplicate pairs.

Given a list of (episode_id_a, episode_id_b) pairs where both phash AND
trajectory similarity confirmed they're near-duplicates, group them into
connected-component clusters.

Each cluster gets a deterministic UUID-style cluster_id. Singleton episodes
(no duplicates) get singleton clusters of size 1.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable


class _UnionFind:
    """Path-compressed + rank-aware union-find. O(α(n)) amortized per op."""

    __slots__ = ("_parent", "_rank")

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        # Path compression
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            self._parent[rx] = ry
        elif self._rank[rx] > self._rank[ry]:
            self._parent[ry] = rx
        else:
            self._parent[ry] = rx
            self._rank[rx] += 1

    def all_roots(self) -> set[str]:
        return {self.find(x) for x in self._parent}


def _stable_cluster_id(member_ids: Iterable[str]) -> str:
    """Deterministic cluster_id from sorted member episode_ids.

    Same set of members → same cluster_id across runs. This means new
    contributions don't churn cluster_ids of existing clusters unless
    membership actually changes (Phase 1 rebuilds-from-scratch each
    pass; Phase 1.5 incremental updates will need a more careful scheme).
    """
    sorted_ids = sorted(member_ids)
    blob = "\x1f".join(sorted_ids).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:16]
    return f"cluster_{digest}"


def assemble_clusters(
    *,
    all_episode_ids: Iterable[str],
    confirmed_pairs: Iterable[tuple[str, str]],
) -> dict[str, list[str]]:
    """Build clusters from confirmed-duplicate pairs.

    Returns a dict mapping cluster_id → sorted list of member episode_ids.
    Includes singleton clusters (size 1) for every episode that has no
    confirmed duplicates — every episode_id appears in exactly one cluster.

    Args:
        all_episode_ids: every episode being clustered (so singletons get
            their own cluster).
        confirmed_pairs: list of (a, b) tuples where stage-2 trajectory
            similarity confirmed the dup.
    """
    uf = _UnionFind()
    for ep_id in all_episode_ids:
        uf.add(ep_id)
    for a, b in confirmed_pairs:
        uf.union(a, b)

    by_root: dict[str, list[str]] = defaultdict(list)
    for ep_id in uf._parent:
        by_root[uf.find(ep_id)].append(ep_id)

    clusters: dict[str, list[str]] = {}
    for root, members in by_root.items():
        cluster_id = _stable_cluster_id(members)
        clusters[cluster_id] = sorted(members)
    return clusters


__all__ = [
    "assemble_clusters",
]
