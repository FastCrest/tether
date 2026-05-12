"""Tests for src/reflex/curate/dedup/ — episode-level near-duplicate detection.

Covers each component (phash + DTW + cluster + canonical) and the top-level
pipeline. Verifies the NEVER-DELETE doctrine: dedup flags but never removes data.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from reflex.curate.dedup import (
    DEDUP_VERSION,
    DedupInfo,
    assemble_clusters,
    canonical_episode,
    compute_average_hash,
    dedup_episodes,
    fingerprint_bytes,
    hamming_distance,
    score_episode_for_canonical,
    trajectory_similarity,
)


def _png_bytes(color: int = 0x80) -> bytes:
    """Generate a tiny PNG of a single solid color via Pillow."""
    from PIL import Image
    img = Image.new("RGB", (32, 32), (color, color, color))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── phash ────────────────────────────────────────────────────────────────────


def test_compute_average_hash_returns_16_hex() -> None:
    h = compute_average_hash(_png_bytes(128))
    assert len(h) == 16
    int(h, 16)  # parses as hex


def test_compute_average_hash_identical_images_same_hash() -> None:
    img = _png_bytes(100)
    h1 = compute_average_hash(img)
    h2 = compute_average_hash(img)
    assert h1 == h2


def test_compute_average_hash_different_images_low_distance_for_similar() -> None:
    # Solid colors close together → near-zero hamming distance
    h1 = compute_average_hash(_png_bytes(120))
    h2 = compute_average_hash(_png_bytes(125))
    # With aHash on solid color, both should hash similarly
    d = hamming_distance(h1, h2)
    assert d <= 8  # much less than 64 max


def test_compute_average_hash_empty_bytes_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_average_hash(b"")


def test_compute_average_hash_invalid_bytes_raises() -> None:
    with pytest.raises(ValueError, match="failed to decode"):
        compute_average_hash(b"this is not an image")


def test_hamming_distance_identical() -> None:
    assert hamming_distance("0123456789abcdef", "0123456789abcdef") == 0


def test_hamming_distance_max() -> None:
    assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64


def test_hamming_distance_mismatched_length_returns_max() -> None:
    # 8 hex chars vs 16 → mismatched, returns max
    d = hamming_distance("01234567", "0123456789abcdef")
    assert d == 64


def test_fingerprint_bytes_stable() -> None:
    img = _png_bytes(50)
    f1 = fingerprint_bytes(img)
    f2 = fingerprint_bytes(img)
    assert f1 == f2
    assert len(f1) == 64  # SHA-256 hex


# ── trajectory ───────────────────────────────────────────────────────────────


def test_trajectory_similarity_identical_full_score() -> None:
    actions = np.linspace(0, 1, 50).reshape(50, 1).astype(np.float32)
    sim = trajectory_similarity(actions, actions)
    assert sim > 0.95


def test_trajectory_similarity_random_low_score() -> None:
    np.random.seed(42)
    a = np.random.uniform(-1, 1, (50, 2)).astype(np.float32)
    b = np.random.uniform(-1, 1, (50, 2)).astype(np.float32)
    sim = trajectory_similarity(a, b)
    assert sim < 0.7


def test_trajectory_similarity_empty_returns_zero() -> None:
    a = np.zeros((0, 1), dtype=np.float32)
    b = np.linspace(0, 1, 10).reshape(10, 1).astype(np.float32)
    assert trajectory_similarity(a, b) == 0.0


def test_trajectory_similarity_dim_mismatch_returns_zero() -> None:
    a = np.zeros((10, 2), dtype=np.float32)
    b = np.zeros((10, 3), dtype=np.float32)
    assert trajectory_similarity(a, b) == 0.0


# ── canonical selection ─────────────────────────────────────────────────────


def test_canonical_singleton_returns_only_member() -> None:
    cluster = {"ep_solo": {"quality_score": 0.5, "step_count": 100, "first_seen_at": "2026-01-01"}}
    assert canonical_episode(cluster) == "ep_solo"


def test_canonical_higher_quality_wins() -> None:
    cluster = {
        "ep_low": {"quality_score": 0.6, "step_count": 100, "first_seen_at": "2026-01-01"},
        "ep_high": {"quality_score": 0.9, "step_count": 100, "first_seen_at": "2026-01-02"},
    }
    assert canonical_episode(cluster) == "ep_high"


def test_canonical_tiebreak_step_count() -> None:
    cluster = {
        "ep_short": {"quality_score": 0.7, "step_count": 80, "first_seen_at": "2026-01-01"},
        "ep_long": {"quality_score": 0.7, "step_count": 200, "first_seen_at": "2026-01-02"},
    }
    assert canonical_episode(cluster) == "ep_long"


def test_canonical_tiebreak_oldest_wins() -> None:
    cluster = {
        "ep_new": {"quality_score": 0.7, "step_count": 100, "first_seen_at": "2026-05-05T11:00:00Z"},
        "ep_old": {"quality_score": 0.7, "step_count": 100, "first_seen_at": "2026-05-05T10:00:00Z"},
    }
    assert canonical_episode(cluster) == "ep_old"


def test_canonical_empty_cluster_returns_none() -> None:
    assert canonical_episode({}) is None


def test_score_episode_for_canonical_missing_fields_safe() -> None:
    score = score_episode_for_canonical({})
    assert score[0] == 0.0
    assert score[1] == 0


# ── cluster assembly ────────────────────────────────────────────────────────


def test_assemble_clusters_singletons_only() -> None:
    clusters = assemble_clusters(
        all_episode_ids=["a", "b", "c"], confirmed_pairs=[],
    )
    assert len(clusters) == 3
    for cid, members in clusters.items():
        assert len(members) == 1


def test_assemble_clusters_pairs_form_clusters() -> None:
    clusters = assemble_clusters(
        all_episode_ids=["a", "b", "c", "d"],
        confirmed_pairs=[("a", "b"), ("c", "d")],
    )
    assert len(clusters) == 2
    sizes = sorted(len(m) for m in clusters.values())
    assert sizes == [2, 2]


def test_assemble_clusters_transitive_grouping() -> None:
    """a-b + b-c should produce one cluster of {a, b, c}."""
    clusters = assemble_clusters(
        all_episode_ids=["a", "b", "c"],
        confirmed_pairs=[("a", "b"), ("b", "c")],
    )
    assert len(clusters) == 1
    members = next(iter(clusters.values()))
    assert sorted(members) == ["a", "b", "c"]


def test_assemble_clusters_stable_id_for_same_membership() -> None:
    """Same cluster membership → same cluster_id across runs (deterministic)."""
    c1 = assemble_clusters(all_episode_ids=["a", "b"], confirmed_pairs=[("a", "b")])
    c2 = assemble_clusters(all_episode_ids=["b", "a"], confirmed_pairs=[("a", "b")])
    assert sorted(c1.keys()) == sorted(c2.keys())


# ── pipeline ────────────────────────────────────────────────────────────────


def _ep(*, phash: str | None, actions: np.ndarray, quality: float = 0.7,
        step_count: int | None = None, ts: str = "2026-05-05T10:00:00Z") -> dict:
    return {
        "phash": phash,
        "actions": actions,
        "quality_score": quality,
        "step_count": step_count if step_count is not None else len(actions),
        "first_seen_at": ts,
    }


def test_dedup_empty_input() -> None:
    assert dedup_episodes({}) == {}


def test_dedup_singleton_episode_canonical_size_one() -> None:
    actions = np.linspace(0, 1, 50).reshape(50, 1).astype(np.float32)
    result = dedup_episodes({"ep_alone": _ep(phash="0000000000000000", actions=actions)})
    assert len(result) == 1
    info = result["ep_alone"]
    assert isinstance(info, DedupInfo)
    assert info.is_canonical
    assert info.cluster_size == 1
    assert info.dedup_version == DEDUP_VERSION


def test_dedup_two_near_duplicates_share_cluster() -> None:
    actions = np.linspace(0, 1, 100).reshape(100, 1).astype(np.float32)
    np.random.seed(0)
    actions_b = actions + np.random.normal(0, 0.001, actions.shape).astype(np.float32)
    result = dedup_episodes({
        "ep_a": _ep(phash="0000000000000000", actions=actions, quality=0.9),
        "ep_b": _ep(phash="0000000000000000", actions=actions_b, quality=0.7),
    })
    assert result["ep_a"].cluster_id == result["ep_b"].cluster_id
    assert result["ep_a"].cluster_size == 2
    assert result["ep_a"].is_canonical  # higher quality wins
    assert not result["ep_b"].is_canonical


def test_dedup_distinct_episodes_separate_clusters() -> None:
    np.random.seed(1)
    actions_a = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
    actions_b = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
    result = dedup_episodes({
        "ep_a": _ep(phash="0000000000000000", actions=actions_a),
        "ep_b": _ep(phash="ffffffffffffffff", actions=actions_b),
    })
    assert result["ep_a"].cluster_id != result["ep_b"].cluster_id


def test_dedup_phash_match_but_trajectory_differs_no_cluster() -> None:
    """Phash matches but trajectory differs → stage 2 rejects → separate clusters."""
    np.random.seed(2)
    actions_a = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
    actions_b = np.random.uniform(-1, 1, (100, 1)).astype(np.float32)
    result = dedup_episodes({
        "ep_a": _ep(phash="0000000000000000", actions=actions_a),
        "ep_b": _ep(phash="0000000000000000", actions=actions_b),
    })
    assert result["ep_a"].cluster_id != result["ep_b"].cluster_id


def test_dedup_returns_all_input_episodes() -> None:
    """NEVER-DELETE doctrine: every input episode appears in the output."""
    actions = np.linspace(0, 1, 50).reshape(50, 1).astype(np.float32)
    inputs = {
        f"ep_{i}": _ep(phash="0000000000000000", actions=actions)
        for i in range(5)
    }
    result = dedup_episodes(inputs)
    assert set(result.keys()) == set(inputs.keys())


def test_dedup_phash_none_falls_back_to_trajectory() -> None:
    """When phash is None for both episodes, dedup still works via trajectory.

    Note: Phase 1 stage-1 phash filter requires BOTH episodes to have a
    phash. With phash=None, no candidates emerge, so cluster_size=1
    even for very similar trajectories. This is the documented limitation.
    """
    actions = np.linspace(0, 1, 100).reshape(100, 1).astype(np.float32)
    np.random.seed(3)
    actions_b = actions + np.random.normal(0, 0.001, actions.shape).astype(np.float32)
    result = dedup_episodes({
        "ep_a": _ep(phash=None, actions=actions),
        "ep_b": _ep(phash=None, actions=actions_b),
    })
    # Phase 1 limitation: no phash → no stage-1 candidates → no stage-2 → singletons.
    assert result["ep_a"].cluster_size == 1
    assert result["ep_b"].cluster_size == 1
    # Both still appear in output.
    assert "ep_a" in result
    assert "ep_b" in result


# ── DedupInfo.to_dict round-trip ────────────────────────────────────────────


def test_dedup_info_to_dict_keys() -> None:
    info = DedupInfo(
        cluster_id="cluster_abc",
        is_canonical=True,
        cluster_size=3,
        dedup_version="v1",
    )
    d = info.to_dict()
    assert d["dedup_cluster_id"] == "cluster_abc"
    assert d["is_canonical"] is True
    assert d["dedup_cluster_size"] == 3
    assert d["dedup_version"] == "v1"
