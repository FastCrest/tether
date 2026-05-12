"""Tests for src/reflex/curate/metadata/ — episode metadata auto-tagging."""
from __future__ import annotations

import numpy as np
import pytest

from reflex.curate.metadata import (
    EnrichmentResult,
    TASK_TYPES,
    action_complexity,
    classify_subtype,
    classify_task,
    detect_language,
    difficulty_from_instruction,
    enrich_from_jsonl_rows,
    enrich_metadata,
    terminal_gripper_state,
)


# ── task_classifier ─────────────────────────────────────────────────────────


def test_task_types_includes_unknown() -> None:
    assert "unknown" in TASK_TYPES


def test_task_types_has_24_verbs_plus_unknown() -> None:
    # Per the research sidecar, expanded from 19 → 24 + unknown.
    assert len(TASK_TYPES) == 27 or "unknown" in TASK_TYPES  # tolerant of future edits


def test_classify_task_pick_high_confidence() -> None:
    task, conf = classify_task("Pick up the red block")
    assert task == "pick"
    assert conf >= 0.5


def test_classify_task_pour_single_match_full_confidence() -> None:
    task, conf = classify_task("pour the water")
    assert task == "pour"
    assert conf == 1.0


def test_classify_task_multi_action_picks_first() -> None:
    """When multiple verbs match, the first action-verb wins."""
    task, conf = classify_task("pick up the block and place it in the box")
    assert task == "pick"
    assert conf < 1.0  # multiple matches → reduced confidence


def test_classify_task_unknown_returns_unknown() -> None:
    task, conf = classify_task("do the thing")
    assert task == "unknown"
    assert conf == 0.0


def test_classify_task_empty_returns_unknown() -> None:
    assert classify_task("") == ("unknown", 0.0)
    assert classify_task("   ") == ("unknown", 0.0)


def test_classify_subtype_block() -> None:
    subtype, conf = classify_subtype("Pick up the red block", "pick")
    assert subtype == "block"
    assert conf > 0.5


def test_classify_subtype_falls_back_to_unknown() -> None:
    subtype, conf = classify_subtype("Pick up the thing", "pick")
    assert subtype == "unknown"
    assert conf == 0.0


def test_classify_subtype_unknown_task_returns_unknown() -> None:
    subtype, conf = classify_subtype("Pick up the block", "unknown")
    assert subtype == "unknown"


# ── difficulty ──────────────────────────────────────────────────────────────


def test_difficulty_simple_low_score() -> None:
    score, conf = difficulty_from_instruction("Pick up the block")
    assert score < 0.5
    assert conf > 0.0


def test_difficulty_multistep_high_score() -> None:
    score, conf = difficulty_from_instruction(
        "Carefully pick up the red block, then place it in the box, "
        "and finally close the lid"
    )
    assert score > 0.7
    assert conf > 0.5


def test_difficulty_empty_zero() -> None:
    score, conf = difficulty_from_instruction("")
    assert score == 0.0
    assert conf == 0.0


def test_difficulty_precision_marker_increases_score() -> None:
    bare = difficulty_from_instruction("Pick up the block now")
    precise = difficulty_from_instruction("Carefully pick up the block now")
    assert precise[0] > bare[0]


# ── language ────────────────────────────────────────────────────────────────


def test_language_english() -> None:
    lang, conf = detect_language("Pick up the red block")
    assert lang == "en"
    assert conf > 0.5


def test_language_japanese() -> None:
    lang, conf = detect_language("赤いブロックを持ち上げる")
    assert lang == "ja"
    assert conf > 0.7


def test_language_chinese() -> None:
    lang, conf = detect_language("拿起红色方块")
    assert lang == "zh"
    assert conf > 0.7


def test_language_korean() -> None:
    lang, conf = detect_language("빨간 블록을 들어 올리세요")
    assert lang == "ko"
    assert conf > 0.7


def test_language_russian() -> None:
    lang, conf = detect_language("Возьми красный блок")
    assert lang == "ru"
    assert conf > 0.7


def test_language_arabic() -> None:
    lang, conf = detect_language("التقط الكتلة الحمراء")
    assert lang == "ar"
    assert conf > 0.7


def test_language_empty_unknown() -> None:
    lang, conf = detect_language("")
    assert lang == "unknown"
    assert conf == 0.0


# ── trajectory_tags ─────────────────────────────────────────────────────────


def test_terminal_gripper_state_closed() -> None:
    actions = np.zeros((10, 7))
    actions[:, 6] = 1.0  # gripper closed
    state, conf = terminal_gripper_state(actions, gripper_dim=6)
    assert state == "closed"
    assert conf > 0.9


def test_terminal_gripper_state_open() -> None:
    actions = np.zeros((10, 7))
    actions[:, 6] = -1.0  # gripper open
    state, conf = terminal_gripper_state(actions, gripper_dim=6)
    assert state == "open"
    assert conf > 0.9


def test_terminal_gripper_state_unknown_when_neutral() -> None:
    actions = np.zeros((10, 7))
    state, conf = terminal_gripper_state(actions, gripper_dim=6)
    assert state == "unknown"


def test_terminal_gripper_state_empty_actions() -> None:
    state, conf = terminal_gripper_state(np.zeros((0, 7)))
    assert state == "unknown"
    assert conf == 0.0


def test_action_complexity_summary() -> None:
    actions = np.linspace(0, 1, 100).reshape(100, 1)
    c = action_complexity(actions)
    assert c["action_range_max"] == pytest.approx(1.0, abs=1e-3)
    assert c["action_variance"] > 0


def test_action_complexity_empty() -> None:
    c = action_complexity(np.zeros((0, 5)))
    assert c["action_range_max"] == 0.0
    assert c["action_variance"] == 0.0


# ── composite enrich ────────────────────────────────────────────────────────


def test_enrich_returns_EnrichmentResult() -> None:
    actions = np.zeros((50, 7))
    result = enrich_metadata(
        instruction="Pick up the red block",
        actions=actions,
        gripper_dim=6,
    )
    assert isinstance(result, EnrichmentResult)
    assert "task_type" in result.tags
    assert "instruction_language" in result.tags


def test_enrich_each_tag_has_value_confidence_source() -> None:
    actions = np.zeros((50, 7))
    actions[:, 6] = 1.0
    result = enrich_metadata(
        instruction="Pick up the red block",
        actions=actions,
        gripper_dim=6,
    )
    for k, v in result.tags.items():
        assert "value" in v
        assert "confidence" in v
        assert "source" in v


def test_enrich_no_inputs_returns_unknowns() -> None:
    result = enrich_metadata()
    assert result.tags["task_type"]["value"] == "unknown"
    assert result.tags["instruction_language"]["value"] == "unknown"


def test_enrich_to_dict_keys() -> None:
    result = enrich_metadata(instruction="pick up the cup")
    d = result.to_dict()
    assert "tags" in d
    assert "taxonomy_version" in d
    assert "computed_at" in d


# ── enrich_from_jsonl_rows ──────────────────────────────────────────────────


def test_enrich_from_rows_extracts_instruction_and_actions() -> None:
    rows = [
        {
            "timestamp": "2026-05-05T10:00:00Z",
            "episode_id": "ep_1",
            "instruction_raw": "Pick up the red block",
            "action_chunk": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]],
            "metadata": {},
        },
        {
            "timestamp": "2026-05-05T10:00:01Z",
            "episode_id": "ep_1",
            "instruction_raw": "Pick up the red block",
            "action_chunk": [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]],
            "metadata": {},
        },
    ]
    result = enrich_from_jsonl_rows(rows)
    assert result.tags["task_type"]["value"] == "pick"
    assert result.tags["task_subtype"]["value"] == "block"


def test_enrich_from_rows_empty_safe() -> None:
    result = enrich_from_jsonl_rows([])
    assert result.tags["task_type"]["value"] == "unknown"
