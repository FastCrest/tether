"""SmolVLA serve-path tokenization must equal LeRobot's preprocessor.

Before 2026-10-28 the monolithic export traced a 16-token language input and the
server tokenized with max_length=16, truncation=True: LIBERO-10 instructions of
20 and 22 tokens were silently cut. These tests compare the serve tokenizer with
LeRobot 0.6.1's own NewLineTaskProcessorStep + TokenizerProcessorStep.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tether.runtime.smolvla_onnx_server import tokenize_smolvla_instruction

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

LONG = "put the white mug on the left plate and put the yellow and white mug on the right plate"


def _tokenizer():
    words = sorted(set((LONG + " extra words to overflow the limit").split()))
    vocab = {"<pad>": 0, "<unk>": 1, "\n": 2, **{w: i + 3 for i, w in enumerate(words)}}
    model = tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>")
    tok = tokenizers.Tokenizer(model)
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex(r"\s+|\n"), behavior="isolated"
    )
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>")
    return fast


def _lerobot(tokenizer, instruction: str, max_length: int):
    pytest.importorskip("lerobot")
    from lerobot.lerobot_types import TransitionKey
    from lerobot.processor.converters import create_transition
    from lerobot.processor.newline_task_processor import NewLineTaskProcessorStep
    from lerobot.processor.tokenizer_processor import TokenizerProcessorStep

    step = TokenizerProcessorStep(
        tokenizer=tokenizer, max_length=max_length, padding="max_length", padding_side="right"
    )
    transition = create_transition(observation={}, complementary_data={"task": instruction})
    transition = NewLineTaskProcessorStep()(transition)
    out = step(transition)
    obs = out[TransitionKey.OBSERVATION]
    return (
        obs["observation.language.tokens"].numpy(),
        obs["observation.language.attention_mask"].numpy().astype(bool),
    )


def _real_tokens(tokenizer, instruction: str) -> int:
    return len(tokenizer(instruction + "\n", truncation=False, padding=False)["input_ids"])


def test_long_instruction_that_16_tokens_cut_now_matches_lerobot_at_48():
    tok = _tokenizer()
    assert _real_tokens(tok, LONG) > 16  # would have been truncated by the old graph
    tokens, masks, report = tokenize_smolvla_instruction(tok, LONG, 48)
    ref_tokens, ref_masks = _lerobot(tok, LONG, 48)
    np.testing.assert_array_equal(tokens, ref_tokens)
    np.testing.assert_array_equal(masks, ref_masks)
    assert report == {"tokens": int(ref_masks.sum()), "max_tokens": 48, "truncated": False, "dropped_tokens": 0}


def test_truncation_matches_lerobot_but_is_reported():
    tok = _tokenizer()
    instruction = " ".join([LONG] * 3)
    full = _real_tokens(tok, instruction)
    assert full > 48
    tokens, masks, report = tokenize_smolvla_instruction(tok, instruction, 48)
    ref_tokens, ref_masks = _lerobot(tok, instruction, 48)
    np.testing.assert_array_equal(tokens, ref_tokens)  # same cut as LeRobot ...
    np.testing.assert_array_equal(masks, ref_masks)
    assert report["truncated"] is True  # ... but not silent
    assert report["tokens"] == full and report["dropped_tokens"] == full - 48


def test_newline_is_not_doubled():
    tok = _tokenizer()
    a, _, _ = tokenize_smolvla_instruction(tok, LONG, 48)
    b, _, _ = tokenize_smolvla_instruction(tok, LONG + "\n", 48)
    np.testing.assert_array_equal(a, b)


def test_language_contract_follows_the_lerobot_config():
    from tether.exporters.monolithic import smolvla_language_contract

    cfg = SimpleNamespace(
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        tokenizer_max_length=48,
        pad_language_to="max_length",
    )
    contract = smolvla_language_contract(cfg)
    assert contract["max_length"] == 48
    assert contract["tokenizer_ref"] == cfg.vlm_model_name
    assert contract["padding_side"] == "right" and contract["truncation"] is True
    assert contract["append_newline"] is True
