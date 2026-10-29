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


def _unloaded_server(tmp_path, *, strict: bool):
    from unittest.mock import MagicMock

    from tether.runtime.smolvla_onnx_server import SmolVLAOnnxServer

    srv = SmolVLAOnnxServer(str(tmp_path), strict_providers=strict)
    srv._ready = True
    srv._session = MagicMock()
    srv._session.run.return_value = [np.zeros((1, 50, 32), np.float32)]
    srv._input_names = ["lang_tokens", "lang_masks", "state", "noise"]
    srv._lang_len = 48
    srv._tokenizer_unavailable = True  # tokenizer failed to load
    return srv


def test_strict_server_refuses_to_act_on_an_instruction_it_cannot_tokenize(tmp_path):
    srv = _unloaded_server(tmp_path, strict=True)
    out = srv.predict(instruction="pick up the red cup")
    assert "error" in out and "actions" not in out
    assert "tokenizer" in out["error"]
    srv._session.run.assert_not_called()


def test_non_strict_server_proceeds_only_with_the_visible_label(tmp_path):
    srv = _unloaded_server(tmp_path, strict=False)
    out = srv.predict(instruction="pick up the red cup")
    assert out["language"]["source"] == "unavailable-zero-tokens"
    assert "actions" in out


def test_truncation_warning_does_not_log_the_unbounded_instruction(caplog):
    import hashlib
    import logging

    tok = _tokenizer()
    instruction = " ".join([LONG] * 200)  # far beyond 48 tokens and the log cap
    with caplog.at_level(logging.WARNING, logger="tether.runtime.smolvla_onnx_server"):
        _, _, report = tokenize_smolvla_instruction(tok, instruction, 48)
    assert report["truncated"] is True
    (record,) = [r for r in caplog.records if "truncated" in r.getMessage()]
    message = record.getMessage()
    assert len(message) < 400
    assert instruction not in message
    assert hashlib.sha256(instruction.encode()).hexdigest()[:12] in message


class _CountingTokenizer:
    """Wraps a tokenizer and counts encode calls (``__call__``)."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.inner(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("repeat", [1, 3])  # fits in 48 / overflows 48
def test_one_tokenizer_pass_matches_lerobot_and_reports_overflow(side, repeat):
    inner = _tokenizer()
    inner.truncation_side = side  # SmolVLM2's tokenizer truncates on the left
    counting = _CountingTokenizer(inner)
    instruction = " ".join([LONG] * repeat)
    tokens, masks, report = tokenize_smolvla_instruction(counting, instruction, 48)
    assert counting.calls == 1
    ref_tokens, ref_masks = _lerobot(inner, instruction, 48)
    np.testing.assert_array_equal(tokens, ref_tokens)
    np.testing.assert_array_equal(masks, ref_masks)
    full = _real_tokens(inner, instruction)
    assert report["tokens"] == full and report["truncated"] == (full > 48)


def test_real_smolvlm2_tokenizer_matches_lerobot_when_cached():
    try:
        real = transformers.AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True
        )
    except OSError:
        pytest.skip("SmolVLM2 tokenizer not cached")
    for instruction in (LONG, " ".join([LONG] * 4)):
        counting = _CountingTokenizer(real)
        tokens, masks, report = tokenize_smolvla_instruction(counting, instruction, 48)
        ref_tokens, ref_masks = _lerobot(real, instruction, 48)
        assert counting.calls == 1
        np.testing.assert_array_equal(tokens, ref_tokens)
        np.testing.assert_array_equal(masks, ref_masks)
        assert report["truncated"] == (report["tokens"] > 48)
