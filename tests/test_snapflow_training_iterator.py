"""CPU- and LeRobot-independent tests for SnapFlow batch iteration."""

from __future__ import annotations

import pytest

from tether.finetune.backends.snapflow_backend import (
    EmptyTrainingEpochError,
    _iter_training_batches,
)


class TestTrainingBatchIterator:
    def test_reiterates_loader_to_exact_step_count(self):
        batches = list(_iter_training_batches(["A", "B"], num_steps=5))
        assert batches == [
            (1, "A"),
            (2, "B"),
            (3, "A"),
            (4, "B"),
            (5, "A"),
        ]

    def test_empty_loader_fails_instead_of_spinning(self):
        with pytest.raises(EmptyTrainingEpochError) as exc_info:
            list(_iter_training_batches([], num_steps=5))

        assert exc_info.value.completed_steps == 0
        assert exc_info.value.requested_steps == 5
        assert "batch_size" in str(exc_info.value)
        assert "drop_last" in str(exc_info.value)

    def test_loader_exception_propagates(self):
        class BrokenLoader:
            def __iter__(self):
                raise RuntimeError("dataset read failed")

        with pytest.raises(RuntimeError, match="dataset read failed"):
            list(_iter_training_batches(BrokenLoader(), num_steps=2))
