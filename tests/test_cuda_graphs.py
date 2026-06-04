"""Unit tests for src/tether/runtime/cuda_graphs.py.

These tests use mock ORT sessions to exercise the wrapper's metric emission +
fallback logic without requiring a GPU. Integration tests covering the actual
ORT capture/replay behavior live in tests/test_cuda_graphs_integration.py and
are gated on CUDA availability.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tether.observability.prometheus import (
    REGISTRY,
    tether_cuda_graph_capture_failed_at_init_total,
    tether_cuda_graph_captured_total,
    tether_cuda_graph_eager_fallback_total,
    tether_cuda_graph_replayed_total,
)
from tether.runtime.cuda_graphs import (
    VALID_SESSION_NAMES,
    CudaGraphWrapper,
    EagerSessionWrapper,
    build_cuda_graph_providers,
    try_capture_or_fall_back,
)


# ---------------------------------------------------------------------------
# Phase 1 Day 8+ CLI / create_app wiring
# ---------------------------------------------------------------------------

def test_cli_serve_help_advertises_cuda_graphs_flag():
    """Guard against accidental --cuda-graphs flag removal or rename."""
    from typer.testing import CliRunner

    from tether.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--cuda-graphs" in result.output


def test_create_app_accepts_cuda_graphs_enabled_kwarg():
    """create_app() must accept cuda_graphs_enabled with default False per ADR
    2026-04-24-cuda-graphs-architecture. Catches signature drift."""
    import inspect

    from tether.runtime.server import create_app

    sig = inspect.signature(create_app)
    assert "cuda_graphs_enabled" in sig.parameters
    assert sig.parameters["cuda_graphs_enabled"].default is False


# ---------------------------------------------------------------------------
# build_cuda_graph_providers
# ---------------------------------------------------------------------------

def test_build_providers_enabled_sets_cuda_graph_flag():
    providers = build_cuda_graph_providers(enabled=True)
    cuda_entry = providers[0]
    assert cuda_entry[0] == "CUDAExecutionProvider"
    assert cuda_entry[1] == {"enable_cuda_graph": "1"}
    assert providers[-1] == "CPUExecutionProvider"


def test_build_providers_disabled_has_empty_cuda_opts():
    providers = build_cuda_graph_providers(enabled=False)
    cuda_entry = providers[0]
    assert cuda_entry[0] == "CUDAExecutionProvider"
    assert cuda_entry[1] == {}
    assert providers[-1] == "CPUExecutionProvider"


# ---------------------------------------------------------------------------
# CudaGraphWrapper construction
# ---------------------------------------------------------------------------

def test_rejects_invalid_session_name():
    mock_sess = MagicMock()
    with pytest.raises(ValueError, match="session_name"):
        CudaGraphWrapper(mock_sess, session_name="bogus", embodiment="franka", model_id="m1")


def test_accepts_both_valid_session_names():
    mock_sess = MagicMock()
    for name in VALID_SESSION_NAMES:
        w = CudaGraphWrapper(mock_sess, session_name=name, embodiment="franka", model_id="m1")
        assert w.captured is False


# ---------------------------------------------------------------------------
# Capture + replay metric emission
# ---------------------------------------------------------------------------

def _get_counter(counter, labels: dict) -> float:
    """Fetch the current value of a Prometheus Counter for given labels."""
    return counter.labels(**labels)._value.get()


def test_first_run_increments_captured_and_replayed():
    mock_sess = MagicMock()
    mock_sess.run.return_value = ["output"]
    w = CudaGraphWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-test-1")

    cap_before = _get_counter(
        tether_cuda_graph_captured_total,
        {"embodiment": "franka", "model_id": "cg-test-1", "session": "vlm_prefix"},
    )
    rep_before = _get_counter(
        tether_cuda_graph_replayed_total,
        {"embodiment": "franka", "model_id": "cg-test-1", "session": "vlm_prefix"},
    )

    result = w.run(None, {"x": [1, 2, 3]})

    assert result == ["output"]
    assert w.captured is True
    assert _get_counter(
        tether_cuda_graph_captured_total,
        {"embodiment": "franka", "model_id": "cg-test-1", "session": "vlm_prefix"},
    ) == cap_before + 1
    assert _get_counter(
        tether_cuda_graph_replayed_total,
        {"embodiment": "franka", "model_id": "cg-test-1", "session": "vlm_prefix"},
    ) == rep_before + 1


def test_subsequent_runs_only_increment_replayed():
    mock_sess = MagicMock()
    mock_sess.run.return_value = ["output"]
    w = CudaGraphWrapper(mock_sess, "expert_denoise", embodiment="so100", model_id="cg-test-2")

    w.run(None, {"x": [1]})  # first call (capture)

    cap_after_first = _get_counter(
        tether_cuda_graph_captured_total,
        {"embodiment": "so100", "model_id": "cg-test-2", "session": "expert_denoise"},
    )
    rep_after_first = _get_counter(
        tether_cuda_graph_replayed_total,
        {"embodiment": "so100", "model_id": "cg-test-2", "session": "expert_denoise"},
    )

    for _ in range(5):
        w.run(None, {"x": [1]})

    # Captured stayed flat
    assert _get_counter(
        tether_cuda_graph_captured_total,
        {"embodiment": "so100", "model_id": "cg-test-2", "session": "expert_denoise"},
    ) == cap_after_first
    # Replayed incremented by 5
    assert _get_counter(
        tether_cuda_graph_replayed_total,
        {"embodiment": "so100", "model_id": "cg-test-2", "session": "expert_denoise"},
    ) == rep_after_first + 5


# ---------------------------------------------------------------------------
# Fallback behavior on exception
# ---------------------------------------------------------------------------

def test_capture_exception_increments_fallback_and_reraises():
    class CudaCaptureError(RuntimeError):
        pass

    mock_sess = MagicMock()
    mock_sess.run.side_effect = CudaCaptureError("mock capture fail")
    w = CudaGraphWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-test-3")

    fb_before = _get_counter(
        tether_cuda_graph_eager_fallback_total,
        {"embodiment": "franka", "model_id": "cg-test-3", "reason": "capture_failed"},
    )

    with pytest.raises(CudaCaptureError):
        w.run(None, {})

    assert w.captured is False  # never flipped
    assert _get_counter(
        tether_cuda_graph_eager_fallback_total,
        {"embodiment": "franka", "model_id": "cg-test-3", "reason": "capture_failed"},
    ) == fb_before + 1


def test_replay_exception_increments_fallback_with_replay_reason():
    class CudaReplayError(RuntimeError):
        pass

    mock_sess = MagicMock()
    # First call succeeds (capture), second raises (replay)
    mock_sess.run.side_effect = [["ok"], CudaReplayError("mock replay fail")]

    w = CudaGraphWrapper(mock_sess, "expert_denoise", embodiment="ur5", model_id="cg-test-4")
    w.run(None, {})  # captures
    assert w.captured is True

    fb_before = _get_counter(
        tether_cuda_graph_eager_fallback_total,
        {"embodiment": "ur5", "model_id": "cg-test-4", "reason": "replay_failed"},
    )

    with pytest.raises(CudaReplayError):
        w.run(None, {})

    assert _get_counter(
        tether_cuda_graph_eager_fallback_total,
        {"embodiment": "ur5", "model_id": "cg-test-4", "reason": "replay_failed"},
    ) == fb_before + 1


# ---------------------------------------------------------------------------
# invalidate()
# ---------------------------------------------------------------------------

def test_invalidate_resets_captured_flag():
    mock_sess = MagicMock()
    mock_sess.run.return_value = ["out"]
    w = CudaGraphWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-test-5")

    w.run(None, {})
    assert w.captured is True

    w.invalidate()
    assert w.captured is False

    # Next run is treated as capture (increments captured counter)
    cap_before = _get_counter(
        tether_cuda_graph_captured_total,
        {"embodiment": "franka", "model_id": "cg-test-5", "session": "vlm_prefix"},
    )
    w.run(None, {})
    assert w.captured is True
    assert _get_counter(
        tether_cuda_graph_captured_total,
        {"embodiment": "franka", "model_id": "cg-test-5", "session": "vlm_prefix"},
    ) == cap_before + 1


# ---------------------------------------------------------------------------
# Pass-through
# ---------------------------------------------------------------------------

def test_run_passes_output_names_and_feed_through():
    mock_sess = MagicMock()
    mock_sess.run.return_value = ["tensor"]

    w = CudaGraphWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-test-6")
    feed = {"lang_tokens": [1, 2, 3], "state": [0.1, 0.2]}
    output_names = ["past_k_0", "past_v_0"]

    w.run(output_names, feed)
    mock_sess.run.assert_called_with(output_names, feed)


def test_session_property_exposes_raw_session():
    mock_sess = MagicMock()
    w = CudaGraphWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-test-7")
    assert w.session is mock_sess


# ---------------------------------------------------------------------------
# EagerSessionWrapper
# ---------------------------------------------------------------------------

def test_eager_session_wrapper_captured_is_false():
    mock_sess = MagicMock()
    w = EagerSessionWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-eager-1")
    assert w.captured is False
    assert w.session is mock_sess


def test_eager_session_wrapper_run_forwards_to_session():
    mock_sess = MagicMock()
    mock_sess.run.return_value = ["out"]
    w = EagerSessionWrapper(mock_sess, "expert_denoise", embodiment="franka", model_id="cg-eager-2")
    result = w.run(None, {"x": [1]})
    assert result == ["out"]
    mock_sess.run.assert_called_with(None, {"x": [1]})


def test_eager_session_wrapper_invalidate_is_noop():
    mock_sess = MagicMock()
    w = EagerSessionWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-eager-3")
    w.invalidate()  # no-op, doesn't raise
    assert w.captured is False


def test_eager_session_wrapper_rejects_invalid_session_name():
    mock_sess = MagicMock()
    with pytest.raises(ValueError, match="session_name"):
        EagerSessionWrapper(mock_sess, "bogus", embodiment="franka", model_id="m1")


# ---------------------------------------------------------------------------
# try_capture_or_fall_back
# ---------------------------------------------------------------------------

def _make_mock_session_with_inputs():
    """Mock ORT session that responds to get_inputs() with one float input."""
    mock_sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "x"
    mock_input.shape = [1, 4]
    mock_input.type = "tensor(float)"
    mock_sess.get_inputs.return_value = [mock_input]
    return mock_sess


def test_try_capture_returns_cuda_graph_wrapper_on_success():
    mock_sess = _make_mock_session_with_inputs()
    mock_sess.run.return_value = ["captured_output"]

    def factory(cg_enabled):
        return mock_sess

    result = try_capture_or_fall_back(
        factory, session_name="expert_denoise",
        embodiment="franka", model_id="cg-try-1",
    )
    assert isinstance(result, CudaGraphWrapper)
    assert result.captured is True
    assert result.session is mock_sess


def test_try_capture_returns_eager_wrapper_on_capture_failure():
    class OOMError(RuntimeError):
        pass

    capture_sess = _make_mock_session_with_inputs()
    capture_sess.run.side_effect = OOMError("BFC arena alloc failed")

    eager_sess = _make_mock_session_with_inputs()

    call_count = {"n": 0}

    def factory(cg_enabled):
        call_count["n"] += 1
        # First call: cg_enabled=True → capture session (raises)
        # Second call: cg_enabled=False → eager fallback session
        return capture_sess if cg_enabled else eager_sess

    fb_before = _get_counter(
        tether_cuda_graph_capture_failed_at_init_total,
        {"embodiment": "franka", "model_id": "cg-try-2",
         "session": "vlm_prefix", "reason": "OOMError"},
    )

    result = try_capture_or_fall_back(
        factory, session_name="vlm_prefix",
        embodiment="franka", model_id="cg-try-2",
    )

    assert isinstance(result, EagerSessionWrapper)
    assert result.captured is False
    assert result.session is eager_sess
    assert call_count["n"] == 2  # tried capture, then built eager
    assert _get_counter(
        tether_cuda_graph_capture_failed_at_init_total,
        {"embodiment": "franka", "model_id": "cg-try-2",
         "session": "vlm_prefix", "reason": "OOMError"},
    ) == fb_before + 1


def test_try_capture_eager_wrapper_forwards_run_to_session():
    """Integration: after graceful-degrade, `.run()` on the returned wrapper
    should forward to the eager session and NOT re-trigger capture."""
    class OOMError(RuntimeError):
        pass

    capture_sess = _make_mock_session_with_inputs()
    capture_sess.run.side_effect = OOMError("BFC alloc fail")

    eager_sess = _make_mock_session_with_inputs()
    eager_sess.run.return_value = ["eager_result"]

    def factory(cg_enabled):
        return capture_sess if cg_enabled else eager_sess

    result = try_capture_or_fall_back(
        factory, session_name="vlm_prefix",
        embodiment="franka", model_id="cg-try-3",
    )

    feed = {"x": [[1.0, 2.0, 3.0, 4.0]]}
    out = result.run(None, feed)
    assert out == ["eager_result"]
    eager_sess.run.assert_called_with(None, feed)
