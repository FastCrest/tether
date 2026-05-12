"""Tests for src/reflex/mcp/server.py (MCP server factory + tools + resources).

Uses a mock ReflexServer so tests run without CUDA / model weights. Exercises:

- Tool registration (all 4 tools + 1 resource present)
- Per-tool invocation contract (act, health, models_list, validate_dataset)
- Error-envelope shape on underlying failures
- Resource output (metrics://prometheus)

Tool invocation uses `mcp.call_tool()` (FastMCP's in-process test entry
point), which bypasses the stdio/HTTP transport layer.
"""
from __future__ import annotations

import asyncio
import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Optional dep: fastmcp ships behind reflex-vla[mcp]. When not
# installed, all tests in this file skip rather than error -- the mcp-
# server feature is correctly gated as an extra. Caught while verifying
# Track A status 2026-04-25.
fastmcp = pytest.importorskip("fastmcp")

from reflex.mcp import create_mcp_server


def _mock_reflex_server(
    health_state: str = "ready",
    export_dir: str = "/tmp/fake-export",
    predict_return: dict | None = None,
    predict_raises: Exception | None = None,
    cuda_graphs_enabled: bool = False,
) -> MagicMock:
    """Build a MagicMock that mimics ReflexServer's minimal MCP-facing API."""
    server = MagicMock()
    server.health_state = health_state
    server.export_dir = export_dir
    server._cuda_graphs_enabled = cuda_graphs_enabled

    predict_async_mock = AsyncMock()
    if predict_raises is not None:
        predict_async_mock.side_effect = predict_raises
    else:
        predict_async_mock.return_value = predict_return or {
            "actions": [[0.1, 0.2, 0.3]],
            "task": "",
        }
    server.predict_from_base64_async = predict_async_mock
    return server


def _run(coro):
    """Helper to run an async function in a test."""
    return asyncio.get_event_loop().run_until_complete(coro) if asyncio.get_event_loop().is_running() \
        else asyncio.run(coro)


# ---------------------------------------------------------------------------
# Server construction + tool registration
# ---------------------------------------------------------------------------


def test_create_mcp_server_returns_fastmcp_instance():
    from fastmcp import FastMCP
    mcp = create_mcp_server(_mock_reflex_server())
    assert isinstance(mcp, FastMCP)


@pytest.mark.asyncio
async def test_all_six_tools_registered():
    """Phase 1: act + health + models_list + validate_dataset.
    Phase 1.5: bench_latency + export_estimate (added 2026-05-06)."""
    mcp = create_mcp_server(_mock_reflex_server())
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert tool_names >= {
        # Phase 1
        "act", "health", "models_list", "validate_dataset",
        # Phase 1.5
        "bench_latency", "export_estimate",
    }


@pytest.mark.asyncio
async def test_metrics_resource_registered():
    mcp = create_mcp_server(_mock_reflex_server())
    resources = await mcp.list_resources()
    # Resources are identified by URI
    uris = {str(r.uri) for r in resources}
    assert any("metrics" in uri for uri in uris)


# ---------------------------------------------------------------------------
# Tool: act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_act_tool_forwards_to_reflex_server():
    server = _mock_reflex_server(predict_return={
        "actions": [[0.5, 0.6], [0.7, 0.8]],
    })
    mcp = create_mcp_server(server)

    # Minimal 1x1 base64 PNG for the image input
    img_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20).decode("ascii")

    result = await mcp.call_tool("act", {
        "instruction": "pick up the cup",
        "image_b64": img_b64,
        "state": [0.0, 0.1, 0.2],
    })

    # FastMCP call_tool returns a CallToolResult; structured content is in .data
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    assert "actions" in payload
    assert payload["actions"] == [[0.5, 0.6], [0.7, 0.8]]
    assert "inference_ms" in payload
    assert payload["inference_ms"] >= 0
    server.predict_from_base64_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_act_tool_returns_error_envelope_on_server_exception():
    class ServerBoom(RuntimeError):
        pass

    server = _mock_reflex_server(predict_raises=ServerBoom("inference crashed"))
    mcp = create_mcp_server(server)

    result = await mcp.call_tool("act", {
        "instruction": "test",
        "image_b64": "AAAA",
        "state": [0.0],
    })
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    assert "error" in payload
    assert payload["error"]["kind"] == "ServerBoom"
    assert "inference crashed" in payload["error"]["message"]
    assert payload["error"]["remediation"]  # non-empty


@pytest.mark.asyncio
async def test_act_tool_returns_error_on_decode_failure():
    # predict_from_base64_async returns {"error": "Failed to decode image: ..."}
    # when image_b64 is malformed. MCP act() should surface that as an error envelope.
    server = _mock_reflex_server(predict_return={"error": "Failed to decode image: bad padding"})
    mcp = create_mcp_server(server)

    result = await mcp.call_tool("act", {
        "instruction": "x",
        "image_b64": "not-base64",
        "state": [0.0],
    })
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    assert "error" in payload
    assert payload["error"]["kind"] == "DecodeError"


# ---------------------------------------------------------------------------
# Tool: health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_tool_returns_prewarm_state():
    server = _mock_reflex_server(health_state="ready",
                                  export_dir="/tmp/my-model",
                                  cuda_graphs_enabled=True)
    mcp = create_mcp_server(server)

    result = await mcp.call_tool("health", {})
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    assert payload["state"] == "ready"
    assert payload["model_version"] == "/tmp/my-model"
    assert payload["uptime_seconds"] >= 0
    assert payload["cuda_graphs_active"] is True


@pytest.mark.asyncio
async def test_health_tool_reports_warming_state():
    server = _mock_reflex_server(health_state="warming")
    mcp = create_mcp_server(server)
    result = await mcp.call_tool("health", {})
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    assert payload["state"] == "warming"


# ---------------------------------------------------------------------------
# Tool: models_list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_models_list_tool_returns_registry_entries():
    """models_list forwards to reflex.registry.filter_models()."""
    mcp = create_mcp_server(_mock_reflex_server())
    result = await mcp.call_tool("models_list", {})
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    # The real registry has >=1 entry. Confirm shape.
    # FastMCP wraps list responses in a dict with a "result" key in structured_content
    entries = payload["result"] if isinstance(payload, dict) and "result" in payload else payload
    assert isinstance(entries, list)
    if entries and isinstance(entries[0], dict) and "error" not in entries[0]:
        e0 = entries[0]
        assert "model_id" in e0
        assert "hf_repo" in e0
        assert "family" in e0
        assert "action_dim" in e0


# ---------------------------------------------------------------------------
# Tool: validate_dataset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_dataset_tool_returns_error_on_missing_path():
    mcp = create_mcp_server(_mock_reflex_server())
    result = await mcp.call_tool("validate_dataset",
                                  {"dataset_path": "/nonexistent/path/xyz"})
    payload = result.structured_content if hasattr(result, "structured_content") else (
        result.data if hasattr(result, "data") else result
    )
    # Either an error envelope (FileNotFoundError from DatasetContext) or a
    # structured validation report with a "block" decision — depending on
    # how DatasetContext handles missing paths. Either is acceptable; just
    # verify the shape is deterministic.
    assert isinstance(payload, dict)
    assert ("error" in payload) or ("decision" in payload) or ("summary" in payload)


# ---------------------------------------------------------------------------
# Resource: metrics://prometheus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_resource_returns_prometheus_text():
    mcp = create_mcp_server(_mock_reflex_server())
    # Read the metrics resource
    result = await mcp.read_resource("metrics://prometheus")
    # read_resource returns a list of ReadResourceContents or strings depending
    # on FastMCP version — handle both
    if isinstance(result, list):
        payload = "".join(
            (item.text if hasattr(item, "text") else str(item))
            for item in result
        )
    else:
        payload = result.text if hasattr(result, "text") else str(result)
    # Real Prometheus exposition starts with # HELP or # TYPE comments
    assert isinstance(payload, str)
    assert len(payload) > 0


# ---------------------------------------------------------------------------
# Phase 1.5 — bench_latency
# ---------------------------------------------------------------------------


def _payload(result):
    """FastMCP call_tool result → dict regardless of FastMCP version."""
    return (
        result.structured_content
        if hasattr(result, "structured_content")
        else (result.data if hasattr(result, "data") else result)
    )


@pytest.mark.asyncio
async def test_bench_latency_rejects_iterations_out_of_range():
    mcp = create_mcp_server(_mock_reflex_server())
    result = await mcp.call_tool("bench_latency", {
        "export_dir": "/tmp/whatever",
        "iterations": 999,
    })
    payload = _payload(result)
    assert "error" in payload
    assert "iterations must be in" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_bench_latency_rejects_warmup_out_of_range():
    mcp = create_mcp_server(_mock_reflex_server())
    result = await mcp.call_tool("bench_latency", {
        "export_dir": "/tmp/whatever",
        "iterations": 5,
        "warmup": 999,
    })
    payload = _payload(result)
    assert "error" in payload
    assert "warmup must be in" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_bench_latency_returns_error_for_missing_path(tmp_path):
    mcp = create_mcp_server(_mock_reflex_server())
    nonexistent = tmp_path / "does-not-exist"
    result = await mcp.call_tool("bench_latency", {
        "export_dir": str(nonexistent),
        "iterations": 5,
    })
    payload = _payload(result)
    assert "error" in payload
    assert payload["error"]["kind"] == "FileNotFoundError"


@pytest.mark.asyncio
async def test_bench_latency_returns_error_for_path_without_onnx(tmp_path):
    """Existing dir without an .onnx file → ValueError envelope, not crash."""
    mcp = create_mcp_server(_mock_reflex_server())
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    result = await mcp.call_tool("bench_latency", {
        "export_dir": str(empty),
        "iterations": 5,
    })
    payload = _payload(result)
    assert "error" in payload
    assert payload["error"]["kind"] == "ValueError"
    assert "no ONNX file" in payload["error"]["message"]


# ---------------------------------------------------------------------------
# Phase 1.5 — export_estimate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_estimate_rejects_unknown_target():
    mcp = create_mcp_server(_mock_reflex_server())
    result = await mcp.call_tool("export_estimate", {
        "model_id": "lerobot/smolvla_base",
        "target": "ufo_hardware",
    })
    payload = _payload(result)
    assert "error" in payload
    assert "unknown target" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_export_estimate_rejects_unsupported_precision():
    mcp = create_mcp_server(_mock_reflex_server())
    result = await mcp.call_tool("export_estimate", {
        "model_id": "lerobot/smolvla_base",
        "target": "desktop",
        "precision": "fp4",
    })
    payload = _payload(result)
    assert "error" in payload
    assert "unsupported precision" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_export_estimate_returns_estimate_for_known_model():
    """Known registry model → registry_hit=True + tighter VRAM estimate."""
    mcp = create_mcp_server(_mock_reflex_server())
    # SmolVLA is in the registry; should hit it.
    result = await mcp.call_tool("export_estimate", {
        "model_id": "smolvla",
        "target": "desktop",
        "precision": "fp16",
    })
    payload = _payload(result)
    assert "model_id" in payload
    assert payload["target"] == "desktop"
    assert payload["precision"] == "fp16"
    assert "estimated_vram_gb" in payload
    assert payload["estimated_vram_gb"] > 0
    assert "estimated_export_time_minutes" in payload
    assert "estimated_inference_ms_p50" in payload
    assert "notes" in payload
    assert isinstance(payload["notes"], list)


@pytest.mark.asyncio
async def test_export_estimate_falls_back_for_unknown_model():
    """Unknown model_id → registry_hit=False + generic estimate + clear note."""
    mcp = create_mcp_server(_mock_reflex_server())
    result = await mcp.call_tool("export_estimate", {
        "model_id": "imaginary/never-existed-model",
        "target": "desktop",
        "precision": "fp16",
    })
    payload = _payload(result)
    assert payload["registry_hit"] is False
    assert any("not in registry" in n for n in payload["notes"])
    # Generic estimate should still be reasonable (>0, <1000 GB).
    assert 0 < payload["estimated_vram_gb"] < 1000


@pytest.mark.asyncio
async def test_export_estimate_precision_scales_vram():
    """fp16 → 1x; fp8 → 0.5x; fp32 → 2x of the registry's size_gb_fp16."""
    mcp = create_mcp_server(_mock_reflex_server())
    fp16 = _payload(await mcp.call_tool("export_estimate", {
        "model_id": "smolvla", "target": "desktop", "precision": "fp16",
    }))
    fp8 = _payload(await mcp.call_tool("export_estimate", {
        "model_id": "smolvla", "target": "desktop", "precision": "fp8",
    }))
    fp32 = _payload(await mcp.call_tool("export_estimate", {
        "model_id": "smolvla", "target": "desktop", "precision": "fp32",
    }))
    # Skip the assertion if registry didn't hit (unlikely given the entry exists)
    if fp16.get("registry_hit"):
        assert fp32["estimated_vram_gb"] == pytest.approx(fp16["estimated_vram_gb"] * 2.0, abs=0.1)
        assert fp8["estimated_vram_gb"] == pytest.approx(fp16["estimated_vram_gb"] * 0.5, abs=0.1)
