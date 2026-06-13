from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tether.cli import app
from tether.runtime.record import RecordWriter
from tether.shadow_rollout import run_shadow_rollout_gate


runner = CliRunner()


def _write_shadow_trace(
    tmp_path: Path,
    *,
    pending: bool = False,
    shadow_actions: list[list[float]] | None = None,
) -> Path:
    writer = RecordWriter(
        record_dir=tmp_path / ("pending_trace" if pending else "ready_trace"),
        model_hash="deadbeefcafe0000",
        config_hash="0011223344556677",
        export_dir=str(tmp_path / "fake_export"),
        model_type="pi0.5",
        export_kind="monolithic",
        providers=["CPUExecutionProvider"],
        gzip_output=False,
    )
    seq = writer.write_request(
        chunk_id=0,
        image_b64="aGVsbG8=",
        instruction="pick",
        state=[0.1, 0.2],
        actions=[[0.1, 0.2]],
        action_dim=2,
        latency_total_ms=100.0,
        routing={
            "shadow_sampled": True,
            "shadow_mode": "background",
            "shadow_pending": pending,
        },
    )
    if not pending:
        actions = shadow_actions or [[0.11, 0.21]]
        writer.write_shadow_result(
            seq=seq,
            actions=actions,
            action_dim=2,
            latency_total_ms=12.0,
            routing={
                "shadow_sampled": True,
                "shadow_mode": "background",
                "shadow_actions": actions,
                "shadow_latency_ms": 12.0,
            },
        )
    writer.write_footer({"total_requests": 1})
    writer.close()
    return writer.filepath


def test_shadow_rollout_gate_promotes_ready_shadow_trace(tmp_path: Path) -> None:
    trace = _write_shadow_trace(tmp_path)
    packet_dir = tmp_path / "packet"

    report = run_shadow_rollout_gate(
        trace=trace,
        packet_dir=packet_dir,
        profile="lab-shadow",
        min_compared=1,
        fail_on="any",
    )

    assert report["decision"] == "PROMOTE"
    assert report["policy_diff"]["summary"]["compared"] == 1
    assert (packet_dir / "deployment-proof.json").exists()
    assert (packet_dir / "policy-diff.json").exists()
    assert (packet_dir / "promotion-decision.json").exists()
    manifest = json.loads((packet_dir / "MANIFEST.json").read_text())
    assert {item["name"] for item in manifest["files"]} == {
        "deployment-proof.json",
        "policy-diff.json",
    }


def test_shadow_rollout_gate_holds_pending_shadow_trace(tmp_path: Path) -> None:
    trace = _write_shadow_trace(tmp_path, pending=True)

    report = run_shadow_rollout_gate(
        trace=trace,
        packet_dir=tmp_path / "packet",
        profile="lab-shadow",
        min_compared=1,
        wait_timeout_s=0.0,
        fail_on="any",
    )

    assert report["decision"] == "HOLD"
    assert report["promotion"]["decision"] == "BLOCK"
    assert "policy_diff_verdict" in report["promotion"]["failed_checks"]
    assert "policy_shadow_pending" in report["promotion"]["failed_checks"]


def test_shadow_gate_cli_json(tmp_path: Path) -> None:
    trace = _write_shadow_trace(tmp_path)
    packet_dir = tmp_path / "cli_packet"

    result = runner.invoke(
        app,
        [
            "policy",
            "shadow-gate",
            str(trace),
            "--packet-dir",
            str(packet_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["decision"] == "PROMOTE"
    assert body["policy_diff"]["summary"]["compared"] == 1
