"""Human CLI rendering must not abort on redirected Windows code pages."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys

import pytest
import typer
from rich.logging import RichHandler
from rich.table import Table

from tether import cli


@pytest.mark.parametrize("encoding", ["cp1252", "ascii", "utf-8"])
def test_cli_console_renders_status_table_and_paths(monkeypatch, encoding):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding, errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    table = Table(title="Export ✓")
    table.add_column("Result")
    table.add_row("✓ model → /exports/策略/model.onnx")

    cli.console.print(table)
    stream.flush()
    output = raw.getvalue().decode(encoding)
    assert "model.onnx" in output
    if encoding == "utf-8":
        assert "✓" in output and "策略" in output
    else:
        assert r"\u2713" in output and r"\u7b56\u7565" in output
    assert stream.errors == "strict"


def test_logging_and_cli_errors_keep_json_stdout_clean(monkeypatch):
    raw_out, raw_err = io.BytesIO(), io.BytesIO()
    stdout = io.TextIOWrapper(raw_out, encoding="cp1252", errors="strict")
    stderr = io.TextIOWrapper(raw_err, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    configured = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: configured.update(kwargs))

    cli._setup_logging()
    handler = configured["handlers"][0]
    assert isinstance(handler, RichHandler)
    record = logging.LogRecord("export", logging.WARNING, __file__, 1, "✓ export → ready", (), None)
    handler.emit(record)
    cli.err_console.print("[red]⚠ export diagnostic[/red]")
    receipt = {"status": "ready", "export": "/exports/策略/model.onnx"}
    typer.echo(json.dumps(receipt))
    stdout.flush()
    stderr.flush()

    assert json.loads(raw_out.getvalue().decode("cp1252")) == receipt
    assert "export diagnostic" in raw_err.getvalue().decode("cp1252")
    assert r"\u2713" in raw_err.getvalue().decode("cp1252")
    assert stdout.errors == stderr.errors == "strict"


def test_explicit_stream_can_be_replaced():
    first, second = io.StringIO(), io.StringIO()
    console = type(cli.console)(file=first)
    console.print("first")
    console.file = second
    console.print("second")
    assert first.getvalue() == "first\n"
    assert second.getvalue() == "second\n"


def test_real_cli_entrypoint_on_strict_cp1252():
    env = dict(
        os.environ,
        PYTHONIOENCODING="cp1252:strict",
        TETHER_SKIP_ONBOARDING="1",
        TETHER_NO_UPGRADE_CHECK="1",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tether.cli"],
        env=env,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("cp1252")
    assert "Core workflow:" in result.stdout.decode("cp1252")
    assert r"\u21b3" in result.stdout.decode("cp1252")
    assert b"UnicodeEncodeError" not in result.stderr


def test_chat_tool_diagnostic_on_strict_cp1252(monkeypatch):
    from tether.chat.console import _make_event_handler, _UIState

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    handler = _make_event_handler(_UIState())
    handler({"kind": "tool_start", "name": "export", "args": {"model": "策略"}})
    stream.flush()
    assert "export" in raw.getvalue().decode("cp1252")
    assert r"\u7b56\u7565" in raw.getvalue().decode("cp1252")


@pytest.mark.parametrize("encoding", ["cp1252", "utf-8"])
def test_chat_tokens_preserve_whitespace_and_flush(monkeypatch, encoding):
    from tether.chat.console import _UIState

    class ObservedStream(io.TextIOWrapper):
        flush_count = 0

        def flush(self):
            self.flush_count += 1
            return super().flush()

    raw = io.BytesIO()
    stream = ObservedStream(raw, encoding=encoding, errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    state = _UIState(streaming_reply=True)
    state.on_token("  ✓")
    state.on_token("\n策略  ")
    state.end_stream()
    expected = "\n  ✓\n策略  \n\n"
    if encoding == "cp1252":
        expected = expected.encode(encoding, errors="backslashreplace").decode(encoding)
    assert raw.getvalue().decode(encoding) == expected
    assert stream.flush_count == 4
    assert not state._stream_started and not state.streaming_reply
    assert stream.errors == "strict"


def test_eval_error_keeps_diagnostic_on_strict_cp1252(monkeypatch, tmp_path):
    from tether.eval import cli as eval_cli

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stderr", stream)

    def fail_eval(config):
        raise ValueError("✓ invalid policy 策略")

    monkeypatch.setattr(eval_cli, "run_aloha_eval", fail_eval)
    with pytest.raises(typer.Exit) as exc:
        eval_cli.aloha_eval_command(
            policy="/policy",
            revision="",
            task="AlohaTransferCube-v0",
            device="cpu",
            seed=0,
            num_episodes=3,
            output=str(tmp_path),
            python="",
            timeout_s=30,
        )
    assert exc.value.exit_code == 1
    stream.flush()
    assert r"\u2713 invalid policy \u7b56\u7565" in raw.getvalue().decode("cp1252")
