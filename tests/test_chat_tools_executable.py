"""Every chat tool must route to a real `tether` CLI command.

This test catches the v0.3.0 regression where 4 of 16 chat tools (`list_traces`,
`show_status`, `show_config`, `replay_trace`) routed to non-existent CLI commands
or wrong signatures, so the chat agent silently failed when users invoked them.

For each tool in `tether.chat.schema.TOOLS`, we synthesize valid placeholder args
from its JSON schema, build the argv via the executor, and run the resulting
`tether <subcommand> --help`. A `--help` invocation succeeds (exit=0) iff the
command exists. We don't actually execute the tool — that would require the full
runtime stack and may have side effects.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

import pytest

from tether.chat.executor import _argv_for
from tether.chat.schema import TOOLS

TETHER_BIN = shutil.which("tether")


def _placeholder_args(schema: dict[str, Any]) -> dict[str, Any]:
    """Build the smallest valid argument dict from a tool's JSON schema."""
    props = schema.get("properties", {})
    required = schema.get("required", [])
    out: dict[str, Any] = {}
    for name in required:
        prop = props.get(name, {})
        t = prop.get("type")
        if t == "string":
            if "enum" in prop:
                out[name] = prop["enum"][0]
            else:
                out[name] = "placeholder"
        elif t == "integer":
            out[name] = 1
        elif t == "boolean":
            out[name] = False
    return out


@pytest.mark.skipif(TETHER_BIN is None, reason="tether CLI not on PATH")
@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t["function"]["name"])
def test_tool_routes_to_real_cli_command(tool: dict[str, Any]) -> None:
    name = tool["function"]["name"]
    schema = tool["function"]["parameters"]
    args = _placeholder_args(schema)
    argv = _argv_for(name, args)

    # Strip placeholder positional / flag values; we only want to check the
    # subcommand path exists, not invoke the operation.
    subcommand_path: list[str] = []
    for a in argv:
        if a.startswith("--"):
            break
        subcommand_path.append(a)

    cmd = [TETHER_BIN, *subcommand_path, "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, (
        f"Chat tool '{name}' routes to `{' '.join(subcommand_path)}` "
        f"which exited {proc.returncode}.\n"
        f"argv: {argv}\nstderr: {proc.stderr[:400]}"
    )
