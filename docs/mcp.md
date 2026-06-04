# MCP integration

Tether exposes a [Model Context Protocol](https://spec.modelcontextprotocol.io/) server so MCP-compatible agents (Claude Desktop, Cursor, custom) can call a VLA policy as a tool. Additive to the HTTP API — both share the same inference engine.

## Install

```bash
pip install tether[mcp]
```

Pulls [`fastmcp`](https://github.com/jlowin/fastmcp) >= 3.0 alongside the core dependencies.

## Quick start: Claude Desktop / Cursor integration

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent on Linux/Windows:

```json
{
  "mcpServers": {
    "tether": {
      "command": "tether",
      "args": [
        "serve",
        "/path/to/your/exported/model/",
        "--mcp",
        "--mcp-transport", "stdio",
        "--embodiment", "franka"
      ]
    }
  }
}
```

Claude Desktop spawns Tether as a subprocess; stdio transport means no ports, no firewall, no auth. Cursor's MCP config is analogous.

Restart Claude Desktop. `tether` now appears in the tool picker. Call it like any MCP tool:

```
/tether act
  instruction: "pick up the red block"
  image_b64: "<base64 PNG from robot camera>"
  state: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  episode_id: "ep-2026-04-24-001"
```

## HTTP transport (for networked agents)

```bash
tether serve ./my-export/ \
  --mcp \
  --mcp-transport http \
  --mcp-port 8001 \
  --port 8000 \
  --embodiment franka
```

Both MCP (streamable-http on `127.0.0.1:8001`) AND the REST API (`0.0.0.0:8000`) run concurrently. The same `ReflexServer` backs both.

For production HTTP deployment, front MCP with a reverse proxy that handles TLS + auth. MCP doesn't ship its own TLS layer.

## Available tools

| Tool | Inputs | Output |
|---|---|---|
| `act` | `instruction`, `image_b64`, `state`, `episode_id?` | `{actions: [[float]], policy_version, inference_ms}` or `{error: ...}` |
| `health` | — | `{state, model_version, uptime_seconds, cuda_graphs_active}` |
| `models_list` | — | `[{model_id, hf_repo, family, action_dim, size_mb, supported_embodiments, supported_devices, license, description}]` |
| `validate_dataset` | `dataset_path` | Validation report with `pass`/`warn`/`block` decision + per-check details |

## Available resources

| URI | Content |
|---|---|
| `metrics://prometheus` | Current Prometheus metrics in text exposition format (same as the `/metrics` HTTP endpoint) |

## Safety

The `act` tool returns action chunks but does NOT actuate them. Callers are responsible for sending actions to the robot's actuation controller (SO-ARM / Trossen / ROS2). Tether's `act` is pure inference.

Safety features that DO run inside `act`:

- `ActionGuard` from the loaded embodiment config (joint-limit clamping, velocity caps, torque caps)
- Per-request circuit breaker (`--max-consecutive-crashes`)
- Optional audit log (`--record <dir>`)

Shadow actions, A/B policy routing, and dataset validation run via explicit tools (the caller decides when to invoke).

## Troubleshooting

**"fastmcp not installed"**
```bash
pip install tether[mcp]
```

**Claude Desktop doesn't list Tether as a tool**
Verify the `claude_desktop_config.json` path. On macOS, quit + relaunch Claude Desktop fully (cmd-Q, not just close the window).

**"Could not find ReflexServer on the app state"**
This shouldn't happen in released versions — file a bug at github.com/FastCrest/tether/issues.

**stdio mode blocks the terminal**
By design. stdio owns stdin/stdout for MCP's bidirectional framing. For interactive dev, use `--mcp-transport http` on a separate port.

**MCP + FastAPI on same port**
Not supported — the two transports use different protocols. `--port` is FastAPI; `--mcp-port` is MCP HTTP; they must differ.

## ROS2 tools (ros2-mcp-bridge)

When running `tether ros2-serve` alongside MCP, four ROS2 tools + one resource become available. They introspect + drive a real ROS2-connected robot through the same MCP surface your agent already queries.

### Tools

| Tool | Purpose | Cooldown | Requires `confirm=True` |
|---|---|---|---|
| `get_joint_state()` | Latest `/joint_states` positions | 100 ms | — |
| `get_camera_frame()` | Latest `/camera/image_raw` as base64 JPEG | 100 ms | — |
| `execute_task(instruction, confirm, max_steps?)` | Runs one Tether inference cycle + publishes the action chunk | 5 s | ✅ |
| `emergency_stop(confirm)` | Publishes to `/tether/e_stop` | 5 s | ✅ |
| `robot://status` (resource) | URDF info + latest state + current task | — | — |

### The `confirm=True` tripwire

`execute_task` and `emergency_stop` will **reject** any call where `confirm` is not literally the boolean `True` — not truthy, not a string. Concretely:

```python
await execute_task(instruction="pick up the red block", confirm=True)
# → runs inference + actuates

await execute_task(instruction="pick up the red block")
# → {"error": {"kind": "ConfirmationRequired", ...}}

await execute_task(instruction="pick up", confirm="yes")
# → {"error": {"kind": "ConfirmationRequired", ...}}  # not identity-True
```

This is intentional friction. An agent forgetting to pass `confirm=True` can't actuate by accident; a prompt-injection payload that tries to slip in `confirm="yes"` fails the identity check.

### Rate limiting

Server-side cooldowns are authoritative — clients can't bypass by issuing parallel calls. The limiter is per-tool (pending e-stop cooldown doesn't block joint-state reads) and uses `time.monotonic()` so wall-clock skew doesn't matter.

Rejected calls return `{"error": {"kind": "RateLimited", "message": "... wait Xs", ...}}`. Well-behaved agents retry after the hint.

### `robot://status`

A JSON resource snapshot combining static robot identity (URDF path, action_dim, frame info) with the latest cached joint state + task. One-shot status for an agent picking up a new task.

```json
{
  "embodiment": "franka",
  "action_dim": 7,
  "image_topic": "/camera/image_raw",
  "state_topic": "/joint_states",
  "action_topic": "/tether/actions",
  "e_stop_topic": "/tether/e_stop",
  "last_joint_positions": [0.0, -0.3, ...],
  "last_task": "pick up the red block",
  "joint_state_stale": false,
  "snapshot_timestamp": 1750000000.0
}
```

### CLI wiring

Phase 1 ships the substrate (tools + live-node context implementation). The end-to-end `tether ros2-serve --mcp` invocation that spins up both the rclpy node AND the MCP stdio loop is Phase 1.5.

## Feature spec

- `features/01_serve/subfeatures/_dx_gaps/mcp-server/mcp-server.md`
- `features/01_serve/subfeatures/_dx_gaps/mcp-server/mcp-server_plan.md`
- `features/01_serve/subfeatures/_ecosystem/ros2-mcp-bridge/ros2-mcp-bridge.md` (ROS2 tools)

Pattern source: [InferScope](https://github.com/rylinjames/easyinference) (sibling project at `EasyInference-main/products/inferscope/`). Tether's MCP server lifts InferScope's FastMCP tool/resource pattern with VLA-specific tool semantics (`act` instead of `chat/completions`).
