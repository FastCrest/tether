"""Tether MCP server — agent-callable surface for /act, health, models_list, validate_dataset.

Exposes a running TetherServer as a Model Context Protocol (MCP) server so
MCP-compatible agents (Claude Desktop, Cursor, custom) can discover Tether in
the mcp.so catalog and call robot-policy inference as a tool.

Pattern source: EasyInference InferScope (sibling project; not a dependency).
    ~/Desktop/building projects/EasyInference-main/products/inferscope/src/inferscope/server.py

Feature spec: features/01_serve/subfeatures/_dx_gaps/mcp-server/mcp-server.md
Execution plan: features/01_serve/subfeatures/_dx_gaps/mcp-server/mcp-server_plan.md
"""
from tether.mcp.server import create_mcp_server
from tether.mcp.ros2_tools import ROS2Context, register_ros2_tools

__all__ = ["create_mcp_server", "ROS2Context", "register_ros2_tools"]
