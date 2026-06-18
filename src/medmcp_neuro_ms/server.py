"""MCP server entrypoint for medmcp-neuro-ms (MS lesion segmentation via LST-AI)."""

from importlib.resources import files as _pkg_files

from mcp.server.fastmcp import FastMCP

from medmcp_neuro_ms.tools.segmentation import list_ms_lesion_regions, segment_ms_lesions

mcp = FastMCP("medmcp-neuro-ms")

mcp.add_tool(segment_ms_lesions)
mcp.add_tool(list_ms_lesion_regions)


def server_config() -> dict[str, object]:
    """Return MCP server metadata for autodiscovery by the local agent."""
    return {
        "name": "medmcp-neuro-ms",
        "command": "medmcp-neuro-ms",
        "skills_path": str(_pkg_files("medmcp_neuro_ms") / "skills"),
        "tool_timeout_sec": 1800.0,
    }


def main() -> None:
    """Launch the MCP server over stdio (JSON-RPC)."""
    mcp.run(transport="stdio")
