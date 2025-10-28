#!/usr/bin/env python3
"""
Simple MCP Server using FastMCP 2.0
Day 8 implementation for Elite Engineers Advent of AI 2025
"""

from mcp.server.fastmcp import FastMCP
import json

# Create an MCP server instance
mcp = FastMCP("Weather Server")


@mcp.tool()
def z_weather() -> str:
    """Get weather information (mock implementation)"""
    return json.dumps({})


if __name__ == "__main__":
    print("Starting Weather MCP Server...")
    mcp.run()
    print("MCP Server finished.")