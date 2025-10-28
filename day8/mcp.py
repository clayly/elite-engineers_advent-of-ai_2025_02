#!/usr/bin/env python3
"""
Simple MCP Server using FastMCP 2.0
Day 8 implementation for Elite Engineers Advent of AI 2025
"""

from mcp.server.fastmcp import FastMCP
import json
import asyncio
import urllib.parse
from datetime import datetime

# Create an MCP server instance
mcp = FastMCP("Weather Server")


async def fetch_weather(latitude: float = 52.52, longitude: float = 13.41) -> dict:
    """Fetch weather data from Open-Meteo API"""
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true",
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m"
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        import urllib.request
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def z_weather(latitude: float = 52.52, longitude: float = 13.41) -> str:
    """
    Get weather information from Open-Meteo API

    Args:
        latitude: Latitude coordinate (default: Berlin 52.52)
        longitude: Longitude coordinate (default: Berlin 13.41)
    """
    weather_data = await fetch_weather(latitude, longitude)
    return json.dumps(weather_data, indent=2)


if __name__ == "__main__":
    print("Starting Weather MCP Server...")
    mcp.run()
    print("MCP Server finished.")