#!/usr/bin/env python3
"""
Simple MCP Server using FastMCP 2.0
Day 8 implementation for Elite Engineers Advent of AI 2025
"""

import random
import string as string_module
import json
import asyncio
import urllib.request
import urllib.parse

from fastmcp import FastMCP
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


@mcp.tool()
async def z_string_length(string: str) -> str:
    """
    Calculate string length

    Args:
        string: string to calculate length of
    """
    length = len(string)
    return json.dumps({"length": length})

@mcp.tool()
async def z_string_random(length: int) -> str:
    """
    Generate random string of specified length

    Args:
        length: length of generated string
    """
    string = ''.join(random.choices(string_module.ascii_uppercase + string_module.digits, k=length))
    return json.dumps({"string": string})

@mcp.tool()
async def z_number_is_even(number: int) -> str:
    """
    Check if number is even

    Args:
        number: number to check oddity of
    """
    isEven = number % 2 == 0
    return json.dumps({"isEven": isEven})

if __name__ == "__main__":
    mcp.run()