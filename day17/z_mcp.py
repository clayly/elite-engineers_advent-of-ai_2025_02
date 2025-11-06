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
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


@mcp.tool()
async def z_docker_run(dockerfile_path: str) -> str:
    """
    Build and run Docker container from local Dockerfile

    Creates a dated directory, copies Dockerfile, builds image, and runs container.
    All errors and logs are printed to stderr for proper MCP stdio logging.

    Args:
        dockerfile_path: Path to local Dockerfile
    """
    result = {
        "success": False,
        "steps": {},
        "error": None,
        "directory_name": None
    }

    def log_to_stderr(message: str):
        """Helper function to log messages to stderr"""
        print(f"[z_docker_run] {message}", file=sys.stderr)

    try:
        log_to_stderr(f"Starting Docker run process for: {dockerfile_path}")

        # Validate Dockerfile exists
        dockerfile_path = Path(dockerfile_path).resolve()
        if not dockerfile_path.exists():
            error_msg = f"Dockerfile not found: {dockerfile_path}"
            result["error"] = error_msg
            log_to_stderr(f"ERROR: {error_msg}")
            return json.dumps(result, indent=2)

        if not dockerfile_path.is_file():
            error_msg = f"Path is not a file: {dockerfile_path}"
            result["error"] = error_msg
            log_to_stderr(f"ERROR: {error_msg}")
            return json.dumps(result, indent=2)

        # Create dated directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"docker_run_{timestamp}"
        work_dir = Path.cwd() / dir_name

        result["directory_name"] = dir_name
        log_to_stderr(f"Created directory name: {dir_name}")

        # Step 1: Create directory and copy Dockerfile
        result["steps"]["directory_creation"] = f"Creating directory: {work_dir}"
        log_to_stderr(f"Creating work directory: {work_dir}")
        work_dir.mkdir(exist_ok=True)

        # Copy Dockerfile to work directory
        target_dockerfile = work_dir / "Dockerfile"
        log_to_stderr(f"Copying Dockerfile from {dockerfile_path} to {target_dockerfile}")
        shutil.copy2(dockerfile_path, target_dockerfile)
        result["steps"]["copy_dockerfile"] = f"Copied Dockerfile to: {target_dockerfile}"
        log_to_stderr("Dockerfile copied successfully")

        # Step 2: Build Docker image
        build_cmd = [
            "docker", "build",
            "--progress=plain",
            "-t", dir_name,
            ".",
            "--file", "Dockerfile"
        ]

        build_cmd_str = " ".join(build_cmd)
        result["steps"]["build_command"] = build_cmd_str
        log_to_stderr(f"Executing build command: {build_cmd_str}")

        try:
            build_process = await asyncio.create_subprocess_exec(
                *build_cmd,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            build_stdout_bytes, build_stderr_bytes = await build_process.communicate()
            build_stdout = build_stdout_bytes.decode('utf-8') if build_stdout_bytes else ""
            build_stderr = build_stderr_bytes.decode('utf-8') if build_stderr_bytes else ""

            log_to_stderr(f"Build process completed with return code: {build_process.returncode}")

            if build_process.returncode != 0:
                error_msg = f"Docker build failed with return code {build_process.returncode}"
                result["steps"]["build_error"] = {
                    "return_code": build_process.returncode,
                    "stdout": build_stdout,
                    "stderr": build_stderr
                }
                result["error"] = error_msg
                log_to_stderr(f"ERROR: {error_msg}")
                log_to_stderr(f"Build stdout:\n{build_stdout}")
                log_to_stderr(f"Build stderr:\n{build_stderr}")
                return json.dumps(result, indent=2)
            else:
                result["steps"]["build_success"] = {
                    "stdout": build_stdout,
                    "stderr": build_stderr
                }
                log_to_stderr("Docker build completed successfully")
                if build_stdout:
                    log_to_stderr(f"Build output:\n{build_stdout}")
                if build_stderr:
                    log_to_stderr(f"Build stderr:\n{build_stderr}")

        except Exception as e:
            error_msg = f"Failed to execute docker build: {str(e)}"
            result["error"] = error_msg
            log_to_stderr(f"ERROR: {error_msg}")
            return json.dumps(result, indent=2)

        # Step 3: Run Docker container
        run_cmd = ["docker", "run", "--rm", "-i", dir_name]

        run_cmd_str = " ".join(run_cmd)
        result["steps"]["run_command"] = run_cmd_str
        log_to_stderr(f"Executing run command: {run_cmd_str}")

        try:
            run_process = await asyncio.create_subprocess_exec(
                *run_cmd,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            run_stdout_bytes, run_stderr_bytes = await run_process.communicate()
            run_stdout = run_stdout_bytes.decode('utf-8') if run_stdout_bytes else ""
            run_stderr = run_stderr_bytes.decode('utf-8') if run_stderr_bytes else ""

            log_to_stderr(f"Run process completed with return code: {run_process.returncode}")

            result["steps"]["run_result"] = {
                "return_code": run_process.returncode,
                "stdout": run_stdout,
                "stderr": run_stderr
            }

            if run_process.returncode != 0:
                error_msg = f"Docker run failed with return code {run_process.returncode}"
                result["error"] = error_msg
                log_to_stderr(f"ERROR: {error_msg}")
                log_to_stderr(f"Run stdout:\n{run_stdout}")
                log_to_stderr(f"Run stderr:\n{run_stderr}")
            else:
                result["success"] = True
                log_to_stderr("Docker run completed successfully")
                if run_stdout:
                    log_to_stderr(f"Run output:\n{run_stdout}")
                if run_stderr:
                    log_to_stderr(f"Run stderr:\n{run_stderr}")

        except Exception as e:
            error_msg = f"Failed to execute docker run: {str(e)}"
            result["error"] = error_msg
            log_to_stderr(f"ERROR: {error_msg}")

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        result["error"] = error_msg
        log_to_stderr(f"ERROR: {error_msg}")

    log_to_stderr(f"Process completed. Success: {result['success']}")
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    mcp.run()