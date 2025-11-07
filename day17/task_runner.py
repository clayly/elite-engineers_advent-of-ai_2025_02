#!/usr/bin/env python3
"""
CLI tool that reads task from file and sends to LLM
Based on main.py ai-agent chat foundation
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Any, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ModelCallLimitMiddleware
from rich.console import Console

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

console = Console()

SYSTEM_MESSAGE = """You are a helpful AI assistant powered by z.ai's GLM-4.6 model.
You can answer questions, help with tasks, and have engaging conversations.
You have access to various tools through the Model Context Protocol (MCP) that can help you perform actions like file operations, git commands, and web searches.
Use these tools when they are relevant to the user's request.

IMPORTANT: When you complete your task, you MUST end your response with the exact word: amen"""

MCP_CONFIG_PATH = Path(__file__).parent / "mcp.json"

def initialize_llm() -> ChatOpenAI:
    """Initialize the LLM"""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "glm-4.6"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4/"),
        streaming=False,
    )

def get_enabled_servers() -> Dict[str, Dict[str, Any]]:
    """Get enabled MCP servers from config file"""
    try:
        if not MCP_CONFIG_PATH.exists():
            return {}
        
        with open(MCP_CONFIG_PATH, 'r') as f:
            config = json.load(f)
            servers = config.get('mcpServers', {})
            return {name: cfg for name, cfg in servers.items() if not cfg.get('disabled', False)}
    except Exception:
        return {}

async def load_mcp_tools() -> List[Any]:
    """Load MCP tools from configuration"""
    enabled_servers = get_enabled_servers()
    if not enabled_servers:
        console.print("[yellow]⚠[/yellow] No MCP servers enabled")
        return []

    client_config = {}
    for name, config in enabled_servers.items():
        client_config[name] = {
            "transport": "stdio",
            "command": config["command"],
            "args": config.get("args", []),
            "env": config.get("env", {}),
        }

    try:
        client = MultiServerMCPClient(client_config)
        tools = await client.get_tools()
        console.print(f"[green]✓[/green] Loaded {len(tools)} MCP tools")
        return tools
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Could not load MCP tools: {e}")
        return []

async def run_task_from_file(file_path: str) -> int:
    """Read task from file and execute with LLM"""
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not found")
        return 1

    task_file = Path(file_path)
    if not task_file.exists():
        console.print(f"[bold red]Error:[/bold red] Task file not found: {file_path}")
        return 1

    task = task_file.read_text().strip()
    console.print(f"[bold blue]Task:[/bold blue] {task}\n")

    llm = initialize_llm()
    mcp_tools = await load_mcp_tools()

    agent = create_agent(
        model=llm,
        tools=mcp_tools,
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=100,
                thread_limit=200,
                exit_behavior="end",
            ),
            SummarizationMiddleware(
                model=llm,
                max_tokens_before_summary=8000,
                messages_to_keep=20,
            ),
        ],
    )

    messages = [
        SystemMessage(content=SYSTEM_MESSAGE),
        HumanMessage(content=task)
    ]

    try:
        with console.status("[bold blue]Processing task...", spinner="dots"):
            with get_openai_callback() as cb:
                response = await agent.ainvoke({
                    "messages": messages
                }, config={"recursion_limit": 200})
                
        ai_response = response["messages"][-1].content
        
        console.print("[bold blue]Response:[/bold blue]")
        console.print(ai_response)
        console.print(f"\n[dim]Tokens: {cb.total_tokens} total[/dim]")
        
        if "amen" in ai_response.lower():
            console.print("\n[green]✓[/green] Task completed successfully")
            return 0
        else:
            console.print("\n[yellow]⚠[/yellow] Task may not be complete (no completion signal)")
            return 0
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="CLI tool to run LLM tasks from file")
    parser.add_argument("-f", "--file", required=True, help="Path to task file")
    args = parser.parse_args()
    
    exit_code = asyncio.run(run_task_from_file(args.file))
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
