#!/usr/bin/env python3
"""
Simple console AI chat using GLM API from z.ai with LangChain
Day 8 implementation for Elite Engineers Advent of AI 2025
"""

import os
import sys
import json
import subprocess
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, time
import threading
import time as time_module

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_community.callbacks import get_openai_callback
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# MCP imports
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()

# System message for the AI
SYSTEM_MESSAGE = """You are a helpful AI assistant powered by z.ai's GLM-4.6 model.
You can answer questions, help with tasks, and have engaging conversations.
You have access to various tools through the Model Context Protocol (MCP) that can help you perform actions like file operations, git commands, and web searches.
Use these tools when they are relevant to the user's request.
Be concise but helpful in your responses."""

# MCP configuration file path
MCP_CONFIG_PATH = Path(__file__).parent / "mcp.json"
PERIODIC_TASK_PATH = Path(__file__).parent / "periodic_task.json"

class MCPServerManager:
    """Manages MCP server registration and lifecycle"""

    def __init__(self, config_path: Path = MCP_CONFIG_PATH):
        self.config_path = config_path
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load MCP server configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.servers = config.get('mcpServers', {})
                console.print(f"[green]✓[/green] Loaded {len(self.servers)} MCP server configurations from {self.config_path}")
            else:
                console.print(f"[yellow]⚠[/yellow] MCP config file not found at {self.config_path}, using empty configuration")
                self.servers = {}
        except json.JSONDecodeError as e:
            console.print(f"[red]✗[/red] Error parsing MCP config file: {e}")
            self.servers = {}
        except Exception as e:
            console.print(f"[red]✗[/red] Error loading MCP config: {e}")
            self.servers = {}

    def get_enabled_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get only enabled MCP servers"""
        return {name: config for name, config in self.servers.items()
                if not config.get('disabled', False)}

    def register_server(self, name: str, command: str, args: List[str] = None,
                       env: Dict[str, str] = None, disabled: bool = False) -> bool:
        """Register a new MCP server configuration"""
        try:
            server_config = {
                "command": command,
                "args": args or [],
                "env": env or {},
                "disabled": disabled
            }
            self.servers[name] = server_config
            self.save_config()
            console.print(f"[green]✓[/green] Registered MCP server '{name}'")
            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to register MCP server '{name}': {e}")
            return False

    def save_config(self) -> bool:
        """Save current server configuration to JSON file"""
        try:
            config = {"mcpServers": self.servers}
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to save MCP config: {e}")
            return False

    def list_servers(self) -> None:
        """Display all registered MCP servers"""
        if not self.servers:
            console.print("[yellow]No MCP servers configured[/yellow]")
            return

        console.print("\n[bold]Registered MCP Servers:[/bold]")
        for name, config in self.servers.items():
            status = "[red]Disabled[/red]" if config.get('disabled', False) else "[green]Enabled[/green]"
            console.print(f"  • {name}: {status}")
            console.print(f"    Command: {config['command']} {' '.join(config.get('args', []))}")
            if config.get('env'):
                env_vars = ', '.join([f"{k}={v}" for k, v in config['env'].items()])
                console.print(f"    Environment: {env_vars}")
        console.print()

    def unregister_server(self, name: str) -> bool:
        """Remove an MCP server configuration"""
        if name in self.servers:
            del self.servers[name]
            self.save_config()
            console.print(f"[green]✓[/green] Unregistered MCP server '{name}'")
            return True
        else:
            console.print(f"[red]✗[/red] MCP server '{name}' not found")
            return False

    def enable_server(self, name: str) -> bool:
        """Enable an MCP server"""
        if name in self.servers:
            self.servers[name]['disabled'] = False
            self.save_config()
            console.print(f"[green]✓[/green] Enabled MCP server '{name}'")
            return True
        else:
            console.print(f"[red]✗[/red] MCP server '{name}' not found")
            return False

    def disable_server(self, name: str) -> bool:
        """Disable an MCP server"""
        if name in self.servers:
            self.servers[name]['disabled'] = True
            self.save_config()
            console.print(f"[green]✓[/green] Disabled MCP server '{name}'")
            return True
        else:
            console.print(f"[red]✗[/red] MCP server '{name}' not found")
            return False

# Global MCP server manager
mcp_manager = MCPServerManager()

async def load_mcp_tools() -> List[Any]:
    """Load all available MCP tools from enabled servers"""
    enabled_servers = mcp_manager.get_enabled_servers()
    if not enabled_servers:
        console.print("[yellow]No MCP servers enabled, no MCP tools loaded[/yellow]")
        return []

    # Convert MCP server config to MultiServerMCPClient format
    client_config = {}
    for name, config in enabled_servers.items():
        if config["command"] == "npx":
            # Node.js-based MCP server
            client_config[name] = {
                "transport": "stdio",
                "command": config["command"],
                "args": config.get("args", []),
                "env": config.get("env", {}),
            }
        else:
            # Generic command-based server
            client_config[name] = {
                "transport": "stdio",
                "command": config["command"],
                "args": config.get("args", []),
                "env": config.get("env", {}),
            }

    try:
        # Create MCP client and load tools
        client = MultiServerMCPClient(client_config)
        tools = await client.get_tools()
        console.print(f"[green]✓[/green] Loaded {len(tools)} MCP tools from {len(enabled_servers)} server(s)")

        # Display loaded tools
        for tool in tools:
            console.print(f"  • [dim]{tool.name}[/dim]: {tool.description[:80]}{'...' if len(tool.description) > 80 else ''}")

        return tools
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load MCP tools: {e}")
        console.print("[dim]This might be due to missing MCP server dependencies.[/dim]")
        return []

def initialize_llm():
    """Initialize the LLM with z.ai API configuration"""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "glm-4.6"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4/"),
        streaming=False  # Set to True for streaming output
    )

def display_welcome():
    """Display welcome message"""
    console.print(Panel.fit("[bold blue]AI Chat with LangChain + MCP Support[/bold blue]", border_style="blue"))
    console.print("Welcome to the AI chat application with Model Context Protocol support!")
    console.print("This application integrates MCP tools to provide enhanced capabilities like file operations, git commands, and more.")

    # Display MCP server information
    enabled_servers = mcp_manager.get_enabled_servers()
    if enabled_servers:
        server_names = list(enabled_servers.keys())
        console.print(f"[green]✓[/green] {len(enabled_servers)} MCP server(s) configured: {', '.join(server_names)}")
        console.print("[dim]Tools will be loaded when you start the conversation[/dim]")
    else:
        console.print("[yellow]⚠[/yellow] No MCP servers enabled")
        console.print("[dim]Use '/mcp enable <name>' to enable configured servers[/dim]")

    console.print("\n[bold]Commands:[/bold]")
    console.print("  Type 'exit' or 'quit' to end the conversation")
    console.print("  Type '/mcp list' to see all MCP servers")
    console.print("  Type '/mcp register <name> <command> [args...]' to register a new MCP server")
    console.print("  Type '/mcp enable <name>' to enable an MCP server")
    console.print("  Type '/mcp disable <name>' to disable an MCP server")
    console.print("  Type '/mcp unregister <name>' to remove an MCP server")
    console.print("\n[dim]Example questions you can ask:[/dim]")
    console.print("  • 'What files are in the current directory?'")
    console.print("  • 'Show me the git status of this repository'")
    console.print("  • 'Create a file called hello.txt with some content'")
    console.print("")

def handle_mcp_command(command: str) -> bool:
    """Handle MCP-related commands. Returns True if command was handled."""
    parts = command.strip().split()
    if len(parts) < 2 or parts[0] != "/mcp":
        return False

    if len(parts) < 2:
        console.print("[red]Usage: /mcp <subcommand> [args...][/red]")
        return True

    subcommand = parts[1]

    if subcommand == "list":
        mcp_manager.list_servers()
        return True

    elif subcommand == "register":
        if len(parts) < 4:
            console.print("[red]Usage: /mcp register <name> <command> [args...][/red]")
            console.print("[dim]Example: /mcp register filesystem npx -y @modelcontextprotocol/server-filesystem /tmp[/dim]")
            return True
        name = parts[2]
        command = parts[3]
        args = parts[4:] if len(parts) > 4 else []
        mcp_manager.register_server(name, command, args)
        return True

    elif subcommand == "enable":
        if len(parts) != 3:
            console.print("[red]Usage: /mcp enable <name>[/red]")
            return True
        name = parts[2]
        mcp_manager.enable_server(name)
        return True

    elif subcommand == "disable":
        if len(parts) != 3:
            console.print("[red]Usage: /mcp disable <name>[/red]")
            return True
        name = parts[2]
        mcp_manager.disable_server(name)
        return True

    elif subcommand == "unregister":
        if len(parts) != 3:
            console.print("[red]Usage: /mcp unregister <name>[/red]")
            return True
        name = parts[2]
        mcp_manager.unregister_server(name)
        return True

    else:
        console.print(f"[red]Unknown MCP subcommand: {subcommand}[/red]")
        console.print("[dim]Available subcommands: list, register, enable, disable, unregister[/dim]")
        return True

def get_user_input() -> str:
    """Get input from user"""
    try:
        return Prompt.ask("[bold green]You[/bold green]")
    except EOFError:
        return "exit"

def display_ai_response(response: str, token_usage: dict = None):
    """Display AI response with rich formatting and token usage"""
    console.print("[bold blue]AI Assistant:[/bold blue]")
    # Try to parse as markdown, fallback to plain text
    try:
        md = Markdown(response)
        console.print(md)
    except:
        console.print(response)
    
    # Display token usage if available
    if token_usage and token_usage.get("total_tokens", 0) > 0:
        console.print(f"[dim]Tokens: {token_usage['prompt_tokens']} prompt, "
                     f"{token_usage['completion_tokens']} completion, "
                     f"{token_usage['total_tokens']} total[/dim]")
    
    console.print("")

def detect_and_display_summarization(original_messages: list, response_messages: list):
    """Detect when summarization occurs and display the summarized content"""
    # Check if the number of messages has decreased significantly
    if len(response_messages) < len(original_messages) and len(original_messages) > 3:
        # Look for messages that contain summary content
        for msg in response_messages:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                # Check if this looks like a summary message
                if 'summary' in msg.content.lower() or 'previous conversation' in msg.content.lower():
                    console.print("[bold yellow]📝 Conversation summarized to manage token usage:[/bold yellow]")
                    # Try to display just the summary part
                    summary_content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                    console.print(f"[dim]{summary_content}[/dim]")
                    console.print(f"[dim]Original messages: {len(original_messages)} → After summarization: {len(response_messages)}[/dim]")
                    console.print("")
                    break

async def main():
    """Main chat loop"""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not found in environment variables")
        console.print("Please set your API key in a .env file")
        sys.exit(1)

    # Initialize components
    display_welcome()
    llm = initialize_llm()

    # Load MCP tools
    console.print("[bold blue]Loading MCP tools...[/bold blue]")
    mcp_tools = await load_mcp_tools()

    # Create agent with MCP tools and SummarizationMiddleware to handle long conversations
    agent = create_agent(
        model=llm,
        tools=mcp_tools,  # Include MCP tools
        middleware=[
            SummarizationMiddleware(
                model=llm,  # Use the same model for summarization
                max_tokens_before_summary=200,  # Trigger summarization at 4000 tokens
                messages_to_keep=10,  # Keep last 10 messages after summary
            ),
        ],
    )

    if mcp_tools:
        console.print(f"[green]✓[/green] Agent initialized with {len(mcp_tools)} MCP tools")
    else:
        console.print("[yellow]⚠[/yellow] Agent initialized without MCP tools")
    
    periodic_task = load_periodic_task()
    if periodic_task and periodic_task.get('scheduled_time'):
        schedule_daily_task(agent, messages, periodic_task)
    
    # Initialize conversation history with system message
    messages = [SystemMessage(content=SYSTEM_MESSAGE)]
    
    # Main chat loop
    while True:
        user_input = get_user_input()

        # Check for exit commands
        if user_input.lower() in ['exit', 'quit']:
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break

        # Handle MCP commands
        if user_input.startswith('/mcp'):
            if handle_mcp_command(user_input):
                continue  # Command handled, continue to next iteration

        # Add user message to history
        messages.append(HumanMessage(content=user_input))
        
        # Get and display AI response using the agent
        try:
            with console.status("[bold blue]Thinking...", spinner="dots"):
                with get_openai_callback() as cb:
                    # Try using async invoke with middleware
                    response = await agent.ainvoke({"messages": messages})
                    token_usage = {
                        "total_tokens": cb.total_tokens,
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_cost": cb.total_cost
                    }
            
            # Extract AI response content
            ai_response = response["messages"][-1].content
            
            # Detect and display summarization info
            detect_and_display_summarization(messages, response["messages"])
            
            # Display AI response
            display_ai_response(ai_response, token_usage)
            
            # Add AI response to history
            messages.append(AIMessage(content=ai_response))
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            console.print("[dim]The conversation will continue, but the last message may not be saved.[/dim]")
            console.print("")

def load_periodic_task() -> Optional[Dict[str, str]]:
    try:
        if PERIODIC_TASK_PATH.exists():
            with open(PERIODIC_TASK_PATH, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load periodic task: {e}")
        return None

def parse_scheduled_time(time_str: str) -> time:
    return datetime.strptime(time_str, "%H:%M").time()

async def execute_periodic_task(agent, messages: List, task_config: Dict[str, str]):
    console.print(f"\n[bold yellow]⏰ Executing scheduled task at {task_config['scheduled_time']}[/bold yellow]")
    
    task_message = HumanMessage(content=task_config['prompt'])
    task_messages = messages + [task_message]
    
    try:
        with console.status("[bold blue]Processing scheduled task...", spinner="dots"):
            with get_openai_callback() as cb:
                response = await agent.ainvoke({"messages": task_messages})
                token_usage = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }
        
        ai_response = response["messages"][-1].content
        display_ai_response(ai_response, token_usage)
        
    except Exception as e:
        console.print(f"[bold red]Error executing periodic task:[/bold red] {str(e)}")

def schedule_daily_task(agent, messages: List, task_config: Dict[str, str]):
    target_time = parse_scheduled_time(task_config['scheduled_time'])
    
    def run_scheduler():
        while True:
            now = datetime.now()
            current_time = now.time()
            
            if current_time.hour == target_time.hour and current_time.minute == target_time.minute:
                asyncio.run(execute_periodic_task(agent, messages, task_config))
                time_module.sleep(61)
            else:
                time_module.sleep(30)
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    console.print(f"[green]✓[/green] Periodic task scheduled for {task_config['scheduled_time']} daily")

if __name__ == "__main__":
    asyncio.run(main())