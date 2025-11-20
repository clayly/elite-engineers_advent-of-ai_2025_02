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

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import RunnableLambda
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ModelCallLimitMiddleware

# RAG imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Local RAG imports
from rag_manager import RAGManager, should_use_rag, format_docs
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# MCP imports
from langchain_mcp_adapters.client import MultiServerMCPClient

# Memory and persistence imports
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig

# Voice input imports
try:
    from microphone_input import MicrophoneInput
    MICROPHONE_AVAILABLE = True
except ImportError:
    MICROPHONE_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()

# System message for the AI
SYSTEM_MESSAGE = """You are a helpful AI assistant.
You can answer questions, help with tasks, and have engaging conversations.
You have access to various tools through the Model Context Protocol (MCP) that can help you perform actions like file operations, git commands, and web searches.
Use these tools when they are relevant to the user's request.
Be concise but helpful in your responses."""

# MCP configuration file path
# MCP configuration file path
MCP_CONFIG_PATH = Path(__file__).parent / "mcp.json"

# Memory database path
MEMORY_DB_PATH = Path(__file__).parent / "chat_memory.db"

# User profile path
USER_PROFILE_PATH = Path(__file__).parent / "user_profile.md"

# Voice mode settings
microphone: Optional[MicrophoneInput] = None
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

class UserProfileManager:
    
    def __init__(self, profile_path: Path = USER_PROFILE_PATH):
        self.profile_path = profile_path
        self.profile_content = ""
        self.load_profile()
    
    def load_profile(self) -> None:
        try:
            if self.profile_path.exists():
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    self.profile_content = f.read()
                console.print(f"[green]✓[/green] Loaded user profile from {self.profile_path}")
            else:
                console.print(f"[yellow]⚠[/yellow] User profile not found at {self.profile_path}")
                self.profile_content = ""
        except Exception as e:
            console.print(f"[red]✗[/red] Error loading user profile: {e}")
            self.profile_content = ""
    
    def get_profile_context(self) -> str:
        if not self.profile_content:
            return ""
        
        return f"\n\n## User Profile and Personalization\n{self.profile_content}"
    
    def reload_profile(self) -> None:
        self.load_profile()

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

class SimpleMemoryManager:
    """Simple file-based persistent memory for AI conversations"""

    def __init__(self, db_path: Path = MEMORY_DB_PATH):
        self.db_path = db_path
        self.sessions_dir = db_path.parent / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        self.store = InMemoryStore()
        console.print(f"[green]✓[/green] Memory directory initialized: {self.sessions_dir}")
        console.print("[green]✓[/green] Semantic memory store initialized")

    def _get_session_file(self, thread_id: str) -> Path:
        """Get the file path for a session's conversation history"""
        return self.sessions_dir / f"{thread_id}.json"

    def get_agent_config(self, thread_id: str = "default", user_id: str = "default_user") -> RunnableConfig:
        """Get configuration for agent with thread and user IDs"""
        return {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }

    def get_conversation_history(self, thread_id: str = "default") -> list:
        """Retrieve conversation history for a specific thread"""
        try:
            session_file = self._get_session_file(thread_id)
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert stored messages back to LangChain message objects
                    messages = []
                    for msg_data in data.get("messages", []):
                        if msg_data["type"] == "human":
                            messages.append(HumanMessage(content=msg_data["content"]))
                        elif msg_data["type"] == "ai":
                            messages.append(AIMessage(content=msg_data["content"]))
                        elif msg_data["type"] == "system":
                            messages.append(SystemMessage(content=msg_data["content"]))
                    return messages
            return []
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to retrieve conversation history: {e}")
            return []

    def save_conversation(self, thread_id: str, messages: list):
        """Save conversation history to file"""
        try:
            session_file = self._get_session_file(thread_id)
            # Convert LangChain messages to serializable format
            serializable_messages = []
            for msg in messages:
                msg_type = "unknown"
                if hasattr(msg, 'type'):
                    msg_type = msg.type
                elif isinstance(msg, HumanMessage):
                    msg_type = "human"
                elif isinstance(msg, AIMessage):
                    msg_type = "ai"
                elif isinstance(msg, SystemMessage):
                    msg_type = "system"

                serializable_messages.append({
                    "type": msg_type,
                    "content": msg.content
                })

            data = {
                "thread_id": thread_id,
                "messages": serializable_messages,
                "updated_at": str(Path().absolute())
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to save conversation: {e}")

    def list_threads(self) -> list:
        """List all available conversation threads"""
        try:
            threads = []
            for session_file in self.sessions_dir.glob("*.json"):
                thread_id = session_file.stem
                threads.append(thread_id)
            return threads if threads else ["default"]
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to list threads: {e}")
            return ["default"]

# Global memory manager
memory_manager = SimpleMemoryManager()

# Global RAG manager with configurable similarity threshold
rag_similarity_threshold = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))
rag_manager = RAGManager(similarity_threshold=rag_similarity_threshold)

# Global RAG prompt template
rag_prompt = None

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

def initialize_rag_chains(llm):
    """Initialize RAG and non-RAG chains"""
    
    # Initialize global RAG prompt template
    global rag_prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to relevant documents.
Answer the question using ONLY the following context. If the context doesn't contain enough information, say so clearly.

CRITICAL INSTRUCTION: You MUST cite your sources using [1], [2], [3], etc. reference markers whenever you use information from a specific document. Place the citation immediately after the relevant information.

EXAMPLE of CORRECT citations:
- "AI is a branch of computer science [1] that aims to create intelligent machines [2]."
- "Key concepts include NLP [1] and computer vision [2]."

INCORRECT (missing citations):
- "AI is a branch of computer science that aims to create intelligent machines."

Context:
{context}

Previous conversation context and tools are also available to you. You can use MCP tools and your general knowledge when needed, but you must cite document sources when using the provided context."""),
        ("human", "{question}")
    ])
    
    # Non-RAG Prompt Template (keep existing system message)
    no_rag_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("human", "{question}")
    ])
    
    # Create chains based on RAG availability
    if rag_manager.get_retriever():
        rag_chain = (
            {"context": rag_manager.get_retriever() | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        console.print("[green]✓[/green] RAG chain initialized")
    else:
        rag_chain = None
        console.print("[yellow]⚠[/yellow] RAG chain not available (no index)")
    
    no_rag_chain = (
        {"question": RunnablePassthrough()}
        | no_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, no_rag_chain

def initialize_llm():
    """Initialize the LLM based on configuration (OpenAI API or Ollama)"""
    # Check if we should use Ollama
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    
    if use_ollama:
        # Configure for Ollama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL_NAME", "llama3.1"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        # Default to OpenAI API (z.ai)
        return ChatOpenAI(
            model=os.getenv("MODEL_NAME", "glm-4.6"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4/"),
            streaming=False  # Set to True for streaming output
        )

def display_session_info():
    """Display session and memory information"""
    console.print("\n[bold]Memory & Session Info:[/bold]")
    console.print(f"  • Memory directory: {memory_manager.sessions_dir}")
    console.print(f"  • Persistence: [green]Enabled[/green]")

    # Show conversation history count
    history = memory_manager.get_conversation_history(current_thread_id)
    if history:
        console.print(f"  • Previous messages: {len(history)}")
    else:
        console.print("  • Previous messages: None (new conversation)")

def display_welcome():
    """Display welcome message"""
    console.print(Panel.fit("[bold blue]AI Chat with LangChain + MCP Support[/bold blue]", border_style="blue"))
    console.print("Welcome to the AI chat application with Model Context Protocol support!")
    console.print("This application integrates MCP tools and now includes **persistent memory** - your conversations are saved and restored after restart.")
    
    # Display session and memory info
    display_session_info()
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
    console.print("  • '/session list' - List all conversation sessions")
    console.print("  • '/session new <name>' - Start a new conversation session")
    console.print("  • '/session switch <name>' - Switch to an existing session")
    console.print("  • '/session info' - Show current session information")
    console.print("  • '/rag status' - Show RAG index status")
    console.print("  • '/rag index <path>' - Index documents from file or directory")
    console.print("  • '/rag search <query>' - Search documents in RAG index")
    console.print("  • '/rag reset' - Reset RAG index")
    console.print("  • '/rag threshold <value>' - Set similarity threshold (0.0-1.0)")
    console.print("  • '/rag threshold show' - Show current similarity threshold")
    
    # Display voice commands if available
    if MICROPHONE_AVAILABLE:
        console.print("\n[bold]Voice Commands:[/bold]")
        console.print("  • '/voice' or '/voice record' - Record and transcribe speech")
        console.print("  • '/voice devices' - List available microphone devices")
    else:
        console.print("\n[yellow]⚠[/yellow] Voice input not available (missing dependencies)")
        console.print("[dim]Install with: uv pip install sounddevice soundfile[/dim]")
    
    # Display RAG status
    if rag_manager.ready:
        stats = rag_manager.get_stats()
        console.print(f"[green]✓[/green] RAG system ready: {stats.get('total_chunks', 0)} chunks indexed")
    else:
        console.print("[yellow]⚠[/yellow] RAG system not ready (use '/rag index <path>' to create index)")
    
    console.print("\n[dim]Example questions you can ask:[/dim]")
    console.print("  • 'What files are in the current directory?'")
    console.print("  • 'Show me the git status of this repository'")
    console.print("  • 'Create a file called hello.txt with some content'")
    console.print("")

# Global session state
current_thread_id = "default"
current_user_id = "default_user"

def handle_session_command(command: str) -> bool:
    """Handle session-related commands. Returns True if command was handled."""
    global current_thread_id
    parts = command.strip().split()
    if len(parts) < 2 or parts[0] != "/session":
        return False

    subcommand = parts[1]

    if subcommand == "list":
        console.print("\n[bold]Available Sessions:[/bold]")
        threads = memory_manager.list_threads()
        for thread in threads:
            status = "✓ [green]Current[/green]" if thread == current_thread_id else f"• {thread}"
            console.print(f"  {status}")
        console.print()
        return True

    elif subcommand == "new":
        if len(parts) < 3:
            console.print("[red]Usage: /session new <session_name>[/red]")
            return True
        session_name = parts[2]
        current_thread_id = session_name
        console.print(f"[green]✓[/green] Started new session: {session_name}")
        console.print("[dim]Tip: Use '/session list' to see all sessions[/dim]")
        return True

    elif subcommand == "switch":
        if len(parts) < 3:
            console.print("[red]Usage: /session switch <session_name>[/red]")
            return True
        session_name = parts[2]
        current_thread_id = session_name
        console.print(f"[green]✓[/green] Switched to session: {session_name}")
        
        # Show conversation history for this session
        history = memory_manager.get_conversation_history(session_name)
        if history:
            console.print(f"[dim]Found {len(history)} previous messages in this session[/dim]")
        else:
            console.print("[dim]No previous messages in this session[/dim]")
        console.print()
        return True

    elif subcommand == "info":
        console.print(f"\n[bold]Current Session Info:[/bold]")
        console.print(f"  • Session ID: {current_thread_id}")
        console.print(f"  • User ID: {current_user_id}")
        
        history = memory_manager.get_conversation_history(current_thread_id)
        console.print(f"  • Messages in session: {len(history)}")
        
        if history:
            console.print(f"  • Last message: {history[-1].content[:50]}{'...' if len(history[-1].content) > 50 else ''}")
        console.print()
        return True

    else:
        console.print(f"[red]Unknown session subcommand: {subcommand}[/red]")
        console.print("[dim]Available subcommands: list, new, switch, info[/dim]")
        return True

def handle_rag_command(command: str) -> bool:
    """Handle RAG-related commands. Returns True if command was handled."""
    parts = command.strip().split()
    if len(parts) < 2 or parts[0] != "/rag":
        return False
    
    subcommand = parts[1]
    
    if subcommand == "status":
        stats = rag_manager.get_stats()
        console.print("\n[bold]RAG System Status:[/bold]")
        console.print(f"  • Status: {stats.get('status', 'unknown')}")
        
        if stats.get('status') == 'ready':
            console.print(f"  • Total chunks: {stats.get('total_chunks', 0)}")
            console.print(f"  • Index path: {stats.get('index_path', 'N/A')}")
            console.print(f"  • Similarity threshold: {rag_manager.similarity_threshold}")
            
            model_info = stats.get('model_info', {})
            if model_info:
                console.print(f"  • Model: {model_info.get('model_name', 'N/A')}")
                console.print(f"  • Model type: {model_info.get('model_type', 'N/A')}")
            
            input_path = stats.get('input_path')
            if input_path:
                console.print(f"  • Source: {input_path}")
        else:
            console.print("  • RAG system not ready")
            console.print("  • Use '/rag index <path>' to create index")
        console.print()
        return True
    
    elif subcommand == "index":
        if len(parts) < 3:
            console.print("[red]Usage: /rag index <file_or_directory_path>[/red]")
            console.print("[dim]Example: /rag index ./documents[/dim]")
            return True
        
        input_path = parts[2]
        if not Path(input_path).exists():
            console.print(f"[red]✗[/red] Path not found: {input_path}")
            return True
        
        success = rag_manager.index_documents(input_path)
        if success:
            console.print("[green]✓[/green] Document indexing completed successfully!")
        return True
    
    elif subcommand == "search":
        if len(parts) < 3:
            console.print("[red]Usage: /rag search <query>[/red]")
            console.print("[dim]Example: /rag search artificial intelligence[/dim]")
            return True
        
        query = " ".join(parts[2:])
        results = rag_manager.search_documents(query, k=5)
        
        if not results:
            console.print(f"[yellow]No results found for: {query}[/yellow]")
            return True
        
        console.print(f"\n[bold]Search results for: {query}[/bold]")
        console.print("=" * 80)
        
        for i, (doc, similarity) in enumerate(results, 1):
            console.print(f"\nResult {i} (Similarity: {similarity:.4f})")
            console.print("-" * 40)
            
            # Show metadata
            metadata = doc.metadata
            if metadata:
                console.print("Metadata:")
                for key, value in metadata.items():
                    console.print(f"  {key}: {value}")
            
            # Show content preview
            content_preview = doc.page_content[:300]
            if len(doc.page_content) > 300:
                content_preview += "..."
            
            console.print(f"\nContent preview:\n{content_preview}")
            console.print("\n" + "=" * 80)
        console.print()
        return True
    
    elif subcommand == "reset":
        console.print("[yellow]⚠[/yellow] This will remove the entire RAG index.")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            success = rag_manager.reset_index()
            if success:
                console.print("[green]✓[/green] RAG index reset successfully")
        else:
            console.print("Operation cancelled")
        return True
    
    elif subcommand == "threshold":
        if len(parts) < 3:
            console.print(f"[red]Usage: /rag threshold <value|show>[/red]")
            console.print(f"[dim]Current threshold: {rag_manager.similarity_threshold}[/dim]")
            console.print(f"[dim]Example: /rag threshold 0.5[/dim]")
            return True
        
        if parts[2] == "show":
            console.print(f"[bold]Current RAG Similarity Threshold:[/bold] {rag_manager.similarity_threshold}")
            console.print(f"[dim]Range: 0.0 (no filtering) to 1.0 (very strict)[/dim]")
            console.print(f"[dim]Higher values = more relevant but fewer results[/dim]")
        else:
            try:
                new_threshold = float(parts[2])
                if 0.0 <= new_threshold <= 1.0:
                    rag_manager.similarity_threshold = new_threshold
                    # Recreate retriever with new threshold
                    if rag_manager.vector_store:
                        rag_manager.retriever = create_retriever_function(rag_manager.vector_store, new_threshold)
                    console.print(f"[green]✓[/green] Similarity threshold updated to {new_threshold}")
                    console.print("[dim]Note: This affects future RAG queries[/dim]")
                else:
                    console.print(f"[red]✗[/red] Threshold must be between 0.0 and 1.0")
            except ValueError:
                console.print(f"[red]✗[/red] Invalid threshold value: {parts[2]}")
        return True
    
    else:
        console.print(f"[red]Unknown RAG subcommand: {subcommand}[/red]")
        console.print("[dim]Available subcommands: status, index, search, reset, threshold[/dim]")
        return True

def handle_voice_command(command: str) -> Optional[str]:
    """Handle voice input commands. Returns transcribed text if recording, True if command handled, False otherwise."""
    global microphone
    
    if not command.startswith("/voice"):
        return False
    
    parts = command.strip().split(maxsplit=1)
    if len(parts) < 2:
        subcommand = ""
    else:
        subcommand = parts[1].lower()
    
    # Record and transcribe (/voice or /voice record)
    if not subcommand or subcommand == "record":
        if not MICROPHONE_AVAILABLE:
            console.print("[red]✗[/red] Voice input not available. Install dependencies:")
            console.print("[dim]uv pip install sounddevice soundfile[/dim]")
            return True
        
        try:
            if microphone is None:
                microphone = MicrophoneInput(model_size=WHISPER_MODEL_SIZE)
            
            console.print("[yellow]Recording... Press Enter when done speaking![/yellow]")
            
            text = microphone.record_and_transcribe_until_enter()
            if text.strip():
                console.print(f"[green]✓[/green] Transcribed: \"{text}\"")
                return text  # Return the transcribed text for processing
            else:
                console.print("[yellow]⚠[/yellow] No speech detected, try again")
        except Exception as e:
            console.print(f"[red]✗[/red] Recording failed: {e}")
            console.print("[dim]Make sure your microphone is connected and accessible[/dim]")
        return True
    
    elif subcommand == "devices":
        if not MICROPHONE_AVAILABLE:
            console.print("[red]✗[/red] Voice input not available")
            return True
        
        try:
            if microphone is None:
                microphone = MicrophoneInput(model_size=WHISPER_MODEL_SIZE)
            microphone.list_devices()
        except Exception as e:
            console.print(f"[red]✗[/red] Error listing devices: {e}")
        return True
    
    else:
        console.print(f"[red]Unknown voice subcommand: {subcommand}[/red]")
        console.print("[dim]Available: /voice (record), /voice devices[/dim]")
        return True

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
    global microphone
    
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not found in environment variables")
        console.print("Please set your API key in a .env file")
        sys.exit(1)

    display_welcome()
    llm = initialize_llm()
    
    profile_manager = UserProfileManager()

    console.print("[bold blue]Loading MCP tools...[/bold blue]")
    mcp_tools = await load_mcp_tools()

    # Create agent with MCP tools, middleware, and persistent memory
    agent = create_agent(
        model=llm,
        tools=mcp_tools,  # Include MCP tools
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=100,  # Allow up to 100 model calls per run for large analysis
                thread_limit=200,  # Allow up to 200 calls per thread
                exit_behavior="end",  # Gracefully end when limit reached
            ),
            SummarizationMiddleware(
                model=llm,  # Use the same model for summarization
                max_tokens_before_summary=8000,  # Higher token limit for large project context
                messages_to_keep=20,  # Keep more messages for complex analysis
            ),
        ],
        store=memory_manager.store,  # Add semantic memory store
    )

    if mcp_tools:
        console.print(f"[green]✓[/green] Agent initialized with {len(mcp_tools)} MCP tools")
    else:
        console.print("[yellow]⚠[/yellow] Agent initialized without MCP tools")
    
    # Initialize RAG chains
    console.print("[bold blue]Initializing RAG chains...[/bold blue]")
    rag_chain, no_rag_chain = initialize_rag_chains(llm)
    
    # Initialize conversation with persistent memory
    config = memory_manager.get_agent_config(current_thread_id, current_user_id)
    
    # Try to load existing conversation history
    messages = memory_manager.get_conversation_history(current_thread_id)
    if messages:
        console.print(f"[green]✓[/green] Restored {len(messages)} previous messages")
        console.print("[dim]Type '/session info' for more details about this session[/dim]")
    else:
        # Start new conversation with system message
        personalized_system_message = SYSTEM_MESSAGE + profile_manager.get_profile_context()
        messages = [SystemMessage(content=personalized_system_message)]
        console.print("[dim]Starting new conversation[/dim]")
        console.print("[dim]Tip: Your conversations are now automatically saved![/dim]")
    
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

        # Handle session commands
        if user_input.startswith('/session'):
            if handle_session_command(user_input):
                # Update config for new session
                config = memory_manager.get_agent_config(current_thread_id, current_user_id)
                # Load conversation history for the new session
                messages = memory_manager.get_conversation_history(current_thread_id)
                if not messages:
                    personalized_system_message = SYSTEM_MESSAGE + profile_manager.get_profile_context()
                    messages = [SystemMessage(content=personalized_system_message)]
                continue

        # Handle RAG commands
        if user_input.startswith('/rag'):
            if handle_rag_command(user_input):
                continue  # Command handled, continue to next iteration

        # Handle voice commands
        if user_input.startswith('/voice'):
            voice_result = handle_voice_command(user_input)
            # If voice command returns text (from recording), use it as input
            if isinstance(voice_result, str) and voice_result.strip():
                user_input = voice_result
                console.print(f"[dim]Using voice input: {user_input}[/dim]")
            else:
                continue  # Command handled, continue to next iteration

        # Check for RAG trigger and process accordingly
        use_rag, cleaned_question = should_use_rag(user_input)
        
        # Get and display AI response using RAG or agent
        try:
            if use_rag and rag_chain:
                # Use RAG chain
                with console.status("[bold blue]🔍 Searching documents...[/bold blue]", spinner="dots"):
                    # Get retrieved documents for citation tracking
                    retrieved_docs = []
                    if rag_manager.get_retriever():
                        try:
                            retrieved_docs = rag_manager.get_retriever()(cleaned_question)
                            console.print(f"[dim]✓ Retrieved {len(retrieved_docs)} relevant documents[/dim]")
                        except Exception as e:
                            console.print(f"[yellow]⚠[/yellow] Error retrieving documents: {e}")
                    
                    # Format context separately to pass to chain
                    from rag_manager import format_docs
                    context = format_docs(retrieved_docs)
                    
                    with get_openai_callback() as cb:
                        # Create a modified chain that uses pre-retrieved documents
                        from langchain_core.runnables import RunnablePassthrough
                        from langchain_core.prompts import ChatPromptTemplate
                        from langchain_core.output_parsers import StrOutputParser
                        
                        # Use the pre-formatted context instead of retriever
                        temp_chain = (
                            {"context": lambda x: context, "question": RunnablePassthrough()}
                            | rag_prompt
                            | llm
                            | StrOutputParser()
                        )
                        
                        ai_response = temp_chain.invoke(cleaned_question)
                        token_usage = {
                            "total_tokens": cb.total_tokens,
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                            "total_cost": cb.total_cost
                        }
                
                # Enhance response with citations
                from rag_manager import enhance_response_with_citations
                enhanced_response = enhance_response_with_citations(ai_response, retrieved_docs)
                
                # Add both original user input and RAG response to conversation
                messages.append(HumanMessage(content=user_input))
                messages.append(AIMessage(content=enhanced_response))
                
                # Display AI response
                display_ai_response(enhanced_response, token_usage)
                
                # Save conversation with RAG context
                memory_manager.save_conversation(current_thread_id, messages)
                console.print("[dim]✓ RAG-enhanced conversation saved to persistent memory[/dim]")
                
            else:
                # Use regular agent with tools
                if use_rag and not rag_chain:
                    console.print("[yellow]⚠[/yellow] RAG requested but no index available. Using agent mode.")
                
                with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                    with get_openai_callback() as cb:
                        # Try using async invoke with middleware and increased recursion limit for large project analysis
                        # Update config with current session info and invoke with persistent memory
                        config = memory_manager.get_agent_config(current_thread_id, current_user_id)
                        config["recursion_limit"] = 200  # Much higher limit for large project analysis
                        
                        # Add user message to conversation
                        messages.append(HumanMessage(content=user_input))

                        # Pass the entire conversation history to the agent
                        response = await agent.ainvoke({
                            "messages": messages
                        }, config=config)
                        token_usage = {
                            "total_tokens": cb.total_tokens,
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                            "total_cost": cb.total_cost
                        }
            
            # Extract AI response content (only for agent mode)
                        ai_response = response["messages"][-1].content
                        
                        # Detect and display summarization info
                        detect_and_display_summarization(messages, response["messages"])
            
            # Display AI response
                        display_ai_response(ai_response, token_usage)
            
            # Add AI response to messages and save to file
                        messages.append(AIMessage(content=ai_response))
                        memory_manager.save_conversation(current_thread_id, messages)
            console.print("[dim]✓ Conversation saved to persistent memory[/dim]")
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            console.print("[dim]The conversation will continue, but the last message may not be saved.[/dim]")
            console.print("")

if __name__ == "__main__":
    asyncio.run(main())