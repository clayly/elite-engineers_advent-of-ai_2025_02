#!/usr/bin/env python3
"""
Simple console AI chat using GLM API from z.ai
Day 1 implementation for Elite Engineers Advent of AI 2025
"""

import asyncio
import os
import sys
from typing import List, Dict

import aiohttp
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

class GLMChatClient:
    """Client for interacting with GLM API from z.ai"""

    def __init__(self, api_key: str, base_url: str = "https://api.z.ai/api/coding/paas/v4"):
        self.api_key = api_key
        self.base_url = base_url
        self.console = Console()
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "glm-4.6",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Send chat completion request to GLM API"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")

                data = await response.json()
                return data["choices"][0]["message"]["content"]

        except asyncio.TimeoutError:
            raise Exception("Request timed out")
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")


class ChatInterface:
    """Rich console interface for the chat"""

    def __init__(self, client: GLMChatClient):
        self.client = client
        self.console = Console()
        self.conversation_history: List[Dict[str, str]] = []

    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
# 🤖 GLM AI Chat - Day 1
Welcome to your AI assistant powered by GLM API from z.ai!

**Commands:**
- Type your message and press Enter to chat
- Type `/help` to see this help
- Type `/clear` to clear conversation history
- Type `/exit` or `Ctrl+C` to quit

**Features:**
- Rich text formatting with syntax highlighting
- Markdown support
- Conversation history
- Streaming responses
        """

        panel = Panel(
            Markdown(welcome_text),
            title="[bold blue]GLM AI Chat[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )

        self.console.print(panel)
        self.console.print()

    def display_help(self):
        """Display help information"""
        help_text = """
## Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/clear` | Clear conversation history |
| `/exit` | Exit the chat application |
| `Ctrl+C` | Emergency exit |

## Tips
- The AI remembers previous messages in the conversation
- You can use markdown formatting in your responses
- Responses are displayed with rich formatting
        """

        panel = Panel(
            Markdown(help_text),
            title="[bold green]Help[/bold green]",
            border_style="green",
            padding=(1, 2)
        )

        self.console.print(panel)
        self.console.print()

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.console.print("[yellow]Conversation history cleared![/yellow]")
        self.console.print()

    def display_user_message(self, message: str):
        """Display user message with nice formatting"""
        panel = Panel(
            Text(message, style="white"),
            title="[bold cyan]You[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )
        self.console.print(panel)
        self.console.print()

    async def display_assistant_message(self, message: str):
        """Display assistant message with markdown rendering"""
        try:
            # Render as markdown for rich formatting
            markdown_content = Markdown(message)
            panel = Panel(
                markdown_content,
                title="[bold magenta]AI Assistant[/bold magenta]",
                border_style="magenta",
                padding=(1, 2)
            )
            self.console.print(panel)
            self.console.print()
        except Exception:
            # Fallback to plain text if markdown parsing fails
            panel = Panel(
                Text(message, style="white"),
                title="[bold magenta]AI Assistant[/bold magenta]",
                border_style="magenta",
                padding=(1, 2)
            )
            self.console.print(panel)
            self.console.print()

    async def get_user_input(self) -> str:
        """Get user input with rich prompt"""
        try:
            message = Prompt.ask("[bold blue]You[/bold blue]", default="")
            return message.strip()
        except KeyboardInterrupt:
            return "/exit"

    async def handle_command(self, command: str) -> bool:
        """Handle special commands"""
        command = command.lower().strip()

        if command in ['/exit', 'exit', 'quit']:
            self.console.print("[yellow]Goodbye! 👋[/yellow]")
            return False

        elif command == '/help':
            self.display_help()
            return True

        elif command == '/clear':
            self.clear_history()
            return True

        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("Type `/help` to see available commands.")
            return True

    async def run(self):
        """Main chat loop"""
        self.display_welcome()

        while True:
            try:
                # Get user input
                user_message = await self.get_user_input()

                # Skip empty messages
                if not user_message:
                    continue

                # Handle commands
                if user_message.startswith('/'):
                    should_continue = await self.handle_command(user_message)
                    if not should_continue:
                        break
                    continue

                # Display user message
                self.display_user_message(user_message)

                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": user_message})

                # Show typing indicator
                with self.console.status("[bold green]AI is thinking...[/bold green]", spinner="dots"):
                    try:
                        # Get AI response
                        ai_response = await self.client.chat_completion(
                            messages=self.conversation_history
                        )

                        # Add to conversation history
                        self.conversation_history.append({"role": "assistant", "content": ai_response})

                        # Display AI response
                        await self.display_assistant_message(ai_response)

                    except Exception as e:
                        error_panel = Panel(
                            Text(f"Error: {str(e)}", style="red"),
                            title="[bold red]Error[/bold red]",
                            border_style="red",
                            padding=(1, 2)
                        )
                        self.console.print(error_panel)
                        self.console.print()

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Unexpected error: {str(e)}[/red]")


async def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("ZAI_API_KEY")

    if not api_key:
        console = Console()
        console.print("[red]Error: GLM API key not found![/red]")
        console.print("Please set the ZAI_API_KEY environment variable.")
        console.print("You can create a .env file with your API key:")
        console.print("ZAI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Create chat client and interface
    async with GLMChatClient(api_key) as client:
        chat_interface = ChatInterface(client)
        await chat_interface.run()


if __name__ == "__main__":
    # Set up event loop for Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
    except Exception as e:
        console = Console()
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        sys.exit(1)