#!/usr/bin/env python3
"""
Simple console AI chat using GLM API from z.ai
Day 1 implementation for Elite Engineers Advent of AI 2025
"""

import asyncio
import os
import sys
from typing import List, Dict, Optional, Union
from enum import Enum

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.json import JSON
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
import json
import re
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


def extract_json_from_markdown(text: str) -> dict:
    """Extract JSON content from markdown code blocks."""
    # Try to find JSON blocks in ```json...``` format
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        # Use the first JSON block found
        json_text = matches[0].strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # Try to find any JSON-like structure in the text
    try:
        # Look for content between { and }
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            json_text = text[brace_start:brace_end + 1]
            return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not extract valid JSON from: {text[:200]}...")


class StructuredOutputMode(Enum):
    """Enum for different structured output modes"""
    NORMAL = "normal"
    STRUCTURED = "structured"


class BasicResponse(BaseModel):
    """Basic structured response with answer and follow-up"""
    answer: str = Field(description="Direct answer to user's question")
    followup_question: Optional[str] = Field(default=None, description="Suggested follow-up question")
    confidence: Optional[float] = Field(default=None, description="Confidence score from 0 to 1")


class StructuredAnswer(BaseModel):
    """Detailed answer with sections and bullet points"""
    summary: str = Field(description="Brief summary of the answer")
    details: List[str] = Field(description="Detailed points in bullet form")
    sources: Optional[List[str]] = Field(default=None, description="Source references if any")
    confidence: Optional[float] = Field(default=None, description="Confidence score from 0 to 1")


class DataExtraction(BaseModel):
    """For extracting specific data from text"""
    extracted_data: Dict[str, Union[str, int, float, bool, List]] = Field(description="Extracted information organized by key")
    extraction_notes: Optional[str] = Field(default=None, description="Notes about the extraction process")
    confidence: Optional[float] = Field(default=None, description="Confidence in extraction accuracy")


class CreativeResponse(BaseModel):
    """For creative content like jokes, stories, poems"""
    content: str = Field(description="The main creative content")
    genre: Optional[str] = Field(default=None, description="Type of creative content (joke, story, poem, etc.)")
    mood: Optional[str] = Field(default=None, description="Mood or tone of the content")
    rating: Optional[float] = Field(default=None, description="Quality rating from 1 to 10")


class ErrorResponse(BaseModel):
    """For structured error responses"""
    error_type: str = Field(description="Type of error that occurred")
    error_message: str = Field(description="Detailed error message")
    suggestion: Optional[str] = Field(default=None, description="Suggested resolution or workaround")


class GLMChatClient:
    """Client for interacting with GLM API from z.ai using LangChain"""

    def __init__(self, api_key: str, base_url: str = "https://api.z.ai/api/coding/paas/v4/"):
        self.api_key = api_key
        self.base_url = base_url
        self.console = Console()

        # Initialize LangChain ChatOpenAI with Z.AI configuration
        self.llm = ChatOpenAI(
            model="glm-4.6",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7,
            max_tokens=1000,
            streaming=False  # We'll handle streaming manually for better control
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "glm-4.6",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Send chat completion request to GLM API using LangChain"""

        try:
            # Convert message format for LangChain
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            # Update LLM parameters if needed
            self.llm.temperature = temperature
            self.llm.max_tokens = max_tokens

            # Call the model
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.llm.invoke, langchain_messages
            )

            return response.content

        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

    async def chat_completion_streaming(
        self,
        messages: List[Dict[str, str]],
        model: str = "glm-4.6",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Send streaming chat completion request to GLM API using LangChain"""

        try:
            # Convert message format for LangChain
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            # Create streaming LLM
            streaming_llm = ChatOpenAI(
                model=model,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )

            # Call the streaming model
            response = await asyncio.get_event_loop().run_in_executor(
                None, streaming_llm.invoke, langchain_messages
            )

            return response.content

        except Exception as e:
            raise Exception(f"Streaming API request failed: {str(e)}")

    async def chat_completion_structured(
        self,
        schema: BaseModel,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """Send structured chat completion request using JsonOutputParser"""
        try:
            # Create base LLM
            llm = ChatOpenAI(
                model="glm-4.6",
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=False
            )

            # Create JsonOutputParser with Pydantic schema
            parser = JsonOutputParser(pydantic_object=schema)

            # Create prompt with explicit JSON formatting instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a helpful assistant. Respond with valid JSON that matches the requested schema. "
                 "IMPORTANT: Do NOT wrap your response in markdown code blocks or backticks. "
                 "Output ONLY the raw JSON object without any formatting.\n\n"
                 "Format instructions:\n{format_instructions}"
                ),
                ("human", "{input}")
            ]).partial(format_instructions=parser.get_format_instructions())

            # Convert message format for LangChain
            # Get the last user message as input
            user_input = ""
            system_message = ""
            for msg in reversed(messages):
                if msg["role"] == "user" and not user_input:
                    user_input = msg["content"]
                elif msg["role"] == "system" and not system_message:
                    system_message = msg["content"]

            # Combine system messages if needed
            if system_message:
                final_input = f"System context: {system_message}\n\nUser request: {user_input}"
            else:
                final_input = user_input

            # Create chain and invoke
            chain = prompt | llm | parser
            result = await asyncio.get_event_loop().run_in_executor(
                None, chain.invoke, {"input": final_input}
            )

            return result

        except Exception as e:
            # Fallback: try to extract JSON from markdown-wrapped response
            try:
                # Create a simple LLM to get raw response
                llm = ChatOpenAI(
                    model="glm-4.6",
                    openai_api_key=self.api_key,
                    openai_api_base=self.base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=False
                )

                # Convert message format for LangChain
                langchain_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        langchain_messages.append(SystemMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))

                # Get raw response
                raw_response = await asyncio.get_event_loop().run_in_executor(
                    None, llm.invoke, langchain_messages
                )

                # Try to extract JSON from the markdown response
                json_data = extract_json_from_markdown(raw_response.content)

                # Validate against schema and create Pydantic object
                return schema.model_validate(json_data)

            except Exception as fallback_error:
                raise Exception(f"Structured API request failed: {str(e)}. Fallback also failed: {str(fallback_error)}")

    def detect_schema_type(self, user_input: str) -> BaseModel:
        """Detect which schema would be most appropriate for the user input"""
        user_input_lower = user_input.lower()

        # Creative content detection
        if any(word in user_input_lower for word in ['joke', 'story', 'poem', 'creative', 'funny', 'humor']):
            return CreativeResponse

        # Data extraction detection
        elif any(word in user_input_lower for word in ['extract', 'find', 'identify', 'list', 'data', 'information']):
            return DataExtraction

        # Detailed/complex question detection
        elif any(word in user_input_lower for word in ['explain', 'detailed', 'comprehensive', 'analysis', 'break down']):
            return StructuredAnswer

        # Default to basic response
        else:
            return BasicResponse


class ChatInterface:
    """Rich console interface for the chat"""

    def __init__(self, client: GLMChatClient):
        self.client = client
        self.console = Console()
        self.conversation_history: List[Dict[str, str]] = []
        self.structured_mode = StructuredOutputMode.NORMAL

    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
# 🤖 GLM AI Chat - Day 1
Welcome to your AI assistant powered by GLM API from z.ai!

**Commands:**
- Type your message and press Enter to chat
- Type `/help` to see this help
- Type `/clear` to clear conversation history
- Type `/structured` to toggle structured output mode
- Type `/exit` or `Ctrl+C` to quit

**Features:**
- Rich text formatting with syntax highlighting
- Markdown support
- Conversation history
- Streaming responses
- **Structured output** with Pydantic schemas
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
| `/structured` | Toggle structured output mode on/off |
| `/exit` | Exit the chat application |
| `Ctrl+C` | Emergency exit |

## Structured Output Mode
When structured mode is **ON**, responses are formatted as structured data with:
- **BasicResponse**: Simple Q&A with follow-up questions
- **StructuredAnswer**: Detailed answers with bullet points
- **CreativeResponse**: Jokes, stories, poems with metadata
- **DataExtraction**: Extracted information organized by keys
- **ErrorResponse**: Structured error information

## Tips
- The AI remembers previous messages in the conversation
- You can use markdown formatting in your responses
- Responses are displayed with rich formatting
- Structured output provides consistent, parseable responses
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

    def display_structured_response(self, response):
        """Display structured response with proper formatting"""
        # Handle both Pydantic models and dictionaries
        if hasattr(response, 'model_dump'):
            response_dict = response.model_dump()
            schema_name = response.__class__.__name__
        else:
            response_dict = response
            schema_name = "Structured Response"

        # Create title with schema type
        title = f"[bold magenta]AI Assistant - {schema_name}[/bold magenta]"

        # Display as formatted JSON
        json_content = JSON.from_data(response_dict, indent=2)
        panel = Panel(
            json_content,
            title=title,
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()

        # Display additional info based on schema type
        # Handle both Pydantic models and dictionaries
        if hasattr(response, 'confidence') and response.confidence:
            confidence_text = f"Confidence: {response.confidence:.1%}"
            self.console.print(f"[dim cyan]{confidence_text}[/dim cyan]")
        elif isinstance(response_dict, dict) and response_dict.get('confidence'):
            confidence = response_dict['confidence']
            if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                confidence_text = f"Confidence: {confidence:.1%}"
                self.console.print(f"[dim cyan]{confidence_text}[/dim cyan]")

        if hasattr(response, 'followup_question') and response.followup_question:
            followup_text = f"Suggested follow-up: {response.followup_question}"
            self.console.print(f"[dim yellow]{followup_text}[/dim yellow]")
        elif isinstance(response_dict, dict) and response_dict.get('followup_question'):
            followup_text = f"Suggested follow-up: {response_dict['followup_question']}"
            self.console.print(f"[dim yellow]{followup_text}[/dim yellow]")

        self.console.print()

    def toggle_structured_mode(self):
        """Toggle between normal and structured output modes"""
        if self.structured_mode == StructuredOutputMode.NORMAL:
            self.structured_mode = StructuredOutputMode.STRUCTURED
            self.console.print("[green]✓ Structured output mode enabled![/green]")
            self.console.print("[dim]Responses will now be formatted as structured data.[/dim]")
        else:
            self.structured_mode = StructuredOutputMode.NORMAL
            self.console.print("[yellow]✓ Structured output mode disabled.[/yellow]")
            self.console.print("[dim]Responses will use normal markdown formatting.[/dim]")
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

        elif command == '/structured':
            self.toggle_structured_mode()
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
                        # Check if structured mode is enabled
                        if self.structured_mode == StructuredOutputMode.STRUCTURED:
                            # Auto-detect schema type
                            schema_class = self.client.detect_schema_type(user_message)

                            # Add system message about structured output
                            structured_messages = self.conversation_history.copy()
                            structured_messages.insert(0, {
                                "role": "system",
                                "content": f"You are a helpful assistant that provides structured responses. "
                                f"Respond with valid JSON that matches the {schema_class.__name__} schema. "
                                f"IMPORTANT: Output ONLY raw JSON without markdown formatting, backticks, or code blocks. "
                                f"Be comprehensive, accurate, and provide complete responses that fully address the user's request."
                            })

                            # Get structured AI response
                            structured_response = await self.client.chat_completion_structured(
                                schema=schema_class,
                                messages=structured_messages
                            )

                            # Convert structured response to string for history
                            if hasattr(structured_response, 'model_dump'):
                                response_dict = structured_response.model_dump()
                            else:
                                response_dict = structured_response
                            response_text = str(response_dict)
                            self.conversation_history.append({"role": "assistant", "content": response_text})

                            # Display structured response
                            self.display_structured_response(structured_response)

                        else:
                            # Normal mode - get regular AI response
                            ai_response = await self.client.chat_completion(
                                messages=self.conversation_history
                            )

                            # Add to conversation history
                            self.conversation_history.append({"role": "assistant", "content": ai_response})

                            # Display AI response
                            await self.display_assistant_message(ai_response)

                    except Exception as e:
                        # Create structured error response
                        error_response = ErrorResponse(
                            error_type="API_Error",
                            error_message=str(e),
                            suggestion="Try again or use /structured to toggle modes"
                        )

                        # Display structured error
                        self.display_structured_response(error_response)

                        # Also add to conversation history
                        error_dict = error_response.model_dump() if hasattr(error_response, 'model_dump') else error_response
                        error_text = str(error_dict)
                        self.conversation_history.append({"role": "assistant", "content": error_text})

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
    client = GLMChatClient(api_key)
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