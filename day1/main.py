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


def load_llm_config(config_path: str = "llm.json") -> dict:
    """Load LLM configuration from JSON file with validation"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Validate and clamp temperature to safe range for z.ai API (0.0 to 1.0)
        if "temperature" in config:
            temp = config["temperature"]
            if temp < 0.0:
                print(f"Warning: Temperature {temp} is too low, setting to 0.0")
                config["temperature"] = 0.0
            elif temp > 1.0:
                print(f"Warning: Temperature {temp} is too high for z.ai API, setting to 1.0")
                config["temperature"] = 1.0

        # Ensure max_tokens is reasonable
        if "max_tokens" in config and config["max_tokens"] > 4000:
            print(f"Warning: max_tokens {config['max_tokens']} may be too high, setting to 4000")
            config["max_tokens"] = 4000

        return config

    except FileNotFoundError:
        # Return default config if file doesn't exist
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "glm-4.6",
            "streaming": False,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {config_path}, using defaults: {e}")
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "glm-4.6",
            "streaming": False,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }


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


# Technical Specification Collector Models
class RequirementCategory(str, Enum):
    """Categories of requirements for technical specifications"""
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    TECHNICAL = "technical"
    BUSINESS = "business"
    UI_UX = "ui_ux"
    SECURITY = "security"
    PERFORMANCE = "performance"


class Requirement(BaseModel):
    """Individual requirement with metadata"""
    id: str = Field(description="Unique identifier for the requirement")
    category: RequirementCategory = Field(description="Category of the requirement")
    title: str = Field(description="Brief title of the requirement")
    description: str = Field(description="Detailed description of the requirement")
    priority: str = Field(description="Priority level (high, medium, low)")
    acceptance_criteria: Optional[List[str]] = Field(default=None, description="Criteria for requirement acceptance")
    dependencies: Optional[List[str]] = Field(default=None, description="Dependencies on other requirements")


class TechnicalSpecification(BaseModel):
    """Complete technical specification document"""
    project_name: str = Field(description="Name of the project")
    project_description: str = Field(description="Brief description of the project")
    requirements: List[Requirement] = Field(description="List of all requirements")
    completeness_score: float = Field(description="Score from 0 to 1 indicating completeness")
    missing_categories: List[RequirementCategory] = Field(description="Categories that need more requirements")
    next_questions: List[str] = Field(description="Suggested questions to gather missing information")
    is_ready_for_review: bool = Field(description="Whether the specification is ready for review")


class TZCollectorState(BaseModel):
    """State tracking for technical specification collection"""
    phase: str = Field(description="Current phase of collection (initial, gathering, reviewing, complete)")
    project_type: Optional[str] = Field(default=None, description="Type of project being specified")
    current_category: Optional[RequirementCategory] = Field(default=None, description="Currently focused category")
    requirements_count: int = Field(description="Total number of requirements collected")
    last_completed_category: Optional[RequirementCategory] = Field(default=None, description="Last category that was completed")
    should_complete: bool = Field(default=False, description="Whether collection should be considered complete")
    accumulated_requirements: List[Requirement] = Field(default_factory=list, description="All requirements collected so far")
    asked_questions: List[str] = Field(default_factory=list, description="Questions that have already been asked")
    collected_info: Dict[str, str] = Field(default_factory=dict, description="Key information collected from user")


class GLMChatClient:
    """Client for interacting with GLM API from z.ai using LangChain"""

    def __init__(self, api_key: str, base_url: str = "https://api.z.ai/api/coding/paas/v4/"):
        self.api_key = api_key
        self.base_url = base_url
        self.console = Console()

        # Load LLM configuration
        self.llm_config = load_llm_config()

        # Initialize LangChain ChatOpenAI with Z.AI configuration
        self.llm = ChatOpenAI(
            model=self.llm_config["model"],
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=self.llm_config["temperature"],
            max_tokens=self.llm_config["max_tokens"],
            streaming=False  # We'll handle streaming manually for better control
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send chat completion request to GLM API using LangChain"""

        try:
            # Reload config on every call to pick up live changes
            current_config = load_llm_config()

            # Use config values or provided parameters
            effective_model = model or current_config["model"]
            effective_temperature = temperature or current_config["temperature"]
            effective_max_tokens = max_tokens or current_config["max_tokens"]

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
            self.llm.temperature = effective_temperature
            self.llm.max_tokens = effective_max_tokens

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
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send streaming chat completion request to GLM API using LangChain"""

        try:
            # Reload config on every call to pick up live changes
            current_config = load_llm_config()

            # Use config values or provided parameters
            effective_model = model or current_config["model"]
            effective_temperature = temperature or current_config["temperature"]
            effective_max_tokens = max_tokens or current_config["max_tokens"]

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
                model=effective_model,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Send structured chat completion request using JsonOutputParser"""
        try:
            # Reload config on every call to pick up live changes
            current_config = load_llm_config()

            # Use config values or provided parameters
            effective_temperature = temperature or current_config["temperature"]
            effective_max_tokens = max_tokens or current_config["max_tokens"]

            # Create base LLM
            llm = ChatOpenAI(
                model=current_config["model"],
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
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
            raw_result = await asyncio.get_event_loop().run_in_executor(
                None, chain.invoke, {"input": final_input}
            )

            # Validate result is a Pydantic model
            if not isinstance(raw_result, schema):
                # Try to convert dict to Pydantic model
                if isinstance(raw_result, dict):
                    try:
                        result = schema.model_validate(raw_result)
                    except Exception as validation_error:
                        # Fallback: create minimal valid response
                        result = schema.model_validate({
                            "project_name": "Unknown Project",
                            "project_description": "Error during validation",
                            "requirements": [],
                            "completeness_score": 0.0,
                            "missing_categories": list(RequirementCategory),
                            "next_questions": ["Please provide more details"],
                            "is_ready_for_review": False
                        })
                        print(f"DEBUG: Validation error handled: {validation_error}")
                else:
                    result = raw_result
            else:
                result = raw_result

            return result

        except Exception as e:
            # Fallback: try to extract JSON from markdown-wrapped response
            try:
                # Reload config again for fallback (in case file changed since start of method)
                fallback_config = load_llm_config()

                # Create a simple LLM to get raw response
                llm = ChatOpenAI(
                    model=fallback_config["model"],
                    openai_api_key=self.api_key,
                    openai_api_base=self.base_url,
                    temperature=effective_temperature,  # Keep original effective temperature
                    max_tokens=effective_max_tokens,    # Keep original effective max_tokens
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

        # Technical specification detection
        if any(word in user_input_lower for word in [
            'тз', 'техническое задание', 'требования', 'specification', 'requirements',
            'project requirements', 'система', 'приложение', 'разработка', 'проект',
            'создать', 'разработать', 'спроектировать', 'спецификация'
        ]):
            return TechnicalSpecification

        # Creative content detection
        elif any(word in user_input_lower for word in ['joke', 'story', 'poem', 'creative', 'funny', 'humor']):
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
        self.tz_collector_state: Optional[TZCollectorState] = None
        self.tz_mode = False

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
- Type `/tz` to start technical specification collector mode
- Type `/exit` or `Ctrl+C` to quit

**Features:**
- Rich text formatting with syntax highlighting
- Markdown support
- Conversation history
- Streaming responses
- **Structured output** with Pydantic schemas
- **Technical Specification collector** - interactive ТЗ gathering
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
| `/tz` | Start Technical Specification collector mode |
| `/exit` | Exit the chat application |
| `Ctrl+C` | Emergency exit |

## Structured Output Mode
When structured mode is **ON**, responses are formatted as structured data with:
- **BasicResponse**: Simple Q&A with follow-up questions
- **StructuredAnswer**: Detailed answers with bullet points
- **CreativeResponse**: Jokes, stories, poems with metadata
- **DataExtraction**: Extracted information organized by keys
- **TechnicalSpecification**: Complete technical specifications
- **ErrorResponse**: Structured error information

## Technical Specification Mode (`/tz`)
Interactive ТЗ collection with automatic completion detection:
- **Smart questions** to gather project requirements
- **Category-based** collection (functional, non-functional, technical, etc.)
- **Automatic stopping** when requirements are complete
- **Ready-to-use** technical specification document

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
        """Clear conversation history and reset all modes"""
        # Clear conversation history
        self.conversation_history = []

        # Reset all modes to defaults
        self.structured_mode = StructuredOutputMode.NORMAL
        self.tz_mode = False
        self.tz_collector_state = None

        # Provide feedback about what was cleared
        self.console.print("[yellow]✓ Conversation history cleared![/yellow]")
        self.console.print("[dim]✓ Modes reset to defaults[/dim]")
        self.console.print("[dim]✓ Technical specification state cleared[/dim]")
        self.console.print()

        # Show current mode status
        status_messages = []
        if self.structured_mode == StructuredOutputMode.STRUCTURED:
            status_messages.append("[green]Structured mode: ON[/green]")
        else:
            status_messages.append("[dim]Structured mode: OFF[/dim]")

        if self.tz_mode:
            status_messages.append("[green]Technical specification mode: ON[/green]")
        else:
            status_messages.append("[dim]Technical specification mode: OFF[/dim]")

        if status_messages:
            self.console.print(" | ".join(status_messages))
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

    def start_tz_mode(self):
        """Start Technical Specification collector mode"""
        self.tz_mode = True
        self.tz_collector_state = TZCollectorState(
            phase="initial",
            requirements_count=0
        )

        self.console.print("[green]🎯 Technical Specification Collector Mode[/green]")
        self.console.print("[dim]I'll help you create a complete technical specification.")
        self.console.print("[dim]I'll ask targeted questions and automatically detect when we have enough information.[/dim]")
        self.console.print()

        # Add initial system message for TZ collection
        tz_welcome = "Давайте создадим полное техническое задание для вашего проекта. Расскажите, что вы хотите разработать, и я буду задавать уточняющие вопросы для сбора всех необходимых требований."
        self.conversation_history.append({"role": "assistant", "content": tz_welcome})

        self.console.print("[bold magenta]AI Assistant[/bold magenta]")
        panel = Panel(
            Text(tz_welcome, style="white"),
            title="[bold magenta]ТЗ Коллектор[/bold magenta]",
            border_style="magenta",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()

    def update_tz_state(self, tz_response):
        """Update TZ collector state based on response"""
        if self.tz_collector_state:
            # Handle both Pydantic objects and dictionaries
            if hasattr(tz_response, 'requirements'):
                requirements = tz_response.requirements
                is_ready = tz_response.is_ready_for_review
                next_questions = tz_response.next_questions if hasattr(tz_response, 'next_questions') else []
            elif isinstance(tz_response, dict):
                requirements = tz_response.get('requirements', [])
                is_ready = tz_response.get('is_ready_for_review', False)
                next_questions = tz_response.get('next_questions', [])
            else:
                requirements = []
                is_ready = False
                next_questions = []

            # Update accumulated requirements
            for req in requirements:
                # Check if requirement already exists (by ID or title)
                req_exists = False
                for existing_req in self.tz_collector_state.accumulated_requirements:
                    if (hasattr(req, 'id') and hasattr(existing_req, 'id') and req.id == existing_req.id) or \
                       (hasattr(req, 'title') and hasattr(existing_req, 'title') and req.title == existing_req.title):
                        req_exists = True
                        break

                if not req_exists:
                    self.tz_collector_state.accumulated_requirements.append(req)

            # Update asked questions
            for question in next_questions:
                if question not in self.tz_collector_state.asked_questions:
                    self.tz_collector_state.asked_questions.append(question)

            self.tz_collector_state.requirements_count = len(self.tz_collector_state.accumulated_requirements)
            self.tz_collector_state.should_complete = is_ready

            if is_ready:
                self.tz_collector_state.phase = "complete"
                self.console.print("[green]✓ Technical Specification is complete![/green]")
                self.console.print("[dim]The specification is ready for review and implementation.[/dim]")
                self.console.print()

    def create_comprehensive_tz_response(self, current_response):
        """Create comprehensive response that includes all accumulated requirements"""
        if not self.tz_collector_state:
            return current_response

        # Get all accumulated requirements
        all_requirements = list(self.tz_collector_state.accumulated_requirements)

        # Add current response requirements if they're not already included
        if hasattr(current_response, 'requirements'):
            current_reqs = current_response.requirements
        elif isinstance(current_response, dict):
            current_reqs = current_response.get('requirements', [])
        else:
            current_reqs = []

        for req in current_reqs:
            req_exists = False
            for existing_req in all_requirements:
                if (hasattr(req, 'id') and hasattr(existing_req, 'id') and req.id == existing_req.id) or \
                   (hasattr(req, 'title') and hasattr(existing_req, 'title') and req.title == existing_req.title):
                    req_exists = True
                    break

            if not req_exists:
                all_requirements.append(req)

        # Calculate completeness based on all requirements
        completeness_score = min(0.9, len(all_requirements) * 0.15)  # Progressive completeness

        # Determine missing categories
        categories_present = set()
        for req in all_requirements:
            if hasattr(req, 'category'):
                categories_present.add(req.category)
            elif isinstance(req, dict):
                categories_present.add(req.get('category', 'unknown'))

        all_categories = set(RequirementCategory)
        missing_categories = list(all_categories - categories_present)

        # Get next questions that haven't been asked
        if hasattr(current_response, 'next_questions'):
            current_questions = current_response.next_questions
        elif isinstance(current_response, dict):
            current_questions = current_response.get('next_questions', [])
        else:
            current_questions = []

        # Filter out already asked questions
        new_questions = [q for q in current_questions if q not in self.tz_collector_state.asked_questions]

        # If no new questions from AI, let LLM handle it naturally (no forced generation)

        # Determine if ready for review
        is_ready = len(all_requirements) >= 5 and len(missing_categories) <= 2

        # Create comprehensive response
        if hasattr(current_response, 'project_name'):
            project_name = current_response.project_name
            project_description = current_response.project_description
        elif isinstance(current_response, dict):
            project_name = current_response.get('project_name', 'Проект')
            project_description = current_response.get('project_description', 'Описание проекта')
        else:
            project_name = 'Проект'
            project_description = 'Описание проекта'

        comprehensive_tz = TechnicalSpecification(
            project_name=project_name,
            project_description=project_description,
            requirements=all_requirements,
            completeness_score=completeness_score,
            missing_categories=list(missing_categories),
            next_questions=new_questions[:3] if new_questions else current_questions[:3],  # Use AI questions or fallback to current
            is_ready_for_review=is_ready
        )

        return comprehensive_tz

    def remove_repeated_questions(self, questions):
        """Remove questions that have been asked too many times"""
        if not questions:
            return []

        # Simple heuristic: if questions are too generic, replace them
        generic_patterns = [
            "Какие еще аспекты проекта важны для вас?",
            "Есть ли специфические требования, которые мы еще не обсудили?",
            "Какие риски нужно учесть при разработке?"
        ]

        filtered_questions = []
        for q in questions:
            # Skip if it's a generic question that was asked before
            if q in generic_patterns and q in self.tz_collector_state.asked_questions:
                continue
            filtered_questions.append(q)

        return filtered_questions

    async def handle_tz_collection(self, user_message: str):
        """Handle technical specification collection process"""

        # Store user input in collected info
        if self.tz_collector_state:
            # Extract key information from user message
            self.tz_collector_state.collected_info[f"user_input_{len(self.tz_collector_state.collected_info)}"] = user_message

        # Create comprehensive context from accumulated requirements
        accumulated_context = ""
        if self.tz_collector_state and self.tz_collector_state.accumulated_requirements:
            accumulated_context = "\nУЖЕ СОБРАННЫЕ ТРЕБОВАНИЯ:\n"
            for i, req in enumerate(self.tz_collector_state.accumulated_requirements, 1):
                if hasattr(req, 'title') and hasattr(req, 'description'):
                    accumulated_context += f"{i}. {req.title}: {req.description}\n"
                elif isinstance(req, dict):
                    accumulated_context += f"{i}. {req.get('title', 'Без названия')}: {req.get('description', 'Без описания')}\n"

        asked_questions_context = ""
        if self.tz_collector_state and self.tz_collector_state.asked_questions:
            asked_questions_context = "\nУЖЕ ЗАДАННЫЕ ВОПРОСЫ (ИЗБЕГАТЬ ПОВТОРЕНИЯ):\n"
            for q in self.tz_collector_state.asked_questions:
                asked_questions_context += f"- {q}\n"

        # Create specialized system prompt for TZ collection with full context
        tz_system_prompt = f"""
        Ты - профессиональный бизнес-аналитик, который помогает создавать полные технические задания.

        ТЕКУЩАЯ ФАЗА: {self.tz_collector_state.phase if self.tz_collector_state else 'initial'}
        УЖЕ СОБРАНО ТРЕБОВАНИЙ: {self.tz_collector_state.requirements_count if self.tz_collector_state else 0}

        {accumulated_context}

        {asked_questions_context}

        ВАЖНО:
        1. НЕ задавай вопросы, которые уже были заданы выше
        2. Учитывай всю информацию, предоставленную ранее
        3. Создавай вопросы РЕЛЕВАНТНЫЕ конкретному проекту (не универсальные)
        4. Избегай общих вопросов вроде "сколько пользователей" для локальных утилит
        5. ОБЯЗАТЕЛЬНО включи поле next_questions с 2-3 специфическими вопросами

        Проанализируй текущий ответ пользователя и:
        1. Извлеки новые требования из ответа
        2. Объедини их с уже собранными требованиями
        3. Определи, какие категории все еще нуждаются в дополнении
        4. Сформулируй 2-3 КОНКРЕТНЫХ вопроса, УЧЕСТЫВАЯ ТИП ПРОЕКТА:
           - Для калькулятора: операции, точность, интерфейс, особенности вычислений
           - Для локальной программы: установка, требования к системе, особенности использования
           - Для веб-приложения: браузеры, хостинг, пользователи, интерфейс
        5. Оцени полноту ВСЕХ собранных требований (0.0 - 1.0)
        6. Определи, готово ли ТЗ к передаче на разработку

        СТРУКТУРА ОТВЕТА (ОБЯЗАТЕЛЬНО ЗАПОЛНИТЬ ВСЕ ПОЛЯ):
        - project_name: название проекта
        - project_description: краткое описание
        - requirements: ВСЕ собранные требования с категориями
        - completeness_score: оценка от 0.0 до 1.0
        - missing_categories: категории для дополнения
        - next_questions: 2-3 КОНКРЕТНЫХ, РЕЛЕВАНТНЫЕ вопроса
        - is_ready_for_review: готово ли к рассмотрению

        ВНИМАНИЕ: Поле next_questions ОБЯЗАТЕЛЬНО должно содержать 2-3 вопроса!
        НЕ Оставляйте next_questions пустым!

        Пример ХОРОШИХ вопросов для калькулятора:
        - Какие математические операции должны поддерживаться?
        - Нужна ли история вычислений или работа с памятью?
        - Какой формат интерфейса предпочитается (кнопки, командная строка)?

        Пример полного ответа в JSON формате:
        {{
          "project_name": "Калькулятор",
          "project_description": "Локальный калькулятор на Python",
          "requirements": [...],
          "completeness_score": 0.4,
          "missing_categories": ["technical", "ui_ux"],
          "next_questions": [
            "Какие математические операции должны поддерживаться?",
            "Нужна ли история вычислений?"
          ],
          "is_ready_for_review": false
        }}

        Включи в ответ ВСЕ собранные требования, оценку полноты, ОБЯЗАТЕЛЬНО с вопросами в next_questions!
        Отвечай на русском языке.
        """

        # Create message history with context preservation
        tz_messages = []

        # Keep system message with accumulated context
        tz_messages.append({
            "role": "system",
            "content": tz_system_prompt
        })

        # Add recent conversation history (last 10 messages to maintain context)
        recent_history = self.conversation_history[-10:]
        tz_messages.extend(recent_history)

        try:
            # Get structured Technical Specification response
            tz_response = await self.client.chat_completion_structured(
                schema=TechnicalSpecification,
                messages=tz_messages
            )

            # Debug: print response type
            self.console.print(f"[dim]Response type: {type(tz_response)}[/dim]")

            # Update collector state
            self.update_tz_state(tz_response)

            # Create comprehensive response that includes all accumulated requirements
            comprehensive_response = self.create_comprehensive_tz_response(tz_response)

            # Debug: check if original response has questions
            original_questions = []
            if hasattr(tz_response, 'next_questions'):
                original_questions = tz_response.next_questions
            elif isinstance(tz_response, dict):
                original_questions = tz_response.get('next_questions', [])

            self.console.print(f"[dim]DEBUG: Original AI questions: {original_questions}[/dim]")

            # Update asked questions with the ones from comprehensive response (with filtering)
            if hasattr(comprehensive_response, 'next_questions') and self.tz_collector_state:
                filtered_questions = self.remove_repeated_questions(comprehensive_response.next_questions)
                self.console.print(f"[dim]DEBUG: Filtered questions: {filtered_questions}[/dim]")
                for question in filtered_questions:
                    if question not in self.tz_collector_state.asked_questions:
                        self.tz_collector_state.asked_questions.append(question)

            # Add response to conversation history
            if hasattr(comprehensive_response, 'model_dump'):
                response_dict = comprehensive_response.model_dump()
            else:
                response_dict = comprehensive_response
            response_text = str(response_dict)
            self.conversation_history.append({"role": "assistant", "content": response_text})

            # Display the structured response
            self.display_structured_response(comprehensive_response)

            # Check if collection should be completed
            is_ready = False
            if hasattr(tz_response, 'is_ready_for_review'):
                is_ready = tz_response.is_ready_for_review
            elif isinstance(tz_response, dict):
                is_ready = tz_response.get('is_ready_for_review', False)

            if is_ready:
                self.tz_mode = False
                self.console.print("[bold green]🎉 Техническое задание готово![/bold green]")
                self.console.print("[dim]Режим сбора ТЗ завершен. Вы можете продолжить обычный диалог или использовать другие команды.[/dim]")
                self.console.print()

        except Exception as e:
            # Enhanced error handling with debug info
            self.console.print(f"[red]DEBUG: Error in TZ collection: {str(e)}[/red]")
            self.console.print(f"[red]Error type: {type(e).__name__}[/red]")

            # Try fallback to regular chat response
            try:
                fallback_response = await self.client.chat_completion(
                    messages=tz_messages
                )
                self.conversation_history.append({"role": "assistant", "content": fallback_response})
                await self.display_assistant_message(fallback_response)
            except Exception as fallback_error:
                # If fallback also fails, show structured error
                error_response = ErrorResponse(
                    error_type="TZ_Collection_Error",
                    error_message=f"Primary error: {str(e)}. Fallback error: {str(fallback_error)}",
                    suggestion="Попробуйте переформулировать ответ или используйте /clear для сброса"
                )
                self.display_structured_response(error_response)

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

        elif command == '/tz':
            self.start_tz_mode()
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
                        # Check if TZ mode is active
                        if self.tz_mode and self.tz_collector_state:
                            # Handle Technical Specification collection
                            await self.handle_tz_collection(user_message)

                        # Check if structured mode is enabled
                        elif self.structured_mode == StructuredOutputMode.STRUCTURED:
                            # Auto-detect schema type
                            schema_class = self.client.detect_schema_type(user_message)

                            # Check if this is a technical specification request
                            if schema_class == TechnicalSpecification:
                                # Switch to TZ mode automatically
                                self.start_tz_mode()
                                await self.handle_tz_collection(user_message)
                            else:
                                # Regular structured response
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