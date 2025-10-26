#!/usr/bin/env python3
"""
Simple console AI chat using GLM API from z.ai with LangChain
Day 8 implementation for Elite Engineers Advent of AI 2025
"""

import os
import sys
from typing import List
from dotenv import load_dotenv

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

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()

# System message for the AI
SYSTEM_MESSAGE = """You are a helpful AI assistant powered by z.ai's GLM-4.6 model. 
You can answer questions, help with tasks, and have engaging conversations. 
Be concise but helpful in your responses."""

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
    console.print(Panel.fit("[bold blue]AI Chat with LangChain[/bold blue]", border_style="blue"))
    console.print("Welcome to the simple AI chat application!")
    console.print("Type 'exit' or 'quit' to end the conversation.")
    console.print("")

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

def main():
    """Main chat loop"""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not found in environment variables")
        console.print("Please set your API key in a .env file")
        sys.exit(1)
    
    # Initialize components
    display_welcome()
    llm = initialize_llm()
    
    # Create agent with SummarizationMiddleware to handle long conversations
    agent = create_agent(
        model=llm,
        tools=[],  # No tools needed for simple chat
        middleware=[
            SummarizationMiddleware(
                model=llm,  # Use the same model for summarization
                max_tokens_before_summary=200,  # Trigger summarization at 4000 tokens
                messages_to_keep=10,  # Keep last 10 messages after summary
            ),
        ],
    )
    
    # Initialize conversation history with system message
    messages = [SystemMessage(content=SYSTEM_MESSAGE)]
    
    # Main chat loop
    while True:
        user_input = get_user_input()
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit']:
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break
        
        # Add user message to history
        messages.append(HumanMessage(content=user_input))
        
        # Get and display AI response using the agent
        try:
            with console.status("[bold blue]Thinking...", spinner="dots"):
                with get_openai_callback() as cb:
                    response = agent.invoke({"messages": messages})
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

if __name__ == "__main__":
    main()