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

def process_message(llm, messages: List) -> str:
    """Process user message and get AI response"""
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return "Sorry, I encountered an error processing your request."

def display_ai_response(response: str):
    """Display AI response with rich formatting"""
    console.print("[bold blue]AI Assistant:[/bold blue]")
    # Try to parse as markdown, fallback to plain text
    try:
        md = Markdown(response)
        console.print(md)
    except:
        console.print(response)
    console.print("")

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
    
    # Initialize conversation history
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
        
        # Get and display AI response
        with console.status("[bold blue]Thinking...", spinner="dots"):
            ai_response = process_message(llm, messages)
        
        display_ai_response(ai_response)
        
        # Add AI response to history
        messages.append(AIMessage(content=ai_response))

if __name__ == "__main__":
    main()