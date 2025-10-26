#!/usr/bin/env python3
"""
Test script to verify token counting functionality
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks import get_openai_callback

# Load environment variables
load_dotenv()

def test_token_counting():
    """Test token counting functionality"""
    # Initialize the LLM with z.ai API configuration
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "glm-4.6"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4/"),
    )
    
    # Test messages
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Hello, how are you?")
    ]
    
    print("Testing token counting...")
    
    # Test with callback
    with get_openai_callback() as cb:
        response = llm.invoke(messages)
        print(f"Response: {response.content}")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost: ${cb.total_cost}")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in environment variables")
        print("Please set your API key in a .env file")
        exit(1)
    
    test_token_counting()