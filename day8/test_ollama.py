#!/usr/bin/env python3
"""
Simple test script to verify Ollama integration
"""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

def test_ollama():
    """Test Ollama integration"""
    print("Testing Ollama integration...")
    
    # Configure for Ollama
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL_NAME", "llama3.1"),
        temperature=0.7,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    
    # Test query
    response = llm.invoke("Hello, who are you?")
    print(f"Response: {response.content}")

if __name__ == "__main__":
    test_ollama()