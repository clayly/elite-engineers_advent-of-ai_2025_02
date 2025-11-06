#!/usr/bin/env python3
"""
Simple test script to verify that the agent with SummarizationMiddleware works correctly
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

# Load environment variables
load_dotenv()

def initialize_llm():
    """Initialize the LLM with z.ai API configuration"""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "glm-4.6"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4/"),
        streaming=False
    )

def test_agent_with_middleware():
    """Test the agent with SummarizationMiddleware"""
    llm = initialize_llm()
    
    # Create agent with SummarizationMiddleware
    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=llm,
                max_tokens_before_summary=4000,
                messages_to_keep=10,
            ),
        ],
    )
    
    # Test with a simple conversation
    messages = [SystemMessage(content="You are a helpful AI assistant.")]
    messages.append(HumanMessage(content="Hello, how are you?"))
    
    try:
        response = agent.invoke({"messages": messages})
        print("Agent response:", response["messages"][-1].content)
        print("Test successful!")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in environment variables")
        exit(1)
    
    test_agent_with_middleware()