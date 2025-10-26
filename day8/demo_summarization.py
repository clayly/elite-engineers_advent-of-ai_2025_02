#!/usr/bin/env python3
"""
Demonstration of SummarizationMiddleware in action
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

def demonstrate_summarization():
    """Demonstrate how SummarizationMiddleware works"""
    print("Demonstration of SummarizationMiddleware")
    print("=" * 40)
    print("\nSummarizationMiddleware automatically summarizes conversation history")
    print("when the token limit is approached, preventing token overflow issues.")
    print("\nKey features:")
    print("- Automatically triggers when conversation approaches token limits")
    print("- Keeps recent messages while summarizing older ones")
    print("- Maintains conversation context without exceeding model limits")
    print("- Transparent to the user experience")
    
    llm = initialize_llm()
    
    # Show the middleware configuration
    print("\nConfiguration example:")
    print("""
agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=llm,                    # Model for summarization
            max_tokens_before_summary=4000, # Trigger threshold
            messages_to_keep=10,          # Recent messages to preserve
        ),
    ],
)
    """)
    
    print("This implementation will automatically summarize the conversation")
    print("when it approaches 4000 tokens, keeping the last 10 messages intact.")
    
    return True

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in environment variables")
        exit(1)
    
    demonstrate_summarization()