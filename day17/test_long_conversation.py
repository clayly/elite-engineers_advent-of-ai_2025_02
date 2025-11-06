#!/usr/bin/env python3
"""
Test script to demonstrate the SummarizationMiddleware with a long conversation
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

def test_long_conversation():
    """Test the agent with SummarizationMiddleware using a long conversation"""
    llm = initialize_llm()
    
    # Create agent with SummarizationMiddleware
    # Set a lower threshold for testing purposes
    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=llm,
                max_tokens_before_summary=1000,  # Lower threshold for testing
                messages_to_keep=5,  # Keep last 5 messages
            ),
        ],
    )
    
    # Initialize conversation
    messages = [SystemMessage(content="You are a helpful AI assistant. Please be concise in your responses.")]
    
    # Add several messages to the conversation
    conversation_prompts = [
        "What is the capital of France?",
        "What is the population of Paris?",
        "What are some famous landmarks in Paris?",
        "What is the history of the Eiffel Tower?",
        "What is French cuisine known for?",
        "What are some popular French dishes?",
        "What is the significance of French wine?",
        "What are some famous French artists?",
        "What is the Louvre Museum known for?",
        "What is the cultural significance of Paris?"
    ]
    
    print("Testing long conversation with SummarizationMiddleware...")
    print("=" * 50)
    
    for i, prompt in enumerate(conversation_prompts):
        print(f"\nRound {i+1}:")
        print(f"User: {prompt}")
        
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = agent.invoke({"messages": messages})
            ai_response = response["messages"][-1].content
            print(f"AI: {ai_response}")
            
            # Add AI response to history
            messages.append(response["messages"][-1])
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
    
    print("\n" + "=" * 50)
    print("Long conversation test completed successfully!")
    print(f"Total messages in history: {len(messages)}")
    return True

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in environment variables")
        exit(1)
    
    test_long_conversation()