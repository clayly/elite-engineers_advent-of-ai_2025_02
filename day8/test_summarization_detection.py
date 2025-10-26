#!/usr/bin/env python3
"""
Test script to demonstrate summarization detection
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

def detect_and_display_summarization(original_messages: list, response_messages: list):
    """Detect when summarization occurs and display the summarized content"""
    # Check if the number of messages has decreased significantly
    if len(response_messages) < len(original_messages) and len(original_messages) > 3:
        # Look for messages that contain summary content
        for msg in response_messages:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                # Check if this looks like a summary message
                if 'summary' in msg.content.lower() or 'previous conversation' in msg.content.lower():
                    print("📝 Conversation summarized to manage token usage:")
                    # Try to display just the summary part
                    summary_content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                    print(f"  Summary: {summary_content}")
                    print(f"  Original messages: {len(original_messages)} → After summarization: {len(response_messages)}")
                    print("")
                    return True
    return False

def test_summarization_detection():
    """Test the summarization detection functionality"""
    print("Testing summarization detection...")
    print("=" * 40)
    
    llm = initialize_llm()
    
    # Create agent with SummarizationMiddleware (low threshold for testing)
    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=llm,
                max_tokens_before_summary=100,  # Low threshold for testing
                messages_to_keep=3,  # Keep last 3 messages
            ),
        ],
    )
    
    # Initialize conversation with system message
    messages = [SystemMessage(content="You are a helpful AI assistant. Please be concise in your responses.")]
    
    # Add several messages to trigger summarization
    conversation_prompts = [
        "What is the capital of France?",
        "What is the population of Paris?",
        "What are some famous landmarks in Paris?",
        "What is the history of the Eiffel Tower?",
        "What is French cuisine known for?",
        "What are some popular French dishes?",
    ]
    
    for i, prompt in enumerate(conversation_prompts):
        print(f"\nRound {i+1}:")
        print(f"User: {prompt}")
        
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = agent.invoke({"messages": messages})
            
            # Detect and display summarization
            detect_and_display_summarization(messages, response["messages"])
            
            ai_response = response["messages"][-1].content
            print(f"AI: {ai_response}")
            
            # Add AI response to history
            messages.append(response["messages"][-1])
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
    
    print("\n" + "=" * 40)
    print("Summarization detection test completed!")
    return True

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in environment variables")
        exit(1)
    
    test_summarization_detection()