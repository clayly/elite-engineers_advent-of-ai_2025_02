#!/usr/bin/env python3
"""
Simple test for summarization detection function
"""

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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

def test_detection_function():
    """Test the detection function with simulated data"""
    print("Testing summarization detection function...")
    print("=" * 40)
    
    # Simulate original messages (many messages)
    original_messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        HumanMessage(content="How are you?"),
        AIMessage(content="I'm doing well!"),
        HumanMessage(content="What's the weather like?"),
        AIMessage(content="It's sunny today."),
        HumanMessage(content="That's nice."),
        AIMessage(content="Yes, it is."),
        HumanMessage(content="What about tomorrow?"),
        AIMessage(content="Tomorrow will be cloudy."),
    ]
    
    # Simulate response messages (summarized, fewer messages)
    # This simulates what might happen after summarization
    response_messages = [
        SystemMessage(content="System message"),
        AIMessage(content="Previous conversation summary: User greeted and we discussed weather..."),
        HumanMessage(content="What about tomorrow?"),
        AIMessage(content="Tomorrow will be cloudy."),
    ]
    
    print(f"Original messages: {len(original_messages)}")
    print(f"Response messages: {len(response_messages)}")
    
    # Test the detection function
    result = detect_and_display_summarization(original_messages, response_messages)
    
    if result:
        print("✓ Summarization detection worked correctly!")
    else:
        print("✗ Summarization detection did not trigger as expected.")
    
    print("\n" + "=" * 40)
    print("Detection function test completed!")

if __name__ == "__main__":
    test_detection_function()