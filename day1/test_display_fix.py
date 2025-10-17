#!/usr/bin/env python3
"""
Test script to verify display_structured_response handles both dicts and Pydantic models
"""

import json
import re
from pydantic import BaseModel, Field

# Mock the display function
def display_structured_response(response):
    """Display structured response with proper formatting"""
    # Handle both Pydantic models and dictionaries
    if hasattr(response, 'model_dump'):
        response_dict = response.model_dump()
        schema_name = response.__class__.__name__
    else:
        response_dict = response
        schema_name = "Structured Response"

    print(f"Schema Name: {schema_name}")
    print(f"Response Dict: {response_dict}")

    # Display additional info based on schema type
    # Handle both Pydantic models and dictionaries
    if hasattr(response, 'confidence') and response.confidence:
        confidence_text = f"Confidence: {response.confidence:.1%}"
        print(f"From Pydantic: {confidence_text}")
    elif isinstance(response_dict, dict) and response_dict.get('confidence'):
        confidence = response_dict['confidence']
        if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
            confidence_text = f"Confidence: {confidence:.1%}"
            print(f"From Dict: {confidence_text}")

    if hasattr(response, 'followup_question') and response.followup_question:
        followup_text = f"Suggested follow-up: {response.followup_question}"
        print(f"From Pydantic: {followup_text}")
    elif isinstance(response_dict, dict) and response_dict.get('followup_question'):
        followup_text = f"Suggested follow-up: {response_dict['followup_question']}"
        print(f"From Dict: {followup_text}")

# Test Pydantic model
class BasicResponse(BaseModel):
    answer: str = Field(description="Direct answer to user's question")
    followup_question: str = Field(default=None, description="Suggested follow-up question")
    confidence: float = Field(default=None, description="Confidence score from 0 to 1")

def test_display_fix():
    print("Testing display_structured_response with both dict and Pydantic models...")
    print("=" * 70)

    # Test 1: Pydantic model
    print("\n[Test 1] Pydantic Model:")
    pydantic_response = BasicResponse(
        answer="The capital of France is Paris.",
        followup_question="Would you like to know more about Paris?",
        confidence=0.95
    )
    display_structured_response(pydantic_response)

    # Test 2: Dictionary (what JsonOutputParser returns)
    print("\n[Test 2] Dictionary:")
    dict_response = {
        "answer": "The capital of France is Paris.",
        "followup_question": "Would you like to know more about Paris?",
        "confidence": 0.95
    }
    display_structured_response(dict_response)

    # Test 3: Dictionary without optional fields
    print("\n[Test 3] Dictionary without optional fields:")
    dict_response_minimal = {
        "answer": "Hello world!"
    }
    display_structured_response(dict_response_minimal)

    print("\n" + "=" * 70)
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_display_fix()