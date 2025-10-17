#!/usr/bin/env python3
"""
Test script for JSON extraction from markdown
"""

import json
import re

def extract_json_from_markdown(text: str) -> dict:
    """Extract JSON content from markdown code blocks."""
    # Try to find JSON blocks in ```json...``` format
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        # Use the first JSON block found
        json_text = matches[0].strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # Try to find any JSON-like structure in the text
    try:
        # Look for content between { and }
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            json_text = text[brace_start:brace_end + 1]
            return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not extract valid JSON from: {text[:200]}...")

def test_json_extraction():
    """Test the JSON extraction function"""

    # Test case 1: JSON in markdown code block
    markdown_json = '''
    Here's the JSON response:

    ```json
    {
      "answer": "The capital of France is Paris.",
      "followup_question": "Would you like to know more about Paris?",
      "confidence": 0.95
    }
    ```

    This should be extracted correctly.
    '''

    try:
        result = extract_json_from_markdown(markdown_json)
        print("[PASS] Test 1: JSON extracted from markdown code block")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"[FAIL] Test 1: {e}")

    # Test case 2: JSON without markdown
    plain_json = '''
    {"setup": "Why did the chicken cross the playground?", "punchline": "To get to the other slide!", "rating": 7}
    '''

    try:
        result = extract_json_from_markdown(plain_json)
        print("[PASS] Test 2: JSON extracted from plain text")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"[FAIL] Test 2: {e}")

    # Test case 3: JSON in markdown without language specifier
    markdown_no_lang = '''
    Here's your response:

    ```
    {
      "content": "Hello world!",
      "genre": "greeting",
      "mood": "friendly"
    }
    ```
    '''

    try:
        result = extract_json_from_markdown(markdown_no_lang)
        print("[PASS] Test 3: JSON extracted from markdown without language specifier")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"[FAIL] Test 3: {e}")

    # Test case 4: Invalid JSON (should fail)
    invalid_json = '''
    ```json
    {"invalid": json content}
    ```
    '''

    try:
        result = extract_json_from_markdown(invalid_json)
        print("[FAIL] Test 4: Should have raised an error for invalid JSON")
    except Exception as e:
        print("[PASS] Test 4: Correctly identified invalid JSON")

if __name__ == "__main__":
    print("Testing JSON extraction functionality...")
    print("=" * 50)
    test_json_extraction()
    print("=" * 50)
    print("Testing completed!")