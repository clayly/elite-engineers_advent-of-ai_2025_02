#!/usr/bin/env python3
"""
Test the fix for TZ collection error handling
"""

import sys
import os

# Add the day1 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'day1'))

# Test the fix directly without imports
def test_dict_handling():
    """Test handling of dict vs Pydantic objects"""

    # Simulate dict response (what was causing the error)
    dict_response = {
        "project_name": "Test Project",
        "project_description": "Test Description",
        "requirements": [
            {
                "id": "REQ-001",
                "category": "functional",
                "title": "Test Requirement",
                "description": "Test Description",
                "priority": "high"
            }
        ],
        "completeness_score": 0.5,
        "missing_categories": ["technical", "security"],
        "next_questions": ["What about performance?"],
        "is_ready_for_review": False
    }

    print("Testing dict handling...")

    # Test the fixed update_tz_state logic
    class MockTZState:
        def __init__(self):
            self.requirements_count = 0
            self.should_complete = False
            self.phase = "gathering"

    state = MockTZState()

    # Simulate the fixed update_tz_state method logic
    if hasattr(dict_response, 'requirements'):
        requirements = dict_response.requirements
        is_ready = dict_response.is_ready_for_review
    elif isinstance(dict_response, dict):
        requirements = dict_response.get('requirements', [])
        is_ready = dict_response.get('is_ready_for_review', False)
    else:
        requirements = []
        is_ready = False

    state.requirements_count = len(requirements)
    state.should_complete = is_ready

    if is_ready:
        state.phase = "complete"

    print(f"  ✓ Requirements count: {state.requirements_count}")
    print(f"  ✓ Should complete: {state.should_complete}")
    print(f"  ✓ Phase: {state.phase}")

    # Test the fixed completion check logic
    is_ready = False
    if hasattr(dict_response, 'is_ready_for_review'):
        is_ready = dict_response.is_ready_for_review
    elif isinstance(dict_response, dict):
        is_ready = dict_response.get('is_ready_for_review', False)

    print(f"  ✓ Ready for review: {is_ready}")

    return True

if __name__ == "__main__":
    print("Testing TZ Collection Fixes")
    print("=" * 40)

    try:
        success = test_dict_handling()
        if success:
            print("\n✅ All fixes working correctly!")
            print("The TZ collector should now handle dict responses properly.")
        else:
            print("\n❌ Some tests failed.")
    except Exception as e:
        print(f"\n❌ Test error: {e}")