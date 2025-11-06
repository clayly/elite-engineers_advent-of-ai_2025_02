#!/usr/bin/env python3
"""
Test script for persistent memory functionality
"""

import subprocess
import time
import os
import sys

def run_test():
    """Test persistent memory by sending two separate conversations"""

    print("🧪 Testing Persistent Memory Functionality")
    print("=" * 50)

    # First conversation - introduce information
    print("\n1️⃣ Starting first conversation...")
    try:
        process1 = subprocess.run([
            'uv', 'run', 'python', 'main.py'
        ], input='Hello, my name is Alice and I love pizza. Please remember this.\nexit\n',
        text=True, capture_output=True, timeout=30)

        if process1.returncode != 0:
            print(f"❌ First conversation failed: {process1.stderr}")
            return False

        print("✅ First conversation completed")

    except subprocess.TimeoutExpired:
        print("⏰ First conversation timed out")
        return False
    except Exception as e:
        print(f"❌ First conversation error: {e}")
        return False

    # Wait a moment
    time.sleep(2)

    # Second conversation - test memory recall
    print("\n2️⃣ Starting second conversation (should remember previous info)...")
    try:
        process2 = subprocess.run([
            'uv', 'run', 'python', 'main.py'
        ], input='What is my name and what do I love?\nexit\n',
        text=True, capture_output=True, timeout=30)

        if process2.returncode != 0:
            print(f"❌ Second conversation failed: {process2.stderr}")
            return False

        print("✅ Second conversation completed")

        # Check if memory recall was successful
        output = process2.stdout.lower()
        if 'alice' in output and 'pizza' in output:
            print("🎉 SUCCESS: Memory persistence is working!")
            print(f"   AI correctly remembered: name=Alice, loves=pizza")
            return True
        else:
            print("❌ Memory persistence failed - AI didn't remember previous conversation")
            print(f"   Output: {process2.stdout}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Second conversation timed out")
        return False
    except Exception as e:
        print(f"❌ Second conversation error: {e}")
        return False

if __name__ == "__main__":
    success = run_test()

    if success:
        print("\n🎯 All tests passed! Persistent memory is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed. Check the implementation.")
        sys.exit(1)