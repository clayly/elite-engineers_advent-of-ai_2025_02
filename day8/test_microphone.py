#!/usr/bin/env python3
"""
Test script for microphone input functionality
"""

from microphone_input import MicrophoneInput

def test_microphone():
    """Test microphone input functionality"""
    try:
        mic = MicrophoneInput()
        
        print("\n=== Microphone Input Test ===")
        mic.list_devices()
        
        # Record and transcribe
        print("\nTest: 5-second recording")
        text = mic.record_and_transcribe(duration=5.0)
        print(f"Transcribed: {text}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed: uv pip install sounddevice soundfile")

if __name__ == "__main__":
    test_microphone()