#!/usr/bin/env python3
"""
Microphone input using OpenAI Whisper for speech-to-text
Compatible with Python 3.11+
"""

import tempfile
import numpy as np
from typing import Optional, Tuple
import warnings
import os

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class MicrophoneInput:
    """Handle microphone input and speech-to-text using Whisper"""
    
    def __init__(self, model_size: str = "base", device: Optional[int] = None):
        """
        Initialize microphone input handler
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, turbo)
            device: Audio device index (None for default)
        """
        if not AUDIO_AVAILABLE:
            raise RuntimeError("Audio libraries not available. Run: uv pip install sounddevice soundfile")
        
        if not WHISPER_AVAILABLE:
            raise RuntimeError("OpenAI Whisper not available. Run: uv pip install openai-whisper")
        
        self.model_size = model_size
        self.device = device
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1  # Mono
        self.whisper_model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        try:
            print(f"Loading Whisper model '{self.model_size}'...")
            self.whisper_model = whisper.load_model(self.model_size)
            print("✓ Whisper model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def list_devices(self) -> None:
        """List available audio input devices"""
        if not AUDIO_AVAILABLE:
            print("Audio libraries not available")
            return
        
        print("\nAvailable audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (channels: {device['max_input_channels']}, sample rate: {device['default_samplerate']})")
    
    def record_audio(self, duration: float = 5.0) -> Tuple[np.ndarray, int]:
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not AUDIO_AVAILABLE:
            raise RuntimeError("Audio libraries not available")
        
        print(f"\nRecording for {duration} seconds... Speak now!")
        print("Press Ctrl+C to stop early")
        
        try:
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device,
                dtype=np.float32
            )
            
            # Wait for recording to complete
            sd.wait()
            
            print("✓ Recording completed")
            return audio_data, self.sample_rate
            
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
            raise
        except Exception as e:
            print(f"✗ Recording failed: {e}")
            raise
    
    def record_until_enter(self) -> Tuple[np.ndarray, int]:
        """
        Record audio from microphone until user presses Enter
        
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not AUDIO_AVAILABLE:
            raise RuntimeError("Audio libraries not available")
        
        print("\nRecording... Press Enter when done speaking!")
        
        audio_chunks = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            audio_chunks.append(indata.copy())
        
        try:
            # Start recording stream
            with sd.InputStream(
                callback=callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                device=self.device,
                dtype=np.float32
            ):
                # Wait for user to press Enter
                input()
            
            if audio_chunks:
                audio_data = np.concatenate(audio_chunks, axis=0)
                duration = len(audio_data) / self.sample_rate
                print(f"✓ Recording completed ({duration:.1f}s)")
                return audio_data, self.sample_rate
            else:
                raise RuntimeError("No audio recorded")
                
        except Exception as e:
            print(f"✗ Recording failed: {e}")
            raise
    
    def save_temp_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Save audio to temporary WAV file
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Path to temporary WAV file
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate, format='WAV')
            return temp_file.name
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, 
                        language: Optional[str] = None) -> str:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            language: Target language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Transcribed text
        """
        if not WHISPER_AVAILABLE or self.whisper_model is None:
            raise RuntimeError("Whisper model not loaded")
        
        # Save to temp file
        temp_file = self.save_temp_audio(audio_data, sample_rate)
        
        try:
            # Transcribe
            options = {}
            if language:
                options['language'] = language
            
            print("Transcribing audio...")
            result = self.whisper_model.transcribe(temp_file, **options)
            
            text = result.get("text", "").strip()
            
            # Show transcription info
            detected_lang = result.get("language", "unknown")
            print(f"Detected language: {detected_lang}")
            
            return text
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def record_and_transcribe(self, duration: float = 5.0, 
                             language: Optional[str] = None) -> str:
        """
        Record audio and transcribe it
        
        Args:
            duration: Recording duration in seconds
            language: Target language code
            
        Returns:
            Transcribed text
        """
        # Record audio
        audio_data, sample_rate = self.record_audio(duration)
        
        # Transcribe
        return self.transcribe_audio(audio_data, sample_rate, language)
    
    def record_and_transcribe_until_enter(self, language: Optional[str] = None) -> str:
        """
        Record audio until Enter is pressed and transcribe it
        
        Args:
            language: Target language code
            
        Returns:
            Transcribed text
        """
        # Record audio
        audio_data, sample_rate = self.record_until_enter()
        
        # Transcribe
        return self.transcribe_audio(audio_data, sample_rate, language)
        # Record audio
        audio_data, sample_rate = self.record_audio(duration)
        
        # Transcribe
        return self.transcribe_audio(audio_data, sample_rate, language)


def test_microphone():
    """Test microphone input functionality"""
    try:
        mic = MicrophoneInput()
        
        print("\n=== Microphone Input Test ===")
        mic.list_devices()
        
        # Record and transcribe
        print("\nTest: 5-second fixed recording")
        text = mic.record_and_transcribe(duration=5.0)
        print(f"Transcribed: {text}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed: uv pip install sounddevice soundfile numpy openai-whisper")


if __name__ == "__main__":
    test_microphone()