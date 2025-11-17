# Voice Input Implementation Guide

## Overview

This document describes the speech-to-text functionality added to the AI chat application using OpenAI Whisper and microphone input.

## Features

- **Voice-to-Text Input**: Record audio from your microphone and transcribe it using Whisper
- **Multiple Whisper Models**: Support for various Whisper model sizes (tiny, base, small, medium, large, turbo)
- **Voice Commands**: Built-in voice input management commands
- **Fallback Support**: Seamlessly switches between voice and text input

## Installation

The required packages are already included in `pyproject.toml`:

```bash
uv pip install -e .
```

This will install:
- `openai-whisper>=20250625` - OpenAI's speech recognition model
- `sounddevice>=0.5.1` - Audio recording library
- `soundfile>=0.13.1` - Audio file handling
- `numpy>=1.24.0` - Numerical operations (already installed)

### Additional System Requirements

Whisper requires `ffmpeg` to be installed on your system:

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```bash
choco install ffmpeg
```
Or download from: https://ffmpeg.org/download.html

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Voice Input Configuration
VOICE_MODE_ENABLED=false           # Set to 'true' to enable voice at startup
WHISPER_MODEL_SIZE=base            # Model size: tiny, base, small, medium, large, turbo
```

### Model Selection

Choose the Whisper model based on your needs:

| Model | Size | Speed | Accuracy | Memory |
|-------|------|-------|----------|--------|
| tiny | 39M | Fastest | Basic | ~1GB |
| base | 74M | Fast | Good | ~1GB |
| small | 244M | Medium | Better | ~2GB |
| medium | 769M | Slow | Very Good | ~5GB |
| large | 1550M | Slowest | Excellent | ~10GB |
| turbo | 809M | Fast | Very Good | ~6GB |

**Recommendation**: Use `base` for most cases (good balance of speed and accuracy). Use `turbo` for faster processing with good accuracy.

## Usage

### Starting the Application

```bash
# Install dependencies
uv pip install -e .

# Run with voice support
uv run python main.py
```

### Voice Commands

Once in the chat application, use these commands:

#### Record Voice
- `/voice` or `/voice record` - Record audio (press Enter when done) and transcribe it to text

#### Device Management
- `/voice devices` - List available microphone devices

#### Quick Start Example

```bash
$ uv run python main.py

# In the chat interface:
You: /voice
Recording... Press Enter when done speaking!
(Speak your message)
(Press Enter)
✓ Recording completed [Detected language: en]
✓ Transcribed: "Hello, can you help me with Python programming?"

AI Assistant: Hello! I'd be happy to help you with Python programming...
```

**Recording:** Recording continues until you press Enter. Speak naturally, then press Enter to stop recording and start transcription.

### Direct Text Input

Text input is always available. Voice input is only used when you explicitly use the `/voice` or `/voice record` command.

## Implementation Details

### Architecture

The voice input system consists of:

1. **MicrophoneInput Class** (`microphone_input.py`)
   - Handles audio recording via sounddevice
   - Manages Whisper model loading and transcription
   - Provides device enumeration and testing

2. **Voice Command Handler** (`main_voice.py`)
   - Processes `/voice` commands
   - Integrates transcription into chat flow
   - Manages voice mode state

3. **Input Processing**
   - Maintains both text and voice input paths
   - Auto-detects language during transcription
   - Cleans up temporary audio files automatically

### Whisper Integration

The implementation uses OpenAI's Whisper library:

```python
import whisper

# Load model
model = whisper.load_model("base")

# Transcribe audio
result = model.transcribe("audio.wav")
text = result["text"]
language = result["language"]
```

### Audio Processing Flow

1. Record audio at 16kHz (Whisper's expected sample rate)
2. Save to temporary WAV file
3. Pass to Whisper for transcription
4. Extract text and language
5. Clean up temporary file
6. Return transcribed text to chat

## Testing

### Test Microphone Only

```bash
uv run python microphone_input.py
```

This will:
1. List available audio devices
2. Record 5 seconds of audio
3. Transcribe using Whisper
4. Display the results

### Test Voice in Chat

```bash
uv run python main.py

# Then use:
/voice on
/voice test
```

## Troubleshooting

### Common Issues

**1. "Audio libraries not available"**
```bash
uv pip install sounddevice soundfile
```

**2. "OpenAI Whisper not available"**
```bash
uv pip install openai-whisper
```

**3. "ffmpeg not found"**
- Install ffmpeg system package (see Installation section)
- Or download from: https://ffmpeg.org/download.html

**4. No microphone detected**
```bash
# List devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Set default device
# In Python: sd.default.device = 2  # Use device index 2
```

**5. Poor transcription quality**
- Try a larger Whisper model (e.g., `small` instead of `base`)
- Check microphone quality and reduce background noise
- Ensure you're speaking clearly and at a reasonable volume

**6. ImportError: No module named 'audioop'**
- This module was removed in Python 3.13
- The code has been updated to not use audioop
- Update to the latest version of the microphone_input.py

### Debug Mode

Enable debug logging:

```python
# In microphone_input.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Configuration

### Custom Device Selection

```python
from microphone_input import MicrophoneInput

# Use specific device
mic = MicrophoneInput(device=2)  # Use device index 2

# Or set default device
import sounddevice as sd
sd.default.device = 2
```

### Language Specification

Force transcription to specific language:

```python
text = mic.record_and_transcribe(duration=5.0, language="en")
# or
# text = mic.record_and_transcribe(duration=5.0, language="es")  # Spanish
# text = mic.record_and_transcribe(duration=5.0, language="fr")  # French
```

### Programmatic Voice Input

```python
from microphone_input import MicrophoneInput

mic = MicrophoneInput(model_size="base")
text = mic.record_and_transcribe(duration=5.0)
print(f"You said: {text}")
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_MODE_ENABLED` | `false` | Enable voice at startup |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model to use |
| `WHISPER_LANGUAGE` | `None` | Force specific language (optional) |
| `MICROPHONE_DEVICE` | `None` | Specific device index (optional) |

## Performance Considerations

- **First Run**: Whisper downloads the model (~150MB for base model)
- **Memory Usage**: Base model uses ~1GB RAM during transcription
- **Speed**: base model processes 5 seconds of audio in ~1-2 seconds
- **CPU**: Uses CPU by default, GPU acceleration available with PyTorch CUDA

## Security Notes

- Temporary audio files are created in system temp directory and automatically deleted
- No audio data is sent to external services (Whisper runs locally)
- Microphone access is only activated during explicit recording commands

## Updates and Maintenance

To update Whisper to the latest version:

```bash
uv pip install --upgrade openai-whisper
```

Check current version:

```bash
python -c "import whisper; print(whisper.__version__)"
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all system dependencies are installed (especially ffmpeg)
3. Verify microphone permissions in your OS settings
4. Test with the standalone `microphone_input.py` script first