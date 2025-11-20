## Using Ollama with the AI Chat Application

The application can be configured to use local Ollama models instead of the default z.ai API.

### Prerequisites

1. Install Ollama from https://ollama.ai/
2. Pull a model (e.g., `ollama pull llama3.1`)

### Configuration

To use Ollama instead of the z.ai API:

1. Copy `.env.example` to `.env`
2. Set `USE_OLLAMA=true`
3. Optionally configure the model name and base URL:
   ```
   USE_OLLAMA=true
   OLLAMA_MODEL_NAME=llama3.1
   OLLAMA_BASE_URL=http://localhost:11434
   ```

### Available Ollama Models

Some popular models that work well with this application:
- `llama3.1` (recommended)
- `mistral-nemo`
- `gemma2`
- `qwen2`

You can pull any of these models using:
```
ollama pull llama3.1
```