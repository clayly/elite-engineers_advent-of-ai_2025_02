# Day 8: Simple AI Chat with LangChain

A simple console-based AI chat application using LangChain with z.ai's GLM API.

## Features

- Interactive console chat interface
- Conversation history tracking
- Rich text formatting with Markdown support
- Environment-based configuration
- Error handling and user-friendly messages
- **Token usage tracking** - Displays input/output token counts for each response
- **Automatic conversation summarization** - Handles long conversations without token overflow

## Requirements

- Python 3.11+
- uv package manager
- z.ai API key

## Installation

1. Clone or navigate to the project directory
2. Create virtual environment and install dependencies with uv:

```bash
cd day8
uv venv
uv pip install -e .
```

This will create a virtual environment and install all required dependencies including:
- LangChain
- LangChain-OpenAI
- LangChain-Community
- Rich for terminal formatting
- python-dotenv for configuration

## Configuration

1. Copy the environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your z.ai API key:

```env
OPENAI_API_KEY=your_z_ai_api_key_here
OPENAI_BASE_URL=https://api.z.ai/api/coding/paas/v4/
MODEL_NAME=glm-4.6
TEMPERATURE=0.7
MAX_TOKENS=1000
```

## Usage

Run the chat application:

```bash
cd day8
.venv/bin/python main.py
```

The application will start and prompt you for input. Type your messages and press Enter to send them to the AI. Type 'exit' or 'quit' to end the conversation.

For each response, the application will display token usage information:
- Prompt tokens: Number of tokens in the input (including conversation history)
- Completion tokens: Number of tokens in the AI's response
- Total tokens: Sum of prompt and completion tokens

## Project Structure

```
day8/
├── main.py          # Main application code
├── pyproject.toml   # Project configuration and dependencies
├── .env.example     # Environment variable template
└── README.md        # This file
```

## How It Works

The application uses LangChain to interface with z.ai's GLM API:

1. **LangChain Integration**: Uses ChatOpenAI wrapper configured for z.ai's API
2. **Conversation Memory**: Maintains conversation history for context
3. **Rich Display**: Uses Rich library for enhanced terminal output
4. **Environment Config**: Loads settings from .env file
5. **Token Tracking**: Uses LangChain's callback system to track token usage
6. **Automatic Summarization**: Uses LangChain's SummarizationMiddleware to handle long conversations

### Automatic Conversation Summarization

The application implements LangChain's latest SummarizationMiddleware to automatically handle long conversations:

- When the conversation approaches the token limit (4000 tokens by default), the middleware automatically summarizes older messages
- The last 10 messages are preserved in their original form
- Older messages are replaced with a concise summary
- This process is transparent to the user experience
- Prevents token overflow errors while maintaining conversation context

## Extending the Application

This is a foundation that can be extended with:
- More sophisticated conversation memory
- Function calling capabilities
- Multi-agent interactions
- Custom prompt templates
- Streaming responses
- Database integration for persistent history

## License

MIT License