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
- **MCP Server Support** - Register and manage Model Context Protocol (MCP) servers from JSON configuration

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

The application will start and prompt you for input.

### Chat Commands

- **Regular chat**: Type your messages and press Enter to send them to the AI
- **Exit**: Type 'exit' or 'quit' to end the conversation
- **MCP Management**: Use `/mcp` commands to manage MCP servers:
  - `/mcp list` - Show all registered MCP servers
  - `/mcp register <name> <command> [args...]` - Register a new MCP server
  - `/mcp enable <name>` - Enable a disabled MCP server
  - `/mcp disable <name>` - Disable an MCP server
  - `/mcp unregister <name>` - Remove an MCP server

- **RAG (Retrieval-Augmented Generation)**: Use `/rag` commands to work with document indexing and retrieval:
  - `/rag status` - Show RAG index status (indexed chunks, model info, similarity threshold)
  - `/rag index <path>` - Index documents from file or directory (documents will be chunked and stored in vector DB)
  - `/rag search <query>` - Search documents in RAG index and display results with similarity scores
  - `/rag reset` - Reset RAG index (permanent deletion, requires confirmation)
  - `/rag threshold <value>` - Set similarity threshold (0.0-1.0, higher = more strict relevance)
  - `/rag threshold show` - Show current similarity threshold setting

For each response, the application will display token usage information:
- Prompt tokens: Number of tokens in the input (including conversation history)
- Completion tokens: Number of tokens in the AI's response
- Total tokens: Sum of prompt and completion tokens

### MCP Server Configuration

The application automatically loads MCP server configurations from `mcp.json`. The configuration format follows the MCP standard:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "executable",
      "args": ["arg1", "arg2"],
      "env": {
        "ENV_VAR1": "value1",
        "ENV_VAR2": "value2"
      },
      "disabled": false
    }
  }
}
```

**Example MCP servers included:**
- **filesystem**: File system access via `@modelcontextprotocol/server-filesystem`
- **git**: Git repository operations via `@modelcontextprotocol/server-git`
- **web-search**: Web search via `@modelcontextprotocol/server-brave-search` (disabled by default)

## Project Structure

```
day8/
├── main.py          # Main application code with MCP integration
├── mcp.json         # MCP server configuration file
├── pyproject.toml   # Project configuration and dependencies
├── .env.example     # Environment variable template
└── README.md        # This file
```

## How It Works

The application uses LangChain to interface with z.ai's GLM API with MCP integration:

1. **LangChain Integration**: Uses ChatOpenAI wrapper configured for z.ai's API
2. **Conversation Memory**: Maintains conversation history for context
3. **Rich Display**: Uses Rich library for enhanced terminal output
4. **Environment Config**: Loads settings from .env file
5. **Token Tracking**: Uses LangChain's callback system to track token usage
6. **Automatic Summarization**: Uses LangChain's SummarizationMiddleware to handle long conversations
7. **MCP Server Management**: Loads and manages MCP servers from `mcp.json` configuration file
   - Automatic configuration loading on startup
   - Runtime server registration, enabling, disabling, and removal
   - Command-line interface for MCP management
   - JSON-based persistent configuration

### Automatic Conversation Summarization

The application implements LangChain's latest SummarizationMiddleware to automatically handle long conversations:

- When the conversation approaches the token limit (4000 tokens by default), the middleware automatically summarizes older messages
- The last 10 messages are preserved in their original form
- Older messages are replaced with a concise summary
- This process is transparent to the user experience
- Prevents token overflow errors while maintaining conversation context
- Displays a notification when summarization occurs, showing the summarized content

## Extending the Application

This is a foundation that can be extended with:
- More sophisticated conversation memory
- Function calling capabilities
- Multi-agent interactions
- Custom prompt templates
- Streaming responses
- Database integration for persistent history

## Dockerfile prompt

```text
Write fully self-contained Dockerfile with Python "hello-world" code -- all python code should be written inside Dockerfile. Add CMD clause, which will execute this code on "docker run". use write_file tool to write this Dockerfile with random name to directory /home/clayly/projects/zva-no-project/elite-engineers_advent-of-ai_2025_02/day8/sandbox/

use z_docker_run to build and run this Dockerfile
```
