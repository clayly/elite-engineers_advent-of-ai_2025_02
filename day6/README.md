# LangGraph Multi-Agent System

A simple multiagent system using LangGraph with logging, tracing, and token counting capabilities.

## Features

- **Two-agent system** with Agent One (analysis/extraction) and Agent Two (synthesis/reporting)
- **Comprehensive logging** with rich console output and structured logs
- **Token counting** for cost monitoring and usage tracking
- **Built on latest LangGraph 1.0+** architecture with modern Python patterns
- **Multiple execution modes**: sync, async, and streaming
- **State persistence** with thread management
- **Easy configuration** with environment variables
- **CLI interface** for convenient usage

## Requirements

- Python 3.11+
- uv package manager
- OpenAI API key

## Installation

1. Clone or navigate to the project directory
2. Install dependencies with uv:

```bash
uv sync
```

This will create a virtual environment and install all required dependencies including:
- LangGraph 1.0.1
- LangChain 1.0.2
- LangChain-OpenAI 1.0.0
- OpenAI 2.6.0
- TikToken 0.12.0

## Configuration

1. Copy the environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1/
MODEL_NAME=gpt-4
TEMPERATURE=0.7
MAX_TOKENS=4000
```

## Usage

### Quick Start

1. **Run the basic example:**

```bash
uv run python examples/basic_usage.py
```

2. **Use the CLI interface:**

```bash
# Basic usage
uv run python cli.py "Analyze this text for key insights"

# With streaming output
uv run python cli.py --stream "Process this with real-time feedback"

# Async processing
uv run python cli.py --async-mode "Async processing example"

# Specific task type
uv run python cli.py --task extract "Extract entities from this text"

# Custom model and settings
uv run python cli.py --model gpt-3.5-turbo --temperature 0.5 "Custom settings"
```

### Programmatic Usage

```python
from src.multiagent import MultiAgentGraph, AgentOneConfig, AgentTwoConfig

# Initialize the multiagent system
graph = MultiAgentGraph(
    agent_one_config=AgentOneConfig(model="gpt-4", temperature=0.7),
    agent_two_config=AgentTwoConfig(model="gpt-4", temperature=0.5)
)

# Process text
result = graph.invoke(
    input_data="Your text to process here",
    task="both"  # "analyze", "extract", or "both"
)

# Get results
if result.get("agent_two_results"):
    report = result["agent_two_results"]["final_report"]
    print(report)
```

### Advanced Usage

**Streaming execution:**
```python
for event in graph.stream(input_data="Your text", task="analyze"):
    print(f"Step: {event}")
```

**Async execution:**
```python
result = await graph.ainvoke(input_data="Your text", task="both")
```

**With persistence:**
```python
result = graph.invoke(
    input_data="Your text",
    task="both",
    thread_id="user-session-123"  # Maintains conversation state
)
```

## Architecture

### Agent One
- Analyzes input text for topics, entities, sentiment
- Extracts structured data and relationships
- Provides initial insights and preprocessing

### Agent Two
- Synthesizes results from Agent One
- Generates comprehensive reports and recommendations
- Provides executive summaries and action items

### Workflow
1. **START** → **Agent One** (analyze/extract)
2. **Agent One** → **Agent Two** (if ready for synthesis)
3. **Agent Two** → **END** (final report)
4. Error handling at each step

## Project Structure

```
.
├── src/
│   └── multiagent/
│       ├── __init__.py          # Package initialization
│       ├── agents.py            # Agent One and Agent Two implementations
│       ├── graph.py             # LangGraph workflow orchestration
│       ├── logging.py           # Enhanced logging utilities
│       └── token_counter.py     # Token usage and cost tracking
├── examples/
│   └── basic_usage.py           # Comprehensive usage examples
├── cli.py                       # Command-line interface
├── pyproject.toml               # Project configuration and dependencies
├── README.md                    # This file
├── .env.example                 # Environment variable template
└── .python-version              # Python version specification
```

## Monitoring

The system includes comprehensive logging and token tracking:

- **Rich console output** with colored logs and progress indicators
- **Token usage tracking** with cost estimation
- **Structured logging** for debugging and monitoring
- **Performance metrics** and execution time tracking

## Examples

The `examples/basic_usage.py` file demonstrates:
- Basic text processing workflow
- Streaming execution
- Async processing
- Performance testing
- Error handling

## License

MIT License - see LICENSE file for details.