# UV-Only Setup Guide

This project uses **only uv** for dependency management and does not require requirements.txt.

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repository>
cd document-indexer

# Install dependencies (reads from pyproject.toml)
uv sync

# Or install in development mode
uv pip install -e .
```

## Running Commands

Use `uv run` to execute any Python command:

```bash
# Index documents
uv run python -m src.main index documents/ -o index/

# Search documents
uv run python -m src.main search "query" -i index/

# Run tests
uv run python test_demo.py

# Start web server
uv run python -m src.main serve index/
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run linting
uv run black src/
uv run flake8 src/

# Run tests
uv run pytest
```

## Project Structure

- `pyproject.toml`: All dependencies and project configuration
- No `requirements.txt`: Everything is in pyproject.toml
- Uses latest versions: LangChain 0.3.x, Sentence Transformers 5.x, etc.

## Benefits

- **Single source of truth**: pyproject.toml manages everything
- **Fast installs**: uv is much faster than pip
- **Lock file**: uv.lock ensures reproducible builds
- **Modern**: Uses latest Python packaging standards