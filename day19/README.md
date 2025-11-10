# Document Indexing Pipeline

A powerful document indexing system built with Python, LangChain, and modern AI tools. This application implements a complete pipeline for processing documents, generating embeddings, and creating searchable indexes.

## 🎯 Features

- **Multi-format Document Support**: PDF, TXT, Markdown, HTML
- **Intelligent Text Processing**: Configurable chunking with overlap
- **Multiple Embedding Models**: OpenAI, HuggingFace, Sentence Transformers
- **Local Vector Storage**: JSON-based storage with similarity search
- **CLI Interface**: Easy-to-use command-line tools
- **Web Interface**: Browser-based search interface
- **Latest Stack**: Built with LangChain 1.0.5+, Sentence Transformers 5.1.2+, and the latest Python libraries (November 2025)

## 📋 Requirements

- Python 3.11+
- **uv package manager (required)** - we use only uv, no pip/requirements.txt
- For OpenAI embeddings: OpenAI API key

## 🚀 Installation

### Using uv (Required)

This project uses **only uv** for dependency management. No requirements.txt needed!

```bash
# Install dependencies from pyproject.toml
uv sync

# Or install in development mode
uv pip install -e .
```

### Why uv?

- **Single source of truth**: All dependencies in pyproject.toml
- **Fast**: Much faster than pip
- **Modern**: Uses latest Python packaging standards
- **No requirements.txt**: Cleaner project structure

See [UV_SETUP.md](UV_SETUP.md) for detailed uv instructions.

## 🔧 Configuration

### Environment Variables

```bash
# For OpenAI embeddings (optional)
export OPENAI_API_KEY="your-api-key-here"
```

## 📖 Usage

### Basic Indexing

Index documents from a directory:

```bash
# Using uv run (recommended)
uv run python -m src.main index documents/

# Or using the installed CLI
doc-indexer index documents/

# With custom settings
uv run python -m src.main index documents/ -o my_index -c 500 -l 50
```

### Searching Documents

Search the indexed documents:

```bash
# Basic search with uv run
uv run python -m src.main search "machine learning" -i index/

# Return top 10 results
uv run python -m src.main search "python best practices" -i index/ -k 10

# Using OpenAI embeddings
uv run python -m src.main search "AI development" -i index/ -t openai -m "text-embedding-3-small"
```

### Web Interface

Start a web server for browser-based searching:

```bash
uv run python -m src.main serve index/
```

Then open http://localhost:8000 in your browser.

### Index Statistics

View information about your index:

```bash
uv run python -m src.main stats index/
```

## 🏗️ Architecture

### Components

1. **DocumentProcessor** (`src/document_processor.py`)
   - Loads documents from various formats
   - Splits text into chunks with configurable overlap
   - Preserves metadata throughout processing

2. **EmbeddingGenerator** (`src/embedding_generator.py`)
   - Generates embeddings using multiple model types
   - Supports OpenAI, HuggingFace, and Sentence Transformers
   - Handles both document and query embeddings

3. **VectorStore** (`src/embedding_generator.py`)
   - Stores documents, embeddings, and metadata
   - Implements cosine similarity search
   - JSON-based persistence

4. **CLI Interface** (`src/main.py`)
   - Command-line tools for indexing and searching
   - Flexible configuration options
   - Web server for browser interface

### Processing Pipeline

```
Input Documents
    ↓
DocumentProcessor (Load & Chunk)
    ↓
EmbeddingGenerator (Create Embeddings)
    ↓
VectorStore (Store & Index)
    ↓
Searchable Index
```

## 🛠️ Advanced Usage

### Using Different Embedding Models

```bash
# OpenAI embeddings (requires API key)
doc-indexer index documents/ -t openai -m "text-embedding-3-small"

# Different HuggingFace model
doc-indexer index documents/ -m "sentence-transformers/all-mpnet-base-v2"

# Multi-lingual model
doc-indexer index documents/ -m "intfloat/multilingual-e5-large"
```

### Custom Chunking Parameters

```bash
# Small chunks for detailed search
doc-indexer index documents/ -c 200 -l 50

# Large chunks for context preservation
doc-indexer index documents/ -c 2000 -l 400
```

### Programmatic Usage

```python
from src.document_processor import DocumentProcessor
from src.embedding_generator import EmbeddingGenerator, VectorStore

# Initialize components
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
embeddings = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = VectorStore(storage_path="index", embedding_generator=embeddings)

# Process documents
documents = processor.process_directory("documents")
vector_store.add_documents(documents)

# Save index
vector_store.save_to_json()

# Search
results = vector_store.similarity_search("your query", k=5)
for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:200]}...")
```

## 📁 Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI interface
│   ├── document_processor.py   # Document loading and chunking
│   ├── embedding_generator.py  # Embeddings and vector storage
│   └── web_server.py          # Web interface
├── documents/                  # Example documents
├── index/                      # Generated indexes
├── data/                       # Additional data
├── pyproject.toml             # Project configuration
├── pyproject.toml            # Project configuration and dependencies
└── example.py                # Usage examples
```

## 🧪 Example Documents

Create example documents:

```bash
python example.py
```

This creates sample documents in the `documents/` directory that demonstrate various content types.

## 🔍 Search Examples

After indexing example documents:

```bash
# Search for AI-related content
doc-indexer search "artificial intelligence" -i index/

# Find Python best practices
doc-indexer search "python naming conventions" -i index/

# Search with high result count
doc-indexer search "development" -i index/ -k 10
```

## 🚀 Performance Tips

1. **Chunk Size**: Larger chunks preserve context but reduce granularity
2. **Chunk Overlap**: Overlap helps maintain context across boundaries
3. **Embedding Models**: 
   - `all-MiniLM-L6-v2`: Fast, good quality (default)
   - `all-mpnet-base-v2`: Better quality, slower
   - OpenAI models: Best quality, requires API key
4. **Storage**: JSON storage works well for small to medium indexes

## 🔧 Troubleshooting

### Common Issues

1. **Missing dependencies**: Install with `uv sync` or `uv pip install -e .`
2. **OpenAI API key**: Required for OpenAI embeddings
3. **Memory issues**: Reduce chunk size or use smaller embedding models
4. **File encoding**: Ensure documents are UTF-8 encoded

### Debug Mode

Enable verbose logging:

```bash
doc-indexer -v index documents/
```

## 📊 AI Challenge Task Completion

This project fulfills the requirements for "День 19. Индексация документов":

✅ **Document Set Processing**: Supports README, articles, code, and PDF documents  
✅ **Text Chunking Pipeline**: Configurable chunking with overlap preservation  
✅ **Embedding Generation**: Multiple model support with latest LangChain  
✅ **Index Storage**: JSON-based local storage with metadata  
✅ **Search Interface**: Both CLI and web interfaces  
✅ **Modern Stack**: Latest versions of LangChain and related libraries  

## 📄 License

MIT License - feel free to use and modify for your AI challenge.

## 🤝 Contributing

This is a demonstration project for the AI challenge. Feel free to extend it with additional features like:

- Additional vector stores (FAISS, ChromaDB)
- More document formats
- Advanced search features
- API endpoints
- Docker deployment