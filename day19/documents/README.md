# Document Indexer

This is a document indexing system that can process various document types
and create searchable indexes using embeddings.

## Features

- Support for multiple document formats (PDF, TXT, MD, HTML)
- Text chunking with configurable size and overlap
- Multiple embedding models (OpenAI, HuggingFace, Sentence Transformers)
- Local vector storage with similarity search
- CLI interface and web server

## Usage

Basic indexing:
```bash
python -m src.main index documents/
```

Search documents:
```bash
python -m src.main search "your query" -i index/
```
