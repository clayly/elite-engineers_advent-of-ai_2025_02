#!/usr/bin/env python3
"""
Example usage of the document indexing pipeline.
"""

import os
from pathlib import Path
from src.document_processor import DocumentProcessor
from src.embedding_generator import EmbeddingGenerator, VectorStore

def main():
    # Example documents directory
    docs_dir = Path("documents")
    docs_dir.mkdir(exist_ok=True)
    
    # Create some example documents if they don't exist
    example_files = [
        ("README.md", """# Document Indexer

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
"""),
        
        ("ai_article.txt", """Artificial Intelligence in Modern Software Development

Artificial Intelligence has revolutionized the way we approach software development.
Machine learning models can now assist developers in writing code, debugging,
and optimizing performance.

Key areas where AI is making an impact:

1. Code Generation: AI-powered tools can generate code snippets based on natural
   language descriptions, significantly speeding up development time.

2. Bug Detection: Machine learning algorithms can identify potential bugs and
   security vulnerabilities before they reach production.

3. Automated Testing: AI can generate test cases and automatically test software
   for various scenarios and edge cases.

4. Performance Optimization: AI systems can analyze code performance and suggest
   optimizations for better efficiency.

The integration of AI in software development workflows continues to grow,
with new tools and capabilities emerging regularly.
"""),
        
        ("python_guide.txt", """Python Best Practices Guide

Writing clean, maintainable Python code requires following established best practices.

Naming Conventions:
- Use descriptive variable and function names
- Follow PEP 8 guidelines for naming
- Use underscores for function and variable names
- Use CapWords for class names

Code Organization:
- Keep functions small and focused on a single task
- Use modules to organize related functionality
- Follow the "import this" zen of Python principles
- Document your code with docstrings

Error Handling:
- Use specific exception types rather than broad except clauses
- Implement proper error logging
- Provide meaningful error messages
- Use context managers for resource management

Performance Considerations:
- Use list comprehensions for simple transformations
- Consider generator expressions for large datasets
- Profile your code to identify bottlenecks
- Use appropriate data structures for your use case
""")
    ]
    
    for filename, content in example_files:
        filepath = docs_dir / filename
        if not filepath.exists():
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Created example document: {filename}")
    
    print(f"\nDocuments available in: {docs_dir}")
    print("You can now run the indexing pipeline:")
    print(f"  uv run python -m src.main index {docs_dir}")
    print("\nOr use the CLI:")
    print(f"  doc-indexer index {docs_dir}")

if __name__ == "__main__":
    main()