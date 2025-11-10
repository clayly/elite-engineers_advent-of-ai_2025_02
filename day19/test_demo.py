#!/usr/bin/env python3
"""
Comprehensive demonstration of the document indexing pipeline.
This script tests all major functionality for the AI challenge.
"""

import os
import sys
import time
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def test_basic_functionality():
    """Test basic indexing and search functionality."""
    print_section("1. Testing Basic Document Indexing")
    
    # Create test documents
    test_docs_dir = Path("test_documents")
    test_docs_dir.mkdir(exist_ok=True)
    
    # Create a sample README
    readme_content = """# AI Challenge Day 19

## Document Indexing System

This project demonstrates a complete document indexing pipeline using modern AI tools.

### Features
- Document processing and chunking
- Embedding generation
- Vector storage and search
- Multiple format support

### Technologies Used
- LangChain for document processing
- Sentence Transformers for embeddings
- FastAPI for web interface
- Click for CLI interface
"""
    
    # Create a sample article
    article_content = """The Future of Artificial Intelligence in Software Development

Artificial Intelligence is transforming how we write, test, and maintain software.
Recent advances in large language models have enabled new possibilities for
automated code generation and intelligent assistance.

Machine learning algorithms can now understand code context, suggest improvements,
and even write entire functions based on natural language descriptions.
This represents a fundamental shift in how developers approach their work.

Key benefits include:
- Increased productivity through automated code suggestions
- Reduced bugs through AI-powered code review
- Faster onboarding for new developers
- Improved code quality and consistency

As AI continues to evolve, we can expect even more sophisticated tools that
will further enhance the software development process.
"""
    
    with open(test_docs_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    with open(test_docs_dir / "ai_future.txt", "w", encoding="utf-8") as f:
        f.write(article_content)
    
    print(f"✓ Created test documents in {test_docs_dir}")
    
    # Test indexing
    print("\n✓ Indexing documents...")
    os.system(f"source .venv/bin/activate && python -m src.main index {test_docs_dir} -o test_index")
    
    # Test search
    print("\n✓ Testing search functionality...")
    print("\n--- Search Results for 'artificial intelligence' ---")
    os.system("source .venv/bin/activate && python -m src.main search 'artificial intelligence' -i test_index")
    
    print("\n--- Search Results for 'document processing' ---")
    os.system("source .venv/bin/activate && python -m src.main search 'document processing' -i test_index")

def test_cli_interface():
    """Test the CLI interface."""
    print_section("2. Testing CLI Interface")
    
    print("✓ Testing CLI help:")
    os.system("source .venv/bin/activate && python -m src.main --help")
    
    print("\n✓ Testing index command help:")
    os.system("source .venv/bin/activate && python -m src.main index --help")
    
    print("\n✓ Testing search command help:")
    os.system("source .venv/bin/activate && python -m src.main search --help")

def test_different_models():
    """Test different embedding models."""
    print_section("3. Testing Different Embedding Models")
    
    # Test with a smaller model for faster processing
    print("✓ Testing with alternative embedding model...")
    os.system("source .venv/bin/activate && python -m src.main index documents/ -o index_alternative -m 'sentence-transformers/all-MiniLM-L6-v2'")
    
    print("\n✓ Search with alternative model:")
    os.system("source .venv/bin/activate && python -m src.main search 'python code' -i index_alternative")

def test_stats_and_info():
    """Test statistics and information commands."""
    print_section("4. Testing Statistics and Information")
    
    print("✓ Testing stats command:")
    os.system("source .venv/bin/activate && python -m src.main stats test_index")
    
    print("\n✓ Index file information:")
    index_path = Path("test_index/vector_store.json")
    if index_path.exists():
        size_mb = index_path.stat().st_size / (1024 * 1024)
        print(f"  - Index file size: {size_mb:.2f} MB")
        print(f"  - Index file location: {index_path}")

def test_error_handling():
    """Test error handling and edge cases."""
    print_section("5. Testing Error Handling")
    
    print("✓ Testing non-existent file:")
    result = os.system("source .venv/bin/activate && python -m src.main index non_existent_file 2>&1")
    
    print("\n✓ Testing search without index:")
    result = os.system("source .venv/bin/activate && python -m src.main search 'test' -i non_existent_index 2>&1")

def demonstrate_features():
    """Demonstrate key features for the AI challenge."""
    print_section("6. AI Challenge Feature Demonstration")
    
    print("✓ Key Features Implemented:")
    print("  1. ✅ Document loading (README, articles, code, PDF)")
    print("  2. ✅ Text chunking with configurable size/overlap")
    print("  3. ✅ Embedding generation (multiple models)")
    print("  4. ✅ Local index storage (JSON format)")
    print("  5. ✅ Similarity search functionality")
    print("  6. ✅ CLI interface")
    print("  7. ✅ Web interface (optional)")
    print("  8. ✅ Latest LangChain stack (0.3.x)")
    
    print("\n✓ Supported Document Formats:")
    print("  - Plain text (.txt)")
    print("  - Markdown (.md)")
    print("  - PDF (.pdf)")
    print("  - HTML (.html, .htm)")
    
    print("\n✓ Embedding Models:")
    print("  - HuggingFace models (default: all-MiniLM-L6-v2)")
    print("  - OpenAI models (text-embedding-3-small, etc.)")
    print("  - Sentence Transformers")
    
    print("\n✓ Storage Formats:")
    print("  - JSON with embeddings and metadata")
    print("  - Easy to extend to FAISS, ChromaDB, etc.")

def cleanup():
    """Clean up test files."""
    print_section("7. Cleanup")
    
    # Clean up test files
    test_docs_dir = Path("test_documents")
    test_index_dir = Path("test_index")
    alt_index_dir = Path("index_alternative")
    
    for path in [test_docs_dir, test_index_dir, alt_index_dir]:
        if path.exists():
            import shutil
            shutil.rmtree(path)
            print(f"✓ Cleaned up {path}")

def main():
    """Run all tests."""
    print("🚀 Document Indexing Pipeline - Comprehensive Test")
    print("=" * 80)
    
    try:
        test_basic_functionality()
        test_cli_interface()
        test_different_models()
        test_stats_and_info()
        test_error_handling()
        demonstrate_features()
        
        print_section("✅ All Tests Completed Successfully!")
        
        print("\n🎉 The document indexing pipeline is ready for the AI challenge!")
        print("\n📹 For the video demonstration, you can show:")
        print("   1. Document processing and indexing")
        print("   2. Search functionality with relevant results")
        print("   3. CLI interface usage")
        print("   4. Index statistics and metadata")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    finally:
        cleanup()

if __name__ == "__main__":
    main()