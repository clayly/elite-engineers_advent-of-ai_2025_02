#!/usr/bin/env python3
"""
Test script to verify RAG citation functionality
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_manager import (RAGManager, format_docs, get_citation_mapping, 
                        enhance_response_with_citations, should_use_rag)
from langchain_core.documents import Document
from rich.console import Console

console = Console()


def test_citation_formatting():
    """Test citation formatting and enhancement functions"""
    console.print("[bold blue]Testing Citation Formatting[/bold blue]")
    console.print("=" * 60)
    
    # Create test documents
    test_docs = [
        Document(
            page_content="Artificial Intelligence is a branch of computer science that creates intelligent machines.",
            metadata={"source": "ai_intro.txt", "chunk_id": 0, "source_path": "/docs/ai_intro.txt"}
        ),
        Document(
            page_content="Machine Learning is a subset of AI that focuses on algorithms that learn from data.",
            metadata={"source": "ml_basics.txt", "chunk_id": 1, "source_path": "/docs/ml_basics.txt"}
        )
    ]
    
    # Test format_docs
    console.print("\n[bold]1. Testing format_docs():[/bold]")
    formatted = format_docs(test_docs)
    console.print(formatted[:400] + "..." if len(formatted) > 400 else formatted)
    
    # Test get_citation_mapping
    console.print("\n[bold]2. Testing get_citation_mapping():[/bold]")
    mapping = get_citation_mapping(test_docs)
    console.print(mapping)
    
    # Test citation detection in responses
    test_responses = [
        "AI is important for technology [1] and ML is a subset [2].",
        "I found the answer. It's machine learning [2].",
        "Artificial Intelligence creates intelligent machines."  # No citations
    ]
    
    for i, response in enumerate(test_responses, 1):
        console.print(f"\n[bold]3-{i}. Testing response {i}:[/bold]")
        console.print(f"Original: {response}")
        enhanced = enhance_response_with_citations(response, test_docs)
        console.print(enhanced)
    
    console.print("\n[green]✓ Citation formatting tests completed[/green]")


def test_rag_integration_with_citations():
    """Test full RAG integration with citation tracking"""
    console.print("\n[bold blue]Testing RAG Integration with Citations[/bold blue]")
    console.print("=" * 60)
    
    # Initialize RAG manager
    rag_manager = RAGManager()
    
    # Check if ready, if not, index sample documents
    if not rag_manager.ready:
        console.print("[yellow]⚠ RAG not ready, indexing sample documents...[/yellow]")
        sample_path = Path("sample_documents.txt")
        if sample_path.exists():
            success = rag_manager.index_documents(str(sample_path))
            if not success:
                console.print("[red]✗ Failed to index documents[/red]")
                return False
        else:
            console.print("[red]✗ Sample documents not found[/red]")
            return False
    
    # Test document search and citation tracking
    test_query = "artificial intelligence"
    console.print(f"\n[bold]1. Searching for: '{test_query}'[/bold]")
    
    if rag_manager.get_retriever():
        try:
            retrieved_docs = rag_manager.get_retriever()(test_query)
            console.print(f"✓ Retrieved {len(retrieved_docs)} documents")
            
            # Show retrieved documents with metadata
            for i, doc in enumerate(retrieved_docs, 1):
                console.print(f"\n  [Source {i}] {doc.metadata.get('source', 'unknown')}")
                console.print(f"  Chunk ID: {doc.metadata.get('chunk_id', 'unknown')}")
                console.print(f"  Source path: {doc.metadata.get('source_path', 'N/A')}")
                console.print(f"  Content preview: {doc.page_content[:100]}...")
            
            # Test context formatting
            console.print("\n[bold]2. Testing context formatting:[/bold]")
            context = format_docs(retrieved_docs)
            console.print(context[:500] + "..." if len(context) > 500 else context)
            
            # Test citation enhancement
            console.print("\n[bold]3. Testing citation enhancement:[/bold]")
            sample_response = "AI is a branch of computer science [1] that creates intelligent machines."
            enhanced = enhance_response_with_citations(sample_response, retrieved_docs)
            console.print(enhanced)
            
        except Exception as e:
            console.print(f"[red]✗ Error during retrieval: {e}[/red]")
            return False
    else:
        console.print("[red]✗ No retriever available[/red]")
        return False
    
    console.print("\n[green]✓ RAG integration tests completed[/green]")
    return True


def test_citation_markers_in_prompt():
    """Test that the prompt includes citation instructions"""
    console.print("\n[bold blue]Testing Citation Instructions in Prompt[/bold blue]")
    console.print("=" * 60)
    
    # Check if the prompt template contains citation instructions
    main_file = Path("main.py")
    if main_file.exists():
        content = main_file.read_text()
        check_items = [
            ("MUST cite", "MUST cite your sources using"),
            ("[1], [2] format", "[1], [2], [3]"),
            ("EXAMPLE", "EXAMPLE of CORRECT citations")
        ]
        
        console.print("\nChecking for citation instructions in prompt:")
        all_found = True
        for check_name, check_string in check_items:
            if check_string in content:
                console.print(f"  [green]✓[/green] {check_name}")
            else:
                console.print(f"  [red]✗[/red] {check_name}")
                all_found = False
        
        if all_found:
            console.print("\n[green]✓ All citation instructions found in prompt[/green]")
            return True
        else:
            console.print("\n[yellow]⚠ Some citation instructions missing[/yellow]")
            return False
    else:
        console.print("[red]✗ main.py not found[/red]")
        return False


def test_document_metadata_enrichment():
    """Test that documents have proper metadata for citations"""
    console.print("\n[bold blue]Testing Document Metadata Enrichment[/bold blue]")
    console.print("=" * 60)
    
    # Initialize RAG manager and check sample document metadata
    rag_manager = RAGManager()
    
    if not rag_manager.ready:
        console.print("[yellow]⚠ RAG not ready, skipping metadata test[/yellow]")
        return False
    
    # Test search to get documents
    test_query = "artificial intelligence"
    if rag_manager.get_retriever():
        retrieved_docs = rag_manager.get_retriever()(test_query)
        
        if retrieved_docs:
            console.print("\nChecking document metadata:")
            required_fields = ['source', 'chunk_id']
            optional_fields = ['source_path']
            
            for i, doc in enumerate(retrieved_docs, 1):
                console.print(f"\nDocument {i}:")
                metadata = doc.metadata
                
                # Check required fields
                for field in required_fields:
                    if field in metadata:
                        console.print(f"  [green]✓[/green] {field}: {metadata[field]}")
                    else:
                        console.print(f"  [red]✗[/red] Missing {field}")
                
                # Check optional fields
                for field in optional_fields:
                    if field in metadata:
                        console.print(f"  [green]✓[/green] {field}: {metadata[field]}")
                    else:
                        console.print(f"  [yellow]⚠[/yellow] No {field} (optional)")
            
            console.print("\n[green]✓ Metadata enrichment test completed[/green]")
            return True
    
    console.print("[yellow]⚠ No documents available for metadata test[/yellow]")
    return False


def main():
    """Run all citation tests"""
    console.print("[bold green]RAG Citation Functionality Test Suite[/bold green]")
    console.print("=" * 70)
    
    all_tests_passed = True
    
    try:
        test_citation_formatting()
    except Exception as e:
        console.print(f"\n[red]✗ Citation formatting test failed: {e}[/red]")
        all_tests_passed = False
    
    try:
        success = test_rag_integration_with_citations()
        if not success:
            all_tests_passed = False
    except Exception as e:
        console.print(f"\n[red]✗ RAG integration test failed: {e}[/red]")
        all_tests_passed = False
    
    try:
        success = test_citation_markers_in_prompt()
        if not success:
            all_tests_passed = False
    except Exception as e:
        console.print(f"\n[red]✗ Prompt test failed: {e}[/red]")
        all_tests_passed = False
    
    try:
        success = test_document_metadata_enrichment()
        if not success:
            console.print("\n[yellow]⚠ Some metadata tests failed (non-critical)[/yellow]")
    except Exception as e:
        console.print(f"\n[red]✗ Metadata test failed: {e}[/red]")
    
    console.print("\n" + "=" * 70)
    if all_tests_passed:
        console.print("[bold green]🎉 All citation tests passed![/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("[dim]1. Run the main app: uv run python main.py[/dim]")
        console.print("[dim]2. Try: use_rag What is artificial intelligence?[/dim]")
        console.print("[dim]3. Look for citations like [1], [2] in responses[/dim]")
        console.print("[dim]4. Check the 'Sources:' section at the end[/dim]")
        return 0
    else:
        console.print("[bold red]❌ Some tests failed[/bold red]")
        console.print("\n[yellow]Manual verification needed:[/yellow]")
        console.print("[dim]1. Check if sample_documents.txt exists[/dim]")
        console.print("[dim]2. Run: /rag index sample_documents.txt[/dim]")
        console.print("[dim]3. Then test citations with: use_rag What is AI?[/dim]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
