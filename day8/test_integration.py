#!/usr/bin/env python3
"""
Test script to verify RAG integration works properly
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_manager import RAGManager, should_use_rag
from rich.console import Console

console = Console()

def test_rag_trigger_detection():
    """Test the RAG trigger detection function"""
    console.print("[bold blue]Testing RAG trigger detection...[/bold blue]")
    
    test_cases = [
        ("What is AI?", False, "What is AI?"),
        ("use_rag What is machine learning?", True, "What is machine learning?"),
        ("Tell me about USE_RAG and deep learning", True, "Tell me about and deep learning"),
        ("Can you explain neural networks?", False, "Can you explain neural networks?"),
        ("use_rag How do computers learn?", True, "How do computers learn?"),
    ]
    
    for question, expected_use_rag, expected_cleaned in test_cases:
        use_rag, cleaned = should_use_rag(question)
        status = "✓" if (use_rag == expected_use_rag and cleaned == expected_cleaned) else "✗"
        color = "green" if use_rag == expected_use_rag else "red"
        
        console.print(f"  {status} Input: '{question}'")
        console.print(f"    → RAG: {use_rag}, Cleaned: '{cleaned}'")
        if use_rag != expected_use_rag or cleaned != expected_cleaned:
            console.print(f"    [red]Expected: RAG={expected_use_rag}, Cleaned='{expected_cleaned}'[/red]")
    
    console.print()

def test_rag_manager_initialization():
    """Test RAG Manager initialization"""
    console.print("[bold blue]Testing RAG Manager initialization...[/bold blue]")
    
    try:
        rag_manager = RAGManager()
        
        if rag_manager.ready:
            console.print("  [green]✓ RAG Manager initialized successfully[/green]")
            stats = rag_manager.get_stats()
            console.print(f"  • Status: {stats.get('status')}")
            console.print(f"  • Documents: {stats.get('total_chunks', 0)} chunks")
        else:
            console.print("  [yellow]⚠ RAG Manager not ready (no index found)[/yellow]")
            console.print("  • This is expected if no index has been created yet")
        
        console.print()
        return rag_manager
        
    except Exception as e:
        console.print(f"  [red]✗ RAG Manager initialization failed: {e}[/red]")
        console.print()
        return None

def test_document_indexing():
    """Test document indexing functionality"""
    console.print("[bold blue]Testing document indexing...[/bold blue]")
    
    rag_manager = RAGManager()
    
    # Check if sample document exists
    sample_doc = Path("sample_documents.txt")
    if not sample_doc.exists():
        console.print("  [yellow]⚠ Sample document not found, creating it...[/yellow]")
        return False
    
    try:
        success = rag_manager.index_documents(str(sample_doc))
        if success:
            console.print("  [green]✓ Document indexing successful[/green]")
            stats = rag_manager.get_stats()
            console.print(f"  • Indexed {stats.get('total_chunks', 0)} chunks")
            return True
        else:
            console.print("  [red]✗ Document indexing failed[/red]")
            return False
            
    except Exception as e:
        console.print(f"  [red]✗ Document indexing error: {e}[/red]")
        return False

def test_basic_search():
    """Test basic search functionality"""
    console.print("[bold blue]Testing basic search...[/bold blue]")
    
    rag_manager = RAGManager()
    
    if not rag_manager.ready:
        console.print("  [yellow]⚠ RAG not ready, skipping search test[/yellow]")
        return False
    
    try:
        test_queries = [
            "artificial intelligence",
            "machine learning applications", 
            "neural networks",
            "computer vision"
        ]
        
        for query in test_queries:
            results = rag_manager.search_documents(query, k=3)
            console.print(f"  Query: '{query}' → {len(results)} results")
            
            for i, (doc, similarity) in enumerate(results[:2], 1):
                console.print(f"    {i}. Similarity: {similarity:.4f}")
                console.print(f"       Content preview: {doc.page_content[:100]}...")
        
        console.print("  [green]✓ Search functionality working[/green]")
        return True
        
    except Exception as e:
        console.print(f"  [red]✗ Search error: {e}[/red]")
        return False

def main():
    """Run all tests"""
    console.print("[bold green]🧪 RAG Integration Test Suite[/bold green]")
    console.print("=" * 60)
    
    # Test 1: RAG trigger detection
    test_rag_trigger_detection()
    
    # Test 2: RAG Manager initialization
    rag_manager = test_rag_manager_initialization()
    
    # Test 3: Document indexing (if needed)
    if not rag_manager or not rag_manager.ready:
        console.print("[bold blue]Testing document indexing...[/bold blue]")
        indexing_success = test_document_indexing()
        if indexing_success:
            console.print("  [green]✓ Indexing completed, proceeding with search tests[/green]")
        else:
            console.print("  [yellow]⚠ Indexing failed, skipping search tests[/yellow]")
            return
    
    # Test 4: Basic search
    test_basic_search()
    
    console.print("[bold green]🎉 Test suite completed![/bold green]")
    console.print("\n[dim]You can now run the main application with:[/dim]")
    console.print("[dim]python main.py[/dim]")
    console.print("\n[dim]Try these commands:[/dim]")
    console.print("[dim]• '/rag status' - Check RAG status[/dim]")
    console.print("[dim]• '/rag search artificial intelligence' - Test search[/dim]")
    console.print("[dim]• 'use_rag What is machine learning?' - Test RAG in chat[/dim]")

if __name__ == "__main__":
    main()