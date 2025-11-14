#!/usr/bin/env python3
"""
Demo script to test RAG citations with enhanced metadata
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_manager import RAGManager
from rich.console import Console

console = Console()


def demo_citation_functionality():
    """Demonstrate the citation functionality"""
    console.print("[bold green]RAG Citation Functionality Demo[/bold green]")
    console.print("=" * 70)
    
    # Test 1: Reset and re-index with enhanced metadata
    console.print("\n[bold blue]1. Testing document indexing with source_path metadata[/bold blue]")
    
    rag_manager = RAGManager()
    
    # Reset existing index to get fresh metadata
    console.print("\n[yellow]Reseting RAG index to apply new metadata enrichment...[/yellow]")
    rag_manager.reset_index()
    
    # Re-index sample documents
    sample_path = Path("sample_documents.txt")
    if sample_path.exists():
        console.print(f"[dim]Indexing: {sample_path.resolve()}[/dim]")
        success = rag_manager.index_documents(str(sample_path))
        
        if success:
            console.print("[green]✓ Documents re-indexed with enhanced metadata[/green]")
            
            # Test retrieval
            test_query = "artificial intelligence"
            if rag_manager.get_retriever():
                docs = rag_manager.get_retriever()(test_query)
                
                if docs:
                    console.print(f"\n[bold] Retrieved {len(docs)} documents:[/bold]")
                    for i, doc in enumerate(docs, 1):
                        console.print(f"\n  Document {i}:")
                        console.print(f"    Source: {doc.metadata.get('source', 'unknown')}")
                        console.print(f"    Chunk ID: {doc.metadata.get('chunk_id', 'unknown')}")
                        console.print(f"    Source path: {doc.metadata.get('source_path', 'N/A')}")
                        console.print(f"    Content: {doc.page_content[:100]}...")
                    
                    # Show formatted context
                    from rag_manager import format_docs
                    console.print("\n[bold]Formatted context for LLM:[/bold]")
                    formatted = format_docs(docs)
                    console.print(formatted[:800] + "..." if len(formatted) > 800 else formatted)
                    
                    return True
                else:
                    console.print("[red]✗ No documents retrieved[/red]")
                    return False
            else:
                console.print("[red]✗ Retriever not available[/red]")
                return False
        else:
            console.print("[red]✗ Failed to index documents[/red]")
            return False
    else:
        console.print("[red]✗ sample_documents.txt not found[/red]")
        return False


def demo_citation_usage():
    """Show how to use citations in practice"""
    console.print("\n[bold blue]2. Citation Usage Example[/bold blue]")
    console.print("\n[yellow]Example interaction:[/yellow]")
    
    example_interaction = """User: use_rag What is artificial intelligence?

AI Assistant: Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence [1]. Machine Learning (ML) is a subset of AI that focuses on algorithms that can learn from data [1].

Key concepts in AI include Natural Language Processing (NLP), Computer Vision, Robotics, and Expert Systems [1].

Modern applications include virtual assistants like Siri and Alexa, recommendation systems, self-driving cars, and medical diagnosis systems [1].

---
**Sources:**
[1] sample_documents.txt (chunk 0) - /home/.../day8/sample_documents.txt
[2] sample_documents.txt (chunk 1) - /home/.../day8/sample_documents.txt
"""
    console.print(example_interaction)
    
    console.print("\n[bold]Citation Features:[/bold]")
    console.print("  • In-text citations like [1], [2] throughout the response")
    console.print("  • 'Sources:' section listing all referenced documents")
    console.print("  • Full paths to source files when available")
    console.print("  • Chunk information to identify specific document sections")


def demo_enhanced_metadata():
    """Show the enhanced metadata structure"""
    console.print("\n[bold blue]3. Enhanced Document Metadata[/bold blue]")
    
    metadata_example = {
        "source": "sample_documents.txt",
        "chunk_id": 0,
        "source_path": "/home/.../day8/sample_documents.txt",
        "other_metadata": "..."
    }
    
    console.print("\nDocument metadata structure:")
    for key, value in metadata_example.items():
        console.print(f"  {key}: {value}")
    
    console.print("\n[dim]This metadata enables proper citation tracking![/dim]")


def main():
    """Run the citation demo"""
    success = demo_citation_functionality()
    
    if success:
        demo_citation_usage()
        demo_enhanced_metadata()
        
        console.print("\n" + "=" * 70)
        console.print("[bold green]✓ Citation demo completed successfully![/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("[dim]1. Start the app: uv run python main.py[/dim]")
        console.print("[dim]2. Ask: use_rag What is AI?[/dim]")
        console.print("[dim]3. Look for [1], [2] citations in the response[/dim]")
        console.print("[dim]4. Check the Sources section at the bottom[/dim]")
    else:
        console.print("\n[bold red]✗ Demo failed[/bold red]")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
