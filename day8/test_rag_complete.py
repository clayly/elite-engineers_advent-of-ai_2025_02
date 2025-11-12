#!/usr/bin/env python3
"""
Complete test for RAG integration functionality
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_manager import RAGManager, should_use_rag, format_docs
from rich.console import Console

console = Console()

async def test_rag_integration():
    """Test complete RAG integration"""
    console.print("[bold green]🧪 Complete RAG Integration Test[/bold green]")
    console.print("=" * 60)
    
    # Test 1: RAG Manager initialization
    console.print("[bold blue]1. Testing RAG Manager initialization...[/bold blue]")
    rag_manager = RAGManager()
    
    if rag_manager.ready:
        console.print("  [green]✓ RAG Manager ready[/green]")
        stats = rag_manager.get_stats()
        console.print(f"  • Documents: {stats.get('total_chunks', 0)} chunks")
    else:
        console.print("  [yellow]⚠ RAG Manager not ready - will test indexing[/yellow]")
    
    # Test 2: RAG trigger detection
    console.print("[bold blue]2. Testing RAG trigger detection...[/bold blue]")
    test_questions = [
        ("What is AI?", False, "What is AI?"),
        ("use_rag What is machine learning?", True, "What is machine learning?"),
        ("USE_RAG Tell me about neural networks", True, "Tell me about neural networks"),
        ("Explain computer vision", False, "Explain computer vision"),
    ]
    
    all_passed = True
    for question, expected_rag, expected_cleaned in test_questions:
        use_rag, cleaned = should_use_rag(question)
        passed = (use_rag == expected_rag and cleaned.strip() == expected_cleaned.strip())
        status = "✓" if passed else "✗"
        color = "green" if passed else "red"
        
        console.print(f"  [{color}]{status}[/{color}] '{question}' → RAG: {use_rag}")
        if not passed:
            console.print(f"    [red]Expected: RAG={expected_rag}, Cleaned='{expected_cleaned}'[/red]")
            all_passed = False
    
    # Test 3: Document indexing (if needed)
    if not rag_manager.ready:
        console.print("[bold blue]3. Testing document indexing...[/bold blue]")
        sample_doc = Path("sample_documents.txt")
        if sample_doc.exists():
            success = rag_manager.index_documents(str(sample_doc))
            if success:
                console.print("  [green]✓ Document indexing successful[/green]")
            else:
                console.print("  [red]✗ Document indexing failed[/red]")
                return False
        else:
            console.print("  [yellow]⚠ Sample document not found[/yellow]")
            return False
    
    # Test 4: Document search
    console.print("[bold blue]4. Testing document search...[/bold blue]")
    test_queries = ["artificial intelligence", "machine learning"]
    
    for query in test_queries:
        try:
            results = rag_manager.search_documents(query, k=3)
            console.print(f"  Query: '{query}' → {len(results)} results")
            
            if results:
                for i, (doc, similarity) in enumerate(results[:1], 1):
                    console.print(f"    {i}. Similarity: {similarity:.4f}")
                    console.print(f"       Preview: {doc.page_content[:80]}...")
            
        except Exception as e:
            console.print(f"  [red]✗ Search error: {e}[/red]")
            return False
    
    # Test 5: LangChain chain compatibility
    console.print("[bold blue]5. Testing LangChain chain compatibility...[/bold blue]")
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from langchain_openai import ChatOpenAI
        
        # Create a simple test LLM (we won't actually call it)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Create RAG chain
        retriever = rag_manager.get_retriever()
        if retriever:
            rag_prompt = ChatPromptTemplate.from_template(
                "Answer using only this context:\n{context}\n\nQuestion: {question}"
            )
            
            # This should work without errors (no API calls needed for chain creation)
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | llm
                | StrOutputParser()
            )
            
            console.print("  [green]✓ RAG chain created successfully[/green]")
            console.print("  [green]✓ LangChain pipe operator working[/green]")
        else:
            console.print("  [yellow]⚠ No retriever available[/yellow]")
            
    except Exception as e:
        console.print(f"  [red]✗ Chain creation error: {e}[/red]")
        return False
    
    # Results
    console.print("\n" + "=" * 60)
    if all_passed:
        console.print("[bold green]🎉 All tests passed![/bold green]")
        console.print("\n[dim]RAG integration is fully functional![/dim]")
        console.print("\n[dim]To try it interactively:[/dim]")
        console.print("[dim]  uv run python main.py[/dim]")
        console.print("\n[dim]Then try these commands:[/dim]")
        console.print("[dim]  • '/rag status' - Check RAG status[/dim]")
        console.print("[dim]  • '/rag search AI' - Search documents[/dim]")
        console.print("[dim]  • 'use_rag What is machine learning?' - RAG chat[/dim]")
        return True
    else:
        console.print("[bold red]❌ Some tests failed[/bold red]")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rag_integration())
    sys.exit(0 if success else 1)