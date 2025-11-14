#!/usr/bin/env python3
"""
RAG Manager for integrating day19 vector store with LangChain 1.0.5
Provides RAG functionality to the day8 AI agent.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

# Add day19 to path to import their modules
day19_path = Path(__file__).parent.parent / "day19" / "src"
if day19_path.exists():
    sys.path.insert(0, str(day19_path))
    try:
        from document_processor import DocumentProcessor
        from embedding_generator import EmbeddingGenerator, VectorStore
        DAY19_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"Day19 modules not available: {e}")
        DAY19_AVAILABLE = False
else:
    DAY19_AVAILABLE = False

# LangChain imports
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from typing import List

# Rich for console output
from rich.console import Console

console = Console()

logger = logging.getLogger(__name__)


def create_retriever_function(vector_store: 'VectorStore', similarity_threshold: float = 0.3):
    """Create a LangChain-compatible retriever function with similarity threshold"""
    
    def retriever_function(query: str) -> List[Document]:
        """Retriever function compatible with LangChain"""
        try:
            results = vector_store.similarity_search(query, k=5)
            # Filter results by similarity threshold
            filtered_results = [
                doc for doc, similarity in results 
                if similarity >= similarity_threshold
            ]
            logger.info(f"RAG retrieval: {len(results)} results found, {len(filtered_results)} passed threshold {similarity_threshold}")
            return filtered_results
        except Exception as e:
            logger.error(f"Error in retriever function: {e}")
            return []
    
    return retriever_function


class RAGManager:
    """RAG Manager that bridges day19 vector store with LangChain 1.0.5"""
    
    def __init__(self, index_path: str = "rag_index", similarity_threshold: float = 0.3):
        self.index_path = Path(index_path)
        self.similarity_threshold = similarity_threshold
        self.embedding_generator = None
        self.vector_store = None
        self.retriever = None
        self.ready = False
        
        if not DAY19_AVAILABLE:
            console.print("[yellow]⚠[/yellow] Day19 modules not available. RAG functionality disabled.")
            return
            
        self._initialize_if_available()
    
    def _initialize_if_available(self):
        """Initialize RAG components if index exists"""
        if self.is_index_ready():
            try:
                console.print("[blue]🔍 Initializing RAG system...[/blue]")
                
                # Use day19 components
                self.embedding_generator = EmbeddingGenerator()
                self.vector_store = VectorStore(
                    storage_path=str(self.index_path), 
                    embedding_generator=self.embedding_generator
                )
                self.vector_store.load_from_json()
                
                # Create LangChain-compatible retriever with similarity threshold
                self.retriever = create_retriever_function(self.vector_store, self.similarity_threshold)
                self.ready = True
                
                console.print(f"[green]✓[/green] RAG system ready with {len(self.vector_store.documents)} documents")
                
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to initialize RAG: {e}")
                logger.error(f"RAG initialization error: {e}")
    
    def is_index_ready(self) -> bool:
        """Check if RAG index is available"""
        vector_store_file = self.index_path / "vector_store.json"
        if not vector_store_file.exists():
            return False
            
        # Check if the file has content
        if vector_store_file.stat().st_size == 0:
            return False
            
        return True
    
    def get_retriever(self):
        """Get LangChain-compatible retriever"""
        return self.retriever if self.ready else None
    
    def index_documents(self, input_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> bool:
        """Index documents from a file or directory using day19 pipeline"""
        if not DAY19_AVAILABLE:
            console.print("[red]✗[/red] Cannot index: Day19 modules not available")
            return False
            
        try:
            console.print(f"[blue]📚 Indexing documents from {input_path}...[/blue]")
            
            # Initialize day19 components
            processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
            embedding_generator = EmbeddingGenerator(
                model_name=model_name, 
                model_type="sentence-transformers"
            )
            
            # Create vector store
            self.vector_store = VectorStore(
                storage_path=str(self.index_path), 
                embedding_generator=embedding_generator
            )
            
            # Process documents
            input_path_obj = Path(input_path)
            if input_path_obj.is_file():
                documents = processor.process_file(input_path)
            else:
                documents = processor.process_directory(input_path)
            
            if not documents:
                console.print("[yellow]⚠[/yellow] No documents found to process")
                return False
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            # Save the index
            self.vector_store.save_to_json()
            
            # Create retriever
            self.retriever = create_retriever_function(self.vector_store)
            self.embedding_generator = embedding_generator
            self.ready = True
            
            # Save metadata
            stats = {
                "total_documents": len(set(doc.metadata.get('source', 'unknown') for doc in documents)),
                "total_chunks": len(self.vector_store.documents),
                "model_info": {
                    "model_name": model_name,
                    "model_type": "sentence-transformers"
                },
                "processing_params": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                },
                "input_path": str(input_path_obj.resolve())
            }
            
            stats_path = self.index_path / "index_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]✓[/green] Successfully indexed {len(documents)} chunks from {stats['total_documents']} documents")
            return True
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to index documents: {e}")
            logger.error(f"Indexing error: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5, apply_threshold: bool = True) -> List[tuple[Document, float]]:
        """Search documents in the RAG index"""
        if not self.ready:
            console.print("[yellow]⚠[/yellow] RAG system not ready")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            if apply_threshold:
                # Filter by similarity threshold
                filtered_results = [
                    (doc, similarity) for doc, similarity in results 
                    if similarity >= self.similarity_threshold
                ]
                logger.info(f"RAG search: {len(results)} results found, {len(filtered_results)} passed threshold {self.similarity_threshold}")
                return filtered_results
            return results
        except Exception as e:
            console.print(f"[red]✗[/red] Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG index statistics"""
        if not self.ready:
            return {"status": "not_ready"}
        
        stats_path = self.index_path / "index_stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                stats["status"] = "ready"
                return stats
            except Exception as e:
                logger.error(f"Error reading stats: {e}")
        
        return {
            "status": "ready",
            "total_chunks": len(self.vector_store.documents) if self.vector_store else 0,
            "index_path": str(self.index_path)
        }
    
    def reset_index(self) -> bool:
        """Reset the RAG index"""
        try:
            if self.index_path.exists():
                import shutil
                shutil.rmtree(self.index_path)
                console.print(f"[green]✓[/green] RAG index removed: {self.index_path}")
            
            # Reset internal state
            self.vector_store = None
            self.embedding_generator = None
            self.retriever = None
            self.ready = False
            
            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to reset index: {e}")
            return False


def should_use_rag(question: str) -> tuple[bool, str]:
    """Check if RAG should be used and return cleaned question"""
    question_lower = question.lower()
    use_rag = "use_rag" in question_lower
    cleaned_question = question.replace("use_rag", "").replace("USE_RAG", "").strip()
    return use_rag, cleaned_question


def format_docs(docs):
    """Format documents for RAG context"""
    if not docs:
        return "No relevant documents found that meet the relevance threshold."
    
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', f'Document {i}')
        content = doc.page_content
        
        # Truncate very long content
        if len(content) > 800:
            content = content[:800] + "..."
        
        formatted_docs.append(f"Источник {i}: {source}\n{content}")
    
    return "\n\n".join(formatted_docs)