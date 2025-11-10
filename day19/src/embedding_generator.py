"""
Embedding generation module using various embedding models.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles generation of embeddings for documents using various models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model_type: str = "huggingface"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model
            model_type: Type of model ('openai', 'huggingface', 'sentence-transformers')
        """
        self.model_name = model_name
        self.model_type = model_type
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model based on model_type."""
        logger.info(f"Initializing embeddings with model: {self.model_name} ({self.model_type})")
        
        try:
            if self.model_type == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
                return OpenAIEmbeddings(model=self.model_name)
            
            elif self.model_type == "huggingface":
                return HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            elif self.model_type == "sentence-transformers":
                return SentenceTransformerEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'}
                )
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def generate_embeddings(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(documents)} documents")
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        return embeddings
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        logger.info(f"Generating embedding for query: {query[:50]}...")
        return self.embeddings.embed_query(query)


class VectorStore:
    """Vector store for storing and retrieving document embeddings."""
    
    def __init__(self, storage_path: Union[str, Path], embedding_generator: EmbeddingGenerator):
        """
        Initialize the vector store.
        
        Args:
            storage_path: Path to store the vector index
            embedding_generator: EmbeddingGenerator instance
        """
        self.storage_path = Path(storage_path)
        self.embedding_generator = embedding_generator
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict[str, Any]] = []
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(documents)
        
        # Store documents, embeddings, and metadata
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadata.extend([doc.metadata for doc in documents])
        
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def save_to_json(self, filename: str = "vector_store.json") -> None:
        """
        Save vector store to JSON file.
        
        Args:
            filename: Name of the JSON file
        """
        file_path = self.storage_path / filename
        logger.info(f"Saving vector store to {file_path}")
        
        data = {
            "documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in self.documents
            ],
            "embeddings": self.embeddings,
            "model_info": {
                "model_name": self.embedding_generator.model_name,
                "model_type": self.embedding_generator.model_type
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Vector store saved successfully")
    
    def load_from_json(self, filename: str = "vector_store.json") -> None:
        """
        Load vector store from JSON file.
        
        Args:
            filename: Name of the JSON file
        """
        file_path = self.storage_path / filename
        logger.info(f"Loading vector store from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Vector store file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct documents
        self.documents = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in data["documents"]
        ]
        
        self.embeddings = data["embeddings"]
        self.metadata = [doc["metadata"] for doc in data["documents"]]
        
        logger.info(f"Loaded {len(self.documents)} documents from vector store")
    
    def similarity_search(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if not self.documents:
            return []
        
        logger.info(f"Performing similarity search for: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Calculate similarities (cosine similarity)
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i, similarity in similarities[:k]:
            results.append((self.documents[i], similarity))
        
        logger.info(f"Found {len(results)} similar documents")
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))