#!/usr/bin/env python3
"""
Demo script showing RAG similarity threshold functionality
"""

import os
import sys
from pathlib import Path

# Mock the day19 imports for demonstration
class MockVectorStore:
    def similarity_search(self, query, k=5):
        # Mock results with different similarity scores
        from langchain_core.documents import Document
        return [
            (Document(page_content="Very relevant content about AI", metadata={"source": "ai_docs.txt"}), 0.85),
            (Document(page_content="Somewhat relevant content", metadata={"source": "tech_docs.txt"}), 0.55),
            (Document(page_content="Barely relevant content", metadata={"source": "random.txt"}), 0.25),
            (Document(page_content="Completely irrelevant", metadata={"source": "unrelated.txt"}), 0.05),
        ]

class MockEmbeddingGenerator:
    pass

# Mock the day19 modules
sys.modules['document_processor'] = type(sys)('document_processor')
sys.modules['embedding_generator'] = type(sys)('embedding_generator')
sys.modules['embedding_generator'].EmbeddingGenerator = MockEmbeddingGenerator
sys.modules['embedding_generator'].VectorStore = MockVectorStore

# Add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_manager import create_retriever_function

def demo_threshold_filtering():
    """Demonstrate how different thresholds filter results"""
    
    print("RAG Similarity Threshold Demo")
    print("=" * 50)
    
    # Create mock vector store
    vector_store = MockVectorStore()
    
    # Test different thresholds
    thresholds = [0.0, 0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        print("-" * 20)
        
        # Create retriever with this threshold
        retriever = create_retriever_function(vector_store, similarity_threshold=threshold)
        
        # Get results
        results = retriever("test query")
        
        print(f"Documents returned: {len(results)}")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.metadata.get('source', 'unknown')}")
    
    print("\n" + "=" * 50)
    print("Key Insights:")
    print("- Lower thresholds (0.0-0.3): More results, potentially less relevant")
    print("- Medium thresholds (0.3-0.6): Balanced relevance and coverage")
    print("- Higher thresholds (0.6-1.0): Fewer results, highly relevant only")
    print("\nRecommended: 0.3 for general use, 0.5-0.7 for precision")

if __name__ == "__main__":
    demo_threshold_filtering()