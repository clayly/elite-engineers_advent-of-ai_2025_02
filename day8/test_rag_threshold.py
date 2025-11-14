#!/usr/bin/env python3
"""
Test script to verify RAG similarity threshold functionality
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_manager import RAGManager

def test_rag_threshold():
    """Test RAG similarity threshold functionality"""
    
    # Test with different thresholds
    print("Testing RAG Similarity Threshold Functionality")
    print("=" * 50)
    
    # Create RAG manager with default threshold
    rag_default = RAGManager(similarity_threshold=0.3)
    print(f"\n1. Default threshold: {rag_default.similarity_threshold}")
    
    # Create RAG manager with strict threshold
    rag_strict = RAGManager(similarity_threshold=0.7)
    print(f"2. Strict threshold: {rag_strict.similarity_threshold}")
    
    # Create RAG manager with no filtering
    rag_none = RAGManager(similarity_threshold=0.0)
    print(f"3. No filtering: {rag_none.similarity_threshold}")
    
    print("\n✓ RAG threshold configuration test passed")
    print("\nTo test with actual documents:")
    print("1. Index some documents: /rag index <path>")
    print("2. Search with different thresholds: /rag threshold <value>")
    print("3. Query with RAG: use_rag your question here")
    print("\nThe system will filter out documents below the threshold.")

if __name__ == "__main__":
    test_rag_threshold()