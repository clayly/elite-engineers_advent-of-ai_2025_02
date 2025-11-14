# RAG Similarity Threshold Implementation

## Overview
Added a cutoff threshold for irrelevant results when RAG (Retrieval-Augmented Generation) is used in the AI agent. This prevents low-relevance documents from being included in the context provided to the LLM.

## Changes Made

### 1. rag_manager.py
- Added `similarity_threshold` parameter to `RAGManager.__init__()` (default: 0.3)
- Modified `create_retriever_function()` to filter results based on similarity score
- Updated `search_documents()` method to support threshold filtering
- Enhanced `format_docs()` to provide better feedback when no documents pass the threshold

### 2. main.py
- Added environment variable loading: `RAG_SIMILARITY_THRESHOLD` (default: 0.3)
- Updated RAG manager initialization with threshold parameter
- Added new commands:
  - `/rag threshold <value>` - Set similarity threshold (0.0-1.0)
  - `/rag threshold show` - Show current threshold
- Updated RAG status display to show current threshold
- Updated help text to include threshold commands

### 3. .env.example
- Added `RAG_SIMILARITY_THRESHOLD` configuration option with documentation

## How It Works

### Similarity Score Filtering
When RAG is triggered (using `use_rag` in your query), the system:

1. Retrieves the top-k most similar documents from the vector store
2. Filters out documents with similarity scores below the threshold
3. Only includes relevant documents in the context provided to the LLM

### Threshold Values
- **0.0**: No filtering (includes all retrieved documents)
- **0.1-0.3**: Lenient filtering (more results, potentially less relevant)
- **0.3-0.6**: Balanced (recommended for general use)
- **0.6-1.0**: Strict filtering (fewer results, highly relevant only)

### Default Behavior
- Default threshold: **0.3**
- This provides a good balance between recall and precision
- Can be adjusted based on your specific use case

## Usage

### Environment Variable
```bash
export RAG_SIMILARITY_THRESHOLD=0.5
uv run python main.py
```

### Runtime Commands
```
# Show current threshold
/rag threshold show

# Set new threshold
/rag threshold 0.7

# Check RAG status (includes threshold)
/rag status
```

### Query with RAG
```
use_rag What is machine learning?
```

## Benefits

1. **Improved Response Quality**: Prevents irrelevant documents from confusing the LLM
2. **Reduced Token Usage**: Fewer irrelevant documents = less token consumption
3. **Configurable**: Easy to adjust based on your document collection and use case
4. **Transparent**: Clear feedback when documents are filtered out
5. **Dynamic**: Can be changed at runtime without restarting

## Testing

Run the verification script:
```bash
uv run python verify_threshold.py
```

Run the demo:
```bash
uv run python demo_rag_threshold.py
```

## Implementation Details

### Core Filtering Logic
```python
def retriever_function(query: str) -> List[Document]:
    results = vector_store.similarity_search(query, k=5)
    filtered_results = [
        doc for doc, similarity in results 
        if similarity >= similarity_threshold
    ]
    return filtered_results
```

### Dynamic Threshold Updates
When the threshold is changed via `/rag threshold <value>`, the retriever function is recreated with the new threshold, affecting all subsequent RAG queries.

## Recommendations

- **Start with default (0.3)**: Good balance for most use cases
- **Increase to 0.5-0.7**: When you need high-precision answers
- **Decrease to 0.1-0.2**: When you want comprehensive coverage
- **Monitor results**: Use `/rag search <query>` to see similarity scores
- **Adjust based on content**: Dense technical docs may need lower thresholds, while general content works well with higher thresholds