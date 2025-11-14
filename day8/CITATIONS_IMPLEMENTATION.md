# RAG Citations Implementation

## Overview

The AI agent's RAG (Retrieval-Augmented Generation) system has been enhanced to always return citations and links to sources from the document database. This ensures transparency, traceability, and trust in AI-generated responses.

## Implementation Summary

### Key Changes Made

#### 1. **Enhanced Document Formatting** (`rag_manager.py`)

- **Modified `format_docs()`**: Now formats documents with citation markers `[Source 1]`, `[Source 2]`, etc.
- **Added `get_citation_mapping()`**: Generates mapping of citation numbers to document metadata
- **Added `enhance_response_with_citations()`**: Appends proper citation references to responses
- **Enhanced metadata enrichment**: Documents now include `source_path` for full file paths

```python
# Example formatted context:
[Source 1]: sample_documents.txt (chunk 0)
Artificial Intelligence (AI) is a branch of computer science that...

[Source 2]: sample_documents.txt (chunk 1)  
The future of AI includes more advanced general intelligence...
```

#### 2. **Strong Prompt Engineering** (`main.py`)

Updated RAG prompt with explicit citation instructions:

- **MUST cite directive**: Model is explicitly told it must cite sources
- **Format examples**: Shows correct `[1]`, `[2]` citation format
- **Consequences**: Model informed that uncited responses are incorrect

```
CRITICAL INSTRUCTION: You MUST cite your sources using [1], [2], [3], etc. 
reference markers whenever you use information from a specific document.

EXAMPLE of CORRECT citations:
- "AI is a branch of computer science [1] that aims to create intelligent machines [2]."
- "Key concepts include NLP [1] and computer vision [2]."
```

#### 3. **Response Enhancement** (`main.py`)

- **Document tracking**: Retrieved documents are tracked through the RAG pipeline
- **Automatic citation appending**: Responses enhanced with "Sources:" section
- **Reference mapping**: Citation numbers mapped back to actual document sources

## Usage

### Basic RAG Query with Citations

```bash
# Start the application
uv run python main.py

# Ask a RAG-enhanced question (prefix with use_rag)
use_rag What is artificial intelligence?
```

### Example Response with Citations

```
AI Assistant: Artificial Intelligence (AI) is a branch of computer science 
that aims to create intelligent machines that can perform tasks that typically 
require human intelligence [1]. Machine Learning (ML) is a subset of AI that 
focuses on algorithms that can learn from data [1].

Key concepts in AI include Natural Language Processing (NLP), Computer Vision, 
Robotics, and Expert Systems [1].

Modern applications include virtual assistants like Siri and Alexa, 
recommendation systems, self-driving cars, and medical diagnosis systems [1].

---
**Sources:**
[1] sample_documents.txt (chunk 0) - /home/.../day8/sample_documents.txt
[2] sample_documents.txt (chunk 1) - /home/.../day8/sample_documents.txt
```

### Citation Features

- **In-text citations**: `[1]`, `[2]`, etc. throughout the response
- **Sources section**: Complete reference list at the end
- **Document metadata**: Source filename, chunk ID, and full path
- **Missing citation detection**: Warning if response lacks citations

## Document Metadata Structure

Each document in the RAG index now includes enhanced metadata:

```json
{
  "metadata": {
    "source": "sample_documents.txt",
    "chunk_id": 0,
    "source_path": "/home/.../day8/sample_documents.txt"
  }
}
```

### Metadata Fields

- **`source`**: Filename of the source document
- **`chunk_id`**: Integer identifier for document chunk
- **`source_path`**: Full filesystem path to source document

## Testing

### Run Citation Tests

```bash
# Test all citation functionality
uv run python test_rag_citations.py

# Run demo showing citation features
uv run python citation_demo.py
```

### Test Coverage

- ✅ Citation formatting with `format_docs()`
- ✅ Metadata mapping with `get_citation_mapping()`
- ✅ Response enhancement with `enhance_response_with_citations()`
- ✅ Document retrieval and metadata enrichment
- ✅ Prompt engineering and instruction effectiveness
- ✅ RAG chain integration and document tracking

## Technical Architecture

### Citations Workflow

1. **Document Retrieval**: User query → RAG retriever fetches relevant documents
2. **Context Formatting**: Documents formatted with `[Source N]` markers
3. **Prompt Enhancement**: Citations instructions added to system prompt
4. **Response Generation**: LLM generates answer citing sources as `[N]`
5. **Citation Enhancement**: Response enhanced with full "Sources:" section
6. **Delivery**: User receives cited response with reference list

### Chain Modification

The RAG chain was modified to track documents:

```python
# Retrieve documents first
retrieved_docs = rag_manager.get_retriever()(cleaned_question)
context = format_docs(retrieved_docs)

# Use pre-formatted context in chain
temp_chain = (
    {"context": lambda x: context, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Enhance response with citations
enhanced_response = enhance_response_with_citations(ai_response, retrieved_docs)
```

## Edge Cases Handled

### No Documents Retrieved
- Response: "No relevant documents found that meet the relevance threshold."
- No citations added

### Documents Retrieved But Not Cited
- Warning added: "*Note: The response didn't explicitly cite specific sources.*"
- All sources still listed in references

### Single Document
- Proper citation `[1]` throughout response
- Single source in references

### Multiple Documents, Same Source
- Separate `[1]`, `[2]`, etc. citations
- Both chunks listed with different chunk IDs

## Benefits

1. **Transparency**: Users can verify information sources
2. **Traceability**: Every claim can be traced to source document
3. **Trust**: Enhanced credibility through source attribution
4. **Debuggability**: Easy to verify which documents influenced response
5. **Compliance**: Meets requirements for cited AI-generated content

## Limitations

- **Local files only**: Currently only supports local file paths, not URLs
- **Chunk-level citation**: Citations reference chunks, not exact lines
- **Model compliance**: Depends on model following citation instructions
- **Path availability**: `source_path` only available for local files

## Future Enhancements

Consider adding:

1. **URL support**: Allow indexing web URLs with proper citation
2. **Line-level precision**: Add line numbers for exact source location
3. **Citation validation**: Verify model actually cited all used sources
4. **Clickable links**: Format file paths as clickable links in terminal
5. **Source preview**: Allow users to view source document chunks
6. **Citation confidence**: Show which sources were most influential

## Integration Notes

- Works with existing RAG commands (`/rag search`, `/rag index`, etc.)
- Compatible with similarity threshold filtering
- Integrates with session management and persistent memory
- Works alongside MCP tools and other agent capabilities
- No breaking changes to existing functionality

## Verification

To verify the implementation works:

```bash
# Run tests
uv run python test_rag_citations.py

# Expected: All tests pass

# Run demo
uv run python citation_demo.py

# Expected: Shows enhanced metadata and example citations

# Manual test
uv run python main.py
# Then: use_rag What is artificial intelligence?

# Expected: Response with [1], [2] citations and Sources section
```
