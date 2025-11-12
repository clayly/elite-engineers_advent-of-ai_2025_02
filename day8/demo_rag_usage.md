# RAG Integration Demo - Usage Guide

## 🚀 Getting Started

The RAG integration is now complete and ready to use! Here's how to use it:

### 1. Start the Application
```bash
uv run python main.py
```

### 2. Check RAG Status
When you start the app, you should see:
```
✓ RAG system ready with 2 chunks indexed
```

### 3. Available RAG Commands

#### `/rag status`
Check if RAG system is ready and see statistics:
```
/rag status
```

#### `/rag search <query>`
Search documents directly:
```
/rag search artificial intelligence
/rag search machine learning
```

#### `/rag index <path>`
Index new documents:
```
/rag index ./documents
/rag index sample_documents.txt
```

#### `/rag reset`
Reset the RAG index (with confirmation):
```
/rag reset
```

### 4. Using RAG in Conversation

Simply prepend **`use_rag`** to any question to get document-enhanced answers:

```
use_rag What is artificial intelligence?
use_rag Tell me about machine learning applications
use_rag How do neural networks work?
```

### 5. Examples to Try

1. **Normal conversation** (with MCP tools):
   ```
   What files are in the current directory?
   ```

2. **RAG-enhanced conversation**:
   ```
   use_rag What are the key concepts in artificial intelligence?
   ```

3. **Compare responses**:
   ```
   What is machine learning?
   use_rag What is machine learning?
   ```

### 6. Sample Session

```
💬 AI Chat with LangChain + MCP Support
✓ RAG system ready with 2 chunks indexed

You: /rag status
RAG System Status:
  • Status: ready
  • Total chunks: 2
  • Model: sentence-transformers/all-MiniLM-L6-v2

You: use_rag What is artificial intelligence?
🔍 Using RAG (searching documents)...
AI Assistant: Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines...

You: What is the weather today?  
🧠 Using general knowledge...
AI Assistant: I don't have access to current weather information, but I can help you check using available tools...
```

## ✅ Features Working

- ✅ **RAG trigger detection** - automatically detects "use_rag"
- ✅ **Document indexing** - processes files and creates searchable index  
- ✅ **Similarity search** - finds relevant document chunks
- ✅ **Persistent memory** - saves RAG-enhanced conversations
- ✅ **MCP tool integration** - works alongside existing tools
- ✅ **Session management** - RAG works across different sessions
- ✅ **Error handling** - graceful fallback when RAG unavailable

## 🔧 Architecture

- **Modern LangChain 1.0.5** with LCEL syntax
- **Day19 vector store** integration for document processing
- **RunnablePassthrough** chains for clean data flow
- **Backward compatible** with existing day8 features

The integration maintains all existing functionality while adding powerful RAG capabilities!