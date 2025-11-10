"""
Web server for searching the document index via a web interface.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from .embedding_generator import EmbeddingGenerator, VectorStore

logger = logging.getLogger(__name__)


def create_app(index_path: str):
    """Create the FastAPI application."""
    app = FastAPI(title="Document Index Search")
    
    # Initialize components
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore(storage_path=index_path, embedding_generator=embedding_generator)
    
    # Load the index
    try:
        vector_store.load_from_json()
        logger.info(f"Loaded index with {len(vector_store.documents)} documents")
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        raise
    
    # DEBUG: Log current working directory and script location
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"Script location: {Path(__file__).parent}")
    
    # Create templates directory and file
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    logger.info(f"Templates directory: {templates_dir.absolute()}")
    
    template_path = templates_dir / "search.html"
    template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Index Search</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
            text-align: center;
        }
        .search-form {
            margin-bottom: 30px;
        }
        .search-input {
            width: 70%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-right: 10px;
        }
        .search-button {
            padding: 12px 24px;
            font-size: 16px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .search-button:hover {
            background-color: #0056b3;
        }
        .results-info {
            margin-bottom: 20px;
            color: #666;
        }
        .result {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 15px;
            background: #fafafa;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .result-title {
            font-weight: bold;
            color: #333;
        }
        .result-score {
            color: #007bff;
            font-weight: bold;
        }
        .result-meta {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        .result-meta span {
            margin-right: 15px;
        }
        .result-content {
            color: #444;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .no-results {
            text-align: center;
            color: #666;
            padding: 40px;
        }
        .stats {
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Document Index Search</h1>
        
        <div class="stats">
            <strong>Index Stats:</strong> {{ total_docs }} documents | 
            <strong>Model:</strong> {{ model_name }}
        </div>
        
        <form method="post" class="search-form">
            <input 
                type="text" 
                name="query" 
                class="search-input" 
                placeholder="Enter your search query..."
                value="{{ query or '' }}"
                required
            >
            <button type="submit" class="search-button">Search</button>
        </form>
        
        {% if error %}
            <div class="error">
                <strong>Error:</strong> {{ error }}
            </div>
        {% endif %}
        
        {% if query is not none %}
            <div class="results-info">
                <strong>Search results for:</strong> "{{ query }}" 
                ({{ results|length }} results found)
            </div>
            
            {% if results %}
                {% for result in results %}
                    <div class="result">
                        <div class="result-header">
                            <div class="result-title">Result #{{ loop.index }}</div>
                            <div class="result-score">Similarity: {{ "%.4f"|format(result.score) }}</div>
                        </div>
                        
                        {% if result.metadata %}
                            <div class="result-meta">
                                {% for key, value in result.metadata.items() %}
                                    <span><strong>{{ key }}:</strong> {{ value }}</span>
                                {% endfor %}
                            </div>
                        {% endif %}
                        
                        <div class="result-content">{{ result.content }}</div>
                    </div>
                {% endfor %}
            {% else %}
                <div class="no-results">
                    <p>No results found for your query.</p>
                </div>
            {% endif %}
        {% endif %}
    </div>
</body>
</html>"""
    
    # Write template file
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    # Log template file info
    logger.info(f"Template file created/updated: {template_path}")
    logger.info(f"Template file size: {template_path.stat().st_size} bytes")
    
    # Initialize Jinja2 templates
    templates = Jinja2Templates(directory=str(templates_dir))
    logger.info(f"Jinja2Templates initialized with directory: {templates_dir}")
    
    # Test template loading
    try:
        test_template = templates.env.get_template('search.html')
        logger.info(f"Template loaded successfully: {test_template}")
        
        # Test rendering
        test_html = test_template.render(
            request=None,
            total_docs=1,
            model_name="test",
            query="test",
            results=[{"content": "test", "score": 0.5, "metadata": {}}],
            error=None
        )
        logger.info(f"Test render successful, length: {len(test_html)}")
        if 'Result #1' in test_html:
            logger.info("✅ Test render contains Result #1")
        else:
            logger.error("❌ Test render missing Result #1")
    except Exception as e:
        logger.error(f"Template test failed: {e}")
    
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Home page with search form."""
        return templates.TemplateResponse(
            "search.html",
            {
                "request": request,
                "total_docs": len(vector_store.documents),
                "model_name": embedding_generator.model_name,
                "query": None,
                "results": None,
                "error": None
            }
        )
    
    @app.post("/", response_class=HTMLResponse)
    async def search(request: Request, query: str = Form(...)):
        """Handle search query."""
        try:
            # Perform search
            search_results = vector_store.similarity_search(query, k=10)
            
            # Format results
            formatted_results = []
            for doc, score in search_results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            
            logger.info(f"Search found {len(formatted_results)} results for query: {query}")
            
            response = templates.TemplateResponse(
                "search.html",
                {
                    "request": request,
                    "total_docs": len(vector_store.documents),
                    "model_name": embedding_generator.model_name,
                    "query": query,
                    "results": formatted_results,
                    "error": None
                }
            )
            
            # DEBUG: Check the rendered HTML
            rendered_html = response.body.decode('utf-8') if isinstance(response.body, bytes) else response.body
            if '{{' in rendered_html or '{%' in rendered_html:
                logger.error("❌ Jinja2 syntax found in response!")
                logger.error(f"First 500 chars: {rendered_html[:500]}")
            else:
                logger.info("✅ Response rendered successfully, no Jinja2 syntax")
                if 'Result #1' in rendered_html:
                    logger.info("✅ Result #1 found in response")
            
            return response
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return templates.TemplateResponse(
                "search.html",
                {
                    "request": request,
                    "total_docs": len(vector_store.documents),
                    "model_name": embedding_generator.model_name,
                    "query": query,
                    "results": None,
                    "error": str(e)
                }
            )
    
    return app


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python web_server.py <index_path>")
        sys.exit(1)
    
    index_path = sys.argv[1]
    app = create_app(index_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)