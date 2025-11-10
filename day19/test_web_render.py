#!/usr/bin/env python3
"""Test web server template rendering."""

import sys
sys.path.insert(0, '.')

from src.embedding_generator import EmbeddingGenerator, VectorStore
from jinja2 import Template

# Replicate the exact template from web_server.py
TEMPLATE_STR = """<!DOCTYPE html>
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

# Test the template with real data
print("=== Testing Jinja2 Template ===")

# Load actual data
embedding_generator = EmbeddingGenerator()
vector_store = VectorStore(storage_path="test_final", embedding_generator=embedding_generator)
vector_store.load_from_json()

# Perform actual search
search_results = vector_store.similarity_search("AI development", k=5)

# Format results like the web server does
formatted_results = []
for doc, score in search_results:
    formatted_results.append({
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    })

print(f"Number of search results: {len(formatted_results)}")
for i, result in enumerate(formatted_results):
    print(f"  Result {i+1}: {result['content'][:50]}... (score: {result['score']:.4f})")

# Test template rendering
template = Template(TEMPLATE_STR)
html = template.render(
    total_docs=len(vector_store.documents),
    model_name=embedding_generator.model_name,
    query="AI development",
    results=formatted_results,
    error=None
)

print(f"\n=== Rendered HTML ===")
print(f"Length: {len(html)}")
print(f"Has Jinja2 syntax: {'{{' in html or '{%' in html}")
print(f"Has Result #1: {'Result #1' in html}")
print(f"Has AI content: {'Artificial Intelligence' in html}")
print(f"Has similarity: {'Similarity:' in html}")

# Show snippet
if 'Result #1' in html:
    idx = html.find('Result #1')
    print(f"\n--- Snippet around Result #1 ---")
    print(html[idx:idx+400])
else:
    print("\n❌ No Result #1 found in rendered HTML")
    print("First 500 chars of HTML:")
    print(html[:500])