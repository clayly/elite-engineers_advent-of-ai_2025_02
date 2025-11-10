#!/usr/bin/env python3
"""Debug template rendering."""

import sys
sys.path.insert(0, '.')

from jinja2 import Template
from src.embedding_generator import EmbeddingGenerator, VectorStore

# Create template like in web_server
TEMPLATE_STR = """<!DOCTYPE html>
<html>
<body>
    <h1>{{ total_docs }} documents</h1>
    {% if query %}
        <p>Query: {{ query }}</p>
        <p>Results: {{ results|length }}</p>
        {% for result in results %}
            <div>Result: {{ result.content[:50] }}</div>
        {% endfor %}
    {% endif %}
</body>
</html>"""

# Test template rendering
template = Template(TEMPLATE_STR)

# Test data
test_data = {
    "total_docs": 5,
    "model_name": "test-model",
    "query": "AI test",
    "results": [{"content": "Artificial Intelligence is amazing", "score": 0.95, "metadata": {}}],
    "error": None
}

# Render
html = template.render(**test_data)
print("=== Rendered HTML ===")
print(html[:500])
print("\n=== Checking for Jinja2 syntax ===")
print("Has {{:", "{{" in html)
print("Has {%:", "{%" in html)
print("Has total_docs:", "5 documents" in html)
print("Has query:", "Query: AI test" in html)
print("Has result content:", "Artificial Intelligence" in html)