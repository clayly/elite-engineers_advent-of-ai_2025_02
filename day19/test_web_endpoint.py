#!/usr/bin/env python3
"""Test the web server endpoint directly."""

import sys
sys.path.insert(0, '.')

from src.web_server import create_app
from fastapi.testclient import TestClient

# Create app
app = create_app('test_final')

# Create test client
client = TestClient(app)

# Test GET request
print("=== Testing GET / ===")
response = client.get("/")
print(f"Status: {response.status_code}")
print(f"Has form: {'search-form' in response.text}")
print(f"Has Jinja2 syntax: {'{%' in response.text or '{{' in response.text}")

# Test POST request
print("\n=== Testing POST / ===")
response = client.post("/", data={"query": "AI development"})
print(f"Status: {response.status_code}")
print(f"Length: {len(response.text)}")
print(f"Has Jinja2 syntax: {'{%' in response.text or '{{' in response.text}")
print(f"Has Result #1: {'Result #1' in response.text}")
print(f"Has AI content: {'Artificial Intelligence' in response.text}")

if '{%' in response.text:
    print("\n❌ Jinja2 syntax found in response!")
    # Show snippet
    idx = response.text.find('{%')
    print("Snippet:")
    print(response.text[idx-100:idx+200])
elif 'Result #1' in response.text:
    print("\n✅ Results rendered correctly!")
    # Show snippet
    idx = response.text.find('Result #1')
    print("Snippet:")
    print(response.text[idx:idx+400])
else:
    print("\n❌ No results and no Jinja2 syntax - empty?")