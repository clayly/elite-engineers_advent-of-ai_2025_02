#!/usr/bin/env python3
"""
Quick verification script for RAG citations implementation
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_implementation():
    """Verify all citation components are in place"""
    print("RAG Citations Implementation Verification")
    print("=" * 50)
    
    checks = []
    
    # 1. Check rag_manager.py for citation functions
    print("\n1. Checking rag_manager.py...")
    rag_manager = Path("rag_manager.py")
    if rag_manager.exists():
        content = rag_manager.read_text()
        
        func_checks = [
            ("format_docs() updated", "[Source " in content and "chunk_id" in content),
            ("get_citation_mapping() added", "def get_citation_mapping" in content),
            ("enhance_response_with_citations() added", "def enhance_response_with_citations" in content),
            ("Metadata enrichment", "source_path" in content)
        ]
        
        for name, check in func_checks:
            status = "✓" if check else "✗"
            print(f"  {status} {name}")
            checks.append(check)
    else:
        print("  ✗ rag_manager.py not found")
        checks.append(False)
    
    # 2. Check main.py for prompt engineering
    print("\n2. Checking main.py...")
    main_file = Path("main.py")
    if main_file.exists():
        content = main_file.read_text()
        
        prompt_checks = [
            ("Citation instruction", "MUST cite your sources using" in content),
            ("Format examples", "EXAMPLE of CORRECT citations" in content),
            ("Document tracking", "retrieved_docs" in content),
            ("Response enhancement", "enhance_response_with_citations" in content)
        ]
        
        for name, check in prompt_checks:
            status = "✓" if check else "✗"
            print(f"  {status} {name}")
            checks.append(check)
    else:
        print("  ✗ main.py not found")
        checks.append(False)
    
    # 3. Check if test files exist
    print("\n3. Checking test files...")
    test_files = [
        "test_rag_citations.py",
        "citation_demo.py"
    ]
    
    for test_file in test_files:
        exists = Path(test_file).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {test_file}")
        checks.append(exists)
    
    # 4. Check documentation
    print("\n4. Checking documentation...")
    docs = ["CITATIONS_IMPLEMENTATION.md"]
    
    for doc in docs:
        exists = Path(doc).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {doc}")
        checks.append(exists)
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print("🎉 All implementation checks passed!")
        print("\nThe RAG citations system is ready to use.")
        print("\nTo test manually:")
        print("  uv run python main.py")
        print("  Then type: use_rag What is artificial intelligence?")
        print("\nYou should see:")
        print("  - In-text citations like [1], [2]")
        print("  - Sources section with document references")
        return 0
    else:
        print(f"❌ {total - passed} of {total} checks failed")
        print("\nReview the implementation or run tests:")
        print("  uv run python test_rag_citations.py")
        return 1


if __name__ == "__main__":
    sys.exit(verify_implementation())
