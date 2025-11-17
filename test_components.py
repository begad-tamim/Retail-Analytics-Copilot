"""
Test script to validate individual components without requiring Ollama.
"""
import sys

def test_sqlite_tool():
    """Test SQLite tool functionality."""
    print("=" * 60)
    print("Testing SQLite Tool")
    print("=" * 60)
    
    from agent.tools.sqlite_tool import get_sqlite_tool
    tool = get_sqlite_tool()
    
    # Test schema summary
    schema = tool.get_schema_summary()
    assert "Products" in schema
    assert "Orders" in schema
    assert "Customers" in schema
    print("✓ Schema summary retrieved successfully")
    
    # Test query execution
    result = tool.execute_sql("SELECT COUNT(*) as total FROM Products")
    assert result["error"] is None
    assert len(result["rows"]) == 1
    assert result["rows"][0]["total"] == 10
    print("✓ SQL query executed successfully")
    
    # Test error handling
    result = tool.execute_sql("SELECT * FROM NonExistentTable")
    assert result["error"] is not None
    print("✓ SQL error handling works")
    
    print("\n✓ All SQLite tool tests passed!\n")
    return True


def test_rag_retrieval():
    """Test RAG retrieval functionality."""
    print("=" * 60)
    print("Testing RAG Retrieval")
    print("=" * 60)
    
    from agent.rag.retrieval import get_rag_retriever
    retriever = get_rag_retriever()
    
    # Check chunks loaded
    assert len(retriever.chunks) > 0
    print(f"✓ Loaded {len(retriever.chunks)} chunks from documents")
    
    # Test retrieval
    chunks = retriever.retrieve("What are the KPI definitions?", k=4)
    assert len(chunks) <= 4
    assert all(hasattr(c, 'id') for c in chunks)
    assert all(hasattr(c, 'score') for c in chunks)
    print(f"✓ Retrieved {len(chunks)} relevant chunks")
    
    # Check citation format
    for chunk in chunks:
        assert "::" in chunk.id
    print("✓ Citation IDs have correct format")
    
    print("\n✓ All RAG retrieval tests passed!\n")
    return True


def test_data_files():
    """Test that all required data files exist."""
    print("=" * 60)
    print("Testing Data Files")
    print("=" * 60)
    
    import os
    
    required_files = [
        "data/northwind.sqlite",
        "docs/marketing_calendar.md",
        "docs/kpi_definitions.md",
        "docs/catalog.md",
        "docs/product_policy.md",
        "sample_questions_hybrid_eval.jsonl"
    ]
    
    for filepath in required_files:
        assert os.path.exists(filepath), f"Missing file: {filepath}"
        print(f"✓ Found {filepath}")
    
    print("\n✓ All required data files exist!\n")
    return True


def test_module_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    try:
        from agent.tools.sqlite_tool import SQLiteTool, get_sqlite_tool
        print("✓ Imported agent.tools.sqlite_tool")
        
        from agent.rag.retrieval import RAGRetriever, DocChunk, get_rag_retriever
        print("✓ Imported agent.rag.retrieval")
        
        from agent.dspy_signatures import (
            RouterSignature, NL2SQLSignature, SynthesizerSignature,
            Router, NL2SQL, Synthesizer
        )
        print("✓ Imported agent.dspy_signatures")
        
        from agent.graph_hybrid import AgentState, HybridAgent
        print("✓ Imported agent.graph_hybrid")
        
        print("\n✓ All modules imported successfully!\n")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RETAIL ANALYTICS COPILOT - COMPONENT TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        ("Data Files", test_data_files),
        ("Module Imports", test_module_imports),
        ("SQLite Tool", test_sqlite_tool),
        ("RAG Retrieval", test_rag_retrieval),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60 + "\n")
    
    if passed == total:
        print("✓ All component tests passed!")
        print("\nNote: To test the full system with DSPy and LangGraph,")
        print("ensure Ollama is running with phi3.5 model:")
        print("  ollama serve")
        print("  ollama pull phi3.5")
        print("\nThen run:")
        print("  python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
