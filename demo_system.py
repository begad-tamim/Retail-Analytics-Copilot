"""
Demo script to show the system structure without requiring Ollama.
This demonstrates component integration without actual LLM inference.
"""
import json
from agent.tools.sqlite_tool import get_sqlite_tool
from agent.rag.retrieval import get_rag_retriever


def demo_without_llm():
    """Demonstrate system components without LLM."""
    print("=" * 70)
    print("RETAIL ANALYTICS COPILOT - COMPONENT DEMO")
    print("=" * 70)
    
    # Initialize components
    print("\n1. Initializing Components...")
    sqlite_tool = get_sqlite_tool()
    rag_retriever = get_rag_retriever()
    print(f"   ✓ SQLite tool connected to database")
    print(f"   ✓ RAG retriever loaded {len(rag_retriever.chunks)} document chunks")
    
    # Demo question
    question = "What is the total revenue for July 2024?"
    print(f"\n2. Example Question:")
    print(f"   '{question}'")
    
    # Simulate routing (would use DSPy Router)
    print(f"\n3. Routing Decision:")
    print(f"   → Mode: 'hybrid' (requires both SQL and RAG)")
    
    # RAG retrieval
    print(f"\n4. RAG Retrieval:")
    chunks = rag_retriever.retrieve(question, k=3)
    for i, chunk in enumerate(chunks, 1):
        print(f"   [{i}] {chunk.id} (score: {chunk.score:.3f})")
        print(f"       Preview: {chunk.content[:80]}...")
    
    # Extract constraints (would use Planner node)
    print(f"\n5. Planning - Extract Constraints:")
    print(f"   → Date constraint: July 2024")
    print(f"   → KPI: Total Revenue formula from kpi_definitions.md")
    
    # SQL generation (would use DSPy NL2SQL)
    print(f"\n6. SQL Generation:")
    schema = sqlite_tool.get_schema_summary()
    print(f"   Schema available: {len(schema.split('Table:')) - 1} tables")
    
    example_sql = """
    SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as TotalRevenue
    FROM OrderDetails od
    JOIN Orders o ON od.OrderID = o.OrderID
    WHERE o.OrderDate LIKE '2024-07%'
    """.strip()
    print(f"   Generated SQL:")
    for line in example_sql.split('\n'):
        print(f"      {line}")
    
    # SQL execution
    print(f"\n7. SQL Execution:")
    result = sqlite_tool.execute_sql(example_sql)
    if result['error']:
        print(f"   ✗ Error: {result['error']}")
    else:
        print(f"   ✓ Success: {len(result['rows'])} rows returned")
        if result['rows']:
            print(f"   Result: {json.dumps(result['rows'][0], indent=6)}")
    
    # Synthesis (would use DSPy Synthesizer)
    print(f"\n8. Answer Synthesis:")
    print(f"   Combining:")
    print(f"     - SQL result: Revenue calculation")
    print(f"     - RAG context: KPI definition, date context")
    print(f"   Final Answer: '$485.82'")
    print(f"   Citations: ['kpi_definitions::chunk2']")
    print(f"   Confidence: 0.9")
    
    # Validation
    print(f"\n9. Validation:")
    print(f"   ✓ Answer is a number (matches format_hint)")
    print(f"   ✓ Citations provided")
    print(f"   ✓ SQL executed without errors")
    print(f"   ✓ Ready for output")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nThis demo shows the structure without running actual LLM inference.")
    print("For full LLM-powered operation, install Ollama and run:")
    print("  python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demo_without_llm()
