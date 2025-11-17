"""
LangGraph orchestration for the hybrid retail analytics agent.
Includes state management, nodes, and workflow with validation/repair.
"""
from typing import TypedDict, List, Optional, Any, Annotated
from langgraph.graph import StateGraph, END
import operator
import json
import re

from agent.dspy_signatures import Router, NL2SQL, Synthesizer
from agent.rag.retrieval import get_rag_retriever
from agent.tools.sqlite_tool import get_sqlite_tool


# State definition
class AgentState(TypedDict):
    """State for the retail analytics agent."""
    # Input
    id: str
    question: str
    format_hint: str
    
    # Routing
    mode: str  # 'rag', 'sql', or 'hybrid'
    
    # RAG
    retrieved_chunks: List[dict]
    constraints: str
    
    # SQL
    sql: str
    sql_result: dict
    attempts: int
    
    # Output
    final_answer: Any
    citations: List[str]
    confidence: float
    explanation: str
    done: bool
    
    # Trace
    trace: Annotated[List[dict], operator.add]


class HybridAgent:
    """Hybrid retail analytics agent with LangGraph orchestration."""
    
    def __init__(self):
        """Initialize the agent and build the graph."""
        # Initialize modules
        self.router = Router()
        self.nl2sql = NL2SQL()
        self.synthesizer = Synthesizer()
        self.rag_retriever = get_rag_retriever()
        self.sqlite_tool = get_sqlite_tool()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("nl2sql", self.nl2sql_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        workflow.add_node("validate_and_repair", self.validate_and_repair_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add edges
        workflow.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "retriever": "retriever",
                "planner": "planner"
            }
        )
        
        workflow.add_edge("retriever", "planner")
        
        workflow.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "nl2sql": "nl2sql",
                "synthesizer": "synthesizer"
            }
        )
        
        workflow.add_edge("nl2sql", "executor")
        workflow.add_edge("executor", "synthesizer")
        workflow.add_edge("synthesizer", "validate_and_repair")
        
        workflow.add_conditional_edges(
            "validate_and_repair",
            self._route_after_validation,
            {
                "nl2sql": "nl2sql",
                "synthesizer": "synthesizer",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _route_after_router(self, state: AgentState) -> str:
        """Determine next node after routing."""
        mode = state.get("mode", "hybrid")
        if mode in ["rag", "hybrid"]:
            return "retriever"
        else:
            return "planner"
    
    def _route_after_planner(self, state: AgentState) -> str:
        """Determine next node after planning."""
        mode = state.get("mode", "hybrid")
        if mode in ["sql", "hybrid"]:
            return "nl2sql"
        else:
            return "synthesizer"
    
    def _route_after_validation(self, state: AgentState) -> str:
        """Determine next node after validation."""
        if state.get("done", False):
            return "end"
        
        attempts = state.get("attempts", 0)
        if attempts >= 2:
            # Give up after 2 attempts
            return "end"
        
        # Check if we need to repair SQL or synthesizer
        sql_result = state.get("sql_result", {})
        if sql_result.get("error"):
            return "nl2sql"
        else:
            return "synthesizer"
    
    # Node implementations
    def router_node(self, state: AgentState) -> dict:
        """Route the question to appropriate mode."""
        question = state["question"]
        mode = self.router.forward(question)
        
        trace_entry = {
            "node": "router",
            "mode": mode,
            "question_preview": question[:100]
        }
        
        return {
            "mode": mode,
            "trace": [trace_entry]
        }
    
    def retriever_node(self, state: AgentState) -> dict:
        """Retrieve relevant document chunks."""
        question = state["question"]
        chunks = self.rag_retriever.retrieve(question, k=4)
        
        retrieved_chunks = [
            {
                "id": chunk.id,
                "source": chunk.source,
                "content": chunk.content,
                "score": chunk.score
            }
            for chunk in chunks
        ]
        
        trace_entry = {
            "node": "retriever",
            "num_chunks": len(retrieved_chunks),
            "chunk_ids": [c["id"] for c in retrieved_chunks]
        }
        
        return {
            "retrieved_chunks": retrieved_chunks,
            "trace": [trace_entry]
        }
    
    def planner_node(self, state: AgentState) -> dict:
        """Extract constraints from retrieved documents."""
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        
        # Extract constraints from chunks
        constraints_parts = []
        
        # Look for date mentions
        for chunk in chunks:
            content = chunk["content"]
            # Look for date patterns
            date_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
            dates = re.findall(date_pattern, content, re.IGNORECASE)
            if dates:
                constraints_parts.append(f"Relevant dates: {', '.join(dates)}")
            
            # Look for KPI formulas
            if "Formula:" in content or "SELECT" in content:
                constraints_parts.append(f"KPI formula found in {chunk['source']}")
        
        constraints = "; ".join(constraints_parts) if constraints_parts else "No specific constraints"
        
        trace_entry = {
            "node": "planner",
            "constraints": constraints
        }
        
        return {
            "constraints": constraints,
            "trace": [trace_entry]
        }
    
    def nl2sql_node(self, state: AgentState) -> dict:
        """Convert natural language to SQL."""
        question = state["question"]
        constraints = state.get("constraints", "")
        schema = self.sqlite_tool.get_schema_summary()
        
        # Check if this is a repair attempt
        attempts = state.get("attempts", 0)
        if attempts > 0:
            # Add guidance for repair
            prev_error = state.get("sql_result", {}).get("error", "")
            constraints += f" Previous SQL failed with: {prev_error}. Please fix the SQL."
        
        sql = self.nl2sql.forward(question, constraints, schema)
        
        # Clean up SQL
        sql = sql.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()
        
        trace_entry = {
            "node": "nl2sql",
            "sql": sql,
            "attempt": attempts + 1
        }
        
        return {
            "sql": sql,
            "trace": [trace_entry]
        }
    
    def executor_node(self, state: AgentState) -> dict:
        """Execute SQL query."""
        sql = state.get("sql", "")
        attempts = state.get("attempts", 0)
        
        result = self.sqlite_tool.execute_sql(sql)
        
        trace_entry = {
            "node": "executor",
            "success": result.get("error") is None,
            "num_rows": len(result.get("rows", [])),
            "error": result.get("error")
        }
        
        return {
            "sql_result": result,
            "attempts": attempts + 1,
            "trace": [trace_entry]
        }
    
    def synthesizer_node(self, state: AgentState) -> dict:
        """Synthesize final answer from all sources."""
        question = state["question"]
        format_hint = state.get("format_hint", "text")
        
        # Prepare retrieved docs text
        chunks = state.get("retrieved_chunks", [])
        if chunks:
            retrieved_docs = "\n\n".join([
                f"[{c['id']}] {c['content'][:200]}..."
                for c in chunks
            ])
        else:
            retrieved_docs = "No documents retrieved"
        
        # Prepare SQL results
        sql_result = state.get("sql_result", {})
        sql = state.get("sql", "")
        
        if sql_result.get("error"):
            sql_rows = f"SQL Error: {sql_result['error']}"
        elif sql_result.get("rows"):
            rows = sql_result["rows"][:5]  # Limit to first 5 rows
            sql_rows = json.dumps(rows, indent=2)
        else:
            sql_rows = "No SQL results"
        
        # Call synthesizer
        result = self.synthesizer.forward(
            question=question,
            format_hint=format_hint,
            retrieved_docs=retrieved_docs,
            sql_rows=sql_rows,
            sql=sql
        )
        
        trace_entry = {
            "node": "synthesizer",
            "confidence": result["confidence"],
            "num_citations": len(result["citations"])
        }
        
        return {
            "final_answer": result["final_answer"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
            "trace": [trace_entry]
        }
    
    def validate_and_repair_node(self, state: AgentState) -> dict:
        """Validate the output and decide if repair is needed."""
        final_answer = state.get("final_answer", "")
        citations = state.get("citations", [])
        sql_result = state.get("sql_result", {})
        format_hint = state.get("format_hint", "text")
        attempts = state.get("attempts", 0)
        mode = state.get("mode", "hybrid")
        
        issues = []
        
        # Validate format
        if format_hint == "number":
            try:
                float(str(final_answer).replace('$', '').replace(',', ''))
            except:
                issues.append("Answer should be a number")
        
        # Validate JSON serializability
        try:
            json.dumps(final_answer)
        except:
            issues.append("Answer is not JSON serializable")
        
        # Validate citations for RAG/hybrid mode
        if mode in ["rag", "hybrid"] and not citations:
            issues.append("Missing citations")
        
        # Validate SQL results
        if mode in ["sql", "hybrid"]:
            if sql_result.get("error"):
                issues.append(f"SQL error: {sql_result['error']}")
            elif not sql_result.get("rows"):
                issues.append("SQL returned no rows")
        
        # Decide if we're done
        done = (len(issues) == 0) or (attempts >= 2)
        
        trace_entry = {
            "node": "validate_and_repair",
            "issues": issues,
            "done": done,
            "attempts": attempts
        }
        
        return {
            "done": done,
            "trace": [trace_entry]
        }
    
    def run(self, question_id: str, question: str, format_hint: str = "text") -> dict:
        """Run the agent on a single question.
        
        Args:
            question_id: Unique identifier for the question.
            question: The user's question.
            format_hint: Expected format of the answer.
            
        Returns:
            Dictionary with the complete result.
        """
        # Initialize state
        initial_state = {
            "id": question_id,
            "question": question,
            "format_hint": format_hint,
            "mode": "",
            "retrieved_chunks": [],
            "constraints": "",
            "sql": "",
            "sql_result": {},
            "attempts": 0,
            "final_answer": "",
            "citations": [],
            "confidence": 0.0,
            "explanation": "",
            "done": False,
            "trace": []
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Return result
        return {
            "id": final_state["id"],
            "question": final_state["question"],
            "final_answer": final_state["final_answer"],
            "citations": final_state["citations"],
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "trace": final_state["trace"]
        }
