"""
DSPy signatures and modules for the retail analytics copilot.
Includes Router, NL2SQL, and Synthesizer with optimization support.
"""
import dspy
from typing import Literal


# Configure DSPy to use local Ollama model
def configure_dspy(model_name: str = "phi3.5", base_url: str = "http://localhost:11434"):
    """Configure DSPy with a local Ollama model.
    
    Args:
        model_name: Name of the Ollama model to use.
        base_url: Base URL for the Ollama API.
    """
    lm = dspy.OllamaLocal(
        model=model_name,
        base_url=base_url,
        max_tokens=2000,
        temperature=0.1
    )
    dspy.settings.configure(lm=lm)


# Signature: Router
class RouterSignature(dspy.Signature):
    """Route a question to the appropriate processing mode."""
    
    question = dspy.InputField(desc="The user's question about retail analytics")
    mode = dspy.OutputField(
        desc="Processing mode: 'rag' for document retrieval only, 'sql' for database queries only, 'hybrid' for both"
    )


# Signature: NL to SQL
class NL2SQLSignature(dspy.Signature):
    """Convert natural language question to SQL query."""
    
    question = dspy.InputField(desc="The user's question in natural language")
    constraints = dspy.InputField(desc="Any constraints or hints derived from documents (dates, KPIs, etc.)")
    db_schema = dspy.InputField(desc="Database schema information")
    sql = dspy.OutputField(desc="Valid SQL query to answer the question")


# Signature: Synthesizer
class SynthesizerSignature(dspy.Signature):
    """Synthesize final answer from retrieved documents and SQL results."""
    
    question = dspy.InputField(desc="The original user question")
    format_hint = dspy.InputField(desc="Expected format for the answer (e.g., 'number', 'list', 'text')")
    retrieved_docs = dspy.InputField(desc="Relevant document excerpts from RAG retrieval")
    sql_rows = dspy.InputField(desc="Results from SQL query execution")
    sql = dspy.InputField(desc="The SQL query that was executed")
    final_answer = dspy.OutputField(desc="The complete answer to the user's question")
    citations = dspy.OutputField(desc="List of document chunk IDs used as sources")
    confidence = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
    explanation = dspy.OutputField(desc="Brief explanation of how the answer was derived")


# Modules (Predictors)
class Router(dspy.Module):
    """Router module to determine processing mode."""
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RouterSignature)
    
    def forward(self, question: str) -> str:
        """Route the question to appropriate mode.
        
        Args:
            question: User's question.
            
        Returns:
            Mode string: 'rag', 'sql', or 'hybrid'.
        """
        result = self.predict(question=question)
        mode = result.mode.lower().strip()
        
        # Validate mode
        if mode not in ['rag', 'sql', 'hybrid']:
            # Default to hybrid if unclear
            mode = 'hybrid'
        
        return mode


class NL2SQL(dspy.Module):
    """Natural language to SQL conversion module."""
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(NL2SQLSignature)
    
    def forward(self, question: str, constraints: str, schema: str) -> str:
        """Convert question to SQL.
        
        Args:
            question: User's question.
            constraints: Any constraints from retrieved docs.
            schema: Database schema.
            
        Returns:
            SQL query string.
        """
        result = self.predict(
            question=question,
            constraints=constraints,
            db_schema=schema
        )
        return result.sql.strip()


class Synthesizer(dspy.Module):
    """Answer synthesizer module."""
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SynthesizerSignature)
    
    def forward(self, question: str, format_hint: str, retrieved_docs: str, 
                sql_rows: str, sql: str) -> dict:
        """Synthesize final answer.
        
        Args:
            question: Original question.
            format_hint: Expected answer format.
            retrieved_docs: Retrieved document content.
            sql_rows: SQL query results.
            sql: SQL query executed.
            
        Returns:
            Dictionary with final_answer, citations, confidence, explanation.
        """
        result = self.predict(
            question=question,
            format_hint=format_hint,
            retrieved_docs=retrieved_docs,
            sql_rows=sql_rows,
            sql=sql
        )
        
        # Parse citations if it's a string
        citations_str = result.citations
        if isinstance(citations_str, str):
            # Try to extract citation IDs
            citations = [c.strip() for c in citations_str.split(',') if '::' in c]
        else:
            citations = []
        
        # Parse confidence
        try:
            confidence = float(result.confidence)
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        except:
            confidence = 0.5
        
        return {
            'final_answer': result.final_answer,
            'citations': citations,
            'confidence': confidence,
            'explanation': result.explanation
        }


# Optimization helpers
def optimize_router(train_data: list) -> Router:
    """Optimize the router module with training data.
    
    Args:
        train_data: List of examples with 'question' and 'mode' fields.
        
    Returns:
        Optimized Router module.
    """
    # Create training examples
    trainset = []
    for item in train_data:
        example = dspy.Example(
            question=item['question'],
            mode=item['mode']
        ).with_inputs('question')
        trainset.append(example)
    
    # Define metric
    def routing_metric(example, pred, trace=None):
        return example.mode.lower() == pred.mode.lower()
    
    # Optimize with BootstrapFewShot
    optimizer = dspy.BootstrapFewShot(
        metric=routing_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3
    )
    
    router = Router()
    optimized_router = optimizer.compile(router, trainset=trainset)
    
    return optimized_router
