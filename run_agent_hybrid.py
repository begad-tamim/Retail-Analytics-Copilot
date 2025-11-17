"""
CLI entrypoint for the retail analytics hybrid agent.
Processes batch questions and outputs JSONL results.
"""
import argparse
import json
import sys
from pathlib import Path

from agent.dspy_signatures import configure_dspy
from agent.graph_hybrid import HybridAgent


def process_batch(batch_file: str, output_file: str):
    """Process a batch of questions and write results.
    
    Args:
        batch_file: Path to input JSONL file with questions.
        output_file: Path to output JSONL file for results.
    """
    # Configure DSPy
    try:
        configure_dspy(model_name="phi3.5")
        print("DSPy configured with Ollama phi3.5 model")
    except Exception as e:
        print(f"Warning: Could not configure DSPy with Ollama: {e}")
        print("Please ensure Ollama is running with: ollama serve")
        print("And that phi3.5 model is available: ollama pull phi3.5")
        sys.exit(1)
    
    # Initialize agent
    print("Initializing hybrid agent...")
    agent = HybridAgent()
    
    # Read batch questions
    print(f"Reading questions from {batch_file}")
    questions = []
    with open(batch_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    
    print(f"Processing {len(questions)} questions...")
    
    # Process each question
    results = []
    for idx, item in enumerate(questions, 1):
        question_id = item.get("id", f"q{idx}")
        question = item.get("question", "")
        format_hint = item.get("format_hint", "text")
        
        print(f"\n[{idx}/{len(questions)}] Processing: {question[:60]}...")
        
        try:
            result = agent.run(question_id, question, format_hint)
            results.append(result)
            print(f"  ✓ Answer: {str(result['final_answer'])[:60]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Add error result
            results.append({
                "id": question_id,
                "question": question,
                "final_answer": "Error processing question",
                "citations": [],
                "confidence": 0.0,
                "explanation": str(e),
                "trace": []
            })
    
    # Write results
    print(f"\nWriting results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"✓ Complete! Processed {len(results)} questions")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run the retail analytics hybrid agent on a batch of questions"
    )
    parser.add_argument(
        "--batch",
        type=str,
        default="sample_questions_hybrid_eval.jsonl",
        help="Path to input JSONL file with questions"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs_hybrid.jsonl",
        help="Path to output JSONL file for results"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.batch).exists():
        print(f"Error: Input file '{args.batch}' not found")
        sys.exit(1)
    
    # Process batch
    process_batch(args.batch, args.out)


if __name__ == "__main__":
    main()
