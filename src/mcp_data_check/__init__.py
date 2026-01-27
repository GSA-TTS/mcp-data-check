"""MCP Data Check - Evaluate MCP server accuracy against known questions and answers."""

from pathlib import Path

from .evaluator import Evaluator, EvalSummary, EvalResult
from .eval_methods import evaluate_numeric, evaluate_string, evaluate_llm_judge


def run_evaluation(
    questions_filepath: str | Path,
    api_key: str,
    server_url: str,
    model: str = "claude-sonnet-4-20250514",
    server_name: str = "mcp-server",
    verbose: bool = False
) -> dict:
    """Run MCP server evaluation and return results as JSON.

    Args:
        questions_filepath: Path to a CSV file with columns: question, expected_answer, eval_type
        api_key: Anthropic API key for authentication
        server_url: URL of the remote MCP server to evaluate
        model: Claude model to use for generating responses (default: claude-sonnet-4-20250514)
        server_name: Name to use for the MCP server (default: mcp-server)
        verbose: Whether to print progress during evaluation (default: False)

    Returns:
        dict containing:
            - summary: dict with total, passed, failed, pass_rate, by_eval_type
            - results: list of individual evaluation results
            - metadata: dict with server_url, model, and timestamp

    Example:
        >>> from mcp_data_check import run_evaluation
        >>> results = run_evaluation(
        ...     questions_filepath="questions.csv",
        ...     api_key="sk-ant-...",
        ...     server_url="https://mcp.example.com/sse"
        ... )
        >>> print(f"Pass rate: {results['summary']['pass_rate']:.1%}")
    """
    from datetime import datetime

    questions_path = Path(questions_filepath)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    evaluator = Evaluator(
        server_url=server_url,
        api_key=api_key,
        model=model,
        server_name=server_name
    )

    questions = evaluator.load_questions(questions_path)
    summary = evaluator.run_evaluation(questions, verbose=verbose)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        **summary.to_dict(),
        "metadata": {
            "server_url": server_url,
            "model": model,
            "timestamp": timestamp
        }
    }


__all__ = [
    "run_evaluation",
    "Evaluator",
    "EvalSummary",
    "EvalResult",
    "evaluate_numeric",
    "evaluate_string",
    "evaluate_llm_judge",
]
