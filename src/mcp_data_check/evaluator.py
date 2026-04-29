"""Core evaluation logic for the MCP server evaluation framework."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

import anthropic
import openai

from .eval_methods import evaluate_numeric, evaluate_string, evaluate_llm_judge

SUPPORTED_PROVIDERS = ("anthropic", "openai")


@dataclass
class EvalResult:
    """Result of evaluating a single question."""
    question: str
    expected_answer: str
    eval_type: str
    model_response: str
    passed: bool
    details: dict = field(default_factory=dict)
    error: str | None = None
    time_to_answer: float | None = None
    tools_called: list[dict] = field(default_factory=list)


@dataclass
class EvalSummary:
    """Summary of all evaluation results."""
    total: int
    passed: int
    failed: int
    pass_rate: float
    by_eval_type: dict = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": self.pass_rate,
                "by_eval_type": self.by_eval_type
            },
            "results": [asdict(r) for r in self.results]
        }


class Evaluator:
    """Evaluates MCP server responses against expected answers."""

    def __init__(
        self,
        server_url: str,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        server_name: str = "mcp-server",
        provider: str = "anthropic",
    ):
        """Initialize the evaluator.

        Args:
            server_url: URL of the MCP server to evaluate
            api_key: API key for the chosen provider (falls back to the provider's
                     standard env var: ANTHROPIC_API_KEY or OPENAI_API_KEY)
            model: Model to use for generating responses and evaluation
            server_name: Name to use for the MCP server in API calls
            provider: Which LLM provider to use ('anthropic' or 'openai')
        """
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider {provider!r}. Must be one of: {SUPPORTED_PROVIDERS}"
            )

        self.server_url = server_url
        self.model = model
        self.server_name = server_name
        self.provider = provider

        if provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        else:
            self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def load_questions(self, csv_path: str | Path) -> list[dict]:
        """Load questions from a CSV file.

        Args:
            csv_path: Path to the CSV file with questions

        Returns:
            List of question dictionaries with 'question', 'expected_answer', 'eval_type'
        """
        questions = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append({
                    "question": row["question"],
                    "expected_answer": row["expected_answer"],
                    "eval_type": row["eval_type"]
                })
        return questions

    def call_model_with_mcp(self, question: str) -> tuple[str, float, list[dict]]:
        """Call the configured model with MCP server access to answer a question.

        Args:
            question: The question to ask

        Returns:
            Tuple of (response_text, time_to_answer_in_seconds, tools_called)
        """
        if self.provider == "anthropic":
            return self._call_anthropic_mcp(question)
        else:
            return self._call_openai_mcp(question)

    def _call_anthropic_mcp(self, question: str) -> tuple[str, float, list[dict]]:
        start_time = time.perf_counter()
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4096,
            betas=["mcp-client-2025-04-04"],
            mcp_servers=[
                {
                    "type": "url",
                    "url": self.server_url,
                    "name": self.server_name
                }
            ],
            messages=[{"role": "user", "content": question}]
        )
        elapsed_time = time.perf_counter() - start_time

        text_parts = []
        tool_uses_by_id = {}
        tools_called = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "mcp_tool_use":
                tool_call = {
                    "tool_name": block.name,
                    "server_name": block.server_name,
                    "input": block.input,
                    "api_response": None,
                }
                tool_uses_by_id[block.id] = tool_call
                tools_called.append(tool_call)
            elif block.type == "mcp_tool_result":
                if block.tool_use_id in tool_uses_by_id:
                    serialized_content = []
                    for content_block in block.content:
                        if hasattr(content_block, "model_dump"):
                            serialized_content.append(content_block.model_dump())
                        elif hasattr(content_block, "text"):
                            serialized_content.append({"type": "text", "text": content_block.text})
                        else:
                            serialized_content.append(str(content_block))
                    tool_uses_by_id[block.tool_use_id]["api_response"] = serialized_content

        return "\n".join(text_parts), elapsed_time, tools_called

    def _call_openai_mcp(self, question: str) -> tuple[str, float, list[dict]]:
        start_time = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            tools=[{
                "type": "mcp",
                "server_label": self.server_name,
                "server_url": self.server_url,
                "require_approval": "never",
            }],
            input=question,
        )
        elapsed_time = time.perf_counter() - start_time

        text_parts = []
        tools_called = []
        tool_calls_by_id = {}

        for item in response.output:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        text_parts.append(content.text)

            elif item_type == "mcp_call":
                args = getattr(item, "arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (ValueError, TypeError):
                        pass
                tool_call = {
                    "tool_name": getattr(item, "name", ""),
                    "server_name": getattr(item, "server_label", self.server_name),
                    "input": args,
                    "api_response": None,
                }
                item_id = getattr(item, "id", None)
                if item_id:
                    tool_calls_by_id[item_id] = tool_call
                tools_called.append(tool_call)

            elif item_type == "mcp_call_result":
                call_id = getattr(item, "call_id", None)
                if call_id and call_id in tool_calls_by_id:
                    output = getattr(item, "output", None)
                    tool_calls_by_id[call_id]["api_response"] = output

        return "\n".join(text_parts), elapsed_time, tools_called

    def evaluate_response(
        self,
        question: str,
        expected_answer: str,
        eval_type: str,
        model_response: str
    ) -> EvalResult:
        """Evaluate a model response against the expected answer.

        Args:
            question: The original question
            expected_answer: The expected answer
            eval_type: Type of evaluation ('numeric', 'string', 'llm_judge')
            model_response: The model's response

        Returns:
            EvalResult with pass/fail status and details
        """
        try:
            if eval_type == "numeric":
                result = evaluate_numeric(
                    model_response, expected_answer,
                    question=question,
                    client=self.client,
                    provider=self.provider,
                    model=self.model,
                )
            elif eval_type == "string":
                result = evaluate_string(model_response, expected_answer)
            elif eval_type == "llm_judge":
                result = evaluate_llm_judge(
                    question, model_response, expected_answer,
                    client=self.client,
                    provider=self.provider,
                    model=self.model,
                )
            else:
                return EvalResult(
                    question=question,
                    expected_answer=expected_answer,
                    eval_type=eval_type,
                    model_response=model_response,
                    passed=False,
                    error=f"Unknown eval_type: {eval_type}"
                )

            return EvalResult(
                question=question,
                expected_answer=expected_answer,
                eval_type=eval_type,
                model_response=model_response,
                passed=result["passed"],
                details=result
            )

        except Exception as e:
            return EvalResult(
                question=question,
                expected_answer=expected_answer,
                eval_type=eval_type,
                model_response=model_response,
                passed=False,
                error=str(e)
            )

    def run_evaluation(
        self,
        questions: list[dict],
        verbose: bool = False
    ) -> EvalSummary:
        """Run evaluation on a list of questions.

        Args:
            questions: List of question dicts with 'question', 'expected_answer', 'eval_type'
            verbose: Whether to print progress

        Returns:
            EvalSummary with results and statistics
        """
        results = []
        by_eval_type = {}

        for i, q in enumerate(questions):
            if verbose:
                print(f"Evaluating question {i+1}/{len(questions)}: {q['question'][:50]}...")

            try:
                model_response, time_to_answer, tools_called = self.call_model_with_mcp(q["question"])
            except Exception as e:
                model_response = ""
                time_to_answer = None
                tools_called = []
                result = EvalResult(
                    question=q["question"],
                    expected_answer=q["expected_answer"],
                    eval_type=q["eval_type"],
                    model_response=model_response,
                    passed=False,
                    error=f"API call failed: {e}",
                    time_to_answer=time_to_answer,
                    tools_called=tools_called
                )
                results.append(result)
                continue

            result = self.evaluate_response(
                q["question"],
                q["expected_answer"],
                q["eval_type"],
                model_response
            )
            result.time_to_answer = time_to_answer
            result.tools_called = tools_called
            results.append(result)

            eval_type = q["eval_type"]
            if eval_type not in by_eval_type:
                by_eval_type[eval_type] = {"total": 0, "passed": 0}
            by_eval_type[eval_type]["total"] += 1
            if result.passed:
                by_eval_type[eval_type]["passed"] += 1

            if verbose:
                status = "PASS" if result.passed else "FAIL"
                time_str = f" ({result.time_to_answer:.2f}s)" if result.time_to_answer else ""
                print(f"  Result: {status}{time_str}")

        total = len(results)
        passed = sum(1 for r in results if r.passed)

        return EvalSummary(
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total if total > 0 else 0.0,
            by_eval_type=by_eval_type,
            results=results
        )

    def save_results(
        self,
        summary: EvalSummary,
        output_dir: str | Path,
        filename_prefix: str = "eval"
    ) -> Path:
        """Save evaluation results to a JSON file.

        Args:
            summary: The evaluation summary to save
            output_dir: Directory to save results
            filename_prefix: Prefix for the output filename

        Returns:
            Path to the saved results file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        output_path = output_dir / filename

        output_data = {
            "summary": {
                "total": summary.total,
                "passed": summary.passed,
                "failed": summary.failed,
                "pass_rate": summary.pass_rate,
                "by_eval_type": summary.by_eval_type
            },
            "results": [asdict(r) for r in summary.results],
            "metadata": {
                "server_url": self.server_url,
                "model": self.model,
                "provider": self.provider,
                "timestamp": timestamp
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        return output_path
