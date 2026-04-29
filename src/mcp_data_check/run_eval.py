#!/usr/bin/env python3
"""CLI entry point for MCP server evaluation."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .evaluator import Evaluator

# Load environment variables from .env file
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MCP server accuracy against known questions and answers"
    )
    parser.add_argument(
        "server_url",
        help="URL of the MCP server to evaluate"
    )
    parser.add_argument(
        "-q", "--questions",
        required=True,
        help="Path to questions CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for results (default: ./results)"
    )
    parser.add_argument(
        "-p", "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider to use (default: anthropic)"
    )
    parser.add_argument(
        "-k", "--api-key",
        default=None,
        help="API key for the chosen provider (defaults to ANTHROPIC_API_KEY or OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "-m", "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use for evaluation (default: claude-sonnet-4-20250514; use e.g. gpt-4o for OpenAI)"
    )
    parser.add_argument(
        "-n", "--server-name",
        default="mcp-server",
        help="Name for the MCP server (default: mcp-server)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run each question with and without MCP server and compare results"
    )
    parser.add_argument(
        "-r", "--repeats",
        type=int,
        default=5,
        help="Number of times to run each question (majority vote determines pass/fail, default: 5)"
    )

    args = parser.parse_args()

    # Determine paths
    questions_path = Path(args.questions)
    output_dir = Path(args.output) if args.output else Path("./results")

    # Validate questions file exists
    if not questions_path.exists():
        print(f"Error: Questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)

    # Create evaluator
    evaluator = Evaluator(
        server_url=args.server_url,
        api_key=args.api_key,
        model=args.model,
        server_name=args.server_name,
        provider=args.provider,
    )

    # Load questions
    print(f"Loading questions from {questions_path}...")
    questions = evaluator.load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    print("-" * 50)

    if args.compare:
        print(f"\nComparing MCP vs baseline against {args.server_url}...")
        print("-" * 50)

        comparison = evaluator.run_comparison(questions, verbose=args.verbose, repeats=args.repeats)

        print("-" * 50)
        print("\nComparison Summary")
        print("=" * 50)
        print(f"Total questions:    {comparison.total}")
        print(f"MCP pass rate:      {comparison.mcp_passed}/{comparison.total} ({comparison.mcp_pass_rate:.1%})")
        print(f"Baseline pass rate: {comparison.baseline_passed}/{comparison.total} ({comparison.baseline_pass_rate:.1%})")
        delta = comparison.mcp_pass_rate - comparison.baseline_pass_rate
        print(f"Delta (MCP - base): {delta:+.1%}")

        print("\nBreakdown:")
        print(f"  Both passed:      {comparison.both_passed}")
        print(f"  Neither passed:   {comparison.neither_passed}")
        print(f"  MCP only passed:  {comparison.mcp_only_passed}")
        print(f"  Baseline only:    {comparison.baseline_only_passed}")

        # Print questions where results differ
        differing = [r for r in comparison.results if r.mcp.passed != r.baseline.passed]
        if differing:
            print(f"\nQuestions where MCP and baseline differ ({len(differing)}):")
            for r in differing:
                mcp_str = "PASS" if r.mcp.passed else "FAIL"
                base_str = "PASS" if r.baseline.passed else "FAIL"
                print(f"\n  Q: {r.mcp.question[:80]}...")
                print(f"  MCP: {mcp_str}  |  Baseline: {base_str}")
                if not r.mcp.passed and r.mcp.error:
                    print(f"  MCP error: {r.mcp.error}")
                if not r.baseline.passed and r.baseline.error:
                    print(f"  Baseline error: {r.baseline.error}")

        output_path = evaluator.save_comparison(comparison, output_dir)
        print(f"\nResults saved to: {output_path}")

        sys.exit(0 if comparison.mcp_passed == comparison.total else 1)

    else:
        # Run evaluation
        print(f"\nEvaluating against {args.server_url}...")
        print("-" * 50)

        summary = evaluator.run_evaluation(questions, verbose=args.verbose, repeats=args.repeats)

        # Print summary
        print("-" * 50)
        print("\nEvaluation Summary")
        print("=" * 50)
        print(f"Total questions: {summary.total}")
        print(f"Passed: {summary.passed}")
        print(f"Failed: {summary.failed}")
        print(f"Pass rate: {summary.pass_rate:.1%}")

        if summary.by_eval_type:
            print("\nBy evaluation type:")
            for eval_type, stats in summary.by_eval_type.items():
                type_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
                print(f"  {eval_type}: {stats['passed']}/{stats['total']} ({type_rate:.1%})")

        # Print failed questions
        failed_results = [r for r in summary.results if not r.passed]
        if failed_results:
            print(f"\nFailed questions ({len(failed_results)}):")
            for r in failed_results:
                print(f"\n  Q: {r.question[:80]}...")
                print(f"  Expected: {r.expected_answer[:50]}...")
                if r.error:
                    print(f"  Error: {r.error}")
                else:
                    print(f"  Details: {r.details.get('details', 'N/A')}")

        # Save results
        output_path = evaluator.save_results(summary, output_dir)
        print(f"\nResults saved to: {output_path}")

        # Exit with non-zero if any failures
        sys.exit(0 if summary.failed == 0 else 1)


if __name__ == "__main__":
    main()
