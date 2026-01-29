"""Tests for the mcp-data-check evaluation framework."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mcp_data_check.evaluator import Evaluator, EvalResult, EvalSummary
from mcp_data_check.eval_methods import (
    evaluate_numeric,
    evaluate_string,
    extract_number,
    extract_all_numbers,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLoadQuestions:
    """Tests for CSV loading functionality."""

    def test_load_questions_success(self):
        """Test loading questions from a valid CSV file."""
        with patch.object(Evaluator, '__init__', lambda self, **kwargs: None):
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.client = Mock()

            questions = evaluator.load_questions(FIXTURES_DIR / "sample_questions.csv")

            assert len(questions) == 4
            assert questions[0]["question"] == "How many grants were awarded in 2023?"
            assert questions[0]["expected_answer"] == "150"
            assert questions[0]["eval_type"] == "numeric"

    def test_load_questions_file_not_found(self):
        """Test that FileNotFoundError is raised for missing CSV."""
        with patch.object(Evaluator, '__init__', lambda self, **kwargs: None):
            evaluator = Evaluator.__new__(Evaluator)
            evaluator.client = Mock()

            with pytest.raises(FileNotFoundError):
                evaluator.load_questions("nonexistent_file.csv")


class TestEvaluateNumeric:
    """Tests for numeric evaluation."""

    def test_exact_match(self):
        """Test exact numeric match."""
        result = evaluate_numeric("The answer is 150", "150")
        assert result["passed"] is True
        assert result["extracted_value"] == 150
        assert result["expected_value"] == 150

    def test_within_tolerance(self):
        """Test numeric match within default 5% tolerance."""
        result = evaluate_numeric("The answer is 152", "150", tolerance=0.05)
        assert result["passed"] is True

    def test_outside_tolerance(self):
        """Test numeric mismatch outside tolerance."""
        result = evaluate_numeric("The answer is 200", "150", tolerance=0.05)
        assert result["passed"] is False

    def test_number_with_commas(self):
        """Test extracting numbers with commas."""
        result = evaluate_numeric("The total is 1,000,000", "1000000")
        assert result["passed"] is True
        assert result["extracted_value"] == 1000000

    def test_no_number_in_response(self):
        """Test when response contains no numbers."""
        result = evaluate_numeric("There are many grants", "150")
        assert result["passed"] is False
        assert result["extracted_value"] is None

    def test_bold_pattern_extraction(self):
        """Test extraction of bold-formatted numbers."""
        result = evaluate_numeric("The answer is **150** grants", "150")
        assert result["passed"] is True
        assert result["extracted_value"] == 150


class TestEvaluateString:
    """Tests for string evaluation."""

    def test_exact_match(self):
        """Test exact string match (case-insensitive)."""
        result = evaluate_string(
            "The agency is National Institutes of Health",
            "National Institutes of Health"
        )
        assert result["passed"] is True
        assert result["match_type"] == "exact"

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        result = evaluate_string(
            "The agency is NATIONAL INSTITUTES OF HEALTH",
            "National Institutes of Health"
        )
        assert result["passed"] is True
        assert result["match_type"] == "exact"

    def test_substring_match(self):
        """Test matching expected value as substring."""
        result = evaluate_string(
            "The program, called Research Excellence Initiative, was launched in 2020",
            "Research Excellence Initiative"
        )
        assert result["passed"] is True
        assert result["match_type"] == "exact"

    def test_fuzzy_match(self):
        """Test fuzzy matching for similar strings."""
        result = evaluate_string(
            "The National Institute of Health",
            "National Institutes of Health",
            fuzzy_threshold=0.8
        )
        assert result["similarity"] > 0.7

    def test_no_match(self):
        """Test when strings don't match."""
        result = evaluate_string(
            "The answer is completely different",
            "National Institutes of Health",
            fuzzy_threshold=0.8
        )
        assert result["passed"] is False


class TestExtractNumber:
    """Tests for number extraction helpers."""

    def test_extract_integer(self):
        """Test extracting an integer."""
        assert extract_number("There are 150 grants") == 150

    def test_extract_float(self):
        """Test extracting a float."""
        assert extract_number("The rate is 3.14 percent") == 3.14

    def test_extract_with_commas(self):
        """Test extracting number with commas."""
        assert extract_number("Total: 1,234,567") == 1234567

    def test_extract_negative(self):
        """Test extracting negative number."""
        assert extract_number("Change: -42 units") == -42

    def test_extract_all_numbers(self):
        """Test extracting all numbers from text."""
        numbers = extract_all_numbers("In 2023, there were 150 grants worth $1000000")
        assert 2023 in numbers
        assert 150 in numbers
        assert 1000000 in numbers


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_run_evaluation_with_mock(self):
        """Test run_evaluation with mocked API calls."""
        mock_client = Mock()

        with patch('mcp_data_check.evaluator.anthropic.Anthropic', return_value=mock_client):
            evaluator = Evaluator(
                server_url="https://example.com/mcp",
                api_key="test-key"
            )

            # Mock the call_model_with_mcp method (returns tuple of response, time, and tools_called)
            evaluator.call_model_with_mcp = Mock(return_value=("The answer is 150 grants", 1.5, []))

            questions = [
                {
                    "question": "How many grants?",
                    "expected_answer": "150",
                    "eval_type": "numeric"
                }
            ]

            summary = evaluator.run_evaluation(questions)

            assert summary.total == 1
            assert summary.passed == 1
            assert summary.failed == 0
            assert summary.pass_rate == 1.0
            assert summary.results[0].time_to_answer == 1.5

    def test_run_evaluation_with_failure(self):
        """Test run_evaluation when evaluation fails."""
        mock_client = Mock()

        with patch('mcp_data_check.evaluator.anthropic.Anthropic', return_value=mock_client):
            evaluator = Evaluator(
                server_url="https://example.com/mcp",
                api_key="test-key"
            )

            # Mock response that doesn't match expected (returns tuple of response, time, and tools_called)
            evaluator.call_model_with_mcp = Mock(return_value=("The answer is 999 grants", 2.3, []))

            questions = [
                {
                    "question": "How many grants?",
                    "expected_answer": "150",
                    "eval_type": "numeric"
                }
            ]

            summary = evaluator.run_evaluation(questions)

            assert summary.total == 1
            assert summary.passed == 0
            assert summary.failed == 1

    def test_evaluate_response_unknown_type(self):
        """Test evaluate_response with unknown eval_type."""
        mock_client = Mock()

        with patch('mcp_data_check.evaluator.anthropic.Anthropic', return_value=mock_client):
            evaluator = Evaluator(
                server_url="https://example.com/mcp",
                api_key="test-key"
            )

            result = evaluator.evaluate_response(
                question="Test question",
                expected_answer="test",
                eval_type="unknown_type",
                model_response="test response"
            )

            assert result.passed is False
            assert "Unknown eval_type" in result.error


class TestEvalSummary:
    """Tests for EvalSummary dataclass."""

    def test_to_dict(self):
        """Test converting EvalSummary to dictionary."""
        result = EvalResult(
            question="Test?",
            expected_answer="42",
            eval_type="numeric",
            model_response="42",
            passed=True,
            details={"extracted_value": 42}
        )

        summary = EvalSummary(
            total=1,
            passed=1,
            failed=0,
            pass_rate=1.0,
            by_eval_type={"numeric": {"total": 1, "passed": 1}},
            results=[result]
        )

        d = summary.to_dict()

        assert d["summary"]["total"] == 1
        assert d["summary"]["pass_rate"] == 1.0
        assert len(d["results"]) == 1
