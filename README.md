# mcp-data-check

Evaluate MCP server accuracy against known questions and answers.

## Installation

```bash
pip install mcp-data-check
```

Or install from source:

```bash
pip install -e .
```

## Usage

### Python API

**Anthropic (default)**

```python
from mcp_data_check import run_evaluation

results = run_evaluation(
    questions_filepath="questions.csv",
    api_key="sk-ant-...",
    server_url="https://mcp.example.com/sse"
)

print(f"Pass rate: {results['summary']['pass_rate']:.1%}")
print(f"Passed: {results['summary']['passed']}/{results['summary']['total']}")
```

**OpenAI**

```python
from mcp_data_check import run_evaluation

results = run_evaluation(
    questions_filepath="questions.csv",
    api_key="sk-...",
    server_url="https://mcp.example.com/sse",
    provider="openai",
    model="gpt-4o"
)
```

### Command Line

**Anthropic (default)**

```bash
mcp-data-check https://mcp.example.com/sse -q questions.csv -k YOUR_API_KEY
```

**OpenAI**

```bash
mcp-data-check https://mcp.example.com/sse -q questions.csv -p openai -m gpt-4o -k YOUR_API_KEY
```

**Baseline comparison (MCP vs no tools)**

```bash
# Anthropic
mcp-data-check https://mcp.example.com/sse -q questions.csv --compare

# OpenAI
mcp-data-check https://mcp.example.com/sse -q questions.csv -p openai -m gpt-4o --compare
```

Each question is run `--repeats` times per mode (default 5). Pass/fail is determined by majority vote. Results are saved to `./results/comparison_<timestamp>.json`.

Options:
- `-q, --questions`: Path to questions CSV file (required)
- `-p, --provider`: LLM provider to use: `anthropic` (default) or `openai`
- `-k, --api-key`: API key for the chosen provider (defaults to `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` env var)
- `-o, --output`: Output directory for results (default: `./results`)
- `-m, --model`: Model to use for evaluation (default: `claude-sonnet-4-20250514`; use e.g. `gpt-4o` for OpenAI)
- `-n, --server-name`: Name for the MCP server (default: `mcp-server`)
- `-r, --repeats`: Number of times to run each question (default: 5; majority vote determines pass/fail)
- `-v, --verbose`: Print detailed progress
- `--compare`: Run each question with and without the MCP server and report the delta

### Baseline Comparison (Python API)

**Anthropic**

```python
from mcp_data_check import Evaluator

evaluator = Evaluator(server_url="https://mcp.example.com/sse", api_key="sk-ant-...")
questions = evaluator.load_questions("questions.csv")

comparison = evaluator.run_comparison(questions, repeats=5, verbose=True)

print(f"MCP pass rate:      {comparison.mcp_pass_rate:.1%}")
print(f"Baseline pass rate: {comparison.baseline_pass_rate:.1%}")
print(f"Delta:              {comparison.mcp_pass_rate - comparison.baseline_pass_rate:+.1%}")

evaluator.save_comparison(comparison, "./results")
```

**OpenAI**

```python
from mcp_data_check import Evaluator

evaluator = Evaluator(
    server_url="https://mcp.example.com/sse",
    api_key="sk-...",
    provider="openai",
    model="gpt-4o"
)
questions = evaluator.load_questions("questions.csv")

comparison = evaluator.run_comparison(questions, repeats=5, verbose=True)

print(f"MCP pass rate:      {comparison.mcp_pass_rate:.1%}")
print(f"Baseline pass rate: {comparison.baseline_pass_rate:.1%}")
```

## Questions CSV Format

The questions CSV file must have three columns:

| Column | Description |
|--------|-------------|
| `question` | The question to ask the MCP server |
| `expected_answer` | The expected answer to compare against |
| `eval_type` | Evaluation method: `numeric`, `string`, or `llm_judge` |

Example:

```csv
question,expected_answer,eval_type
How many grants were awarded in 2023?,1234,numeric
What organization received the most funding?,NIH,string
Explain the grant distribution,Most grants went to research institutions...,llm_judge
```

### Evaluation Types

- **numeric**: Extracts numbers from responses and compares with 5% tolerance
- **string**: Checks if expected string appears in response (case-insensitive)
- **llm_judge**: Uses the selected model to semantically evaluate if the response is correct

## Return Value

The `run_evaluation` function returns a dictionary:

```python
{
    "summary": {
        "total": 10,
        "passed": 8,
        "failed": 2,
        "pass_rate": 0.8,
        "by_eval_type": {
            "numeric": {"total": 5, "passed": 4},
            "string": {"total": 3, "passed": 3},
            "llm_judge": {"total": 2, "passed": 1}
        }
    },
    "results": [
        {
            "question": "...",
            "expected_answer": "...",
            "eval_type": "numeric",
            "model_response": "...",
            "passed": True,
            "details": {...},
            "error": None,
            "time_to_answer": 2.35,
            "tools_called": [
                {
                    "tool_name": "get_grants",
                    "server_name": "mcp-server",
                    "input": {"year": 2023}
                }
            ]
        },
        ...
    ],
    "metadata": {
        "server_url": "https://mcp.example.com/sse",
        "model": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "timestamp": "20250127_143022"
    }
}
```

### Result Fields

Each result in the `results` array contains:

| Field | Description |
|-------|-------------|
| `question` | The original question asked |
| `expected_answer` | The expected answer from the CSV |
| `eval_type` | Evaluation method used |
| `model_response` | The model's full response text |
| `passed` | Whether the evaluation passed |
| `details` | Additional evaluation details |
| `error` | Error message if the evaluation failed |
| `time_to_answer` | Average response time in seconds across repeats |
| `tools_called` | List of MCP tools invoked during the response |
| `repeat_count` | Number of times the question was run |
| `repeat_pass_count` | Number of runs that passed (out of `repeat_count`) |

The `tools_called` array contains objects with:
- `tool_name`: Name of the MCP tool called
- `server_name`: Name of the MCP server that provided the tool
- `input`: Parameters passed to the tool

## Requirements

- Python 3.10+
- API key for your chosen provider (Anthropic or OpenAI)
