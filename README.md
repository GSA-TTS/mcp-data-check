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

### Command Line

```bash
mcp-data-check https://mcp.example.com/sse -q questions.csv -k YOUR_API_KEY
```

Options:
- `-q, --questions`: Path to questions CSV file (required)
- `-k, --api-key`: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
- `-o, --output`: Output directory for results (default: ./results)
- `-m, --model`: Claude model to use (default: claude-sonnet-4-20250514)
- `-n, --server-name`: Name for the MCP server (default: mcp-server)
- `-v, --verbose`: Print detailed progress

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
- **llm_judge**: Uses Claude to semantically evaluate if the response is correct

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
| `time_to_answer` | Response time in seconds for the MCP server call |
| `tools_called` | List of MCP tools invoked during the response |

The `tools_called` array contains objects with:
- `tool_name`: Name of the MCP tool called
- `server_name`: Name of the MCP server that provided the tool
- `input`: Parameters passed to the tool

## Requirements

- Python 3.10+
- Anthropic API key with MCP beta access
