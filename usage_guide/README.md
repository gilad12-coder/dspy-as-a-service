# Usage Guide

This guide is for **clients** who want to submit optimization jobs to the DSPy service. For service operators, see the [main README](../README.md).

## Contents
- [Prerequisites](#prerequisites)
- [Notebooks](#notebooks)
- [Datasets](#datasets)
- [GEPA Optimizer](#gepa-optimizer)
- [API Client Classes](#api-client-classes)
- [Typical Workflow](#typical-workflow)

## Prerequisites
- The DSPy service running (locally via `uv run python main.py` or deployed)
- Python environment with `dspy` and `requests` installed

### Quick Setup with `uv`
```bash
cd usage_guide
uv venv .venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

## Notebooks

| Notebook | Dataset | Description |
|----------|---------|-------------|
| `hotpotqa_gepa_optimization.ipynb` | HotPotQA | Multi-hop QA requiring reasoning over multiple facts. GEPA improves accuracy from ~24% to ~51%. |
| `gsm8k_gepa_optimization.ipynb` | GSM8K | Grade school math word problems. GEPA achieves up to 93% accuracy through reflective prompt evolution. |

Both notebooks demonstrate the complete optimization workflow using the GEPA optimizer, from job submission to loading and using the optimized program.

## Datasets

| File | Source | Description |
|------|--------|-------------|
| `data/hotpotqa.json` | [HotPotQA](https://hotpotqa.github.io/) | 100 multi-hop QA examples requiring reasoning over multiple Wikipedia facts |
| `data/gsm8k.json` | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 100 grade school math word problems with step-by-step solutions |

Load datasets directly:
```python
import json
with open("data/hotpotqa.json") as f:
    dataset = json.load(f)
```

## GEPA Optimizer

[GEPA](https://dspy.ai/api/optimizers/GEPA) (Genetic-Pareto) is an advanced optimizer that uses reflective prompt evolution to optimize DSPy programs. Unlike traditional optimizers that rely only on numerical scores, GEPA analyzes execution traces and uses an LLM to understand *why* certain prompts work better, then evolves improved versions. Key characteristics:

**Metric Requirements:**
GEPA metrics must accept five arguments and return a `dspy.Prediction` with `score` and `feedback`:

```python
def gepa_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None,
                pred_name=None, pred_trace=None) -> dspy.Prediction:
    # Compare gold.answer with pred.answer
    if gold.answer == pred.answer:
        return dspy.Prediction(score=1.0, feedback="Correct answer.")
    else:
        feedback = "Incorrect. Expected '" + str(gold.answer) + "', got '" + str(pred.answer) + "'."
        return dspy.Prediction(score=0.0, feedback=feedback)
```

**Configuration:**
```python
payload = {
    "optimizer_name": "dspy.GEPA",
    "optimizer_kwargs": {
        "auto": "heavy",  # Options: "light", "medium", "heavy" (recommended)
    },
    "compile_kwargs": {},
    "reflection_model_config": {  # Required for GEPA - the LLM that analyzes failures
        "name": "openai/gpt-4o-mini",
        "temperature": 1.0,  # Higher temperature for creative reflection
        "max_tokens": 20000,
    },
}
```

**GEPA Parameters:**
| Parameter | Description |
|-----------|-------------|
| `auto` | Budget level: `"light"` (fast), `"medium"` (balanced), `"heavy"` (best results) |
| `max_full_evals` | Alternative to `auto`: explicit max number of rollouts |
| `use_merge` | Enable merge-based optimization (default: `True`) |
| `num_threads` | Parallel evaluation threads (e.g., `32`) |

## API Client Classes

Each notebook includes reusable client classes:

### DSPyServiceClient

HTTP client for all API operations:

```python
client = DSPyServiceClient("http://localhost:8000")

# Health check
client.health()  # Returns {"status": "ok", "registered_assets": {...}}

# Submit optimization job
job_id = client.submit(payload)

# Poll job status
status = client.status(job_id)

# Get job summary (lightweight)
summary = client.summary(job_id)

# Get full logs
logs = client.logs(job_id)

# Download artifact
artifact = client.artifact(job_id)

# Load optimized program directly
program = client.load_program(job_id)
response = program(question="What is 2+2?")
```

### JobMonitor

Real-time progress monitoring with formatted output:

```python
monitor = JobMonitor(client, job_id)
result = monitor.poll(interval=3)  # Streams progress until completion
```

Output includes:
- Job status transitions (pending → running → completed/failed)
- Progress percentage and ETA
- Baseline and optimized metrics
- DSPy evaluation logs

### serialize_source

Helper to serialize DSPy signatures and metrics for the API:

```python
class QASignature(dspy.Signature):
    """Answer questions accurately."""
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="The answer")

payload = {
    "signature_code": serialize_source(QASignature),
    "metric_code": serialize_source(my_metric),
    # ...
}
```

## Typical Workflow

1. **Start the service**
   ```bash
   uv run python main.py
   ```

2. **Open a notebook** (Jupyter, VS Code, or export via `nbconvert`)

3. **Configure the client**
   ```python
   BASE_URL = "http://localhost:8000"
   client = DSPyServiceClient(BASE_URL)
   client.health()  # Verify connection
   ```

4. **Define signature and metric**
   ```python
   class QASignature(dspy.Signature):
       """Answer multi-hop questions by reasoning over facts."""
       question: str = dspy.InputField(desc="The question")
       answer: str = dspy.OutputField(desc="The answer")

   def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
       correct = gold.answer.lower() == pred.answer.lower()
       return dspy.Prediction(
           score=1.0 if correct else 0.0,
           feedback="Correct!" if correct else f"Expected {gold.answer}"
       )
   ```

5. **Build and submit a job**
   ```python
   payload = {
       "module_name": "dspy.ChainOfThought",
       "optimizer_name": "dspy.GEPA",
       "signature_code": serialize_source(QASignature),
       "metric_code": serialize_source(metric),
       "dataset": dataset,
       "column_mapping": {"question": "question", "answer": "answer"},
       "model_config": MODEL_CONFIG,
       "reflection_model_config": MODEL_CONFIG,  # Required for GEPA
       "optimizer_kwargs": {"auto": "heavy"},  # "light", "medium", or "heavy"
       "compile_kwargs": {},
       "split_fractions": {"train": 0.5, "val": 0.3, "test": 0.2},
       "shuffle": True,
       "seed": 42,
   }
   job_id = client.submit(payload)
   ```

6. **Monitor progress**
   ```python
   monitor = JobMonitor(client, job_id)
   result = monitor.poll(interval=3)
   ```

7. **Retrieve and use the optimized program**
   ```python
   program = client.load_program(job_id)
   response = program(question="Who wrote Romeo and Juliet?")
   print(response.answer)
   ```
