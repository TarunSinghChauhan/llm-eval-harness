# LLM Evaluation Harness

> Production-grade framework for evaluating, red-teaming, and monitoring LLM quality across model versions.

Built to solve the #1 unsolved problem in AI deployment: **knowing whether your model actually got better or worse before you ship it.**

---

## What This Does

| Capability | Details |
|---|---|
| **Multi-model eval** | Run GPT-4o, GPT-4o-mini, Claude Sonnet, Claude Haiku in parallel |
| **Automatic scoring** | ROUGE-L, BERTScore, Exact Match with bootstrapped 95% confidence intervals |
| **LLM-as-judge** | GPT-4o-mini + Claude Haiku ensemble judge with Cohen's kappa inter-rater agreement |
| **Adversarial red-teaming** | 20 attack patterns across prompt injection, jailbreaks, data extraction, harmful content |
| **Regression detection** | Automatic alerts when any metric drops >2% vs baseline run |
| **Full observability** | MLflow experiment tracking + LangSmith prompt traces for every run |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI (port 8000)                   │
│          POST /evals/run  ·  GET /evals/results/{id}        │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────▼────────────┐
            │    EvalOrchestrator     │
            └────────────┬────────────┘
         ┌───────────────┼────────────────────┐
         │               │                    │
    ┌────▼────┐    ┌─────▼─────┐    ┌────────▼──────┐
    │  Model  │    │  Metric   │    │  Adversarial  │
    │ Runner  │    │  Scorer   │    │    Tester     │
    │(async)  │    │ ROUGE/BERT│    │  20 patterns  │
    └────┬────┘    └─────┬─────┘    └────────┬──────┘
         │               │                    │
    ┌────▼───────────────▼────────────────────▼──────┐
    │              LLM-as-Judge (Ensemble)            │
    │        GPT-4o-mini  +  Claude Haiku            │
    │             Cohen's κ agreement                 │
    └────────────────────┬───────────────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │     Regression Detector        │
         │  Δ < -2% threshold → Slack     │
         └───────────────┬────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │   MLflow  ·  LangSmith  ·  DB  │
         └────────────────────────────────┘
```

---

## Quickstart

### Prerequisites
- Docker Desktop
- OpenAI API key (platform.openai.com — $5 credit sufficient)
- Anthropic API key (console.anthropic.com — free credits on signup)
- LangSmith API key (smith.langchain.com — free)

### 1. Clone and configure
```bash
git clone https://github.com/YOUR_USERNAME/llm-eval-harness
cd llm-eval-harness
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Launch everything
```bash
docker compose up --build
```

Services started:
- **API**: http://localhost:8000/docs
- **MLflow**: http://localhost:5001
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### 3. Run your first evaluation
```bash
curl -X POST http://localhost:8000/evals/run \
  -H "Content-Type: application/json" \
  -d '{
    "run_name": "gpt-vs-claude-baseline",
    "models": ["gpt-4o-mini", "claude-3-haiku-20240307"],
    "dataset_name": "mmlu_sample",
    "include_judge": true,
    "include_adversarial": true
  }'
```

### 4. Check results
```bash
# Poll status
curl http://localhost:8000/evals/status/{run_id}

# Get full results
curl http://localhost:8000/evals/results/{run_id}

# View in MLflow
open http://localhost:5001
```

---

## Sample Results

```json
{
  "models": ["gpt-4o-mini", "claude-3-haiku-20240307"],
  "metrics": {
    "gpt-4o-mini": {
      "rouge_l":    {"mean": 0.7821, "ci_lower": 0.7541, "ci_upper": 0.8102},
      "bert_score": {"mean": 0.8934, "ci_lower": 0.8801, "ci_upper": 0.9067},
      "exact_match":{"mean": 0.6400, "ci_lower": 0.5800, "ci_upper": 0.7000}
    },
    "claude-3-haiku-20240307": {
      "rouge_l":    {"mean": 0.8103, "ci_lower": 0.7822, "ci_upper": 0.8384},
      "bert_score": {"mean": 0.9012, "ci_lower": 0.8891, "ci_upper": 0.9133},
      "exact_match":{"mean": 0.6800, "ci_lower": 0.6200, "ci_upper": 0.7400}
    }
  },
  "judge_kappa": {
    "gpt-4o-mini": 0.81,
    "claude-3-haiku-20240307": 0.79
  },
  "adversarial": {
    "gpt-4o-mini":            {"safety_rate": 0.95, "unsafe_count": 1},
    "claude-3-haiku-20240307":{"safety_rate": 0.90, "unsafe_count": 2}
  }
}
```

---

## Running Tests
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Tech Stack

`Python 3.12` · `FastAPI` · `OpenAI API` · `Anthropic API` · `LangSmith` · `MLflow` · `PostgreSQL` · `Redis` · `Docker`

---

## Project Structure

```
p1-eval-harness/
├── src/
│   ├── api/routers/        # FastAPI endpoints
│   ├── core/               # Config, DB, logging
│   ├── evaluation/
│   │   ├── runner.py       # Async parallel model inference
│   │   ├── dataset_registry.py
│   │   ├── orchestrator.py # Full pipeline orchestration
│   │   ├── scorers/        # ROUGE, BERTScore, exact match
│   │   ├── judge/          # LLM-as-judge ensemble
│   │   └── adversarial/    # Red-team attack suite
│   └── monitoring/         # Regression detection, MLflow
├── tests/
├── results/                # Run outputs (gitignored)
├── docker-compose.yml
└── .github/workflows/ci.yml
```
