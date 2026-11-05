# LLM Evaluation Harness & Red-Teaming Framework

A framework for evaluating language models: async multi-model inference, automatic scoring with statistical confidence intervals, an LLM-as-judge ensemble with inter-rater agreement, adversarial red-team prompts, and regression alerts when quality drops.

**Try the scoring logic in your browser:** [Scoring Playground](https://tarunsinghchauhan.github.io/llm-eval-harness/)
It runs ROUGE-L, exact match, bootstrapped 95% confidence intervals and the regression check entirely client-side. It is a demo of the math, **not** a benchmark of real models.

## What it does

- **Evaluation runner:** sends the same prompts to multiple models concurrently and collects the outputs.
- **Scoring:** ROUGE-L, BERTScore and exact match, each reported with a percentile bootstrap 95% confidence interval.
- **LLM-as-judge ensemble:** two judge models label each output independently, and Cohen's κ measures how much they agree.
- **Red-teaming:** a suite of 20 adversarial prompt patterns (prompt injection, jailbreaks, data extraction, harmful content) with a safety rate per model.
- **Regression detection:** compares a candidate run to a stored baseline and flags any metric that drops by more than a threshold (default 2%).
- **Tracking:** runs are logged to MLflow, and traces go to LangSmith.
- **Stack:** Python, FastAPI, Docker, Redis, PostgreSQL, GitHub Actions CI.

## What is verified, and what isn't

This table is kept honest on purpose.

| Item | Status |
| --- | --- |
| Scoring logic (ROUGE-L, exact match, bootstrap CI, regression check) | Verifiable now in the [Scoring Playground](https://tarunsinghchauhan.github.io/llm-eval-harness/) |
| Reproducible small benchmark (50 MMLU questions, two models, two judges, κ) | Run with `eval_run.py`; output in `results/summary.json` once committed |
| Red-team suite results (safety rate per model) | Suite implemented; a committed results file is pending |
| CI runtime | See the duration of runs in the repo's Actions tab |

Nothing in this README claims a result that isn't backed by a file in the repo.

## Reproducible benchmark (`eval_run.py`)

A standalone script that anyone can run, with a fixed seed.

```bash
pip install datasets openai numpy
export GROQ_API_KEY=your_key        # Windows PowerShell: $env:GROQ_API_KEY="your_key"
python eval_run.py
```

What it does:

1. Samples 50 questions from MMLU (`cais/mmlu`) with a fixed seed.
2. Asks each model for an answer letter and scores exact match with a bootstrapped 95% CI (1,000 resamples).
3. Has two judge models label every answer CORRECT or INCORRECT **without seeing the reference**, then reports Cohen's κ between the judges and each judge's accuracy against the ground truth.
4. Writes everything, including model IDs, seed and date, to `results/summary.json`.

It caches calls in `results/cache.json`, so an interrupted run resumes instead of repeating work. Any OpenAI-compatible provider works through `--base-url`, `--key-env`, `--models` and `--judges`. It defaults to Groq's free tier.

### Results

Filled in from `results/summary.json` after a run.

| Model | Exact match (95% CI) | n |
| --- | --- | --- |
| _model 1_ | _fill from summary.json_ | 50 |
| _model 2_ | _fill from summary.json_ | 50 |

| Judge agreement | Value |
| --- | --- |
| Cohen's κ between judges | _fill from summary.json_ |
| Items labelled by both judges | _fill from summary.json_ |

**Caveats**

- With 50 questions, confidence intervals are wide (roughly ±13 points for an accuracy near 60%). Treat the numbers as a demonstration of the method, not a leaderboard.
- By default the judges are the same models being evaluated, which can bias their judgments. Use a separate judge model with `--judges` when you have one.

## Metric choice

ROUGE-L and BERTScore measure text overlap and semantic similarity, so they suit free-text outputs such as summaries. MMLU answers are single letters, where ROUGE-L reduces to exact match and BERTScore says very little. For MMLU the meaningful metric is exact match, and this repo uses ROUGE-L and BERTScore for the free-text evaluation paths.

## Full pipeline (Docker)

```bash
cp .env.example .env     # add your API keys
docker compose up --build
```

MLflow is available at <http://localhost:5001> and the FastAPI docs at `/docs` on the API service. The CI workflow in `.github/workflows/ci.yml` runs the tests on every push.

## Statistical method

Confidence intervals use the percentile bootstrap: resample the per-item scores with replacement 1,000 times (seed 42), take the mean of each resample, and report the 2.5th and 97.5th percentiles. Regression detection flags a metric when the candidate mean falls below the baseline mean by more than the configured threshold.

## Limitations

- Small evaluation samples give wide confidence intervals.
- LLM judges can disagree with ground truth, and a judge that is also an evaluated model can favor its own style.
- Pattern-based red-team prompts cover known attack styles and don't exhaust what an attacker could try.
- Free-tier providers are rate-limited, so large runs take time.

## Roadmap

- [ ] Commit a reference `results/summary.json` and a red-team results file
- [ ] Add a separate third-party judge model
- [ ] Larger MMLU sample, with a per-subject breakdown
- [ ] Publish CI run time in the README from a measured run

## License

MIT
