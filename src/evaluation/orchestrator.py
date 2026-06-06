import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.evaluation.runner import ModelRunner
from src.evaluation.dataset_registry import DatasetRegistry
from src.evaluation.scorers.metrics import MetricScorer, ScoredPair
from src.evaluation.judge.llm_judge import LLMJudge
from src.evaluation.adversarial.red_team import AdversarialTester
from src.monitoring.regression import RegressionDetector
from src.monitoring.tracker import EvalTracker
from src.core.logging import get_logger

logger = get_logger(__name__)


class EvalOrchestrator:
    """
    Orchestrates a full evaluation run:
    1. Load dataset
    2. Run models in parallel
    3. Score with ROUGE-L, BERTScore, exact match
    4. Judge with LLM ensemble
    5. Red-team adversarial testing
    6. Detect regressions vs baseline
    7. Log everything to MLflow
    8. Save results to disk
    """

    def __init__(self):
        self.runner = ModelRunner()
        self.registry = DatasetRegistry()
        self.scorer = MetricScorer()
        self.judge = LLMJudge()
        self.adversarial = AdversarialTester()
        self.detector = RegressionDetector()
        self.tracker = EvalTracker()

    async def run(
        self,
        run_id: str,
        run_name: str,
        models: list[str],
        dataset_name: str = "mmlu_sample",
        dataset_version: str = "v1",
        include_judge: bool = True,
        include_adversarial: bool = True,
        baseline_run_id: str | None = None,
    ) -> dict:
        logger.info("eval_run_started", run_id=run_id, models=models, dataset=dataset_name)
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # ── 1. Load dataset ───────────────────────────────────────────────────
        dataset = self.registry.load(dataset_name, dataset_version)
        prompts = dataset["prompts"]
        logger.info("dataset_loaded", n_prompts=len(prompts))

        # ── 2. Run all models ─────────────────────────────────────────────────
        model_responses = await self.runner.run_all_models(models, prompts)

        # ── 3. Score each model ───────────────────────────────────────────────
        metrics_by_model = {}
        scored_by_model = {}

        for model, responses in model_responses.items():
            resp_map = {r.prompt_id: r for r in responses}
            predictions, references, ids = [], [], []

            for p in prompts:
                if p["id"] in resp_map:
                    predictions.append(resp_map[p["id"]].response_text)
                    references.append(p["reference"])
                    ids.append(p["id"])

            scored_pairs = self.scorer.score_all(predictions, references, ids)
            scored_by_model[model] = scored_pairs
            metrics_by_model[model] = self.scorer.aggregate(scored_pairs)

            logger.info(
                "model_scored",
                model=model,
                rouge_l=metrics_by_model[model]["rouge_l"].mean,
                bert_score=metrics_by_model[model]["bert_score"].mean,
                exact_match=metrics_by_model[model]["exact_match"].mean,
            )

        # ── 4. LLM-as-judge ───────────────────────────────────────────────────
        judge_results_by_model = {}
        kappa_by_model = {}
        if include_judge:
            for model, responses in model_responses.items():
                resp_map = {r.prompt_id: r for r in responses}
                items = [
                    {
                        "prompt_id": p["id"],
                        "model_judged": model,
                        "question": p["prompt"],
                        "reference": p["reference"],
                        "response": resp_map[p["id"]].response_text,
                    }
                    for p in prompts if p["id"] in resp_map
                ]
                judge_results = await self.judge.judge_batch(items)
                judge_results_by_model[model] = judge_results

                gpt_scores = [r.gpt_score for r in judge_results if r.gpt_score is not None]
                claude_scores = [r.claude_score for r in judge_results if r.claude_score is not None]
                if gpt_scores and claude_scores:
                    kappa = self.judge.cohens_kappa(gpt_scores, claude_scores)
                    kappa_by_model[model] = kappa
                    logger.info("judge_agreement", model=model, cohens_kappa=kappa)

        # ── 5. Adversarial red-teaming ────────────────────────────────────────
        adversarial_by_model = {}
        if include_adversarial:
            for model in models:
                adv_results = await self.adversarial.test_model(model)
                adversarial_by_model[model] = self.adversarial.summarize(adv_results)

        # ── 6. Regression detection ───────────────────────────────────────────
        regression_alerts = []
        if baseline_run_id:
            baseline_path = results_dir / f"{baseline_run_id}_metrics.json"
            if baseline_path.exists():
                with open(baseline_path) as f:
                    baseline_data = json.load(f)
                current_means = {
                    model: {k: v.mean for k, v in metrics.items()}
                    for model, metrics in metrics_by_model.items()
                }
                baseline_means = {
                    model: {k: v["mean"] for k, v in metrics.items()}
                    for model, metrics in baseline_data.get("metrics", {}).items()
                }
                regression_alerts = self.detector.detect(current_means, baseline_means, run_id, baseline_run_id)
                if regression_alerts:
                    await self.detector.send_slack_alert(regression_alerts)

        # ── 7. Log to MLflow ──────────────────────────────────────────────────
        adv_summary = adversarial_by_model.get(models[0]) if adversarial_by_model else None
        mlflow_run_id = self.tracker.log_run(
            run_name=run_name,
            models=models,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            metrics_by_model=metrics_by_model,
            adversarial_summary=adv_summary,
            run_id=run_id,
        )

        # ── 8. Build and save full results ────────────────────────────────────
        output = {
            "run_id": run_id,
            "run_name": run_name,
            "mlflow_run_id": mlflow_run_id,
            "dataset": dataset_name,
            "dataset_version": dataset_version,
            "n_prompts": len(prompts),
            "models": models,
            "completed_at": datetime.utcnow().isoformat(),
            "metrics": {
                model: {
                    k: {
                        "mean": v.mean,
                        "std": v.std,
                        "ci_lower": v.ci_lower,
                        "ci_upper": v.ci_upper,
                        "n_samples": v.n_samples,
                    }
                    for k, v in metrics.items()
                }
                for model, metrics in metrics_by_model.items()
            },
            "judge_kappa": kappa_by_model,
            "adversarial": adversarial_by_model,
            "regressions": [
                {
                    "model": a.model,
                    "metric": a.metric,
                    "delta": a.delta,
                    "severity": a.severity,
                }
                for a in regression_alerts
            ],
        }

        output_path = results_dir / f"{run_id}_full_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info("eval_run_complete", run_id=run_id, output_path=str(output_path))
        return output

    async def close(self):
        await self.runner.close()
