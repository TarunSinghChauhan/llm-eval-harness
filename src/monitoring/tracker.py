import mlflow
import json
from pathlib import Path

from src.core.config import get_settings
from src.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class EvalTracker:
    """MLflow experiment tracking for eval runs."""

    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    def log_run(
        self,
        run_name: str,
        models: list[str],
        dataset_name: str,
        dataset_version: str,
        metrics_by_model: dict,
        adversarial_summary: dict | None = None,
        run_id: str | None = None,
    ) -> str:
        """Log a complete eval run to MLflow."""
        with mlflow.start_run(run_name=run_name) as run:
            # Parameters
            mlflow.log_params({
                "models": ",".join(models),
                "dataset": dataset_name,
                "dataset_version": dataset_version,
                "n_models": len(models),
                "run_id": run_id or run.info.run_id,
            })

            # Metrics per model
            for model, metrics in metrics_by_model.items():
                safe_model = model.replace("-", "_").replace(".", "_")
                for metric_name, result in metrics.items():
                    mlflow.log_metrics({
                        f"{safe_model}_{metric_name}_mean": result.mean,
                        f"{safe_model}_{metric_name}_ci_lower": result.ci_lower,
                        f"{safe_model}_{metric_name}_ci_upper": result.ci_upper,
                        f"{safe_model}_{metric_name}_std": result.std,
                    })

            # Adversarial summary
            if adversarial_summary:
                mlflow.log_metrics({
                    "adversarial_safety_rate": adversarial_summary.get("safety_rate", 0),
                    "adversarial_unsafe_count": adversarial_summary.get("unsafe_count", 0),
                })

            # Save full results as artifact
            results_path = Path("results") / f"{run_id}_metrics.json"
            results_path.parent.mkdir(exist_ok=True)
            with open(results_path, "w") as f:
                json.dump({
                    "metrics": {
                        model: {
                            k: {"mean": v.mean, "std": v.std, "ci_lower": v.ci_lower, "ci_upper": v.ci_upper}
                            for k, v in metrics.items()
                        }
                        for model, metrics in metrics_by_model.items()
                    },
                    "adversarial": adversarial_summary,
                }, f, indent=2)
            mlflow.log_artifact(str(results_path))

            mlflow_run_id = run.info.run_id
            logger.info("mlflow_run_logged", mlflow_run_id=mlflow_run_id)
            return mlflow_run_id
