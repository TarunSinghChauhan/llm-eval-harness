import json
from dataclasses import dataclass
from datetime import datetime

import httpx

from src.core.config import get_settings
from src.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


@dataclass
class RegressionAlert:
    model: str
    metric: str
    baseline_value: float
    current_value: float
    delta: float
    threshold: float
    severity: str  # warning | critical
    run_id: str
    baseline_run_id: str
    detected_at: str


class RegressionDetector:
    """
    Detects metric regressions between eval runs.
    Fires alerts when a metric drops beyond the configured threshold.
    """

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold or settings.regression_threshold

    def detect(
        self,
        current_metrics: dict[str, dict[str, float]],
        baseline_metrics: dict[str, dict[str, float]],
        run_id: str,
        baseline_run_id: str,
    ) -> list[RegressionAlert]:
        """
        current_metrics:  {model: {metric: mean_value}}
        baseline_metrics: {model: {metric: mean_value}}
        """
        alerts = []

        for model in current_metrics:
            if model not in baseline_metrics:
                continue

            for metric, current_val in current_metrics[model].items():
                baseline_val = baseline_metrics[model].get(metric)
                if baseline_val is None:
                    continue

                delta = current_val - baseline_val

                # Regression = metric got worse (lower is worse for accuracy metrics)
                if delta < -self.threshold:
                    severity = "critical" if abs(delta) > self.threshold * 2 else "warning"
                    alert = RegressionAlert(
                        model=model,
                        metric=metric,
                        baseline_value=round(baseline_val, 4),
                        current_value=round(current_val, 4),
                        delta=round(delta, 4),
                        threshold=self.threshold,
                        severity=severity,
                        run_id=run_id,
                        baseline_run_id=baseline_run_id,
                        detected_at=datetime.utcnow().isoformat(),
                    )
                    alerts.append(alert)
                    logger.warning(
                        "regression_detected",
                        model=model,
                        metric=metric,
                        delta=delta,
                        severity=severity,
                    )

        return alerts

    async def send_slack_alert(self, alerts: list[RegressionAlert]) -> None:
        if not alerts or not settings.slack_webhook_url:
            return

        critical = [a for a in alerts if a.severity == "critical"]
        warnings = [a for a in alerts if a.severity == "warning"]

        emoji = "🚨" if critical else "⚠️"
        lines = [f"{emoji} *LLM Eval Regression Detected*\n"]

        for alert in alerts:
            icon = "🔴" if alert.severity == "critical" else "🟡"
            lines.append(
                f"{icon} `{alert.model}` | `{alert.metric}` | "
                f"{alert.baseline_value:.4f} → {alert.current_value:.4f} "
                f"(Δ {alert.delta:+.4f})"
            )

        lines.append(f"\n_Run ID: {alerts[0].run_id}_")
        lines.append(f"_Baseline: {alerts[0].baseline_run_id}_")

        payload = {"text": "\n".join(lines)}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(settings.slack_webhook_url, json=payload, timeout=5)
                resp.raise_for_status()
            logger.info("slack_alert_sent", n_alerts=len(alerts))
        except Exception as e:
            logger.error("slack_alert_failed", error=str(e))
