import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.monitoring.regression import RegressionDetector


def test_severity_is_warning_just_above_threshold():
    detector = RegressionDetector(threshold=0.02)
    current = {"gpt-4o-mini": {"rouge_l": 0.729}}
    baseline = {"gpt-4o-mini": {"rouge_l": 0.75}}
    alerts = detector.detect(current, baseline, "run_002", "run_001")
    assert len(alerts) == 1
    assert alerts[0].severity == "warning"


def test_severity_is_critical_when_delta_exceeds_double_threshold():
    detector = RegressionDetector(threshold=0.02)
    current = {"gpt-4o-mini": {"rouge_l": 0.65}}
    baseline = {"gpt-4o-mini": {"rouge_l": 0.75}}
    alerts = detector.detect(current, baseline, "run_002", "run_001")
    assert len(alerts) == 1
    assert alerts[0].severity == "critical"


def test_detect_ignores_model_not_in_baseline():
    detector = RegressionDetector(threshold=0.02)
    current = {"new-model": {"rouge_l": 0.5}}
    baseline = {"gpt-4o-mini": {"rouge_l": 0.75}}
    alerts = detector.detect(current, baseline, "run_002", "run_001")
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_send_slack_alert_noop_when_no_webhook(monkeypatch):
    from src.monitoring import regression
    monkeypatch.setattr(regression.settings, "slack_webhook_url", None)
    detector = RegressionDetector(threshold=0.02)

    with patch("src.monitoring.regression.httpx.AsyncClient") as mock_client:
        await detector.send_slack_alert([])
        mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_send_slack_alert_noop_when_no_alerts(monkeypatch):
    from src.monitoring import regression
    monkeypatch.setattr(regression.settings, "slack_webhook_url", "https://hooks.slack.test/x")
    detector = RegressionDetector(threshold=0.02)

    with patch("src.monitoring.regression.httpx.AsyncClient") as mock_client:
        await detector.send_slack_alert([])
        mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_send_slack_alert_posts_payload_when_alerts_exist(monkeypatch):
    from src.monitoring import regression
    monkeypatch.setattr(regression.settings, "slack_webhook_url", "https://hooks.slack.test/x")
    detector = RegressionDetector(threshold=0.02)

    current = {"gpt-4o-mini": {"rouge_l": 0.65}}
    baseline = {"gpt-4o-mini": {"rouge_l": 0.75}}
    alerts = detector.detect(current, baseline, "run_002", "run_001")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)

    with patch("src.monitoring.regression.httpx.AsyncClient", return_value=mock_async_client):
        await detector.send_slack_alert(alerts)

    mock_async_client.__aenter__.return_value.post.assert_called_once()
    call_kwargs = mock_async_client.__aenter__.return_value.post.call_args
    assert "gpt-4o-mini" in call_kwargs.kwargs["json"]["text"]
