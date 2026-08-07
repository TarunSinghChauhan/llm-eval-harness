from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json

from src.monitoring.tracker import EvalTracker


def make_metric_result(mean, std, ci_lower, ci_upper):
    return SimpleNamespace(mean=mean, std=std, ci_lower=ci_lower, ci_upper=ci_upper)


def make_tracker(mock_mlflow, run_id):
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    return EvalTracker()


@patch("src.monitoring.tracker.mlflow")
def test_log_run_sanitizes_model_names_with_dashes_and_dots(mock_mlflow, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = make_tracker(mock_mlflow, "run_123")

    metrics_by_model = {
        "gpt-4o-mini": {"rouge_l": make_metric_result(0.8, 0.05, 0.75, 0.85)}
    }

    tracker.log_run(
        run_name="test-run",
        models=["gpt-4o-mini"],
        dataset_name="mmlu_sample",
        dataset_version="v1",
        metrics_by_model=metrics_by_model,
        run_id="run_123",
    )

    logged_metrics = mock_mlflow.log_metrics.call_args_list[0][0][0]
    assert "gpt_4o_mini_rouge_l_mean" in logged_metrics
    assert logged_metrics["gpt_4o_mini_rouge_l_mean"] == 0.8


@patch("src.monitoring.tracker.mlflow")
def test_log_run_skips_adversarial_metrics_when_none(mock_mlflow, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = make_tracker(mock_mlflow, "run_456")

    metrics_by_model = {
        "gpt-4o-mini": {"rouge_l": make_metric_result(0.8, 0.05, 0.75, 0.85)}
    }

    tracker.log_run(
        run_name="test-run",
        models=["gpt-4o-mini"],
        dataset_name="mmlu_sample",
        dataset_version="v1",
        metrics_by_model=metrics_by_model,
        adversarial_summary=None,
        run_id="run_456",
    )

    assert mock_mlflow.log_metrics.call_count == 1


@patch("src.monitoring.tracker.mlflow")
def test_log_run_writes_results_json_with_expected_structure(mock_mlflow, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = make_tracker(mock_mlflow, "run_789")

    metrics_by_model = {
        "gpt-4o-mini": {"rouge_l": make_metric_result(0.8, 0.05, 0.75, 0.85)}
    }

    tracker.log_run(
        run_name="test-run",
        models=["gpt-4o-mini"],
        dataset_name="mmlu_sample",
        dataset_version="v1",
        metrics_by_model=metrics_by_model,
        run_id="run_789",
    )

    results_path = tmp_path / "results" / "run_789_metrics.json"
    assert results_path.exists()

    with open(results_path) as f:
        data = json.load(f)

    assert data["metrics"]["gpt-4o-mini"]["rouge_l"]["mean"] == 0.8


@patch("src.monitoring.tracker.mlflow")
def test_log_run_returns_mlflow_run_id(mock_mlflow, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = make_tracker(mock_mlflow, "abc-123")

    metrics_by_model = {
        "gpt-4o-mini": {"rouge_l": make_metric_result(0.8, 0.05, 0.75, 0.85)}
    }

    result = tracker.log_run(
        run_name="test-run",
        models=["gpt-4o-mini"],
        dataset_name="mmlu_sample",
        dataset_version="v1",
        metrics_by_model=metrics_by_model,
        run_id="abc-123",
    )

    assert result == "abc-123"
