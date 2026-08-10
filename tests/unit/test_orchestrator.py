from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.orchestrator import EvalOrchestrator


def make_response(prompt_id, text):
    return SimpleNamespace(prompt_id=prompt_id, response_text=text)


@pytest.mark.asyncio
@patch("src.evaluation.orchestrator.EvalTracker")
@patch("src.evaluation.orchestrator.RegressionDetector")
@patch("src.evaluation.orchestrator.AdversarialTester")
@patch("src.evaluation.orchestrator.LLMJudge")
@patch("src.evaluation.orchestrator.ModelRunner")
async def test_run_skips_regression_when_baseline_file_missing(
    mock_runner_cls, mock_judge_cls, mock_adv_cls, mock_detector_cls, mock_tracker_cls,
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    mock_runner = mock_runner_cls.return_value
    mock_runner.run_all_models = AsyncMock(return_value={
        "gpt-4o-mini": [make_response("r001", "150 miles")]
    })

    mock_tracker = mock_tracker_cls.return_value
    mock_tracker.log_run = MagicMock(return_value="mlflow_run_1")

    orchestrator = EvalOrchestrator()

    result = await orchestrator.run(
        run_id="run_test_1",
        run_name="test-run",
        models=["gpt-4o-mini"],
        dataset_name="reasoning",
        dataset_version="v1",
        include_judge=False,
        include_adversarial=False,
        baseline_run_id="nonexistent_baseline",
    )

    assert result["regressions"] == []
    mock_detector_cls.return_value.detect.assert_not_called()

    output_path = tmp_path / "results" / "run_test_1_full_results.json"
    assert output_path.exists()
    assert result["mlflow_run_id"] == "mlflow_run_1"
