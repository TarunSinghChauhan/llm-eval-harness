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


@pytest.mark.asyncio
@patch("src.evaluation.orchestrator.EvalTracker")
@patch("src.evaluation.orchestrator.RegressionDetector")
@patch("src.evaluation.orchestrator.AdversarialTester")
@patch("src.evaluation.orchestrator.LLMJudge")
@patch("src.evaluation.orchestrator.ModelRunner")
async def test_run_includes_judge_results_and_kappa_when_both_scores_present(
    mock_runner_cls, mock_judge_cls, mock_adv_cls, mock_detector_cls, mock_tracker_cls,
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    mock_runner = mock_runner_cls.return_value
    mock_runner.run_all_models = AsyncMock(return_value={
        "gpt-4o-mini": [make_response("r001", "150 miles")]
    })

    mock_judge = mock_judge_cls.return_value
    fake_judge_result = SimpleNamespace(gpt_score=8.0, claude_score=6.0)
    mock_judge.judge_batch = AsyncMock(return_value=[fake_judge_result])
    mock_judge.cohens_kappa = MagicMock(return_value=0.75)

    mock_tracker = mock_tracker_cls.return_value
    mock_tracker.log_run = MagicMock(return_value="mlflow_run_2")

    orchestrator = EvalOrchestrator()

    result = await orchestrator.run(
        run_id="run_test_2",
        run_name="test-run",
        models=["gpt-4o-mini"],
        dataset_name="reasoning",
        dataset_version="v1",
        include_judge=True,
        include_adversarial=False,
    )

    assert result["judge_kappa"]["gpt-4o-mini"] == 0.75
    mock_judge.judge_batch.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.evaluation.orchestrator.EvalTracker")
@patch("src.evaluation.orchestrator.RegressionDetector")
@patch("src.evaluation.orchestrator.AdversarialTester")
@patch("src.evaluation.orchestrator.LLMJudge")
@patch("src.evaluation.orchestrator.ModelRunner")
async def test_run_skips_kappa_when_only_one_judge_scored(
    mock_runner_cls, mock_judge_cls, mock_adv_cls, mock_detector_cls, mock_tracker_cls,
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    mock_runner = mock_runner_cls.return_value
    mock_runner.run_all_models = AsyncMock(return_value={
        "gpt-4o-mini": [make_response("r001", "150 miles")]
    })

    mock_judge = mock_judge_cls.return_value
    fake_judge_result = SimpleNamespace(gpt_score=8.0, claude_score=None)
    mock_judge.judge_batch = AsyncMock(return_value=[fake_judge_result])
    mock_judge.cohens_kappa = MagicMock()

    mock_tracker = mock_tracker_cls.return_value
    mock_tracker.log_run = MagicMock(return_value="mlflow_run_3")

    orchestrator = EvalOrchestrator()

    result = await orchestrator.run(
        run_id="run_test_3",
        run_name="test-run",
        models=["gpt-4o-mini"],
        dataset_name="reasoning",
        dataset_version="v1",
        include_judge=True,
        include_adversarial=False,
    )

    assert "gpt-4o-mini" not in result["judge_kappa"]
    mock_judge.cohens_kappa.assert_not_called()


@pytest.mark.asyncio
@patch("src.evaluation.orchestrator.EvalTracker")
@patch("src.evaluation.orchestrator.RegressionDetector")
@patch("src.evaluation.orchestrator.AdversarialTester")
@patch("src.evaluation.orchestrator.LLMJudge")
@patch("src.evaluation.orchestrator.ModelRunner")
async def test_run_includes_adversarial_summary_when_enabled(
    mock_runner_cls, mock_judge_cls, mock_adv_cls, mock_detector_cls, mock_tracker_cls,
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    mock_runner = mock_runner_cls.return_value
    mock_runner.run_all_models = AsyncMock(return_value={
        "gpt-4o-mini": [make_response("r001", "150 miles")]
    })

    mock_adv = mock_adv_cls.return_value
    mock_adv.test_model = AsyncMock(return_value=["fake_result"])
    mock_adv.summarize = MagicMock(return_value={"safety_rate": 0.95})

    mock_tracker = mock_tracker_cls.return_value
    mock_tracker.log_run = MagicMock(return_value="mlflow_run_4")

    orchestrator = EvalOrchestrator()

    result = await orchestrator.run(
        run_id="run_test_4",
        run_name="test-run",
        models=["gpt-4o-mini"],
        dataset_name="reasoning",
        dataset_version="v1",
        include_judge=False,
        include_adversarial=True,
    )

    assert result["adversarial"]["gpt-4o-mini"] == {"safety_rate": 0.95}
    mock_adv.test_model.assert_awaited_once_with("gpt-4o-mini")


@pytest.mark.asyncio
@patch("src.evaluation.orchestrator.EvalTracker")
@patch("src.evaluation.orchestrator.RegressionDetector")
@patch("src.evaluation.orchestrator.AdversarialTester")
@patch("src.evaluation.orchestrator.LLMJudge")
@patch("src.evaluation.orchestrator.ModelRunner")
async def test_run_detects_and_sends_regression_alert_when_baseline_exists(
    mock_runner_cls, mock_judge_cls, mock_adv_cls, mock_detector_cls, mock_tracker_cls,
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    baseline_data = {
        "metrics": {
            "gpt-4o-mini": {"rouge_l": {"mean": 0.85}}
        }
    }
    with open(results_dir / "baseline_run_full_results.json", "w") as f:
        pass  # placeholder; actual baseline file uses "_metrics.json" suffix per orchestrator.py
    with open(results_dir / "baseline_run_metrics.json", "w") as f:
        import json
        json.dump(baseline_data, f)

    mock_runner = mock_runner_cls.return_value
    mock_runner.run_all_models = AsyncMock(return_value={
        "gpt-4o-mini": [make_response("r001", "150 miles")]
    })

    fake_alert = SimpleNamespace(model="gpt-4o-mini", metric="rouge_l", delta=-0.2, severity="critical")
    mock_detector = mock_detector_cls.return_value
    mock_detector.detect = MagicMock(return_value=[fake_alert])
    mock_detector.send_slack_alert = AsyncMock()

    mock_tracker = mock_tracker_cls.return_value
    mock_tracker.log_run = MagicMock(return_value="mlflow_run_5")

    orchestrator = EvalOrchestrator()

    result = await orchestrator.run(
        run_id="run_test_5",
        run_name="test-run",
        models=["gpt-4o-mini"],
        dataset_name="reasoning",
        dataset_version="v1",
        include_judge=False,
        include_adversarial=False,
        baseline_run_id="baseline_run",
    )

    assert len(result["regressions"]) == 1
    assert result["regressions"][0]["severity"] == "critical"
    mock_detector.send_slack_alert.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.evaluation.orchestrator.EvalTracker")
@patch("src.evaluation.orchestrator.RegressionDetector")
@patch("src.evaluation.orchestrator.AdversarialTester")
@patch("src.evaluation.orchestrator.LLMJudge")
@patch("src.evaluation.orchestrator.ModelRunner")
async def test_close_closes_the_runner(
    mock_runner_cls, mock_judge_cls, mock_adv_cls, mock_detector_cls, mock_tracker_cls,
):
    mock_runner = mock_runner_cls.return_value
    mock_runner.close = AsyncMock()

    orchestrator = EvalOrchestrator()
    await orchestrator.close()

    mock_runner.close.assert_awaited_once()
