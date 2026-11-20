"""A failed BERTScore must be reported as unavailable, never as rouge_l under another name."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.orchestrator import EvalOrchestrator
from src.evaluation.scorers.metrics import MetricScorer, ScoredPair

BERT = "src.evaluation.scorers.metrics.bert_score_fn"


def scorer_with_rouge(value=0.5):
    s = MetricScorer()
    s._rouge = MagicMock()
    s._rouge.score.return_value = {"rougeL": SimpleNamespace(fmeasure=value)}
    return s


def test_a_failure_gives_none_per_pair_and_never_calls_rouge():
    s = scorer_with_rouge()
    with patch(BERT, side_effect=RuntimeError("no model")):
        assert s.bert_score_batch(["a", "b"], ["c", "d"]) == [None, None]
    s._rouge.score.assert_not_called()


def test_success_is_unchanged():
    s = scorer_with_rouge()
    with patch(BERT, return_value=(None, None, [0.812345, 0.55])):
        assert s.bert_score_batch(["a", "b"], ["c", "d"]) == [0.8123, 0.55]


def test_score_all_keeps_the_other_metrics_when_bert_fails():
    s = scorer_with_rouge(0.5)
    with patch(BERT, side_effect=RuntimeError("no model")):
        pairs = s.score_all(["Paris"], ["Paris"], ["p1"])
    assert pairs[0].bert_score is None
    assert pairs[0].rouge_l == 0.5 and pairs[0].exact_match == 1.0


def test_aggregate_leaves_bert_score_out_when_no_pair_has_one():
    out = MetricScorer().aggregate([
        ScoredPair("p1", 0.5, None, 1.0), ScoredPair("p2", 0.7, None, 0.0),
    ], n_iterations=20)
    assert set(out) == {"rouge_l", "exact_match"}
    assert out["rouge_l"].n_samples == 2


def test_aggregate_averages_only_the_pairs_that_have_a_bert_score():
    out = MetricScorer().aggregate([
        ScoredPair("p1", 0.5, 0.5, 1.0), ScoredPair("p2", 0.7, None, 0.0), ScoredPair("p3", 0.6, 0.7, 1.0),
    ], n_iterations=20)
    assert out["bert_score"].n_samples == 2
    assert out["bert_score"].mean == 0.6


def test_aggregate_keeps_the_metric_order():
    out = MetricScorer().aggregate([ScoredPair("p1", 0.5, 0.6, 1.0)], n_iterations=5)
    assert list(out) == ["rouge_l", "bert_score", "exact_match"]


# ---- orchestrator: the run output says what is missing -----------------------------

async def run_eval(tmp_path, monkeypatch, bert_patch):
    monkeypatch.chdir(tmp_path)
    with patch("src.evaluation.orchestrator.EvalTracker") as tracker, \
         patch("src.evaluation.orchestrator.RegressionDetector"), \
         patch("src.evaluation.orchestrator.AdversarialTester"), \
         patch("src.evaluation.orchestrator.LLMJudge"), \
         patch("src.evaluation.orchestrator.ModelRunner") as runner_cls:
        runner_cls.return_value.run_all_models = AsyncMock(return_value={
            "gpt-4o-mini": [SimpleNamespace(prompt_id="r001", response_text="150 miles")]
        })
        tracker.return_value.log_run = MagicMock(return_value="mlflow_1")
        orch = EvalOrchestrator()
        orch.scorer = scorer_with_rouge(0.5)
        with bert_patch:
            result = await orch.run(
                run_id="r1", run_name="n", models=["gpt-4o-mini"], dataset_name="reasoning",
                dataset_version="v1", include_judge=False, include_adversarial=False,
            )
        return result, tracker.return_value.log_run


@pytest.mark.asyncio
async def test_a_run_with_failed_bert_score_lists_it_as_unavailable(tmp_path, monkeypatch):
    result, log_run = await run_eval(tmp_path, monkeypatch, patch(BERT, side_effect=RuntimeError("x")))
    assert result["unavailable_metrics"] == {"gpt-4o-mini": ["bert_score"]}
    assert "bert_score" not in result["metrics"]["gpt-4o-mini"]
    assert "rouge_l" in result["metrics"]["gpt-4o-mini"]
    assert "bert_score" not in log_run.call_args.kwargs["metrics_by_model"]["gpt-4o-mini"]


@pytest.mark.asyncio
async def test_a_healthy_run_has_nothing_unavailable(tmp_path, monkeypatch):
    result, _ = await run_eval(tmp_path, monkeypatch, patch(BERT, return_value=(None, None, [0.9])))
    assert result["unavailable_metrics"] == {}
    assert result["metrics"]["gpt-4o-mini"]["bert_score"]["mean"] == 0.9
