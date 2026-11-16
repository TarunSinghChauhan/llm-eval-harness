"""Judge agreement must be computed on matched items, and a non-answer is not a score of 0."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.judge.llm_judge import LLMJudge, parse_judge_score
from src.evaluation.orchestrator import EvalOrchestrator
from src.monitoring.regression import RegressionDetector


# ---- parse_judge_score ---------------------------------------------------------

@pytest.mark.parametrize("data,expected", [
    ({"score": 7}, 7.0), ({"score": "8.5"}, 8.5), ({"score": 0}, 0.0), ({"score": 10}, 10.0),
])
def test_valid_scores_are_accepted(data, expected):
    assert parse_judge_score(data) == expected


@pytest.mark.parametrize("data", [
    {}, {"reasoning": "no score here"}, {"score": None}, {"score": True},
    {"score": "eight"}, {"score": float("nan")}, {"score": float("inf")},
    {"score": -1}, {"score": 10.5}, {"score": 85}, [], "text", None,
])
def test_unusable_scores_are_rejected(data):
    with pytest.raises((ValueError, TypeError)):
        parse_judge_score(data)


def make_judge(content):
    judge = LLMJudge.__new__(LLMJudge)
    judge.client = MagicMock()
    reply = MagicMock()
    reply.choices = [MagicMock(message=MagicMock(content=content))]
    judge.client.chat.completions.create = AsyncMock(return_value=reply)
    return judge


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["_judge_openai", "_judge_anthropic"])
@pytest.mark.parametrize("content", [
    "", None, "{}", json.dumps({"reasoning": "forgot the score"}), json.dumps({"score": 42}),
])
async def test_a_reply_without_a_valid_score_is_a_failed_judgment_not_a_zero(method, content):
    judge = make_judge(content)
    assert await getattr(judge, method)("p1", "model", "q", "ref", "resp") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["_judge_openai", "_judge_anthropic"])
async def test_a_real_zero_from_the_judge_is_still_recorded(method):
    judge = make_judge(json.dumps({"score": 0, "reasoning": "completely wrong"}))
    result = await getattr(judge, method)("p1", "model", "q", "ref", "resp")
    assert result is not None and result.score == 0.0


@pytest.mark.asyncio
async def test_ensemble_ignores_a_judge_that_returned_no_score():
    judge = LLMJudge.__new__(LLMJudge)
    judge._judge_openai = AsyncMock(return_value=SimpleNamespace(score=9.0, reasoning="good"))
    judge._judge_anthropic = AsyncMock(return_value=None)  # e.g. empty reply
    result = await judge.judge_single("p1", "m", "q", "ref", "resp")
    assert result.gpt_score == 9.0 and result.claude_score is None
    assert result.ensemble_score == 9.0  # not dragged down to 4.5 by a phantom 0


# ---- orchestrator: kappa on paired items --------------------------------------

def _run(judge_results, tmp_path, monkeypatch, real_kappa=True):
    monkeypatch.chdir(tmp_path)
    prompts_ids = [f"r{i:03d}" for i in range(len(judge_results))]
    with patch("src.evaluation.orchestrator.EvalTracker") as tracker, \
         patch("src.evaluation.orchestrator.RegressionDetector"), \
         patch("src.evaluation.orchestrator.AdversarialTester"), \
         patch("src.evaluation.orchestrator.LLMJudge") as judge_cls, \
         patch("src.evaluation.orchestrator.ModelRunner") as runner_cls:
        runner_cls.return_value.run_all_models = AsyncMock(return_value={
            "m": [SimpleNamespace(prompt_id=pid, response_text="x") for pid in prompts_ids]
        })
        judge = judge_cls.return_value
        judge.judge_batch = AsyncMock(return_value=judge_results)
        judge.cohens_kappa = MagicMock(side_effect=LLMJudge.cohens_kappa)
        tracker.return_value.log_run = MagicMock(return_value="mlflow")
        return judge, EvalOrchestrator()


@pytest.mark.asyncio
async def test_kappa_uses_only_items_both_judges_scored(tmp_path, monkeypatch):
    results = [
        SimpleNamespace(gpt_score=9.0, claude_score=9.0),
        SimpleNamespace(gpt_score=None, claude_score=1.0),   # GPT failed here
        SimpleNamespace(gpt_score=1.0, claude_score=None),   # Claude failed here
        SimpleNamespace(gpt_score=5.0, claude_score=5.0),
    ]
    judge, orch = _run(results, tmp_path, monkeypatch)
    out = await orch.run(run_id="r1", run_name="n", models=["m"], dataset_name="reasoning",
                         dataset_version="v1", include_judge=True, include_adversarial=False)

    gpt_arg, claude_arg = judge.cohens_kappa.call_args.args
    assert gpt_arg == [9.0, 5.0]
    assert claude_arg == [9.0, 5.0]          # aligned item by item
    assert out["judge_kappa_n"] == {"m": 2}
    assert out["judge_kappa"]["m"] == 1.0    # the two judges agree on both shared items


@pytest.mark.asyncio
async def test_kappa_is_not_the_silent_zero_when_failures_differ_in_number(tmp_path, monkeypatch):
    results = [
        SimpleNamespace(gpt_score=2.0, claude_score=2.0),
        SimpleNamespace(gpt_score=8.0, claude_score=8.0),
        SimpleNamespace(gpt_score=None, claude_score=6.0),   # only Claude scored this one
    ]
    judge, orch = _run(results, tmp_path, monkeypatch)
    out = await orch.run(run_id="r2", run_name="n", models=["m"], dataset_name="reasoning",
                         dataset_version="v1", include_judge=True, include_adversarial=False)
    # Independent filtering gave lists of length 2 and 3, and cohens_kappa answered 0.0.
    assert out["judge_kappa"]["m"] == 1.0
    assert out["judge_kappa_n"]["m"] == 2


@pytest.mark.asyncio
async def test_no_kappa_when_no_item_was_scored_by_both(tmp_path, monkeypatch):
    results = [
        SimpleNamespace(gpt_score=8.0, claude_score=None),
        SimpleNamespace(gpt_score=None, claude_score=7.0),
    ]
    judge, orch = _run(results, tmp_path, monkeypatch)
    out = await orch.run(run_id="r3", run_name="n", models=["m"], dataset_name="reasoning",
                         dataset_version="v1", include_judge=True, include_adversarial=False)
    judge.cohens_kappa.assert_not_called()
    assert out["judge_kappa"] == {} and out["judge_kappa_n"] == {}


# ---- regression detector -----------------------------------------------------

def test_a_threshold_of_zero_flags_any_drop():
    detector = RegressionDetector(threshold=0.0)
    assert detector.threshold == 0.0
    alerts = detector.detect({"m": {"rouge_l": 0.499}}, {"m": {"rouge_l": 0.5}}, "run", "base")
    assert [a.metric for a in alerts] == ["rouge_l"]


def test_default_threshold_still_comes_from_settings():
    from src.monitoring import regression
    assert RegressionDetector().threshold == regression.settings.regression_threshold
