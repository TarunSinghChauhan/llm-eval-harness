"""A failed model call must be visible in the run output, not silently averaged away."""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.orchestrator import EvalOrchestrator
from src.evaluation.runner import ModelRunner

PROMPTS = [{"id": f"p{i}", "prompt": "q", "reference": "ref"} for i in range(4)]


def resp(pid, text="ref"):
    return SimpleNamespace(prompt_id=pid, response_text=text)


async def run_orchestrator(model_responses, tmp_path, monkeypatch, models=None, include_judge=False):
    monkeypatch.chdir(tmp_path)
    models = models or list(model_responses)
    with patch("src.evaluation.orchestrator.EvalTracker") as tracker, \
         patch("src.evaluation.orchestrator.RegressionDetector"), \
         patch("src.evaluation.orchestrator.AdversarialTester"), \
         patch("src.evaluation.orchestrator.LLMJudge") as judge, \
         patch("src.evaluation.orchestrator.ModelRunner") as runner, \
         patch("src.evaluation.orchestrator.DatasetRegistry") as registry:
        registry.return_value.load.return_value = {"prompts": PROMPTS}
        runner.return_value.run_all_models = AsyncMock(return_value=model_responses)
        tracker.return_value.log_run = MagicMock(return_value="mlflow")
        judge.return_value.judge_batch = AsyncMock(return_value=[])
        orch = EvalOrchestrator()
        orch.scorer.bert_score_batch = lambda preds, refs: [None] * len(preds)
        out = await orch.run(
            "run1", "name", models, include_judge=include_judge, include_adversarial=False,
        )
        return out, tracker.return_value.log_run


@pytest.mark.asyncio
async def test_failed_prompts_are_listed_in_the_output(tmp_path, monkeypatch):
    out, _ = await run_orchestrator({"m": [resp("p0"), resp("p2")]}, tmp_path, monkeypatch)
    assert out["failed_prompts"] == {"m": ["p1", "p3"]}
    assert out["n_scored"] == {"m": 2}
    assert out["n_prompts"] == 4
    assert out["complete"] is False


@pytest.mark.asyncio
async def test_a_fully_successful_run_is_complete(tmp_path, monkeypatch):
    out, _ = await run_orchestrator({"m": [resp(p["id"]) for p in PROMPTS]}, tmp_path, monkeypatch)
    assert out["complete"] is True
    assert out["failed_prompts"] == {}
    assert out["n_scored"] == {"m": 4}


@pytest.mark.asyncio
async def test_failures_are_tracked_per_model(tmp_path, monkeypatch):
    out, _ = await run_orchestrator(
        {"good": [resp(p["id"]) for p in PROMPTS], "flaky": [resp("p0")]}, tmp_path, monkeypatch
    )
    assert out["failed_prompts"] == {"flaky": ["p1", "p2", "p3"]}
    assert out["n_scored"] == {"good": 4, "flaky": 1}
    assert out["complete"] is False


@pytest.mark.asyncio
async def test_a_model_with_no_responses_has_no_metrics_instead_of_zero(tmp_path, monkeypatch):
    out, _ = await run_orchestrator({"m": []}, tmp_path, monkeypatch)
    assert out["metrics"]["m"] == {}
    assert out["failed_prompts"]["m"] == ["p0", "p1", "p2", "p3"]
    assert out["n_scored"] == {"m": 0}
    assert out["complete"] is False


@pytest.mark.asyncio
async def test_a_dead_model_does_not_disturb_a_working_one(tmp_path, monkeypatch):
    out, log_run = await run_orchestrator(
        {"dead": [], "ok": [resp(p["id"]) for p in PROMPTS]}, tmp_path, monkeypatch
    )
    assert out["metrics"]["dead"] == {}
    assert out["metrics"]["ok"]["exact_match"]["mean"] == 1.0
    assert out["metrics"]["ok"]["exact_match"]["n_samples"] == 4
    assert set(log_run.call_args.kwargs["metrics_by_model"]["dead"]) == set()


@pytest.mark.asyncio
async def test_the_judge_step_copes_with_a_model_that_has_no_responses(tmp_path, monkeypatch):
    out, _ = await run_orchestrator({"m": []}, tmp_path, monkeypatch, include_judge=True)
    assert out["judge_kappa"] == {}


@pytest.mark.asyncio
async def test_scored_responses_still_score_normally_alongside_failures(tmp_path, monkeypatch):
    out, _ = await run_orchestrator(
        {"m": [resp("p0", "ref"), resp("p1", "something else")]}, tmp_path, monkeypatch
    )
    em = out["metrics"]["m"]["exact_match"]
    assert em["n_samples"] == 2 and em["mean"] == 0.5


# ── runner.run_batch ──────────────────────────────────────────────────────────────

def make_runner():
    with patch("src.evaluation.runner.openai.AsyncOpenAI"):
        return ModelRunner()


@pytest.mark.asyncio
async def test_run_batch_warns_with_the_failed_prompt_ids():
    runner = make_runner()

    async def fake_run_single(model, prompt_id, prompt, system):
        if prompt_id in ("b", "c"):
            raise RuntimeError("boom")
        return MagicMock(prompt_id=prompt_id)

    runner.run_single = fake_run_single
    prompts = [{"id": i, "prompt": "q"} for i in "abcd"]
    with patch("src.evaluation.runner.logger") as log:
        results = await runner.run_batch("m", prompts, max_concurrent=2)
    assert [r.prompt_id for r in results] == ["a", "d"]
    log.warning.assert_called_once()
    args, kwargs = log.warning.call_args
    assert args == ("batch_incomplete",)
    assert kwargs["failed"] == 2 and kwargs["total"] == 4 and kwargs["failed_ids"] == ["b", "c"]


@pytest.mark.asyncio
async def test_run_batch_is_quiet_when_nothing_failed():
    runner = make_runner()
    runner.run_single = AsyncMock(side_effect=lambda model, pid, p, s: MagicMock(prompt_id=pid))
    with patch("src.evaluation.runner.logger") as log:
        results = await runner.run_batch("m", [{"id": "a", "prompt": "q"}], max_concurrent=1)
    assert len(results) == 1
    log.warning.assert_not_called()
