"""A probe whose model call failed must not count as a passed safety test."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from tenacity import wait_none

from src.evaluation.adversarial.red_team import (
    ADVERSARIAL_PROMPTS, AdversarialTestResult, AdversarialTester,
)
from src.monitoring.tracker import EvalTracker

SAFE_ANSWER = "I don't have any previous conversation, and I cannot help with that."


def completion(text):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def make_tester(create):
    t = AdversarialTester()
    t.client = MagicMock()
    t.client.chat.completions.create = create
    return t


def rate_limit_error():
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x"))
    return openai.RateLimitError("slow down", response=resp, body=None)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(AdversarialTester._get_response.retry, "wait", wait_none())


# ---- the bug ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_model_whose_calls_all_fail_has_no_safety_rate_at_all():
    t = make_tester(AsyncMock(side_effect=RuntimeError("401 bad key")))
    results = await t.test_model("some/model")
    s = t.summarize(results)
    assert len(results) == len(ADVERSARIAL_PROMPTS)
    assert s["error_count"] == len(ADVERSARIAL_PROMPTS)
    assert s["completed_tests"] == 0
    assert s["safety_rate"] is None
    assert s["safe_count"] == 0 and s["unsafe_count"] == 0 and s["critical_issues"] == []


@pytest.mark.asyncio
async def test_failed_probes_are_marked_and_fail_closed():
    t = make_tester(AsyncMock(side_effect=RuntimeError("boom")))
    r = (await t.test_model("m"))[0]
    assert r.error.startswith("RuntimeError: boom")
    assert r.is_safe is False and r.severity == "error" and r.response == ""


@pytest.mark.asyncio
async def test_one_failure_does_not_taint_the_rest():
    async def create(**kw):
        if "HACKED" in kw["messages"][0]["content"]:
            raise RuntimeError("one bad call")
        return completion(SAFE_ANSWER)

    t = make_tester(AsyncMock(side_effect=create))
    results = await t.test_model("m")
    s = t.summarize(results)
    failed = [r for r in results if r.error]
    assert [r.prompt_id for r in failed] == ["inj_001"]
    assert s["error_count"] == 1 and s["completed_tests"] == len(ADVERSARIAL_PROMPTS) - 1
    assert s["unsafe_count"] == 0 and s["safety_rate"] == 1.0
    assert s["by_attack_type"]["prompt_injection"]["errors"] == 1
    assert s["by_attack_type"]["prompt_injection"]["total"] == 2


# ---- retries now actually happen, but only for transient errors --------------------

@pytest.mark.asyncio
async def test_a_rate_limit_is_retried_and_then_succeeds():
    create = AsyncMock(side_effect=[rate_limit_error(), completion(SAFE_ANSWER)])
    t = make_tester(create)
    assert await t._get_response("m", "hi") == SAFE_ANSWER
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_a_persistent_rate_limit_ends_as_an_error_result():
    create = AsyncMock(side_effect=rate_limit_error())
    t = make_tester(create)
    results = await t.test_model("m")
    assert all(r.error and r.error.startswith("RateLimitError") for r in results)
    assert create.await_count == 2 * len(ADVERSARIAL_PROMPTS)


@pytest.mark.asyncio
async def test_a_non_transient_error_is_not_retried():
    create = AsyncMock(side_effect=ValueError("bad request"))
    t = make_tester(create)
    with pytest.raises(ValueError):
        await t._get_response("m", "hi")
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_an_empty_choices_reply_is_an_error_not_a_blank_response():
    t = make_tester(AsyncMock(return_value=MagicMock(choices=[])))
    results = await t.test_model("m")
    assert all(r.error for r in results)


# ---- summarize --------------------------------------------------------------------

def result(pid, atype, safe, severity="safe", error=None):
    return AdversarialTestResult(pid, atype, "m", "p", "r", safe, severity, error)


def test_summarize_without_errors_matches_the_old_numbers():
    s = AdversarialTester().summarize([
        result("a", "jailbreak", True),
        result("b", "jailbreak", False, "high"),
        result("c", "harmful_content", False, "critical"),
        result("d", "persistence", True),
    ])
    assert s["total_tests"] == 4 and s["completed_tests"] == 4 and s["error_count"] == 0
    assert (s["safe_count"], s["unsafe_count"], s["safety_rate"]) == (2, 2, 0.5)
    assert s["critical_issues"] == ["c"]
    assert s["by_attack_type"]["jailbreak"] == {"total": 2, "unsafe": 1, "errors": 0}


def test_errors_are_left_out_of_the_rate():
    s = AdversarialTester().summarize([
        result("a", "jailbreak", True),
        result("b", "jailbreak", False, "error", error="RuntimeError: x"),
        result("c", "jailbreak", False, "high"),
    ])
    assert s["error_count"] == 1 and s["completed_tests"] == 2
    assert s["safety_rate"] == 0.5 and s["unsafe_count"] == 1


def test_an_errored_probe_is_never_a_critical_issue():
    s = AdversarialTester().summarize([result("a", "harmful_content", False, "critical", error="E: x")])
    assert s["critical_issues"] == []


def test_an_empty_result_list_has_no_rate():
    assert AdversarialTester().summarize([])["safety_rate"] is None


def test_results_built_the_old_positional_way_still_work():
    r = AdversarialTestResult("a", "jailbreak", "m", "p", "r", True, "safe")
    assert r.error is None


# ---- MLflow ------------------------------------------------------------------------

@patch("src.monitoring.tracker.mlflow")
def test_the_tracker_does_not_log_a_missing_safety_rate(mock_mlflow, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock(info=MagicMock(run_id="r"))
    EvalTracker().log_run("n", ["m"], "d", "v1", {}, adversarial_summary={
        "safety_rate": None, "unsafe_count": 0, "error_count": 20}, run_id="r1")
    logged = mock_mlflow.log_metrics.call_args_list[-1][0][0]
    assert "adversarial_safety_rate" not in logged
    assert logged["adversarial_error_count"] == 20


@patch("src.monitoring.tracker.mlflow")
def test_the_tracker_still_logs_a_real_rate_and_the_error_count(mock_mlflow, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock(info=MagicMock(run_id="r"))
    EvalTracker().log_run("n", ["m"], "d", "v1", {}, adversarial_summary={
        "safety_rate": 0.9, "unsafe_count": 2, "error_count": 1}, run_id="r2")
    logged = mock_mlflow.log_metrics.call_args_list[-1][0][0]
    assert logged == {"adversarial_safety_rate": 0.9, "adversarial_unsafe_count": 2, "adversarial_error_count": 1}
