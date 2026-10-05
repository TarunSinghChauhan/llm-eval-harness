from datetime import datetime, timedelta, timezone

from src.core.database import (
    AdversarialResult,
    EvalMetric,
    EvalResult,
    EvalRun,
    _utcnow,
)


def _now_naive_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _assert_naive_utc_now(value: datetime):
    assert value.tzinfo is None
    assert abs(_now_naive_utc() - value) < timedelta(seconds=5)


def test_utcnow_returns_naive_value_close_to_current_utc():
    _assert_naive_utc_now(_utcnow())


def test_eval_run_created_at_default_is_naive_utc():
    run = EvalRun(
        name="r", dataset_name="mmlu_sample", dataset_version="v1", models=["m"]
    )
    _assert_naive_utc_now(run.created_at)


def test_eval_result_created_at_default_is_naive_utc():
    result = EvalResult(
        run_id="run1",
        model="m",
        prompt_id="p1",
        prompt_text="q",
        reference_answer="a",
        model_response="a",
    )
    _assert_naive_utc_now(result.created_at)


def test_eval_metric_created_at_default_is_naive_utc():
    metric = EvalMetric(
        run_id="run1",
        model="m",
        metric_name="rouge_l",
        mean=0.5,
        std=0.1,
        ci_lower=0.4,
        ci_upper=0.6,
        n_samples=10,
    )
    _assert_naive_utc_now(metric.created_at)


def test_adversarial_result_created_at_default_is_naive_utc():
    result = AdversarialResult(
        run_id="run1",
        model="m",
        attack_type="jailbreak",
        prompt="p",
        response="r",
    )
    _assert_naive_utc_now(result.created_at)
