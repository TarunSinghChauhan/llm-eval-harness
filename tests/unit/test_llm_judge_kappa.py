import pytest

from src.evaluation.judge.llm_judge import LLMJudge, EnsembleJudgeResult


def test_cohens_kappa_perfect_agreement():
    scores_a = [8, 8, 9, 5, 2]
    scores_b = [8, 8, 9, 5, 2]
    kappa = LLMJudge.cohens_kappa(scores_a, scores_b)
    assert kappa == 1.0


def test_cohens_kappa_no_agreement_returns_low_value():
    scores_a = [10, 10, 10, 10]
    scores_b = [0, 0, 0, 0]
    kappa = LLMJudge.cohens_kappa(scores_a, scores_b)
    assert kappa < 0.5


def test_cohens_kappa_empty_lists_returns_zero():
    kappa = LLMJudge.cohens_kappa([], [])
    assert kappa == 0.0


def test_cohens_kappa_mismatched_lengths_returns_zero():
    kappa = LLMJudge.cohens_kappa([8, 9], [8])
    assert kappa == 0.0


def test_cohens_kappa_returns_value_in_valid_range():
    scores_a = [7, 8, 6, 9, 5, 7, 8]
    scores_b = [7, 7, 6, 8, 5, 6, 9]
    kappa = LLMJudge.cohens_kappa(scores_a, scores_b)
    assert -1.0 <= kappa <= 1.0


@pytest.mark.asyncio
async def test_judge_single_ensemble_score_is_average_when_both_succeed(monkeypatch):
    judge = LLMJudge.__new__(LLMJudge)

    from src.evaluation.judge.llm_judge import JudgeScore

    async def fake_openai(*args, **kwargs):
        return JudgeScore("p1", "gpt-4o-mini", "gpt-4o-mini", 8.0, "good", "{}")

    async def fake_anthropic(*args, **kwargs):
        return JudgeScore("p1", "gpt-4o-mini", "claude-3-haiku", 6.0, "ok", "{}")

    monkeypatch.setattr(judge, "_judge_openai", fake_openai)
    monkeypatch.setattr(judge, "_judge_anthropic", fake_anthropic)

    result = await judge.judge_single("p1", "gpt-4o-mini", "Q", "ref", "resp")

    assert result.ensemble_score == 7.0
    assert result.gpt_score == 8.0
    assert result.claude_score == 6.0


@pytest.mark.asyncio
async def test_judge_single_uses_only_available_score_when_one_fails(monkeypatch):
    judge = LLMJudge.__new__(LLMJudge)

    from src.evaluation.judge.llm_judge import JudgeScore

    async def fake_openai(*args, **kwargs):
        return JudgeScore("p1", "gpt-4o-mini", "gpt-4o-mini", 9.0, "great", "{}")

    async def fake_anthropic(*args, **kwargs):
        return None

    monkeypatch.setattr(judge, "_judge_openai", fake_openai)
    monkeypatch.setattr(judge, "_judge_anthropic", fake_anthropic)

    result = await judge.judge_single("p1", "gpt-4o-mini", "Q", "ref", "resp")

    assert result.ensemble_score == 9.0
    assert result.claude_score is None
    assert result.agreement == 0.0


@pytest.mark.asyncio
async def test_judge_single_returns_zero_score_when_both_fail(monkeypatch):
    judge = LLMJudge.__new__(LLMJudge)

    async def fake_openai(*args, **kwargs):
        return None

    async def fake_anthropic(*args, **kwargs):
        return None

    monkeypatch.setattr(judge, "_judge_openai", fake_openai)
    monkeypatch.setattr(judge, "_judge_anthropic", fake_anthropic)

    result = await judge.judge_single("p1", "gpt-4o-mini", "Q", "ref", "resp")

    assert result.ensemble_score == 0.0
    assert result.gpt_score is None
    assert result.claude_score is None
