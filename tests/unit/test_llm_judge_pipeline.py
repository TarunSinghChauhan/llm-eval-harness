import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.judge.llm_judge import LLMJudge, JudgeScore


def make_llm_response(score, reasoning):
    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({"score": score, "reasoning": reasoning})))
    ]
    return fake_response


def test_init_constructs_openai_client_with_settings():
    with patch("src.evaluation.judge.llm_judge.openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        judge = LLMJudge()
    mock_cls.assert_called_once()
    assert judge.client is mock_cls.return_value


def test_build_eval_prompt_includes_all_fields():
    judge = LLMJudge.__new__(LLMJudge)
    prompt = judge._build_eval_prompt("What is 2+2?", "4", "The answer is 4")
    assert "What is 2+2?" in prompt
    assert "4" in prompt
    assert "The answer is 4" in prompt


@pytest.mark.asyncio
async def test_judge_openai_returns_judge_score_on_success():
    judge = LLMJudge.__new__(LLMJudge)
    judge.client = MagicMock()
    judge.client.chat.completions.create = AsyncMock(
        return_value=make_llm_response(8, "mostly correct")
    )
    result = await judge._judge_openai("p1", "gpt-4o-mini", "q", "ref", "resp")
    assert result.score == 8.0
    assert result.reasoning == "mostly correct"
    assert result.judge_model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_judge_openai_returns_none_on_exception():
    judge = LLMJudge.__new__(LLMJudge)
    judge.client = MagicMock()
    judge.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("api down"))
    result = await judge._judge_openai("p1", "gpt-4o-mini", "q", "ref", "resp")
    assert result is None


@pytest.mark.asyncio
async def test_judge_anthropic_returns_judge_score_on_success():
    judge = LLMJudge.__new__(LLMJudge)
    judge.client = MagicMock()
    judge.client.chat.completions.create = AsyncMock(
        return_value=make_llm_response(6, "partially correct")
    )
    result = await judge._judge_anthropic("p1", "gpt-4o-mini", "q", "ref", "resp")
    assert result.score == 6.0
    assert result.judge_model == "claude-3-haiku"


@pytest.mark.asyncio
async def test_judge_anthropic_returns_none_on_exception():
    judge = LLMJudge.__new__(LLMJudge)
    judge.client = MagicMock()
    judge.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("api down"))
    result = await judge._judge_anthropic("p1", "gpt-4o-mini", "q", "ref", "resp")
    assert result is None


@pytest.mark.asyncio
async def test_judge_single_combines_both_scores_and_computes_agreement():
    judge = LLMJudge.__new__(LLMJudge)
    gpt_score = JudgeScore("p1", "gpt-4o-mini", "gpt-4o-mini", 8.0, "gpt reasoning", "{}")
    claude_score = JudgeScore("p1", "gpt-4o-mini", "claude-3-haiku", 6.0, "claude reasoning", "{}")

    with patch.object(judge, "_judge_openai", AsyncMock(return_value=gpt_score)), \
         patch.object(judge, "_judge_anthropic", AsyncMock(return_value=claude_score)):
        result = await judge.judge_single("p1", "gpt-4o-mini", "q", "ref", "resp")

    assert result.gpt_score == 8.0
    assert result.claude_score == 6.0
    assert result.ensemble_score == 7.0
    assert result.agreement == round(1.0 - abs(8.0 - 6.0) / 10.0, 4)
    assert "gpt reasoning" in result.reasoning
    assert "claude reasoning" in result.reasoning


@pytest.mark.asyncio
async def test_judge_single_handles_one_judge_failing():
    judge = LLMJudge.__new__(LLMJudge)
    gpt_score = JudgeScore("p1", "gpt-4o-mini", "gpt-4o-mini", 8.0, "gpt reasoning", "{}")

    with patch.object(judge, "_judge_openai", AsyncMock(return_value=gpt_score)), \
         patch.object(judge, "_judge_anthropic", AsyncMock(return_value=None)):
        result = await judge.judge_single("p1", "gpt-4o-mini", "q", "ref", "resp")

    assert result.gpt_score == 8.0
    assert result.claude_score is None
    assert result.ensemble_score == 8.0
    assert result.agreement == 0.0
    assert result.reasoning == "gpt reasoning"


@pytest.mark.asyncio
async def test_judge_single_handles_both_judges_failing():
    judge = LLMJudge.__new__(LLMJudge)

    with patch.object(judge, "_judge_openai", AsyncMock(return_value=None)), \
         patch.object(judge, "_judge_anthropic", AsyncMock(return_value=None)):
        result = await judge.judge_single("p1", "gpt-4o-mini", "q", "ref", "resp")

    assert result.gpt_score is None
    assert result.claude_score is None
    assert result.ensemble_score == 0.0
    assert result.agreement == 0.0
    assert result.reasoning == ""


@pytest.mark.asyncio
async def test_judge_batch_runs_all_items_and_filters_exceptions():
    judge = LLMJudge.__new__(LLMJudge)

    async def fake_judge_single(prompt_id, model_judged, question, reference, response):
        if prompt_id == "bad":
            raise RuntimeError("boom")
        return MagicMock(prompt_id=prompt_id)

    items = [
        {"prompt_id": "p1", "model_judged": "m", "question": "q", "reference": "r", "response": "resp"},
        {"prompt_id": "bad", "model_judged": "m", "question": "q", "reference": "r", "response": "resp"},
        {"prompt_id": "p2", "model_judged": "m", "question": "q", "reference": "r", "response": "resp"},
    ]

    with patch.object(judge, "judge_single", side_effect=fake_judge_single):
        results = await judge.judge_batch(items, max_concurrent=2)

    result_ids = [r.prompt_id for r in results]
    assert result_ids == ["p1", "p2"]  # the "bad" one raised and got filtered out
