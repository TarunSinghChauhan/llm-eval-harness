import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.runner import ModelRunner, ModelResponse, compute_cache_key


def make_runner():
    """Build a ModelRunner without running __init__ (no real OpenAI client)."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.client = MagicMock()
    runner._redis = None
    return runner


def make_chat_response(text="hello", prompt_tokens=10, completion_tokens=5, with_usage=True):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens) if with_usage else None
    return resp


# ── ModelResponse ────────────────────────────────────────────────────────────

def test_model_response_computes_cost_and_defaults_cached_false():
    r = ModelResponse("openai/gpt-4o-mini", "p1", "text", 1_000_000, 1_000_000, 12.5)
    assert r.cached is False
    assert r.cost_usd == pytest.approx(0.15 + 0.60)
    assert r.prompt_id == "p1"
    assert r.latency_ms == 12.5


def test_model_response_cached_flag_is_stored():
    r = ModelResponse("openai/gpt-4o-mini", "p1", "text", 0, 0, 1.0, cached=True)
    assert r.cached is True


# ── __init__ / get_redis / close ─────────────────────────────────────────────

def test_init_constructs_openai_client_and_starts_without_redis():
    with patch("src.evaluation.runner.openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        runner = ModelRunner()
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert runner.client is mock_cls.return_value
    assert runner._redis is None


@pytest.mark.asyncio
async def test_get_redis_creates_connection_once_and_caches_it():
    runner = make_runner()
    fake_redis = MagicMock()
    with patch("src.evaluation.runner.aioredis.from_url", AsyncMock(return_value=fake_redis)) as mock_from_url:
        first = await runner.get_redis()
        second = await runner.get_redis()
    assert first is fake_redis
    assert second is fake_redis
    mock_from_url.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_closes_redis_when_connected():
    runner = make_runner()
    runner._redis = MagicMock()
    runner._redis.close = AsyncMock()
    await runner.close()
    runner._redis.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_is_noop_when_redis_never_opened():
    runner = make_runner()
    await runner.close()  # should not raise
    assert runner._redis is None


# ── _call_model ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_model_returns_text_and_token_counts():
    runner = make_runner()
    runner.client.chat.completions.create = AsyncMock(
        return_value=make_chat_response("the answer", 20, 7)
    )
    text, in_tok, out_tok = await runner._call_model("openai/gpt-4o-mini", "q", "sys")
    assert text == "the answer"
    assert (in_tok, out_tok) == (20, 7)


@pytest.mark.asyncio
async def test_call_model_defaults_tokens_to_zero_when_usage_missing():
    runner = make_runner()
    runner.client.chat.completions.create = AsyncMock(
        return_value=make_chat_response("x", with_usage=False)
    )
    text, in_tok, out_tok = await runner._call_model("openai/gpt-4o-mini", "q", "sys")
    assert (in_tok, out_tok) == (0, 0)


@pytest.mark.asyncio
async def test_call_model_returns_empty_string_when_content_is_none():
    runner = make_runner()
    runner.client.chat.completions.create = AsyncMock(
        return_value=make_chat_response(None)
    )
    text, _, _ = await runner._call_model("openai/gpt-4o-mini", "q", "sys")
    assert text == ""


# ── run_single ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_single_returns_cached_response_on_cache_hit():
    runner = make_runner()
    cached_payload = json.dumps(
        {"text": "cached answer", "input_tokens": 11, "output_tokens": 4, "latency_ms": 3.0}
    )
    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=cached_payload)
    fake_redis.setex = AsyncMock()
    runner._call_model = AsyncMock()

    with patch.object(runner, "get_redis", AsyncMock(return_value=fake_redis)):
        result = await runner.run_single("openai/gpt-4o-mini", "p1", "question", "sys")

    assert result.cached is True
    assert result.response_text == "cached answer"
    assert result.input_tokens == 11
    runner._call_model.assert_not_awaited()
    fake_redis.setex.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_single_calls_model_and_caches_result_on_miss():
    runner = make_runner()
    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.setex = AsyncMock()
    runner._call_model = AsyncMock(return_value=("fresh answer", 9, 3))

    with patch.object(runner, "get_redis", AsyncMock(return_value=fake_redis)):
        result = await runner.run_single("openai/gpt-4o-mini", "p1", "question", "sys")

    assert result.cached is False
    assert result.response_text == "fresh answer"
    assert (result.input_tokens, result.output_tokens) == (9, 3)
    fake_redis.setex.assert_awaited_once()
    key, ttl, payload = fake_redis.setex.call_args[0]
    assert key == compute_cache_key("openai/gpt-4o-mini", "question", "sys")
    assert ttl == 86400
    assert json.loads(payload)["text"] == "fresh answer"


@pytest.mark.asyncio
async def test_run_single_reraises_model_errors_and_does_not_cache():
    runner = make_runner()
    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.setex = AsyncMock()
    runner._call_model = AsyncMock(side_effect=RuntimeError("model exploded"))

    with patch.object(runner, "get_redis", AsyncMock(return_value=fake_redis)):
        with pytest.raises(RuntimeError, match="model exploded"):
            await runner.run_single("openai/gpt-4o-mini", "p1", "question", "sys")

    fake_redis.setex.assert_not_awaited()


# ── run_batch / run_all_models ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_batch_filters_out_failed_prompts():
    runner = make_runner()

    async def fake_run_single(model, prompt_id, prompt, system):
        if prompt_id == "bad":
            raise RuntimeError("boom")
        return MagicMock(prompt_id=prompt_id)

    runner.run_single = fake_run_single
    prompts = [
        {"id": "a", "prompt": "qa"},
        {"id": "bad", "prompt": "qb"},
        {"id": "c", "prompt": "qc"},
    ]
    results = await runner.run_batch("openai/gpt-4o-mini", prompts, max_concurrent=2)
    assert [r.prompt_id for r in results] == ["a", "c"]


@pytest.mark.asyncio
async def test_run_batch_falls_back_to_settings_concurrency_when_not_given():
    runner = make_runner()
    runner.run_single = AsyncMock(return_value=MagicMock(prompt_id="a"))
    results = await runner.run_batch("openai/gpt-4o-mini", [{"id": "a", "prompt": "q"}])
    assert len(results) == 1


@pytest.mark.asyncio
async def test_run_all_models_returns_results_keyed_by_model():
    runner = make_runner()
    runner.run_batch = AsyncMock(side_effect=lambda model, prompts, system: [f"result-for-{model}"])
    results = await runner.run_all_models(["model-a", "model-b"], [{"id": "p1", "prompt": "q"}])
    assert results == {
        "model-a": ["result-for-model-a"],
        "model-b": ["result-for-model-b"],
    }
