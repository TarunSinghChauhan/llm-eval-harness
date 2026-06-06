import asyncio
import time
import hashlib
import json

import openai
import redis.asyncio as aioredis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.config import get_settings
from src.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# OpenRouter base URL — OpenAI-compatible
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models available via OpenRouter (free or cheap)
MODEL_PRICING = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "mistralai/mistral-7b-instruct": {"input": 0.07, "output": 0.07},
    "meta-llama/llama-3.1-8b-instruct:free": {"input": 0.0, "output": 0.0},
}


def compute_cache_key(model: str, prompt: str, system: str) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "system": system}, sort_keys=True)
    return f"eval:response:{hashlib.sha256(payload.encode()).hexdigest()}"


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


class ModelResponse:
    def __init__(self, model, prompt_id, response_text, input_tokens, output_tokens, latency_ms, cached=False):
        self.model = model
        self.prompt_id = prompt_id
        self.response_text = response_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.cost_usd = compute_cost(model, input_tokens, output_tokens)
        self.cached = cached


class ModelRunner:
    """
    Async model runner using OpenRouter (OpenAI-compatible API).
    Supports GPT-4o, Claude, Mistral, LLaMA through one endpoint.
    """

    def __init__(self):
        # OpenRouter uses OpenAI SDK with custom base URL
        self.client = openai.AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        self._redis: aioredis.Redis | None = None

    async def get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
    )
    async def _call_model(self, model: str, prompt: str, system: str) -> tuple[str, int, int]:
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.0,
            extra_headers={
                "HTTP-Referer": "https://github.com/llm-eval-harness",
                "X-Title": "LLM Eval Harness",
            },
        )
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return text, input_tokens, output_tokens

    async def run_single(
        self,
        model: str,
        prompt_id: str,
        prompt: str,
        system: str = "You are a helpful assistant. Answer concisely and accurately.",
    ) -> ModelResponse:
        redis = await self.get_redis()
        cache_key = compute_cache_key(model, prompt, system)

        cached = await redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            logger.info("cache_hit", model=model, prompt_id=prompt_id)
            return ModelResponse(
                model=model,
                prompt_id=prompt_id,
                response_text=data["text"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                latency_ms=data["latency_ms"],
                cached=True,
            )

        start = time.perf_counter()
        try:
            text, in_tok, out_tok = await self._call_model(model, prompt, system)
        except Exception as e:
            logger.error("model_call_failed", model=model, prompt_id=prompt_id, error=str(e))
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        cache_data = {"text": text, "input_tokens": in_tok, "output_tokens": out_tok, "latency_ms": latency_ms}
        await redis.setex(cache_key, 86400, json.dumps(cache_data))

        logger.info("model_call_success", model=model, prompt_id=prompt_id, latency_ms=round(latency_ms, 1))
        return ModelResponse(model, prompt_id, text, in_tok, out_tok, latency_ms)

    async def run_batch(self, model, prompts, system="You are a helpful assistant. Answer concisely and accurately.", max_concurrent=None):
        concurrency = max_concurrent or settings.eval_max_concurrent
        semaphore = asyncio.Semaphore(concurrency)

        async def run_with_semaphore(p):
            async with semaphore:
                return await self.run_single(model, p["id"], p["prompt"], system)

        results = await asyncio.gather(*[run_with_semaphore(p) for p in prompts], return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def run_all_models(self, models, prompts, system="You are a helpful assistant. Answer concisely and accurately."):
        results = {}
        for model in models:
            results[model] = await self.run_batch(model, prompts, system)
        return results

    async def close(self):
        if self._redis:
            await self._redis.close()