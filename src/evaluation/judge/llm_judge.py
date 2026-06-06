import asyncio
import json
from dataclasses import dataclass

import openai
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI model responses.
Given a question, a reference answer, and a model's response, score the response from 0 to 10.

Scoring criteria:
- 9-10: Correct, complete, and clear
- 7-8: Mostly correct with minor omissions
- 5-6: Partially correct but missing key information
- 3-4: Related but significantly wrong
- 0-2: Completely incorrect or irrelevant

Respond ONLY with valid JSON in this exact format:
{"score": <number 0-10>, "reasoning": "<one sentence explanation>"}"""


@dataclass
class JudgeScore:
    prompt_id: str
    model_judged: str
    judge_model: str
    score: float
    reasoning: str
    raw_response: str


@dataclass
class EnsembleJudgeResult:
    prompt_id: str
    model_judged: str
    gpt_score: float | None
    claude_score: float | None
    ensemble_score: float
    agreement: float
    reasoning: str


class LLMJudge:
    """
    LLM-as-judge pipeline using GPT-4o-mini and Claude-3-Haiku via OpenRouter.
    Computes inter-rater agreement (Cohen's kappa) for statistical validity.
    """

    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def _build_eval_prompt(self, question: str, reference: str, response: str) -> str:
        return f"""Question: {question}

Reference Answer: {reference}

Model Response: {response}

Score the model response."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _judge_openai(
        self, prompt_id: str, model_judged: str, question: str, reference: str, response: str
    ) -> JudgeScore | None:
        try:
            eval_prompt = self._build_eval_prompt(question, reference, response)
            result = await self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": eval_prompt},
                ],
                max_tokens=150,
                temperature=0.0,
                extra_headers={
                    "HTTP-Referer": "https://github.com/llm-eval-harness",
                    "X-Title": "LLM Eval Harness",
                },
            )
            raw = result.choices[0].message.content or "{}"
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            data = json.loads(raw)
            return JudgeScore(
                prompt_id=prompt_id,
                model_judged=model_judged,
                judge_model="gpt-4o-mini",
                score=float(data.get("score", 0)),
                reasoning=data.get("reasoning", ""),
                raw_response=raw,
            )
        except Exception as e:
            logger.error("gpt_judge_failed", prompt_id=prompt_id, error=str(e))
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _judge_anthropic(
        self, prompt_id: str, model_judged: str, question: str, reference: str, response: str
    ) -> JudgeScore | None:
        try:
            eval_prompt = self._build_eval_prompt(question, reference, response)
            result = await self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": eval_prompt},
                ],
                max_tokens=150,
                temperature=0.0,
                extra_headers={
                    "HTTP-Referer": "https://github.com/llm-eval-harness",
                    "X-Title": "LLM Eval Harness",
                },
            )
            raw = result.choices[0].message.content or "{}"
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            data = json.loads(raw)
            return JudgeScore(
                prompt_id=prompt_id,
                model_judged=model_judged,
                judge_model="claude-3-haiku",
                score=float(data.get("score", 0)),
                reasoning=data.get("reasoning", ""),
                raw_response=raw,
            )
        except Exception as e:
            logger.error("claude_judge_failed", prompt_id=prompt_id, error=str(e))
            return None

    async def judge_single(
        self,
        prompt_id: str,
        model_judged: str,
        question: str,
        reference: str,
        response: str,
    ) -> EnsembleJudgeResult:
        gpt_result, claude_result = await asyncio.gather(
            self._judge_openai(prompt_id, model_judged, question, reference, response),
            self._judge_anthropic(prompt_id, model_judged, question, reference, response),
        )

        scores = []
        gpt_score = None
        claude_score = None

        if gpt_result:
            gpt_score = gpt_result.score
            scores.append(gpt_score)
        if claude_result:
            claude_score = claude_result.score
            scores.append(claude_score)

        ensemble_score = float(np.mean(scores)) if scores else 0.0
        agreement = (
            1.0 - abs(gpt_score - claude_score) / 10.0
            if (gpt_score is not None and claude_score is not None)
            else 0.0
        )

        reasoning = " | ".join(filter(None, [
            gpt_result.reasoning if gpt_result else None,
            claude_result.reasoning if claude_result else None,
        ]))

        return EnsembleJudgeResult(
            prompt_id=prompt_id,
            model_judged=model_judged,
            gpt_score=gpt_score,
            claude_score=claude_score,
            ensemble_score=round(ensemble_score, 2),
            agreement=round(agreement, 4),
            reasoning=reasoning,
        )

    async def judge_batch(
        self,
        items: list[dict],
        max_concurrent: int = 5,
    ) -> list[EnsembleJudgeResult]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def judge_with_sem(item: dict) -> EnsembleJudgeResult:
            async with semaphore:
                return await self.judge_single(**item)

        results = await asyncio.gather(
            *[judge_with_sem(i) for i in items], return_exceptions=True
        )
        return [r for r in results if not isinstance(r, Exception)]

    @staticmethod
    def cohens_kappa(scores_a: list[float], scores_b: list[float], bins: int = 5) -> float:
        if len(scores_a) != len(scores_b) or not scores_a:
            return 0.0

        def discretize(scores):
            return [min(int(s / (10 / bins)), bins - 1) for s in scores]

        a = discretize(scores_a)
        b = discretize(scores_b)

        po = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        n = len(a)
        pe = sum((a.count(k) / n) * (b.count(k) / n) for k in range(bins))

        if pe == 1.0:
            return 1.0
        return round((po - pe) / (1 - pe), 4)