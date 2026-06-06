import math
import numpy as np
from dataclasses import dataclass
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

from src.core.logging import get_logger

logger = get_logger(__name__)


def safe_float(val):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return 0.0
    return float(val)


@dataclass
class MetricResult:
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    raw_scores: list[float]


@dataclass
class ScoredPair:
    prompt_id: str
    rouge_l: float
    bert_score: float
    exact_match: float


class MetricScorer:
    def __init__(self):
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def rouge_l(self, prediction: str, reference: str) -> float:
        score = self._rouge.score(reference, prediction)
        return round(safe_float(score["rougeL"].fmeasure), 4)

    def exact_match(self, prediction: str, reference: str) -> float:
        pred_clean = prediction.strip().lower()
        ref_clean = reference.strip().lower()
        if pred_clean == ref_clean:
            return 1.0
        if ref_clean in pred_clean:
            return 0.8
        ref_tokens = set(ref_clean.split())
        pred_tokens = set(pred_clean.split())
        if ref_tokens and ref_tokens.issubset(pred_tokens):
            return 0.6
        return 0.0

    def bert_score_batch(self, predictions: list[str], references: list[str]) -> list[float]:
        if not predictions:
            return []
        try:
            _, _, f1 = bert_score_fn(
                predictions,
                references,
                lang="en",
                model_type="distilbert-base-uncased",
                verbose=False,
            )
            return [round(safe_float(s), 4) for s in f1]
        except Exception as e:
            logger.error("bert_score_failed", error=str(e))
            return [self.rouge_l(p, r) for p, r in zip(predictions, references)]

    def score_all(self, predictions, references, prompt_ids):
        rouge_scores = [self.rouge_l(p, r) for p, r in zip(predictions, references)]
        exact_scores = [self.exact_match(p, r) for p, r in zip(predictions, references)]
        bert_scores = self.bert_score_batch(predictions, references)
        return [
            ScoredPair(pid, rl, bs, em)
            for pid, rl, bs, em in zip(prompt_ids, rouge_scores, bert_scores, exact_scores)
        ]

    def bootstrap_ci(self, scores: list[float], n_iterations: int = 1000, ci: float = 0.95) -> MetricResult:
        scores = [safe_float(s) for s in scores]
        arr = np.array(scores)
        n = len(arr)
        bootstrap_means = []
        rng = np.random.default_rng(42)
        for _ in range(n_iterations):
            sample = rng.choice(arr, size=n, replace=True)
            bootstrap_means.append(safe_float(np.mean(sample)))
        bootstrap_means = np.array(bootstrap_means)
        alpha = (1 - ci) / 2
        return MetricResult(
            mean=round(safe_float(np.mean(arr)), 4),
            std=round(safe_float(np.std(arr)), 4),
            ci_lower=round(safe_float(np.percentile(bootstrap_means, alpha * 100)), 4),
            ci_upper=round(safe_float(np.percentile(bootstrap_means, (1 - alpha) * 100)), 4),
            n_samples=n,
            raw_scores=scores,
        )

    def aggregate(self, scored_pairs: list[ScoredPair], n_iterations: int = 1000) -> dict[str, MetricResult]:
        return {
            "rouge_l": self.bootstrap_ci([s.rouge_l for s in scored_pairs], n_iterations),
            "bert_score": self.bootstrap_ci([s.bert_score for s in scored_pairs], n_iterations),
            "exact_match": self.bootstrap_ci([s.exact_match for s in scored_pairs], n_iterations),
        }
