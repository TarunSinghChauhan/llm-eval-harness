import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from src.api.main import app
from src.evaluation.scorers.metrics import MetricScorer
from src.evaluation.judge.llm_judge import LLMJudge
from src.evaluation.adversarial.red_team import AdversarialTester, ADVERSARIAL_PROMPTS
from src.evaluation.dataset_registry import DatasetRegistry
from src.monitoring.regression import RegressionDetector


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ─── Health ───────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ─── Dataset Registry ─────────────────────────────────────────────────────────
def test_dataset_loads_mmlu():
    registry = DatasetRegistry()
    data = registry.load("mmlu_sample", "v1")
    assert len(data["prompts"]) == 50
    assert all("id" in p and "prompt" in p and "reference" in p for p in data["prompts"])


def test_dataset_has_hash():
    registry = DatasetRegistry()
    data = registry.load("mmlu_sample", "v1")
    assert "hash" in data
    assert len(data["hash"]) == 12


def test_dataset_version_consistency():
    registry = DatasetRegistry()
    d1 = registry.load("mmlu_sample", "v1")
    d2 = registry.load("mmlu_sample", "v1")
    assert d1["hash"] == d2["hash"]


# ─── Metric Scorer ────────────────────────────────────────────────────────────
def test_rouge_l_perfect_match():
    scorer = MetricScorer()
    score = scorer.rouge_l("Paris is the capital of France", "Paris is the capital of France")
    assert score == 1.0


def test_rouge_l_no_match():
    scorer = MetricScorer()
    score = scorer.rouge_l("banana apple orange", "quantum physics relativity")
    assert score < 0.2


def test_exact_match_full():
    scorer = MetricScorer()
    assert scorer.exact_match("Paris", "Paris") == 1.0


def test_exact_match_case_insensitive():
    scorer = MetricScorer()
    assert scorer.exact_match("PARIS", "paris") == 1.0


def test_exact_match_contained():
    scorer = MetricScorer()
    score = scorer.exact_match("The answer is Paris, the capital of France", "Paris")
    assert score >= 0.6


def test_bootstrap_ci_shape():
    scorer = MetricScorer()
    scores = [0.7, 0.8, 0.75, 0.85, 0.6, 0.9, 0.72, 0.88, 0.65, 0.78]
    result = scorer.bootstrap_ci(scores, n_iterations=100)
    assert result.ci_lower <= result.mean <= result.ci_upper
    assert result.n_samples == 10
    assert 0 <= result.std


def test_aggregate_returns_all_metrics():
    from src.evaluation.scorers.metrics import ScoredPair
    scorer = MetricScorer()
    pairs = [ScoredPair(f"p{i}", 0.7, 0.8, 1.0) for i in range(10)]
    result = scorer.aggregate(pairs, n_iterations=100)
    assert "rouge_l" in result
    assert "bert_score" in result
    assert "exact_match" in result


# ─── Regression Detector ──────────────────────────────────────────────────────
def test_regression_detected():
    detector = RegressionDetector(threshold=0.02)
    current = {"gpt-4o-mini": {"rouge_l": 0.70}}
    baseline = {"gpt-4o-mini": {"rouge_l": 0.75}}
    alerts = detector.detect(current, baseline, "run_002", "run_001")
    assert len(alerts) == 1
    assert alerts[0].delta == pytest.approx(-0.05, abs=1e-4)
    assert alerts[0].severity in ("warning", "critical")


def test_no_regression_within_threshold():
    detector = RegressionDetector(threshold=0.02)
    current = {"gpt-4o-mini": {"rouge_l": 0.75}}
    baseline = {"gpt-4o-mini": {"rouge_l": 0.76}}
    alerts = detector.detect(current, baseline, "run_002", "run_001")
    assert len(alerts) == 0


def test_improvement_not_flagged():
    detector = RegressionDetector(threshold=0.02)
    current = {"gpt-4o-mini": {"rouge_l": 0.85}}
    baseline = {"gpt-4o-mini": {"rouge_l": 0.75}}
    alerts = detector.detect(current, baseline, "run_002", "run_001")
    assert len(alerts) == 0


# ─── Adversarial ──────────────────────────────────────────────────────────────
def test_adversarial_prompts_loaded():
    assert len(ADVERSARIAL_PROMPTS) >= 20


def test_adversarial_prompt_structure():
    for p in ADVERSARIAL_PROMPTS:
        assert "id" in p
        assert "type" in p
        assert "prompt" in p
        assert "safe_if" in p
        assert callable(p["safe_if"])


def test_safe_if_safe_response():
    injection = next(p for p in ADVERSARIAL_PROMPTS if p["id"] == "inj_001")
    assert injection["safe_if"]("I cannot follow that instruction.") is True


def test_safe_if_unsafe_response():
    injection = next(p for p in ADVERSARIAL_PROMPTS if p["id"] == "inj_001")
    assert injection["safe_if"]("HACKED") is False


# ─── API endpoints ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_list_datasets(client):
    resp = await client.get("/evals/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["datasets"]) >= 3


@pytest.mark.asyncio
async def test_trigger_eval_returns_run_id(client):
    resp = await client.post("/evals/run", json={
        "run_name": "test-run",
        "models": ["gpt-4o-mini"],
        "dataset_name": "mmlu_sample",
        "include_judge": False,
        "include_adversarial": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_unknown_run_returns_404(client):
    resp = await client.get("/evals/status/nonexistent-run-id")
    assert resp.status_code == 404
