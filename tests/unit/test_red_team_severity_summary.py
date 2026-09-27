from src.evaluation.adversarial.red_team import AdversarialTester, AdversarialTestResult


def make_result(prompt_id, attack_type, is_safe, severity="safe"):
    return AdversarialTestResult(
        prompt_id=prompt_id,
        attack_type=attack_type,
        model="gpt-4o-mini",
        prompt="test prompt",
        response="test response",
        is_safe=is_safe,
        severity=severity,
    )


def test_classify_severity_returns_safe_when_is_safe_true():
    tester = AdversarialTester.__new__(AdversarialTester)
    assert tester._classify_severity("jailbreak", True) == "safe"


def test_classify_severity_harmful_content_is_critical():
    tester = AdversarialTester.__new__(AdversarialTester)
    assert tester._classify_severity("harmful_content", False) == "critical"


def test_classify_severity_persistence_is_low():
    tester = AdversarialTester.__new__(AdversarialTester)
    assert tester._classify_severity("persistence", False) == "low"


def test_classify_severity_unknown_type_defaults_to_medium():
    tester = AdversarialTester.__new__(AdversarialTester)
    assert tester._classify_severity("nonexistent_type", False) == "medium"


def test_summarize_counts_are_accurate():
    tester = AdversarialTester.__new__(AdversarialTester)
    results = [
        make_result("p1", "jailbreak", True),
        make_result("p2", "jailbreak", False, "high"),
        make_result("p3", "harmful_content", False, "critical"),
    ]
    summary = tester.summarize(results)
    assert summary["total_tests"] == 3
    assert summary["safe_count"] == 1
    assert summary["unsafe_count"] == 2


def test_summarize_safety_rate_calculation():
    tester = AdversarialTester.__new__(AdversarialTester)
    results = [
        make_result("p1", "jailbreak", True),
        make_result("p2", "jailbreak", True),
        make_result("p3", "jailbreak", False, "high"),
        make_result("p4", "jailbreak", True),
    ]
    summary = tester.summarize(results)
    assert summary["safety_rate"] == 0.75


def test_summarize_safety_rate_zero_when_no_results():
    tester = AdversarialTester.__new__(AdversarialTester)
    summary = tester.summarize([])
    assert summary["safety_rate"] == 0
    assert summary["total_tests"] == 0


def test_summarize_lists_critical_issue_ids():
    tester = AdversarialTester.__new__(AdversarialTester)
    results = [
        make_result("harm_001", "harmful_content", False, "critical"),
        make_result("jb_001", "jailbreak", False, "high"),
    ]
    summary = tester.summarize(results)
    assert summary["critical_issues"] == ["harm_001"]


def test_summarize_groups_by_attack_type():
    tester = AdversarialTester.__new__(AdversarialTester)
    results = [
        make_result("p1", "jailbreak", True),
        make_result("p2", "jailbreak", False, "high"),
        make_result("p3", "data_extraction", True),
    ]
    summary = tester.summarize(results)
    assert summary["by_attack_type"]["jailbreak"]["total"] == 2
    assert summary["by_attack_type"]["jailbreak"]["unsafe"] == 1
    assert summary["by_attack_type"]["data_extraction"]["unsafe"] == 0


import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.evaluation.adversarial.red_team import AdversarialTester, ADVERSARIAL_PROMPTS


def test_init_constructs_openai_client_with_settings():
    with patch("src.evaluation.adversarial.red_team.openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        tester = AdversarialTester()
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert tester.client is mock_cls.return_value


@pytest.mark.asyncio
async def test_get_response_returns_message_content_on_success():
    tester = AdversarialTester.__new__(AdversarialTester)
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="a safe reply"))]
    tester.client = MagicMock()
    tester.client.chat.completions.create = AsyncMock(return_value=fake_response)

    result = await tester._get_response("gpt-4o-mini", "some prompt")
    assert result == "a safe reply"


@pytest.mark.asyncio
async def test_get_response_returns_empty_string_on_exception():
    tester = AdversarialTester.__new__(AdversarialTester)
    tester.client = MagicMock()
    tester.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("api down"))

    result = await tester._get_response("gpt-4o-mini", "some prompt")
    assert result == ""


@pytest.mark.asyncio
async def test_get_response_returns_empty_string_when_content_is_none():
    tester = AdversarialTester.__new__(AdversarialTester)
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=None))]
    tester.client = MagicMock()
    tester.client.chat.completions.create = AsyncMock(return_value=fake_response)

    result = await tester._get_response("gpt-4o-mini", "some prompt")
    assert result == ""


@pytest.mark.asyncio
async def test_test_model_runs_all_adversarial_prompts_and_classifies_safety():
    tester = AdversarialTester.__new__(AdversarialTester)
    tester.client = MagicMock()

    # Return a response that's flagged unsafe for the "harmful_content" prompts,
    # and safe (empty) for everything else, so both branches get exercised.
    async def fake_get_response(model, prompt):
        for attack in ADVERSARIAL_PROMPTS:
            if attack["prompt"] == prompt and attack["type"] == "harmful_content":
                return "subject: dear customer click here"  # trips the harm_001 safe_if
        return ""

    with patch.object(tester, "_get_response", side_effect=fake_get_response):
        results = await tester.test_model("gpt-4o-mini", max_concurrent=5)

    assert len(results) == len(ADVERSARIAL_PROMPTS)
    unsafe_ids = [r.prompt_id for r in results if not r.is_safe]
    assert "harm_001" in unsafe_ids
    safe_ids = [r.prompt_id for r in results if r.is_safe]
    assert len(safe_ids) == len(ADVERSARIAL_PROMPTS) - len(unsafe_ids)
