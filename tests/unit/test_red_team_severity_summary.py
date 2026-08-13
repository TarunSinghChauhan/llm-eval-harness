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
