import math

from src.evaluation.scorers.metrics import safe_float, MetricScorer


def test_safe_float_passes_through_normal_value():
    assert safe_float(0.75) == 0.75


def test_safe_float_converts_int_to_float():
    result = safe_float(5)
    assert result == 5.0
    assert isinstance(result, float)


def test_safe_float_replaces_none_with_zero():
    assert safe_float(None) == 0.0


def test_safe_float_replaces_nan_with_zero():
    assert safe_float(float("nan")) == 0.0


def test_safe_float_replaces_positive_inf_with_zero():
    assert safe_float(float("inf")) == 0.0


def test_safe_float_replaces_negative_inf_with_zero():
    assert safe_float(float("-inf")) == 0.0


def test_exact_match_full_match_scores_one():
    scorer = MetricScorer()
    assert scorer.exact_match("Paris", "Paris") == 1.0


def test_exact_match_containment_scores_point_eight():
    scorer = MetricScorer()
    score = scorer.exact_match("The capital is Paris, France", "Paris")
    assert score == 0.8


def test_exact_match_token_subset_scores_point_six():
    scorer = MetricScorer()
    score = scorer.exact_match("France Paris capital city europe", "Paris France")
    assert score == 0.6


def test_exact_match_no_overlap_scores_zero():
    scorer = MetricScorer()
    score = scorer.exact_match("banana apple", "quantum physics")
    assert score == 0.0
