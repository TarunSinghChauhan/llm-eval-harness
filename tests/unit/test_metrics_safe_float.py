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


from unittest.mock import MagicMock, patch

from src.evaluation.scorers.metrics import MetricResult, ScoredPair


def test_rouge_l_uses_the_stubbed_scorer_and_rounds_result():
    scorer = MetricScorer()
    fake_score_obj = MagicMock()
    fake_score_obj.fmeasure = 0.73456
    scorer._rouge.score.return_value = {"rougeL": fake_score_obj}
    result = scorer.rouge_l("the model said this", "the reference")
    assert result == 0.7346
    scorer._rouge.score.assert_called_once_with("the reference", "the model said this")


def test_bert_score_batch_returns_empty_list_for_empty_predictions():
    scorer = MetricScorer()
    assert scorer.bert_score_batch([], []) == []


def test_bert_score_batch_returns_rounded_f1_scores_on_success():
    scorer = MetricScorer()
    fake_f1 = [0.812345, 0.55]
    with patch("src.evaluation.scorers.metrics.bert_score_fn", return_value=(None, None, fake_f1)):
        result = scorer.bert_score_batch(["pred1", "pred2"], ["ref1", "ref2"])
    assert result == [0.8123, 0.55]


def test_bert_score_batch_falls_back_to_rouge_l_on_exception():
    scorer = MetricScorer()
    fake_score_obj = MagicMock()
    fake_score_obj.fmeasure = 0.5
    scorer._rouge.score.return_value = {"rougeL": fake_score_obj}
    with patch("src.evaluation.scorers.metrics.bert_score_fn", side_effect=RuntimeError("model load failed")):
        result = scorer.bert_score_batch(["pred1"], ["ref1"])
    assert result == [0.5]


def test_score_all_builds_scored_pairs_for_each_prompt():
    scorer = MetricScorer()
    fake_score_obj = MagicMock()
    fake_score_obj.fmeasure = 1.0
    scorer._rouge.score.return_value = {"rougeL": fake_score_obj}
    with patch("src.evaluation.scorers.metrics.bert_score_fn", return_value=(None, None, [0.9])):
        results = scorer.score_all(["Paris"], ["Paris"], ["p1"])
    assert len(results) == 1
    assert isinstance(results[0], ScoredPair)
    assert results[0].prompt_id == "p1"
    assert results[0].exact_match == 1.0
    assert results[0].bert_score == 0.9


def test_bootstrap_ci_returns_metric_result_with_expected_fields():
    scorer = MetricScorer()
    result = scorer.bootstrap_ci([0.5, 0.6, 0.7, 0.8], n_iterations=50)
    assert isinstance(result, MetricResult)
    assert result.n_samples == 4
    assert result.mean == 0.65
    assert result.ci_lower <= result.mean <= result.ci_upper
    assert result.raw_scores == [0.5, 0.6, 0.7, 0.8]


def test_bootstrap_ci_sanitizes_nan_and_inf_in_input_scores():
    scorer = MetricScorer()
    result = scorer.bootstrap_ci([0.5, float("nan"), float("inf")], n_iterations=20)
    assert result.raw_scores == [0.5, 0.0, 0.0]


def test_aggregate_returns_metric_result_for_each_metric_type():
    scorer = MetricScorer()
    pairs = [
        ScoredPair("p1", rouge_l=0.5, bert_score=0.6, exact_match=1.0),
        ScoredPair("p2", rouge_l=0.7, bert_score=0.8, exact_match=0.0),
    ]
    result = scorer.aggregate(pairs, n_iterations=20)
    assert set(result.keys()) == {"rouge_l", "bert_score", "exact_match"}
    assert isinstance(result["rouge_l"], MetricResult)
    assert result["rouge_l"].n_samples == 2
