"""exact_match must compare words, not raw substrings."""
import pytest

from src.evaluation.scorers.metrics import MetricScorer

S = MetricScorer()


@pytest.mark.parametrize("prediction,reference", [
    ("I know the answer", "no"),             # "no" inside "know"
    ("The Parisian cafe", "Paris"),          # prefix of a longer word
    ("catalog", "cat"),
    ("a 10 b", "1"),                         # digit inside a number
])
def test_a_substring_of_a_longer_word_is_not_a_match(prediction, reference):
    assert S.exact_match(prediction, reference) == 0.0


@pytest.mark.parametrize("reference", ["", "   "])
def test_an_empty_reference_matches_nothing_non_empty(reference):
    assert S.exact_match("anything at all", reference) == 0.0


def test_empty_against_empty_is_still_identical():
    assert S.exact_match("", "") == 1.0


@pytest.mark.parametrize("prediction,reference,expected", [
    ("The answer is Paris, the capital", "Paris", 0.8),
    ("It is Paris.", "paris", 0.8),
    ("(Paris)", "Paris", 0.8),
    ("answer: no", "no", 0.8),
    ("PARIS", "paris", 1.0),
    ("France, Paris and more", "Paris France", 0.6),   # punctuation no longer hides a word
    ("banana apple", "quantum physics", 0.0),
])
def test_genuine_matches_keep_their_scores(prediction, reference, expected):
    assert S.exact_match(prediction, reference) == expected


def test_multi_word_reference_must_appear_as_a_phrase_for_point_eight():
    assert S.exact_match("the capital of France is Paris", "capital of France") == 0.8
    assert S.exact_match("France has a capital city", "capital of France") == 0.0


def test_regex_metacharacters_in_the_reference_are_literal():
    assert S.exact_match("the cost is $5.00 today", "$5.00") == 0.8
    assert S.exact_match("the cost is $5x00 today", "$5.00") == 0.0
    assert S.exact_match("a (b) c", "(b)") == 0.8   # must not raise
