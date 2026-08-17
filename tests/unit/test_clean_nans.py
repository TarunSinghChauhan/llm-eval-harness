import math

from src.api.routers.evals import clean_nans


def test_replaces_nan_with_zero():
    assert clean_nans(float("nan")) == 0.0


def test_replaces_positive_inf_with_zero():
    assert clean_nans(float("inf")) == 0.0


def test_replaces_negative_inf_with_zero():
    assert clean_nans(float("-inf")) == 0.0


def test_leaves_normal_float_unchanged():
    assert clean_nans(3.14) == 3.14


def test_leaves_non_float_values_unchanged():
    assert clean_nans("hello") == "hello"
    assert clean_nans(42) == 42
    assert clean_nans(None) is None
    assert clean_nans(True) is True


def test_cleans_nan_inside_dict():
    data = {"score": float("nan"), "name": "gpt-4o-mini"}
    result = clean_nans(data)
    assert result["score"] == 0.0
    assert result["name"] == "gpt-4o-mini"


def test_cleans_nan_inside_nested_dict():
    data = {"metrics": {"rouge_l": {"mean": float("nan"), "std": 0.05}}}
    result = clean_nans(data)
    assert result["metrics"]["rouge_l"]["mean"] == 0.0
    assert result["metrics"]["rouge_l"]["std"] == 0.05


def test_cleans_nan_inside_list():
    data = [1.0, float("nan"), 3.0, float("inf")]
    result = clean_nans(data)
    assert result == [1.0, 0.0, 3.0, 0.0]


def test_cleans_nan_inside_list_of_dicts():
    data = [{"score": float("nan")}, {"score": 8.5}]
    result = clean_nans(data)
    assert result[0]["score"] == 0.0
    assert result[1]["score"] == 8.5


def test_result_is_json_serializable():
    import json
    data = {"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": float("-inf")}}
    result = clean_nans(data)
    serialized = json.dumps(result)
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
