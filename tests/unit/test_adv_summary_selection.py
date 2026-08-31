from src.evaluation.orchestrator import select_primary_adversarial_summary


def test_selects_summary_for_first_model():
    adversarial_by_model = {
        "gpt-4o-mini": {"safety_rate": 0.9},
        "claude-3-haiku": {"safety_rate": 0.95},
    }
    result = select_primary_adversarial_summary(adversarial_by_model, ["gpt-4o-mini", "claude-3-haiku"])
    assert result == {"safety_rate": 0.9}


def test_returns_none_when_dict_is_empty():
    result = select_primary_adversarial_summary({}, ["gpt-4o-mini"])
    assert result is None


def test_returns_none_when_dict_is_none_falsy():
    result = select_primary_adversarial_summary(None, ["gpt-4o-mini"])
    assert result is None


def test_returns_none_if_first_model_has_no_entry():
    adversarial_by_model = {"claude-3-haiku": {"safety_rate": 0.95}}
    result = select_primary_adversarial_summary(adversarial_by_model, ["gpt-4o-mini", "claude-3-haiku"])
    assert result is None
