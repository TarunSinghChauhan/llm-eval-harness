from src.evaluation.runner import compute_cache_key, compute_cost, MODEL_PRICING


def test_cache_key_is_deterministic():
    k1 = compute_cache_key("openai/gpt-4o-mini", "What is 2+2?", "You are helpful.")
    k2 = compute_cache_key("openai/gpt-4o-mini", "What is 2+2?", "You are helpful.")
    assert k1 == k2


def test_cache_key_differs_for_different_prompt():
    k1 = compute_cache_key("openai/gpt-4o-mini", "What is 2+2?", "sys")
    k2 = compute_cache_key("openai/gpt-4o-mini", "What is 3+3?", "sys")
    assert k1 != k2


def test_cache_key_differs_for_different_model():
    k1 = compute_cache_key("openai/gpt-4o-mini", "prompt", "sys")
    k2 = compute_cache_key("anthropic/claude-3-haiku", "prompt", "sys")
    assert k1 != k2


def test_cache_key_differs_for_different_system_prompt():
    k1 = compute_cache_key("openai/gpt-4o-mini", "prompt", "sys A")
    k2 = compute_cache_key("openai/gpt-4o-mini", "prompt", "sys B")
    assert k1 != k2


def test_cache_key_has_expected_prefix():
    k = compute_cache_key("openai/gpt-4o-mini", "prompt", "sys")
    assert k.startswith("eval:response:")


def test_compute_cost_known_model():
    cost = compute_cost("openai/gpt-4o-mini", input_tokens=1000, output_tokens=500)
    expected = (1000 * 0.15 + 500 * 0.60) / 1_000_000
    assert cost == expected


def test_compute_cost_free_model_is_zero():
    cost = compute_cost("meta-llama/llama-3.1-8b-instruct:free", input_tokens=1000, output_tokens=1000)
    assert cost == 0.0


def test_compute_cost_unknown_model_defaults_to_zero():
    cost = compute_cost("some/unknown-model", input_tokens=1000, output_tokens=1000)
    assert cost == 0.0


def test_compute_cost_zero_tokens_is_zero():
    cost = compute_cost("openai/gpt-4o-mini", input_tokens=0, output_tokens=0)
    assert cost == 0.0


def test_all_pricing_entries_are_non_negative():
    for model, pricing in MODEL_PRICING.items():
        assert pricing["input"] >= 0
        assert pricing["output"] >= 0
