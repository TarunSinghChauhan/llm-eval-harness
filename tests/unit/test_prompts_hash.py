from src.evaluation.dataset_registry import compute_prompts_hash


def test_hash_is_12_chars():
    prompts = [{"id": "p1", "prompt": "hi", "reference": "hello"}]
    assert len(compute_prompts_hash(prompts)) == 12


def test_hash_is_deterministic():
    prompts = [{"id": "p1", "prompt": "hi", "reference": "hello"}]
    assert compute_prompts_hash(prompts) == compute_prompts_hash(prompts)


def test_hash_is_independent_of_dict_key_order():
    prompts_a = [{"id": "p1", "prompt": "hi", "reference": "hello"}]
    prompts_b = [{"reference": "hello", "prompt": "hi", "id": "p1"}]
    assert compute_prompts_hash(prompts_a) == compute_prompts_hash(prompts_b)


def test_hash_differs_for_different_content():
    prompts_a = [{"id": "p1", "prompt": "hi"}]
    prompts_b = [{"id": "p1", "prompt": "bye"}]
    assert compute_prompts_hash(prompts_a) != compute_prompts_hash(prompts_b)


def test_hash_differs_for_different_prompt_order_in_list():
    prompts_a = [{"id": "p1"}, {"id": "p2"}]
    prompts_b = [{"id": "p2"}, {"id": "p1"}]
    assert compute_prompts_hash(prompts_a) != compute_prompts_hash(prompts_b)
