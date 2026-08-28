from src.evaluation.judge.llm_judge import strip_json_fence


def test_strips_json_labeled_fence():
    raw = '```json\n{"score": 8}\n```'
    assert strip_json_fence(raw) == '{"score": 8}'


def test_strips_plain_fence():
    raw = '```\n{"score": 8}\n```'
    assert strip_json_fence(raw) == '{"score": 8}'


def test_returns_unchanged_when_no_fence():
    raw = '{"score": 8}'
    assert strip_json_fence(raw) == raw


def test_handles_empty_string():
    assert strip_json_fence("") == ""
