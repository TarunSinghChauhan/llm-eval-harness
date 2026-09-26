import pytest

from src.evaluation.dataset_registry import DatasetRegistry


def test_reasoning_dataset_loads():
    registry = DatasetRegistry()
    data = registry.load("reasoning", "v1")
    assert len(data["prompts"]) == 5
    assert all("id" in p and "prompt" in p and "reference" in p for p in data["prompts"])


def test_instruction_following_dataset_loads():
    registry = DatasetRegistry()
    data = registry.load("instruction_following", "v1")
    assert len(data["prompts"]) == 5
    assert all("id" in p and "prompt" in p and "reference" in p for p in data["prompts"])


def test_unknown_dataset_raises_file_not_found():
    registry = DatasetRegistry()
    with pytest.raises(FileNotFoundError):
        registry.load("nonexistent_dataset", "v1")


def test_reasoning_dataset_has_hash():
    registry = DatasetRegistry()
    data = registry.load("reasoning", "v1")
    assert "hash" in data
    assert len(data["hash"]) == 12


def test_dataset_persisted_and_reloaded_matches(tmp_path, monkeypatch):
    from src.evaluation import dataset_registry
    monkeypatch.setattr(dataset_registry, "DATASETS_DIR", tmp_path)

    registry = DatasetRegistry()
    first_load = registry.load("reasoning", "v1")

    saved_path = tmp_path / "reasoning_v1.json"
    assert saved_path.exists()

    second_load = registry.load("reasoning", "v1")
    assert first_load["hash"] == second_load["hash"]
    assert first_load["prompts"] == second_load["prompts"]


def test_mmlu_sample_dataset_loads(tmp_path, monkeypatch):
    from src.evaluation import dataset_registry
    monkeypatch.setattr(dataset_registry, "DATASETS_DIR", tmp_path)

    registry = DatasetRegistry()
    data = registry.load("mmlu_sample", "v1")
    assert len(data["prompts"]) == 50
    assert all("id" in p and "prompt" in p and "reference" in p for p in data["prompts"])
    assert data["name"] == "mmlu_sample"
    assert "hash" in data


def test_mmlu_sample_dataset_persisted_and_reloaded(tmp_path, monkeypatch):
    from src.evaluation import dataset_registry
    monkeypatch.setattr(dataset_registry, "DATASETS_DIR", tmp_path)

    registry = DatasetRegistry()
    first_load = registry.load("mmlu_sample", "v1")

    saved_path = tmp_path / "mmlu_sample_v1.json"
    assert saved_path.exists()

    second_load = registry.load("mmlu_sample", "v1")
    assert first_load["hash"] == second_load["hash"]


def test_instruction_following_dataset_generates_fresh(tmp_path, monkeypatch):
    """Forces generation via a fresh tmp_path, since a previously-persisted
    datasets/instruction_following_v1.json on disk would otherwise make
    load() skip straight to the file-read path and never exercise
    _create_instruction_dataset."""
    from src.evaluation import dataset_registry
    monkeypatch.setattr(dataset_registry, "DATASETS_DIR", tmp_path)

    registry = DatasetRegistry()
    data = registry.load("instruction_following", "v1")
    assert len(data["prompts"]) == 5
    assert data["name"] == "instruction_following"
    assert (tmp_path / "instruction_following_v1.json").exists()
