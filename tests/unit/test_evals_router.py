import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import evals as evals_module
from src.api.routers.evals import RunEvalRequest


@pytest.fixture(autouse=True)
def clean_state():
    """Module-level dicts leak between tests, so reset them around each one."""
    evals_module._run_status.clear()
    evals_module._run_results.clear()
    yield
    evals_module._run_status.clear()
    evals_module._run_results.clear()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(evals_module.router, prefix="/evals")
    return TestClient(app)


def _mock_orchestrator(run_result=None, run_side_effect=None):
    orch = MagicMock()
    orch.run = AsyncMock(return_value=run_result, side_effect=run_side_effect)
    orch.close = AsyncMock()
    return orch


# ---------- RunEvalRequest ----------

def test_run_eval_request_defaults():
    req = RunEvalRequest()
    assert req.run_name == "eval-run"
    assert req.models == ["gpt-4o-mini", "claude-3-haiku-20240307"]
    assert req.dataset_name == "mmlu_sample"
    assert req.dataset_version == "v1"
    assert req.include_judge is True
    assert req.include_adversarial is True
    assert req.baseline_run_id is None


# ---------- POST /run ----------

def test_trigger_eval_returns_pending_and_schedules_task(client):
    with patch.object(evals_module, "_run_eval_task", new=AsyncMock()) as task:
        resp = client.post("/evals/run", json={"run_name": "my-run"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert len(body["run_id"]) == 8
    assert body["run_id"] in body["message"]
    assert evals_module._run_status[body["run_id"]] == "pending"

    task.assert_awaited_once()
    run_id_arg, request_arg = task.await_args.args
    assert run_id_arg == body["run_id"]
    assert request_arg.run_name == "my-run"


def test_trigger_eval_rejects_invalid_body(client):
    resp = client.post("/evals/run", json={"models": "not-a-list"})
    assert resp.status_code == 422


# ---------- GET /status/{run_id} ----------

def test_get_run_status_found(client):
    evals_module._run_status["abc12345"] = "running"
    resp = client.get("/evals/status/abc12345")
    assert resp.status_code == 200
    assert resp.json() == {"run_id": "abc12345", "status": "running"}


def test_get_run_status_not_found(client):
    resp = client.get("/evals/status/missing")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Run not found"


# ---------- GET /results/{run_id} ----------

def test_get_results_unknown_run_is_404(client):
    resp = client.get("/evals/results/missing")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Run not found"


@pytest.mark.parametrize("status", ["pending", "running", "failed: boom"])
def test_get_results_not_completed_returns_202(client, status):
    evals_module._run_status["r1"] = status
    resp = client.get("/evals/results/r1")
    assert resp.status_code == 202
    assert resp.json()["detail"] == f"Run status: {status}"


def test_get_results_from_memory_cleans_nans(client):
    evals_module._run_status["r1"] = "completed"
    evals_module._run_results["r1"] = {
        "score": float("nan"),
        "nested": {"x": float("inf")},
        "items": [1.5, float("-inf")],
    }
    resp = client.get("/evals/results/r1")
    assert resp.status_code == 200
    assert resp.json() == {"score": 0.0, "nested": {"x": 0.0}, "items": [1.5, 0.0]}


def test_get_results_falls_back_to_disk(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r2_full_results.json").write_text(
        json.dumps({"accuracy": 0.9, "models": ["gpt-4o-mini"]})
    )
    evals_module._run_status["r2"] = "completed"

    resp = client.get("/evals/results/r2")
    assert resp.status_code == 200
    assert resp.json() == {"accuracy": 0.9, "models": ["gpt-4o-mini"]}


def test_get_results_disk_fallback_cleans_nans(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    # json.dumps emits bare NaN, which json.load accepts
    (tmp_path / "results" / "r3_full_results.json").write_text(
        json.dumps({"score": float("nan")})
    )
    evals_module._run_status["r3"] = "completed"

    resp = client.get("/evals/results/r3")
    assert resp.status_code == 200
    assert resp.json() == {"score": 0.0}


def test_get_results_completed_but_no_data_is_404(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # empty dir, no results/ folder
    evals_module._run_status["r4"] = "completed"

    resp = client.get("/evals/results/r4")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Results not found"


# ---------- GET /runs ----------

def test_list_runs_empty(client):
    resp = client.get("/evals/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_list_runs_returns_all_statuses(client):
    evals_module._run_status["a"] = "pending"
    evals_module._run_status["b"] = "completed"
    resp = client.get("/evals/runs")
    assert resp.status_code == 200
    assert resp.json()["runs"] == [
        {"run_id": "a", "status": "pending"},
        {"run_id": "b", "status": "completed"},
    ]


# ---------- GET /datasets ----------

def test_list_datasets(client):
    resp = client.get("/evals/datasets")
    assert resp.status_code == 200
    datasets = resp.json()["datasets"]
    assert [d["name"] for d in datasets] == [
        "mmlu_sample",
        "reasoning",
        "instruction_following",
    ]
    assert datasets[0]["n_prompts"] == 50
    assert all(d["version"] == "v1" for d in datasets)


# ---------- _run_eval_task ----------

def test_run_eval_task_success_stores_cleaned_result_and_closes():
    orch = _mock_orchestrator(run_result={"score": float("nan"), "ok": 1.0})
    request = RunEvalRequest(
        run_name="n",
        models=["m1"],
        dataset_name="reasoning",
        dataset_version="v2",
        include_judge=False,
        include_adversarial=False,
        baseline_run_id="base1",
    )

    with patch("src.api.routers.evals.EvalOrchestrator", return_value=orch):
        asyncio.run(evals_module._run_eval_task("run1", request))

    orch.run.assert_awaited_once_with(
        run_id="run1",
        run_name="n",
        models=["m1"],
        dataset_name="reasoning",
        dataset_version="v2",
        include_judge=False,
        include_adversarial=False,
        baseline_run_id="base1",
    )
    assert evals_module._run_status["run1"] == "completed"
    assert evals_module._run_results["run1"] == {"score": 0.0, "ok": 1.0}
    orch.close.assert_awaited_once()


def test_run_eval_task_failure_records_status_and_still_closes():
    orch = _mock_orchestrator(run_side_effect=RuntimeError("boom"))

    with patch("src.api.routers.evals.EvalOrchestrator", return_value=orch):
        # Must swallow the exception rather than propagate it to the background runner
        asyncio.run(evals_module._run_eval_task("run2", RunEvalRequest()))

    assert evals_module._run_status["run2"] == "failed: boom"
    assert "run2" not in evals_module._run_results
    orch.close.assert_awaited_once()
