import uuid
import math
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import json

from src.evaluation.orchestrator import EvalOrchestrator
from src.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

_run_status: dict[str, str] = {}
_run_results: dict[str, dict] = {}


def clean_nans(obj):
    """Recursively replace NaN and Inf values with 0 so JSON serialization works."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    return obj


class RunEvalRequest(BaseModel):
    run_name: str = "eval-run"
    models: list[str] = ["gpt-4o-mini", "claude-3-haiku-20240307"]
    dataset_name: str = "mmlu_sample"
    dataset_version: str = "v1"
    include_judge: bool = True
    include_adversarial: bool = True
    baseline_run_id: str | None = None


async def _run_eval_task(run_id: str, request: RunEvalRequest):
    orchestrator = EvalOrchestrator()
    try:
        _run_status[run_id] = "running"
        result = await orchestrator.run(
            run_id=run_id,
            run_name=request.run_name,
            models=request.models,
            dataset_name=request.dataset_name,
            dataset_version=request.dataset_version,
            include_judge=request.include_judge,
            include_adversarial=request.include_adversarial,
            baseline_run_id=request.baseline_run_id,
        )
        _run_results[run_id] = clean_nans(result)
        _run_status[run_id] = "completed"
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error("eval_run_failed", run_id=run_id, error=str(e), traceback=error_detail)
        _run_status[run_id] = f"failed: {str(e)}"
    finally:
        await orchestrator.close()


@router.post("/run")
async def trigger_eval(request: RunEvalRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())[:8]
    _run_status[run_id] = "pending"
    background_tasks.add_task(_run_eval_task, run_id, request)
    return {
        "run_id": run_id,
        "status": "pending",
        "message": f"Eval run started. Poll /evals/status/{run_id} for updates.",
    }


@router.get("/status/{run_id}")
async def get_run_status(run_id: str):
    if run_id not in _run_status:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "status": _run_status[run_id]}


@router.get("/results/{run_id}")
async def get_run_results(run_id: str):
    if run_id not in _run_status:
        raise HTTPException(status_code=404, detail="Run not found")
    if _run_status[run_id] != "completed":
        raise HTTPException(status_code=202, detail=f"Run status: {_run_status[run_id]}")

    if run_id in _run_results:
        return JSONResponse(content=clean_nans(_run_results[run_id]))

    results_path = Path("results") / f"{run_id}_full_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        return JSONResponse(content=clean_nans(data))

    raise HTTPException(status_code=404, detail="Results not found")


@router.get("/runs")
async def list_runs():
    return {"runs": [{"run_id": k, "status": v} for k, v in _run_status.items()]}


@router.get("/datasets")
async def list_datasets():
    return {
        "datasets": [
            {"name": "mmlu_sample", "version": "v1", "n_prompts": 50, "description": "50 MMLU-style questions across science, math, history, CS, and reasoning"},
            {"name": "reasoning", "version": "v1", "n_prompts": 5, "description": "Step-by-step reasoning tasks"},
            {"name": "instruction_following", "version": "v1", "n_prompts": 5, "description": "Instruction adherence tasks"},
        ]
    }