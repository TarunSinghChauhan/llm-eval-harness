from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "service": "llm-eval-harness"}
