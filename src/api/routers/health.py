from fastapi import APIRouter
from datetime import datetime
from sqlalchemy import text

from src.core.database import engine
from src.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/")
async def health():
    checks = {"database": "unknown"}

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = "unreachable"
        logger.error("health_check_db_failed", error=str(e))

    overall_status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "service": "llm-eval-harness",
        "checks": checks,
    }
