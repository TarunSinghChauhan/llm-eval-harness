from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import health as health_module


def _client():
    app = FastAPI()
    app.include_router(health_module.router, prefix="/health")
    return TestClient(app)


def _assert_utc_timestamp(value: str):
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_health_timestamp_is_timezone_aware_utc_when_db_ok():
    conn = MagicMock()
    conn.execute = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    fake_engine = MagicMock()
    fake_engine.connect.return_value = ctx

    with patch.object(health_module, "engine", fake_engine):
        resp = _client().get("/health/")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"database": "ok"}
    assert body["service"] == "llm-eval-harness"
    _assert_utc_timestamp(body["timestamp"])


def test_health_reports_degraded_with_utc_timestamp_when_db_unreachable():
    fake_engine = MagicMock()
    fake_engine.connect.side_effect = RuntimeError("db down")

    with patch.object(health_module, "engine", fake_engine):
        resp = _client().get("/health/")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"] == {"database": "unreachable"}
    _assert_utc_timestamp(body["timestamp"])
