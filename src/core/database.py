from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional, Any
import uuid

from src.core.config import get_settings

settings = get_settings()

# ─── Engine ───────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


# ─── Models ───────────────────────────────────────────────────────────────────
class EvalRun(SQLModel, table=True):
    """A single evaluation run across models and datasets."""
    __tablename__ = "eval_runs"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str
    dataset_name: str
    dataset_version: str
    models: list[str] = Field(sa_column=Column(JSON))
    status: str = Field(default="pending")  # pending | running | completed | failed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    mlflow_run_id: Optional[str] = None
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))


class EvalResult(SQLModel, table=True):
    """Results for one model on one prompt."""
    __tablename__ = "eval_results"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    run_id: str = Field(foreign_key="eval_runs.id", index=True)
    model: str
    prompt_id: str
    prompt_text: str
    reference_answer: str
    model_response: str
    rouge_l: float = 0.0
    bert_score: float = 0.0
    exact_match: float = 0.0
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    latency_ms: float = 0.0
    token_cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvalMetric(SQLModel, table=True):
    """Aggregated metrics per run per model."""
    __tablename__ = "eval_metrics"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    run_id: str = Field(foreign_key="eval_runs.id", index=True)
    model: str
    metric_name: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AdversarialResult(SQLModel, table=True):
    """Results from adversarial / red-team prompts."""
    __tablename__ = "adversarial_results"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    run_id: str = Field(foreign_key="eval_runs.id", index=True)
    model: str
    attack_type: str
    prompt: str
    response: str
    was_jailbroken: bool = False
    severity: str = "low"  # low | medium | high | critical
    created_at: datetime = Field(default_factory=datetime.utcnow)
