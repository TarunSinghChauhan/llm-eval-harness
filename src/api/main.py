from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.database import create_tables
from src.core.logging import setup_logging, get_logger
from src.api.routers import evals, health

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", service="llm-eval-harness")
    await create_tables()
    yield
    logger.info("shutdown", service="llm-eval-harness")


app = FastAPI(
    title="LLM Evaluation Harness",
    description="Production-grade LLM evaluation framework with adversarial red-teaming",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(evals.router, prefix="/evals", tags=["Evaluations"])
