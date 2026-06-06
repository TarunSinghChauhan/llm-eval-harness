from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ← this is the key fix
    )

    # API Keys
    openrouter_api_key: str = ""
    langchain_api_key: str = ""
    cohere_api_key: str = ""

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_project: str = "llm-eval-harness"

    # Database
    database_url: str = "postgresql+asyncpg://eval_user:eval_pass@localhost:5432/eval_db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "llm-eval-harness"

    # Eval settings
    eval_batch_size: int = 10
    eval_max_concurrent: int = 5
    bootstrap_iterations: int = 1000
    regression_threshold: float = 0.02
    slack_webhook_url: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()