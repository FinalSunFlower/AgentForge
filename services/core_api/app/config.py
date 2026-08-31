from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"
    # The local profile is deliberately zero-infrastructure. Production uses PostgreSQL/Redis.
    database_url: str = "sqlite+aiosqlite:///./agentforge.local.db"
    redis_url: str | None = None
    agent_runtime_url: str = "http://localhost:8101"
    jwt_secret: str = "local-development-secret-change-me"
    jwt_expire_minutes: int = 30
    run_max_steps: int = 8
    run_max_tool_depth: int = 6
    run_timeout_seconds: int = 120
    event_stream_maxlen: int = 10_000
    event_stream_retention_seconds: int = 86_400
    runtime_internal_token: str = "local-runtime-token-change-me"
    webhook_signing_secret: str = "local-webhook-secret-change-me"
    # Public-demo abuse controls. Reuse UsageDaily + budget_exceeded; not a new limiter stack.
    demo_mode: bool = False
    daily_token_budget: int = 200_000
    demo_runs_per_ip_per_hour: int = 5
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def cors_origin_list() -> list[str]:
    return [origin.strip() for origin in get_settings().cors_origins.split(",") if origin.strip()]
