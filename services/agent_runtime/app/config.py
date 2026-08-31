from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./agentforge.local.db"
    redis_url: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_model: str | None = None
    run_max_steps: int = 8
    run_max_tool_depth: int = 6
    run_timeout_seconds: int = 120
    run_max_context_chars: int = 120_000
    run_keep_recent_messages: int = 6
    run_compress_after_messages: int = 12
    run_max_input_tokens: int = 128_000
    run_max_cost_micros: int = 1_000_000
    model_input_cost_per_1m_micros: int = 0
    model_output_cost_per_1m_micros: int = 0
    runtime_internal_token: str = "local-runtime-token-change-me"
    demo_mode: bool = False
    daily_token_budget: int = 200_000
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
