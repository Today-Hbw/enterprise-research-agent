from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Enterprise Research Agent"
    app_env: str = "development"
    max_steps: int = Field(default=8, ge=1, le=32)
    run_timeout_seconds: float = Field(default=30, gt=0, le=300)
    tool_timeout_seconds: float = Field(default=5, gt=0, le=60)
    max_parallel_tools: int = Field(default=4, ge=1, le=16)
    llm_provider: Literal["deterministic", "openai"] = "deterministic"
    openai_api_key: SecretStr | None = None
    openai_model: str = Field(default="gpt-5-mini", min_length=1)
    openai_base_url: str = Field(default="https://api.openai.com/v1", min_length=1)
    openai_timeout_seconds: float = Field(default=45, gt=0, le=300)


@lru_cache
def get_settings() -> Settings:
    return Settings()
