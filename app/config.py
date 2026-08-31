from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import ToolPermission


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Enterprise Research Agent"
    app_env: str = "development"
    max_steps: int = Field(default=8, ge=1, le=32)
    run_timeout_seconds: float = Field(default=30, gt=0, le=300)
    run_token_budget: int | None = Field(default=None, ge=1)
    run_cost_budget_usd: float | None = Field(default=None, gt=0)
    tool_timeout_seconds: float = Field(default=5, gt=0, le=60)
    tool_max_permission: ToolPermission = ToolPermission.HIGH
    max_parallel_tools: int = Field(default=4, ge=1, le=16)
    llm_provider: Literal["deterministic", "openai"] = "deterministic"
    openai_api_key: SecretStr | None = None
    openai_model: str = Field(default="gpt-5-mini", min_length=1)
    openai_base_url: str = Field(default="https://api.openai.com/v1", min_length=1)
    openai_timeout_seconds: float = Field(default=45, gt=0, le=300)
    llm_input_cost_per_million_tokens: float | None = Field(default=None, ge=0)
    llm_output_cost_per_million_tokens: float | None = Field(default=None, ge=0)
    knowledge_backend: Literal["memory", "qdrant"] = "memory"
    knowledge_default_tenant: str = Field(default="demo", min_length=1, max_length=128)
    knowledge_default_principal: str = Field(default="demo-user", min_length=1, max_length=128)
    knowledge_admin_token: SecretStr | None = None
    knowledge_trust_access_headers: bool = False
    knowledge_embedding_dimensions: int = Field(default=256, ge=16, le=4096)
    knowledge_chunk_size: int = Field(default=800, ge=32, le=20_000)
    knowledge_chunk_overlap: int = Field(default=120, ge=0, le=10_000)
    knowledge_ranking: Literal["semantic", "hybrid"] = "semantic"
    knowledge_hybrid_rrf_k: int = Field(default=60, ge=1, le=1_000)
    knowledge_reranker: Literal["none", "token_overlap"] = "none"
    knowledge_rerank_candidate_k: int = Field(default=30, ge=1, le=100)
    knowledge_metadata_filter_keys: str = ""
    qdrant_url: str = Field(default="http://localhost:6333", min_length=1)
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(default="enterprise_knowledge", min_length=1)

    web_search_backend: Literal["stub", "brave"] = "stub"
    brave_search_api_key: SecretStr | None = None
    brave_search_base_url: str = Field(
        default="https://api.search.brave.com/res/v1/web/search", min_length=1
    )
    web_search_timeout_seconds: float = Field(default=10, gt=0, le=60)
    http_fetch_backend: Literal["stub", "safe"] = "stub"
    http_allowed_hosts: str = ""
    http_fetch_timeout_seconds: float = Field(default=10, gt=0, le=60)
    http_fetch_max_bytes: int = Field(default=1_000_000, ge=1_024, le=10_000_000)
    http_fetch_max_redirects: int = Field(default=3, ge=0, le=10)
    http_fetch_allow_http: bool = False

    sql_backend: Literal["stub", "postgres"] = "stub"
    postgres_dsn: SecretStr | None = None
    postgres_allowed_schemas: str = Field(default="public", min_length=1)
    postgres_query_timeout_ms: int = Field(default=5_000, ge=100, le=60_000)
    postgres_max_rows: int = Field(default=500, ge=1, le=10_000)

    python_backend: Literal["stub", "isolated"] = "stub"
    python_worker_timeout_seconds: float = Field(default=5, gt=0, le=60)
    python_worker_max_output_bytes: int = Field(default=65_536, ge=256, le=1_000_000)

    browser_backend: Literal["stub", "playwright"] = "stub"
    browser_allowed_hosts: str = ""
    browser_timeout_seconds: float = Field(default=15, gt=0, le=60)

    mcp_servers_json: str = ""
    mcp_allowed_hosts: str = ""

    state_backend: Literal["memory", "postgres"] = "memory"
    state_postgres_dsn: SecretStr | None = None
    redis_url: SecretStr | None = None
    redis_event_ttl_seconds: int = Field(default=3600, ge=60, le=86_400)

    @model_validator(mode="after")
    def require_rates_for_cost_budget(self) -> Self:
        has_input_rate = self.llm_input_cost_per_million_tokens is not None
        has_output_rate = self.llm_output_cost_per_million_tokens is not None
        if has_input_rate != has_output_rate:
            raise ValueError("LLM input and output token cost rates must be configured together")
        if self.run_cost_budget_usd is not None and not has_input_rate and not has_output_rate:
            raise ValueError("RUN_COST_BUDGET_USD requires configured LLM token cost rates")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
