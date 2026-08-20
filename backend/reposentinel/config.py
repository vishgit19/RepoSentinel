"""Central configuration for RepoSentinel.

Every tunable knob lives here so that experiments (model comparison, memory
on/off, retrieval strategy) can be reproduced from a single settings object.
Values are read from the environment with the ``REPOSENTINEL_`` prefix, or from
a ``.env`` file at the repository root.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Limits(BaseSettings):
    """Hard stops that bound a single agent run.

    These are guardrails, not hints: the graph checks them between nodes and
    terminates with a partial result rather than exceeding them.
    """

    model_config = SettingsConfigDict(env_prefix="REPOSENTINEL_LIMIT_", extra="ignore")

    max_repair_attempts: int = 3
    max_tool_calls: int = 60
    max_llm_calls: int = 40
    max_tokens: int = 400_000
    max_cost_usd: float = 2.00
    wall_clock_seconds: int = 900
    sandbox_command_seconds: int = 180
    max_tool_output_chars: int = 20_000
    max_file_read_bytes: int = 200_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REPOSENTINEL_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Filesystem layout -------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    workspaces_dir: Path = PROJECT_ROOT / "data" / "workspaces"
    benchmarks_dir: Path = PROJECT_ROOT / "benchmarks"

    # --- Models ------------------------------------------------------------
    # Provider is resolved per-run; this is only the default offered by the UI.
    default_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.0
    llm_request_timeout: int = 120
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # --- Backends (each degrades gracefully; see README) --------------------
    vector_store: Literal["sqlite", "pgvector"] = "sqlite"
    sandbox_backend: Literal["auto", "local", "docker"] = "auto"
    security_backend: Literal["auto", "builtin", "semgrep"] = "auto"
    database_url: str | None = None  # required only for vector_store=pgvector
    sandbox_docker_image: str = "reposentinel-sandbox:latest"

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = 24  # candidates pulled from each retriever
    retrieval_final_k: int = 8  # chunks surviving rerank
    retrieval_graph_hops: int = 1
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_dense_weight: float = 0.6

    # --- Agent behaviour ---------------------------------------------------
    memory_enabled: bool = True
    memory_top_k: int = 3
    require_human_approval: bool = True
    allow_github_push: bool = False
    github_token: str | None = None

    limits: Limits = Field(default_factory=Limits)

    @field_validator("data_dir", "workspaces_dir", "benchmarks_dir", mode="after")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @model_validator(mode="after")
    def _adopt_standard_credential_names(self) -> Settings:
        """Also honour the conventional, unprefixed provider variables.

        Users expect ``OPENAI_API_KEY`` to work; requiring
        ``REPOSENTINEL_OPENAI_API_KEY`` would be a gratuitous surprise. The
        prefixed form still wins when both are set.
        """
        fallbacks = {
            "openai_api_key": "OPENAI_API_KEY",
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "github_token": "GITHUB_TOKEN",
            "database_url": "DATABASE_URL",
        }
        for field_name, env_name in fallbacks.items():
            if getattr(self, field_name):
                continue
            value = os.environ.get(env_name)
            if value:
                object.__setattr__(self, field_name, value)
        return self

    def provider_availability(self) -> dict[str, bool]:
        return {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "ollama": True,  # probed lazily; presence is decided at call time
        }

    @property
    def db_path(self) -> Path:
        return self.data_dir / "reposentinel.db"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.workspaces_dir):
            path.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
