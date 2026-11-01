"""Typed, strict configuration for RAG Foundry.

Sources: init arguments, ``RAG_FOUNDRY_*`` process environment variables and an
optional env file passed explicitly via ``load_settings(env_file=...)``.

Secrets: any future secret must be a ``pydantic.SecretStr`` field read from the
environment or a secret store, and must never get a default value.

Unknown keys: ``extra="forbid"`` rejects unknown init arguments and unknown keys
in an env file (both tested). Pydantic ignores unknown ``RAG_FOUNDRY_*`` process
environment variables, so ``load_settings()`` checks them itself and raises
``ValueError`` listing only the unknown names (never values). Constructing
``Settings()`` directly does not perform this process-env check.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_FOUNDRY_",
        env_file=None,
        extra="forbid",
        frozen=True,
    )

    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(8000, ge=1, le=65535)
    worker_poll_seconds: float = Field(5.0, gt=0)


def load_settings(env_file: str | None = None) -> Settings:
    """Load settings; ``env_file`` is only read when given explicitly."""
    prefix = Settings.model_config["env_prefix"]
    unknown = sorted(
        k for k in os.environ if k.upper().startswith(prefix) and k[len(prefix) :].lower() not in Settings.model_fields
    )
    if unknown:
        raise ValueError(f"Unknown {prefix}* environment variables: {', '.join(unknown)}")
    return Settings(_env_file=env_file)
