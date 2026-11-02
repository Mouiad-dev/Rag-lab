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
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_FOUNDRY_",
        env_file=None,
        extra="forbid",
        frozen=True,
        # Pydantic echoes rejected input in ValidationError text; hide it so a URL with user info is not leaked.
        hide_input_in_errors=True,
    )

    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(8000, ge=1, le=65535)
    worker_poll_seconds: float = Field(5.0, gt=0)
    # Not a secret: a plain str. URL user info, query and fragment are rejected so no credential can hide here.
    ollama_base_url: str = "http://127.0.0.1:11434"

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_base_url(cls, value: str) -> str:
        # urlsplit silently drops whitespace and control characters; reject them so the
        # stored URL is exactly the URL that was checked.
        if any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in value):
            raise ValueError("ollama_base_url must not contain whitespace or control characters")
        parts = urlsplit(value)
        try:
            _ = parts.port
        except ValueError:
            raise ValueError("ollama_base_url has an invalid port") from None
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("ollama_base_url must be an http or https URL with a host")
        if parts.username is not None or parts.password is not None or "@" in parts.netloc:
            raise ValueError("ollama_base_url must not contain user info")
        if "?" in value or "#" in value:
            raise ValueError("ollama_base_url must not contain a query or fragment")
        return value[:-1] if value.endswith("/") else value


def load_settings(env_file: str | None = None) -> Settings:
    """Load settings; ``env_file`` is only read when given explicitly."""
    prefix = Settings.model_config["env_prefix"]
    unknown = sorted(
        k for k in os.environ if k.upper().startswith(prefix) and k[len(prefix) :].lower() not in Settings.model_fields
    )
    if unknown:
        raise ValueError(f"Unknown {prefix}* environment variables: {', '.join(unknown)}")
    return Settings(_env_file=env_file)
