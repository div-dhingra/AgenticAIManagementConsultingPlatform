"""Environment configuration, loaded explicitly; no network or key reads on import."""

from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import Field, SecretStr

from consulting_platform.contracts import ContractModel, Mode


class Settings(ContractModel):
    mode: Mode = "live"
    provider: Literal["cohere", "mistral"] = "cohere"
    api_key: SecretStr = SecretStr("")
    model: str = "command-a-03-2025"
    router_model: str | None = None
    specialist_model: str | None = None
    synthesis_model: str | None = None
    requests_per_minute: int = Field(default=18, ge=1, le=1000)
    max_tokens: int = Field(default=4000, ge=256, le=8000)
    timeout_seconds: float = Field(default=90, gt=0, le=90)
    llm_timeout_seconds: float = Field(default=30, gt=0, le=90)
    max_llm_calls: int = Field(default=12, ge=1, le=12)
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Settings":
        import os

        values = {**(dotenv_values(env_file) if env_file else {}), **os.environ}
        provider = values.get("LLM_PROVIDER", "cohere")
        prefix = provider.upper()
        return cls(
            mode=values.get("APP_MODE", "live"),
            provider=provider,
            api_key=values.get(f"{prefix}_API_KEY") or "",
            model=values.get(f"{prefix}_MODEL")
            or ("command-a-03-2025" if provider == "cohere" else "mistral-small-latest"),
            router_model=values.get(f"{prefix}_ROUTER_MODEL") or None,
            specialist_model=values.get(f"{prefix}_SPECIALIST_MODEL") or None,
            synthesis_model=values.get(f"{prefix}_SYNTHESIS_MODEL") or None,
            requests_per_minute=values.get("LLM_REQUESTS_PER_MINUTE", 18),
            max_tokens=values.get("LLM_MAX_TOKENS", 4000),
            timeout_seconds=values.get("CONSULT_TIMEOUT_SECONDS", 90),
            llm_timeout_seconds=values.get("LLM_TIMEOUT_SECONDS", 30),
            max_llm_calls=values.get("MAX_LLM_CALLS", 12),
            data_dir=values.get("DATA_DIR", "data"),
            artifact_dir=values.get("ARTIFACT_DIR", "artifacts"),
        )
