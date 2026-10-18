"""API settings: Person A's shared Settings plus repo-relative paths for the API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from ..config import Settings
from ..contracts import Mode

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ApiSettings:
    core: Settings = field(default_factory=Settings)
    controlled_evidence_file: Path | None = None

    def __post_init__(self):
        if self.controlled_evidence_file is not None and self.mode != "live":
            raise ValueError("CONSULT_EVIDENCE_FILE requires APP_MODE=live.")

    @property
    def mode(self) -> Mode:
        return self.core.mode

    @property
    def consult_timeout_seconds(self) -> float:
        return self.core.timeout_seconds

    @property
    def artifact_dir(self) -> Path:
        path = self.core.artifact_dir
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def retail_model_path(self) -> Path:
        return self.artifact_dir / "retail" / "model.json"

    @classmethod
    def from_env(cls) -> "ApiSettings":
        env_path = REPO_ROOT / ".env"
        core = Settings.from_env(env_path)
        value = os.environ.get(
            "CONSULT_EVIDENCE_FILE", dotenv_values(env_path).get("CONSULT_EVIDENCE_FILE")
        )
        path = Path(value.strip()) if value and value.strip() else None
        if path is not None and not path.is_absolute():
            path = REPO_ROOT / path
        return cls(core=core, controlled_evidence_file=path)
