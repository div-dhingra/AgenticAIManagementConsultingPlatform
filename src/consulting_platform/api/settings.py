"""API settings: Person A's shared Settings plus repo-relative paths for the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..contracts import Mode

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ApiSettings:
    core: Settings = field(default_factory=Settings)

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
        return cls(core=Settings.from_env(REPO_ROOT / ".env"))
