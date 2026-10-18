"""Service construction (once, at startup) and request-scoped dependencies."""

from __future__ import annotations

import importlib
import logging
import uuid
from dataclasses import dataclass, field

from fastapi import Request

from ..contracts import ConsultingService, KnowledgeService, RetailService, ServiceUnavailableError
from .settings import ApiSettings

log = logging.getLogger("consulting_platform.api")

# Person B's no-argument factory (see docs/INTEGRATION.md). Change here if B picks another name.
KNOWLEDGE_FACTORY = ("consulting_platform.knowledge.service", "create_knowledge_service")


@dataclass
class Services:
    consulting: ConsultingService | None
    knowledge: KnowledgeService | None
    retail: RetailService | None
    warnings: list[str] = field(default_factory=list)


def _load_knowledge() -> KnowledgeService:
    module_name, attr = KNOWLEDGE_FACTORY
    return getattr(importlib.import_module(module_name), attr)()


def build_services(settings: ApiSettings) -> Services:
    """Each subsystem loads independently: one failing never blocks the others."""
    from ..agents import create_consulting_service

    if settings.mode == "demo":
        from ..agents.mocks import FakeKnowledgeService
        from ..retail.fixtures import build_demo_service

        knowledge = FakeKnowledgeService()
        return Services(
            consulting=create_consulting_service(knowledge, settings=settings.core),
            knowledge=knowledge,
            retail=build_demo_service(),
            warnings=["Demo mode: all consulting and retail outputs are synthetic fixtures."],
        )

    warnings: list[str] = []

    retail = None
    try:
        from ..retail.service import RetailModelService

        retail = RetailModelService.from_artifact(settings.retail_model_path, mode="live")
    except FileNotFoundError:
        warnings.append("Retail model artifact not found; run `python scripts/train_retail.py`.")
    except Exception:
        log.exception("retail artifact failed to load")
        warnings.append("Retail model artifact could not be loaded; retrain it.")

    knowledge = None
    try:
        if settings.controlled_evidence_file is not None:
            from ..agents.controlled_evidence import NOTICE, ControlledEvidenceService

            warnings.append(NOTICE)
            knowledge = ControlledEvidenceService.from_file(settings.controlled_evidence_file)
        else:
            knowledge = _load_knowledge()
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("consulting_platform.knowledge"):
            warnings.append("Knowledge service is not installed yet.")
        else:
            warnings.append(
                "Knowledge dependencies are missing; install them with "
                "`python -m pip install -e '.[knowledge]'`."
            )
    except ImportError:
        warnings.append("Knowledge service could not be imported; check its installed dependencies.")
    except Exception:
        log.exception("knowledge service failed to start")
        warnings.append(
            "Controlled evidence file could not be loaded; consulting is unavailable."
            if settings.controlled_evidence_file is not None
            else "Knowledge service failed to start."
        )

    consulting = None
    try:
        if knowledge is not None and all(
            status.status == "unavailable" for status in knowledge.get_status().domains.values()
        ):
            raise ServiceUnavailableError(
                "Knowledge index is unavailable; run `python scripts/build_index.py`."
            )
        consulting = create_consulting_service(knowledge, settings=settings.core)
    except ServiceUnavailableError as exc:
        # Missing knowledge service or credentials: unavailable, never a demo fallback.
        warnings.append(f"Consulting unavailable: {exc}")
    except Exception:
        log.exception("consulting service failed to start")
        warnings.append("Consulting workflow failed to start.")

    return Services(consulting=consulting, knowledge=knowledge, retail=retail, warnings=warnings)


def get_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if request_id is None:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
    return request_id


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_settings(request: Request) -> ApiSettings:
    return request.app.state.settings
