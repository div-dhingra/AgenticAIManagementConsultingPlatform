"""Thin HTTP handlers: validate, call the injected service, validate the result."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from ..contracts import (
    CatalogResponse,
    ConsultRequest,
    ConsultResponse,
    Domain,
    DomainKnowledgeStatus,
    HealthResponse,
    KnowledgeStatus,
    RetailRequest,
    RetailResponse,
    ServiceReadiness,
    UpstreamTimeoutError,
)
from .deps import Services, get_request_id, get_services, get_settings
from .errors import ContractViolation, PublicServiceUnavailable
from .settings import ApiSettings

router = APIRouter()


@router.post("/consult", response_model=ConsultResponse)
async def consult(
    body: ConsultRequest,
    request_id: str = Depends(get_request_id),
    services: Services = Depends(get_services),
    settings: ApiSettings = Depends(get_settings),
) -> ConsultResponse:
    if services.consulting is None:
        raise PublicServiceUnavailable("Consulting is unavailable: the agent workflow is not loaded.")
    try:
        result = await asyncio.wait_for(
            services.consulting.run_consultation(body, request_id=request_id),
            timeout=settings.consult_timeout_seconds,
        )
    except TimeoutError as exc:  # wait_for cancels the workflow task before raising
        raise UpstreamTimeoutError() from exc

    result = ConsultResponse.model_validate(result.model_dump())
    if result.request_id != request_id or result.mode != settings.mode:
        raise ContractViolation("consulting response request_id/mode mismatch")
    if len(result.recommendations) > body.max_recommendations:
        raise ContractViolation("consulting response exceeds max_recommendations")
    return result


def _retail(services: Services):
    if services.retail is None:
        raise PublicServiceUnavailable("Retail recommendations are unavailable: the model artifact is not loaded.")
    return services.retail


@router.get("/catalog", response_model=CatalogResponse)
def catalog(request_id: str = Depends(get_request_id), services: Services = Depends(get_services)) -> CatalogResponse:
    return _retail(services).get_catalog(request_id=request_id)


@router.post("/recommend", response_model=RetailResponse)
def recommend(
    body: RetailRequest,
    request_id: str = Depends(get_request_id),
    services: Services = Depends(get_services),
) -> RetailResponse:
    return _retail(services).recommend(body, request_id=request_id)


def _knowledge_status(services: Services, settings: ApiSettings, warnings: list[str]) -> KnowledgeStatus:
    if services.knowledge is not None:
        try:
            return KnowledgeStatus.model_validate(services.knowledge.get_status().model_dump())
        except Exception:
            warnings.append("Knowledge status check failed.")
    reason = "Knowledge service is not loaded."
    return KnowledgeStatus(
        mode=settings.mode,
        domains={d: DomainKnowledgeStatus(status="unavailable", document_count=0, chunk_count=0, warnings=[reason]) for d in Domain},
    )


@router.get("/health", response_model=HealthResponse)
def health(
    request_id: str = Depends(get_request_id),
    services: Services = Depends(get_services),
    settings: ApiSettings = Depends(get_settings),
) -> HealthResponse:
    """Readiness from loaded objects only; never calls paid model inference."""
    warnings = list(services.warnings)
    knowledge = _knowledge_status(services, settings, warnings)
    domain_states = {d.status for d in knowledge.domains.values()}
    if services.knowledge is None or domain_states == {"unavailable"}:
        knowledge_state = "unavailable"
    elif domain_states == {"ready"}:
        knowledge_state = "ready"
    else:
        knowledge_state = "partial"

    consulting_ready = services.consulting is not None
    is_ready = getattr(services.consulting, "is_ready", None)
    if consulting_ready and callable(is_ready):
        consulting_ready = bool(is_ready())

    states = ServiceReadiness(
        consulting="ready" if consulting_ready else "unavailable",
        knowledge=knowledge_state,
        retail="ready" if services.retail is not None else "unavailable",
    )
    all_ready = states.consulting == "ready" and states.knowledge == "ready" and states.retail == "ready"
    return HealthResponse(
        request_id=request_id,
        mode=settings.mode,
        status="ok" if all_ready else "degraded",
        services=states,
        knowledge=knowledge,
        warnings=warnings,
    )
