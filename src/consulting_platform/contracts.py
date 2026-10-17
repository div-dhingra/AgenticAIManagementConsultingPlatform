"""Version 1.0 integration contract. Import these models instead of redefining them."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Protocol, Self, runtime_checkable
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Year = Annotated[int, Field(strict=True, ge=1900, le=2100)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Mode = Literal["live", "demo"]


class Domain(StrEnum):
    TECHNOLOGY = "technology"
    CLINICAL = "clinical"
    SEC = "sec"
    CORPORATE_RESPONSIBILITY = "corporate_responsibility"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


def unique(values: list) -> list:
    if len(values) != len(set(values)):
        raise ValueError("Duplicate values are not allowed.")
    return values


def safe_source_uri(value: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(value)
    if parsed.scheme == "https" and parsed.netloc and not parsed.username and not parsed.password:
        return value
    path = PurePosixPath(value)
    if parsed.scheme or value.startswith(("/", "\\")) or "\\" in value or ".." in path.parts:
        raise ValueError("Source must be an HTTPS URL or safe repository-relative path.")
    return value


class ConsultContext(ContractModel):
    company: Text | None = None
    product_model: Text | None = None
    symptoms: list[Text] = Field(default_factory=list)
    requirements: list[Text] = Field(default_factory=list)
    report_years: list[Year] = Field(default_factory=list)
    region: Text | None = None

    _years_unique = field_validator("report_years")(unique)


class ConsultRequest(ContractModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]
    context: ConsultContext = Field(default_factory=ConsultContext)
    max_recommendations: Annotated[int, Field(strict=True, ge=1, le=5)] = 3


class RetrievalFilters(ContractModel):
    companies: list[Text] = Field(default_factory=list)
    report_years: list[Year] = Field(default_factory=list)
    document_ids: list[Text] = Field(default_factory=list)

    _unique = field_validator("companies", "report_years", "document_ids")(unique)


class EvidenceChunk(ContractModel):
    chunk_id: Text
    document_id: Text
    text: Text
    title: Text
    domain: Domain
    company: Text | None
    report_year: Year | None
    source_uri: Text
    page: PositiveInt | None
    printed_page: Text | None
    retrieval_score: Score

    _source = field_validator("source_uri")(safe_source_uri)


class RetrievalResult(ContractModel):
    mode: Mode
    status: Literal["ok", "partial", "unavailable"]
    chunks: list[EvidenceChunk]
    warnings: list[Text]

    @model_validator(mode="after")
    def valid_chunks(self) -> Self:
        unique([c.chunk_id for c in self.chunks])
        if self.status == "unavailable" and self.chunks:
            raise ValueError("Unavailable retrieval cannot contain evidence.")
        return self


class DomainKnowledgeStatus(ContractModel):
    status: Literal["ready", "partial", "unavailable"]
    document_count: NonnegativeInt
    chunk_count: NonnegativeInt
    warnings: list[Text]


class KnowledgeStatus(ContractModel):
    mode: Mode
    domains: dict[Domain, DomainKnowledgeStatus]

    @field_validator("domains")
    @classmethod
    def all_domains(cls, value: dict) -> dict:
        if set(value) != set(Domain):
            raise ValueError("Knowledge status must describe all four domains.")
        return value


class Finding(ContractModel):
    id: Text
    domain: Domain
    statement: Text
    assessment: Literal["supported", "inference"]
    evidence_ids: list[Text] = Field(min_length=1)

    _ids_unique = field_validator("evidence_ids")(unique)


class ConsultRecommendation(ContractModel):
    id: Text
    rank: PositiveInt
    kind: Literal["product", "component", "business_action", "diagnostic_action"]
    title: Text
    rationale: Text
    priority: Literal["high", "medium", "low"]
    assessment: Literal["supported", "conditional", "proposed"]
    evidence_ids: list[Text] = Field(min_length=1)
    prerequisites: list[Text]
    next_steps: list[Text]
    success_metrics: list[Text]

    _ids_unique = field_validator("evidence_ids")(unique)

    @model_validator(mode="after")
    def conditional_needs_prerequisites(self) -> Self:
        if self.assessment == "conditional" and not self.prerequisites:
            raise ValueError("Conditional recommendations require prerequisites.")
        return self


class Citation(ContractModel):
    evidence_id: Text
    document_id: Text
    title: Text
    domain: Domain
    company: Text | None
    report_year: Year | None
    source_uri: Text
    page: PositiveInt | None
    printed_page: Text | None
    excerpt: Annotated[str, StringConstraints(min_length=1, max_length=400)]

    _source = field_validator("source_uri")(safe_source_uri)


class Uncertainty(ContractModel):
    level: Literal["low", "medium", "high"]
    reasons: list[Text]


class TraceEvent(ContractModel):
    step: Literal["route", "retrieve", "rank", "validate", "synthesize"]
    domain: Domain | None
    status: Literal["completed", "skipped", "failed"]
    detail: Text


class ResponseBase(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    mode: Mode

    @field_validator("request_id")
    @classmethod
    def uuid_string(cls, value: str) -> str:
        UUID(value)
        return value


class ConsultResponse(ResponseBase):
    status: Literal["answered", "partial", "needs_clarification", "insufficient_evidence"]
    domains: list[Domain]
    summary: Text
    findings: list[Finding]
    recommendations: list[ConsultRecommendation]
    citations: list[Citation]
    clarification_questions: list[Text]
    uncertainty: Uncertainty
    trace: list[TraceEvent]
    warnings: list[Text]

    _domains_unique = field_validator("domains")(unique)

    @model_validator(mode="after")
    def references(self) -> Self:
        ids = unique([c.evidence_id for c in self.citations])
        unique([f.id for f in self.findings])
        unique([r.id for r in self.recommendations])
        for item in [*self.findings, *self.recommendations]:
            if not set(item.evidence_ids) <= set(ids):
                raise ValueError("Evidence references must resolve to citations.")
        if any(c.domain not in self.domains for c in self.citations):
            raise ValueError("Citation domain must be selected.")
        if any(f.domain not in self.domains for f in self.findings):
            raise ValueError("Finding domain must be selected.")
        if [r.rank for r in self.recommendations] != list(range(1, len(self.recommendations) + 1)):
            raise ValueError("Recommendation ranks must be contiguous.")
        if self.status == "needs_clarification" and not self.clarification_questions:
            raise ValueError("Clarification status requires questions.")
        return self


class Product(ContractModel):
    product_id: Text
    name: Text


class CatalogResponse(ResponseBase):
    products: list[Product]

    @field_validator("products")
    @classmethod
    def sorted_products(cls, value: list[Product]) -> list[Product]:
        ids = unique([p.product_id for p in value])
        if ids != sorted(ids):
            raise ValueError("Catalog must be sorted by product_id.")
        return value


class RetailRequest(ContractModel):
    cart_items: list[Text] = Field(default_factory=list, max_length=40)
    top_k: Annotated[int, Field(strict=True, ge=1, le=10)] = 5

    _unique_cart = field_validator("cart_items")(unique)


class RetailRecommendation(Product):
    rank: PositiveInt
    score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    support: Score
    confidence: Score
    lift: Annotated[float, Field(gt=1, allow_inf_nan=False)]
    evidence_count: Annotated[int, Field(strict=True, ge=3)]
    antecedent_count: Annotated[int, Field(strict=True, ge=3)]
    reason: Text

    @model_validator(mode="after")
    def counts(self) -> Self:
        if self.evidence_count > self.antecedent_count or not math.isclose(
            self.confidence, self.evidence_count / self.antecedent_count, abs_tol=1e-8
        ):
            raise ValueError("Confidence must match the evidence counts.")
        return self


class RetailResponse(ResponseBase):
    status: Literal["ok", "empty_cart", "no_supported_pattern"]
    cart_items: list[Text]
    recommendations: list[RetailRecommendation]
    explanation: Text
    model_version: Text
    warnings: list[Text]

    @model_validator(mode="after")
    def valid_recommendations(self) -> Self:
        unique(self.cart_items)
        ids = unique([r.product_id for r in self.recommendations])
        if set(ids) & set(self.cart_items):
            raise ValueError("Recommendations cannot already be in the cart.")
        if [r.rank for r in self.recommendations] != list(range(1, len(ids) + 1)):
            raise ValueError("Ranks must be contiguous.")
        if self.status != "ok" and self.recommendations:
            raise ValueError("An abstention must have no recommendations.")
        if self.status == "empty_cart" and self.cart_items:
            raise ValueError("empty_cart status requires an empty cart.")
        return self


class ServiceReadiness(ContractModel):
    consulting: Literal["ready", "unavailable"]
    knowledge: Literal["ready", "partial", "unavailable"]
    retail: Literal["ready", "unavailable"]


class HealthResponse(ResponseBase):
    status: Literal["ok", "degraded"]
    services: ServiceReadiness
    knowledge: KnowledgeStatus
    warnings: list[Text]


class ErrorDetail(ContractModel):
    field: Text
    message: Text


class ErrorBody(ContractModel):
    code: Text
    message: Text
    details: list[ErrorDetail]


class ErrorResponse(ResponseBase):
    error: ErrorBody


class ServiceUnavailableError(RuntimeError):
    """Map to HTTP 503 without exposing internal details."""


class UpstreamTimeoutError(RuntimeError):
    """Map to HTTP 504."""


class UpstreamModelError(RuntimeError):
    """Map to HTTP 502. Never attach raw provider bodies to HTTP responses."""


@runtime_checkable
class KnowledgeService(Protocol):
    async def retrieve(
        self,
        query: str,
        domains: list[Domain],
        filters: RetrievalFilters | None = None,
        top_k: int = 5,
    ) -> RetrievalResult: ...

    def get_status(self) -> KnowledgeStatus: ...


@runtime_checkable
class ConsultingService(Protocol):
    async def run_consultation(
        self,
        request: ConsultRequest,
        *,
        request_id: str,
    ) -> ConsultResponse: ...


@runtime_checkable
class RetailService(Protocol):
    def get_catalog(self, *, request_id: str) -> CatalogResponse: ...

    def recommend(self, request: RetailRequest, *, request_id: str) -> RetailResponse: ...
