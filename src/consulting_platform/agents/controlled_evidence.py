"""Preselected real passages for live-model tests, independent of the full KB.

This is a controlled context provider, not a semantic search implementation.
It is selected explicitly at CLI/API startup and never used as an outage fallback.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from consulting_platform.agents.grounding import filter_evidence
from consulting_platform.contracts import (
    ContractModel,
    Domain,
    DomainKnowledgeStatus,
    EvidenceChunk,
    KnowledgeStatus,
    RetrievalFilters,
    RetrievalResult,
    Text,
    unique,
)

NOTICE = (
    "Controlled evidence test: real passages were preselected; "
    "this does not evaluate search quality or full source coverage."
)


class ControlledEvidenceSet(ContractModel):
    provenance: Literal["real_source_passages"]
    description: Text
    chunks: list[EvidenceChunk] = Field(min_length=1)

    @field_validator("chunks")
    @classmethod
    def unique_chunks(cls, value):
        unique([chunk.chunk_id for chunk in value])
        return value


class ControlledEvidenceService:
    """Filter developer-selected passages; preserve order, without query scoring."""

    supports_query_refinement = False

    def __init__(self, evidence: ControlledEvidenceSet):
        self.evidence = evidence

    @classmethod
    def from_file(cls, path: str | Path) -> "ControlledEvidenceService":
        return cls(ControlledEvidenceSet.model_validate_json(Path(path).read_text()))

    async def retrieve(
        self,
        query: str,
        domains: list[Domain],
        filters: RetrievalFilters | None = None,
        top_k: int = 5,
    ) -> RetrievalResult:
        if not query.strip() or not domains or not 1 <= top_k <= 20:
            raise ValueError("Invalid controlled retrieval input")
        filters = filters or RetrievalFilters()
        allowed = []
        for chunk in self.evidence.chunks:
            if chunk.domain in domains:
                matching, _ = filter_evidence([chunk], chunk.domain, filters)
                allowed.extend(matching)
        return RetrievalResult(
            mode="live",
            status="ok",
            chunks=allowed[:top_k],
            warnings=[NOTICE],
        )

    def get_status(self) -> KnowledgeStatus:
        return KnowledgeStatus(
            mode="live",
            domains={
                domain: DomainKnowledgeStatus(
                    status="partial",
                    document_count=len(
                        {c.document_id for c in self.evidence.chunks if c.domain == domain}
                    ),
                    chunk_count=sum(c.domain == domain for c in self.evidence.chunks),
                    warnings=[NOTICE]
                    + (
                        []
                        if any(c.domain == domain for c in self.evidence.chunks)
                        else ["No passages selected for this domain."]
                    ),
                )
                for domain in Domain
            },
        )
