
from consulting_platform.contracts import (
    Domain,
    DomainKnowledgeStatus,
    KnowledgeStatus,
    RetrievalFilters,
    RetrievalResult,
)


class LocalKnowledgeService:
    def __init__(self):
        self.mode = "live"

    async def retrieve(
        self,
        query: str,
        domains: list[Domain],
        filters: RetrievalFilters | None = None,
        top_k: int = 5,
    ) -> RetrievalResult:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not domains:
            raise ValueError("At least one domain is required.")

        if len(set(domains)) != len(domains):
            raise ValueError("Duplicate domains are not allowed.")

        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")

        return RetrievalResult(
            mode=self.mode,
            status="unavailable",
            chunks=[],
            warnings=[
                "Knowledge index has not been built yet."
            ],
        )

    def get_status(self) -> KnowledgeStatus:

        domain_statuses = {}

        for domain in Domain:
            domain_statuses[domain] = DomainKnowledgeStatus(
                status="unavailable",
                document_count=0,
                chunk_count=0,
                warnings=[
                    "No indexed documents are available."
                ],
            )

        return KnowledgeStatus(
            mode=self.mode,
            domains=domain_statuses,
        )
