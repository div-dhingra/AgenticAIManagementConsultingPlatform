from pathlib import Path

from consulting_platform.contracts import (
    Domain,
    DomainKnowledgeStatus,
    KnowledgeStatus,
    RetrievalFilters,
    RetrievalResult,
)

from consulting_platform.knowledge.retrieval import HybridRetriever


class LocalKnowledgeService:

    def __init__(self, index_dir=None):
        self.mode = "live"
        self.retriever = None
        self.load_error = None

        if index_dir is None:
            project_root = Path(__file__).resolve().parents[3]
            index_dir = project_root / "data" / "index"

        try:
            self.retriever = HybridRetriever(index_dir)
        except (OSError, ValueError, ImportError) as error:
            self.load_error = str(error)

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

        if self.retriever is None:
            return RetrievalResult(
                mode=self.mode,
                status="unavailable",
                chunks=[],
                warnings=["Knowledge index is unavailable."],
            )

        chunks = self.retriever.search(
            query=query,
            domains=domains,
            filters=filters,
            top_k=top_k,
        )

        available_domains = set()

        for chunk in self.retriever.chunks:
            available_domains.add(Domain(chunk["domain"]))

        missing_domains = []

        for domain in domains:
            if domain not in available_domains:
                missing_domains.append(domain.value)

        if missing_domains:
            return RetrievalResult(
                mode=self.mode,
                status="partial" if chunks else "unavailable",
                chunks=chunks,
                warnings=[
                    "No indexed documents for domains: "
                    + ", ".join(missing_domains)
                ],
            )

        return RetrievalResult(
            mode=self.mode,
            status="ok",
            chunks=chunks,
            warnings=[],
        )

    def get_status(self) -> KnowledgeStatus:

        domain_statuses = {}

        for domain in Domain:

            if self.retriever is None:
                domain_statuses[domain] = DomainKnowledgeStatus(
                    status="unavailable",
                    document_count=0,
                    chunk_count=0,
                    warnings=["Knowledge index is unavailable."],
                )
                continue

            matching_chunks = []

            for chunk in self.retriever.chunks:
                if chunk["domain"] == domain.value:
                    matching_chunks.append(chunk)

            document_ids = set()

            for chunk in matching_chunks:
                document_ids.add(chunk["document_id"])

            if matching_chunks:
                status = "ready"
                warnings = []
            else:
                status = "unavailable"
                warnings = ["No indexed documents are available."]

            domain_statuses[domain] = DomainKnowledgeStatus(
                status=status,
                document_count=len(document_ids),
                chunk_count=len(matching_chunks),
                warnings=warnings,
            )

        return KnowledgeStatus(
            mode=self.mode,
            domains=domain_statuses,
        )


def create_knowledge_service():
    """
    Factory used by the consulting agent and API.
    """
    return LocalKnowledgeService()