import asyncio
from pathlib import Path

from consulting_platform.contracts import (
    Domain,
    DomainKnowledgeStatus,
    EvidenceChunk,
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

        # Embedding inference is synchronous; keep the API event loop available
        # for other domains, requests and workflow deadlines.
        chunks = await asyncio.to_thread(
            self.retriever.search,
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
                warnings=["No indexed documents for domains: " + ", ".join(missing_domains)],
            )

        return RetrievalResult(
            mode=self.mode,
            status="ok",
            chunks=chunks,
            warnings=[],
        )

    async def expand_context(self, chunks: list[EvidenceChunk], max_chunks: int = 4):
        """Exact neighboring records, never text merged under an existing ID.

        Optional capability: other KnowledgeService implementations need not add it.
        Domain/company/year are inherited from the same source document and checked
        again at the agent boundary.
        """
        if self.retriever is None or max_chunks <= 0:
            return []
        records = self.retriever.chunks
        positions = {c["chunk_id"]: i for i, c in enumerate(records)}
        seen = {c.chunk_id for c in chunks}
        neighbors = []
        for chunk in chunks:
            position = positions.get(chunk.chunk_id)
            if position is None:
                continue
            for offset in (1, -1):
                index = position + offset
                if not 0 <= index < len(records):
                    continue
                record = records[index]
                if (
                    record["document_id"] != chunk.document_id
                    or record["domain"] != chunk.domain
                    or record.get("company") != chunk.company
                    or record.get("report_year") != chunk.report_year
                    or record.get("source_uri") != chunk.source_uri
                    or record["chunk_id"] in seen
                    or (chunk.page and record.get("page") and abs(record["page"] - chunk.page) > 1)
                ):
                    continue
                neighbors.append(
                    EvidenceChunk.model_validate(
                        {
                            "retrieval_score": 0.0,  # Context expansion is not a scored search hit.
                            **{
                                k: record.get(k)
                                for k in (
                                    "chunk_id",
                                    "document_id",
                                    "text",
                                    "title",
                                    "domain",
                                    "company",
                                    "report_year",
                                    "source_uri",
                                    "page",
                                    "printed_page",
                                )
                            },
                        }
                    )
                )
                seen.add(record["chunk_id"])
                if len(neighbors) >= max_chunks:
                    return neighbors
        return neighbors

    def get_status(self) -> KnowledgeStatus:

        domain_statuses = {}

        for domain in Domain:
            if self.retriever is None:
                domain_statuses[domain] = DomainKnowledgeStatus(
                    status="unavailable",
                    document_count=0,
                    chunk_count=0,
                    warnings=[
                        "Knowledge index is unavailable; run `python scripts/build_index.py` after installing `.[knowledge]`."
                    ],
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
