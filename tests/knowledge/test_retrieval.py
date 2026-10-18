
import asyncio

import pytest

from consulting_platform.contracts import (
    Domain,
    KnowledgeService,
    RetrievalFilters,
)
from consulting_platform.knowledge.service import (
    LocalKnowledgeService,
    create_knowledge_service,
)


@pytest.fixture
def indexed_service():
    """
    Create a knowledge service with a real index.

    Skip integration tests if the index is unavailable.
    """
    service = create_knowledge_service()

    if service.retriever is None:
        pytest.skip("Real knowledge index is not available.")

    return service


def test_factory_implements_protocol():
    service = create_knowledge_service()

    assert isinstance(service, KnowledgeService)


def test_domain_status(indexed_service):
    service = indexed_service

    status = service.get_status()

    assert status.mode == "live"
    assert set(status.domains.keys()) == set(Domain)

    assert status.domains[Domain.SEC].status == "ready"
    assert status.domains[Domain.SEC].document_count > 0
    assert status.domains[Domain.SEC].chunk_count > 0


def test_cisco_filter(indexed_service):
    service = indexed_service

    async def run_test():
        return await service.retrieve(
            query="Cisco risk factors",
            domains=[Domain.SEC],
            filters=RetrievalFilters(
                companies=["Cisco"],
                report_years=[2025],
            ),
            top_k=3,
        )

    result = asyncio.run(run_test())

    assert result.mode == "live"
    assert result.status == "ok"
    assert len(result.chunks) > 0
    assert len(result.chunks) <= 3

    for chunk in result.chunks:
        assert chunk.domain == Domain.SEC
        assert chunk.company == "Cisco"
        assert chunk.report_year == 2025
        assert 0.0 <= chunk.retrieval_score <= 1.0


def test_missing_domain(indexed_service):
    service = indexed_service

    async def run_test():
        return await service.retrieve(
            query="Cisco router specifications",
            domains=[Domain.TECHNOLOGY],
            top_k=3,
        )

    result = asyncio.run(run_test())

    assert result.mode == "live"
    assert result.status == "ok"
    assert len(result.chunks) > 0
    assert len(result.chunks) <= 3

    for chunk in result.chunks:
        assert chunk.domain == Domain.TECHNOLOGY
        assert chunk.company == "Cisco"
        assert chunk.source_uri.startswith("https://")


def test_missing_index(tmp_path):
    missing_index_dir = tmp_path / "missing_index"

    service = LocalKnowledgeService(
        index_dir=missing_index_dir,
    )

    status = service.get_status()

    assert status.mode == "live"

    for domain in Domain:
        assert status.domains[domain].status == "unavailable"
        assert status.domains[domain].document_count == 0
        assert status.domains[domain].chunk_count == 0

    async def run_test():
        return await service.retrieve(
            query="Cisco risk factors",
            domains=[Domain.SEC],
            top_k=3,
        )

    result = asyncio.run(run_test())

    assert result.mode == "live"
    assert result.status == "unavailable"
    assert result.chunks == []
    assert len(result.warnings) > 0


def test_strict_company_filter(indexed_service):
    service = indexed_service

    async def run_test():
        return await service.retrieve(
            query="Cisco risk factors",
            domains=[Domain.SEC],
            filters=RetrievalFilters(
                companies=["CompanyThatDoesNotExist"],
            ),
            top_k=3,
        )

    result = asyncio.run(run_test())

    assert result.mode == "live"
    assert result.status == "ok"
    assert result.chunks == []
    assert result.warnings == []
