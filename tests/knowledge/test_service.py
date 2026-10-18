
import asyncio

from consulting_platform.contracts import Domain
from consulting_platform.knowledge.service import LocalKnowledgeService


def test_get_status(tmp_path):
    # Use an empty temporary directory instead of the real index.
    service = LocalKnowledgeService(
        index_dir=tmp_path / "missing_index"
    )

    result = service.get_status()

    assert result.mode == "live"
    assert len(result.domains) == 4

    for domain in Domain:
        assert result.domains[domain].status == "unavailable"
        assert result.domains[domain].document_count == 0
        assert result.domains[domain].chunk_count == 0


def test_retrieve_without_index():
    service = LocalKnowledgeService()

    result = asyncio.run(
        service.retrieve(
            query="Cisco Catalyst 8300 specifications",
            domains=[Domain.TECHNOLOGY],
            top_k=5,
        )
    )

    assert result.status == "unavailable"
    assert result.chunks == []
    assert len(result.warnings) > 0
