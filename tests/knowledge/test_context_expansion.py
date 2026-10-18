from types import SimpleNamespace

from consulting_platform.agents.mocks import fixture_chunks
from consulting_platform.knowledge.service import LocalKnowledgeService


async def test_context_expansion_keeps_exact_ids_text_and_document_boundaries():
    first = fixture_chunks()[2].model_copy(update={"page": 3})
    neighbor = first.model_copy(
        update={"chunk_id": "next", "text": "Continued financial metric label.", "page": 4}
    )
    unrelated = first.model_copy(update={"chunk_id": "other-doc", "document_id": "unrelated"})
    distant = first.model_copy(update={"chunk_id": "distant", "page": 15})
    service = LocalKnowledgeService.__new__(LocalKnowledgeService)
    service.retriever = SimpleNamespace(
        chunks=[c.model_dump() for c in [first, neighbor, unrelated, distant]]
    )
    expanded = await service.expand_context([first], max_chunks=1)
    assert [c.chunk_id for c in expanded] == ["next"]
    assert expanded[0].text == neighbor.text and expanded[0].page == 4
    assert first.text == fixture_chunks()[2].text  # no merged/rewritten provenance
    assert await service.expand_context([distant]) == []
