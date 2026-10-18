from uuid import uuid4

import pytest

from consulting_platform.agents.mocks import FakeKnowledgeService, ScriptedLLMClient
from consulting_platform.agents.service import create_consulting_service
from consulting_platform.config import Settings
from consulting_platform.contracts import ConsultRequest, Domain


@pytest.mark.parametrize("accept_paraphrase", [True, False])
async def test_paraphrase_needs_semantic_acceptance_while_exact_proof_survives(accept_paraphrase):
    kb = FakeKnowledgeService()
    chunk = next(c for c in kb.chunks if c.domain == Domain.SEC)
    quote = "Changes in demand could adversely affect our operating results."
    kb.chunks = [chunk.model_copy(update={"text": quote})]
    paraphrase = "Cisco's 2025 10-K identifies operating risk when customer demand changes."
    accepted = ["sec:f1", "sec:explain1"] if accept_paraphrase else ["sec:f1"]
    llm = ScriptedLLMClient(
        {
            "sec": [
                {
                    "findings": [
                        {
                            "statement": paraphrase,
                            "supports": [{"evidence_id": chunk.chunk_id, "quote": quote}],
                        }
                    ]
                }
            ],
            "review": [{"accepted_finding_ids": accepted, "accepted_recommendation_ids": []}],
        }
    )
    response = await create_consulting_service(
        kb, settings=Settings(mode="demo"), llm=llm
    ).run_consultation(
        ConsultRequest(question="Summarize Cisco 2025 filing risks."), request_id=str(uuid4())
    )
    assert (paraphrase in response.summary) == accept_paraphrase
    assert any(quote in f.statement for f in response.findings)
    assert response.citations and "[1]" in response.summary
    assert response.citations[0].excerpt == quote
    assert "Cisco — fiscal 2025" in response.summary
    assert response.summary.count("filing discloses") == 0
