"""Relevant, true datasheet facts must not masquerade as a fault diagnosis."""

from uuid import uuid4

import pytest

from consulting_platform.agents.diagnostics import compatible_with_reported_model, diagnostic_intake
from consulting_platform.agents.mocks import FakeKnowledgeService, ScriptedLLMClient
from consulting_platform.agents.service import create_consulting_service
from consulting_platform.config import Settings
from consulting_platform.contracts import ConsultContext, ConsultRequest, Domain


@pytest.mark.parametrize("model", ["", "Cisco Catalyst 8300", "C8300-2N2S-6T"])
async def test_model_or_reviewer_cannot_turn_specs_into_a_repair(model):
    kb = FakeKnowledgeService()
    chunk = next(c for c in kb.chunks if c.domain == Domain.TECHNOLOGY)
    text = "The router supports dual power supplies. SRST supports voice survivability."
    kb.chunks = [chunk.model_copy(update={"text": text})]
    llm = ScriptedLLMClient(
        {
            "technology": [
                {
                    "findings": [
                        {
                            "statement": "The router supports dual power supplies.",
                            "supports": [
                                {
                                    "evidence_id": chunk.chunk_id,
                                    "quote": "The router supports dual power supplies.",
                                }
                            ],
                        }
                    ]
                }
            ],
            "review": [
                {"accepted_finding_ids": ["technology:f1"], "accepted_recommendation_ids": []}
            ],
        }
    )
    response = await create_consulting_service(
        kb, settings=Settings(mode="demo"), llm=llm
    ).run_consultation(
        ConsultRequest(
            question="Our Cisco router has intermittent connectivity failures. Which replacement parts should we order and how can we prevent outages?",
            context=ConsultContext(product_model=model or None),
        ),
        request_id=str(uuid4()),
    )
    assert response.status == "needs_clarification"
    assert "Do not order replacement parts yet" in response.summary
    assert "Suggested diagnostic plan" in response.summary
    assert "outage frequency and duration" in response.summary
    assert "dual power supplies" not in response.summary
    assert not response.recommendations
    assert any("fault logs" in q for q in response.clarification_questions)
    assert any("exact router model" in q for q in response.clarification_questions) == (
        model != "C8300-2N2S-6T"
    )


async def test_supplied_diagnostic_account_is_not_reasked_or_confirmed_by_datasheet():
    request = ConsultRequest(
        question="Our Cisco C8300-2N2S-6T router has intermittent failures. Diagnostic logs show interface errors after each drop. Which replacement parts should we order?"
    )
    assert not diagnostic_intake(request).questions
    response = await create_consulting_service(settings=Settings(mode="demo")).run_consultation(
        request, request_id=str(uuid4())
    )
    assert response.status != "answered"
    assert "Do not order replacement parts yet" in response.summary
    assert not any(r.kind == "component" for r in response.recommendations)


def test_procurement_comparison_is_not_an_incident():
    request = ConsultRequest(
        question="Compare Cisco routers for at least 5 ports; overall deployment suitability remains unverified."
    )
    assert not diagnostic_intake(request).requested
    planned = ConsultRequest(
        question="Compare Cisco router power redundancy options to prevent future outages."
    )
    assert not diagnostic_intake(planned).requested


def test_model_compatibility_is_not_evidence_of_a_failed_component():
    from consulting_platform.agents.models import Candidate

    request = ConsultRequest(
        question="Our Cisco C8300-2N2S-6T router has intermittent failures. Diagnostic logs show interface errors after each drop. Which replacement parts should we order?"
    )
    candidate = Candidate.model_validate(
        {
            "id": "psu",
            "label": "Example PSU",
            "identity_supports": [{"evidence_id": "fixture", "quote": "Example PSU"}],
            "attributes": {
                "compatible_model": {
                    "value": "C8300-2N2S-6T",
                    "supports": [
                        {"evidence_id": "fixture", "quote": "Compatible with C8300-2N2S-6T"}
                    ],
                }
            },
        }
    )
    assert not compatible_with_reported_model(candidate, diagnostic_intake(request))


def test_negated_or_merely_requested_diagnostics_are_not_observations():
    for details in ["We have no diagnostic logs.", "What diagnostic tests should we run?"]:
        assert not diagnostic_intake(
            ConsultRequest(question="Our Cisco router has intermittent failures. " + details)
        ).observations_supplied


async def test_agilent_laboratory_task_cannot_use_cisco_technology_index():
    kb = FakeKnowledgeService()
    llm = ScriptedLLMClient(
        {
            "router": [
                {
                    "tasks": {
                        "technology": {
                            "query": "What do Agilent Seahorse XF laboratory analyzers measure?"
                        }
                    }
                }
            ]
        }
    )
    response = await create_consulting_service(
        kb, settings=Settings(mode="demo"), llm=llm
    ).run_consultation(
        ConsultRequest(question="What do Agilent Seahorse XF laboratory analyzers measure?"),
        request_id=str(uuid4()),
    )
    assert response.domains == [Domain.CLINICAL]
    assert kb.calls and all(call["domains"] == [Domain.CLINICAL] for call in kb.calls)
