from uuid import uuid4

import pytest

from consulting_platform.agents.controlled_evidence import ControlledEvidenceService
from consulting_platform.agents.csr_actions import qualify_csr_actions
from consulting_platform.agents.grounding import ground_analysis
from consulting_platform.agents.mocks import ScriptedLLMClient
from consulting_platform.agents.models import SpecialistAnalysis
from consulting_platform.agents.service import create_consulting_service
from consulting_platform.config import Settings
from consulting_platform.contracts import ConsultRequest


def example():
    kb = ControlledEvidenceService.from_file("fixtures/consulting/controlled_real_passages.json")
    chunk = kb.evidence.chunks[-1]
    quote = chunk.text[chunk.text.index("data sources") : chunk.text.index("This report")].strip()

    def action(title, steps):
        return {
            "kind": "business_action",
            "title": title,
            "rationale": "Connectivity can help meet the report's data needs.",
            "supports": [{"evidence_id": chunk.chunk_id, "quote": quote}],
            "next_steps": steps,
            "success_metrics": ["Completeness of data in a proposed pilot."],
        }

    return (
        kb,
        quote,
        {
            "findings": [],
            "actions": [
                action(
                    "Assess connectivity options for nutrition-program data sharing",
                    ["Conduct connectivity audit", "Define data sharing standards"],
                ),
                action(
                    "Invest in digital infrastructure",
                    ["Procure connectivity solutions", "Implement data platform"],
                ),
            ],
        },
    )


def test_source_premise_is_visible_and_connectivity_spending_is_not_endorsed():
    kb, quote, response = example()
    grounded, warnings = ground_analysis(
        SpecialistAnalysis.model_validate(response), kb.evidence.chunks
    )
    assert not warnings
    qualified, notes = qualify_csr_actions(grounded)
    assert [a.title for a in qualified.actions] == [
        "Assess connectivity options for nutrition-program data sharing"
    ]
    assert "untested hypothesis" in qualified.actions[0].rationale
    assert "before any spending decision" in qualified.actions[0].prerequisites[-1]
    assert qualified.findings[0].supports[0].quote == quote
    assert notes and "Invest" in notes[0]


@pytest.mark.parametrize(
    "title,steps",
    [
        ("Review connectivity options", ["Then procure routers"]),
        ("Deploy a nutrition data platform", ["Schedule rollout"]),
        ("Guarantee improved nutrition with connectivity", ["Do an audit"]),
    ],
)
def test_premature_commitments_are_not_hidden_in_assessment_steps(title, steps):
    kb, _, response = example()
    response["actions"] = [response["actions"][0] | {"title": title, "next_steps": steps}]
    grounded, _ = ground_analysis(SpecialistAnalysis.model_validate(response), kb.evidence.chunks)
    qualified, _ = qualify_csr_actions(grounded)
    assert not qualified.actions and qualified.findings  # source premise survives


def test_investigate_is_an_assessment_not_an_investment_commitment():
    kb, _, response = example()
    response["actions"] = [response["actions"][0] | {"title": "Investigate connectivity options"}]
    grounded, _ = ground_analysis(SpecialistAnalysis.model_validate(response), kb.evidence.chunks)
    qualified, _ = qualify_csr_actions(grounded)
    assert len(qualified.actions) == 1


@pytest.mark.parametrize("keep_premise", [True, False])
async def test_csr_summary_distinguishes_source_basis_from_hypothetical_actions(keep_premise):
    kb, _, response = example()
    scripts = {
        "router": [
            {"tasks": {"corporate_responsibility": {"query": "SOFI 2024 data-quality priorities"}}}
        ],
        "corporate_responsibility": [response],
    }
    if not keep_premise:
        scripts["review"] = [
            {
                "accepted_finding_ids": [],
                "accepted_recommendation_ids": ["corporate_responsibility:a1"],
            }
        ]
    service = create_consulting_service(
        kb, settings=Settings(mode="live"), llm=ScriptedLLMClient(scripts)
    )
    result = await service.run_consultation(
        ConsultRequest(
            question="Explain SOFI 2024 priorities and assess a hypothetical nutrition-data connectivity initiative."
        ),
        request_id=str(uuid4()),
    )
    assert len(result.recommendations) == 1
    assert "untested hypothesis" in result.recommendations[0].rationale
    assert "This part could not be answered" not in result.summary
    assert "data sources" in result.summary
    assert "Invest" not in " ".join(r.title for r in result.recommendations)


def test_requested_assessment_can_be_built_from_cited_data_need_without_causal_claim():
    kb, quote, response = example()
    grounded, _ = ground_analysis(
        SpecialistAnalysis.model_validate(
            {
                "findings": [
                    {
                        "statement": "The report identifies data needs.",
                        "supports": [
                            {"evidence_id": kb.evidence.chunks[-1].chunk_id, "quote": quote}
                        ],
                    }
                ],
            }
        ),
        kb.evidence.chunks,
    )
    request = ConsultRequest(
        question="How could connectivity support nutrition-program data sharing? Explain a hypothetical feasibility test."
    )
    qualified, _ = qualify_csr_actions(grounded, request)
    assert len(qualified.actions) == 1
    assert qualified.actions[0].supports[0].quote == quote
    assert "untested hypothesis" in qualified.actions[0].rationale
    assert "baseline" in qualified.actions[0].success_metrics[0]
    qualified, _ = qualify_csr_actions(
        grounded, ConsultRequest(question="Explain SOFI 2024 priorities.")
    )
    assert not qualified.actions


def test_feasibility_proposal_does_not_answer_an_unrelated_causal_question():
    from types import SimpleNamespace

    from consulting_platform.agents.coverage import source_proof_covers
    from consulting_platform.agents.models import AnswerRequirement
    from consulting_platform.contracts import Domain

    kb, quote, _ = example()
    grounded, _ = ground_analysis(
        SpecialistAnalysis.model_validate(
            {
                "findings": [
                    {
                        "statement": "Documented data needs",
                        "supports": [
                            {"evidence_id": kb.evidence.chunks[-1].chunk_id, "quote": quote}
                        ],
                    }
                ],
            }
        ),
        kb.evidence.chunks,
    )
    request = ConsultRequest(
        question="How could connectivity support nutrition-program data sharing? Explain a hypothetical feasibility test."
    )
    qualified, _ = qualify_csr_actions(grounded, request)
    result = {"request": request, "analysis": qualified}
    rec = SimpleNamespace(
        id="corporate_responsibility:a1", rationale=qualified.actions[0].rationale
    )
    requirement = AnswerRequirement(
        id="r", question="How could connectivity support nutrition-program data sharing?"
    )
    assert source_proof_covers(requirement, Domain.CORPORATE_RESPONSIBILITY, result, [], {}, [rec])
    requirement.question = "Could buying routers reduce food prices?"
    assert not source_proof_covers(
        requirement, Domain.CORPORATE_RESPONSIBILITY, result, [], {}, [rec]
    )
