import copy
from collections import deque
from uuid import uuid4

from consulting_platform.agents.controlled_evidence import ControlledEvidenceService
from consulting_platform.agents.mocks import ScriptedLLMClient
from consulting_platform.agents.reliability import consolidate_questions
from consulting_platform.agents.service import create_consulting_service
from consulting_platform.config import Settings
from consulting_platform.contracts import ConsultContext, ConsultRequest


class Recorder(ScriptedLLMClient):
    def __init__(self, scripts):
        super().__init__(scripts)
        self.inputs = []

    async def generate_json(self, **kwargs):
        self.inputs.append(copy.deepcopy(kwargs))
        return await super().generate_json(**kwargs)


def setup_case(*, invalid_attributes=False, no_requirements=False):
    kb = ControlledEvidenceService.from_file("fixtures/consulting/controlled_real_passages.json")
    tech, sec, csr, _ = kb.evidence.chunks
    header, row_a, row_b = tech.text.splitlines()

    def support(chunk, quote):
        return {"evidence_id": chunk.chunk_id, "quote": quote}

    candidates = []
    for label, row, count in [("C8300-2N2S-4T2X", row_a, 4), ("C8300-2N2S-6T", row_b, 6)]:
        candidates.append(
            {
                "id": label,
                "label": label,
                "identity_supports": [support(tech, row)],
                "attributes": {
                    "1g_ports": {
                        "value": count,
                        "unit": "ports",
                        "supports": [
                            support(tech, "Invented quote" if invalid_attributes else row),
                            support(tech, header),
                        ],
                    }
                },
            }
        )
    requirements = (
        []
        if no_requirements
        else [
            {
                "id": "five_ports",
                "attribute": "1g_ports",
                "operator": "gte",
                "expected_value": 5,
                "unit": "ports",
                "required": True,
            }
        ]
    )
    router = {
        "tasks": {
            "technology": {
                "query": "Compare Cisco C8300 router 1G ports",
                "compare_candidates": True,
                "constraints": requirements,
            },
            "sec": {"query": "Cisco and Agilent fiscal 2025 10-K risks"},
            "corporate_responsibility": {"query": "SOFI 2024 food-security priorities"},
        }
    }
    sec_quote = sec.text.splitlines()[0]
    csr_quote = csr.text[:150]
    scripts = {
        "router": [router],
        "technology": [
            {
                "candidates": candidates,
                "findings": [
                    {
                        "statement": "C8300-2N2S-6T meets the 5-port requirement.",
                        "supports": [support(tech, row_b)],
                    }
                ],
                "clarification_questions": [
                    "Please provide Cisco 10-K excerpts.",
                    "Please upload SOFI documentation.",
                ],
                "limitations": ["No Cisco or SOFI evidence is available."],
            }
        ],
        "sec": [
            {
                "findings": [{"statement": sec_quote, "supports": [support(sec, sec_quote)]}],
                "missing_evidence": ["Agilent filing evidence is unavailable."],
            }
        ],
        "corporate_responsibility": [
            {"findings": [{"statement": csr_quote, "supports": [support(csr, csr_quote)]}]}
        ],
        "synthesis": [
            {
                "selected_finding_ids": ["technology:f1", "sec:f1", "corporate_responsibility:f1"],
                "selected_recommendation_ids": [
                    "technology:c:C8300-2N2S-6T",
                    "technology:c:C8300-2N2S-4T2X",
                ],
                "dependencies": [
                    {
                        "before_id": "technology:c:C8300-2N2S-6T",
                        "after_id": "technology:c:C8300-2N2S-4T2X",
                    }
                ],
            }
        ],
    }
    return kb, Recorder(scripts)


async def run_case(**kwargs):
    kb, llm = setup_case(**kwargs)
    if kwargs.get("invalid_attributes"):
        # No unambiguous source column exists either: prose must not rescue this.
        chunk = kb.evidence.chunks[0]
        kb.evidence.chunks[0] = chunk.model_copy(
            update={"text": chunk.text.replace("1G port density", "Unspecified port density")}
        )
    service = create_consulting_service(kb, settings=Settings(mode="live"), llm=llm)
    result = await service.run_consultation(
        ConsultRequest(
            question="Compare Cisco C8300 routers; assess Cisco and Agilent 2025 risks and SOFI 2024 priorities."
            if kwargs.get("no_requirements")
            else "Compare Cisco C8300 routers for 5 ports; assess Cisco and Agilent 2025 risks and SOFI 2024 priorities.",
            context=ConsultContext(
                requirements=[] if kwargs.get("no_requirements") else ["At least 5 1G ports"],
                report_years=[2024, 2025],
            ),
        ),
        request_id=str(uuid4()),
    )
    return result, llm


async def test_cross_domain_comparison_is_decisive_and_notes_cannot_override_sources():
    result, llm = await run_case()
    assert result.status == "partial" and not result.clarification_questions
    assert [r.title for r in result.recommendations] == ["Evaluate C8300-2N2S-6T"]
    assert "C8300-2N2S-6T meets" in result.summary
    assert "C8300-2N2S-4T2X does not meet" in result.summary
    assert "Agilent" in result.summary and "unresolved" in result.summary
    assert "Food-security priorities and proposed implications" in result.summary
    assert all("No Cisco or SOFI evidence" not in w for w in result.warnings)
    assert any("No Cisco or SOFI evidence" in t.detail for t in result.trace)
    assert not any(
        "Proposed execution order" in s for r in result.recommendations for s in r.next_steps
    )
    sec_input = next(c["payload"] for c in llm.inputs if c["role"] == "sec")
    assert "SOFI" not in sec_input["question"] and sec_input["context"]["report_years"] == [2025]
    assert sec_input["context"]["requirements"] == []


async def test_rejected_attributes_cannot_be_rescued_by_a_suitability_finding():
    result, _ = await run_case(invalid_attributes=True)
    assert not result.recommendations
    assert "No product recommendation could be validated" in result.summary
    assert not any("meets the 5-port" in f.statement for f in result.findings)
    assert result.status == "partial"  # evidence gaps, not unanswered customer preferences
    assert any("Removed unsupported attribute" in t.detail for t in result.trace)


async def test_missing_comparison_requirements_asks_customer_and_does_not_rank_products():
    result, _ = await run_case(no_requirements=True)
    assert result.status == "needs_clarification" and not result.recommendations
    assert any("specific requirements" in q for q in result.clarification_questions)


def test_clarification_deduplication_keeps_preferences_and_filters_source_requests():
    questions = consolidate_questions(
        [
            "What port and bandwidth requirements apply?",
            "What port and bandwidth requirements apply?",
            "Please provide Agilent filing excerpts.",
            "Which risk categories should be prioritized?",
        ]
    )
    assert len(questions) == 2 and any("risk categories" in q for q in questions)


async def test_two_eligible_product_alternatives_cannot_gain_execution_dependencies():
    kb, llm = setup_case()
    # Lower the threshold so both products are valid alternatives.
    llm.scripts["router"][0]["tasks"]["technology"]["constraints"][0]["expected_value"] = 4
    service = create_consulting_service(kb, settings=Settings(mode="live"), llm=llm)
    result = await service.run_consultation(
        ConsultRequest(
            question="Compare Cisco routers for 4 ports and assess financial and food risks."
        ),
        request_id=str(uuid4()),
    )
    assert len(result.recommendations) == 2
    assert not any(
        "Proposed execution order" in s for r in result.recommendations for s in r.next_steps
    )


async def test_product_selection_cannot_bypass_ranker_as_business_action():
    kb, llm = setup_case(invalid_attributes=True)
    tech = kb.evidence.chunks[0]
    llm.scripts["technology"][0]["actions"] = [
        {
            "kind": "business_action",
            "title": "Select C8300-2N2S-4T2X",
            "rationale": "Purchase this model.",
            "supports": [{"evidence_id": tech.chunk_id, "quote": tech.text.splitlines()[1]}],
            "next_steps": ["Order the router."],
            "success_metrics": ["Router ordered."],
        }
    ]
    service = create_consulting_service(kb, settings=Settings(mode="live"), llm=llm)
    result = await service.run_consultation(
        ConsultRequest(question="Compare Cisco routers for 5 ports."), request_id=str(uuid4())
    )
    assert [r.title for r in result.recommendations] == ["Evaluate C8300-2N2S-6T"]
    assert all(r.kind == "product" for r in result.recommendations)


async def test_explicit_port_requirement_is_recovered_without_reasking_customer():
    kb, llm = setup_case(no_requirements=True)
    service = create_consulting_service(kb, settings=Settings(mode="live"), llm=llm)
    result = await service.run_consultation(
        ConsultRequest(
            question="Compare Cisco routers.",
            context=ConsultContext(requirements=["At least 5 1G ports"]),
        ),
        request_id=str(uuid4()),
    )
    assert [r.title for r in result.recommendations] == ["Evaluate C8300-2N2S-6T"]
    assert not result.clarification_questions


async def test_review_can_remove_unnecessary_customer_question_without_losing_answer():
    kb, llm = setup_case()
    llm.scripts["corporate_responsibility"][0]["clarification_questions"] = [
        "Which food-security priorities should be explained?"
    ]

    class ReviewQuestions(Recorder):
        async def generate_json(self, **kwargs):
            result = await super().generate_json(**kwargs)
            if kwargs["role"] == "review":
                result["accepted_clarification_ids"] = []
            return result

    reviewer = ReviewQuestions({})
    reviewer.scripts = llm.scripts
    service = create_consulting_service(kb, settings=Settings(mode="live"), llm=reviewer)
    result = await service.run_consultation(
        ConsultRequest(question="Compare Cisco routers for 5 ports and explain SOFI priorities."),
        request_id=str(uuid4()),
    )
    assert not result.clarification_questions and result.status == "answered"


def test_table_header_and_nonadjacent_row_become_separate_verbatim_supports():
    from consulting_platform.agents.grounding import grounded_candidates
    from consulting_platform.agents.models import Candidate

    kb, llm = setup_case()
    chunk = kb.evidence.chunks[0]
    candidate = Candidate.model_validate(llm.scripts["technology"][0]["candidates"][1])
    header, _, row = chunk.text.splitlines()
    fact = candidate.attributes["1g_ports"]
    fact.unit = "1G Ethernet ports"
    fact.supports = [fact.supports[0].model_copy(update={"quote": f"{header}\n{row}"})]
    [accepted], warnings = grounded_candidates([candidate], {chunk.chunk_id: chunk})
    assert not warnings and accepted.attributes["1g_ports"].unit == "ports"
    assert [s.quote for s in accepted.attributes["1g_ports"].supports] == [header, row]
    # Altered words and another model's row remain invalid.
    fact.supports[0].quote = f"{header.replace('density', 'capacity')}\n{row}"
    [rejected], warnings = grounded_candidates([candidate], {chunk.chunk_id: chunk})
    assert not rejected.attributes and warnings


def test_table_attribute_cannot_borrow_another_candidates_row():
    from consulting_platform.agents.grounding import grounded_candidates
    from consulting_platform.agents.models import Candidate

    kb, llm = setup_case()
    chunk = kb.evidence.chunks[0]
    a, b = [Candidate.model_validate(c) for c in llm.scripts["technology"][0]["candidates"]]
    b.attributes = a.attributes
    [rejected], warnings = grounded_candidates([b], {chunk.chunk_id: chunk})
    assert not rejected.attributes and any("candidate's row" in w for w in warnings)


def test_port_table_extracts_header_units_and_rejects_wrong_speed_column():
    from consulting_platform.agents.grounding import grounded_candidates
    from consulting_platform.agents.models import Candidate

    kb, llm = setup_case()
    chunk = kb.evidence.chunks[0]
    candidate = Candidate.model_validate(llm.scripts["technology"][0]["candidates"][0])
    fact = candidate.attributes["1g_ports"]
    fact.supports = fact.supports[:1]  # row alone; verified column supplies header
    fact.unit = "1G"
    [accepted], warnings = grounded_candidates([candidate], {chunk.chunk_id: chunk})
    assert not warnings and accepted.attributes["1g_ports"].unit == "ports"
    assert len(accepted.attributes["1g_ports"].supports) == 2
    fact.value = 2  # exists in row, but belongs to 10G rather than 1G
    [rejected], warnings = grounded_candidates([candidate], {chunk.chunk_id: chunk})
    assert not rejected.attributes and any("column" in w for w in warnings)


async def test_successful_follow_up_replaces_stale_comparison_warnings():
    kb, llm = setup_case(invalid_attributes=True)
    from consulting_platform.agents.mocks import FakeKnowledgeService

    kb = FakeKnowledgeService(kb.evidence.chunks, mode="live")
    _, valid = setup_case()
    llm.scripts["technology"][0]["follow_up_query"] = "Specific Cisco C8300 1G port density table"
    llm.scripts["technology"].append(valid.scripts["technology"][0])
    service = create_consulting_service(kb, settings=Settings(mode="live"), llm=llm)
    result = await service.run_consultation(
        ConsultRequest(
            question="Compare Cisco routers for 5 ports, vendor risks and SOFI priorities."
        ),
        request_id=str(uuid4()),
    )
    assert any("6T" in r.title for r in result.recommendations)
    assert not any("Required candidate attributes could not" in w for w in result.warnings)
    assert any("Removed unsupported attribute" in t.detail for t in result.trace)


async def test_controlled_source_does_not_repeat_retrieval_for_an_unchanged_context():
    kb, llm = setup_case(invalid_attributes=True)
    llm.scripts["technology"][0]["follow_up_query"] = "A more focused Cisco ports query"
    service = create_consulting_service(kb, settings=Settings(mode="live"), llm=llm)
    result = await service.run_consultation(
        ConsultRequest(question="Compare Cisco routers for 5 ports."), request_id=str(uuid4())
    )
    assert llm.calls.count("technology") == 1
    assert [r.title for r in result.recommendations] == ["Evaluate C8300-2N2S-6T"]


def test_optional_follow_up_is_skipped_when_final_stages_need_remaining_time():
    import time

    service = create_consulting_service(settings=Settings(mode="demo"))
    assert service._follow_up_target({"deadline": time.monotonic() + 5}) is None


async def test_explicit_source_verified_port_proof_survives_model_omission():
    kb, llm = setup_case()
    llm.scripts["technology"][0]["candidates"] = []
    llm.scripts["review"] = deque(
        [
            {
                "accepted_finding_ids": ["sec:f1", "corporate_responsibility:f1"],
                "accepted_recommendation_ids": [],
            }
        ]
    )
    llm.scripts["synthesis"][0]["selected_recommendation_ids"] = []
    service = create_consulting_service(kb, settings=Settings(mode="live"), llm=llm)
    result = await service.run_consultation(
        ConsultRequest(
            question="Compare Cisco C8300-2N2S-4T2X and C8300-2N2S-6T for 1G ports.",
            context=ConsultContext(requirements=["At least 5 1G Ethernet ports per site"]),
        ),
        request_id=str(uuid4()),
    )
    assert [r.title for r in result.recommendations] == ["Evaluate C8300-2N2S-6T"]
    assert "C8300-2N2S-4T2X does not meet" in result.summary
    assert any("exact source counts" in t.detail for t in result.trace)


async def test_table_proof_does_not_override_semantic_review_for_unparsed_requirements():
    kb, llm = setup_case()
    llm.scripts["review"] = deque(
        [{"accepted_finding_ids": ["sec:f1"], "accepted_recommendation_ids": []}]
    )
    service = create_consulting_service(kb, settings=Settings(mode="live"), llm=llm)
    result = await service.run_consultation(
        ConsultRequest(question="Compare Cisco routers for 5 ports."), request_id=str(uuid4())
    )
    assert (
        not result.recommendations
    )  # no complete explicit context -> model review stays authoritative
