"""Regression checks for missing subquestions, false partials and financial labels."""

from uuid import uuid4

from consulting_platform.agents.coverage import coverage_gaps
from consulting_platform.agents.grounding import ground_analysis
from consulting_platform.agents.mocks import FakeKnowledgeService, ScriptedLLMClient
from consulting_platform.agents.models import (
    AnswerRequirement,
    CoverageCheck,
    DomainTask,
    SpecialistAnalysis,
)
from consulting_platform.agents.service import create_consulting_service
from consulting_platform.config import Settings
from consulting_platform.contracts import (
    Citation,
    ConsultRequest,
    Domain,
    Finding,
    RetrievalFilters,
)


def evidence_case():
    chunk = next(c for c in FakeKnowledgeService().chunks if c.domain == Domain.SEC)
    task = DomainTask(
        query="Cisco 2025 demand and inventory risk",
        answer_requirements=[
            AnswerRequirement(
                id="sec:r1", question="Cisco demand risk", company="Cisco", report_year=2025
            ),
            AnswerRequirement(
                id="sec:r2", question="Cisco inventory risk", company="Cisco", report_year=2025
            ),
        ],
    )
    result = {
        Domain.SEC: {
            "task": task,
            "chunks": [chunk],
            "filters": RetrievalFilters(companies=["Cisco"]),
        }
    }
    finding = Finding(
        id="sec:f1",
        domain=Domain.SEC,
        statement="Demand risk",
        assessment="supported",
        evidence_ids=[chunk.chunk_id],
    )
    citation = Citation.model_validate(
        {
            **chunk.model_dump(exclude={"text", "retrieval_score", "chunk_id"}),
            "evidence_id": chunk.chunk_id,
            "excerpt": chunk.text,
        }
    )
    return result, [finding], {chunk.chunk_id: citation}


async def test_sofi_financing_does_not_create_an_unanswerable_sec_subquestion():
    kb = FakeKnowledgeService()
    llm = ScriptedLLMClient(
        {
            "router": [
                {
                    "tasks": {
                        "sec": {
                            "query": "SOFI 2024 financing priorities",
                            "answer_requirements": [
                                {
                                    "id": "finance",
                                    "question": "SOFI 2024 financing priorities",
                                    "company": "FAO",
                                    "report_year": 2024,
                                }
                            ],
                        },
                        "corporate_responsibility": {"query": "SOFI 2024 data priorities"},
                    }
                }
            ]
        }
    )
    response = await create_consulting_service(
        kb, settings=Settings(mode="demo"), llm=llm
    ).run_consultation(
        ConsultRequest(question="Explain SOFI 2024 financing and data priorities."),
        request_id=str(uuid4()),
    )
    assert response.domains == [Domain.CORPORATE_RESPONSIBILITY]
    assert kb.calls and all(c["domains"] == [Domain.CORPORATE_RESPONSIBILITY] for c in kb.calls)
    assert not any(
        "SEC filings" in t.detail for t in response.trace if t.detail.startswith("Answer coverage")
    )


def test_one_finding_in_a_domain_does_not_answer_two_questions():
    results, findings, citations = evidence_case()
    gaps = coverage_gaps(
        results,
        findings,
        [],
        citations,
        [CoverageCheck(requirement_id="sec:r1", answered=True, finding_ids=["sec:f1"])],
    )
    assert len(gaps) == 1 and "inventory" in gaps[0]


def test_rejected_or_wrong_company_year_cannot_complete_an_answer():
    results, findings, citations = evidence_case()
    checks = [
        CoverageCheck(requirement_id=f"sec:r{i}", answered=True, finding_ids=["sec:invented"])
        for i in (1, 2)
    ]
    assert len(coverage_gaps(results, findings, [], citations, checks)) == 2
    for check in checks:
        check.finding_ids = ["sec:f1"]
    assert not coverage_gaps(results, findings, [], citations, checks)
    citation = next(iter(citations.values()))
    citation.company = "Agilent"
    assert len(coverage_gaps(results, findings, [], citations, checks)) == 2
    citation.company, citation.report_year = "Cisco", 2024
    assert len(coverage_gaps(results, findings, [], citations, checks)) == 2


async def test_complete_answer_retains_third_fact_and_does_not_turn_partial_for_diagnostics():
    kb = FakeKnowledgeService()
    chunk = next(c for c in kb.chunks if c.domain == Domain.SEC)
    sentences = [f"Documented risk fact {i}." for i in range(3)]
    chunk = chunk.model_copy(update={"text": " ".join(sentences)})
    kb.chunks = [chunk if c.chunk_id == chunk.chunk_id else c for c in kb.chunks]
    requirements = [{"id": str(i), "question": f"Risk topic {i}"} for i in range(3)]
    scripts = {
        "router": [
            {"tasks": {"sec": {"query": "Cisco 2025 risks", "answer_requirements": requirements}}}
        ],
        "sec": [
            {
                "findings": [
                    {
                        "statement": f"Accepted risk fact {i}",
                        "supports": [{"evidence_id": chunk.chunk_id, "quote": sentences[i]}],
                    }
                    for i in range(3)
                ],
                "limitations": ["Check the complete filing before a procurement decision."],
            }
        ],
        "review": [
            {
                "accepted_finding_ids": [f"sec:f{i}" for i in (1, 2, 3)],
                "accepted_recommendation_ids": [],
                "limitations": ["A redundant draft was excluded."],
                "coverage": [
                    {"requirement_id": f"sec:r{i}", "answered": True, "finding_ids": [f"sec:f{i}"]}
                    for i in (1, 2, 3)
                ],
            }
        ],
        "synthesis": [{"selected_finding_ids": ["sec:f1"], "selected_recommendation_ids": []}],
    }
    service = create_consulting_service(
        kb, settings=Settings(mode="demo"), llm=ScriptedLLMClient(scripts)
    )
    response = await service.run_consultation(
        ConsultRequest(question="Summarize Cisco 2025 risks"), request_id=str(uuid4())
    )
    assert response.status == "answered" and response.warnings
    assert len(response.findings) == 3 and "Documented risk fact 2" in response.summary
    assert "[1]" in response.summary


def test_combined_financial_measure_is_not_a_standalone_liability():
    chunk = next(c for c in FakeKnowledgeService().chunks if c.domain == Domain.SEC)
    text = "Our inventory provisions and purchase commitment liabilities totaled $712 million in fiscal 2025."
    chunk = chunk.model_copy(update={"text": text})
    analysis = SpecialistAnalysis.model_validate(
        {
            "findings": [
                {
                    "statement": "Purchase commitment liabilities totaled $712 million.",
                    "supports": [{"evidence_id": chunk.chunk_id, "quote": text}],
                },
                {"statement": text, "supports": [{"evidence_id": chunk.chunk_id, "quote": text}]},
            ]
        }
    )
    grounded, warnings = ground_analysis(analysis, [chunk])
    assert len(grounded.findings) == 1 and grounded.findings[0].statement == text
    assert any("combined financial measure" in w for w in warnings)


async def test_missing_accepted_answer_drives_one_followup_and_resolves_partial():
    kb = FakeKnowledgeService()
    chunk = next(c for c in kb.chunks if c.domain == Domain.SEC)
    finding = {
        "statement": "Risk disclosed in the Cisco source",
        "supports": [{"evidence_id": chunk.chunk_id, "quote": chunk.text}],
    }
    llm = ScriptedLLMClient(
        {
            "router": [
                {
                    "tasks": {
                        "sec": {
                            "query": "Cisco 2025 risk overview",
                            "answer_requirements": [
                                {
                                    "id": "r",
                                    "question": "Cisco 2025 supplier disruption risk ratio",
                                    "company": "Cisco",
                                    "report_year": 2025,
                                }
                            ],
                        }
                    }
                }
            ],
            "sec": [{"findings": [finding]}, {"findings": [finding]}],
            "review": [
                {
                    "accepted_finding_ids": ["sec:f1"],
                    "accepted_recommendation_ids": [],
                    "coverage": [
                        {
                            "requirement_id": "sec:r1",
                            "answered": False,
                            "reason": "Requested risk detail unresolved.",
                        }
                    ],
                },
                {
                    "accepted_finding_ids": ["sec:f1"],
                    "accepted_recommendation_ids": [],
                    "coverage": [
                        {"requirement_id": "sec:r1", "answered": True, "finding_ids": ["sec:f1"]}
                    ],
                },
            ],
        }
    )
    service = create_consulting_service(
        kb, settings=Settings(mode="demo", llm_timeout_seconds=1), llm=llm
    )
    response = await service.run_consultation(
        ConsultRequest(question="Explain Cisco 2025 supplier disruption risk ratio"),
        request_id=str(uuid4()),
    )
    assert response.status == "answered"
    assert llm.calls.count("sec") == 2
    assert sum(t.step == "retrieve" for t in response.trace) == 2
    assert all(not t.detail.startswith("Answer coverage: Still unresolved") for t in response.trace)


async def test_coverage_uses_only_final_retained_recommendations():
    results, findings, citations = evidence_case()
    # A reviewer may name a valid earlier draft that did not survive selection.
    checks = [
        CoverageCheck(requirement_id=f"sec:r{i}", answered=True, recommendation_ids=["sec:a1"])
        for i in (1, 2)
    ]
    assert len(coverage_gaps(results, findings, [], citations, checks)) == 2


def test_actual_combined_measure_wording_is_protected_with_line_breaks():
    chunk = next(c for c in FakeKnowledgeService().chunks if c.domain == Domain.SEC)
    quote = "Our total provisions for inventory and the liability related to purchase commitments were $493\nmillion in fiscal 2025."
    chunk = chunk.model_copy(update={"text": quote})
    analysis = SpecialistAnalysis.model_validate(
        {
            "findings": [
                {
                    "statement": "Purchase-commitment liabilities were $493 million in fiscal 2025.",
                    "supports": [{"evidence_id": chunk.chunk_id, "quote": quote}],
                }
            ]
        }
    )
    grounded, warnings = ground_analysis(analysis, [chunk])
    assert not grounded.findings
    assert any("combined financial measure" in warning for warning in warnings)


async def test_synthesis_cannot_drop_the_action_that_answers_a_requested_implication():
    kb = FakeKnowledgeService()
    chunk = next(
        c
        for c in kb.chunks
        if c.domain == Domain.CORPORATE_RESPONSIBILITY and c.report_year == 2024
    )
    role = Domain.CORPORATE_RESPONSIBILITY.value
    action_id = f"{role}:a1"
    scripts = {
        "router": [
            {
                "tasks": {
                    role: {
                        "query": "SOFI 2024 financing priorities and a hypothetical connectivity pilot",
                        "answer_requirements": [
                            {
                                "id": "pilot",
                                "question": "Propose a measurable feasibility pilot based on SOFI 2024 priorities",
                                "report_year": 2024,
                            }
                        ],
                    }
                }
            }
        ],
        role: [
            {
                "findings": [
                    {
                        "statement": chunk.text,
                        "supports": [{"evidence_id": chunk.chunk_id, "quote": chunk.text}],
                    }
                ],
                "actions": [
                    {
                        "kind": "business_action",
                        "title": "Assess nutrition-program data sharing in a pilot",
                        "rationale": "Investigate whether a pilot helps address the documented financing barrier.",
                        "supports": [{"evidence_id": chunk.chunk_id, "quote": chunk.text}],
                        "next_steps": [
                            "Audit existing sharing and financing gaps before proposing a pilot."
                        ],
                        "success_metrics": ["Completeness of the proposed pilot's data."],
                    }
                ],
            }
        ],
        "review": [
            {
                "accepted_finding_ids": [f"{role}:f1"],
                "accepted_recommendation_ids": [action_id],
                "coverage": [
                    {
                        "requirement_id": f"{role}:r1",
                        "answered": True,
                        "recommendation_ids": [action_id],
                    }
                ],
            }
        ],
        "synthesis": [{"selected_finding_ids": [], "selected_recommendation_ids": []}],
    }
    service = create_consulting_service(
        kb, settings=Settings(mode="demo"), llm=ScriptedLLMClient(scripts)
    )
    response = await service.run_consultation(
        ConsultRequest(
            question="Explain SOFI 2024 priorities and propose a measurable feasibility pilot."
        ),
        request_id=str(uuid4()),
    )
    assert response.status == "answered"
    assert [r.id for r in response.recommendations] == [action_id]
    assert "Measure the result" in response.summary


async def test_multi_company_extraction_receives_only_its_company_sources():
    from consulting_platform.agents.llm import RequestBudget
    from consulting_platform.agents.mocks import FakeLLMClient
    from consulting_platform.agents.specialists import SpecialistAgent

    class Isolated(FakeLLMClient):
        async def generate_json(self, **kwargs):
            if kwargs["role"] == "sec":
                payload = kwargs["payload"]
                requested = payload["requested_companies"]
                assert len(requested) == 1
                assert all(c["company"].casefold() == requested[0] for c in payload["chunks"])
            return await super().generate_json(**kwargs)

    kb, llm = FakeKnowledgeService(), Isolated()
    agent = SpecialistAgent(Domain.SEC, kb, llm, "demo")
    state = {
        "task": DomainTask(query="Cisco and Agilent 2025 risks"),
        "request": ConsultRequest(question="Assess Cisco and Agilent 2025 risks"),
        "chunks": [c for c in kb.chunks if c.domain == Domain.SEC],
        "filters": RetrievalFilters(companies=["cisco", "agilent"], report_years=[2025]),
        "budget": RequestBudget(),
        "rounds": 1,
    }
    result = await agent._analyze(state)
    assert len(result["analysis"].findings) == 2
    assert [c["role"] for c in llm.calls] == ["sec", "sec"]


def test_review_cannot_mark_rejected_content_as_an_answer():
    import pytest
    from pydantic import ValidationError

    from consulting_platform.agents.models import EvidenceReview

    with pytest.raises(ValidationError, match="accepted content"):
        EvidenceReview.model_validate(
            {
                "accepted_finding_ids": ["sec:f1"],
                "accepted_recommendation_ids": [],
                "coverage": [
                    {
                        "requirement_id": "sec:r1",
                        "answered": True,
                        "finding_ids": ["sec:f1", "sec:rejected"],
                    }
                ],
            }
        )


async def test_semantic_review_is_isolated_by_domain():
    from consulting_platform.agents.mocks import FakeLLMClient

    class ScopedReview(FakeLLMClient):
        def __init__(self):
            super().__init__()
            self.scopes = set()

        async def generate_json(self, **kwargs):
            if kwargs["role"] == "review":
                payload = kwargs["payload"]
                scope = payload["review_scope"]
                self.scopes.add(scope)
                assert all(c["domain"] == scope for c in payload["sources"])
                assert all(f["domain"] == scope for f in payload["findings"])
            return await super().generate_json(**kwargs)

    llm = ScopedReview()
    service = create_consulting_service(
        FakeKnowledgeService(), settings=Settings(mode="demo"), llm=llm
    )
    response = await service.run_consultation(
        ConsultRequest(
            question="Assess Cisco router connectivity, Cisco and Agilent 2025 financial risks, and SOFI 2024 priorities."
        ),
        request_id=str(uuid4()),
    )
    assert response.status == "answered"
    assert llm.scopes == {"technology", "sec", "corporate_responsibility"}


def test_literal_disclosure_coverage_requires_source_words_topic_company_and_year():
    from consulting_platform.agents.coverage import source_proof_covers

    results, findings, citations = evidence_case()
    result = results[Domain.SEC]
    chunk = result["chunks"][0]
    findings[0].statement = f"Cisco's fiscal 2025 filing discloses: “{chunk.text}”"
    requirement = AnswerRequirement(
        id="r",
        question="What supply-chain risks does Cisco disclose?",
        company="Cisco",
        report_year=2025,
    )
    assert source_proof_covers(requirement, Domain.SEC, result, findings, citations)
    requirement.question = "What legal risks does Cisco disclose?"
    assert not source_proof_covers(requirement, Domain.SEC, result, findings, citations)
    requirement.question = "What supply-chain risks does Cisco disclose?"
    requirement.report_year = 2024
    assert not source_proof_covers(requirement, Domain.SEC, result, findings, citations)
    requirement.report_year = 2025
    findings[
        0
    ].statement = (
        "Cisco's fiscal 2025 filing discloses: “Invented supply-chain risks may hurt sales.”"
    )
    assert not source_proof_covers(requirement, Domain.SEC, result, findings, citations)
