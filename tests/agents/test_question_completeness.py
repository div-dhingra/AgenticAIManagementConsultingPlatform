"""Reproduce false port clarifications and a silently omitted second subject."""

from uuid import uuid4

import pytest

from consulting_platform.agents.controlled_evidence import ControlledEvidenceService
from consulting_platform.agents.mocks import FakeKnowledgeService, ScriptedLLMClient
from consulting_platform.agents.models import AnswerRequirement, DomainTask
from consulting_platform.agents.questions import preserve_request_questions
from consulting_platform.agents.service import create_consulting_service
from consulting_platform.config import Settings
from consulting_platform.contracts import ConsultContext, ConsultRequest, Domain


async def test_port_proof_survives_leaked_report_year_and_false_reviewer_objection():
    kb = ControlledEvidenceService.from_file("fixtures/consulting/controlled_real_passages.json")
    llm = ScriptedLLMClient(
        {
            "router": [
                {
                    "tasks": {
                        "technology": {
                            "query": "Compare Cisco C8300 1G port counts",
                            "compare_candidates": True,
                            "answer_requirements": [
                                {
                                    "id": "port-choice",
                                    "question": "Which model meets the 1G Ethernet port requirement?",
                                    "company": "Cisco",
                                    "report_year": 2025,
                                }
                            ],
                        }
                    },
                    "clarification_questions": [
                        "Are there additional port requirements beyond 1G Ethernet?"
                    ],
                }
            ],
            "technology": [{"findings": [], "candidates": []}],
            "review": [
                {
                    "accepted_finding_ids": [],
                    "accepted_recommendation_ids": [],
                    "accepted_clarification_ids": ["router:q0"],
                    "coverage": [{"requirement_id": "technology:r1", "status": "unresolved"}],
                    "unanswered_questions": ["Which model meets the 1G Ethernet port requirement?"],
                }
            ],
        }
    )
    response = await create_consulting_service(
        kb, settings=Settings(mode="live"), llm=llm
    ).run_consultation(
        ConsultRequest(
            question="Compare Cisco C8300-2N2S-4T2X and C8300-2N2S-6T for at least 5 1G Ethernet ports.",
            context=ConsultContext(
                report_years=[2024, 2025], requirements=["At least 5 1G Ethernet ports"]
            ),
        ),
        request_id=str(uuid4()),
    )
    assert response.status == "answered"
    assert not response.clarification_questions
    assert "Still unresolved" not in response.summary
    assert [r.title for r in response.recommendations] == ["Evaluate C8300-2N2S-6T"]


@pytest.mark.parametrize("has_scanner_source", [True, False])
async def test_two_clinical_subjects_cannot_be_reported_complete_with_only_the_first(
    has_scanner_source,
):
    template = next(c for c in FakeKnowledgeService().chunks if c.domain == Domain.CLINICAL)
    seahorse = template.model_copy(
        update={
            "chunk_id": "cell-1",
            "document_id": "cell-doc",
            "title": "Seahorse XF cell metabolic analysis",
            "text": "Seahorse XF analyzers measure energy production in live cells.",
        }
    )
    scanner = template.model_copy(
        update={
            "chunk_id": "slide-1",
            "document_id": "slide-doc",
            "title": "Digital pathology slide scanners",
            "text": "Digital pathology slide scanners convert glass slides into digital images. The S540MD scanner provides continuous loading. The NanoZoomer® range supports fluorescence imaging.",
        }
    )
    if not has_scanner_source:
        # A broad document title must not make a Seahorse-only statement count
        # as an explanation of the scanner function.
        seahorse = seahorse.model_copy(
            update={"title": "Seahorse XF and digital pathology slide scanners overview"}
        )
    kb = FakeKnowledgeService([seahorse, scanner] if has_scanner_source else [seahorse])
    queries = []

    class Recorder(ScriptedLLMClient):
        async def generate_json(self, **kwargs):
            if kwargs["role"] == "clinical":
                queries.append(kwargs["payload"])
            return await super().generate_json(**kwargs)

    def finding(chunk):
        function = chunk.text.split(".")[0] + "."
        return {
            "statement": function,
            "supports": [{"evidence_id": chunk.chunk_id, "quote": function}],
        }

    llm = Recorder(
        {
            # Intentionally omit the second question in the router output.
            "router": [
                {
                    "tasks": {
                        "clinical": {
                            "query": "Agilent Seahorse XF analyzer functions",
                            "compare_candidates": True,
                            "answer_requirements": [
                                {"id": "one", "question": "What do Seahorse XF analyzers measure?"}
                            ],
                        }
                    }
                }
            ],
            "clinical": [
                {"findings": [finding(seahorse)]},
                {
                    "findings": [
                        finding(scanner),
                        {
                            "statement": "The NanoZoomer scanner provides continuous loading.",
                            "supports": [
                                {
                                    "evidence_id": scanner.chunk_id,
                                    "quote": "The S540MD scanner provides continuous loading.",
                                }
                            ],
                        },
                    ]
                },
            ],
            "review": [
                {
                    "accepted_finding_ids": ["clinical:f1", "clinical:f2"]
                    if has_scanner_source
                    else ["clinical:f1"],
                    "accepted_recommendation_ids": [],
                    "coverage": [
                        {
                            "requirement_id": "clinical:r1",
                            "status": "answered",
                            "finding_ids": ["clinical:f1"],
                        },
                        # Deliberately let a bad reviewer reuse the Seahorse claim.
                        {
                            "requirement_id": "clinical:r2",
                            "status": "answered",
                            "finding_ids": ["clinical:f2"]
                            if has_scanner_source
                            else ["clinical:f1"],
                        },
                    ],
                }
            ],
        }
    )
    response = await create_consulting_service(
        kb, settings=Settings(mode="demo"), llm=llm
    ).run_consultation(
        ConsultRequest(
            question="According to the indexed Agilent product pages, what do Seahorse XF analyzers measure, and what do digital pathology slide scanners do?"
        ),
        request_id=str(uuid4()),
    )
    if has_scanner_source:
        assert response.status == "answered"
        assert len(queries) == 2
        assert [c["chunk_id"] for c in queries[1]["chunks"]] == ["slide-1"]
        assert "convert glass slides into digital images" in response.summary
        assert len(response.findings) == 2
    else:
        assert response.status == "partial"
        assert "could not be answered" in response.summary
        assert "Still unresolved" in response.summary
    assert "Seahorse XF analyzers" in response.summary
    assert "Digital pathology slide scanners" in response.summary


def test_an_explicit_technical_document_date_is_not_erased():
    request = ConsultRequest(question="Use the 2024 datasheet to compare Cisco router port counts.")
    task = DomainTask(
        query=request.question,
        answer_requirements=[
            AnswerRequirement(
                id="year",
                question="Compare port counts from the requested datasheet",
                report_year=2024,
            )
        ],
    )
    preserve_request_questions(request, Domain.TECHNOLOGY, task)
    assert task.answer_requirements[0].report_year == 2024


def test_clinical_subquestions_do_not_inherit_another_vendors_context():
    request = ConsultRequest(
        question="What do Agilent Seahorse XF analyzers measure, and what do digital pathology slide scanners do? Also summarize Cisco filing risks.",
        context=ConsultContext(company="Cisco"),
    )
    task = DomainTask(query=request.question)
    preserve_request_questions(request, Domain.CLINICAL, task)
    assert len(task.answer_requirements) == 2
    assert all(r.company == "Agilent" for r in task.answer_requirements)
