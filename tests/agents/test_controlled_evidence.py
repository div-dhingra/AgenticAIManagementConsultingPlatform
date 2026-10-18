import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from consulting_platform.agents.controlled_evidence import (
    NOTICE,
    ControlledEvidenceService,
    ControlledEvidenceSet,
)
from consulting_platform.agents.mocks import ScriptedLLMClient
from consulting_platform.agents.service import create_consulting_service
from consulting_platform.config import Settings
from consulting_platform.contracts import ConsultRequest, Domain, RetrievalFilters

PATH = Path("fixtures/consulting/controlled_real_passages.json")


async def test_controlled_source_filters_and_discloses_limitations():
    service = ControlledEvidenceService.from_file(PATH)
    result = await service.retrieve(
        "Cisco 2025 risks",
        [Domain.SEC],
        RetrievalFilters(
            companies=["Cisco"],
            report_years=[2025],
        ),
    )
    assert result.mode == "live" and result.status == "ok"
    assert len(result.chunks) == 1
    assert result.chunks[0].page == 13 and result.chunks[0].printed_page == "11"
    assert NOTICE in result.warnings
    wrong_year = await service.retrieve(
        "Cisco 2024 risks",
        [Domain.SEC],
        RetrievalFilters(
            report_years=[2024],
        ),
    )
    assert wrong_year.chunks == [] and wrong_year.status == "ok"


async def test_controlled_missing_domain_is_explicit_empty_context():
    service = ControlledEvidenceService.from_file(PATH)
    result = await service.retrieve("Agilent capacity", [Domain.CLINICAL])
    assert result.chunks == [] and result.warnings
    status = service.get_status()
    assert status.domains[Domain.CLINICAL].chunk_count == 0
    assert len(status.domains) == 4 and status.mode == "live"


def test_synthetic_demo_file_cannot_be_mistaken_for_real_source_set():
    with pytest.raises(ValidationError):
        ControlledEvidenceService.from_file("fixtures/consulting/evidence.json")


def test_duplicate_ids_are_rejected():
    data = json.loads(PATH.read_text())
    data["chunks"].append(data["chunks"][0])
    with pytest.raises(ValidationError):
        ControlledEvidenceSet.model_validate(data)


async def test_graph_accepts_real_passages_and_preserves_test_warning_without_network():
    knowledge = ControlledEvidenceService.from_file(PATH)
    chunk = next(c for c in knowledge.evidence.chunks if c.domain == Domain.SEC)
    quote = chunk.text.split("\n")[0]
    llm = ScriptedLLMClient(
        {
            "sec": [
                {
                    "findings": [
                        {
                            "statement": quote,
                            "supports": [{"evidence_id": chunk.chunk_id, "quote": quote}],
                        }
                    ]
                }
            ]
        }
    )
    agent = create_consulting_service(knowledge, settings=Settings(mode="live"), llm=llm)
    result = await agent.run_consultation(
        ConsultRequest(question="What risks does Cisco disclose in its 2025 filing?"),
        request_id=str(uuid4()),
    )
    assert result.mode == "live" and result.status == "partial"
    assert NOTICE in result.warnings
    assert result.citations[0].page == 13 and result.findings[0].statement == quote


def test_cli_forbids_using_real_evidence_file_with_fake_demo_model(monkeypatch):
    from consulting_platform.cli import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "consult",
            "ask",
            "--mode",
            "demo",
            "--evidence-file",
            str(PATH),
            "--question",
            "Cisco risks",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
