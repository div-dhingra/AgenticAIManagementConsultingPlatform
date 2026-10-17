import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from consulting_platform.contracts import (
    ConsultRequest,
    ConsultResponse,
    ErrorResponse,
    EvidenceChunk,
    KnowledgeStatus,
    RetailRequest,
    RetailResponse,
)


@pytest.mark.parametrize(
    "payload",
    [
        {"question": " "},
        {"question": "a" * 4001},
        {"question": "Hi", "agent": "sec"},
        {"question": "Hi", "max_recommendations": True},
        {"question": "Hi", "context": {"report_years": [2025, 2025]}},
        {"question": "Hi", "context": {"report_years": [True]}},
    ],
)
def test_reject_bad_consult_inputs(payload):
    with pytest.raises(ValidationError):
        ConsultRequest.model_validate(payload)


def test_defaults_and_trim():
    request = ConsultRequest(question="  Compare instruments  ")
    assert request.question == "Compare instruments"
    assert request.context.report_years == []
    assert request.max_recommendations == 3


@pytest.mark.parametrize("cart", [["A", "A"], [str(i) for i in range(41)]])
def test_invalid_cart(cart):
    with pytest.raises(ValidationError):
        RetailRequest(cart_items=cart)


def test_handoff_examples_round_trip():
    fixtures = Path("fixtures/consulting/contract_examples.json")
    for example in json.loads(fixtures.read_text()):
        model = (
            ConsultRequest
            if "question" in example
            else ConsultResponse
            if "findings" in example
            else RetailResponse
            if "model_version" in example
            else ErrorResponse
            if "error" in example
            else RetailRequest
        )
        parsed = model.model_validate(example)
        assert model.model_validate_json(parsed.model_dump_json()) == parsed


def test_citation_references_are_enforced():
    examples = json.loads(Path("fixtures/consulting/contract_examples.json").read_text())
    example = next(e for e in examples if e.get("findings"))
    example["findings"][0]["evidence_ids"] = ["invented-id"]
    with pytest.raises(ValidationError, match="references"):
        ConsultResponse.model_validate(example)


@pytest.mark.parametrize(
    "uri", ["/Users/me/private.pdf", "../secret", "file:///tmp/a", "https://u:p@example.com/a"]
)
def test_source_paths_are_portable_and_safe(uri):
    from consulting_platform.agents.mocks import fixture_chunks

    data = fixture_chunks()[0].model_dump()
    data["source_uri"] = uri
    with pytest.raises(ValidationError):
        EvidenceChunk.model_validate(data)


def test_health_requires_all_domains():
    with pytest.raises(ValidationError):
        KnowledgeStatus(mode="demo", domains={})
