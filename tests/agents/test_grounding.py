from consulting_platform.agents.grounding import ground_analysis
from consulting_platform.agents.mocks import fixture_chunks
from consulting_platform.agents.models import SpecialistAnalysis


def test_numeric_attribute_cannot_be_fabricated():
    chunk = fixture_chunks()[1]
    quote = "Fixture Analyzer A supports 16 samples per run."
    supports = [{"evidence_id": chunk.chunk_id, "quote": quote}]
    analysis = SpecialistAnalysis.model_validate(
        {
            "candidates": [
                {
                    "id": "a",
                    "label": "Fixture Analyzer A",
                    "identity_supports": supports,
                    "attributes": {
                        "capacity": {"value": 999, "unit": "samples/run", "supports": supports}
                    },
                }
            ]
        }
    )
    clean, warnings = ground_analysis(analysis, [chunk])
    assert clean.candidates[0].attributes == {}
    assert warnings


def test_identity_and_quote_must_be_present():
    chunk = fixture_chunks()[1]
    analysis = SpecialistAnalysis.model_validate(
        {
            "candidates": [
                {
                    "id": "a",
                    "label": "Invented Product",
                    "identity_supports": [
                        {
                            "evidence_id": chunk.chunk_id,
                            "quote": "Fixture Analyzer A supports 16 samples per run.",
                        }
                    ],
                    "attributes": {},
                }
            ]
        }
    )
    clean, warnings = ground_analysis(analysis, [chunk])
    assert clean.candidates == [] and warnings


def test_nonverbatim_excerpt_is_rejected():
    chunk = fixture_chunks()[0]
    analysis = SpecialistAnalysis.model_validate(
        {
            "findings": [
                {
                    "statement": "The router has 100 ports.",
                    "supports": [
                        {"evidence_id": chunk.chunk_id, "quote": "The router has 100 ports."}
                    ],
                }
            ]
        }
    )
    clean, warnings = ground_analysis(analysis, [chunk])
    assert clean.findings == [] and warnings


def test_wrong_unit_is_not_supported_by_an_unrelated_number():
    chunk = fixture_chunks()[1]
    supports = [
        {"evidence_id": chunk.chunk_id, "quote": "Fixture Analyzer A supports 16 samples per run."}
    ]
    analysis = SpecialistAnalysis.model_validate(
        {
            "candidates": [
                {
                    "id": "a",
                    "label": "Fixture Analyzer A",
                    "identity_supports": supports,
                    "attributes": {"memory": {"value": 16, "unit": "GB", "supports": supports}},
                }
            ]
        }
    )
    clean, warnings = ground_analysis(analysis, [chunk])
    assert clean.candidates[0].attributes == {} and warnings
