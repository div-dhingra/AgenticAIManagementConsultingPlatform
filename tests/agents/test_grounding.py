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


def test_pdf_line_wraps_are_resolved_to_original_source_quote():
    chunk = fixture_chunks()[0].model_copy(
        update={
            "text": "Operating results fluctuate because of\nchanges in customer demand and inventory levels.",
        }
    )
    quote = (
        "Operating results fluctuate because of changes in customer demand and inventory levels."
    )
    analysis = SpecialistAnalysis.model_validate(
        {
            "findings": [
                {
                    "statement": "The source discloses demand and inventory variability.",
                    "supports": [{"evidence_id": chunk.chunk_id, "quote": quote}],
                }
            ]
        }
    )
    clean, warnings = ground_analysis(analysis, [chunk])
    assert len(clean.findings) == 1 and warnings == []
    assert clean.findings[0].supports[0].quote == chunk.text


def test_whitespace_normalization_does_not_admit_changed_numbers_or_ellipsis():
    chunk = fixture_chunks()[1]
    for quote in [
        "Fixture Analyzer A supports 99 samples per run.",
        "Fixture Analyzer A ... 16 samples per run.",
    ]:
        analysis = SpecialistAnalysis.model_validate(
            {
                "findings": [
                    {
                        "statement": quote,
                        "supports": [{"evidence_id": chunk.chunk_id, "quote": quote}],
                    }
                ]
            }
        )
        clean, warnings = ground_analysis(analysis, [chunk])
        assert clean.findings == []
        assert any("words and punctuation" in warning for warning in warnings)


def test_unknown_evidence_id_has_specific_diagnostic():
    chunk = fixture_chunks()[0]
    analysis = SpecialistAnalysis.model_validate(
        {
            "findings": [
                {
                    "statement": "The router has two ports.",
                    "supports": [{"evidence_id": "unknown", "quote": chunk.text[:100]}],
                }
            ]
        }
    )
    clean, warnings = ground_analysis(analysis, [chunk])
    assert not clean.findings
    assert "Evidence rejection: unknown evidence ID." in warnings


def test_candidate_identity_and_attributes_keep_original_line_breaks():
    chunk = fixture_chunks()[1].model_copy(
        update={
            "text": "Fixture Analyzer\nA supports 16 samples per run.",
        }
    )
    support = {
        "evidence_id": chunk.chunk_id,
        "quote": "Fixture Analyzer A supports 16 samples per run.",
    }
    analysis = SpecialistAnalysis.model_validate(
        {
            "candidates": [
                {
                    "id": "a",
                    "label": "Fixture Analyzer A",
                    "identity_supports": [support],
                    "attributes": {
                        "capacity": {"value": 16, "unit": "samples/run", "supports": [support]}
                    },
                }
            ]
        }
    )
    clean, warnings = ground_analysis(analysis, [chunk])
    assert len(clean.candidates) == 1 and warnings == []
    assert clean.candidates[0].attributes["capacity"].supports[0].quote == chunk.text


def test_pdf_smart_apostrophes_retain_original_source_glyphs():
    chunk = fixture_chunks()[0].model_copy(
        update={
            "text": "Changes in our customers’ spending plans affect revenue.",
        }
    )
    analysis = SpecialistAnalysis.model_validate(
        {
            "findings": [
                {
                    "statement": "The source discusses customer spending changes.",
                    "supports": [
                        {
                            "evidence_id": chunk.chunk_id,
                            "quote": "Changes in our customers' spending plans affect revenue.",
                        }
                    ],
                }
            ]
        }
    )
    clean, warnings = ground_analysis(analysis, [chunk])
    assert len(clean.findings) == 1 and warnings == []
    assert clean.findings[0].supports[0].quote == chunk.text


def test_span_ids_resolve_only_to_exact_original_source_text():
    from consulting_platform.agents.evidence_spans import evidence_spans

    chunk = fixture_chunks()[0]
    spans = evidence_spans(chunk.text)
    analysis = SpecialistAnalysis.model_validate({"findings": [{
        "statement": "A documented port fact.",
        "supports": [{"evidence_id": chunk.chunk_id, "span_id": spans[0]["span_id"]}],
    }]})
    clean, warnings = ground_analysis(analysis, [chunk])
    assert not warnings
    assert clean.findings[0].supports[0].quote == spans[0]["quote"]
    assert clean.findings[0].supports[0].quote in chunk.text
    analysis.findings[0].supports[0].span_id = "invented"
    clean, warnings = ground_analysis(analysis, [chunk])
    assert not clean.findings and any("unknown source span" in w for w in warnings)


def test_span_reference_cannot_override_a_contradictory_quote():
    chunk = fixture_chunks()[0]
    analysis = SpecialistAnalysis.model_validate({"findings": [{
        "statement": "Wrong count", "supports": [{
            "evidence_id": chunk.chunk_id, "span_id": "s0", "quote": "There are 999 ports.",
        }],
    }]})
    clean, warnings = ground_analysis(analysis, [chunk])
    assert not clean.findings and any("disagrees" in w for w in warnings)
