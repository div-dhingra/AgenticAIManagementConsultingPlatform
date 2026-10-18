from consulting_platform.ui.consulting_view import diagnostic_warning, readable_rationale


def test_ranking_presentation_keeps_unknowns_and_failures_visible():
    assert (
        readable_rationale("1 matched, 0 unknown, 0 unmet constraints. Eligibility: eligible.")
        == "Meets 1 of your stated requirements."
    )
    result = readable_rationale(
        "1 matched, 2 unknown, 1 unmet constraints. Eligibility: conditional."
    )
    assert "2 requirements still need verification" in result
    assert "Does not meet 1 requirement." in result


def test_free_text_explanation_is_preserved():
    explanation = "Compare the documented interface count before purchase."
    assert readable_rationale(explanation) == explanation


def test_diagnostics_are_distinct_from_customer_coverage_notices():
    assert diagnostic_warning("Removed unsupported attribute port_count.")
    assert not diagnostic_warning("Controlled evidence test: real passages were preselected.")
    assert not diagnostic_warning("SEC filings: No matching passages for Agilent were retrieved.")
