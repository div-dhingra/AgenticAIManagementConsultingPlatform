
from consulting_platform.knowledge.sections import (
    detect_section_heading,
    detect_section_boundary,
    assign_sections,
)


def make_chunk(document_id, domain, text):
    """Create a simple test chunk."""
    return {
        "document_id": document_id,
        "domain": domain,
        "text": text,
    }


def test_detect_item_1a_heading():
    text = "Table of Contents Item 1A. Risk Factors"

    result = detect_section_heading(text)

    assert result == "item_1a"


def test_detect_item_1c_heading():
    text = "Item 1C. Cybersecurity"

    result = detect_section_heading(text)

    assert result == "item_1c"


def test_ignore_item_1a_reference_in_body():
    text = (
        "Our business faces various uncertainties. "
        "Please refer to Item 1A. Risk Factors for details."
    )

    result = detect_section_heading(text)

    assert result is None


def test_detect_item_1b_boundary():
    text = (
        "Additional discussion of risk factors appears here. "
        "Item 1B. Unresolved Staff Comments None."
    )

    result = detect_section_boundary(text)

    assert result == "item_1b"


def test_assign_sections_within_document():
    chunks = [
        make_chunk(
            "cisco_2025",
            "sec",
            "Item 1A. Risk Factors",
        ),
        make_chunk(
            "cisco_2025",
            "sec",
            "We face risks from competition.",
        ),
        make_chunk(
            "cisco_2025",
            "sec",
            "Item 1C. Cybersecurity",
        ),
    ]

    result = assign_sections(chunks)

    assert result[0]["section"] == "item_1a"
    assert result[1]["section"] == "item_1a"
    assert result[2]["section"] == "item_1c"


def test_reset_section_for_new_document():
    chunks = [
        make_chunk(
            "cisco_2025",
            "sec",
            "Item 1A. Risk Factors",
        ),
        make_chunk(
            "alphabet_2025",
            "sec",
            "Introduction to the annual report.",
        ),
    ]

    result = assign_sections(chunks)

    assert result[0]["section"] == "item_1a"
    assert result[1]["section"] is None


def test_non_sec_document_has_no_section():
    chunks = [
        make_chunk(
            "cisco_2025",
            "sec",
            "Item 1A. Risk Factors",
        ),
        make_chunk(
            "sofi_2024",
            "corporate_responsibility",
            "Item 1A. Risk Factors",
        ),
    ]

    result = assign_sections(chunks)

    assert result[0]["section"] == "item_1a"
    assert result[1]["section"] is None


def test_original_chunks_are_not_modified():
    chunks = [
        make_chunk(
            "cisco_2025",
            "sec",
            "Item 1A. Risk Factors",
        ),
    ]

    result = assign_sections(chunks)

    assert "section" not in chunks[0]
    assert result[0]["section"] == "item_1a"
