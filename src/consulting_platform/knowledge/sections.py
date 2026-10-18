
import re


SECTION_PATTERNS = {
    "item_1a": r"item\s+1a\s*\.\s*risk\s+factors",
    "item_1b": r"item\s+1b\s*\.\s*unresolved\s+staff\s+comments",
    "item_1c": r"item\s+1c\s*\.\s*cybersecurity",
    "item_2": r"item\s+2\s*\.\s*properties",
    "item_3": r"item\s+3\s*\.\s*legal\s+proceedings",
    "item_4": r"item\s+4\s*\.\s*mine\s+safety\s+disclosures",
    "item_5": r"item\s+5\s*\.\s*market\s+for",
    "item_6": r"item\s+6\s*\.\s*reserved",
    "item_7": r"item\s+7\s*\.\s*management",
    "item_7a": r"item\s+7a\s*\.\s*quantitative",
    "item_8": r"item\s+8\s*\.\s*financial\s+statements",
    "item_9": r"item\s+9\s*\.\s*changes\s+in",
    "item_9a": r"item\s+9a\s*\.\s*controls",
    "item_9b": r"item\s+9b\s*\.\s*other\s+information",
    "item_10": r"item\s+10\s*\.\s*directors",
    "item_11": r"item\s+11\s*\.\s*executive\s+compensation",
    "item_12": r"item\s+12\s*\.\s*security\s+ownership",
    "item_13": r"item\s+13\s*\.\s*certain\s+relationships",
    "item_14": r"item\s+14\s*\.\s*principal\s+accountant",
    "item_15": r"item\s+15\s*\.\s*exhibits",
}


def detect_section_heading(text: str) -> str | None:
    """
    Detect an SEC section heading near the beginning of a chunk.

    Returns a section name such as 'item_1a',
    or None if no heading is found.
    """

    normalized_text = " ".join(text.lower().split())

    # Some PDF pages begin with a repeated
    # "table of contents" navigation label.
    if normalized_text.startswith("table of contents "):
        normalized_text = normalized_text[len("table of contents "):]

    for section_name, pattern in SECTION_PATTERNS.items():
        match = re.match(pattern, normalized_text)

        if match:
            return section_name

    return None



def detect_section_boundary(text: str) -> str | None:
    """
    Detect a reliable section boundary inside a chunk.

    Currently handles the short Item 1B section
    found in the Cisco FY2025 report.
    """

    normalized_text = " ".join(text.lower().split())

    # Item 1B is a short section with "none."
    # immediately after its heading.
    pattern = (
        r"item\s+1b\s*\.\s*"
        r"unresolved\s+staff\s+comments\s+none\s*\."
    )

    match = re.search(pattern, normalized_text)

    if match is None:
        return None

    # Reject a heading at the beginning of a chunk.
    # detect_section_heading() handles that case.
    if match.start() == 0:
        return None

    # Only accept a boundary near the end.
    remaining_text = normalized_text[match.end():]

    if len(remaining_text) < 30:
        return "item_1b"

    return None


def assign_sections(chunks: list[dict]) -> list[dict]:
    """
    Assign SEC sections to chunks in document order.

    This function does not modify the original chunks.
    """

    updated_chunks = []

    current_document_id = None
    current_section = None

    for chunk in chunks:
        document_id = chunk["document_id"]

        # Reset section when moving to another document.
        if document_id != current_document_id:
            current_document_id = document_id
            current_section = None

        updated_chunk = chunk.copy()

        # Only detect SEC sections.
        if chunk["domain"] == "sec":
            heading = detect_section_heading(chunk["text"])

            if heading is not None:
                current_section = heading

            # Assign current section before processing
            # a boundary at the end of the chunk.
            updated_chunk["section"] = current_section

            boundary = detect_section_boundary(chunk["text"])

            if boundary is not None:
                current_section = boundary

        else:
            updated_chunk["section"] = None

        updated_chunks.append(updated_chunk)

    return updated_chunks
