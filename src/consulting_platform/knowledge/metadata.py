
import re
from pathlib import Path

import pymupdf


def identify_company(file_path: Path) -> str | None:
    """Identify company using its folder name."""

    folder_names = [
        folder.lower()
        for folder in file_path.parts
    ]

    if "cisco" in folder_names:
        return "Cisco"

    if "google" in folder_names or "gcp" in folder_names:
        return "Google"

    if "agilent" in folder_names:
        return "Agilent"

    if "alphabet" in folder_names:
        return "Alphabet"

    return None


def identify_report_year(file_path: Path) -> int | None:
    """Read the fiscal year from the first pages of a PDF."""

    with pymupdf.open(file_path) as pdf:
        first_pages = ""

        for page_number in range(min(5, len(pdf))):
            first_pages += pdf[page_number].get_text()
            first_pages += "\n"

    patterns = [
        r"fiscal year ended\s+\w+\s+\d{1,2},?\s+(20\d{2})",
        r"fiscal year ended\s+\w+\s+(20\d{2})",
        r"for the fiscal year ended\s+\w+\s+\d{1,2},?\s+(20\d{2})",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            first_pages,
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

    return None


def identify_filing_type(file_path: Path) -> str | None:
    """Identify SEC filing type from PDF content."""

    with pymupdf.open(file_path) as pdf:
        text = ""

        for page_number in range(min(3, len(pdf))):
            text += pdf[page_number].get_text()

    if re.search(r"FORM\s+10-K", text, re.IGNORECASE):
        return "10-K"

    if re.search(r"FORM\s+10-Q", text, re.IGNORECASE):
        return "10-Q"

    return None
