"""Presentation helpers; never add facts or change recommendation eligibility."""

import re

RANK_CHECKS = re.compile(
    r"(\d+) matched, (\d+) unknown, (\d+) unmet constraints\. Eligibility: (eligible|conditional|ineligible)\."
)


def diagnostic_warning(text: str) -> bool:
    return text.startswith(
        (
            "Removed ",
            "Evidence rejection:",
            "Rejected ",
            "Discarded ",
            "Evidence review reported limitations",
        )
    ) or any(
        phrase in text
        for phrase in (
            "see evidence check details",
            "See evidence check details",
            "The specialist reported analysis limitations",
        )
    )


def readable_rationale(text: str) -> str:
    match = RANK_CHECKS.fullmatch(text)
    if not match:
        return text
    matched, unknown, unmet = (int(match[i]) for i in (1, 2, 3))
    parts = [f"Meets {matched} of your stated requirements."]
    if unknown:
        parts.append(
            f"{unknown} {'requirement still needs' if unknown == 1 else 'requirements still need'} verification."
        )
    if unmet:
        parts.append(f"Does not meet {unmet} {'requirement' if unmet == 1 else 'requirements'}.")
    return " ".join(parts)


def citation_title(citation: dict) -> str:
    """Replace opaque SEC accession IDs with their known company/year metadata."""
    title = citation["title"]
    if citation.get("domain") == "sec" and re.fullmatch(r"[\d-]+", title):
        company = citation.get("company") or "Company"
        year = citation.get("report_year")
        return f"{company} Form 10-K" + (f", fiscal {year}" if year else "")
    return title
