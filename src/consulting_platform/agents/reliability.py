"""Deterministic presentation and coverage decisions; no generated facts."""

import re

from consulting_platform.agents.grounding import normalize_company
from consulting_platform.agents.port_tables import port_dimension
from consulting_platform.contracts import Domain

LABELS = {
    Domain.TECHNOLOGY: "Technology",
    Domain.CLINICAL: "Clinical",
    Domain.SEC: "SEC filings",
    Domain.CORPORATE_RESPONSIBILITY: "Food security and CSR",
}


def readable_candidate_label(label: str) -> str:
    # Older corpus chunks tokenize product identifiers with spaces at hyphens.
    # Keep their exact source evidence, but display technical codes conventionally.
    if re.fullmatch(r"[A-Za-z0-9]+(?:\s*-\s*[A-Za-z0-9]+)+", label) and re.search(r"\d", label):
        return re.sub(r"\s*-\s*", "-", label).upper()
    return label


def source_request(text: str) -> bool:
    """Legacy model questions about supplying sources are not customer preferences."""
    return bool(
        re.search(r"\b(provide|upload|supply|share|attach|missing|available)\b", text, re.I)
        and re.search(
            r"\b(evidence|documents?|documentation|excerpts?|filings?|10-k|datasheets?|specifications?|sources?|reports?)\b",
            text,
            re.I,
        )
    )


def consolidate_questions(questions: list[str]) -> list[str]:
    kept, tokens = [], []
    for question in questions:
        if source_request(question):
            continue
        words = set(re.findall(r"\w+", question.casefold())) - {
            "what",
            "which",
            "the",
            "a",
            "are",
            "is",
            "for",
            "do",
            "you",
            "your",
            "and",
            "of",
            "to",
        }
        if any(
            words == prior or (words and len(words & prior) / len(words | prior) >= 0.65)
            for prior in tokens
        ):
            continue
        kept.append(question)
        tokens.append(words)
    return kept


def coverage_warnings(results) -> list[str]:
    warnings = []
    for domain, result in results.items():
        label = LABELS[domain]
        chunks = result.get("chunks", [])
        if not chunks:
            warnings.append(
                f"{label}: No matching source passages were retrieved; this part remains unassessed."
            )
            continue
        companies = {normalize_company(c.company, domain) for c in chunks if c.company}
        for company in result["filters"].companies:
            if normalize_company(company, domain) not in companies:
                display = {
                    "cisco": "Cisco",
                    "agilent": "Agilent",
                    "alphabet": "Alphabet",
                    "google": "Google",
                    "ibm": "IBM",
                }.get(company.casefold(), company)
                warnings.append(
                    f"{label}: No matching passages for {display} were retrieved; that company's assessment remains unresolved."
                )
    return warnings


def comparison_summary(domain, result, recommendations) -> str | None:
    if not result["task"].compare_candidates:
        return None
    label = LABELS[domain]
    if not result["task"].constraints:
        return f"{label}: No product selected; specify the requirements to compare."
    ranked = result.get("rankings", [])
    accepted = {r.id for r in recommendations}
    eligible = [
        readable_candidate_label(r.label)
        for r in ranked
        if r.eligibility == "eligible" and f"{domain.value}:c:{r.id}" in accepted
    ]
    failed = [readable_candidate_label(r.label) for r in ranked if r.eligibility == "excluded"]
    parts = []
    criteria = "requirement" if len(result["task"].constraints) == 1 else "requirements"
    if len(result["task"].constraints) == 1:
        constraint = result["task"].constraints[0]
        speed = port_dimension(constraint.attribute, constraint.unit)
        if speed and isinstance(constraint.expected_value, (int, float)):
            rate = f"{speed / 1000:g}G" if speed >= 1000 else f"{speed:g}M"
            operator = {"gte": "at least", "lte": "at most", "eq": "exactly"}.get(constraint.operator)
            if operator:
                criteria += f" for {operator} {constraint.expected_value:g} {rate} Ethernet ports"
    if eligible:
        parts.append(
            f"{', '.join(eligible)} {'meets' if len(eligible) == 1 else 'meet'} your stated {criteria}. Overall deployment suitability still needs checking."
        )
    else:
        parts.append(
            f"{label}: No product recommendation could be validated for the stated requirements."
        )
    if failed:
        parts.append(f"{', '.join(failed)} {'does' if len(failed) == 1 else 'do'} not meet your stated {criteria}.")
    if ranked and all(r.eligibility == "excluded" for r in ranked):
        parts.append("No evaluated candidate satisfies all documented required constraints.")
    if not ranked or any(r.eligibility == "conditional" for r in ranked):
        parts.append(
            "One or more candidate comparisons remain unresolved because required attributes could not be validated."
        )
    return " ".join(parts)
