"""Mechanical grounding checks. Semantic entailment additionally needs review."""

import re

from consulting_platform.agents.models import Candidate, SpecialistAnalysis, Support
from consulting_platform.contracts import Domain, EvidenceChunk, RetrievalFilters


def normalize_company(company: str, domain: Domain) -> str:
    value = company.strip().casefold()
    if domain == Domain.SEC and value in {"google", "alphabet", "alphabet inc."}:
        return "alphabet"
    return {
        "cisco systems": "cisco",
        "cisco systems, inc.": "cisco",
        "agilent technologies": "agilent",
    }.get(value, value)


def filter_evidence(
    chunks: list[EvidenceChunk],
    domain: Domain,
    filters: RetrievalFilters,
) -> tuple[list[EvidenceChunk], list[str]]:
    accepted, rejected, seen = [], 0, set()
    for c in chunks:
        valid = c.domain == domain
        valid &= not filters.report_years or c.report_year in filters.report_years
        valid &= not filters.document_ids or c.document_id in filters.document_ids
        if filters.companies:
            valid &= c.company is not None and normalize_company(c.company, domain) in {
                normalize_company(name, domain) for name in filters.companies
            }
        if valid and c.chunk_id not in seen:
            accepted.append(c)
            seen.add(c.chunk_id)
        else:
            rejected += 1
    return accepted, (
        [f"Rejected {rejected} duplicate or out-of-filter evidence chunks."] if rejected else []
    )


def resolve_supports(
    supports: list[Support],
    registry: dict[str, EvidenceChunk],
) -> tuple[list[Support], str | None]:
    """Recover verbatim spans allowing whitespace and straight/curly quote glyphs.

    PDF extraction inserts line breaks; a model often emits spaces instead.
    Match the same case-sensitive words/numbers and punctuation in order (except
    typographic quote glyphs), then retain the ORIGINAL source substring. Never
    fuzzy-match wording, quantities, dashes, or ellipses.
    """
    if not supports:
        return [], "missing evidence support"
    resolved = []
    for support in supports:
        chunk = registry.get(support.evidence_id)
        if chunk is None:
            return [], "unknown evidence ID"
        quote = support.quote.strip()
        if not quote:
            return [], "empty evidence quote"
        if quote in chunk.text:
            original = quote
        else:
            glyphs = {character: "['‘’]" for character in "'‘’"}
            glyphs.update({character: '["“”]' for character in '"“”'})
            pattern = r"\s+".join(
                "".join(glyphs.get(character, re.escape(character)) for character in token)
                for token in quote.split()
            )
            match = re.search(pattern, chunk.text)
            if not match:
                return [], "quote does not match source words and punctuation"
            original = match.group(0)
        if len(original) > 400:
            return [], "source quote exceeds 400 characters; use a shorter excerpt"
        resolved.append(support.model_copy(update={"quote": original}))
    return resolved, None


def _value_in_quotes(value: str | float | bool, supports: list[Support]) -> bool:
    text = " ".join(s.quote for s in supports)
    if isinstance(value, bool):
        # Boolean capabilities often need semantic interpretation; conservatively
        # require explicit true/false instead of inferring capability from a noun.
        return str(value).casefold() in text.casefold()
    if isinstance(value, (float, int)):
        numbers = re.findall(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w.])", text)
        return any(float(n.replace(",", "")) == value for n in numbers)
    return str(value).casefold() in text.casefold()


def _unit_in_quotes(unit: str | None, supports: list[Support]) -> bool:
    if not unit:
        return True
    text = " ".join(s.quote for s in supports).casefold()
    aliases = {
        "samples/run": ["samples per run", "samples/run"],
        "samples per run": ["samples per run", "samples/run"],
        "samples": ["samples", "sample"],
        "sample": ["samples", "sample"],
        "ports": ["ports", "port"],
        "port": ["ports", "port"],
        "w": ["w", "watts"],
        "watts": ["w", "watts"],
    }
    return any(
        re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text)
        for value in aliases.get(unit.casefold(), [unit.casefold()])
    )


def grounded_candidates(
    candidates: list[Candidate],
    registry: dict[str, EvidenceChunk],
) -> tuple[list[Candidate], list[str]]:
    accepted, warnings, seen = [], [], set()
    for candidate in candidates:
        identity, issue = resolve_supports(candidate.identity_supports, registry)
        if (
            candidate.id in seen
            or issue
            or " ".join(candidate.label.casefold().split())
            not in " ".join(" ".join(s.quote for s in identity).casefold().split())
        ):
            warnings.append(
                "Removed a candidate with unsupported or duplicate identity."
                + (f" Reason: {issue}." if issue else "")
            )
            continue
        seen.add(candidate.id)
        attributes = {}
        for name, fact in candidate.attributes.items():
            supports, issue = resolve_supports(fact.supports, registry)
            if (
                not issue
                and _value_in_quotes(fact.value, supports)
                and _unit_in_quotes(fact.unit, supports)
            ):
                attributes[name] = fact.model_copy(update={"supports": supports})
            else:
                warnings.append(
                    f"Removed unsupported attribute {name} from candidate {candidate.id}."
                )
        accepted.append(
            candidate.model_copy(
                update={
                    "identity_supports": identity,
                    "attributes": attributes,
                }
            )
        )
    return accepted, warnings


def ground_analysis(
    analysis: SpecialistAnalysis,
    chunks: list[EvidenceChunk],
) -> tuple[SpecialistAnalysis, list[str]]:
    registry = {c.chunk_id: c for c in chunks}
    findings, actions, rejected = [], [], []
    for items, output in [(analysis.findings, findings), (analysis.actions, actions)]:
        for item in items:
            supports, issue = resolve_supports(item.supports, registry)
            if issue:
                rejected.append(issue)
            else:
                output.append(item.model_copy(update={"supports": supports}))
    candidates, warnings = grounded_candidates(analysis.candidates, registry)
    removed = len(analysis.findings) - len(findings) + len(analysis.actions) - len(actions)
    if removed:
        warnings.append(
            f"Removed {removed} claims/actions with invalid evidence references or quotes."
        )
        warnings.extend(f"Evidence rejection: {reason}." for reason in dict.fromkeys(rejected))
    return analysis.model_copy(
        update={
            "findings": findings,
            "actions": actions,
            "candidates": candidates,
        }
    ), warnings
