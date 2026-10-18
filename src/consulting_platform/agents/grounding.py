"""Mechanical grounding checks. Semantic entailment additionally needs review."""

import re

from consulting_platform.agents.evidence_spans import evidence_spans
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
        if support.span_id is not None:
            span = next(
                (s for s in evidence_spans(chunk.text) if s["span_id"] == support.span_id), None
            )
            if span is None:
                return [], "unknown source span ID"
            if quote and " ".join(quote.split()) != " ".join(span["quote"].split()):
                return [], "quote disagrees with referenced source span"
            quote = span["quote"]
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
        resolved.append(support.model_copy(update={"quote": original, "span_id": None}))
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


def _table_port_supports(candidate, name, fact, supports, registry):
    """Verify a speed-specific count against its actual table cell and header.

    This is deterministic extraction from a retrieved table, not quote repair.
    A model may cite the full row but omit the header needed to interpret it.
    """
    speed = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?[gm])(?![a-z0-9])", name, re.I)
    if not speed or "port" not in name.casefold():
        return supports, None, False
    for support in supports:
        chunk = registry[support.evidence_id]
        lines = [line for line in chunk.text.splitlines() if "|" in line]
        for header in lines:
            cells = [c.strip() for c in header.split("|")]
            if cells[0].casefold() not in {"model", "product", "instrument"}:
                continue
            columns = [
                i
                for i, c in enumerate(cells)
                if "port" in c.casefold()
                and re.search(r"(?<![a-z0-9])" + re.escape(speed[1]) + r"(?![a-z0-9])", c, re.I)
            ]
            rows = [
                line
                for line in lines
                if line.split("|")[0].strip() == candidate.label
                and " ".join(line.split()) in " ".join(support.quote.split())
            ]
            if len(columns) != 1 or len(rows) != 1:
                continue
            row_cells = [c.strip() for c in rows[0].split("|")]
            column = columns[0]
            if len(row_cells) != len(cells):
                return supports, "ambiguous source table layout", True
            try:
                valid = not isinstance(fact.value, bool) and float(row_cells[column]) == float(
                    fact.value
                )
            except (ValueError, TypeError):
                valid = False
            if not valid:
                return supports, "source table column does not support the attribute value", True
            if not any(
                s.evidence_id == support.evidence_id and s.quote == header for s in supports
            ):
                supports = [*supports, support.model_copy(update={"quote": header})]
            return supports, None, True
    return supports, None, False


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
            if issue:
                # A table header plus a nonadjacent model row must become TWO
                # verbatim supports, never an invented contiguous quotation.
                expanded = []
                for support in fact.supports:
                    chunk = registry.get(support.evidence_id)
                    lines = chunk.text.splitlines() if chunk else []
                    headers = [
                        line
                        for line in lines
                        if "|" in line
                        and line.split("|")[0].strip().casefold()
                        in {"model", "product", "instrument"}
                    ]
                    rows = [
                        line
                        for line in lines
                        if "|" in line and line.split("|")[0].strip() == candidate.label
                    ]
                    pair = next(
                        (
                            (h, r)
                            for h in headers
                            for r in rows
                            if " ".join(support.quote.split()) == " ".join(f"{h} {r}".split())
                        ),
                        None,
                    )
                    expanded.extend(
                        [support.model_copy(update={"quote": q}) for q in pair]
                        if pair
                        else [support]
                    )
                supports, issue = resolve_supports(expanded, registry)
            if not issue and any("|" in s.quote for s in supports):
                rows = [
                    line
                    for s in supports
                    for line in registry[s.evidence_id].text.splitlines()
                    if "|" in line and line.split("|")[0].strip() == candidate.label
                ]
                if rows and not any(
                    " ".join(row.split()) in " ".join(s.quote.split())
                    for row in rows
                    for s in supports
                ):
                    issue = "table attribute does not cite this candidate's row"
            table_verified = False
            if not issue:
                supports, table_issue, table_verified = _table_port_supports(
                    candidate, name, fact, supports, registry
                )
                issue = table_issue
            # A speed-qualified count unit is redundant only if the same speed
            # remains explicit in the attribute key. No rate/count conversion.
            unit = "ports" if table_verified and not issue else fact.unit
            speed_unit = re.fullmatch(
                r"(\d+(?:\.\d+)?[gm])\s+(?:ethernet\s+)?ports?", unit or "", re.I
            )
            if speed_unit and re.search(
                r"(?<![a-z0-9])" + re.escape(speed_unit[1]) + r"(?![a-z0-9])", name, re.I
            ):
                unit = "ports"
            if (
                not issue
                and _value_in_quotes(fact.value, supports)
                and _unit_in_quotes(unit, supports)
            ):
                attributes[name] = fact.model_copy(update={"supports": supports, "unit": unit})
            else:
                reason = issue or (
                    "attribute value absent from quotes"
                    if not _value_in_quotes(fact.value, supports)
                    else "attribute unit absent from quotes"
                )
                warnings.append(
                    f"Removed unsupported attribute {name} from candidate {candidate.id}. Reason: {reason}."
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
            if not issue:
                statement = getattr(item, "statement", getattr(item, "rationale", ""))
                source = " ".join(
                    " ".join(
                        registry[e].text for e in dict.fromkeys(s.evidence_id for s in supports)
                    ).split()
                )
                # A combined financial measure cannot be assigned to one of its
                # components. No amount or company is hard-coded here.
                amounts = re.findall(r"\$?\s*\d[\d,.]*\s*(?:million|billion)", statement, re.I)
                for amount in amounts:
                    pattern = re.escape(amount.strip()).replace(r"\ ", r"\s+")
                    if (
                        re.search(pattern, source, re.I)
                        and re.search(
                            r"inventory\s+provisions|provisions\s+for\s+inventory", source, re.I
                        )
                        and re.search(
                            r"purchase.commitment.*liabilit|liabilit.*purchase.commitment",
                            source,
                            re.I,
                        )
                        and re.search(
                            r"purchase.commitment.*liabilit|liabilit.*purchase.commitment",
                            statement,
                            re.I,
                        )
                        and not re.search(
                            r"inventory\s+provisions|provisions\s+for\s+inventory", statement, re.I
                        )
                    ):
                        issue = "combined financial measure mislabeled as a standalone liability"
                        break
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
