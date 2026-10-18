"""Customer-facing prose composed exclusively from accepted claims and actions."""

import re

from consulting_platform.agents.questions import CLINICAL_SUBJECTS
from consulting_platform.agents.reliability import comparison_summary
from consulting_platform.contracts import Domain

HEADINGS = {
    Domain.TECHNOLOGY: "Equipment decision",
    Domain.CLINICAL: "Laboratory assessment",
    Domain.SEC: "Vendor risks",
    Domain.CORPORATE_RESPONSIBILITY: "Food-security priorities and proposed implications",
}


def compose_answer(results, findings, recommendations, citations, gaps, questions):
    numbers = {c.evidence_id: i for i, c in enumerate(citations, 1)}

    def refs(ids):
        return " ".join(f"[{numbers[i]}]" for i in dict.fromkeys(ids) if i in numbers)

    paragraphs = []
    for domain, result in results.items():
        parts = []
        decision = comparison_summary(domain, result, recommendations)
        if decision:
            evidence = [
                e
                for c in result["analysis"].candidates
                for s in [
                    *c.identity_supports,
                    *[s for f in c.attributes.values() for s in f.supports],
                ]
                for e in [s.evidence_id]
            ]
            parts.append(decision + " " + refs(evidence))
        domain_findings = [f for f in findings if f.domain == domain]
        clinical_shown = set()
        if domain == Domain.CLINICAL:
            labels = ("Seahorse XF analyzers", "Digital pathology slide scanners")
            request_text = (
                result["request"].question if result.get("request") else result["task"].query
            )
            for (subject, pattern), label in zip(CLINICAL_SUBJECTS, labels):
                if not re.search(subject, request_text, re.I):
                    continue
                matches = [
                    f
                    for f in domain_findings
                    if re.search(pattern, f.statement, re.I)
                    or (
                        # A source title helps place a pronoun-only follow-on, but
                        # a broad title cannot turn a Seahorse claim into a scanner
                        # explanation (or the reverse).
                        not any(
                            re.search(other, f.statement, re.I) for _, other in CLINICAL_SUBJECTS
                        )
                        and re.search(
                            pattern,
                            " ".join(c.title for c in citations if c.evidence_id in f.evidence_ids),
                            re.I,
                        )
                    )
                ]
                if matches:
                    parts.append(
                        f"**{label}**\n\n"
                        + "\n".join(
                            "- "
                            + ("Assessment (inference): " if f.assessment == "inference" else "")
                            + f.statement.strip()
                            + " "
                            + refs(f.evidence_ids)
                            for f in matches
                        )
                    )
                    clinical_shown.update(f.id for f in matches)
                else:
                    parts.append(
                        f"**{label}**\n\nThis requested part could not be answered from validated evidence."
                    )
        explanations = {
            f.id.removeprefix("sec:explain"): f
            for f in domain_findings
            if f.id.startswith("sec:explain")
        }
        if domain == Domain.SEC:
            grouped = {}
            for finding in domain_findings:
                if finding.id.startswith("sec:explain"):
                    continue
                display = explanations.get(finding.id.removeprefix("sec:f"), finding)
                if (
                    display == finding
                    and explanations
                    and re.search(r"%|[$€£]|\b(?:million|billion)\b", finding.statement)
                    and not re.search(
                        r"how much|amount|percent|quantif|calculat|ratio|\bfigures?\b",
                        result["task"].query,
                        re.I,
                    )
                ):
                    # Keep unsolicited numerical detail in validated findings,
                    # not ahead of the requested qualitative risk explanation.
                    continue
                source = next((c for c in citations if c.evidence_id in finding.evidence_ids), None)
                company = source.company if source and source.company else "Company disclosures"
                period = f" — fiscal {source.report_year}" if source and source.report_year else ""
                statement = display.statement.strip()
                if display == finding and "filing discloses: “" in statement:
                    statement = "“" + statement.split("filing discloses: “", 1)[1]
                label = "Assessment (inference): " if display.assessment == "inference" else ""
                grouped.setdefault(company + period, []).append(
                    label + statement + " " + refs(display.evidence_ids)
                )
            for company, statements in grouped.items():
                parts.append(
                    f"**{company}**\n\n" + "\n".join(f"- {s}" for s in dict.fromkeys(statements))
                )
        csr_years = (
            {m[1] for f in domain_findings if (m := re.match(r"SOFI\s+(20\d{2})\s+", f.statement))}
            if domain == Domain.CORPORATE_RESPONSIBILITY
            else set()
        )
        csr_points = []
        for finding in domain_findings:
            if finding.id in clinical_shown:
                continue
            if finding.domain != domain:
                continue
            if domain == Domain.SEC:
                continue
            prefix = "Assessment (inference): " if finding.assessment == "inference" else ""
            statement = finding.statement.strip()
            if not statement.endswith((".", "!", "?", "”")):
                statement += "."
            if len(csr_years) == 1:
                year = next(iter(csr_years))
                statement = re.sub(
                    rf"^SOFI\s+{year}\s+(?:states that\s+|identifies the need to\s+)?",
                    "",
                    statement,
                )
                statement = statement[:1].upper() + statement[1:]
                csr_points.append("- " + prefix + statement + " " + refs(finding.evidence_ids))
            else:
                parts.append(prefix + statement + " " + refs(finding.evidence_ids))
        if csr_points:
            parts.append(
                f"SOFI {next(iter(csr_years))} highlights these priorities and constraints:\n\n"
                + "\n".join(csr_points)
            )
        for action in recommendations:
            if action.kind == "product" or not action.id.startswith(f"{domain.value}:"):
                continue
            if not any(f.domain == domain for f in findings):
                premise = next((c for c in citations if c.evidence_id in action.evidence_ids), None)
                if premise:
                    parts.append(
                        f"Source premise: “{premise.excerpt}” " + refs([premise.evidence_id])
                    )
            title = action.title.rstrip(".! ")
            rationale = action.rationale
            if rationale.startswith(
                "Proposed feasibility assessment motivated by the cited report's data needs."
            ):
                rationale = (
                    "First check whether sharing standardized nutrition-program records would improve "
                    "data completeness or reporting speed. That is a hypothesis to test, not a benefit "
                    "established by SOFI or proof that buying a router improves food security."
                )
            parts.append(
                f"**Suggested next step: {title}.** {rationale} "
                + refs(action.evidence_ids)
                + "\n\n"
                + " ".join(action.next_steps)
                + "\n\nMeasure the result: "
                + "; ".join(action.success_metrics)
            )
        if parts:
            paragraphs.append(f"**{HEADINGS[domain]}**\n\n" + "\n\n".join(parts))
    if gaps:
        paragraphs.append("**What remains unresolved**\n\n" + "\n".join(f"- {g}" for g in gaps))
    if questions:
        paragraphs.append(
            "The questions below identify the details still needed before a decision."
        )
    return re.sub(r"[ \t]+\n", "\n", "\n\n".join(paragraphs))
