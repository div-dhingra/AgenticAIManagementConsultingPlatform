"""Completeness is about accepted answers, not the number of warning messages."""

import re

from consulting_platform.agents.csr_actions import FEASIBILITY_RATIONALE
from consulting_platform.agents.grounding import normalize_company
from consulting_platform.agents.models import AnswerRequirement
from consulting_platform.agents.questions import covers_subjects
from consulting_platform.agents.reliability import LABELS, coverage_warnings
from consulting_platform.contracts import Domain


def requirements_for(results):
    return [
        {**requirement.model_dump(mode="json"), "domain": domain.value}
        for domain, result in results.items()
        for requirement in result["task"].answer_requirements
    ]


def source_proof_covers(requirement, domain, result, findings, citations, recommendations=()):
    """Only narrow, mechanically checkable answers bypass a model completeness vote.

    Free interpretations, numbers/ratios, causality and general suitability still
    require semantic review. Exact report quotations can answer a qualitative
    disclosure question without claiming that a forecast is an achieved outcome.
    """
    question = requirement.question.casefold()
    if domain == Domain.TECHNOLOGY and result["task"].compare_candidates:
        if not re.search(r"ports?|counts?", question) or re.search(
            r"security|encrypt|vpn|compatib|throughput|bandwidth|cost|price|latency|budget|reliab|redundan|power|overall|suitabil|deployment|repair|fault|failure|additional|beyond",
            question,
        ):
            return False
        evidence = [
            citations[e]
            for f in findings
            if f.id.startswith("technology:verified:")
            for e in f.evidence_ids
            if e in citations
        ]
        if requirement.company and not any(
            normalize_company(c.company or "", domain)
            == normalize_company(requirement.company, domain)
            for c in evidence
        ):
            return False
        if requirement.report_year and not any(
            c.report_year == requirement.report_year for c in evidence
        ):
            return False
        ranked = result.get("rankings", [])
        proven = set(result.get("table_proven_candidates", []))
        labels = {re.sub(r"[^a-z0-9]", "", r.label.casefold()) for r in ranked}
        named = (
            {
                re.sub(r"[^a-z0-9]", "", value.casefold())
                for value in re.findall(
                    r"\bC\d{4}(?:-[A-Z0-9]+)+", result["request"].question, re.I
                )
            }
            if result.get("request")
            else set()
        )
        return (
            bool(ranked)
            and named.issubset(labels)
            and all(r.id in proven and r.eligibility != "conditional" for r in ranked)
            and any(f.id.startswith("technology:verified:") for f in findings)
        )
    if domain == Domain.CORPORATE_RESPONSIBILITY:
        if not re.search(r"\b(data|information|reporting|monitoring)\b", question):
            return False
        if not re.search(r"hypothet|could|pilot|connectivity", question) or re.search(
            r"guarantee|roi|return on|how much", question
        ):
            return False
        approved = proposed_assessment_ids(result)
        return any(
            r.id in approved and r.rationale == FEASIBILITY_RATIONALE for r in recommendations
        )
    if domain != Domain.SEC or not re.search(r"\brisks?\b", question):
        return False
    if re.search(r"ratio|calculate|how much|percentage|return on|roi|guarantee|predict", question):
        return False
    quoted = []
    for finding in findings:
        if finding.domain != domain or "filing discloses: “" not in finding.statement:
            continue
        evidence = [citations[i] for i in finding.evidence_ids if i in citations]
        quote = finding.statement.split("filing discloses: “", 1)[1].removesuffix("”")
        sources = {c.chunk_id: c for c in result.get("chunks", [])}
        if not any(
            c.evidence_id in sources
            and " ".join(quote.casefold().split())
            in " ".join(sources[c.evidence_id].text.casefold().split())
            and (
                not requirement.company
                or normalize_company(c.company or "", domain)
                == normalize_company(requirement.company, domain)
            )
            and (not requirement.report_year or c.report_year == requirement.report_year)
            for c in evidence
        ):
            continue
        quoted.append(finding.statement.split("filing discloses: “", 1)[1].casefold())
    text = " ".join(quoted)
    # Match requested topics generically rather than treating any risk quotation
    # as an answer to every question about that company.
    stop = {
        "what",
        "which",
        "are",
        "the",
        "and",
        "for",
        "from",
        "does",
        "that",
        "this",
        "risks",
        "risk",
        "disclosed",
        "disclose",
        "filing",
        "filings",
        "report",
        "fiscal",
        "summarize",
        "summary",
        "list",
        "identify",
        "explain",
        "any",
        "also",
        "about",
        "cisco",
        "agilent",
        "alphabet",
        "google",
        "requested",
        "unavailable",
        "available",
        "assessment",
        "unresolved",
        "state",
        "indicate",
        "remains",
        "remain",
        "leave",
        "explicitly",
        "factors",
        "provided",
        "not",
        "only",
        "when",
        "otherwise",
    }
    topics = set(re.findall(r"[a-z]{3,}", question)) - stop
    if topics and not all(re.search(r"\b" + re.escape(word[:6]), text) for word in topics):
        return False
    return bool(quoted) and bool(
        re.search(
            r"risk|could|may|advers|uncertain|negativ|affect|declin|harm|sensitive|exposure", text
        )
    )


def proposed_assessment_ids(result):
    """Publish bounded feasibility hypotheses with exact, relevant data premises.

    This never endorses spending, rollout, causal benefits or achieved results.
    qualify_csr_actions owns the rationale and measurement template after checking
    the proposal's title/steps for premature implementation commitments.
    """
    request = result.get("request")
    if not request or not re.search(r"hypothet|could|pilot|feasibility", request.question, re.I):
        return set()
    return {
        f"corporate_responsibility:a{i + 1}"
        for i, action in enumerate(result["analysis"].actions)
        if action.rationale == FEASIBILITY_RATIONALE
        and any(
            re.search(r"\b(data|information|accounting|tracking|transparen\w*)\b", s.quote, re.I)
            for s in action.supports
        )
    }


def coverage_gaps(results, findings, recommendations, citations, checks):
    """A review cannot count rejected IDs or evidence from the wrong company/year."""
    gaps = list(coverage_warnings(results))
    findings_by_id = {f.id: f for f in findings}
    actions_by_id = {r.id: r for r in recommendations}
    lookup = {check.requirement_id: check for check in checks}
    for domain, result in results.items():
        task = result["task"]
        if not task.answer_requirements:
            # Compatibility with offline fixtures and older routing adapters.
            if not any(f.domain == domain for f in findings) and not any(
                r.id.startswith(f"{domain.value}:") for r in recommendations
            ):
                gaps.append(f"{LABELS[domain]}: No validated answer is available.")
        for requirement in task.answer_requirements:
            if source_proof_covers(
                requirement, domain, result, findings, citations, recommendations
            ):
                continue
            check = lookup.get(requirement.id)
            items = (
                []
                if check is None
                else [
                    *[findings_by_id[i] for i in check.finding_ids if i in findings_by_id],
                    *[actions_by_id[i] for i in check.recommendation_ids if i in actions_by_id],
                ]
            )
            scoped = [item for item in items if item.id.startswith(f"{domain.value}:")]
            subjects_ok = domain != Domain.CLINICAL or covers_subjects(requirement.question, scoped)
            evidence = [
                citations[e] for item in scoped for e in item.evidence_ids if e in citations
            ]
            company_ok = not requirement.company or any(
                normalize_company(c.company or "", domain)
                == normalize_company(requirement.company, domain)
                for c in evidence
            )
            year_ok = not requirement.report_year or any(
                c.report_year == requirement.report_year
                and (
                    not requirement.company
                    or normalize_company(c.company or "", domain)
                    == normalize_company(requirement.company, domain)
                )
                for c in evidence
            )
            if not (check and check.answered and scoped and company_ok and year_ok and subjects_ok):
                gaps.append(f"{LABELS[domain]}: Still unresolved — {requirement.question}")
        if task.compare_candidates and (
            not task.constraints
            or not result.get("rankings")
            or any(r.eligibility == "conditional" for r in result.get("rankings", []))
            or (
                any(r.eligibility == "eligible" for r in result.get("rankings", []))
                and not any(
                    r.eligibility == "eligible" and f"{domain.value}:c:{r.id}" in actions_by_id
                    for r in result.get("rankings", [])
                )
            )
        ):
            gaps.append(f"{LABELS[domain]}: The requested candidate comparison remains incomplete.")
    return list(dict.fromkeys(gaps))


def reconcile_review_questions(questions, results, findings, citations, recommendations):
    """A free-text reviewer objection cannot undo an exact verified port answer.

    Other domains and non-port interpretations continue to require normal review.
    """
    kept = []
    for question in questions:
        if question.startswith("Technology:") and Domain.TECHNOLOGY in results:
            text = question.split(":", 1)[1].removeprefix(" Still unresolved —").strip()
            years = re.findall(r"\b20\d{2}\b", text)
            requirement = AnswerRequirement(
                id="review-check",
                question=text,
                report_year=int(years[0]) if len(years) == 1 else None,
            )
            if source_proof_covers(
                requirement,
                Domain.TECHNOLOGY,
                results[Domain.TECHNOLOGY],
                findings,
                citations,
                recommendations,
            ):
                continue
        kept.append(question)
    return kept


def redundant_port_clarification(question, results, findings, citations, recommendations):
    if not re.search(r"additional|other|beyond|further", question, re.I) or not re.search(
        r"ports?|requirements?", question, re.I
    ):
        return False
    result = results.get(Domain.TECHNOLOGY)
    if not result:
        return False
    requirement = AnswerRequirement(
        id="scope-check", question="Which model meets the stated port count requirement?"
    )
    return source_proof_covers(
        requirement, Domain.TECHNOLOGY, result, findings, citations, recommendations
    )
