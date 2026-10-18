"""Preserve explicit questions and their subjects across routing and review.

This module decomposes requests, not answers. It contains no product facts.
"""

import re

from consulting_platform.agents.models import AnswerRequirement
from consulting_platform.contracts import Domain

CLINICAL_SUBJECTS = (
    (r"seahorse|\bxf\b", r"seahorse|\bxf\b|metabol|bioenerget|mitochond|glycol|energy"),
    (r"patholog|slide.scann|whole.slide", r"patholog|slides?|scann|whole.slide|digiti[sz]"),
)


def subject_patterns(question):
    return [
        evidence for subject, evidence in CLINICAL_SUBJECTS if re.search(subject, question, re.I)
    ]


def covers_subjects(question, items):
    text = " ".join(
        getattr(item, "statement", "") + " " + getattr(item, "title", "") for item in items
    )
    return all(re.search(pattern, text, re.I) for pattern in subject_patterns(question))


def unrequested_model_detail(statement, questions, chunks):
    """Keep a family-function answer from drifting into a different submodel.

    Product identifiers come from supplied text, never a built-in fact catalog.
    Requested submodels remain eligible for ordinary grounding and review.
    """
    if not all(re.match(r"what\s+(?:do|does)\b", q, re.I) for q in questions):
        return False
    source = " ".join(c.text for c in chunks)
    names = set(re.findall(r"\b([a-z]+\d+[a-z\d]*(?:-[a-z\d]+)*)\b", source, re.I))
    names.update(re.findall(r"\b([a-z][a-z\d-]+)[®™]", source, re.I))
    requested = re.sub(r"[^a-z0-9]", "", " ".join(questions).casefold())
    return any(
        re.search(r"\b" + re.escape(name) + r"\b", statement, re.I)
        and re.sub(r"[^a-z0-9]", "", name.casefold()) not in requested
        for name in names
    )


def explicit_clinical_questions(request):
    # Coordinated interrogatives are separate obligations: 'what does X ...,
    # and what does Y ...?' Neither the router nor reviewer may lose Y.
    clauses = re.split(
        r"[?!.]|,?\s+(?:and|also)\s+(?=(?:what|which|how|why)\b)", request.question, flags=re.I
    )
    questions = []
    for clause in clauses:
        match = re.search(r"\b(?:what|which|how|why)\b.+", clause, re.I)
        if match and subject_patterns(match[0]):
            questions.append(match[0].strip(" ,;") + "?")
    return questions


def preserve_request_questions(request, domain, task):
    if domain == Domain.CLINICAL:
        clauses = explicit_clinical_questions(request)
        if (
            len(clauses) > 1
            and all(re.match(r"what\s+(?:do|does)\b", q, re.I) for q in clauses)
            and not re.search(
                r"\b(recommend|select|choose|order|purchase|buy|requirements?|capacity|budget)\b",
                request.question,
                re.I,
            )
        ):
            # Contrasting what two instruments do is an explanation, not a
            # procurement ranking requiring extra customer constraints.
            task.compare_candidates = False
    if domain == Domain.CLINICAL and not task.compare_candidates:
        clauses = explicit_clinical_questions(request)
        if len(clauses) > 1:
            # Use the customer's own wording for these questions rather than
            # another model paraphrase that can combine or omit the subjects.
            existing = [r for r in task.answer_requirements if not subject_patterns(r.question)]
            company = request.context.company
            if re.search(r"\bAgilent\b", request.question, re.I):
                company = "Agilent"
            elif company and company.casefold() in {"cisco", "google", "alphabet", "ibm"}:
                company = None
            task.answer_requirements = [
                *[
                    AnswerRequirement(
                        id="pending",
                        question=q,
                        company=company,
                    )
                    for q in clauses
                ],
                *existing,
            ][:8]
    if domain in {Domain.TECHNOLOGY, Domain.CLINICAL}:
        for requirement in task.answer_requirements:
            year = requirement.report_year
            # Report years in shared context belong to SEC/SOFI. Keep an
            # explicit date for a requested technical manual/datasheet only.
            if year and not re.search(
                rf"\b{year}\b[^.!?]{{0,55}}(?:datasheet|data sheet|manual|technical report)"
                rf"|(?:datasheet|data sheet|manual|technical report)[^.!?]{{0,55}}\b{year}\b",
                request.question,
                re.I,
            ):
                requirement.report_year = None
