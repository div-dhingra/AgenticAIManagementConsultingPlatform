"""Keep cross-domain connectivity proposals at the feasibility stage."""

import re

from consulting_platform.agents.models import ActionDraft, FindingDraft

TECH = re.compile(
    r"connect\w*|routers?|networks?|digital|data.sharing|platforms?|infrastructure|hardware", re.I
)
COMMIT = re.compile(
    r"\b(invest(?:s|ed|ing|ments?)?|procur\w*|purchas\w*|buy|order|deploy\w*|implement\w*|install\w*|roll.out)\b",
    re.I,
)
ASSESS = re.compile(
    r"\b(assess\w*|evaluat\w*|review\w*|investigat\w*|pilot\w*|test\w*|audit\w*|map\w*|defin\w*|consult\w*)\b",
    re.I,
)

FEASIBILITY_RATIONALE = (
    "Proposed feasibility assessment motivated by the cited report's data needs. "
    "Sharing standardized nutrition-program records between sites could improve "
    "data completeness and reporting timeliness; this is an untested hypothesis. "
    "Those effects and any food-security outcomes remain unproven."
)


def qualify_csr_actions(analysis, request=None):
    actions, findings, notes = [], list(analysis.findings), []
    for action in analysis.actions:
        # Grounding has already validated this premise, independently of whether
        # the proposed intervention is acceptable. Keep the report's own words.
        if not findings:
            support = next((s for s in action.supports if len(s.quote.split()) >= 8), None)
            if support:
                findings.append(
                    FindingDraft(
                        statement=f"The report states: “{' '.join(support.quote.split())}”",
                        supports=[support],
                    )
                )
        text = " ".join([action.title, action.rationale, *action.next_steps])
        if TECH.search(text):
            if COMMIT.search(" ".join([action.title, *action.next_steps])) or not ASSESS.search(
                action.title
            ):
                notes.append(
                    f"Excluded premature connectivity implementation/spending proposal: {action.title}"
                )
                continue
            action = action.model_copy(
                update={
                    "rationale": FEASIBILITY_RATIONALE,
                    "prerequisites": list(
                        dict.fromkeys(
                            [
                                *action.prerequisites,
                                "Validate the proposed benefit and compare alternatives before any spending decision.",
                            ]
                        )
                    ),
                    "success_metrics": [
                        "Compare data completeness and reporting delay with the measured baseline; agree acceptance targets with stakeholders."
                    ],
                }
            )
        actions.append(action)
    # A user-requested feasibility assessment is a bounded domain method, not a
    # factual prediction by the model. Supply it when there is an exact cited
    # data-needs premise, even if generation omitted the proposed assessment.
    wants_assessment = (
        request
        and re.search(r"hypothet|could|pilot|feasibility", request.question, re.I)
        and TECH.search(request.question)
    )
    relevant = next(
        (
            support
            for finding in findings
            for support in finding.supports
            if re.search(r"\bdata\b", support.quote, re.I)
            and re.search(
                r"need|must|better|advanc|improv|gap|lack|develop|standard|track",
                support.quote,
                re.I,
            )
        ),
        None,
    )
    if (
        wants_assessment
        and relevant
        and not any(
            a.rationale == FEASIBILITY_RATIONALE
            and any(re.search(r"\bdata\b", q.quote, re.I) for q in a.supports)
            for a in actions
        )
    ):
        actions.append(
            ActionDraft(
                kind="diagnostic_action",
                title="Assess nutrition-program data sharing before a connectivity investment",
                rationale=FEASIBILITY_RATIONALE,
                supports=[relevant],
                prerequisites=[
                    "Validate the proposed benefit and compare alternatives before any spending decision."
                ],
                next_steps=[
                    "Map existing data-sharing bottlenecks and measure a baseline.",
                    "Design a small feasibility test comparing connectivity and non-connectivity options.",
                ],
                success_metrics=[
                    "Compare data completeness and reporting delay with the measured baseline; agree acceptance targets with stakeholders."
                ],
            )
        )
    return analysis.model_copy(update={"actions": actions, "findings": findings}), notes
