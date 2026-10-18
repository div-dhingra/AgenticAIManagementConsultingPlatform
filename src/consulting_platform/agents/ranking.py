"""Deterministic suitability comparison, independent of retail and model providers."""

import math
import re

from consulting_platform.agents.models import Candidate, Constraint, RankedCandidate

UNITS = {
    "gbps": ("network_rate", 1000),
    "mbps": ("network_rate", 1),
    "gb": ("memory", 1000),
    "mb": ("memory", 1),
    "samples": ("samples", 1),
    "sample": ("samples", 1),
    "samples/run": ("samples_per_run", 1),
    "samples per run": ("samples_per_run", 1),
    "ports": ("ports", 1),
    "port": ("ports", 1),
    "w": ("power", 1),
    "watts": ("power", 1),
}


def _compare(fact, constraint: Constraint) -> bool | None:
    left, right = fact.value, constraint.expected_value
    a, b = (fact.unit or "").strip().lower(), (constraint.unit or "").strip().lower()

    def count_unit(unit):
        match = re.fullmatch(r"(\d+(?:\.\d+)?[gm])\s+(?:ethernet\s+)?ports?", unit, re.I)
        if match and re.search(
            r"(?<![a-z0-9])" + re.escape(match[1]) + r"(?![a-z0-9])", constraint.attribute, re.I
        ):
            return "ports"
        return unit

    a, b = count_unit(a), count_unit(b)
    if a != b:
        if a not in UNITS or b not in UNITS or UNITS[a][0] != UNITS[b][0]:
            return None
        if isinstance(left, bool) or isinstance(right, bool):
            return None
        try:
            left, right = float(left) * UNITS[a][1], float(right) * UNITS[b][1]
        except (ValueError, TypeError):
            return None
    if constraint.operator in {"gte", "lte"}:
        if isinstance(left, bool) or isinstance(right, bool):
            return None
        try:
            lnum, rnum = float(left), float(right)
            if not math.isfinite(lnum) or not math.isfinite(rnum):
                return None
            return lnum >= rnum if constraint.operator == "gte" else lnum <= rnum
        except (ValueError, TypeError):
            return None
    if isinstance(left, bool) != isinstance(right, bool):
        return None
    if constraint.operator == "contains":
        return str(right).casefold() in str(left).casefold()
    return str(left).casefold() == str(right).casefold()


def rank_candidates(
    candidates: list[Candidate],
    requirements: list[Constraint],
    top_k: int = 5,
) -> list[RankedCandidate]:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    if len({c.id for c in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    if len({c.id for c in requirements}) != len(requirements):
        raise ValueError("Constraint IDs must be unique")
    results = []
    for candidate in candidates:
        matched, unknown, violated = [], [], []
        numerator = 0.0
        hard_unknown = not requirements
        hard_fail = False
        for requirement in requirements:
            fact = candidate.attributes.get(requirement.attribute)
            comparison = _compare(fact, requirement) if fact else None
            if comparison is None:
                unknown.append(requirement.id)
                hard_unknown |= requirement.required
            elif comparison:
                matched.append(requirement.id)
                numerator += requirement.weight
            else:
                violated.append(requirement.id)
                hard_fail |= requirement.required
        eligibility = "excluded" if hard_fail else "conditional" if hard_unknown else "eligible"
        total = sum(c.weight for c in requirements)
        evidence_ids = sorted(
            {s.evidence_id for s in candidate.identity_supports}
            | {s.evidence_id for fact in candidate.attributes.values() for s in fact.supports}
        )
        results.append(
            RankedCandidate(
                id=candidate.id,
                label=candidate.label,
                rank=1,
                score=numerator / total if total else 0,
                eligibility=eligibility,
                matched_constraint_ids=matched,
                unknown_constraint_ids=unknown,
                violated_constraint_ids=violated,
                evidence_ids=evidence_ids,
                explanation=f"{len(matched)} matched, {len(unknown)} unknown, {len(violated)} unmet constraints.",
            )
        )
    order = {"eligible": 0, "conditional": 1, "excluded": 2}
    results.sort(key=lambda r: (order[r.eligibility], -r.score, r.id))
    return [r.model_copy(update={"rank": i + 1}) for i, r in enumerate(results[:top_k])]
