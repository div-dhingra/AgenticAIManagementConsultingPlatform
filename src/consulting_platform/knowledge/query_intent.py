
import re


def get_risk_section_boost(query: str) -> float:
    """
    Decide how much to boost SEC Item 1A Risk Factors.

    Returns:
        0.10 for explicit Item 1A or Risk Factors queries.
        0.05 for selected risk-related queries.
        0.00 for other queries.
    """

    query_lower = query.lower()

    # Explicit references to the SEC Risk Factors section.
    if (
        re.search(r"\bitem\s+1a\b", query_lower)
        or "risk factors" in query_lower
    ):
        return 0.10

    # Governance queries should not automatically
    # prefer the Risk Factors section.
    if (
        "governance" in query_lower
        or "manage cybersecurity" in query_lower
    ):
        return 0.00

    # Selected business risk topics.
    risk_terms = [
        "risk",
        "risks",
        "threat",
        "threats",
    ]

    business_topics = [
        "competition",
        "competitors",
        "supply chain",
        "suppliers",
        "market",
        "operations",
    ]

    has_risk_term = any(
        re.search(r"\b" + term + r"\b", query_lower)
        for term in risk_terms
    )

    has_business_topic = any(
        topic in query_lower
        for topic in business_topics
    )

    if has_risk_term and has_business_topic:
        return 0.05

    return 0.00
