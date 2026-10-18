
import pytest

from consulting_platform.knowledge.query_intent import (
    get_risk_section_boost,
)


@pytest.mark.parametrize(
    "query, expected_boost",
    [
        (
            "What are Cisco's major risk factors?",
            0.10,
        ),
        (
            "Cisco Item 1A Risk Factors cybersecurity",
            0.10,
        ),
        (
            "What risks does Cisco discuss regarding competition and supply chains?",
            0.05,
        ),
        (
            "What cybersecurity risks does Cisco identify?",
            0.00,
        ),
        (
            "How does Cisco manage cybersecurity governance?",
            0.00,
        ),
        (
            "What are Cisco's major revenue drivers?",
            0.00,
        ),
    ],
)
def test_risk_section_boost(query, expected_boost):
    actual_boost = get_risk_section_boost(query)

    assert actual_boost == expected_boost
