import pytest

from consulting_platform.agents.models import AttributeFact, Candidate, Constraint, Support
from consulting_platform.agents.ranking import rank_candidates


def candidate(name="A", value=16, unit="samples/run", attribute="capacity"):
    support = Support(evidence_id="e1", quote="A supports 16 samples per run.")
    return Candidate(
        id=name,
        label=name,
        identity_supports=[support],
        attributes={
            attribute: AttributeFact(value=value, unit=unit, supports=[support]),
        },
    )


def constraint(value=12, unit="samples/run", attribute="capacity", **kwargs):
    return Constraint(
        id="minimum", attribute=attribute, operator="gte", expected_value=value, unit=unit, **kwargs
    )


def test_hard_constraint_excludes_wrong_capacity():
    results = rank_candidates([candidate("B", 8), candidate("A", 16)], [constraint()])
    assert [(r.id, r.eligibility) for r in results] == [("A", "eligible"), ("B", "excluded")]


def test_unknown_is_conditional_and_does_not_receive_full_score():
    result = rank_candidates([candidate()], [constraint(attribute="unknown")])[0]
    assert result.eligibility == "conditional"
    assert result.score == 0
    assert result.unknown_constraint_ids == ["minimum"]


def test_no_requirements_never_establish_suitability():
    assert rank_candidates([candidate()], [])[0].eligibility == "conditional"


def test_convert_compatible_units_only():
    rate = candidate(value=1000, unit="Mbps", attribute="rate")
    assert (
        rank_candidates([rate], [constraint(value=1, unit="Gbps", attribute="rate")])[0].eligibility
        == "eligible"
    )
    assert (
        rank_candidates([rate], [constraint(value=1, unit="GB", attribute="rate")])[0].eligibility
        == "conditional"
    )


def test_bool_cannot_satisfy_numeric_requirement():
    assert rank_candidates([candidate(value=True)], [constraint()])[0].eligibility == "conditional"


def test_deterministic_tie_order():
    assert [r.id for r in rank_candidates([candidate("B"), candidate("A")], [constraint()])] == [
        "A",
        "B",
    ]


def test_duplicate_candidate_ids_fail():
    with pytest.raises(ValueError, match="unique"):
        rank_candidates([candidate(), candidate()], [constraint()])
