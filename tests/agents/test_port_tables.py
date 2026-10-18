import pytest

from consulting_platform.agents.controlled_evidence import ControlledEvidenceService
from consulting_platform.agents.models import Constraint, DomainTask
from consulting_platform.agents.port_tables import canonical_port_task, table_candidates
from consulting_platform.agents.ranking import rank_candidates


def chunk(text=None):
    original = ControlledEvidenceService.from_file(
        "fixtures/consulting/controlled_real_passages.json"
    ).evidence.chunks[0]
    return original.model_copy(update={"text": text}) if text else original


def task(attribute="ports_1g", unit="ports"):
    return DomainTask(
        query="Compare documented ports",
        compare_candidates=True,
        constraints=[
            Constraint(
                id="minimum", attribute=attribute, operator="gte", expected_value=5, unit=unit
            )
        ],
    )


@pytest.mark.parametrize(
    "attribute,unit",
    [
        ("1G_ethernet_ports", "count"),
        ("ports_1Gbps", "ports"),
        ("number_of_1g_ports", "1G Ethernet ports"),
        ("ports_1000mbps", "ports"),
        ("oneGPortCount", "1G ports"),
    ],
)
def test_table_counts_do_not_depend_on_model_candidates_or_key_spelling(attribute, unit):
    candidates, parsed, notes = table_candidates(
        [chunk()], task(attribute, unit), [], "Compare Cisco C8300 models"
    )
    ranked = rank_candidates(candidates, parsed.constraints)
    assert not notes
    assert [(r.label, r.eligibility) for r in ranked] == [
        ("C8300-2N2S-6T", "eligible"),
        ("C8300-2N2S-4T2X", "excluded"),
    ]
    assert all(
        s.quote in chunk().text
        for c in candidates
        for fact in c.attributes.values()
        for s in fact.supports
    )


def test_changed_source_values_change_selection_without_a_hardcoded_winner():
    source = chunk("Model | 10G port density | 1G port density\nBench-A | 2 | 9\nBench-B | - | 3")
    candidates, parsed, _ = table_candidates([source], task(), [], "Compare Bench-A and Bench-B")
    assert [(r.label, r.eligibility) for r in rank_candidates(candidates, parsed.constraints)] == [
        ("Bench-A", "eligible"),
        ("Bench-B", "excluded"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Model | 10G port density | 1G port density\nBench-A | 6 | -",
        "Model | 1G port density | 1G port density\nBench-A | 6 | 8",
        "Model | 1G port density\nBench-A | 6.5",
    ],
)
def test_dash_ambiguous_and_fractional_counts_remain_unknown(text):
    candidates, parsed, notes = table_candidates([chunk(text)], task(), [], "Compare Bench-A")
    assert notes and rank_candidates(candidates, parsed.constraints)[0].eligibility == "conditional"


def test_conflicting_sources_do_not_select_an_arbitrary_count():
    a = chunk("Model | 1G port density\nBench-A | 6")
    b = a.model_copy(
        update={"chunk_id": "conflicting-count", "text": "Model | 1G port density\nBench-A | 8"}
    )
    candidates, parsed, notes = table_candidates([a, b], task(), [], "Compare Bench-A")
    assert notes and rank_candidates(candidates, parsed.constraints)[0].eligibility == "conditional"


def test_explicit_customer_requirement_overrides_model_spelling_and_keeps_other_requirements():
    model_task = task("ports_10g")
    other = Constraint(
        id="power", attribute="power", operator="lte", expected_value=100, unit="watts"
    )
    model_task.constraints.append(other)
    canonical = canonical_port_task(model_task, ["At least 5 1G Ethernet ports per site"])
    candidates, canonical, _ = table_candidates([chunk()], canonical, [], "Compare Cisco routers")
    port = next(c for c in canonical.constraints if c.unit == "ports")
    assert port.attribute == "ports_1000mbps" and port.expected_value == 5
    assert canonical.constraints[0] == other
    # The other required attribute is unknown: no false overall suitability.
    assert all(
        r.eligibility != "eligible" for r in rank_candidates(candidates, canonical.constraints)
    )
    assert (
        canonical_port_task(model_task, ["At least 5 1G ports and redundant power"]) == model_task
    )


def test_only_explicitly_named_table_models_are_compared_when_present():
    candidates, _, _ = table_candidates([chunk()], task(), [], "Evaluate C8300-2N2S-6T")
    assert [c.label for c in candidates] == ["C8300-2N2S-6T"]


def test_rate_unit_cannot_be_silently_converted_into_port_count():
    _, parsed, _ = table_candidates(
        [chunk()], task(unit="1 Gbps per port"), [], "Compare Cisco routers"
    )
    assert parsed.constraints[0].unit == "1 Gbps per port"


def test_explicit_port_prose_with_tokenized_model_identifiers():
    from consulting_platform.agents.grounding import grounded_candidates

    source = chunk(
        "edge - a1 is a 2ru platform with 2 slots plus 2 x 10gbps and 4 x 1gbps embedded layer3 ethernet ports "
        "edge - b2 is a 2ru platform with 2 slots plus 6 x 1gbps embedded layer3 ethernet ports "
        "edge - c3 is a platform with 8 x 1gbps ethernet ports"
    )
    proven = set()
    candidates, parsed, notes = table_candidates(
        [source], task(), [], "Compare edge-a1 and edge-b2", proven=proven
    )
    candidates, rejected = grounded_candidates(candidates, {source.chunk_id: source})
    ranked = rank_candidates(candidates, parsed.constraints)
    assert not notes and not rejected
    assert [(r.label, r.eligibility) for r in ranked] == [
        ("edge - b2", "eligible"), ("edge - a1", "excluded")
    ]
    assert len(proven) == 2


@pytest.mark.parametrize("qualifier", ["optional ", "up to ", "not "])
def test_conditional_or_negated_prose_is_not_a_verified_port_count(qualifier):
    source = chunk(f"Edge-A1 is a platform with {qualifier}6 x 1gbps ethernet ports")
    candidates, _, _ = table_candidates([source], task(), [], "Compare Edge-A1")
    assert not candidates


def test_prose_does_not_confuse_ten_gigabit_with_one_gigabit_ports():
    candidates, parsed, _ = table_candidates(
        [chunk("Edge-A1 is a platform with 6 x 10gbps ethernet ports")],
        task(), [], "Compare Edge-A1",
    )
    assert rank_candidates(candidates, parsed.constraints)[0].eligibility == "conditional"


def test_conflicting_prose_port_counts_remain_unresolved():
    source = chunk("Edge-A1 is a platform with 6 x 1gbps ethernet ports")
    conflict = source.model_copy(update={"chunk_id": "conflict", "text": "Edge-A1 is a platform with 4 x 1gbps ethernet ports"})
    candidates, parsed, notes = table_candidates([source, conflict], task(), [], "Compare Edge-A1")
    assert notes and rank_candidates(candidates, parsed.constraints)[0].eligibility == "conditional"
