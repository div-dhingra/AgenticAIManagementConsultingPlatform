import pytest

from consulting_platform.agents.llm import (
    InvalidStructuredOutput,
    RequestBudget,
    generate,
    parse_json_object,
    restore_cohere_maps,
)
from consulting_platform.agents.mocks import ScriptedLLMClient
from consulting_platform.agents.models import RoutePlan
from consulting_platform.contracts import UpstreamModelError


async def test_specialist_wire_constrains_evidence_to_supplied_chunk_ids():
    from consulting_platform.agents.models import SpecialistAnalysis

    class Recorder:
        schema = None

        async def generate_json(self, **kwargs):
            self.schema = kwargs["schema"]
            return {"findings": []}

    client = Recorder()
    await generate(
        client,
        RequestBudget(),
        SpecialistAnalysis,
        role="sec",
        system="JSON",
        payload={
            "chunks": [
                {
                    "chunk_id": "original-chunk-17",
                    "document_id": "not-a-chunk-id",
                    "quote_spans": [],
                }
            ]
        },
    )
    support = client.schema["$defs"]["Support"]["properties"]
    assert support["evidence_id"]["enum"] == ["original-chunk-17"]
    assert "quote" not in support


async def test_known_model_wrapper_is_normalized_without_retry():
    client = ScriptedLLMClient({"router": [{"RoutePlan": {"tasks": {}}}]})
    budget = RequestBudget()
    result = await generate(client, budget, RoutePlan, role="router", system="JSON", payload={})
    assert result.tasks == {} and budget.calls == 1


async def test_wrapper_does_not_hide_other_unknown_fields():
    client = ScriptedLLMClient(
        {
            "router": [
                {"RoutePlan": {"tasks": {}}, "unexpected": "not accepted"},
                {"tasks": {}},
            ]
        }
    )
    budget = RequestBudget()
    await generate(client, budget, RoutePlan, role="router", system="JSON", payload={})
    assert budget.calls == 2 and budget.repair_used


async def test_repair_receives_field_paths_and_types_without_input_values():
    class Recorder:
        def __init__(self):
            self.payloads = []

        async def generate_json(self, **kwargs):
            self.payloads.append(kwargs["payload"])
            return {"unexpected": "PRIVATE_VALUE"} if len(self.payloads) == 1 else {"tasks": {}}

    client = Recorder()
    await generate(client, RequestBudget(), RoutePlan, role="router", system="JSON", payload={})
    correction = client.payloads[1]["format_correction"]
    assert {"path": "tasks", "type": "missing"} in correction["validation_errors"]
    assert "PRIVATE_VALUE" not in str(correction)


async def test_final_error_identifies_role_and_bad_fields_not_values():
    client = ScriptedLLMClient({"router": [{"unexpected": "PRIVATE_VALUE"}] * 2})
    with pytest.raises(UpstreamModelError) as error:
        await generate(client, RequestBudget(), RoutePlan, role="router", system="JSON", payload={})
    assert "router" in str(error.value) and "tasks (missing)" in str(error.value)
    assert "PRIVATE_VALUE" not in str(error.value)


def test_explanatory_prefix_is_discarded_and_final_object_is_parsed():
    assert parse_json_object('Explanatory text to discard.\n{"tasks": {}}') == {"tasks": {}}
    assert parse_json_object('```json\n{"tasks": {}}\n```') == {"tasks": {}}


def test_malformed_root_cannot_be_rescued_with_an_inner_object():
    import json

    with pytest.raises(json.JSONDecodeError):
        parse_json_object('{"incomplete_wrapper": {"tasks": {}}')
    with pytest.raises(json.JSONDecodeError):
        parse_json_object('```json\n{"incomplete_wrapper": {"tasks": {}}\n```')


def test_duplicate_json_fields_are_not_silently_overwritten():
    with pytest.raises(TypeError, match="Duplicate"):
        parse_json_object('{"tasks": {}, "tasks": {}}')


async def test_adapter_failure_reports_safe_path_when_request_repair_cap_is_exhausted():
    class BadMapClient:
        async def generate_json(self, **kwargs):
            return restore_cohere_maps(
                {
                    "tasks": [
                        {"key": "PRIVATE_VALUE", "value": {}},
                        {"key": "PRIVATE_VALUE", "value": {}},
                    ]
                },
                RoutePlan.model_json_schema(),
            )

    budget = RequestBudget(calls=2, repair_used=True, repair_count=3)
    with pytest.raises(
        UpstreamModelError, match=r"router: root.tasks.1 .*duplicate_map_key"
    ) as error:
        await generate(BadMapClient(), budget, RoutePlan, role="router", system="JSON", payload={})
    assert "PRIVATE_VALUE" not in str(error.value)
    assert budget.calls == 3


@pytest.mark.parametrize(
    "text,code",
    [("[]", "json_object_required"), ('{"tasks": {}, "tasks": {}}', "duplicate_json_key")],
)
def test_json_shape_errors_have_specific_safe_categories(text, code):
    with pytest.raises(InvalidStructuredOutput) as error:
        parse_json_object(text)
    assert error.value.code == code
