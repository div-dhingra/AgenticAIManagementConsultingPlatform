import json

import httpx
import pytest

from consulting_platform.agents.llm import (
    HostedLLMClient,
    RequestBudget,
    cohere_schema,
    generate,
    restore_cohere_maps,
)
from consulting_platform.agents.models import RoutePlan, SpecialistAnalysis
from consulting_platform.config import Settings
from consulting_platform.contracts import (
    ServiceUnavailableError,
    UpstreamModelError,
    UpstreamTimeoutError,
)


async def call(client):
    return await client.generate_json(
        role="router", system="Generate JSON", payload={"question": "Hi"}, schema={}
    )


async def test_cohere_payload_and_text_only_response():
    def handler(request):
        body = json.loads(request.content)
        assert request.url == "https://api.cohere.com/v2/chat"
        assert request.headers["Authorization"] == "Bearer test-only"
        assert body["model"] == "command-r7b-12-2024"
        assert body["response_format"] == {"type": "json_object"}
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "finish_reason": "COMPLETE",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "PRIVATE CONTENT"},
                        {"type": "text", "text": '{"tasks":{}}'},
                    ]
                },
            },
        )

    client = HostedLLMClient(
        Settings(api_key="test-only", router_model="command-r7b-12-2024"),
        transport=httpx.MockTransport(handler),
    )
    assert await call(client) == {"tasks": {}}


async def test_mistral_is_an_explicit_alternative():
    def handler(request):
        assert request.url == "https://api.mistral.ai/v1/chat/completions"
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
        )

    assert (
        await call(
            HostedLLMClient(
                Settings(provider="mistral", model="mistral-small-latest", api_key="test-only"),
                transport=httpx.MockTransport(handler),
            )
        )
        == {}
    )


@pytest.mark.parametrize(
    "status,error",
    [
        (401, ServiceUnavailableError),
        (429, ServiceUnavailableError),
        (503, ServiceUnavailableError),
        (400, UpstreamModelError),
        (500, UpstreamModelError),
    ],
)
async def test_provider_errors_are_sanitized(status, error):
    client = HostedLLMClient(
        Settings(api_key="test-only"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json={"secret": "DO NOT LEAK"})
        ),
    )
    with pytest.raises(error) as caught:
        await call(client)
    assert "DO NOT LEAK" not in str(caught.value)


async def test_provider_timeout():
    def handler(request):
        raise httpx.ReadTimeout("raw provider detail")

    with pytest.raises(UpstreamTimeoutError):
        await call(
            HostedLLMClient(Settings(api_key="test-only"), transport=httpx.MockTransport(handler))
        )


async def test_truncation_is_not_treated_as_valid_output():
    with pytest.raises(UpstreamModelError, match="incomplete"):
        await call(
            HostedLLMClient(
                Settings(api_key="test-only"),
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200, json={"finish_reason": "MAX_TOKENS", "message": {"content": []}}
                    )
                ),
            )
        )


@pytest.mark.parametrize("always_truncate,call_limit", [(False, 12), (True, 12), (True, 1)])
async def test_truncation_recovery_is_bounded_and_increases_output_budget(
    always_truncate, call_limit
):
    limits = []

    def handler(request):
        limits.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(
            200,
            json={
                "finish_reason": "max_tokens"
                if always_truncate or len(limits) == 1
                else "COMPLETE",
                # Even complete-looking JSON must be discarded on a truncation stop.
                "message": {"content": [{"type": "text", "text": '{"tasks": {}}'}]},
            },
        )

    client = HostedLLMClient(
        Settings(api_key="test-only", requests_per_minute=1000),
        transport=httpx.MockTransport(handler),
    )
    budget = RequestBudget(limit=call_limit)
    if always_truncate:
        with pytest.raises(UpstreamModelError, match="router.*finish_reason=MAX_TOKENS"):
            await generate(client, budget, RoutePlan, role="router", system="JSON", payload={})
    else:
        result = await generate(client, budget, RoutePlan, role="router", system="JSON", payload={})
        assert result.tasks == {}
    assert limits == ([4000] if call_limit == 1 else [4000, 8000])
    assert budget.calls == len(limits)
    assert budget.repair_used == (call_limit != 1)


@pytest.mark.parametrize(
    "reason,error", [("ERROR", UpstreamModelError), ("TIMEOUT", UpstreamTimeoutError)]
)
async def test_provider_stop_errors_are_not_retried_as_truncation(reason, error):
    client = HostedLLMClient(
        Settings(api_key="test-only"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"finish_reason": reason})
        ),
    )
    budget = RequestBudget()
    with pytest.raises(error):
        await generate(client, budget, RoutePlan, role="router", system="JSON", payload={})
    assert budget.calls == 1 and not budget.repair_used


def test_cohere_wire_schema_uses_unique_nullable_domain_slots():
    schema = cohere_schema(RoutePlan.model_json_schema())
    tasks = schema["properties"]["tasks"]
    assert tasks["type"] == "object"
    assert set(tasks["properties"]) == {"technology", "clinical", "sec", "corporate_responsibility"}
    assert tasks["required"] == list(tasks["properties"])
    assert tasks["properties"]["clinical"]["anyOf"][-1] == {"type": "null"}
    assert "minimum" not in str(schema) and "default" not in str(schema)
    assert "tasks" in schema["required"]


def test_cohere_wire_decode_recurses_through_nested_candidate_attributes():
    value = {
        "findings": [],
        "candidates": [
            {
                "id": "a",
                "label": "A",
                "identity_supports": [{"evidence_id": "e1", "quote": "A has 6 ports."}],
                "attributes": [
                    {
                        "key": "port_count",
                        "value": {
                            "value": 6,
                            "unit": "ports",
                            "supports": [{"evidence_id": "e1", "quote": "A has 6 ports."}],
                        },
                    }
                ],
            }
        ],
        "actions": [],
        "clarification_questions": [],
        "follow_up_query": None,
        "limitations": [],
    }
    decoded = restore_cohere_maps(value, SpecialistAnalysis.model_json_schema())
    result = SpecialistAnalysis.model_validate(decoded)
    assert result.candidates[0].attributes["port_count"].value == 6


def test_cohere_wire_duplicate_map_keys_cannot_silently_overwrite():
    with pytest.raises(TypeError, match="duplicate"):
        restore_cohere_maps(
            {"tasks": [{"key": "sec", "value": {}}, {"key": "sec", "value": {}}]},
            RoutePlan.model_json_schema(),
        )


def test_schema_property_names_are_not_removed_as_annotations():
    schema = cohere_schema(
        {
            "type": "object",
            "properties": {
                "title": {"type": "string", "title": "Display name", "maxLength": 20},
            },
            "required": ["title"],
        }
    )
    assert schema["properties"]["title"] == {"type": "string"}


async def test_cohere_uses_server_schema_and_restores_canonical_map():
    def handler(request):
        body = json.loads(request.content)
        assert body["response_format"]["schema"]["properties"]["tasks"]["type"] == "object"
        return httpx.Response(
            200,
            json={
                "finish_reason": "COMPLETE",
                "message": {
                    "content": [
                        {"type": "text", "text": '{"tasks": [], "clarification_questions": []}'},
                    ]
                },
            },
        )

    client = HostedLLMClient(Settings(api_key="test-only"), transport=httpx.MockTransport(handler))
    result = await client.generate_json(
        role="router", system="JSON", payload={}, schema=RoutePlan.model_json_schema()
    )
    assert result == {"tasks": {}, "clarification_questions": []}


async def test_specialist_wire_schema_requires_span_ids_instead_of_retyped_quotes():
    def handler(request):
        body = json.loads(request.content)
        support = body["response_format"]["schema"]["$defs"]["Support"]
        assert set(support["properties"]) == {"evidence_id", "span_id"}
        assert support["required"] == ["evidence_id", "span_id"]
        assert "candidates" not in body["response_format"]["schema"]["properties"]
        assert "actions" not in body["response_format"]["schema"]["properties"]
        assert body["thinking"] == {"type": "enabled", "token_budget": 256}
        return httpx.Response(
            200,
            json={
                "finish_reason": "COMPLETE",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "findings": [
                                        {
                                            "statement": "A documented fact",
                                            "supports": [{"evidence_id": "e1", "span_id": "s0"}],
                                        }
                                    ]
                                }
                            ),
                        }
                    ]
                },
            },
        )

    client = HostedLLMClient(
        Settings(api_key="test-only", model="command-a-plus-05-2026"),
        transport=httpx.MockTransport(handler),
    )
    analysis = await generate(
        client,
        RequestBudget(),
        SpecialistAnalysis,
        role="sec",
        system="JSON",
        payload={
            "allow_actions": False,
            "chunks": [
                {"chunk_id": "e1", "quote_spans": [{"span_id": "s0", "quote": "Source fact."}]}
            ],
        },
    )
    assert analysis.findings[0].supports[0].span_id == "s0"
    assert analysis.findings[0].supports[0].quote == ""


async def test_invalid_optional_hypothesis_does_not_destroy_supported_facts():
    from consulting_platform.agents.mocks import ScriptedLLMClient

    client = ScriptedLLMClient(
        {
            "sec": [
                {
                    "findings": [
                        {
                            "statement": "Documented risk",
                            "supports": [{"evidence_id": "e1", "quote": "Exact source text"}],
                        },
                        {
                            "statement": "Unsupported hypothetical benefit",
                            "assessment": "inference",
                            "supports": [],
                        },
                    ],
                    "actions": [{"title": "Unsupported proposal", "supports": []}],
                }
            ]
        }
    )
    budget = RequestBudget()
    result = await generate(
        client, budget, SpecialistAnalysis, role="sec", system="JSON", payload={}
    )
    assert [f.statement for f in result.findings] == ["Documented risk"]
    assert result.actions == [] and budget.calls == 1
    assert any("empty evidence supports" in note for note in result.limitations)


async def test_independent_roles_can_repair_without_an_unbounded_retry_loop():
    from consulting_platform.agents.mocks import ScriptedLLMClient

    client = ScriptedLLMClient(
        {"one": [{"wrong": True}, {"tasks": {}}], "two": [{"wrong": True}, {"tasks": {}}]}
    )
    budget = RequestBudget()
    for role in ("one", "two"):
        result = await generate(client, budget, RoutePlan, role=role, system="JSON", payload={})
        assert result.tasks == {}
    assert budget.calls == 4 and budget.repair_count == 2
    for _ in range(2):
        await budget.reserve()
    await budget.reserve(repair=True)
    with pytest.raises(UpstreamModelError, match="budget"):
        await budget.reserve(repair=True)


def test_nullable_router_slots_restore_only_active_domains():
    result = restore_cohere_maps(
        {
            "tasks": {
                "technology": None,
                "clinical": None,
                "sec": {"query": "Cisco 2025 risk factors"},
                "corporate_responsibility": None,
            }
        },
        RoutePlan.model_json_schema(),
    )
    plan = RoutePlan.model_validate(result)
    assert list(plan.tasks) == ["sec"]
    assert plan.tasks["sec"].query == "Cisco 2025 risk factors"
