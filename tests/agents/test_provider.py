import json

import httpx
import pytest

from consulting_platform.agents.llm import HostedLLMClient
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
