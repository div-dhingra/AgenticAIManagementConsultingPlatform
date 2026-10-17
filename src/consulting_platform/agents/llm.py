"""Small asynchronous provider adapters and per-request generation budgets."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from consulting_platform.config import Settings
from consulting_platform.contracts import (
    ServiceUnavailableError,
    UpstreamModelError,
    UpstreamTimeoutError,
)

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    async def generate_json(
        self,
        *,
        role: str,
        system: str,
        payload: dict[str, Any],
        schema: dict,
    ) -> dict: ...


@dataclass
class RequestBudget:
    limit: int = 12
    calls: int = 0
    repair_used: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def reserve(self, *, repair: bool = False) -> None:
        async with self.lock:
            if self.calls >= self.limit or (repair and self.repair_used):
                raise UpstreamModelError("Model call or JSON repair budget exhausted.")
            self.calls += 1
            if repair:
                self.repair_used = True


async def generate(
    client: LLMClient,
    budget: RequestBudget,
    model: type[T],
    *,
    role: str,
    system: str,
    payload: dict,
) -> T:
    schema = model.model_json_schema()
    for attempt in range(2):
        await budget.reserve(repair=bool(attempt))
        try:
            result = await client.generate_json(
                role=role,
                system=system,
                payload=payload,
                schema=schema,
            )
            return model.model_validate(result)
        except (ValidationError, json.JSONDecodeError, TypeError) as exc:
            if attempt:
                raise UpstreamModelError("Model returned invalid structured output.") from exc
            # No raw generated output or customer/provider secrets in logs/errors.
            payload = {
                **payload,
                "format_correction": "Return valid JSON matching the supplied schema.",
            }
    raise AssertionError("Unreachable")


class HostedLLMClient:
    """One client/limiter per service process. No automatic provider/model fallback."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport
        self._pace_lock = asyncio.Lock()
        self._last_request = 0.0

    async def _pace(self) -> None:
        async with self._pace_lock:
            delay = 60 / self.settings.requests_per_minute - (time.monotonic() - self._last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request = time.monotonic()

    async def generate_json(
        self,
        *,
        role: str,
        system: str,
        payload: dict,
        schema: dict,
    ) -> dict:
        key = self.settings.api_key.get_secret_value()
        if not key:
            raise ServiceUnavailableError("The configured model credential is missing.")
        await self._pace()
        override = {
            "router": self.settings.router_model,
            "synthesis": self.settings.synthesis_model,
        }.get(role, self.settings.specialist_model)
        messages = [
            {
                "role": "system",
                "content": system
                + "\nGenerate JSON only, matching this schema:\n"
                + json.dumps(schema),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        body: dict = {
            "model": override or self.settings.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            # Local Pydantic validation handles references and semantics. Using JSON
            # mode avoids provider-specific unsupported JSON Schema keywords.
            "response_format": {"type": "json_object"},
        }
        if self.settings.provider == "cohere":
            url = "https://api.cohere.com/v2/chat"
            body["stream"] = False
            if "plus" in body["model"] or "reasoning" in body["model"]:
                body["thinking"] = {"type": "disabled"}
        else:
            url = "https://api.mistral.ai/v1/chat/completions"
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.llm_timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                )
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError("The model request timed out.") from exc
        except httpx.RequestError as exc:
            raise ServiceUnavailableError("The model provider could not be reached.") from exc
        if response.status_code in {401, 403, 429, 503}:
            raise ServiceUnavailableError("The model credential, quota, or service is unavailable.")
        if response.status_code >= 400:
            raise UpstreamModelError("The model provider rejected the request.")
        try:
            data = response.json()
            if self.settings.provider == "cohere":
                if data.get("finish_reason") not in {None, "COMPLETE"}:
                    raise UpstreamModelError("Model output was truncated or incomplete.")
                content = "".join(
                    b["text"] for b in data["message"]["content"] if b.get("type") == "text"
                )
            else:
                choice = data["choices"][0]
                if choice.get("finish_reason") not in {None, "stop"}:
                    raise UpstreamModelError("Model output was truncated or incomplete.")
                content = choice["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            if isinstance(exc, json.JSONDecodeError):
                raise
            raise UpstreamModelError("The provider returned an unexpected response.") from exc
