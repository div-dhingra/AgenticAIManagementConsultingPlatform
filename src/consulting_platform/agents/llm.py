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


class InvalidStructuredOutput(TypeError):
    """A provider's JSON shape cannot be decoded to the canonical model."""

    def __init__(self, message: str, *, path: str = "root", code: str = "invalid_type"):
        super().__init__(message)
        self.path = path
        self.code = code


class OutputTruncatedError(UpstreamModelError):
    """A recoverable token-limit stop; never accept the partial JSON payload."""

    def __init__(self, role: str, token_limit: int):
        self.role = role
        self.token_limit = token_limit
        super().__init__(
            f"Model output was truncated or incomplete for {role} "
            f"(finish_reason=MAX_TOKENS; token_limit={token_limit})."
        )


def parse_json_object(content: str) -> dict:
    """Keep only a complete final JSON object, never model explanatory prefixes.

    Some providers emit surrounding prose/fences even in JSON mode. Accept a
    complete object only when the remaining suffix is empty (or a closing fence).
    Do not repair missing fields, values, punctuation, or incomplete JSON.
    """

    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise InvalidStructuredOutput(
                    "Duplicate JSON object key.", code="duplicate_json_key"
                )
            output[key] = value
        return output

    decoder = json.JSONDecoder(object_pairs_hook=pairs)
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        header, separator, body = text.partition("\n")
        if separator and header.strip().casefold() in {"```", "```json"}:
            text = body[:-3].strip()
    try:
        result = decoder.decode(text)
    except json.JSONDecodeError as original:
        # A malformed JSON document must not be rescued by taking an inner object
        # from it. Prefix extraction is only for a surrounding prose/fence format.
        if text.startswith("{"):
            raise original
        for position, character in enumerate(text):
            if character != "{":
                continue
            try:
                result, end = decoder.raw_decode(text, position)
            except (json.JSONDecodeError, InvalidStructuredOutput):
                continue
            if isinstance(result, dict) and text[end:].strip() in {"", "```"}:
                return result
        raise original
    if not isinstance(result, dict):
        raise InvalidStructuredOutput("Expected a JSON object.", code="json_object_required")
    return result


def cohere_schema(schema: dict) -> dict:
    """Keep Cohere's supported schema subset; enforce all remaining rules locally.

    Preserve actual property/definition names, even if named 'title' or 'minimum'.
    https://docs.cohere.com/v2/docs/structured-outputs#unsupported-schema-features
    """
    unsupported = {
        "title",
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "propertyNames",
        "allOf",
        "oneOf",
        "not",
    }

    def convert(node):
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node
        if node.get("type") == "object" and isinstance(node.get("additionalProperties"), dict):
            # Cohere requires fixed properties and a required field on every
            # object. Encode maps as typed key/value records on the provider wire,
            # then restore dictionaries before validating the canonical model.
            return {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": convert(node.get("propertyNames", {"type": "string"})),
                        "value": convert(node["additionalProperties"]),
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            }
        output = {}
        for key, value in node.items():
            if key in unsupported:
                continue
            if key in {"properties", "$defs", "definitions"}:
                output[key] = {name: convert(child) for name, child in value.items()}
            else:
                output[key] = convert(value)
        if output.get("type") == "object" and output.get("properties"):
            # Include explicit null/empty values for optional fields on the wire.
            # Canonical Pydantic models still own defaults and semantic checks.
            output["required"] = list(output["properties"])
        return output

    return convert(schema)


def restore_cohere_maps(value, schema: dict):
    """Decode only schema-declared map fields, rejecting duplicate/ambiguous keys."""
    definitions = schema.get("$defs", {})

    def restore(item, node, path="root"):
        if "$ref" in node:
            name = node["$ref"].removeprefix("#/$defs/")
            node = definitions[name]
        if node.get("type") == "object" and isinstance(node.get("additionalProperties"), dict):
            if isinstance(item, dict):
                # A provider can return an already-canonical map despite its wire
                # format request. Validate its keys/values normally; never drop data.
                return {
                    key: restore(child, node["additionalProperties"], path)
                    for key, child in item.items()
                }
            if not isinstance(item, list):
                raise InvalidStructuredOutput(
                    "Expected typed key/value records for a provider map.",
                    path=path,
                    code="map_records_required",
                )
            output = {}
            for index, pair in enumerate(item):
                if (
                    not isinstance(pair, dict)
                    or set(pair) != {"key", "value"}
                    or not isinstance(pair["key"], str)
                    or pair["key"] in output
                ):
                    raise InvalidStructuredOutput(
                        "Invalid or duplicate provider map key.",
                        path=f"{path}.{index}",
                        code="invalid_or_duplicate_map_key",
                    )
                output[pair["key"]] = restore(pair["value"], node["additionalProperties"], path)
            return output
        if isinstance(item, dict) and node.get("type") == "object":
            properties = node.get("properties", {})
            return {
                key: restore(child, properties[key], f"{path}.{key}")
                if key in properties
                else child
                for key, child in item.items()
            }
        if isinstance(item, list) and node.get("type") == "array":
            return [
                restore(child, node.get("items", {}), f"{path}.{i}") for i, child in enumerate(item)
            ]
        return item

    return restore(value, schema)


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
            # Some models wrap a valid instance in its schema title. Unwrap
            # exactly this one unambiguous envelope, then validate everything.
            if (
                isinstance(result, dict)
                and set(result) == {model.__name__}
                and isinstance(result[model.__name__], dict)
            ):
                result = result[model.__name__]
            return model.model_validate(result)
        except (ValidationError, json.JSONDecodeError, TypeError, OutputTruncatedError) as exc:
            truncated = isinstance(exc, OutputTruncatedError)
            if isinstance(exc, InvalidStructuredOutput):
                errors = [{"path": exc.path, "type": exc.code}]
            elif isinstance(exc, ValidationError):
                errors = [
                    {
                        "path": ".".join(str(part) for part in error["loc"]) or "root",
                        "type": error["type"],
                    }
                    for error in exc.errors(
                        include_input=False, include_context=False, include_url=False
                    )[:6]
                ]
            else:
                errors = [
                    {
                        "path": "root",
                        "type": "output_truncated"
                        if truncated
                        else "invalid_json"
                        if isinstance(exc, json.JSONDecodeError)
                        else "invalid_type",
                    }
                ]
            if attempt or budget.repair_used or budget.calls >= budget.limit:
                if truncated:
                    raise UpstreamModelError(str(exc)) from exc
                details = "; ".join(f"{error['path']} ({error['type']})" for error in errors)
                raise UpstreamModelError(
                    f"Model returned invalid structured output for {role}: {details}."
                ) from exc
            # No raw generated output or customer/provider secrets in logs/errors.
            payload = {
                **payload,
                "format_correction": {
                    "instruction": "Return a JSON instance matching the supplied schema, not a schema "
                    "or model-name wrapper. Correct these field paths and types. "
                    "Be concise: at most 3 findings, requested attributes only, no explanatory prose. "
                    "For candidate comparisons return actions=[]; the ranker supplies recommendations.",
                    "validation_errors": errors,
                    **({"output_token_limit": min(exc.token_limit * 2, 8000)} if truncated else {}),
                },
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
        output_schema = (
            cohere_schema(schema) if self.settings.provider == "cohere" and schema else schema
        )
        correction = payload.get("format_correction", {})
        retry_limit = correction.get("output_token_limit") if isinstance(correction, dict) else None
        token_limit = self.settings.max_tokens
        if isinstance(retry_limit, int) and not isinstance(retry_limit, bool):
            token_limit = min(8000, max(token_limit, retry_limit))
        messages = [
            {
                "role": "system",
                "content": system + "\nReturn compact JSON only, without reasoning or commentary. "
                "Use at most 3 findings and 3 candidates, only requested attributes, and at most 2 actions. "
                "Generate the JSON INSTANCE only, matching this schema. "
                "Do not wrap it in the schema title/model name. "
                "The top-level instance fields are: "
                + ", ".join(schema.get("properties", {}))
                + ".\nSchema:\n"
                + json.dumps(output_schema),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        body: dict = {
            "model": override or self.settings.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": token_limit,
            # Mistral uses JSON mode; Cohere additionally constrains field shapes
            # server-side below. Full local validation is always authoritative.
            "response_format": {"type": "json_object"},
        }
        if self.settings.provider == "cohere":
            url = "https://api.cohere.com/v2/chat"
            body["stream"] = False
            if schema:
                body["response_format"]["schema"] = output_schema
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
                reason = str(data.get("finish_reason") or "COMPLETE").upper()
                if reason == "MAX_TOKENS":
                    raise OutputTruncatedError(role, token_limit)
                if reason == "TIMEOUT":
                    raise UpstreamTimeoutError(f"Model generation timed out for {role}.")
                if reason != "COMPLETE":
                    reason = (
                        reason if reason in {"STOP_SEQUENCE", "TOOL_CALL", "ERROR"} else "UNKNOWN"
                    )
                    raise UpstreamModelError(
                        f"Model output incomplete for {role} (finish_reason={reason})."
                    )
                content = "".join(
                    b["text"] for b in data["message"]["content"] if b.get("type") == "text"
                )
            else:
                choice = data["choices"][0]
                reason = str(choice.get("finish_reason") or "stop").lower()
                if reason == "length":
                    raise OutputTruncatedError(role, token_limit)
                if reason != "stop":
                    raise UpstreamModelError(f"Model output incomplete for {role}.")
                content = choice["message"]["content"]
            result = parse_json_object(content)
            if self.settings.provider == "cohere" and schema:
                result = restore_cohere_maps(result, schema)
            return result
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            if isinstance(exc, (json.JSONDecodeError, InvalidStructuredOutput)):
                raise
            raise UpstreamModelError("The provider returned an unexpected response.") from exc
