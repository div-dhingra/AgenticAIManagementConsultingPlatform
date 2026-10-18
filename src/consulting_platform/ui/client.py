"""HTTP client for the UI. The UI never imports agents or the retail model directly."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
CONSULT_TIMEOUT_SECONDS = 100
DEFAULT_TIMEOUT_SECONDS = 10


@dataclass
class ApiResult:
    ok: bool
    status_code: int | None
    data: dict[str, Any] | None
    error_message: str | None = None
    request_id: str | None = None

    @property
    def error_details(self) -> list[dict[str, str]]:
        if self.data and isinstance(self.data.get("error"), dict):
            return self.data["error"].get("details", [])
        return []


def _call(method: str, path: str, payload: dict | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> ApiResult:
    try:
        response = httpx.request(method, f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
    except httpx.TimeoutException:
        return ApiResult(False, None, None, "The API did not respond in time.")
    except httpx.TransportError:
        return ApiResult(False, None, None, f"Cannot reach the API at {API_BASE_URL}. Is it running?")

    request_id = response.headers.get("X-Request-ID")
    try:
        data = response.json()
    except ValueError:
        return ApiResult(False, response.status_code, None, "The API returned a non-JSON response.", request_id)
    if response.is_success:
        return ApiResult(True, response.status_code, data, None, request_id)
    message = data.get("error", {}).get("message", "Request failed.") if isinstance(data, dict) else "Request failed."
    return ApiResult(False, response.status_code, data, message, request_id)


def get_health() -> ApiResult:
    return _call("GET", "/health")


def get_catalog() -> ApiResult:
    return _call("GET", "/catalog")


def recommend(cart_items: list[str], top_k: int = 5) -> ApiResult:
    return _call("POST", "/recommend", {"cart_items": cart_items, "top_k": top_k})


def consult(question: str, context: dict[str, Any], max_recommendations: int) -> ApiResult:
    payload = {"question": question, "context": context, "max_recommendations": max_recommendations}
    return _call("POST", "/consult", payload, timeout=CONSULT_TIMEOUT_SECONDS)
