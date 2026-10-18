"""Map every failure to the shared ErrorResponse envelope."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..agents.llm import ModelTimeoutError
from ..contracts import (
    ErrorBody,
    ErrorDetail,
    ErrorResponse,
    ServiceUnavailableError,
    UpstreamModelError,
    UpstreamTimeoutError,
)
from ..retail.service import UnknownProductError
from .deps import get_request_id

log = logging.getLogger("consulting_platform.api")

_LOCATION_ROOTS = {"body", "query", "path", "header", "cookie"}


class PublicServiceUnavailable(ServiceUnavailableError):
    """Raised by our routes; its message is safe to show to clients."""


class ContractViolation(Exception):
    """A service returned a result that does not satisfy the API contract."""


def error_response(request: Request, status: int, code: str, message: str, details: list[ErrorDetail] | None = None, headers: dict | None = None) -> JSONResponse:
    request_id = get_request_id(request)
    body = ErrorResponse(
        request_id=request_id,
        mode=request.app.state.settings.mode,
        error=ErrorBody(code=code, message=message, details=details or []),
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=status, headers={**(headers or {}), "X-Request-ID": request_id})


def _field(loc: tuple) -> str:
    parts = [str(p) for p in loc]
    if parts and parts[0] in _LOCATION_ROOTS and len(parts) > 1:
        parts = parts[1:]
    return ".".join(parts)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [ErrorDetail(field=_field(tuple(e.get("loc", ()))), message=e.get("msg", "Invalid value.")) for e in exc.errors()]
        return error_response(request, 422, "VALIDATION_ERROR", "Invalid request.", details)

    @app.exception_handler(UnknownProductError)
    async def _unknown_product(request: Request, exc: UnknownProductError) -> JSONResponse:
        details = [ErrorDetail(field=f, message=m) for f, m in exc.details]
        return error_response(request, 422, "VALIDATION_ERROR", "Invalid request.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            return error_response(request, 404, "NOT_FOUND", "Resource not found.")
        if exc.status_code == 405:
            return error_response(request, 405, "METHOD_NOT_ALLOWED", "Method not allowed.", headers=exc.headers)
        if 400 <= exc.status_code < 500:
            return error_response(request, exc.status_code, "VALIDATION_ERROR", "Invalid request.")
        return error_response(request, 500, "INTERNAL_ERROR", "Internal server error.")

    @app.exception_handler(ServiceUnavailableError)
    async def _unavailable(request: Request, exc: ServiceUnavailableError) -> JSONResponse:
        message = str(exc) if isinstance(exc, PublicServiceUnavailable) else "A required service is unavailable."
        log.warning("request_id=%s service unavailable: %s", get_request_id(request), exc)
        return error_response(request, 503, "SERVICE_UNAVAILABLE", message)

    @app.exception_handler(UpstreamTimeoutError)
    async def _timeout(request: Request, exc: UpstreamTimeoutError) -> JSONResponse:
        provider_timeout = isinstance(exc, ModelTimeoutError)
        log.warning("request_id=%s timeout_kind=%s", get_request_id(request), "model_call" if provider_timeout else "consultation")
        message = (
            "The model provider did not respond within the per-call time limit. Please try again."
            if provider_timeout else
            "The consulting workflow did not finish within the overall deadline. Please try again or ask a narrower question."
        )
        return error_response(request, 504, "UPSTREAM_TIMEOUT", message)

    @app.exception_handler(UpstreamModelError)
    async def _upstream(request: Request, exc: UpstreamModelError) -> JSONResponse:
        log.warning("request_id=%s upstream model error: %r", get_request_id(request), exc)
        return error_response(request, 502, "UPSTREAM_ERROR", "The hosted model provider returned an error.")


def internal_error(request: Request, exc: BaseException) -> JSONResponse:
    log.error("request_id=%s unhandled error", get_request_id(request), exc_info=exc)
    return error_response(request, 500, "INTERNAL_ERROR", "Internal server error.")
