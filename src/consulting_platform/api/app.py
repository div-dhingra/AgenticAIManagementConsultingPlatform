"""FastAPI application factory.

Run locally:  uvicorn consulting_platform.api.app:create_app --factory --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .deps import Services, build_services
from .errors import install_error_handlers, internal_error
from .routes import router
from .settings import ApiSettings

log = logging.getLogger("consulting_platform.api")


def create_app(settings: ApiSettings | None = None, services: Services | None = None) -> FastAPI:
    """Tests pass `services` to inject fakes; otherwise services are built once at startup."""
    settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "services", None) is None:
            app.state.services = build_services(settings)
        for warning in app.state.services.warnings:
            log.warning("startup: %s", warning)
        yield

    app = FastAPI(title="Consulting Platform API", version="1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.services = services
    install_error_handlers(app)
    app.include_router(router)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # anything the typed handlers did not map
            response = internal_error(request, exc)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request_id=%s %s %s -> %s (%.1f ms)",
            request_id, request.method, request.url.path, response.status_code, (time.perf_counter() - start) * 1000,
        )
        return response

    return app
