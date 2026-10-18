import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from consulting_platform.agents.llm import ModelTimeoutError
from consulting_platform.agents.mocks import FakeConsultingService, FakeKnowledgeService
from consulting_platform.api.app import create_app
from consulting_platform.api.deps import Services
from consulting_platform.api.settings import ApiSettings
from consulting_platform.config import Settings
from consulting_platform.contracts import ServiceUnavailableError, UpstreamModelError
from consulting_platform.retail.fixtures import build_demo_service

DEMO = ApiSettings(core=Settings(mode="demo"))
DEMO_FAST_TIMEOUT = ApiSettings(core=Settings(mode="demo", timeout_seconds=0.2))


def client_for(services: Services, settings: ApiSettings = DEMO) -> TestClient:
    return TestClient(create_app(settings, services))


def demo_services(**overrides) -> Services:
    values = dict(consulting=FakeConsultingService(), knowledge=FakeKnowledgeService(), retail=build_demo_service())
    values.update(overrides)
    return Services(**values)


def assert_envelope(response, status: int, code: str) -> dict:
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"schema_version", "request_id", "mode", "error"}
    assert body["error"]["code"] == code
    assert response.headers["X-Request-ID"] == body["request_id"]
    uuid.UUID(body["request_id"])
    return body


@pytest.fixture
def client():
    with client_for(demo_services()) as c:
        yield c


def test_catalog_and_request_id_header(client):
    response = client.get("/catalog")
    assert response.status_code == 200
    body = response.json()
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert body["mode"] == "demo" and body["schema_version"] == "1.0"
    ids = [p["product_id"] for p in body["products"]]
    assert ids == sorted(ids)


def test_recommend_ok_excludes_cart(client):
    body = client.post("/recommend", json={"cart_items": ["FX-PANEL"], "top_k": 5}).json()
    assert body["status"] == "ok"
    assert body["cart_items"] == ["FX-PANEL"]
    assert "FX-PANEL" not in [r["product_id"] for r in body["recommendations"]]
    assert [r["rank"] for r in body["recommendations"]] == list(range(1, len(body["recommendations"]) + 1))
    assert body["warnings"]  # demo disclosure


def test_empty_and_unsupported_carts_are_200(client):
    assert client.post("/recommend", json={}).json()["status"] == "empty_cart"
    body = client.post("/recommend", json={"cart_items": ["FX-NVR", "FX-PANEL"]}).json()
    assert body["status"] == "no_supported_pattern" and body["recommendations"] == []
    assert body["cart_items"] == ["FX-NVR", "FX-PANEL"]


@pytest.mark.parametrize(
    "payload, field",
    [
        ({"cart_items": ["NOPE"]}, "cart_items.0"),
        ({"cart_items": ["FX-PANEL", "FX-PANEL"]}, "cart_items"),
        ({"cart_items": [], "extra": 1}, "extra"),
        ({"cart_items": ["FX-PANEL"], "top_k": 0}, "top_k"),
        ({"cart_items": "FX-PANEL"}, "cart_items"),
    ],
)
def test_recommend_validation_errors(client, payload, field):
    body = assert_envelope(client.post("/recommend", json=payload), 422, "VALIDATION_ERROR")
    assert field in [d["field"] for d in body["error"]["details"]]


def test_malformed_json_is_422_envelope(client):
    response = client.post("/recommend", content=b"{not json", headers={"Content-Type": "application/json"})
    assert_envelope(response, 422, "VALIDATION_ERROR")


def test_not_found_and_method_not_allowed(client):
    assert_envelope(client.get("/missing"), 404, "NOT_FOUND")
    assert_envelope(client.delete("/catalog"), 405, "METHOD_NOT_ALLOWED")


def test_consult_demo_fixture(client):
    response = client.post("/consult", json={"question": "Which fixture analyzer supports at least 12 samples per run?"})
    body = response.json()
    assert response.status_code == 200 and body["status"] == "answered"
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_consult_validation(client):
    assert_envelope(client.post("/consult", json={"question": "   "}), 422, "VALIDATION_ERROR")
    body = assert_envelope(
        client.post("/consult", json={"question": "q", "context": {"report_years": [1800]}}), 422, "VALIDATION_ERROR"
    )
    assert body["error"]["details"][0]["field"] == "context.report_years.0"


class SlowConsulting:
    cancelled = False

    async def run_consultation(self, request, *, request_id):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            SlowConsulting.cancelled = True
            raise


class FailingConsulting:
    def __init__(self, exc):
        self.exc = exc

    async def run_consultation(self, request, *, request_id):
        raise self.exc


class WrongIdConsulting(FakeConsultingService):
    async def run_consultation(self, request, *, request_id):
        return await super().run_consultation(request, request_id=str(uuid.uuid4()))


def test_consult_timeout_maps_to_504_and_cancels():
    with client_for(demo_services(consulting=SlowConsulting()), DEMO_FAST_TIMEOUT) as c:
        assert_envelope(c.post("/consult", json={"question": "q"}), 504, "UPSTREAM_TIMEOUT")
    assert SlowConsulting.cancelled


def test_model_call_timeout_is_distinct_from_overall_deadline():
    with client_for(demo_services(consulting=FailingConsulting(ModelTimeoutError("private details")))) as c:
        response = c.post("/consult", json={"question": "q"})
    body = assert_envelope(response, 504, "UPSTREAM_TIMEOUT")
    assert "per-call time limit" in body["error"]["message"]
    assert "private details" not in response.text


@pytest.mark.parametrize(
    "exc, status, code",
    [
        (ServiceUnavailableError("secret index path /Users/x"), 503, "SERVICE_UNAVAILABLE"),
        (UpstreamModelError("provider body: api key sk-123"), 502, "UPSTREAM_ERROR"),
        (RuntimeError("Traceback secret"), 500, "INTERNAL_ERROR"),
    ],
)
def test_consult_errors_map_without_leaking(exc, status, code):
    with client_for(demo_services(consulting=FailingConsulting(exc))) as c:
        response = c.post("/consult", json={"question": "q"})
    assert_envelope(response, status, code)
    assert "secret" not in response.text and "sk-123" not in response.text


def test_consult_response_contract_is_enforced():
    with client_for(demo_services(consulting=WrongIdConsulting())) as c:
        assert_envelope(c.post("/consult", json={"question": "q"}), 500, "INTERNAL_ERROR")


def test_missing_retail_only_affects_retail_routes():
    with client_for(demo_services(retail=None)) as c:
        assert_envelope(c.get("/catalog"), 503, "SERVICE_UNAVAILABLE")
        assert_envelope(c.post("/recommend", json={}), 503, "SERVICE_UNAVAILABLE")
        assert c.post("/consult", json={"question": "fixture analyzer?"}).status_code == 200
        health = c.get("/health").json()
        assert health["services"]["retail"] == "unavailable" and health["status"] == "degraded"


def test_missing_consulting_only_affects_consult():
    with client_for(demo_services(consulting=None, knowledge=None)) as c:
        assert_envelope(c.post("/consult", json={"question": "q"}), 503, "SERVICE_UNAVAILABLE")
        assert c.post("/recommend", json={"cart_items": ["FX-PANEL"]}).status_code == 200
        health = c.get("/health").json()
        assert health["services"] == {"consulting": "unavailable", "knowledge": "unavailable", "retail": "ready"}
        assert set(health["knowledge"]["domains"]) == {"technology", "clinical", "sec", "corporate_responsibility"}


def test_live_mode_without_agents_never_fakes_answers(tmp_path):
    settings = ApiSettings(core=Settings(mode="live", artifact_dir=tmp_path))  # no retail artifact either
    with TestClient(create_app(settings)) as c:
        body = assert_envelope(c.post("/consult", json={"question": "q"}), 503, "SERVICE_UNAVAILABLE")
        assert body["mode"] == "live"
        assert_envelope(c.get("/catalog"), 503, "SERVICE_UNAVAILABLE")
        health = c.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "degraded"
