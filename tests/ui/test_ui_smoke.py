"""Smoke test: the Streamlit app renders when the API is unreachable (no live calls)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src" / "consulting_platform" / "ui" / "app.py"


def test_app_renders_with_api_down(monkeypatch):
    monkeypatch.setattr("consulting_platform.ui.client.API_BASE_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    errors = " ".join(e.value for e in app.error)
    assert "Cannot reach the API" in errors


def test_live_presets_clear_stale_filters_and_cover_all_specialists(monkeypatch):
    from fastapi.testclient import TestClient

    from consulting_platform.api.app import create_app
    from consulting_platform.api.settings import ApiSettings
    from consulting_platform.config import Settings
    from consulting_platform.ui import client

    with TestClient(create_app(ApiSettings(core=Settings(mode="demo")))) as api:
        data = api.get("/health").json()
    data["mode"] = "live"
    monkeypatch.setattr(client, "get_health", lambda: client.ApiResult(True, 200, data))
    monkeypatch.setattr(client, "get_catalog", lambda: client.ApiResult(False, None, None, "unavailable"))
    ui = AppTest.from_file(str(APP), default_timeout=30).run()
    next(b for b in ui.button if b.label == "Cisco risks (2025)").click().run()
    assert ui.session_state.ctx_company == "Cisco" and ui.session_state.ctx_years == "2025"
    next(b for b in ui.button if b.label == "Agilent laboratory options").click().run()
    assert ui.session_state.ctx_company == "Agilent" and ui.session_state.ctx_years == ""
    next(b for b in ui.button if b.label == "Cross-domain example").click().run()
    assert ui.session_state.ctx_company == "" and ui.session_state.ctx_model == ""
    assert ui.session_state.ctx_years == "2024, 2025"
    assert "At least 5" in ui.session_state.ctx_requirements
    assert not ui.exception


def test_lost_backend_connection_refreshes_cached_ready_status(monkeypatch):
    from fastapi.testclient import TestClient

    from consulting_platform.api.app import create_app
    from consulting_platform.api.settings import ApiSettings
    from consulting_platform.config import Settings
    from consulting_platform.ui import client

    with TestClient(create_app(ApiSettings(core=Settings(mode="demo")))) as api:
        healthy = client.ApiResult(True, 200, api.get("/health").json())
    unreachable = client.ApiResult(False, None, None, "Cannot reach the API")
    responses = iter([healthy, unreachable])
    monkeypatch.setattr(client, "get_health", lambda: next(responses))
    monkeypatch.setattr(client, "get_catalog", lambda: unreachable)
    monkeypatch.setattr(client, "consult", lambda *args: unreachable)

    ui = AppTest.from_file(str(APP), default_timeout=30).run()
    assert ui.session_state.health_result.ok
    next(b for b in ui.button if b.label == "Cross-domain example").click().run()
    next(b for b in ui.button if b.label == "Ask").click().run()
    assert not ui.exception
    assert not ui.session_state.health_result.ok
    assert not any("Overall:" in item.value for item in ui.markdown)
    assert any("Cannot reach the API" in item.value for item in ui.error)


def test_controlled_live_example_reaches_api_and_renders_cited_answer(monkeypatch):
    from urllib.parse import urlsplit

    from fastapi.testclient import TestClient

    from consulting_platform.agents.controlled_evidence import ControlledEvidenceService
    from consulting_platform.agents.llm import HostedLLMClient
    from consulting_platform.agents.mocks import ScriptedLLMClient
    from consulting_platform.api.app import create_app
    from consulting_platform.api.settings import ApiSettings
    from consulting_platform.config import Settings
    from consulting_platform.ui import client

    path = APP.parents[3] / "fixtures/consulting/controlled_real_passages.json"
    knowledge = ControlledEvidenceService.from_file(path)
    chunk = next(c for c in knowledge.evidence.chunks if c.domain == "sec")
    quote = chunk.text.split("\n")[0]
    model = ScriptedLLMClient(
        {
            "sec": [
                {
                    "findings": [
                        {
                            "statement": quote,
                            "supports": [{"evidence_id": chunk.chunk_id, "quote": quote}],
                        }
                    ]
                }
            ]
        }
    )
    monkeypatch.setattr(
        HostedLLMClient, "generate_json", lambda self, **kw: model.generate_json(**kw)
    )
    settings = ApiSettings(
        core=Settings(mode="live", api_key="test-only"), controlled_evidence_file=path
    )
    with TestClient(create_app(settings)) as api:

        def request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            return api.request(method, urlsplit(url).path, **kwargs)

        monkeypatch.setattr(client.httpx, "request", request)
        ui = AppTest.from_file(str(APP), default_timeout=30).run()
        assert not ui.exception
        assert any("Live test on selected real passages" in w.value for w in ui.warning)
        next(b for b in ui.button if b.label == "Cisco risks (2025)").click().run()
        next(b for b in ui.button if b.label == "Ask").click().run()
        assert not ui.exception
        result = ui.session_state.consult_result
        assert result.ok and result.data["mode"] == "live"
        assert result.data["citations"][0]["evidence_id"] == chunk.chunk_id
        next(b for b in ui.button if b.label == "SOFI financing (2024)").click().run()
        assert ui.session_state.ctx_company == ""
        assert ui.session_state.ctx_years == "2024"
