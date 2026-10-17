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
