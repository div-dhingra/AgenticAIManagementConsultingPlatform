from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from consulting_platform.agents.controlled_evidence import NOTICE
from consulting_platform.agents.llm import HostedLLMClient
from consulting_platform.agents.mocks import ScriptedLLMClient
from consulting_platform.api.app import create_app
from consulting_platform.api.settings import ApiSettings
from consulting_platform.config import Settings

EVIDENCE = Path("fixtures/consulting/controlled_real_passages.json").resolve()


def test_api_controlled_source_uses_live_graph_and_keeps_retail_independent(monkeypatch):
    from consulting_platform.agents.controlled_evidence import ControlledEvidenceService

    knowledge = ControlledEvidenceService.from_file(EVIDENCE)
    chunk = next(c for c in knowledge.evidence.chunks if c.domain == "sec")
    quote = chunk.text.split("\n")[0]
    scripted = ScriptedLLMClient(
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
        HostedLLMClient, "generate_json", lambda self, **kw: scripted.generate_json(**kw)
    )
    monkeypatch.setattr(
        "consulting_platform.api.deps._load_knowledge",
        lambda: pytest.fail("Must not load full KB when explicitly using controlled evidence"),
    )
    settings = ApiSettings(
        core=Settings(mode="live", api_key="test-only"), controlled_evidence_file=EVIDENCE
    )
    with TestClient(create_app(settings)) as api:
        health = api.get("/health").json()
        assert health["mode"] == "live" and NOTICE in health["warnings"]
        assert health["services"]["knowledge"] == "partial"
        response = api.post(
            "/consult", json={"question": "What risks does Cisco disclose in its 2025 filing?"}
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["mode"] == "live" and result["status"] == "answered"
        assert quote in result["findings"][0]["statement"]
        assert result["citations"][0]["evidence_id"] == chunk.chunk_id
        assert NOTICE in result["warnings"]


def test_bad_controlled_file_does_not_fall_back_to_full_kb(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "consulting_platform.api.deps._load_knowledge", lambda: pytest.fail("No fallback allowed")
    )
    settings = ApiSettings(
        core=Settings(mode="live", api_key="test-only"),
        controlled_evidence_file=tmp_path / "missing.json",
    )
    with TestClient(create_app(settings)) as api:
        assert api.post("/consult", json={"question": "Cisco risks?"}).status_code == 503
        assert api.get("/health").json()["services"]["knowledge"] == "unavailable"


def test_demo_rejects_live_controlled_source():
    with pytest.raises(ValueError, match="requires APP_MODE=live"):
        ApiSettings(core=Settings(mode="demo"), controlled_evidence_file=EVIDENCE)


def test_controlled_path_env_override_and_empty_value(tmp_path, monkeypatch):
    monkeypatch.setattr("consulting_platform.api.settings.REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text("APP_MODE=live\nCONSULT_EVIDENCE_FILE=from-file.json\n")
    monkeypatch.setenv("APP_MODE", "live")
    monkeypatch.setenv("CONSULT_EVIDENCE_FILE", "selected.json")
    assert ApiSettings.from_env().controlled_evidence_file == tmp_path / "selected.json"
    monkeypatch.setenv("CONSULT_EVIDENCE_FILE", "")
    assert ApiSettings.from_env().controlled_evidence_file is None
    monkeypatch.delenv("CONSULT_EVIDENCE_FILE")
    assert ApiSettings.from_env().controlled_evidence_file == tmp_path / "from-file.json"


def test_missing_knowledge_dependency_has_install_instruction(monkeypatch):
    def missing_dependency():
        raise ModuleNotFoundError("missing optional dependency", name="sentence_transformers")

    monkeypatch.setattr("consulting_platform.api.deps._load_knowledge", missing_dependency)
    with TestClient(create_app(ApiSettings(core=Settings(mode="live", api_key="test")))) as api:
        result = api.get("/health").json()
        assert any(".[knowledge]" in warning for warning in result["warnings"])
        assert "Knowledge service is not installed yet." not in result["warnings"]
        assert result["services"]["consulting"] == "unavailable"


def test_missing_index_disables_consulting(monkeypatch):
    from consulting_platform.agents.mocks import FakeKnowledgeService

    knowledge = FakeKnowledgeService()
    status = knowledge.get_status()
    for domain in status.domains.values():
        domain.status = "unavailable"
    monkeypatch.setattr(knowledge, "get_status", lambda: status)
    monkeypatch.setattr("consulting_platform.api.deps._load_knowledge", lambda: knowledge)
    with TestClient(create_app(ApiSettings(core=Settings(mode="live", api_key="test")))) as api:
        result = api.get("/health").json()
        assert result["services"]["consulting"] == "unavailable"
        assert any("build_index.py" in warning for warning in result["warnings"])
