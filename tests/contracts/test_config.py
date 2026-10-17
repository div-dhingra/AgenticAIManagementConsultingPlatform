from consulting_platform.config import Settings


def test_env_file_and_process_override_without_leaking_key(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "LLM_PROVIDER=cohere\nCOHERE_MODEL=command-a-03-2025\nCOHERE_API_KEY=local-test-secret\n"
    )
    monkeypatch.setenv("COHERE_MODEL", "command-a-plus-05-2026")
    settings = Settings.from_env(path)
    assert settings.model == "command-a-plus-05-2026"
    assert settings.api_key.get_secret_value() == "local-test-secret"
    assert "local-test-secret" not in repr(settings)
