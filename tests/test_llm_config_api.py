import json

from fastapi.testclient import TestClient

from backend.main import app


def test_llm_provider_lifecycle_never_returns_plain_api_key(monkeypatch, tmp_path):
    config_path = tmp_path / "llm_providers.json"
    monkeypatch.setenv("LLM_CONFIG_PATH", str(config_path))
    client = TestClient(app)

    create_response = client.post(
        "/api/llm/providers",
        json={
            "name": "Qwen Max",
            "type": "openai",
            "description": "DashScope compatible endpoint",
            "enabled": False,
            "config": {
                "api_key": "sk-test-secret",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-max",
            },
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    provider_id = created["id"]
    assert created["config"]["api_key_masked"] == "sk-...cret"
    assert "sk-test-secret" not in json.dumps(created, ensure_ascii=False)

    list_response = client.get("/api/llm/providers")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["statistics"]["total_providers"] == 1
    assert payload["active_provider"] is None
    assert payload["providers"][0]["id"] == provider_id
    assert "sk-test-secret" not in json.dumps(payload, ensure_ascii=False)

    patch_response = client.patch(
        f"/api/llm/providers/{provider_id}",
        json={
            "name": "Qwen Max Tuned",
            "config": {
                "api_key": "",
                "model": "qwen-max-latest",
            },
        },
    )
    assert patch_response.status_code == 200
    patched = patch_response.json()
    assert patched["name"] == "Qwen Max Tuned"
    assert patched["enabled"] is False
    assert patched["config"]["model"] == "qwen-max-latest"
    assert patched["config"]["api_key_masked"] == "sk-...cret"
    assert "sk-test-secret" not in json.dumps(patched, ensure_ascii=False)

    test_response = client.post(f"/api/llm/providers/{provider_id}/test")
    assert test_response.status_code == 200
    assert test_response.json()["ok"] is False
    assert {"provider_id", "ok", "message"} <= set(test_response.json())
    assert "sk-test-secret" not in json.dumps(test_response.json(), ensure_ascii=False)

    enable_response = client.patch(f"/api/llm/providers/{provider_id}", json={"enabled": True})
    assert enable_response.status_code == 200
    assert enable_response.json()["enabled"] is True

    activate_response = client.post(f"/api/llm/providers/{provider_id}/activate")
    assert activate_response.status_code == 200
    assert activate_response.json()["active_provider"]["id"] == provider_id

    delete_response = client.delete(f"/api/llm/providers/{provider_id}")
    assert delete_response.status_code == 204
    assert client.get("/api/llm/providers").json()["providers"] == []


def test_llm_provider_api_rejects_invalid_url_type_and_id(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_CONFIG_PATH", str(tmp_path / "llm_providers.json"))
    client = TestClient(app)

    invalid_type_response = client.post(
        "/api/llm/providers",
        json={
            "name": "bad",
            "type": "unknown",
            "config": {"api_key": "secret", "model": "m"},
        },
    )
    invalid_url_response = client.post(
        "/api/llm/providers",
        json={
            "name": "bad-url",
            "type": "openai",
            "config": {"api_key": "secret", "base_url": "file:///tmp/key", "model": "m"},
        },
    )
    invalid_id_response = client.patch("/api/llm/providers/..%2Fsecret", json={"name": "x"})

    assert invalid_type_response.status_code == 422
    assert invalid_url_response.status_code == 422
    assert invalid_id_response.status_code == 400


def test_llm_provider_base_url_or_type_change_requires_fresh_api_key(monkeypatch, tmp_path):
    """修改 base_url 或 type 时必须重新提交 api_key，防止存量密钥被重定向外泄。"""
    monkeypatch.setenv("LLM_CONFIG_PATH", str(tmp_path / "llm_providers.json"))
    client = TestClient(app)

    create_response = client.post(
        "/api/llm/providers",
        json={
            "name": "victim",
            "type": "openai",
            "enabled": False,
            "config": {
                "api_key": "sk-victim-secret",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
            },
        },
    )
    assert create_response.status_code == 201
    provider_id = create_response.json()["id"]

    redirect_response = client.patch(
        f"/api/llm/providers/{provider_id}",
        json={"config": {"base_url": "https://attacker.example/v1"}},
    )
    assert redirect_response.status_code == 422

    blank_key_response = client.patch(
        f"/api/llm/providers/{provider_id}",
        json={"config": {"base_url": "https://attacker.example/v1", "api_key": ""}},
    )
    assert blank_key_response.status_code == 422

    type_response = client.patch(
        f"/api/llm/providers/{provider_id}",
        json={"type": "qwen"},
    )
    assert type_response.status_code == 422

    same_url_response = client.patch(
        f"/api/llm/providers/{provider_id}",
        json={"config": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o"}},
    )
    assert same_url_response.status_code == 200
    assert same_url_response.json()["config"]["model"] == "gpt-4o"

    rotated_response = client.patch(
        f"/api/llm/providers/{provider_id}",
        json={"config": {"base_url": "https://attacker.example/v1", "api_key": "sk-rotated-key"}},
    )
    assert rotated_response.status_code == 200
    rotated = rotated_response.json()
    assert rotated["config"]["base_url"] == "https://attacker.example/v1"
    assert "sk-victim-secret" not in json.dumps(rotated, ensure_ascii=False)
    assert "sk-rotated-key" not in json.dumps(rotated, ensure_ascii=False)


def test_llm_provider_create_rejects_duplicate_name_and_type(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_CONFIG_PATH", str(tmp_path / "llm_providers.json"))
    client = TestClient(app)
    payload = {
        "name": "same-provider",
        "type": "openai",
        "enabled": False,
        "config": {
            "api_key": "sk-one",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        },
    }

    first = client.post("/api/llm/providers", json=payload)
    second = client.post("/api/llm/providers", json=payload)

    assert first.status_code == 201
    assert second.status_code == 409
    assert "已存在" in second.json()["detail"]


def test_llm_provider_base_url_rejects_metadata_and_link_local(monkeypatch, tmp_path):
    """base_url 必须封禁云元数据/链路本地地址段（防 SSRF 内网探测），
    同时保留本地与内网 LLM 端点（Ollama/vLLM 等核心用途）。"""
    monkeypatch.setenv("LLM_CONFIG_PATH", str(tmp_path / "llm_providers.json"))
    client = TestClient(app)

    def create(name, base_url):
        return client.post(
            "/api/llm/providers",
            json={
                "name": name,
                "type": "openai",
                "enabled": False,
                "config": {"api_key": "sk-test", "base_url": base_url, "model": "m"},
            },
        )

    # 云元数据服务与链路本地段：经典 SSRF 目标，必须拒绝
    assert create("metadata-v4", "http://169.254.169.254/latest/meta-data").status_code == 422
    assert create("metadata-v6", "http://[fd00:ec2::254]/latest/meta-data").status_code == 422
    assert create("link-local", "http://169.254.10.20/v1").status_code == 422
    # 回环（本地 Ollama）与内网（私有化 LLM 网关）是合法用途，必须保持可用
    assert create("local-ollama", "http://127.0.0.1:11434/v1").status_code == 201
    assert create("private-gateway", "http://10.0.0.5:8000/v1").status_code == 201
