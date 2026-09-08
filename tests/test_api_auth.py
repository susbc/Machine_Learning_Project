import asyncio

import httpx
from fastapi.testclient import TestClient

from backend.main import app


def test_api_key_auth_enforced_when_configured(monkeypatch, tmp_path):
    """配置 API_KEY 后，/api/* 必须携带匹配的密钥才能访问。"""
    monkeypatch.setenv("API_KEY", "topsecret")
    monkeypatch.setenv("LLM_CONFIG_PATH", str(tmp_path / "llm_providers.json"))
    client = TestClient(app)

    missing_header = client.get("/api/llm/providers")
    assert missing_header.status_code == 401

    wrong_key = client.get("/api/llm/providers", headers={"X-API-Key": "wrong"})
    assert wrong_key.status_code == 401

    valid_key = client.get("/api/llm/providers", headers={"X-API-Key": "topsecret"})
    assert valid_key.status_code == 200

    bearer_key = client.get("/api/llm/providers", headers={"Authorization": "Bearer topsecret"})
    assert bearer_key.status_code == 200

    # 非 /api 路径（健康检查、前端页面）不受影响
    assert client.get("/health").status_code == 200


def test_api_key_auth_disabled_by_default(monkeypatch, tmp_path):
    """未配置 API_KEY 时保持本地开发体验，不做鉴权。"""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("LLM_CONFIG_PATH", str(tmp_path / "llm_providers.json"))
    client = TestClient(app)

    assert client.get("/api/llm/providers").status_code == 200


def test_api_rejects_remote_client_when_api_key_not_set(monkeypatch, tmp_path):
    """未配置 API_KEY 时必须拒绝非回环客户端访问 /api/*，
    避免服务被绑定到 0.0.0.0 后管理端点被匿名访问（安全审查共性暴露面）。"""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("ALLOW_NETWORK_API_WITHOUT_AUTH", raising=False)
    monkeypatch.setenv("LLM_CONFIG_PATH", str(tmp_path / "llm_providers.json"))

    async def call(client_host):
        transport = httpx.ASGITransport(app=app, client=(client_host, 54321))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
            return await http.get("/api/llm/providers")

    # 远程客户端：未鉴权暴露面必须被拒绝
    assert asyncio.run(call("203.0.113.5")).status_code == 403
    # 本机回环：保持本地开发体验
    assert asyncio.run(call("127.0.0.1")).status_code == 200

    # 显式放开开关（反向代理等场景）
    monkeypatch.setenv("ALLOW_NETWORK_API_WITHOUT_AUTH", "1")
    assert asyncio.run(call("203.0.113.5")).status_code == 200
