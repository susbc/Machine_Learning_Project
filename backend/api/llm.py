"""LLM provider configuration API.

The API keeps the existing ``config/llm_providers.json`` storage format so the
legacy Streamlit pages and the Vue frontend can coexist during migration. API
keys are write-only: callers can create or replace them, but responses only
include masked metadata.
"""

from __future__ import annotations

import ipaddress
import os
import hashlib
import socket
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field, validator

from model_code.llm.llm_connector import LLMManager

from ._common import is_sensitive_key, sanitize_config, validate_entity_id

router = APIRouter(prefix="/api/llm", tags=["LLM Config"])

ProviderType = Literal["openai", "anthropic", "qwen", "zhipu", "baidu"]

# 云元数据服务地址：SSRF 的经典目标，base_url 绝不允许指向它们
_METADATA_IPS = {
    ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure/阿里云 IPv4 元数据
    ipaddress.ip_address("fd00:ec2::254"),    # AWS IPv6 元数据
}


def _assert_not_dangerous_endpoint(hostname: Optional[str]) -> None:
    """拒绝指向链路本地 / 保留地址段的连接目标（防 SSRF 探测元数据等服务）。

    回环地址与 RFC1918 内网段保持放行：本应用的核心用途之一就是连接
    本地/内网部署的 Ollama、vLLM 等 LLM 服务。
    """
    if not hostname:
        return
    try:
        resolved = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            addr_infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            # 无法解析的主机名反正连不上，放行并由后续连接测试给出失败提示
            return
        resolved = [ipaddress.ip_address(info[4][0]) for info in addr_infos]
    for ip in resolved:
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if (
            ip in _METADATA_IPS
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("base_url 不能指向链路本地/保留地址段（如云元数据服务）")


class ProviderConfigPayload(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    secret_key: Optional[str] = None

    @validator("base_url")
    def validate_base_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return value
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url 必须是 http/https URL")
        # SSRF 防护：禁止指向云元数据/链路本地等危险地址段。
        # test 端点会让服务端向 base_url 发起请求，攻击者可控的
        # 目标地址会成为内网探测跳板。
        _assert_not_dangerous_endpoint(parsed.hostname)
        return value.rstrip("/")

    def to_clean_dict(self, *, drop_blank_secrets: bool = False) -> Dict[str, Any]:
        data = self.dict(exclude_none=True)
        if drop_blank_secrets:
            data = {
                key: value
                for key, value in data.items()
                if not (is_sensitive_key(key) and isinstance(value, str) and value == "")
            }
        return data


class ProviderCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    type: ProviderType
    config: ProviderConfigPayload
    description: str = Field(default="", max_length=500)
    enabled: bool = True


class ProviderUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    type: Optional[ProviderType] = None
    config: Optional[ProviderConfigPayload] = None
    description: Optional[str] = Field(default=None, max_length=500)
    enabled: Optional[bool] = None


def _manager() -> LLMManager:
    return LLMManager(storage_path=os.getenv("LLM_CONFIG_PATH"))


def _stored_secret_fields(config: Dict[str, Any]) -> list:
    """返回已存储且非空的敏感字段名（如 api_key、secret_key）。"""
    return [
        key
        for key, value in config.items()
        if is_sensitive_key(key) and isinstance(value, str) and value
    ]


def _provider_snapshot(raw: Dict[str, Any], active_provider: Optional[str]) -> Dict[str, Any]:
    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "type": raw.get("type"),
        "description": raw.get("description", ""),
        "enabled": bool(raw.get("enabled", True)),
        "created_at": raw.get("created_at"),
        "is_active": raw.get("id") == active_provider,
        "config": sanitize_config(raw.get("config", {})),
    }


def _require_provider(manager: LLMManager, provider_id: str) -> Dict[str, Any]:
    validate_entity_id(provider_id, "provider_id")
    provider = manager.configs.get(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail="未找到大模型配置")
    return provider


def _statistics(manager: LLMManager) -> Dict[str, Any]:
    enabled_count = sum(1 for config in manager.configs.values() if config.get("enabled", True))
    provider_types = sorted({config.get("type") for config in manager.configs.values() if config.get("type")})
    active = manager.configs.get(manager.active_provider or "")
    return {
        "total_providers": len(manager.configs),
        "enabled_providers": enabled_count,
        "active_provider": active.get("name") if active else None,
        "provider_types": provider_types,
    }


@router.get("/providers")
def list_providers() -> Dict[str, Any]:
    manager = _manager()
    providers = [
        _provider_snapshot(provider, manager.active_provider)
        for provider in sorted(manager.configs.values(), key=lambda item: item.get("created_at", ""))
    ]
    active = manager.configs.get(manager.active_provider or "")
    return {
        "providers": providers,
        "active_provider": _provider_snapshot(active, manager.active_provider) if active else None,
        "statistics": _statistics(manager),
    }


@router.post("/providers", status_code=status.HTTP_201_CREATED)
def create_provider(payload: ProviderCreateRequest) -> Dict[str, Any]:
    config = payload.config.to_clean_dict()
    if not config.get("api_key"):
        raise HTTPException(status_code=422, detail="api_key 不能为空")

    manager = _manager()
    provider_id = hashlib.sha256(f"{payload.name.strip()}_{payload.type}".encode()).hexdigest()[:8]
    if provider_id in manager.configs:
        raise HTTPException(status_code=409, detail="同名同类型的大模型配置已存在，请编辑现有 Provider")

    provider_id = manager.add_provider(
        name=payload.name.strip(),
        provider_type=payload.type,
        config=config,
        description=payload.description.strip(),
        enabled=payload.enabled,
    )
    return _provider_snapshot(manager.configs[provider_id], manager.active_provider)


@router.patch("/providers/{provider_id}")
def update_provider(provider_id: str, payload: ProviderUpdateRequest) -> Dict[str, Any]:
    manager = _manager()
    raw = dict(_require_provider(manager, provider_id))

    type_changed = payload.type is not None and payload.type != raw.get("type")
    if payload.name is not None:
        raw["name"] = payload.name.strip()
    if payload.type is not None:
        raw["type"] = payload.type
    if payload.description is not None:
        raw["description"] = payload.description.strip()
    if payload.enabled is not None:
        raw["enabled"] = payload.enabled
    current_config = dict(raw.get("config", {}))
    incoming = payload.config.to_clean_dict(drop_blank_secrets=True) if payload.config is not None else {}
    base_url_changed = "base_url" in incoming and incoming["base_url"] != current_config.get("base_url")
    if base_url_changed or type_changed:
        # 安全约束：密钥是“只写”字段，调用方无法读取存量密钥。
        # 若允许在不重新提交密钥的情况下改 base_url / type，攻击者可将
        # 存量密钥重定向到自己控制的端点并通过 test 连接外泄。
        missing = [key for key in _stored_secret_fields(current_config) if not incoming.get(key)]
        if missing:
            raise HTTPException(
                status_code=422,
                detail="修改 base_url 或类型时必须同时重新提交密钥字段：" + ", ".join(missing),
            )
    if incoming:
        current_config.update(incoming)
        raw["config"] = current_config

    manager.configs[provider_id] = raw
    if not raw.get("enabled", True) and manager.active_provider == provider_id:
        manager.active_provider = None
    if provider_id in manager.providers:
        del manager.providers[provider_id]
    if raw.get("enabled", True):
        manager._init_provider(provider_id, raw)
    if not manager._save_configs():
        raise HTTPException(status_code=500, detail="保存大模型配置失败")

    return _provider_snapshot(raw, manager.active_provider)


@router.delete("/providers/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_provider(provider_id: str) -> Response:
    manager = _manager()
    _require_provider(manager, provider_id)
    manager.delete_provider(provider_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/providers/{provider_id}/activate")
def activate_provider(provider_id: str) -> Dict[str, Any]:
    manager = _manager()
    raw = _require_provider(manager, provider_id)
    if not raw.get("enabled", True):
        raise HTTPException(status_code=409, detail="不能激活已停用的大模型配置")
    if not manager.set_active_provider(provider_id):
        raise HTTPException(status_code=500, detail="激活大模型配置失败")
    raw = manager.configs[provider_id]
    return {
        "active_provider": _provider_snapshot(raw, manager.active_provider),
        "statistics": _statistics(manager),
    }


@router.post("/providers/{provider_id}/test")
def test_provider(provider_id: str) -> Dict[str, Any]:
    manager = _manager()
    raw = _require_provider(manager, provider_id)
    if not raw.get("enabled", True):
        return {"provider_id": provider_id, "ok": False, "message": "配置已停用，无法测试连接"}
    try:
        ok = manager.test_provider(provider_id)
    except Exception:
        ok = False
    return {
        "provider_id": provider_id,
        "ok": bool(ok),
        "message": "连接测试通过" if ok else "连接测试失败，请检查配置或网络",
    }


@router.patch("/providers/{invalid_provider_id:path}")
@router.delete("/providers/{invalid_provider_id:path}")
def reject_invalid_provider_path(invalid_provider_id: str) -> None:
    validate_entity_id(invalid_provider_id, "provider_id")
    raise HTTPException(status_code=404, detail="未找到大模型配置")
