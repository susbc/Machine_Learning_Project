"""API 层共享工具：实体 ID 校验、敏感字段掩码等。"""

from __future__ import annotations

import re
from typing import Any, Dict, Pattern

from fastapi import HTTPException

DEFAULT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SENSITIVE_TOKENS = ("api_key", "apikey", "password", "passwd", "secret", "token", "credential")


def validate_entity_id(value: str, name: str, pattern: Pattern[str] = DEFAULT_ID_PATTERN) -> str:
    """校验路径参数中的实体 ID，不合法时抛出 400。"""
    if not pattern.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"无效的 {name}")
    return value


def is_sensitive_key(key: str) -> bool:
    """判断配置键名是否属于敏感字段（密钥、口令、令牌等）。"""
    lowered = key.lower().replace("-", "_")
    return any(token in lowered for token in SENSITIVE_TOKENS)


def mask_secret(value: str) -> str:
    """掩码敏感值：短值全掩码，长值只保留首尾少量字符。"""
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}...{value[-4:]}"


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """移除配置中的敏感字段，替换为 ``<key>_masked`` / ``has_<key>`` 标记。"""
    sanitized: Dict[str, Any] = {}
    for key, value in config.items():
        if is_sensitive_key(key):
            sanitized[f"{key}_masked"] = mask_secret(str(value))
            sanitized[f"has_{key}"] = bool(value)
        else:
            sanitized[key] = value
    return sanitized
