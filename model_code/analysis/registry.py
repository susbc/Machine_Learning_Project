"""Lightweight JSONL registry for analysis artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from .schemas import AnalysisResult, json_default


SENSITIVE_KEYWORDS = ("api_key", "apikey", "password", "passwd", "secret", "token", "credential")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_registry_path() -> Path:
    return project_root() / "model_code" / "analysis_reports" / "analysis_registry.jsonl"


def ensure_within_project(path: Path) -> Path:
    resolved = path.resolve()
    root = project_root().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"路径必须位于项目目录内：{resolved}") from exc
    return resolved


def scrub_sensitive(payload: Any) -> Any:
    """Remove obvious secrets before writing registry records."""
    if isinstance(payload, dict):
        cleaned = {}
        for key, value in payload.items():
            key_str = str(key)
            if any(token in key_str.lower() for token in SENSITIVE_KEYWORDS):
                cleaned[key_str] = "***REDACTED***"
            else:
                cleaned[key_str] = scrub_sensitive(value)
        return cleaned
    if isinstance(payload, list):
        return [scrub_sensitive(item) for item in payload]
    return payload


def record_analysis(
    result: AnalysisResult,
    report_path: Optional[str] = None,
    registry_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    registry_file = ensure_within_project(registry_path or default_registry_path())
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "analysis_id": uuid4().hex[:12],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_type": result.analysis_type,
        "risk_level": result.risk_level,
        "confidence": result.confidence,
        "summary": result.summary,
        "report_path": report_path,
        "metadata": result.metadata,
        "extra": extra or {},
    }
    record = scrub_sensitive(record)

    with open(registry_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")

    return record


def list_recent_records(limit: int = 20, registry_path: Optional[Path] = None) -> list[Dict[str, Any]]:
    registry_file = registry_path or default_registry_path()
    if not registry_file.exists():
        return []

    records = []
    with open(registry_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:][::-1]
