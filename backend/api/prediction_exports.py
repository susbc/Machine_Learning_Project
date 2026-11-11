"""Prediction result export API: server-side file export or database table export."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text, inspect
from sqlalchemy.exc import SQLAlchemyError

from core.paths import ARTIFACT_ROOT

from .database import get_connector
from .models import validate_model_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/prediction-exports", tags=["Prediction Export"])

DEFAULT_EXPORT_DIR = "output"
DEFAULT_EXPORT_TABLE = "prediction_results"
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
MAX_EXPORT_ROWS = 100000

# 各驱动"权限不足"错误码：MySQL 1044/1045/1142/1143/1227/1370、
# SQL Server 229/262、PostgreSQL SQLSTATE 42501/28000。
_MYSQL_PRIVILEGE_CODES = {1044, 1045, 1142, 1143, 1227, 1370}
_MSSQL_PRIVILEGE_CODES = {229, 262}
_PG_PRIVILEGE_CODES = {"42501", "28000"}
_PRIVILEGE_TEXT_MARKERS = (
    "permission denied",
    "privilege",
    "access denied",
    "not authorized",
    "readonly",
    "read-only",
)


class PredictionExportDiagnostics(BaseModel):
    unknown_input_keys: List[str] = Field(default_factory=list)
    missing_columns_imputed: List[str] = Field(default_factory=list)


class PredictionExportRow(BaseModel):
    index: int = Field(ge=0)
    features: Dict[str, Any] = Field(default_factory=dict)
    prediction: Union[str, int, float, bool]
    encoded_prediction: Optional[float] = None
    probability: Optional[float] = None
    diagnostics: Optional[PredictionExportDiagnostics] = None


class PredictionExportRequest(BaseModel):
    model_id: str
    target: Literal["file", "database"]
    rows: List[PredictionExportRow] = Field(min_length=1, max_length=MAX_EXPORT_ROWS)
    output_dir: Optional[str] = Field(default=None, max_length=500)
    connection_id: Optional[str] = Field(default=None, max_length=64)
    table_name: Optional[str] = Field(default=None, max_length=128)


def _resolve_export_dir(raw: Optional[str]) -> Path:
    """导出目录限定在项目产物根目录内，缺省为 ./output。"""
    root = ARTIFACT_ROOT.resolve()
    text = (raw or "").strip()
    candidate = Path(text) if text else Path(DEFAULT_EXPORT_DIR)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="导出目录必须位于项目目录内")
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"导出目录不可用：{exc}") from exc
    return resolved


def _feature_columns(rows: List[PredictionExportRow]) -> List[str]:
    columns: List[str] = []
    for row in rows:
        for key in row.features:
            if key not in columns:
                columns.append(key)
    return columns


def export_rows_to_file(model_id: str, rows: List[PredictionExportRow], directory: Path) -> Path:
    feature_columns = _feature_columns(rows)
    records = [
        {
            "row_index": row.index,
            **{name: row.features.get(name) for name in feature_columns},
            "prediction": row.prediction,
            "encoded_prediction": row.encoded_prediction,
            "probability": row.probability,
        }
        for row in rows
    ]
    frame = pd.DataFrame(
        records,
        columns=["row_index", *feature_columns, "prediction", "encoded_prediction", "probability"],
    )
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", model_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"predictions_{slug}_{stamp}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _privilege_warning(exc: SQLAlchemyError) -> Optional[str]:
    """识别驱动层权限错误，转换为面向用户的警告文案；非权限错误返回 None。"""
    orig = getattr(exc, "orig", None)
    args = list(getattr(orig, "args", ()) or ())
    code = args[0] if args else None
    pgcode = getattr(orig, "pgcode", None) or getattr(orig, "sqlstate", None)
    text = " ".join(str(item) for item in args)
    lowered = f"{text} {exc}".lower()
    denied = (
        pgcode in _PG_PRIVILEGE_CODES
        or code in _MYSQL_PRIVILEGE_CODES
        or code in _MSSQL_PRIVILEGE_CODES
        or any(marker in lowered for marker in _PRIVILEGE_TEXT_MARKERS)
    )
    if not denied:
        return None
    detail = text.strip() or str(exc)
    return (
        f"数据库权限不足：当前连接账号缺少建表或写入权限（{detail}）。"
        "请联系数据库管理员授权，或改用具备权限的连接。"
    )


def _export_table(table_name: str) -> Table:
    schema, _, name = table_name.rpartition(".")
    return Table(
        name,
        MetaData(),
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("exported_at", DateTime, nullable=False),
        Column("model_id", String(255), nullable=False),
        Column("row_index", Integer, nullable=False),
        Column("features_json", Text, nullable=False),
        Column("prediction", String(512), nullable=False),
        Column("encoded_prediction", Float, nullable=True),
        Column("probability", Float, nullable=True),
        Column("diagnostics_json", Text, nullable=True),
        schema=schema or None,
    )


def _insert_params(model_id: str, rows: List[PredictionExportRow], exported_at: datetime) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []
    for row in rows:
        diagnostics = row.diagnostics.dict() if row.diagnostics else None
        params.append(
            {
                "exported_at": exported_at,
                "model_id": model_id,
                "row_index": row.index,
                "features_json": json.dumps(row.features, ensure_ascii=False),
                "prediction": str(row.prediction),
                "encoded_prediction": row.encoded_prediction,
                "probability": row.probability,
                "diagnostics_json": json.dumps(diagnostics, ensure_ascii=False) if diagnostics else None,
            }
        )
    return params


def export_rows_to_database(connector, table_name: str, model_id: str, rows: List[PredictionExportRow]) -> Dict[str, Any]:
    """写入数据库表；表不存在时按默认结构自动创建。权限不足返回警告而非抛错。"""
    if connector.engine is None and not connector.connect():
        raise HTTPException(status_code=400, detail=f"数据库连接失败：{connector.last_error or '请检查连接配置'}")
    engine = connector.engine
    table = _export_table(table_name)
    created = False
    try:
        inspector = inspect(engine)
        if not inspector.has_table(table.name, schema=table.schema):
            table.create(engine, checkfirst=True)
            created = True
        with engine.begin() as connection:
            connection.execute(table.insert(), _insert_params(model_id, rows, datetime.now()))
    except SQLAlchemyError as exc:
        warning = _privilege_warning(exc)
        if warning is None:
            detail = str(getattr(exc, "orig", exc))
            raise HTTPException(status_code=400, detail=f"数据库导出失败：{detail}") from exc
        logger.warning("预测结果导出权限不足 table=%s：%s", table_name, exc)
        return {
            "exported": False,
            "status": "permission_denied",
            "warnings": [warning],
            "table_created": created,
            "row_count": 0,
        }
    return {
        "exported": True,
        "status": "ok",
        "warnings": [],
        "table_created": created,
        "row_count": len(rows),
    }


@router.post("")
async def export_predictions(payload: PredictionExportRequest) -> Dict[str, Any]:
    validate_model_id(payload.model_id)
    if payload.target == "file":
        directory = _resolve_export_dir(payload.output_dir)
        path = export_rows_to_file(payload.model_id, payload.rows, directory)
        return {
            "target": "file",
            "exported": True,
            "status": "ok",
            "warnings": [],
            "row_count": len(payload.rows),
            "path": str(path.relative_to(ARTIFACT_ROOT.resolve())),
            "connection_id": None,
            "table_name": None,
            "table_created": None,
        }

    if not payload.connection_id:
        raise HTTPException(status_code=400, detail="数据库导出需要选择数据库连接")
    table_name = (payload.table_name or "").strip() or DEFAULT_EXPORT_TABLE
    if not TABLE_NAME_PATTERN.fullmatch(table_name):
        raise HTTPException(status_code=400, detail="无效的表名：仅支持字母/数字/下划线，可带 schema 前缀")
    connector = get_connector(payload.connection_id)
    try:
        outcome = export_rows_to_database(connector, table_name, payload.model_id, payload.rows)
    finally:
        connector.disconnect()
    return {
        "target": "database",
        "path": None,
        "connection_id": payload.connection_id,
        "table_name": table_name,
        **outcome,
    }
