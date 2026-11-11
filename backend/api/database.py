"""Database connection management API.

This keeps the legacy ``config/saved_connections.json`` shape while adding the
security semantics required by the TypeScript frontend: masked credentials in
responses, encrypted credentials for newly written configs, and read-only query
preview with server-side row limits.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, Field, validator
from sqlalchemy import inspect as sqlalchemy_inspect

from model_code.database import DatabaseConnector
from model_code.database.connection_manager import ConnectionManager

from ._common import mask_secret, validate_entity_id

router = APIRouter(prefix="/api/database", tags=["Database Config"])

# 表名白名单：标识符（可带 schema 前缀），与连接器 _quote_identifier 的语义一致。
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
TABLE_ROWS_LIMIT_MAX = 10000

WRITE_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|merge|grant|revoke|call|execute|vacuum|attach|detach)\b",
    re.IGNORECASE,
)
# SELECT 内部即可调用的副作用原语（文件读写、外部连接、命令执行代理）。
# 关键词黑名单无法覆盖这类"读起来像查询"的函数，必须单独枚举并在
# _is_readonly_sql 中拦截。注意这只是纵深防御：真正的只读语义应由
# 只读数据库角色保证（见安全审查报告）。
DANGEROUS_SQL_PATTERN = re.compile(
    r"\b(load_file|lo_import|lo_export|lo_get|lo_put|lo_creat|lo_unlink"
    r"|pg_read_file|pg_read_binary_file|pg_write_file|pg_ls_dir|pg_stat_file"
    r"|dblink|dblink_connect|dblink_exec|openrowset|opendatasource"
    r"|xp_cmdshell|sp_oacreate|sp_oamethod|copy|exec)\b"
    r"|\binto\s+(outfile|dumpfile)\b",
    re.IGNORECASE,
)
# 禁止出现的数据库驱动连接参数。这些参数能把"连接数据库"变成
# "让恶意服务器读取本机任意文件"或"加载任意驱动/执行任意 SQL"：
#   local_infile      —— pymysql 启用 LOAD DATA LOCAL，恶意 MySQL 服务器
#                        可在应答任意查询时索取客户端本机文件（已实证可利用）
#   read_default_file —— pymysql 读取任意路径的 MySQL 选项文件
#   client_flag       —— 可重新打开 CLIENT_LOCAL_FILES 能力位
#   init_command      —— pymysql 连接建立后自动执行任意 SQL
#   dsn/odbc_connect  —— pyodbc 任意 DSN / 完整连接串（可夹带任意 Driver/FILEDSN）
FORBIDDEN_EXTRA_PARAM_KEYS = {
    "local_infile",
    "read_default_file",
    "read_default_group",
    "client_flag",
    "init_command",
    "dsn",
    "odbc_connect",
}
SECRET_PREFIX = "enc:v1:"
DbType = Literal["sqlite", "mysql", "postgresql", "sqlserver"]


class DatabaseConfigPayload(BaseModel):
    db_type: DbType
    host: Optional[str] = Field(default=None, max_length=255)
    port: Optional[int] = Field(default=None, ge=1, le=65535)
    database: str = Field(min_length=1, max_length=1000)
    user: Optional[str] = Field(default=None, max_length=255)
    password: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None

    @validator("host")
    def validate_host(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return value
        lowered = value.lower()
        if "://" in lowered or "/" in lowered or "\\" in lowered:
            raise ValueError("host 不能包含 URL scheme 或路径")
        return value

    @validator("database")
    def validate_database(cls, value: str, values: Dict[str, Any]) -> str:
        # '?'/'#' 会向连接串注入 URL 查询段（如 "appdb?local_infile=1"），
        # 绕过 extra_params 的校验；'@'/'/'/'\\' 可篡改凭证/主机段。
        if "?" in value or "#" in value:
            raise ValueError("database 不能包含 URL 查询字符 '?' 或 '#'")
        if values.get("db_type") != "sqlite" and re.search(r"[@/\\]", value):
            raise ValueError("非 SQLite 的 database 名称不能包含 '@'、'/' 或 '\\'")
        return value

    @validator("extra_params")
    def validate_extra_params(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is None:
            return value
        for key, item in value.items():
            lowered_key = str(key).strip().lower()
            if lowered_key in FORBIDDEN_EXTRA_PARAM_KEYS:
                raise ValueError(f"禁止使用危险的数据库连接参数：{key}")
            if not re.fullmatch(r"[A-Za-z0-9_]+", str(key)):
                raise ValueError(f"非法的数据库连接参数名：{key}")
            if isinstance(item, (bool, int, float)):
                continue
            if isinstance(item, str):
                # '&'/'?'/'#'/';' 是 URL 查询串与 ODBC 连接串的分隔符，
                # 攻击者可用它们注入 local_infile 等其他参数
                if re.search(r"[&?#;]", item):
                    raise ValueError(f"数据库连接参数 {key} 的值包含非法字符")
                continue
            raise ValueError(f"数据库连接参数 {key} 的值类型不受支持")
        return value

    def to_clean_dict(self, *, drop_blank_password: bool = False) -> Dict[str, Any]:
        data = self.dict(exclude_none=True)
        if drop_blank_password and data.get("password") == "":
            data.pop("password", None)
        return data


class ConnectionCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    config: DatabaseConfigPayload
    description: str = Field(default="", max_length=500)
    tags: List[str] = Field(default_factory=list)


class ConnectionUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    config: Optional[DatabaseConfigPayload] = None
    description: Optional[str] = Field(default=None, max_length=500)
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None


class QueryPreviewRequest(BaseModel):
    connection_id: str
    sql: str = Field(min_length=1, max_length=20000)
    limit: int = Field(default=100, ge=1, le=1000)


def _manager() -> ConnectionManager:
    return ConnectionManager(storage_path=os.getenv("DATABASE_CONNECTIONS_PATH"))


def _local_secret_path() -> Path:
    """每安装随机密钥的持久化路径（可用 DATABASE_CONFIG_SECRET_PATH 覆盖）。"""
    configured = os.getenv("DATABASE_CONFIG_SECRET_PATH")
    if configured:
        return Path(configured)
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "config" / ".db_config_secret"


def _load_or_create_local_secret() -> str:
    """读取或生成每安装随机密钥，避免把加密密钥硬编码进源码。

    密钥写入 config/.db_config_secret（0600 权限，已被 .gitignore 排除）。
    若文件系统不可写，则退化为进程内随机密钥——重启后旧凭证需重新录入，
    但绝不会使用公开源码里的常量作为密钥。
    """
    path = _local_secret_path()
    try:
        if path.exists():
            seed = path.read_text(encoding="utf-8").strip()
            if seed:
                return seed
        seed = secrets.token_hex(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "x", encoding="utf-8") as handle:
                handle.write(seed)
        except FileExistsError:
            # 并发初始化：另一个进程已写入，直接读回
            return path.read_text(encoding="utf-8").strip()
        path.chmod(0o600)
        return seed
    except OSError:
        return secrets.token_hex(32)


def _secret_key() -> bytes:
    seed = os.getenv("DATABASE_CONFIG_SECRET") or os.getenv("SECRET_KEY")
    if not seed:
        seed = _load_or_create_local_secret()
    return hashlib.sha256(seed.encode("utf-8")).digest()


def _keystream(length: int, nonce: bytes) -> bytes:
    chunks: List[bytes] = []
    counter = 0
    key = _secret_key()
    while sum(len(chunk) for chunk in chunks) < length:
        chunks.append(hmac.new(key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest())
        counter += 1
    return b"".join(chunks)[:length]


def _encrypt_secret(value: str) -> str:
    if not value or value.startswith(SECRET_PREFIX):
        return value
    nonce = os.urandom(12)
    raw = value.encode("utf-8")
    stream = _keystream(len(raw), nonce)
    cipher = bytes(left ^ right for left, right in zip(raw, stream))
    signature = hmac.new(_secret_key(), nonce + cipher, hashlib.sha256).digest()[:16]
    token = base64.urlsafe_b64encode(nonce + signature + cipher).decode("ascii")
    return f"{SECRET_PREFIX}{token}"


def _decrypt_secret(value: str) -> str:
    if not value or not value.startswith(SECRET_PREFIX):
        return value
    try:
        payload = base64.urlsafe_b64decode(value[len(SECRET_PREFIX) :].encode("ascii"))
        nonce, signature, cipher = payload[:12], payload[12:28], payload[28:]
        expected = hmac.new(_secret_key(), nonce + cipher, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(signature, expected):
            raise ValueError("bad signature")
        stream = _keystream(len(cipher), nonce)
        return bytes(left ^ right for left, right in zip(cipher, stream)).decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="数据库凭证解密失败") from exc


def _mask_secret(value: str) -> str:
    # 存储值可能是 enc:v1: 密文，必须先解密再掩码，否则会泄露密文片段
    plain = _decrypt_secret(value) if value.startswith(SECRET_PREFIX) else value
    return mask_secret(plain)


def _encrypt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    encrypted = dict(config)
    if "password" in encrypted and encrypted["password"]:
        encrypted["password"] = _encrypt_secret(str(encrypted["password"]))
    return encrypted


def _runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = dict(config)
    if runtime.get("password"):
        runtime["password"] = _decrypt_secret(str(runtime["password"]))
    return runtime


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {key: value for key, value in config.items() if key != "password"}
    password = config.get("password")
    if password:
        sanitized["has_password"] = True
        try:
            sanitized["password_masked"] = _mask_secret(str(password))
            sanitized["credential_status"] = "ok"
        except HTTPException:
            # A rotated/missing installation key must not make the entire
            # connection registry unavailable. The ciphertext stays hidden and
            # actual connection attempts still fail closed in _runtime_config.
            sanitized["password_masked"] = "无法解密，请重新录入"
            sanitized["credential_status"] = "unreadable"
    else:
        sanitized["has_password"] = False
        sanitized["credential_status"] = "not_set"
    return sanitized


def _connection_snapshot(raw: Dict[str, Any], current_connection: Optional[str]) -> Dict[str, Any]:
    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "db_type": raw.get("db_type") or raw.get("config", {}).get("db_type"),
        "description": raw.get("description", ""),
        "tags": raw.get("tags", []),
        "created_at": raw.get("created_at"),
        "last_used": raw.get("last_used"),
        "use_count": raw.get("use_count", 0),
        "is_active": raw.get("id") == current_connection or bool(raw.get("is_active", False)),
        "config": _sanitize_config(raw.get("config", {})),
    }


def _require_connection(manager: ConnectionManager, connection_id: str) -> Dict[str, Any]:
    validate_entity_id(connection_id, "connection_id")
    connection = manager.connections.get(connection_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="未找到数据库连接")
    return connection


def _connector_for(raw: Dict[str, Any]) -> DatabaseConnector:
    return DatabaseConnector(_runtime_config(raw.get("config", {})))


def get_connector(connection_id: str) -> DatabaseConnector:
    """按已保存连接 ID 返回凭证已解密的连接器，供同级 API 模块复用（不存在时 404）。"""
    return _connector_for(_require_connection(_manager(), connection_id))


def _is_readonly_sql(sql: str) -> bool:
    normalized = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    normalized = re.sub(r"--.*?$", " ", normalized, flags=re.MULTILINE).strip().rstrip(";").strip()
    if ";" in normalized:
        return False
    lowered = normalized.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False
    if WRITE_SQL_PATTERN.search(normalized) is not None:
        return False
    # 仅拦截写关键词不够：SELECT 内部可调用的副作用函数
    # （LOAD_FILE/lo_import/dblink/OPENROWSET/COPY 等）同样必须拒绝
    return DANGEROUS_SQL_PATTERN.search(normalized) is None


def _preview_sql(sql: str, limit: int) -> str:
    stripped = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    stripped = re.sub(r"--.*?$", " ", stripped, flags=re.MULTILINE).strip().rstrip(";")
    return f"SELECT * FROM ({stripped}) AS preview_query LIMIT {limit}"  # nosec B608


def _dataframe_payload(df) -> Dict[str, Any]:
    # where(..., None) retains NaT/NaN in datetime/float columns. Let pandas
    # normalize these to JSON null and serialize actual dates as ISO strings.
    rows = json.loads(df.to_json(orient="records", date_format="iso"))
    return {
        "columns": [str(column) for column in df.columns],
        "rows": rows,
        "row_count": int(len(df)),
    }


class ConnectionTestRequest(BaseModel):
    config: DatabaseConfigPayload
    connection_id: Optional[str] = None


@router.post("/test")
def test_adhoc_connection(payload: ConnectionTestRequest) -> Dict[str, Any]:
    """按请求中的配置直接测试连接（不要求先保存）。

    编辑现有连接且密码留空时，通过 connection_id 复用已存储的加密凭证，
    与列表里"测试连接"按钮的语义保持一致。
    """
    incoming = payload.config.to_clean_dict(drop_blank_password=True)
    config = {}
    if payload.connection_id:
        raw = _require_connection(_manager(), payload.connection_id)
        saved = raw.get("config", {})
        if saved.get("db_type") == incoming["db_type"]:
            # Match update_connection: preserve credentials and driver options
            # omitted by the editor, but test all supplied edits without saving.
            config.update(saved)
    config.update(incoming)
    return _test_config(config)


def _test_config(config: Dict[str, Any]) -> Dict[str, Any]:
    connector = None
    runtime = {}
    try:
        runtime = _runtime_config(config)
        connector = DatabaseConnector(runtime)
        ok = connector.test_connection()
        detail = getattr(connector, "last_error", "") or "请检查配置"
    except HTTPException as exc:
        ok, detail = False, str(exc.detail)
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    finally:
        if connector is not None:
            connector.disconnect()
    # Driver errors must not echo a submitted credential into the page.
    password = runtime.get("password")
    if password:
        detail = detail.replace(str(password), "****")
    return {"ok": bool(ok), "message": "连接测试通过" if ok else f"连接测试失败：{detail}"}


@router.get("/connections")
def list_connections() -> Dict[str, Any]:
    manager = _manager()
    connections = [
        _connection_snapshot(connection, manager.current_connection)
        for connection in sorted(manager.connections.values(), key=lambda item: item.get("last_used", ""), reverse=True)
    ]
    active = manager.connections.get(manager.current_connection or "")
    return {
        "connections": connections,
        "active_connection": _connection_snapshot(active, manager.current_connection) if active else None,
        "statistics": {
            "total_connections": len(connections),
            "active_connection": active.get("name") if active else None,
            "db_types": sorted({connection["db_type"] for connection in connections if connection.get("db_type")}),
        },
    }


@router.post("/connections", status_code=status.HTTP_201_CREATED)
def create_connection(payload: ConnectionCreateRequest) -> Dict[str, Any]:
    manager = _manager()
    config = _encrypt_config(payload.config.to_clean_dict())
    connection_id = manager._generate_connection_id(config)
    if connection_id in manager.connections:
        raise HTTPException(status_code=409, detail="相同数据库连接配置已存在，请编辑现有连接")
    connection_id = manager.add_connection(
        name=payload.name.strip(),
        config=config,
        description=payload.description.strip(),
        tags=[tag.strip() for tag in payload.tags if tag.strip()],
    )
    return _connection_snapshot(manager.connections[connection_id], manager.current_connection)


@router.patch("/connections/{connection_id}")
def update_connection(connection_id: str, payload: ConnectionUpdateRequest) -> Dict[str, Any]:
    manager = _manager()
    raw = dict(_require_connection(manager, connection_id))
    if payload.name is not None:
        raw["name"] = payload.name.strip()
    if payload.description is not None:
        raw["description"] = payload.description.strip()
    if payload.tags is not None:
        raw["tags"] = [tag.strip() for tag in payload.tags if tag.strip()]
    if payload.config is not None:
        current_config = dict(raw.get("config", {}))
        incoming = payload.config.to_clean_dict(drop_blank_password=True)
        current_config.update(_encrypt_config(incoming))
        raw["config"] = current_config
        raw["db_type"] = current_config.get("db_type", raw.get("db_type"))
    if payload.is_active is not None:
        raw["is_active"] = payload.is_active
        if payload.is_active:
            manager.current_connection = connection_id
    raw["last_updated"] = datetime.now().isoformat()
    manager.connections[connection_id] = raw
    if not manager._save_connections():
        raise HTTPException(status_code=500, detail="保存数据库连接失败")
    return _connection_snapshot(raw, manager.current_connection)


@router.delete("/connections/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_connection(connection_id: str) -> Response:
    manager = _manager()
    _require_connection(manager, connection_id)
    manager.delete_connection(connection_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/connections/{connection_id}/test")
def test_connection(connection_id: str) -> Dict[str, Any]:
    manager = _manager()
    raw = _require_connection(manager, connection_id)
    return {"connection_id": connection_id, **_test_config(raw.get("config", {}))}


@router.get("/connections/{connection_id}/schema")
def get_schema(connection_id: str, sample_limit: int = 5) -> Dict[str, Any]:
    manager = _manager()
    raw = _require_connection(manager, connection_id)
    try:
        connector = _connector_for(raw)
        tables = []
        for table_name in connector.list_tables():
            columns = [
                {
                    "name": str(column.get("name")),
                    "type": str(column.get("type")),
                    "nullable": bool(column.get("nullable", True)),
                }
                for column in connector.get_table_columns(table_name)
            ]
            sample = connector.read_table(table_name, limit=max(1, min(sample_limit, 20)))
            tables.append({"name": table_name, "columns": columns, "sample": _dataframe_payload(sample)})
        connector.disconnect()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Schema 读取失败：{type(exc).__name__}") from exc
    return {"connection": _connection_snapshot(raw, manager.current_connection), "tables": tables}


@router.post("/query-preview")
def query_preview(payload: QueryPreviewRequest) -> Dict[str, Any]:
    if not _is_readonly_sql(payload.sql):
        raise HTTPException(status_code=400, detail="仅允许 SELECT/CTE 只读查询预览")

    manager = _manager()
    raw = _require_connection(manager, payload.connection_id)
    try:
        connector = _connector_for(raw)
        df = connector.query(_preview_sql(payload.sql, payload.limit))
        connector.disconnect()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"查询预览失败：{type(exc).__name__}") from exc
    return {
        "connection_id": payload.connection_id,
        "limited": True,
        "limit": payload.limit,
        **_dataframe_payload(df),
    }


@router.get("/connections/{connection_id}/tables/{table_name}/rows")
def get_table_rows(
    connection_id: str,
    table_name: str,
    limit: int = Query(default=1000, ge=1, le=TABLE_ROWS_LIMIT_MAX),
) -> Dict[str, Any]:
    """只读读取指定表数据，供批量预测等场景导入特征行。"""
    if not TABLE_NAME_PATTERN.fullmatch(table_name):
        raise HTTPException(status_code=400, detail="无效的表名：仅支持字母/数字/下划线，可带 schema 前缀")
    manager = _manager()
    raw = _require_connection(manager, connection_id)
    connector = _connector_for(raw)
    try:
        if connector.engine is None and not connector.connect():
            raise HTTPException(status_code=400, detail=f"数据库连接失败：{connector.last_error or '请检查连接配置'}")
        schema, _, name = table_name.rpartition(".")
        if not sqlalchemy_inspect(connector.engine).has_table(name, schema=schema or None):
            raise HTTPException(status_code=404, detail="未找到数据表")
        df = connector.read_table(table_name, limit=limit)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"表数据读取失败：{type(exc).__name__}") from exc
    finally:
        connector.disconnect()
    return _dataframe_payload(df)


@router.get("/connections/{invalid_connection_id:path}/schema")
@router.patch("/connections/{invalid_connection_id:path}")
@router.delete("/connections/{invalid_connection_id:path}")
def reject_invalid_connection_path(invalid_connection_id: str) -> None:
    validate_entity_id(invalid_connection_id, "connection_id")
    raise HTTPException(status_code=404, detail="未找到数据库连接")
