import hashlib
import json
import sqlite3

from fastapi.testclient import TestClient

from backend.api import database as database_api
from backend.main import app


def _create_sqlite_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
    conn.executemany(
        "INSERT INTO customers (name, score) VALUES (?, ?)",
        [("Ada", 0.91), ("Lin", 0.73), ("Max", 0.42)],
    )
    conn.commit()
    conn.close()


def test_database_connection_schema_and_readonly_preview(monkeypatch, tmp_path):
    db_path = tmp_path / "sample.db"
    storage_path = tmp_path / "saved_connections.json"
    _create_sqlite_db(db_path)
    monkeypatch.setenv("DATABASE_CONNECTIONS_PATH", str(storage_path))
    monkeypatch.setenv("DATABASE_CONFIG_SECRET", "test-secret")
    client = TestClient(app)

    create_response = client.post(
        "/api/database/connections",
        json={
            "name": "local sqlite",
            "config": {"db_type": "sqlite", "database": str(db_path)},
            "description": "unit test database",
            "tags": ["test"],
        },
    )
    assert create_response.status_code == 201
    connection = create_response.json()
    connection_id = connection["id"]
    assert connection["config"]["database"].endswith("sample.db")

    list_response = client.get("/api/database/connections")
    assert list_response.status_code == 200
    assert list_response.json()["connections"][0]["id"] == connection_id

    test_response = client.post(f"/api/database/connections/{connection_id}/test")
    assert test_response.status_code == 200
    assert test_response.json()["ok"] is True

    schema_response = client.get(f"/api/database/connections/{connection_id}/schema")
    assert schema_response.status_code == 200
    schema = schema_response.json()
    assert schema["tables"][0]["name"] == "customers"
    assert any(column["name"] == "score" for column in schema["tables"][0]["columns"])

    preview_response = client.post(
        "/api/database/query-preview",
        json={"connection_id": connection_id, "sql": "SELECT id, name, score FROM customers ORDER BY id", "limit": 2},
    )
    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["limited"] is True
    assert preview["row_count"] == 2
    assert preview["rows"][0]["name"] == "Ada"

    write_response = client.post(
        "/api/database/query-preview",
        json={"connection_id": connection_id, "sql": "DELETE FROM customers", "limit": 2},
    )
    assert write_response.status_code == 400

    invalid_id_response = client.get("/api/database/connections/..%2Fsecret/schema")
    assert invalid_id_response.status_code == 400


def test_connection_list_survives_an_unreadable_saved_password(monkeypatch, tmp_path):
    storage_path = tmp_path / "saved_connections.json"
    storage_path.write_text(
        json.dumps(
            {
                "connections": {
                    "deadbeef": {
                        "id": "deadbeef",
                        "name": "legacy database",
                        "db_type": "postgresql",
                        "config": {
                            "db_type": "postgresql",
                            "database": "legacy",
                            "password": "enc:v1:not-a-valid-token",
                        },
                    }
                },
                "current_connection": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_CONNECTIONS_PATH", str(storage_path))
    monkeypatch.setenv("DATABASE_CONFIG_SECRET", "rotated-secret")
    client = TestClient(app)

    response = client.get("/api/database/connections")

    assert response.status_code == 200
    config = response.json()["connections"][0]["config"]
    assert config["credential_status"] == "unreadable"
    assert config["password_masked"] == "无法解密，请重新录入"
    assert "enc:v1:" not in response.text


def test_database_password_is_encrypted_at_rest_and_masked(monkeypatch, tmp_path):
    storage_path = tmp_path / "saved_connections.json"
    monkeypatch.setenv("DATABASE_CONNECTIONS_PATH", str(storage_path))
    monkeypatch.setenv("DATABASE_CONFIG_SECRET", "test-secret")
    client = TestClient(app)

    create_response = client.post(
        "/api/database/connections",
        json={
            "name": "prod pg",
            "config": {
                "db_type": "postgresql",
                "host": "db.example.com",
                "port": 5432,
                "database": "warehouse",
                "user": "analyst",
                "password": "plain-secret",
            },
        },
    )

    assert create_response.status_code == 201
    body = create_response.json()
    assert body["config"]["password_masked"] == "pla...cret"
    assert "plain-secret" not in json.dumps(body, ensure_ascii=False)
    assert "plain-secret" not in storage_path.read_text(encoding="utf-8")
    assert "enc:v1:" in storage_path.read_text(encoding="utf-8")


def test_database_secret_key_never_uses_hardcoded_fallback(monkeypatch, tmp_path):
    """未配置环境变量时必须生成并持久化随机密钥，禁止回退到源码内的硬编码密钥。"""
    monkeypatch.delenv("DATABASE_CONFIG_SECRET", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    secret_path = tmp_path / "db_config_secret"
    monkeypatch.setenv("DATABASE_CONFIG_SECRET_PATH", str(secret_path))

    key = database_api._secret_key()

    hardcoded = hashlib.sha256("agentic-ml-studio-local-secret".encode("utf-8")).digest()
    assert key != hardcoded
    assert secret_path.exists()
    assert (secret_path.stat().st_mode & 0o777) == 0o600
    # 再次调用必须复用同一密钥，否则已加密凭证无法解密
    assert database_api._secret_key() == key


def test_database_secret_key_prefers_configured_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_CONFIG_SECRET", "env-secret")
    monkeypatch.setenv("DATABASE_CONFIG_SECRET_PATH", str(tmp_path / "should_not_exist"))

    key = database_api._secret_key()

    assert key == hashlib.sha256("env-secret".encode("utf-8")).digest()
    assert not (tmp_path / "should_not_exist").exists()


def test_database_create_rejects_duplicate_connection_config(monkeypatch, tmp_path):
    db_path = tmp_path / "sample.db"
    storage_path = tmp_path / "saved_connections.json"
    _create_sqlite_db(db_path)
    monkeypatch.setenv("DATABASE_CONNECTIONS_PATH", str(storage_path))
    client = TestClient(app)
    payload = {
        "name": "local sqlite",
        "config": {"db_type": "sqlite", "database": str(db_path)},
    }

    first = client.post("/api/database/connections", json=payload)
    second = client.post("/api/database/connections", json={**payload, "name": "local sqlite copy"})

    assert first.status_code == 201
    assert second.status_code == 409
    assert "已存在" in second.json()["detail"]


def test_database_connection_rejects_dangerous_extra_params(monkeypatch, tmp_path):
    """extra_params 必须封禁 local_infile 等危险驱动参数，防止恶意 MySQL 服务器
    通过 LOAD DATA LOCAL INFILE 读取本机任意文件（安全审查 Vuln 1）。"""
    storage_path = tmp_path / "saved_connections.json"
    monkeypatch.setenv("DATABASE_CONNECTIONS_PATH", str(storage_path))
    monkeypatch.setenv("DATABASE_CONFIG_SECRET", "test-secret")
    client = TestClient(app)

    def create(name, **overrides):
        config = {
            "db_type": "mysql",
            "host": "db.example.com",
            "database": "appdb",
            "user": "analyst",
            "password": "pw",
        }
        config.update(overrides)
        return client.post("/api/database/connections", json={"name": name, "config": config})

    # local_infile 会让恶意 MySQL 服务器读取客户端本机文件
    assert create("bad-local-infile", extra_params={"local_infile": 1}).status_code == 422
    assert create("bad-read-default-file", extra_params={"read_default_file": "/etc/my.cnf"}).status_code == 422
    assert create("bad-init-command", extra_params={"init_command": "SELECT 1"}).status_code == 422
    # 参数值中的分隔符可用于注入其他连接参数
    assert create("bad-param-value", extra_params={"charset": "utf8&local_infile=1"}).status_code == 422
    # database 字段的查询串注入路径同样被封堵
    assert create("bad-database-query", database="appdb?local_infile=1").status_code == 422
    assert create("bad-database-at", database="app@db").status_code == 422
    # 合法参数不受影响
    assert create("good-params", extra_params={"charset": "utf8mb4", "connect_timeout": 5}).status_code == 201


def test_query_preview_blocks_side_effect_functions(monkeypatch, tmp_path):
    """只读预览必须拒绝 SELECT 内可调用的副作用函数（文件读取/外部连接/RCE 代理），
    防止"只读"保护被绕过（安全审查边缘发现）。"""
    db_path = tmp_path / "sample.db"
    storage_path = tmp_path / "saved_connections.json"
    _create_sqlite_db(db_path)
    monkeypatch.setenv("DATABASE_CONNECTIONS_PATH", str(storage_path))
    monkeypatch.setenv("DATABASE_CONFIG_SECRET", "test-secret")
    client = TestClient(app)

    created = client.post(
        "/api/database/connections",
        json={"name": "sqlite for guard test", "config": {"db_type": "sqlite", "database": str(db_path)}},
    )
    assert created.status_code == 201
    connection_id = created.json()["id"]

    dangerous_statements = [
        "SELECT LOAD_FILE('/etc/passwd')",
        "SELECT lo_import('/etc/passwd')",
        "SELECT lo_get(lo_import('/etc/passwd'))",
        "SELECT * FROM dblink('dbname=prod','SELECT 1') AS t(x int)",
        "SELECT * FROM OPENROWSET('SQLNCLI', 'Server=x;', 'SELECT 1')",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM customers INTO OUTFILE '/tmp/dump.txt'",
    ]
    for sql in dangerous_statements:
        response = client.post(
            "/api/database/query-preview",
            json={"connection_id": connection_id, "sql": sql, "limit": 5},
        )
        assert response.status_code == 400, f"应拒绝的危险 SQL 未被拦截：{sql}"

    # 合法查询不受影响
    normal = client.post(
        "/api/database/query-preview",
        json={"connection_id": connection_id, "sql": "SELECT id, name FROM customers", "limit": 5},
    )
    assert normal.status_code == 200
