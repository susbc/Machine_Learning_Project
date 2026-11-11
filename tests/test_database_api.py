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


def test_draft_connection_reuses_credentials_and_driver_options_without_saving(monkeypatch, tmp_path):
    storage = tmp_path / 'connections.json'
    monkeypatch.setenv('DATABASE_CONNECTIONS_PATH', str(storage))
    monkeypatch.setenv('DATABASE_CONFIG_SECRET', 'draft-test-secret')
    client = TestClient(app)
    saved_config = {
        'db_type': 'mysql', 'host': 'localhost', 'database': 'original',
        'user': 'tester', 'password': 'saved-secret',
        'extra_params': {'charset': 'utf8mb4'},
    }
    connection = client.post('/api/database/connections', json={
        'name': 'draft test', 'config': saved_config,
    }).json()
    before = storage.read_bytes()
    seen, closed = [], []

    class Connector:
        last_error = None

        def __init__(self, config):
            seen.append(config)

        def test_connection(self):
            return True

        def disconnect(self):
            closed.append(True)

    monkeypatch.setattr(database_api, 'DatabaseConnector', Connector)
    draft = {**saved_config, 'database': 'edited', 'password': ''}
    draft.pop('extra_params')
    for password, expected in [('', 'saved-secret'), ('replacement-secret', 'replacement-secret')]:
        draft['password'] = password
        response = client.post('/api/database/test', json={
            'connection_id': connection['id'], 'config': draft,
        })
        assert response.status_code == 200
        assert response.json()['ok'] is True
        assert seen[-1]['database'] == 'edited'
        assert seen[-1]['password'] == expected
        assert seen[-1]['extra_params'] == {'charset': 'utf8mb4'}
        assert storage.read_bytes() == before
    assert len(closed) == 2


def test_draft_connection_failure_closes_connector_and_redacts_password(monkeypatch):
    closed = []

    class Connector:
        def __init__(self, config):
            self.password = config['password']

        def test_connection(self):
            raise RuntimeError('connection rejected: ' + self.password)

        def disconnect(self):
            closed.append(True)

    monkeypatch.setattr(database_api, 'DatabaseConnector', Connector)
    response = TestClient(app).post('/api/database/test', json={'config': {
        'db_type': 'mysql', 'host': 'localhost', 'database': 'draft', 'password': 'private-secret',
    }})
    assert response.status_code == 200
    assert response.json()['ok'] is False
    assert 'connection rejected' in response.json()['message']
    assert 'private-secret' not in response.text
    assert closed == [True]


def test_schema_and_preview_serialize_missing_dates_and_numbers(monkeypatch, tmp_path):
    import pandas as pd

    monkeypatch.setenv('DATABASE_CONNECTIONS_PATH', str(tmp_path / 'connections.json'))
    client = TestClient(app)
    connection = client.post('/api/database/connections', json={
        'name': 'nullable data', 'config': {'db_type': 'sqlite', 'database': ':memory:'},
    }).json()
    frame = pd.DataFrame({
        'recorded_at': pd.to_datetime(['2026-11-06 03:00:00', None, None]),
        'score': [1.25, float('nan'), float('inf')],
        'note': ['中文', None, None],
    })

    class Connector:
        def list_tables(self):
            return ['records']

        def get_table_columns(self, name):
            return [{'name': 'recorded_at', 'type': 'DATETIME', 'nullable': True}]

        def read_table(self, name, limit):
            return frame

        def query(self, sql):
            return frame

        def disconnect(self):
            pass

    monkeypatch.setattr(database_api, '_connector_for', lambda raw: Connector())
    schema = client.get(f"/api/database/connections/{connection['id']}/schema")
    assert schema.status_code == 200
    preview = client.post('/api/database/query-preview', json={
        'connection_id': connection['id'], 'sql': 'SELECT * FROM records',
    })
    assert preview.status_code == 200
    for payload in [schema.json()['tables'][0]['sample'], preview.json()]:
        assert payload['row_count'] == 3
        assert payload['rows'][0] == {
            'recorded_at': '2026-11-06T03:00:00.000', 'score': 1.25, 'note': '中文',
        }
        assert payload['rows'][1] == {'recorded_at': None, 'score': None, 'note': None}
        assert payload['rows'][2]['score'] is None


def test_table_rows_endpoint_reads_features_for_batch_import(monkeypatch, tmp_path):
    db_path = tmp_path / "sample.db"
    _create_sqlite_db(db_path)
    monkeypatch.setenv("DATABASE_CONNECTIONS_PATH", str(tmp_path / "saved_connections.json"))
    monkeypatch.setenv("DATABASE_CONFIG_SECRET", "test-secret")
    client = TestClient(app)
    connection_id = client.post(
        "/api/database/connections",
        json={"name": "rows sqlite", "config": {"db_type": "sqlite", "database": str(db_path)}},
    ).json()["id"]

    response = client.get(f"/api/database/connections/{connection_id}/tables/customers/rows?limit=2")
    assert response.status_code == 200
    payload = response.json()
    assert payload["columns"] == ["id", "name", "score"]
    assert payload["row_count"] == 2
    assert payload["rows"][0] == {"id": 1, "name": "Ada", "score": 0.91}

    full = client.get(f"/api/database/connections/{connection_id}/tables/customers/rows")
    assert full.status_code == 200
    assert full.json()["row_count"] == 3

    missing = client.get(f"/api/database/connections/{connection_id}/tables/no_such_table/rows")
    assert missing.status_code == 404

    invalid = client.get(f"/api/database/connections/{connection_id}/tables/bad%20name/rows")
    assert invalid.status_code == 400

    oversized = client.get(f"/api/database/connections/{connection_id}/tables/customers/rows?limit=10001")
    assert oversized.status_code == 422
