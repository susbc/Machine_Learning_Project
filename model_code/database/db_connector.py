"""
数据库连接器模块

提供统一的数据库连接接口，支持 MySQL、PostgreSQL、SQLite、SQL Server
"""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, Dict, Any, List
import pandas as pd
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 驱动层禁止的连接参数（与 backend/api/database.py 的 FORBIDDEN_EXTRA_PARAM_KEYS 对齐）。
# 存量连接配置（config/saved_connections.json）可能绕过 API 层新增的校验，
# 因此连接串构造层必须再拦截一次：
#   local_infile      —— pymysql 启用 LOAD DATA LOCAL，恶意服务器可读取本机任意文件
#   read_default_file —— pymysql 读取任意路径的 MySQL 选项文件
#   client_flag       —— 可重新打开 CLIENT_LOCAL_FILES 能力位
#   init_command      —— 连接建立后自动执行任意 SQL
#   dsn/odbc_connect  —— pyodbc 任意 DSN / 完整连接串
FORBIDDEN_DRIVER_PARAM_KEYS = {
    "local_infile",
    "read_default_file",
    "read_default_group",
    "client_flag",
    "init_command",
    "dsn",
    "odbc_connect",
}


class DatabaseConnector:
    """
    数据库连接器类
    
    支持多种数据库类型，提供统一的连接和查询接口
    """
    
    # 数据库类型映射
    DB_DRIVERS = {
        'mysql': 'mysql+pymysql',
        'postgresql': 'postgresql+psycopg2',
        'sqlite': 'sqlite',
        'sqlserver': 'mssql+pyodbc'
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据库连接器
        
        Args:
            config: 数据库配置字典，包含:
                - db_type: 数据库类型 (mysql/postgresql/sqlite/sqlserver)
                - host: 主机地址 (SQLite 不需要)
                - port: 端口号 (SQLite 不需要)
                - user: 用户名 (SQLite 不需要)
                - password: 密码 (SQLite 不需要)
                - database: 数据库名 (SQLite 为文件路径)
                - extra_params: 额外参数 (可选)
        """
        self.config = config
        self.db_type = config.get('db_type', 'mysql').lower()
        self.engine: Optional[Any] = None
        self.connection = None
        self._connection_string = None
        # 最近一次连接/测试失败的原因，供 API 层返回给用户定位问题
        self.last_error: Optional[str] = None
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证数据库配置"""
        if self.db_type not in self.DB_DRIVERS:
            raise ValueError(f"不支持的数据库类型：{self.db_type}。支持的类型：{list(self.DB_DRIVERS.keys())}")
        
        # SQLite 只需要 database 字段
        if self.db_type == 'sqlite':
            if 'database' not in self.config:
                raise ValueError("SQLite 需要指定 database 字段 (文件路径)")
        else:
            # 其他数据库需要完整配置
            required_fields = ['host', 'database']
            for field in required_fields:
                if field not in self.config:
                    raise ValueError(f"{self.db_type} 需要指定 {field} 字段")
    
    def _build_connection_string(self) -> str:
        """构建数据库连接字符串。

        使用 SQLAlchemy 的 URL.create 组装（而非手工字符串拼接），
        并对 extra_params 做驱动层黑名单校验，防止：
        1) 危险驱动参数（如 local_infile）把连接变成任意文件读取原语；
        2) 参数值中的 '&'/'?'/'#' 注入额外连接参数；
        3) 用户名/密码/库名中的特殊字符破坏或注入 URL 段。
        """
        database = str(self.config['database'])

        if self.db_type == 'sqlite':
            return URL.create('sqlite', database=database).render_as_string(hide_password=False)

        driver = self.DB_DRIVERS[self.db_type]
        user = self.config.get('user', '') or None
        password = self.config.get('password', '') or None
        host = self.config.get('host', 'localhost')
        port = int(self.config.get('port', self._get_default_port()))

        query: Dict[str, str] = {}
        extra_params = self.config.get('extra_params') or {}
        for key, value in extra_params.items():
            str_key = str(key)
            if str_key.strip().lower() in FORBIDDEN_DRIVER_PARAM_KEYS:
                raise ValueError(f"数据库连接配置包含被禁止的危险参数：{str_key}")
            if isinstance(value, bool):
                query[str_key] = "1" if value else "0"
            elif isinstance(value, (int, float)):
                query[str_key] = str(value)
            elif isinstance(value, str):
                if re.search(r"[&?#;]", value):
                    raise ValueError(f"数据库连接参数 {str_key} 的值包含非法字符")
                query[str_key] = value
            else:
                raise ValueError(f"数据库连接参数 {str_key} 的值类型不受支持")

        url = URL.create(
            driver,
            username=user,
            password=password,
            host=host,
            port=port,
            database=database,
            query=query,
        )
        return url.render_as_string(hide_password=False)
    
    def _get_default_port(self) -> int:
        """获取数据库默认端口"""
        default_ports = {
            'mysql': 3306,
            'postgresql': 5432,
            'sqlserver': 1433
        }
        return default_ports.get(self.db_type, 3306)
    
    def connect(self) -> bool:
        """
        建立数据库连接
        
        Returns:
            连接是否成功
        """
        try:
            self._connection_string = self._build_connection_string()
            self.engine = create_engine(
                self._connection_string,
                pool_pre_ping=True,  # 连接前验证
                pool_recycle=3600,   # 1 小时后回收连接
                echo=False  # 不打印 SQL 日志
            )
            self.connection = self.engine.connect()
            logger.info(f"成功连接到 {self.db_type} 数据库")
            self.last_error = None
            return True
        except SQLAlchemyError as e:
            logger.error(f"数据库连接失败：{str(e)}")
            self.last_error = str(e)
            self.engine = None
            self.connection = None
            return False
    
    def test_connection(self) -> bool:
        """
        测试数据库连接
        
        Returns:
            连接是否正常
        """
        if self.engine is None:
            if not self.connect():
                return False
        
        try:
            # 执行简单查询测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("数据库连接测试成功")
            self.last_error = None
            return True
        except SQLAlchemyError as e:
            logger.error(f"数据库连接测试失败：{str(e)}")
            self.last_error = str(e)
            return False
    
    def disconnect(self) -> None:
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.engine:
            self.engine.dispose()
            self.engine = None
        logger.info("数据库连接已关闭")
    
    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        获取数据库中的所有表
        
        Args:
            schema: 模式名 (可选，PostgreSQL/SQL Server 使用)
        
        Returns:
            表名列表
        """
        if self.engine is None:
            if not self.connect():
                return []
        
        try:
            # 使用 SQLAlchemy 的 inspector 获取表列表
            from sqlalchemy import inspect
            inspector = inspect(self.engine)
            
            if schema:
                tables = inspector.get_table_names(schema=schema)
            else:
                tables = inspector.get_table_names()
            
            logger.info(f"找到 {len(tables)} 个表")
            return tables
        except SQLAlchemyError as e:
            logger.error(f"获取表列表失败：{str(e)}")
            return []
    
    def query(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        执行 SQL 查询
        
        Args:
            sql: SQL 查询语句
            params: 查询参数 (用于防止 SQL 注入)
        
        Returns:
            查询结果 DataFrame
        """
        if self.engine is None:
            if not self.connect():
                return pd.DataFrame()
        
        try:
            # 使用 pandas 直接读取
            df = pd.read_sql_query(text(sql), self.engine, params=params)
            logger.info(f"查询返回 {len(df)} 行")
            return df
        except SQLAlchemyError as e:
            logger.error(f"查询执行失败：{str(e)}")
            return pd.DataFrame()

    def _quote_identifier(self, identifier: str) -> str:
        """
        安全引用表名/列名。

        只允许常见的 schema.table / column 标识符格式，避免用户输入拼入 SQL。
        """
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", identifier):
            raise ValueError(f"无效的 SQL 标识符：{identifier}")

        if self.engine is None:
            if not self.connect():
                raise RuntimeError("数据库连接失败")

        preparer = self.engine.dialect.identifier_preparer
        return ".".join(preparer.quote(part) for part in identifier.split("."))
    
    def read_table(self, table: str, limit: int = 1000, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        读取数据库表数据
        
        Args:
            table: 表名
            limit: 最大读取行数
            columns: 要读取的列 (可选)
        
        Returns:
            表数据 DataFrame
        """
        safe_table = self._quote_identifier(table)
        safe_limit = max(1, min(int(limit), 100000))

        if columns:
            cols = ", ".join(self._quote_identifier(column) for column in columns)
            sql = f"SELECT {cols} FROM {safe_table} LIMIT {safe_limit}"  # nosec B608
        else:
            sql = f"SELECT * FROM {safe_table} LIMIT {safe_limit}"  # nosec B608
        
        return self.query(sql)

    def count_table(self, table: str) -> int:
        """安全统计表行数。"""
        safe_table = self._quote_identifier(table)
        df = self.query(f"SELECT COUNT(*) as count FROM {safe_table}")  # nosec B608
        if len(df) == 0:
            return 0
        return int(df["count"].iloc[0])
    
    def get_table_columns(self, table: str) -> List[Dict[str, Any]]:
        """
        获取表的列信息
        
        Args:
            table: 表名
        
        Returns:
            列信息列表，每项包含 name, type, nullable 等
        """
        if self.engine is None:
            if not self.connect():
                return []
        
        try:
            from sqlalchemy import inspect
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table)
            logger.info(f"表 {table} 有 {len(columns)} 列")
            return columns
        except SQLAlchemyError as e:
            logger.error(f"获取列信息失败：{str(e)}")
            return []
    
    def execute(self, sql: str, params: Optional[Dict] = None) -> int:
        """
        执行 SQL 语句 (INSERT/UPDATE/DELETE)
        
        Args:
            sql: SQL 语句
            params: 参数
        
        Returns:
            受影响的行数
        """
        if self.engine is None:
            if not self.connect():
                return 0
        
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text(sql), params or {})
                return result.rowcount
        except SQLAlchemyError as e:
            logger.error(f"SQL 执行失败：{str(e)}")
            return 0
    
    def save_query(self, name: str, sql: str, description: str = "") -> bool:
        """
        保存查询到配置文件
        
        Args:
            name: 查询名称
            sql: SQL 语句
            description: 查询描述
        
        Returns:
            是否保存成功
        """
        import json
        from pathlib import Path
        
        queries_file = Path(__file__).parent / "saved_queries.json"
        
        # 加载现有查询
        queries = {}
        if queries_file.exists():
            try:
                with open(queries_file, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
            except:
                queries = {}
        
        # 添加新查询
        queries[name] = {
            'sql': sql,
            'description': description,
            'db_type': self.db_type,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        # 保存
        try:
            with open(queries_file, 'w', encoding='utf-8') as f:
                json.dump(queries, f, ensure_ascii=False, indent=2)
            logger.info(f"查询已保存：{name}")
            return True
        except Exception as e:
            logger.error(f"保存查询失败：{str(e)}")
            return False
    
    def load_saved_queries(self) -> Dict[str, Dict]:
        """
        加载已保存的查询
        
        Returns:
            查询字典
        """
        import json
        from pathlib import Path
        
        queries_file = Path(__file__).parent / "saved_queries.json"
        
        if not queries_file.exists():
            return {}
        
        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def load_query(self, name: str) -> Optional[str]:
        """
        加载指定名称的查询
        
        Args:
            name: 查询名称
        
        Returns:
            SQL 语句或 None
        """
        queries = self.load_saved_queries()
        return queries.get(name, {}).get('sql')
    
    @property
    def connection_string(self) -> Optional[str]:
        """获取连接字符串 (隐藏密码)"""
        if not self._connection_string:
            return None
        
        # 隐藏密码
        if '@' in self._connection_string:
            parts = self._connection_string.split('@')
            if '://' in parts[0]:
                prefix, creds = parts[0].split('://')
                if ':' in creds:
                    user, _ = creds.split(':')
                    return f"{prefix}://{user}:***@{parts[1]}"
        
        return self._connection_string
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
