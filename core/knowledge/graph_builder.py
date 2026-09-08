"""
知识图谱构建引擎 - Graph Builder

从关系型数据库自动提取元数据，构建 Neo4j 知识图谱。

支持数据库:
- MySQL / PostgreSQL
- Oracle
- Hive / ClickHouse
- SQL Server

核心流程：Extract (抽取) → Vectorize (向量化) → Load (加载入图)

设计模式:
- 策略模式 (Strategy Pattern): 不同数据库的元数据提取策略
- 抽象工厂 (Abstract Factory): 根据配置创建对应提取器
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import logging
import re

from .neo4j_client import Neo4jClient
from .schema_embedder import SchemaEmbedder
from .faiss_client import FAISSClient

logger = logging.getLogger(__name__)

_IDENTIFIER_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _require_identifier_path(value: str) -> str:
    """Validate identifiers used by dialects that cannot bind object names."""
    if not _IDENTIFIER_PATH.fullmatch(str(value)):
        raise ValueError(f"非法数据库标识符：{value!r}")
    return str(value)


def _quote_mysql_identifier(value: str) -> str:
    """Quote a MySQL identifier returned by trusted metadata queries."""
    return f"`{str(value).replace('`', '``')}`"


# ==================== 基础接口定义 ====================

class BaseMetadataExtractor(ABC):
    """
    元数据提取器基础接口
    
    所有数据库提取器必须实现此接口，返回统一格式的数据
    """
    
    def __init__(self, config: Dict):
        """
        初始化提取器
        
        Args:
            config: 数据库配置
                {
                    "db_type": "mysql",  # mysql/oracle/hive/postgresql/sqlserver
                    "host": "localhost",
                    "port": 3306,
                    "user": "root",
                    "password": "password",
                    "database": "mydb"
                }
        """
        self.config = config
        self.db_name = config.get("database", "")
        self.db_type = config.get("db_type", "mysql")
        self.connection = None
    
    @abstractmethod
    def connect(self) -> bool:
        """建立数据库连接"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """关闭数据库连接"""
        pass
    
    @abstractmethod
    def extract_tables(self) -> List[Dict]:
        """
        提取表元数据
        
        Returns:
            List[Dict]: [
                {
                    "table_name": "user_info",
                    "comment": "用户信息表",
                    "row_count": 100000,
                    "engine": "InnoDB"  # MySQL 特有
                },
                ...
            ]
        """
        pass
    
    @abstractmethod
    def extract_columns(self, table_name: Optional[str] = None) -> List[Dict]:
        """
        提取字段元数据
        
        Args:
            table_name: 表名 (可选，None 表示提取所有表)
            
        Returns:
            List[Dict]: [
                {
                    "table_name": "user_info",
                    "column_name": "user_id",
                    "data_type": "BIGINT",
                    "is_primary": True,
                    "is_nullable": False,
                    "comment": "用户 ID",
                    "default_value": None,
                    "sample_value": "1001"  # 可选，抽样获取
                },
                ...
            ]
        """
        pass
    
    @abstractmethod
    def extract_foreign_keys(self) -> List[Dict]:
        """
        提取外键关系
        
        Returns:
            List[Dict]: [
                {
                    "table": "orders",
                    "column": "user_id",
                    "ref_table": "user_info",
                    "ref_column": "user_id",
                    "constraint_name": "fk_orders_user",
                    "join_type": "N:1"
                },
                ...
            ]
        """
        pass
    
    def extract_all(self) -> Dict[str, Any]:
        """
        提取全部元数据
        
        Returns:
            Dict: 包含 tables, columns, foreign_keys 的字典
        """
        logger.info(f"开始提取 {self.db_type} 数据库 '{self.db_name}' 元数据...")
        
        tables = self.extract_tables()
        logger.info(f"提取到 {len(tables)} 个表")
        
        columns = self.extract_columns()
        logger.info(f"提取到 {len(columns)} 个字段")
        
        foreign_keys = self.extract_foreign_keys()
        logger.info(f"提取到 {len(foreign_keys)} 个外键关系")
        
        return {
            "tables": tables,
            "columns": columns,
            "foreign_keys": foreign_keys
        }


# ==================== MySQL 提取器 ====================

class MySQLMetadataExtractor(BaseMetadataExtractor):
    """MySQL 元数据提取器"""
    
    def connect(self) -> bool:
        try:
            import pymysql
            
            self.connection = pymysql.connect(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 3306),
                user=self.config.get("user", "root"),
                password=self.config.get("password", ""),
                database=self.db_name,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor
            )
            logger.info(f"MySQL 连接成功：{self.db_name}")
            return True
        except Exception as e:
            logger.error(f"MySQL 连接失败：{e}")
            return False
    
    def disconnect(self):
        if self.connection:
            self.connection.close()
            logger.info("MySQL 连接已关闭")
    
    def _query(self, sql: str, params: Optional[tuple] = None) -> List[Dict]:
        """执行 SQL 查询"""
        with self.connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()
    
    def extract_tables(self) -> List[Dict]:
        sql = """
        SELECT 
            TABLE_NAME as table_name,
            TABLE_COMMENT as comment,
            TABLE_ROWS as row_count,
            ENGINE as engine
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = %s
        """
        tables = self._query(sql, (self.db_name,))
        
        return [
            {
                "table_name": t["table_name"],
                "comment": t["comment"] or "",
                "row_count": t["row_count"] or 0,
                "engine": t["engine"] or "InnoDB"
            }
            for t in tables
        ]
    
    def extract_columns(self, table_name: Optional[str] = None) -> List[Dict]:
        sql = """
        SELECT 
            TABLE_NAME as table_name,
            COLUMN_NAME as column_name,
            DATA_TYPE as data_type,
            COLUMN_TYPE as column_type,
            COLUMN_COMMENT as comment,
            IS_NULLABLE as is_nullable,
            COLUMN_KEY as column_key,
            COLUMN_DEFAULT as default_value
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        params = (self.db_name,)
        if table_name:
            sql = """
            SELECT
                TABLE_NAME as table_name,
                COLUMN_NAME as column_name,
                DATA_TYPE as data_type,
                COLUMN_TYPE as column_type,
                COLUMN_COMMENT as comment,
                IS_NULLABLE as is_nullable,
                COLUMN_KEY as column_key,
                COLUMN_DEFAULT as default_value
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """
            params = (self.db_name, table_name)
        columns = self._query(sql, params)
        
        result = []
        for col in columns:
            result.append({
                "table_name": col["table_name"],
                "column_name": col["column_name"],
                "data_type": col["data_type"].upper(),
                "is_primary": col["column_key"] == "PRI",
                "is_nullable": col["is_nullable"] == "YES",
                "comment": col["comment"] or "",
                "default_value": col["default_value"],
                "sample_value": self._get_sample_value(col["table_name"], col["column_name"])
            })
        
        return result
    
    def _get_sample_value(self, table_name: str, column_name: str) -> Optional[str]:
        """获取字段样例值 (可选，抽样 3 条取第一条)"""
        try:
            quoted_column = _quote_mysql_identifier(column_name)
            quoted_table = _quote_mysql_identifier(table_name)
            sql = f"SELECT {quoted_column} FROM {quoted_table} LIMIT 3"  # nosec B608
            result = self._query(sql)
            if result:
                return str(result[0][column_name])
        except Exception as exc:
            logger.debug("获取字段样例失败：%s.%s: %s", table_name, column_name, exc)
        return None
    
    def extract_foreign_keys(self) -> List[Dict]:
        sql = """
        SELECT 
            TABLE_NAME as table,
            COLUMN_NAME as column,
            REFERENCED_TABLE_NAME as ref_table,
            REFERENCED_COLUMN_NAME as ref_column,
            CONSTRAINT_NAME as constraint_name
        FROM information_schema.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = %s
        AND REFERENCED_TABLE_NAME IS NOT NULL
        """
        fks = self._query(sql, (self.db_name,))
        
        return [
            {
                "table": fk["table"],
                "column": fk["column"],
                "ref_table": fk["ref_table"],
                "ref_column": fk["ref_column"],
                "constraint_name": fk["constraint_name"],
                "join_type": "N:1"  # 默认假设
            }
            for fk in fks
        ]


# ==================== Oracle 提取器 ====================

class OracleMetadataExtractor(BaseMetadataExtractor):
    """Oracle 元数据提取器"""
    
    def connect(self) -> bool:
        try:
            import cx_Oracle
            
            dsn = cx_Oracle.makedsn(
                self.config.get("host", "localhost"),
                self.config.get("port", 1521),
                service_name=self.db_name
            )
            
            self.connection = cx_Oracle.connect(
                user=self.config.get("user", "system"),
                password=self.config.get("password", ""),
                dsn=dsn
            )
            logger.info(f"Oracle 连接成功：{self.db_name}")
            return True
        except Exception as e:
            logger.error(f"Oracle 连接失败：{e}")
            return False
    
    def disconnect(self):
        if self.connection:
            self.connection.close()
            logger.info("Oracle 连接已关闭")
    
    def _query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """执行 SQL 查询"""
        cursor = self.connection.cursor()
        cursor.execute(sql, params or {})
        columns = [col[0] for col in cursor.description]
        results = []
        for row in cursor:
            results.append(dict(zip(columns, row)))
        cursor.close()
        return results
    
    def extract_tables(self) -> List[Dict]:
        sql = """
        SELECT t.TABLE_NAME as table_name, c.COMMENTS as comment
        FROM USER_TABLES t
        LEFT JOIN USER_TAB_COMMENTS c ON t.TABLE_NAME = c.TABLE_NAME
        """
        
        tables = self._query(sql)
        
        return [
            {
                "table_name": t["TABLE_NAME"],
                "comment": t["COMMENT"] or "",
                "row_count": 0,  # Oracle 需要单独查询
                "engine": "Oracle"
            }
            for t in tables
        ]
    
    def extract_columns(self, table_name: Optional[str] = None) -> List[Dict]:
        sql = """
        SELECT 
            t.TABLE_NAME as table_name,
            t.COLUMN_NAME as column_name,
            t.DATA_TYPE as data_type,
            c.COMMENTS as comment,
            t.NULLABLE as is_nullable,
            t.DATA_DEFAULT as default_value
        FROM USER_TAB_COLUMNS t
        LEFT JOIN USER_COL_COMMENTS c 
            ON t.TABLE_NAME = c.TABLE_NAME AND t.COLUMN_NAME = c.COLUMN_NAME
        ORDER BY t.TABLE_NAME, t.COLUMN_ID
        """
        params = None
        if table_name:
            sql = """
            SELECT
                t.TABLE_NAME as table_name,
                t.COLUMN_NAME as column_name,
                t.DATA_TYPE as data_type,
                c.COMMENTS as comment,
                t.NULLABLE as is_nullable,
                t.DATA_DEFAULT as default_value
            FROM USER_TAB_COLUMNS t
            LEFT JOIN USER_COL_COMMENTS c
                ON t.TABLE_NAME = c.TABLE_NAME AND t.COLUMN_NAME = c.COLUMN_NAME
            WHERE t.TABLE_NAME = :table_name
            ORDER BY t.TABLE_NAME, t.COLUMN_ID
            """
            params = {"table_name": table_name}
        columns = self._query(sql, params)
        
        # 查询主键信息
        pk_sql = """
        SELECT cols.TABLE_NAME, cols.COLUMN_NAME
        FROM USER_CONST_COLUMNS cols
        JOIN USER_CONSTRAINTS cons 
            ON cols.CONSTRAINT_NAME = cons.CONSTRAINT_NAME
        WHERE cons.CONSTRAINT_TYPE = 'P'
        """
        pk_cols = {(r["TABLE_NAME"], r["COLUMN_NAME"]) for r in self._query(pk_sql)}
        
        result = []
        for col in columns:
            is_pk = (col["TABLE_NAME"], col["COLUMN_NAME"]) in pk_cols
            
            result.append({
                "table_name": col["TABLE_NAME"],
                "column_name": col["COLUMN_NAME"],
                "data_type": col["DATA_TYPE"],
                "is_primary": is_pk,
                "is_nullable": col["IS_NULLABLE"] == "Y",
                "comment": col["COMMENT"] or "",
                "default_value": col["DATA_DEFAULT"],
                "sample_value": None  # Oracle 抽样需要额外查询
            })
        
        return result
    
    def extract_foreign_keys(self) -> List[Dict]:
        sql = """
        SELECT 
            a.table_name as table,
            a.column_name as column,
            c.table_name as ref_table,
            c.column_name as ref_col,
            b.constraint_name as constraint_name
        FROM user_cons_columns a
        JOIN user_constraints b ON a.constraint_name = b.constraint_name
        JOIN user_cons_columns c ON b.r_constraint_name = c.constraint_name
        WHERE b.constraint_type = 'R'
        """
        
        fks = self._query(sql)
        
        return [
            {
                "table": fk["TABLE"],
                "column": fk["COLUMN"],
                "ref_table": fk["REF_TABLE"],
                "ref_column": fk["REF_COL"],
                "constraint_name": fk["CONSTRAINT_NAME"],
                "join_type": "N:1"
            }
            for fk in fks
        ]


# ==================== Hive 提取器 ====================

class HiveMetadataExtractor(BaseMetadataExtractor):
    """
    Hive 元数据提取器
    
    特殊处理:
    - Hive 通常没有物理外键，需要推断同名字段
    - 通过 HiveServer2 或 Metastore 获取元数据
    """
    
    def connect(self) -> bool:
        try:
            from pyhive import hive
            
            self.connection = hive.connect(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 10000),
                database=self.db_name,
                auth=self.config.get("auth", "NOSASL")
            )
            logger.info(f"Hive 连接成功：{self.db_name}")
            return True
        except Exception as e:
            logger.error(f"Hive 连接失败：{e}")
            return False
    
    def disconnect(self):
        if self.connection:
            self.connection.close()
            logger.info("Hive 连接已关闭")
    
    def _query(self, sql: str) -> List[Dict]:
        """执行 SQL 查询"""
        import pandas as pd
        df = pd.read_sql(sql, self.connection)
        return df.to_dict("records")
    
    def extract_tables(self) -> List[Dict]:
        # 方法 1: SHOW TABLES + DESCRIBE FORMATTED (较慢但通用)
        tables_result = self._query("SHOW TABLES")
        
        result = []
        for row in tables_result:
            table_name = _require_identifier_path(str(list(row.values())[0]))
            
            # 获取表信息
            desc = self._query(f"DESCRIBE FORMATTED {table_name}")  # nosec B608
            
            # 解析描述获取注释
            comment = ""
            row_count = 0
            for item in desc:
                key = str(list(item.values())[0]).strip()
                value = str(list(item.values())[1]).strip() if len(item) > 1 else ""
                if "Comment:" in key:
                    comment = value
                elif "numRows:" in key:
                    try:
                        row_count = int(value)
                    except (TypeError, ValueError):
                        logger.debug("Hive 表 %s 的 numRows 无法解析：%r", table_name, value)
            
            result.append({
                "table_name": table_name,
                "comment": comment,
                "row_count": row_count,
                "engine": "Hive"
            })
        
        return result
    
    def extract_columns(self, table_name: Optional[str] = None) -> List[Dict]:
        if not table_name:
            # 提取所有表的字段
            all_columns = []
            tables = self.extract_tables()
            for t in tables:
                columns = self._extract_columns_for_table(t["table_name"])
                all_columns.extend(columns)
            return all_columns
        else:
            return self._extract_columns_for_table(table_name)
    
    def _extract_columns_for_table(self, table_name: str) -> List[Dict]:
        """提取单个表的字段"""
        table_name = _require_identifier_path(table_name)
        desc = self._query(f"DESCRIBE {table_name}")  # nosec B608
        
        result = []
        for row in desc:
            parts = str(list(row.values())[0]).strip().split("\t")
            if len(parts) >= 1 and parts[0]:  # 跳过空行
                col_name = parts[0]
                data_type = parts[1] if len(parts) > 1 else "STRING"
                comment = parts[2] if len(parts) > 2 else ""
                
                # Hive 没有主键概念
                result.append({
                    "table_name": table_name,
                    "column_name": col_name,
                    "data_type": data_type.upper(),
                    "is_primary": False,
                    "is_nullable": True,
                    "comment": comment,
                    "default_value": None,
                    "sample_value": None
                })
        
        return result
    
    def extract_foreign_keys(self) -> List[Dict]:
        """
        Hive 没有物理外键，使用启发式规则打分算法推断软外键
        
        算法策略:
        1. 降噪过滤：剔除审计字段 (create_time, status 等)
        2. 特征匹配与打分：
           - 策略 A (0.9 分): 字段名完全一致 + 数据类型完全一致
           - 策略 B (0.8 分): 主键包含匹配 (driver_info.id ↔ orders.driver_id)
           - 策略 C (0.6 分): 血缘后缀匹配 (_id, _no, _code)
        3. 阈值截断：score >= 0.7 才注册为 POSSIBLE_JOIN
        4. LLM 语义校验 (可选): 二次校验减少假阳性
        """
        columns = self.extract_columns()
        
        # 使用软外键推断引擎
        inferencer = HiveSoftForeignKeyInferencer(
            use_llm_validation=False  # 可选 LLM 校验
        )
        
        inferred_fks = inferencer.infer_foreign_keys(columns)
        
        # 转换为标准格式
        result = []
        for fk in inferred_fks:
            result.append({
                "table": fk["source_table"],
                "column": fk["source_col"],
                "ref_table": fk["target_table"],
                "ref_column": fk["target_col"],
                "constraint_name": f"soft_{fk['source_col']}_{fk['confidence']:.1f}",
                "join_type": "N:1",
                "inferred": True,
                "confidence": fk["confidence"],
                "relation_type": "POSSIBLE_JOIN"  # 区分物理外键
            })
        
        logger.info(f"Hive 推断出 {len(inferred_fks)} 个潜在外键关系 (置信度 >= 0.7)")
        return result


# ==================== Hive 软外键推断引擎 ====================

class HiveSoftForeignKeyInferencer:
    """
    Hive 软外键推断引擎
    
    采用启发式规则打分 + 业务黑白名单算法，模拟资深数据工程师"找表"的直觉
    
    核心策略:
    1. 降噪过滤 (Noise Reduction)
    2. 特征匹配与打分 (Matching & Scoring)
    3. 阈值截断 (Thresholding)
    4. LLM 语义校验 (可选，Agentic 增强)
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 use_llm_validation: bool = False,
                 llm_provider=None):
        """
        初始化推断引擎
        
        Args:
            confidence_threshold: 置信度阈值 (默认 0.7)
            use_llm_validation: 是否使用 LLM 二次校验
            llm_provider: LLM 提供者 (用于语义校验)
        """
        self.confidence_threshold = confidence_threshold
        self.use_llm_validation = use_llm_validation
        self.llm_provider = llm_provider
        
        # 业务黑名单：绝对不可能作为关联键的通用字段
        self.stop_words = {
            'id', 'create_time', 'update_time', 'dt', 'partition_date',
            'is_deleted', 'status', 'remark', 'type', 'version', 'creator',
            'modifier', 'modified_time', 'created_time', 'ext_info', 'json',
            'comment', 'memo', 'note', 'flag', 'is_valid', 'deleted'
        }
        
        # 业务主键特征词 (高权重)
        self.key_suffixes = ('_id', '_no', '_code', '_sn', '_key', '_uuid')
        
        # 典型业务实体名 (用于策略 B 匹配)
        self.entity_keywords = (
            'user', 'customer', 'member', 'account',
            'order', 'trade', 'deal',
            'product', 'item', 'goods',
            'driver', 'passenger', 'merchant',
            'city', 'region', 'area', 'district',
            'time', 'date', 'month', 'year'
        )
        
        # 类型兼容组
        self.numeric_types = {'int', 'integer', 'bigint', 'smallint', 'tinyint', 'float', 'double', 'decimal', 'number'}
        self.string_types = {'string', 'varchar', 'char', 'text', 'nvarchar'}
        self.datetime_types = {'timestamp', 'datetime', 'date', 'time'}
    
    def _is_valid_candidate(self, col_name: str, data_type: str) -> bool:
        """
        判断一个字段是否有资格作为外键候选人
        
        过滤规则:
        1. 剔除黑名单字段 (create_time, status 等)
        2. 必须包含特定后缀 (_id, _no, _code 等)
        3. 剔除不适合 JOIN 的类型 (float, boolean, tinyint)
        """
        col_name_lower = col_name.lower()
        data_type_lower = data_type.lower()
        
        # 1. 剔除黑名单
        if col_name_lower in self.stop_words:
            return False
        
        # 2. 必须包含特定的后缀，或者本身就是典型的业务实体名
        has_key_suffix = col_name_lower.endswith(self.key_suffixes)
        is_entity_name = any(kw in col_name_lower for kw in self.entity_keywords)
        
        if not (has_key_suffix or is_entity_name):
            return False
        
        # 3. 剔除浮点型、布尔型等不适合做 JOIN 的类型
        if data_type_lower in ('float', 'double', 'boolean', 'tinyint', 'blob', 'json'):
            return False
        
        return True
    
    def _get_type_group(self, data_type: str) -> str:
        """获取类型所属组 (用于兼容性判断)"""
        data_type_lower = data_type.lower()
        
        if data_type_lower in self.numeric_types:
            return 'numeric'
        elif data_type_lower in self.string_types:
            return 'string'
        elif data_type_lower in self.datetime_types:
            return 'datetime'
        else:
            return 'unknown'
    
    def _types_compatible(self, type1: str, type2: str) -> tuple:
        """
        检查两个类型是否兼容
        
        Returns:
            (is_compatible: bool, score: float)
        """
        type1_lower = type1.lower()
        type2_lower = type2.lower()
        
        # 完全相同：高分
        if type1_lower == type2_lower:
            return True, 1.0
        
        # 同组兼容：中分
        group1 = self._get_type_group(type1)
        group2 = self._get_type_group(type2)
        
        if group1 == group2 and group1 != 'unknown':
            return True, 0.7
        
        # 跨组兼容 (如 string 和 bigint): 低分
        if group1 in ('numeric', 'string') and group2 in ('numeric', 'string'):
            return True, 0.5
        
        return False, 0.0
    
    def _calculate_score(self, col1: Dict, col2: Dict) -> float:
        """
        计算两个字段的匹配得分
        
        策略:
        - 策略 A (0.9 分): 字段名完全一致 + 数据类型完全一致
        - 策略 B (0.8 分): 主键包含匹配 (driver_info.id ↔ orders.driver_id)
        - 策略 C (0.6 分): 血缘后缀匹配 (_id, _no, _code)
        """
        score = 0.0
        
        # 字段名完全一致
        if col1["column_name"].lower() == col2["column_name"].lower():
            # 策略 A: 名称一致 + 类型一致
            type_compatible, type_score = self._types_compatible(
                col1["data_type"],
                col2["data_type"]
            )
            
            if type_score >= 1.0:
                score = 0.9  # 完全匹配
            elif type_score >= 0.7:
                score = 0.8  # 同组兼容
            else:
                score = 0.6  # 跨组兼容
        else:
            # 策略 B: 主键包含匹配
            # 例如：driver_info.id ↔ orders.driver_id
            table1 = col1["table_name"].lower()
            table2 = col2["table_name"].lower()
            col1_name = col1["column_name"].lower()
            col2_name = col2["column_name"].lower()
            
            # 检查是否一个表名包含另一个字段名
            # 例如：table1="driver_info", col2_name="driver_id"
            table1_prefix = table1.replace("_", "").replace("info", "").replace("dim", "").replace("ods", "")
            table2_prefix = table2.replace("_", "").replace("info", "").replace("dim", "").replace("ods", "")
            
            if (table1_prefix in col2_name.replace("_", "") or 
                table2_prefix in col1_name.replace("_", "")):
                score = 0.75  # 策略 B
            
            # 策略 C: 都有典型后缀
            elif (col1_name.endswith(self.key_suffixes) and 
                  col2_name.endswith(self.key_suffixes)):
                score = 0.6
        
        return score
    
    def infer_foreign_keys(self, columns_data: List[Dict]) -> List[Dict]:
        """
        基于全量字段信息，计算软外键
        
        Args:
            columns_data: List of dict
                [{'table_name': 'x', 'column_name': 'y', 'data_type': 'z', ...}, ...]
        
        Returns:
            List[Dict]: 推断的外键关系列表
        """
        import itertools
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("开始执行 Hive 软外键推断算法...")
        
        # 1. 过滤并构建倒排索引 (Inverted Index)
        # 结构：{ 'driver_id': [('table_a', 'bigint', col_dict), ('table_b', 'bigint', col_dict)], ... }
        inverted_index = {}
        
        for col in columns_data:
            t_name = col["table_name"]
            c_name = col["column_name"].lower()
            d_type = col["data_type"].lower()
            
            if self._is_valid_candidate(c_name, d_type):
                if c_name not in inverted_index:
                    inverted_index[c_name] = []
                inverted_index[c_name].append((t_name, d_type, col))
        
        inferred_fks = []
        processed_pairs = set()  # 避免重复处理
        
        # 2. 基于倒排索引计算笛卡尔积组合，生成关联边
        for col_name, tables in inverted_index.items():
            # 如果这个字段只在一张表里出现，无法关联
            if len(tables) < 2:
                continue
            
            # 在拥有同名字段的表之间，两两组合建立联系
            for (t1, type1, col1), (t2, type2, col2) in itertools.combinations(tables, 2):
                # 避免自关联
                if t1 == t2:
                    continue
                
                # 避免重复处理 (t1.col ↔ t2.col 和 t2.col ↔ t1.col 是同一个关系)
                pair_key = tuple(sorted([f"{t1}.{col_name}", f"{t2}.{col_name}"]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                # 计算得分
                score = self._calculate_score(col1, col2)
                
                # 3. 阈值截断
                if score >= self.confidence_threshold:
                    # 4. LLM 语义校验 (可选)
                    if self.use_llm_validation and self.llm_provider:
                        is_valid = self._llm_validate(col1, col2, score)
                        if not is_valid:
                            logger.debug(f"LLM 校验拒绝：{t1}.{col_name} ↔ {t2}.{col_name}")
                            continue
                    
                    inferred_fks.append({
                        "source_table": t1,
                        "source_col": col1["column_name"],
                        "target_table": t2,
                        "target_col": col2["column_name"],
                        "confidence": score,
                        "relation_type": "POSSIBLE_JOIN",
                        "inference_strategy": "heuristic_scoring"
                    })
        
        logger.info(f"推断完成，共发现 {len(inferred_fks)} 条潜在关联边 (置信度 >= {self.confidence_threshold})")
        return inferred_fks
    
    def _llm_validate(self, col1: Dict, col2: Dict, score: float) -> bool:
        """
        使用 LLM 进行语义校验 (Agentic 增强)
        
        提取相连字段所在表的"表注释"，让 LLM 判断关联可能性
        
        Returns:
            bool: 是否通过校验
        """
        if not self.llm_provider:
            return True
        
        try:
            # 构建校验 Prompt
            prompt = f"""
你是一个数据仓库专家。请基于业务常识判断以下两个表是否可能发生 JOIN 关联：

表 A【{col1.get("table_name", "未知")}】
- 字段：{col1.get("column_name", "未知")}
- 类型：{col1.get("data_type", "未知")}
- 注释：{col1.get("comment", "无")}

表 B【{col2.get("table_name", "未知")}】
- 字段：{col2.get("column_name", "未知")}
- 类型：{col2.get("data_type", "未知")}
- 注释：{col2.get("comment", "无")}

当前推断置信度：{score:.2f}

问题：这两个表在实际数据分析中，发生 JOIN 关联的可能性高吗？
请只回复 True 或 False，不要解释。
"""
            
            # 调用 LLM
            response = self.llm_provider.generate(prompt)
            response_text = response.strip().lower()
            
            # 解析响应
            is_valid = response_text.startswith('true') or response_text.startswith('是')
            return is_valid
            
        except Exception as e:
            logger.error(f"LLM 校验失败：{e}")
            return True  # 失败时默认通过，避免误杀


# ==================== 提取器工厂 ====================

class ExtractorFactory:
    """
    元数据提取器工厂
    
    根据数据库类型创建对应的提取器
    """
    
    _registry = {
        "mysql": MySQLMetadataExtractor,
        "oracle": OracleMetadataExtractor,
        "hive": HiveMetadataExtractor,
        # 可以继续扩展
        # "postgresql": PostgreSQLMetadataExtractor,
        # "sqlserver": SQLServerMetadataExtractor,
        # "clickhouse": ClickHouseMetadataExtractor,
    }
    
    @classmethod
    def register(cls, db_type: str, extractor_class):
        """注册新的提取器"""
        cls._registry[db_type.lower()] = extractor_class
    
    @classmethod
    def get_extractor(cls, config: Dict) -> BaseMetadataExtractor:
        """
        根据配置获取提取器
        
        Args:
            config: 数据库配置，必须包含 db_type
            
        Returns:
            BaseMetadataExtractor: 提取器实例
        """
        db_type = config.get("db_type", "mysql").lower()
        
        if db_type not in cls._registry:
            raise ValueError(
                f"不支持的数据库类型：{db_type}\n"
                f"支持的类型：{', '.join(cls._registry.keys())}"
            )
        
        extractor_class = cls._registry[db_type]
        return extractor_class(config)
    
    @classmethod
    def list_supported_databases(cls) -> List[str]:
        """列出支持的数据库类型"""
        return list(cls._registry.keys())


# ==================== 知识图谱构建引擎 ====================

class KnowledgeGraphBuilder:
    """
    知识图谱构建引擎
    
    从任意关系型数据库提取元数据，构建 Neo4j 知识图谱
    """
    
    def __init__(self, source_config: Dict, neo4j_config: Dict,
                 embedder: Optional[SchemaEmbedder] = None,
                 faiss_client: Optional[FAISSClient] = None):
        """
        初始化图谱构建引擎
        
        Args:
            source_config: 源数据库配置
            neo4j_config: Neo4j 配置
            embedder: Schema Embedder (可选)
            faiss_client: FAISS 客户端 (可选)
        """
        # 1. 创建元数据提取器
        self.extractor = ExtractorFactory.get_extractor(source_config)
        self.db_name = source_config.get("database", "unknown")
        
        # 2. 创建 Neo4j 客户端
        self.neo4j = Neo4jClient(
            uri=neo4j_config.get("uri", "bolt://localhost:7687"),
            user=neo4j_config.get("user", "neo4j"),
            password=neo4j_config.get("password", "neo4j")
        )
        
        # 3. Embedding 和 FAISS (可选)
        self.embedder = embedder or SchemaEmbedder()
        self.faiss = faiss_client
        
        logger.info(f"KnowledgeGraphBuilder 初始化完成，源数据库：{self.db_name}")
    
    def run_pipeline(self, create_domain: bool = True,
                     create_embeddings: bool = True,
                     save_faiss: bool = True) -> Dict[str, int]:
        """
        运行完整的图谱构建流程
        
        Args:
            create_domain: 是否创建 Database 域节点
            create_embeddings: 是否生成 Embedding
            save_faiss: 是否保存 FAISS 索引
            
        Returns:
            Dict: 构建统计信息
        """
        logger.info(f"开始构建知识图谱：{self.db_name}")
        
        # Step 1: 提取元数据
        logger.info("Step 1: 提取元数据...")
        metadata = self.extractor.extract_all()
        
        # Step 2: 构建 Neo4j 图谱
        logger.info("Step 2: 构建 Neo4j 图谱...")
        stats = self._build_neo4j_graph(
            metadata["tables"],
            metadata["columns"],
            metadata["foreign_keys"],
            create_domain=create_domain
        )
        
        # Step 3: 生成 Embedding 并保存到 FAISS
        if create_embeddings:
            logger.info("Step 3: 生成 Embedding...")
            self._build_faiss_index(metadata, save_faiss=save_faiss)
        
        logger.info(f"知识图谱构建完成：{stats}")
        return stats
    
    def _build_neo4j_graph(self, tables: List[Dict], columns: List[Dict],
                           foreign_keys: List[Dict],
                           create_domain: bool = True) -> Dict[str, int]:
        """
        构建 Neo4j 图谱
        
        Returns:
            Dict: 节点和关系数量统计
        """
        stats = {
            "databases": 0,
            "tables": 0,
            "columns": 0,
            "business_concepts": 0,
            "relationships": 0
        }
        
        # 1. 创建 Database 域节点
        if create_domain:
            self.neo4j.create_database_node(
                name=self.db_name,
                description=f"{self.extractor.db_type} 数据库"
            )
            stats["databases"] = 1
        
        # 2. 创建表节点
        for table in tables:
            self.neo4j.create_table_node(
                name=table["table_name"],
                description=table.get("comment", ""),
                row_count=table.get("row_count", 0),
                update_freq="",
                database=self.db_name if create_domain else ""
            )
            
            # 建立 CONTAINS 关系
            if create_domain:
                self.neo4j._link_table_to_database(
                    table["table_name"],
                    self.db_name
                )
                stats["relationships"] += 1
            
            stats["tables"] += 1
        
        # 3. 创建字段节点
        for col in columns:
            self.neo4j.create_column_node(
                table=col["table_name"],
                name=col["column_name"],
                col_type=col.get("data_type", ""),
                description=col.get("comment", ""),
                is_primary=col.get("is_primary", False),
                is_nullable=col.get("is_nullable", True)
            )
            
            # 建立 HAS_COLUMN 关系
            # (在 create_column_node 中已自动创建)
            stats["columns"] += 1
        
        # 4. 创建外键关系
        for fk in foreign_keys:
            self.neo4j.create_foreign_key_to(
                from_table=fk["table"],
                from_col=fk["column"],
                to_table=fk["ref_table"],
                to_col=fk["ref_column"],
                join_type=fk.get("join_type", "N:1")
            )
            stats["relationships"] += 1
        
        return stats
    
    def _build_faiss_index(self, metadata: Dict, save_faiss: bool = True):
        """
        生成 Embedding 并构建 FAISS 索引
        
        Args:
            metadata: 元数据字典
            save_faiss: 是否保存到文件
        """
        # 准备 Schema 数据
        tables_dict = {}
        for table in metadata["tables"]:
            table_name = table["table_name"]
            tables_dict[table_name] = {
                "table_name": table_name,
                "description": table.get("comment", ""),
                "columns": [
                    {
                        "name": col["column_name"],
                        "type": col.get("data_type", ""),
                        "description": col.get("comment", "")
                    }
                    for col in metadata["columns"]
                    if col["table_name"] == table_name
                ]
            }
        
        tables_list = list(tables_dict.values())
        
        # 批量生成 Embedding
        embeddings_dict = self.embedder.embed_schema_batch(
            tables_list,
            show_progress=True
        )
        
        # 如果提供了 FAISS 客户端，添加到索引
        if self.faiss:
            import numpy as np
            
            keys = list(embeddings_dict.keys())
            embeddings = np.array(list(embeddings_dict.values()))
            
            metadata_list = []
            for key in keys:
                parts = key.split(":")
                if parts[0] == "table":
                    metadata_list.append({
                        "key": key,
                        "type": "table",
                        "table_name": parts[1],
                        "database": self.db_name
                    })
                elif parts[0] == "column":
                    table_col = parts[1].split(".")
                    metadata_list.append({
                        "key": key,
                        "type": "column",
                        "table_name": table_col[0],
                        "column_name": table_col[1],
                        "database": self.db_name
                    })
            
            self.faiss.add_embeddings(embeddings, metadata_list)
            
            # 保存索引
            if save_faiss:
                index_path = f"faiss_index_{self.db_name}.index"
                self.faiss.save(index_path)
                logger.info(f"FAISS 索引已保存：{index_path}")


# ==================== 便捷函数 ====================

def build_knowledge_graph(source_config: Dict, neo4j_config: Dict,
                          create_embeddings: bool = True,
                          faiss_index_path: Optional[str] = None) -> Dict[str, int]:
    """
    便捷函数：一键构建知识图谱
    
    Args:
        source_config: 源数据库配置
        neo4j_config: Neo4j 配置
        create_embeddings: 是否生成 Embedding
        faiss_index_path: FAISS 索引保存路径
        
    Returns:
        Dict: 构建统计信息
    """
    # 创建 Embedder
    embedder = SchemaEmbedder()
    
    # 创建 FAISS 客户端
    faiss = None
    if create_embeddings:
        faiss = FAISSClient(dimension=embedder.get_dimension())
    
    # 创建构建引擎
    builder = KnowledgeGraphBuilder(
        source_config=source_config,
        neo4j_config=neo4j_config,
        embedder=embedder,
        faiss_client=faiss
    )
    
    # 运行流程
    stats = builder.run_pipeline(
        create_domain=True,
        create_embeddings=create_embeddings,
        save_faiss=faiss_index_path is not None
    )
    
    return stats
