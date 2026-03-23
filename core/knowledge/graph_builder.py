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

from .neo4j_client import Neo4jClient
from .schema_embedder import SchemaEmbedder
from .faiss_client import FAISSClient

logger = logging.getLogger(__name__)


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
    
    def _query(self, sql: str) -> List[Dict]:
        """执行 SQL 查询"""
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
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
        """ % self.db_name
        
        # 注意：实际使用时应该用参数化查询，这里简化演示
        tables = self._query(sql)
        
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
        table_filter = f"AND TABLE_NAME = '{table_name}'" if table_name else ""
        
        sql = f"""
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
        WHERE TABLE_SCHEMA = '{self.db_name}'
        {table_filter}
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        
        columns = self._query(sql)
        
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
            sql = f"SELECT `{column_name}` FROM `{table_name}` LIMIT 3"
            result = self._query(sql)
            if result:
                return str(result[0][column_name])
        except:
            pass
        return None
    
    def extract_foreign_keys(self) -> List[Dict]:
        sql = f"""
        SELECT 
            TABLE_NAME as table,
            COLUMN_NAME as column,
            REFERENCED_TABLE_NAME as ref_table,
            REFERENCED_COLUMN_NAME as ref_column,
            CONSTRAINT_NAME as constraint_name
        FROM information_schema.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = '{self.db_name}'
        AND REFERENCED_TABLE_NAME IS NOT NULL
        """
        
        fks = self._query(sql)
        
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
    
    def _query(self, sql: str) -> List[Dict]:
        """执行 SQL 查询"""
        cursor = self.connection.cursor()
        cursor.execute(sql)
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
        table_filter = f"WHERE t.TABLE_NAME = '{table_name}'" if table_name else ""
        
        sql = f"""
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
        {table_filter}
        ORDER BY t.TABLE_NAME, t.COLUMN_ID
        """
        
        columns = self._query(sql)
        
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
            table_name = list(row.values())[0]
            
            # 获取表信息
            desc = self._query(f"DESCRIBE FORMATTED {table_name}")
            
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
                    except:
                        pass
            
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
        desc = self._query(f"DESCRIBE {table_name}")
        
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
        Hive 没有物理外键，使用同名字段推断算法
        
        推断规则:
        1. 两张表都有相同名称的字段
        2. 字段类型相同或兼容
        3. 字段名包含常见关联键 (id, user_id, order_id 等)
        """
        columns = self.extract_columns()
        
        # 按字段名分组
        from collections import defaultdict
        col_groups = defaultdict(list)
        
        for col in columns:
            col_name = col["column_name"].lower()
            # 只考虑可能是外键的字段
            if any(kw in col_name for kw in ["_id", "id_", "code", "_no", "no_"]):
                col_groups[col_name].append(col)
        
        # 生成推断的外键关系
        inferred_fks = []
        
        for col_name, cols in col_groups.items():
            if len(cols) > 1:
                # 同一字段名出现在多张表中
                for i, col1 in enumerate(cols):
                    for col2 in cols[i+1:]:
                        # 避免自关联
                        if col1["table_name"] != col2["table_name"]:
                            # 类型兼容检查
                            if self._types_compatible(col1["data_type"], col2["data_type"]):
                                inferred_fks.append({
                                    "table": col1["table_name"],
                                    "column": col1["column_name"],
                                    "ref_table": col2["table_name"],
                                    "ref_column": col2["column_name"],
                                    "constraint_name": f"inferred_{col_name}",
                                    "join_type": "N:1",
                                    "inferred": True  # 标记为推断的外键
                                })
        
        logger.info(f"Hive 推断出 {len(inferred_fks)} 个潜在外键关系")
        return inferred_fks
    
    def _types_compatible(self, type1: str, type2: str) -> bool:
        """检查两个类型是否兼容"""
        type1 = type1.upper()
        type2 = type2.upper()
        
        # 完全相同
        if type1 == type2:
            return True
        
        # 数值类型兼容
        numeric_types = {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "FLOAT", "DOUBLE", "DECIMAL"}
        if type1 in numeric_types and type2 in numeric_types:
            return True
        
        # 字符串类型兼容
        string_types = {"STRING", "VARCHAR", "CHAR", "TEXT"}
        if type1 in string_types and type2 in string_types:
            return True
        
        return False


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
