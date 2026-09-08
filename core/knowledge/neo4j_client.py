"""
Neo4j 客户端 - 图数据库连接 (Graph RAG 数据地图)

用于存储和查询数据库表结构、字段信息、表关联关系、业务概念映射。

功能:
- Neo4j 数据库连接
- Database/Domain 业务域节点
- Table 表节点 CRUD
- Column 字段节点 CRUD (含样例值)
- BusinessConcept 业务概念节点
- 关系查询 (CONTAINS, HAS_COLUMN, FOREIGN_KEY_TO, MAPS_TO, UPSTREAM_OF)
- Schema 导出
"""

from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase, Driver, Session
import logging

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j 图数据库客户端"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化 Neo4j 客户端
        
        Args:
            uri: Neo4j URI (e.g., "bolt://localhost:7687")
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """建立 Neo4j 连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                database=self.database
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Neo4j 连接成功：{self.uri}")
        except Exception as e:
            logger.error(f"Neo4j 连接失败：{e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 连接已关闭")
    
    # ==================== Database/Domain 节点操作 ====================
    
    def create_database_node(self, name: str, description: str = "") -> bool:
        """
        创建数据库/业务域节点
        
        Args:
            name: 业务域名 (e.g., "banking", "risk_control")
            description: 业务域描述
            
        Returns:
            bool: 是否成功
        """
        cypher = """
        MERGE (db:Database {name: $name})
        SET db.description = $description,
            db.updated_at = datetime()
        RETURN db
        """
        try:
            with self.driver.session() as session:
                session.run(cypher, name=name, description=description)
            logger.debug(f"创建业务域节点：{name}")
            return True
        except Exception as e:
            logger.error(f"创建业务域节点失败：{e}")
            return False
    
    def list_databases(self) -> List[str]:
        """列出所有业务域"""
        cypher = """
        MATCH (db:Database)
        RETURN db.name AS name
        ORDER BY db.name
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                return [record["name"] for record in result]
        except Exception as e:
            logger.error(f"列出业务域失败：{e}")
            return []

    def _link_table_to_database(self, table_name: str, db_name: str) -> bool:
        """
        建立表到业务域的 CONTAINS 关系

        Args:
            table_name: 表名
            db_name: 业务域名

        Returns:
            bool: 是否成功
        """
        cypher = """
        MATCH (db:Database {name: $db_name})
        MATCH (t:Table {name: $table_name})
        MERGE (db)-[:CONTAINS]->(t)
        """
        try:
            with self.driver.session() as session:
                session.run(cypher, db_name=db_name, table_name=table_name)
            logger.debug(f"建立业务域关联：{db_name} CONTAINS {table_name}")
            return True
        except Exception as e:
            logger.error(f"建立业务域关联失败：{e}")
            return False
    
    # ==================== 表节点操作 ====================
    
    def create_table_node(self, name: str, description: str = "",
                          row_count: int = 0, update_freq: str = "",
                          database: str = "") -> bool:
        """
        创建表节点
        
        Args:
            name: 表名
            description: 表描述
            
        Returns:
            bool: 是否成功
        """
        cypher = """
        MERGE (t:Table {name: $name})
        SET t.description = $description,
            t.updated_at = datetime()
        RETURN t
        """
        try:
            with self.driver.session() as session:
                session.run(cypher, name=name, description=description)
            logger.debug(f"创建表节点：{name}")
            return True
        except Exception as e:
            logger.error(f"创建表节点失败：{e}")
            return False
    
    def get_table_node(self, name: str) -> Optional[Dict]:
        """获取表节点信息"""
        cypher = """
        MATCH (t:Table {name: $name})
        RETURN t.name AS name, t.description AS description, t.created_at AS created_at
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, name=name)
                record = result.single()
                if record:
                    return dict(record)
                return None
        except Exception as e:
            logger.error(f"获取表节点失败：{e}")
            return None
    
    def delete_table_node(self, name: str) -> bool:
        """删除表节点及其关系"""
        cypher = """
        MATCH (t:Table {name: $name})
        DETACH DELETE t
        """
        try:
            with self.driver.session() as session:
                session.run(cypher, name=name)
            logger.debug(f"删除表节点：{name}")
            return True
        except Exception as e:
            logger.error(f"删除表节点失败：{e}")
            return False
    
    def list_tables(self) -> List[str]:
        """列出所有表名"""
        cypher = """
        MATCH (t:Table)
        RETURN t.name AS name
        ORDER BY t.name
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                return [record["name"] for record in result]
        except Exception as e:
            logger.error(f"列出表名失败：{e}")
            return []
    
    # ==================== 字段节点操作 ====================
    
    def create_column_node(self, table: str, name: str, col_type: str = "",
                           description: str = "", is_primary: bool = False,
                           is_nullable: bool = True) -> bool:
        """
        创建字段节点
        
        Args:
            table: 所属表名
            name: 字段名
            col_type: 字段类型
            description: 字段描述
            is_primary: 是否主键
            is_nullable: 是否可空
            
        Returns:
            bool: 是否成功
        """
        cypher = """
        MATCH (t:Table {name: $table})
        MERGE (c:Column {table: $table, name: $name})
        SET c.type = $col_type,
            c.description = $description,
            c.is_primary = $is_primary,
            c.is_nullable = $is_nullable,
            c.updated_at = datetime()
        MERGE (t)-[:HAS_COLUMN]->(c)
        RETURN c
        """
        try:
            with self.driver.session() as session:
                session.run(
                    cypher,
                    table=table,
                    name=name,
                    col_type=col_type,
                    description=description,
                    is_primary=is_primary,
                    is_nullable=is_nullable
                )
            logger.debug(f"创建字段节点：{table}.{name}")
            return True
        except Exception as e:
            logger.error(f"创建字段节点失败：{e}")
            return False
    
    def get_columns(self, table: str) -> List[Dict]:
        """获取表的所有字段"""
        cypher = """
        MATCH (t:Table {name: $table})-[:HAS_COLUMN]->(c:Column)
        RETURN c.name AS name, c.type AS type, c.description AS description,
               c.is_primary AS is_primary, c.is_nullable AS is_nullable
        ORDER BY c.name
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, table=table)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"获取字段失败：{e}")
            return []
    
    # ==================== BusinessConcept 节点操作 ====================
    
    def create_business_concept(self, name: str, description: str = "") -> bool:
        """
        创建业务概念节点 (用于映射业务黑话)
        
        Args:
            name: 业务概念名 (e.g., "高价值客户", "完单金额")
            description: 业务含义描述
            
        Returns:
            bool: 是否成功
        """
        cypher = """
        MERGE (bc:BusinessConcept {name: $name})
        SET bc.description = $description,
            bc.updated_at = datetime()
        RETURN bc
        """
        try:
            with self.driver.session() as session:
                session.run(cypher, name=name, description=description)
            logger.debug(f"创建业务概念节点：{name}")
            return True
        except Exception as e:
            logger.error(f"创建业务概念节点失败：{e}")
            return False
    
    def map_business_concept(self, concept_name: str,
                              table_name: str, column_name: str,
                              rule: str = "") -> bool:
        """
        将业务概念映射到具体字段
        
        Args:
            concept_name: 业务概念名
            table_name: 表名
            column_name: 字段名
            rule: 映射规则 (e.g., "customer_level IN ['VIP', '黑金']")
            
        Returns:
            bool: 是否成功
        """
        cypher = """
        MATCH (bc:BusinessConcept {name: $concept_name})
        MATCH (c:Column {table: $table_name, name: $column_name})
        MERGE (bc)-[r:MAPS_TO]->(c)
        SET r.rule = $rule
        RETURN r
        """
        try:
            with self.driver.session() as session:
                session.run(
                    cypher,
                    concept_name=concept_name,
                    table_name=table_name,
                    column_name=column_name,
                    rule=rule
                )
            logger.debug(f"业务概念映射：{concept_name} -> {table_name}.{column_name}")
            return True
        except Exception as e:
            logger.error(f"业务概念映射失败：{e}")
            return False
    
    # ==================== 关系操作 ====================
    
    def create_foreign_key_to(self, from_table: str, from_col: str,
                               to_table: str, to_col: str,
                               join_type: str = "N:1") -> bool:
        """
        创建 FOREIGN_KEY_TO 关系 (用于自动 JOIN 推导)
        
        Args:
            from_table: 源表名
            from_col: 源字段名
            to_table: 目标表名
            to_col: 目标字段名
            join_type: 关联类型 (1:1, 1:N, N:1, M:N)
            
        Returns:
            bool: 是否成功
        """
        cypher = """
        MATCH (ft:Table {name: $from_table})-[:HAS_COLUMN]->(fc:Column {table: $from_table, name: $from_col})
        MATCH (tt:Table {name: $to_table})-[:HAS_COLUMN]->(tc:Column {table: $to_table, name: $to_col})
        MERGE (fc)-[r:FOREIGN_KEY_TO]->(tc)
        SET r.join_type = $join_type,
            r.created_at = datetime()
        RETURN r
        """
        try:
            with self.driver.session() as session:
                session.run(
                    cypher,
                    from_table=from_table,
                    from_col=from_col,
                    to_table=to_table,
                    to_col=to_col,
                    join_type=join_type
                )
            logger.debug(f"创建外键关系：{from_table}.{from_col} -> {to_table}.{to_col}")
            return True
        except Exception as e:
            logger.error(f"创建外键关系失败：{e}")
            return False
    
    def create_foreign_key(self, from_table: str, from_col: str,
                           to_table: str, to_col: str,
                           constraint_name: str = "") -> bool:
        """
        创建外键关系
        
        Args:
            from_table: 源表名
            from_col: 源字段名
            to_table: 目标表名
            to_col: 目标字段名
            constraint_name: 约束名称
            
        Returns:
            bool: 是否成功
        """
        cypher = """
        MATCH (ft:Table {name: $from_table})-[:HAS_COLUMN]->(fc:Column {table: $from_table, name: $from_col})
        MATCH (tt:Table {name: $to_table})-[:HAS_COLUMN]->(tc:Column {table: $to_table, name: $to_col})
        MERGE (fc)-[r:REFERENCES]->(tc)
        SET r.constraint_name = $constraint_name,
            r.created_at = datetime()
        RETURN r
        """
        try:
            with self.driver.session() as session:
                session.run(
                    cypher,
                    from_table=from_table,
                    from_col=from_col,
                    to_table=to_table,
                    to_col=to_col,
                    constraint_name=constraint_name
                )
            logger.debug(f"创建外键关系：{from_table}.{from_col} -> {to_table}.{to_col}")
            return True
        except Exception as e:
            logger.error(f"创建外键关系失败：{e}")
            return False
    
    def get_related_tables(self, table_name: str, depth: int = 2) -> List[Dict]:
        """
        获取相关表 (通过外键关系)
        
        Args:
            table_name: 表名
            depth: 查询深度
            
        Returns:
            List[Dict]: 相关表信息
        """
        cypher = """
        MATCH (start:Table {name: $table_name})
        MATCH (start)-[:HAS_COLUMN]-()-[:REFERENCES*1..$depth]-()-[:HAS_COLUMN]-(related:Table)
        RETURN DISTINCT related.name AS name,
               related.description AS description
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, table_name=table_name, depth=depth)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"获取相关表失败：{e}")
            return []
    
    def get_foreign_keys(self, table_name: str) -> List[Dict]:
        """获取表的所有外键"""
        cypher = """
        MATCH (t:Table {name: $table_name})-[:HAS_COLUMN]->(c:Column)-[r:REFERENCES]->(rc:Column)<-[:HAS_COLUMN]-(rt:Table)
        RETURN t.name AS from_table, c.name AS from_column,
               rt.name AS to_table, rc.name AS to_column,
               r.constraint_name AS constraint_name
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, table_name=table_name)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"获取外键失败：{e}")
            return []
    
    # ==================== Schema 导出 ====================
    
    def export_schema(self) -> Dict[str, Any]:
        """
        导出完整 Schema
        
        Returns:
            Dict: 包含所有表、字段、关系的字典
        """
        schema = {
            "tables": {},
            "relationships": []
        }
        
        # 导出所有表
        for table_name in self.list_tables():
            table_info = self.get_table_node(table_name)
            columns = self.get_columns(table_name)
            foreign_keys = self.get_foreign_keys(table_name)
            
            schema["tables"][table_name] = {
                "description": table_info.get("description", "") if table_info else "",
                "columns": columns,
                "foreign_keys": foreign_keys
            }
        
        return schema
    
    def get_table_schema_text(self, table_name: str) -> str:
        """
        获取表的 Schema 文本描述 (用于 LLM 提示词)
        
        Args:
            table_name: 表名
            
        Returns:
            str: Schema 文本描述
        """
        table_info = self.get_table_node(table_name)
        columns = self.get_columns(table_name)
        foreign_keys = self.get_foreign_keys(table_name)
        
        if not table_info:
            return f"表 '{table_name}' 不存在"
        
        lines = [f"表名：{table_name}"]
        if table_info.get("description"):
            lines.append(f"描述：{table_info['description']}")
        
        lines.append("\n字段:")
        for col in columns:
            pk = " [主键]" if col.get("is_primary") else ""
            nullable = " [可空]" if col.get("is_nullable") else " [非空]"
            lines.append(f"  - {col['name']}: {col['type']}{pk}{nullable}")
            if col.get("description"):
                lines.append(f"    描述：{col['description']}")
        
        if foreign_keys:
            lines.append("\n外键:")
            for fk in foreign_keys:
                lines.append(f"  - {fk['from_column']} -> {fk['to_table']}.{fk['to_column']}")
        
        return "\n".join(lines)
    
    def clear_all(self):
        """清空所有数据 (慎用!)"""
        cypher = "MATCH (n) DETACH DELETE n"
        try:
            with self.driver.session() as session:
                session.run(cypher)
            logger.warning("已清空 Neo4j 所有数据")
        except Exception as e:
            logger.error(f"清空数据失败：{e}")
            raise
