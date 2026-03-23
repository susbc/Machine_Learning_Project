"""
NL2SQL 引擎 - 自然语言转 SQL

将用户自然语言查询转换为 SQL 语句并执行，返回 DataFrame。

功能:
- 自然语言解析
- 检索 Neo4j 获取表关系
- 检索 FAISS 获取相关字段
- 构建提示词调用 LLM 生成 SQL
- SQL 验证 (SQLFluff)
- 执行 SQL 返回 DataFrame
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging
import re

from .neo4j_client import Neo4jClient
from .faiss_client import FAISSClient
from .schema_embedder import SchemaEmbedder

logger = logging.getLogger(__name__)


class NL2SQLEngine:
    """NL2SQL 引擎"""
    
    def __init__(self, neo4j_client: Neo4jClient,
                 faiss_client: FAISSClient,
                 embedder: SchemaEmbedder,
                 llm_provider=None,
                 database_config: Optional[Dict] = None):
        """
        初始化 NL2SQL 引擎
        
        Args:
            neo4j_client: Neo4j 客户端
            faiss_client: FAISS 客户端
            embedder: Schema Embedder
            llm_provider: LLM 提供者 (可选，默认使用项目现有 LLM)
            database_config: 目标数据库配置 (用于执行 SQL)
        """
        self.neo4j = neo4j_client
        self.faiss = faiss_client
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.database_config = database_config or {}
        
        # SQL 生成提示词模板
        self.prompt_template = """
你是一个 SQL 专家。根据给定的数据库 Schema，将用户的自然语言查询转换为 SQL 语句。

## 数据库 Schema

{schema_context}

## 查询要求

1. 只输出 SQL 语句，不要解释
2. 使用 {dialect} 语法
3. 确保表名和字段名正确
4. 使用合适的 JOIN 连接相关表
5. 如果查询模糊，选择最合理的解释

## 用户查询

{natural_language}

## SQL
"""
    
    def query(self, natural_language: str, 
              dialect: str = "mysql",
              top_k_tables: int = 5,
              execute: bool = True) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        自然语言查询
        
        Args:
            natural_language: 用户自然语言查询
            dialect: SQL 方言 ("mysql", "postgresql", "sqlite")
            top_k_tables: 检索 Top-K 个相关表
            execute: 是否执行 SQL
            
        Returns:
            Tuple[str, Optional[pd.DataFrame]]: (SQL 语句，执行结果)
        """
        logger.info(f"NL2SQL 查询：{natural_language}")
        
        # 1. 生成查询 Embedding
        query_embedding = self.embedder.embed_text(natural_language)
        
        # 2. FAISS 检索相关字段
        faiss_results = self.faiss.search(query_embedding, top_k=top_k_tables * 3)
        related_tables = set()
        for metadata, _ in faiss_results:
            if "table_name" in metadata:
                related_tables.add(metadata["table_name"])
        
        logger.debug(f"检索到相关表：{related_tables}")
        
        # 3. Neo4j 检索表关系和 Schema
        schema_context = self._build_schema_context(list(related_tables))
        
        # 4. 构建提示词
        prompt = self.prompt_template.format(
            schema_context=schema_context,
            dialect=dialect,
            natural_language=natural_language
        )
        
        # 5. 调用 LLM 生成 SQL
        sql = self._call_llm(prompt)
        sql = self._extract_sql(sql)
        
        logger.info(f"生成的 SQL: {sql}")
        
        # 6. SQLFluff 验证
        is_valid = self._validate_sql(sql, dialect)
        if not is_valid:
            logger.warning("SQL 验证失败，尝试修复...")
            sql = self._fix_sql(sql, dialect)
        
        # 7. 执行 SQL (可选)
        result_df = None
        if execute and sql:
            result_df = self._execute_sql(sql)
        
        return sql, result_df
    
    def _build_schema_context(self, table_names: List[str]) -> str:
        """
        构建 Schema 上下文
        
        Args:
            table_names: 表名列表
            
        Returns:
            str: Schema 文本描述
        """
        lines = []
        
        for table_name in table_names:
            # 获取表 Schema 文本
            schema_text = self.neo4j.get_table_schema_text(table_name)
            lines.append(schema_text)
            
            # 获取相关表
            related = self.neo4j.get_related_tables(table_name, depth=1)
            if related:
                related_names = [r["name"] for r in related]
                lines.append(f"\n相关表：{', '.join(related_names)}")
        
        return "\n\n".join(lines)
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用 LLM 生成 SQL
        
        Args:
            prompt: 提示词
            
        Returns:
            str: LLM 响应
        """
        # 如果未配置 LLM，使用简单的规则生成 (演示用)
        if not self.llm_provider:
            logger.warning("未配置 LLM，使用演示模式")
            return self._demo_sql_generation(prompt)
        
        # 调用实际 LLM
        try:
            response = self.llm_provider.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM 调用失败：{e}")
            return ""
    
    def _demo_sql_generation(self, prompt: str) -> str:
        """
        演示用 SQL 生成 (无 LLM 时的降级方案)
        
        实际使用时会被 LLM 调用替代
        """
        # 简单提取表名，生成 SELECT *
        import re
        match = re.search(r"表名：(\w+)", prompt)
        if match:
            table = match.group(1)
            return f"SELECT * FROM {table} LIMIT 10;"
        return ""
    
    def _extract_sql(self, text: str) -> str:
        """
        从 LLM 响应中提取 SQL
        
        Args:
            text: LLM 响应文本
            
        Returns:
            str: 纯 SQL 语句
        """
        # 移除 Markdown 代码块标记
        text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        
        # 提取第一条 SQL 语句
        lines = text.strip().split("\n")
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("--") and not line.startswith("#"):
                sql_lines.append(line)
        
        sql = " ".join(sql_lines)
        
        # 移除末尾分号 (可选)
        sql = sql.rstrip(";")
        
        return sql.strip()
    
    def _validate_sql(self, sql: str, dialect: str = "mysql") -> bool:
        """
        使用 SQLFluff 验证 SQL
        
        Args:
            sql: SQL 语句
            dialect: SQL 方言
            
        Returns:
            bool: 是否有效
        """
        try:
            import sqlfluff
            
            # Lint SQL
            result = sqlfluff.lint(sql, dialect=dialect)
            
            # 检查严重错误
            for violation in result:
                if violation.get("code") in ["PRS", "LXR"]:  # 解析/词法错误
                    logger.warning(f"SQL 语法错误：{violation.get('description')}")
                    return False
            
            return True
        except ImportError:
            logger.warning("SQLFluff 未安装，跳过验证")
            return True
        except Exception as e:
            logger.error(f"SQL 验证失败：{e}")
            return False
    
    def _fix_sql(self, sql: str, dialect: str) -> str:
        """
        尝试修复 SQL
        
        Args:
            sql: SQL 语句
            dialect: SQL 方言
            
        Returns:
            str: 修复后的 SQL
        """
        try:
            import sqlfluff
            
            # 使用 SQLFluff 修复
            fixed = sqlfluff.fix(sql, dialect=dialect)
            logger.info(f"SQL 已修复：{fixed}")
            return fixed
        except Exception as e:
            logger.error(f"SQL 修复失败：{e}")
            return sql
    
    def _execute_sql(self, sql: str) -> Optional[pd.DataFrame]:
        """
        执行 SQL 并返回 DataFrame
        
        Args:
            sql: SQL 语句
            
        Returns:
            pd.DataFrame: 查询结果
        """
        # 如果有数据库配置，执行实际查询
        if self.database_config:
            try:
                from model_code.database import DatabaseConnector
                db = DatabaseConnector(self.database_config)
                db.connect()
                df = db.query(sql)
                db.disconnect()
                return df
            except Exception as e:
                logger.error(f"SQL 执行失败：{e}")
                return None
        
        # 演示模式：返回空 DataFrame
        logger.warning("未配置数据库，返回空 DataFrame")
        return pd.DataFrame()
    
    def explain_schema(self, table_name: Optional[str] = None) -> str:
        """
        返回数据库 Schema 说明
        
        Args:
            table_name: 表名 (可选，默认返回所有表)
            
        Returns:
            str: Schema 说明
        """
        if table_name:
            return self.neo4j.get_table_schema_text(table_name)
        
        # 返回所有表
        tables = self.neo4j.list_tables()
        lines = ["# 数据库 Schema\n"]
        
        for table in tables:
            lines.append(self.neo4j.get_table_schema_text(table))
            lines.append("")
        
        return "\n".join(lines)
