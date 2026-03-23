"""
NL2SQL 引擎 - 自然语言转 SQL (Agentic Workflow)

将用户自然语言查询转换为 SQL 语句并执行，返回 DataFrame。

核心特性:
- 强制 JSON 输出 + 思维链 (Chain of Thought)
- is_executable 安全阻断机制
- 方言适配 (MySQL/Hive/PostgreSQL 等)
- XML 标签结构化上下文
- 防幻觉严格约束

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
        
        # DataRetrieval_Agent 系统提示词模板 (强制 JSON 输出 + 思维链)
        self.prompt_template = """
# Role
你是一个顶级的数据库架构师和高级数据开发工程师（DataRetrieval_Agent）。你的核心任务是根据用户提供的业务需求，结合图数据库检索出的"局部表结构与血缘关联"，编写出精准、高效、无语法的 {dialect} SQL 查询语句，用于为机器学习提取训练数据集。

# Context (来自图谱检索的知识)
以下是经过精准召回的数据库表结构、字段说明以及表之间的关联关系（Graph Schema）：
<graph_schema>
{graph_schema}
</graph_schema>

# User Request
用户的业务需求是：
<user_query>
{user_query}
</user_query>

# ⚠️ Strict Rules (严格约束)
为了保证生产环境的稳定，你必须绝对遵守以下规则，任何违背将导致系统崩溃：
1. 【禁止幻觉】你只能使用 `<graph_schema>` 中明确提供的表名和字段名。严禁凭空捏造、猜测任何未提及的字段或表！
2. 【严格关联】如果需要多表查询，必须严格按照 `<graph_schema>` 中提供的关联关系（如 A.id = B.id）进行 JOIN 操作。
3. 【方言适配】你编写的 SQL 必须完全符合 {dialect} 数据库的语法规范（例如日期处理、字符串拼接等函数的差异）。
4. 【精准选取】禁止使用 `SELECT *`。请根据用户需求，明确 SELECT 出所需的特征字段和标签字段，并使用 AS 赋予易读的英文别名（如果原字段名不够清晰）。
5. 【安全阻断】如果用户的需求无法用 `<graph_schema>` 中的表和字段实现（即缺失关键数据），你必须拒绝生成 SQL，并在 thought_process 中说明缺失了什么。

# Output Format (输出格式)
你必须且只能输出一个合法的 JSON 对象，不要包含任何 Markdown 标记（如 ```json），不要有任何额外的寒暄。JSON 结构如下：
{{
 "thought_process": "简要描述你的思考过程：你选择了哪几张表？你使用了什么字段做 JOIN？你如何处理了用户的过滤条件？如果数据不足以满足需求，请在这里说明缺失了什么概念。",
 "is_executable": true, // 布尔值。如果数据充足可以生成完整 SQL 则为 true；如果缺失关键表/字段无法生成则为 false。
 "sql": "具体的 SQL 语句。如果 is_executable 为 false，此字段请返回 null。"
}}
"""
    
    def query(self, natural_language: str, 
              dialect: str = "mysql",
              top_k_tables: int = 5,
              execute: bool = True) -> Dict[str, Any]:
        """
        自然语言查询 (Agentic Workflow)
        
        Args:
            natural_language: 用户自然语言查询
            dialect: SQL 方言 ("mysql", "postgresql", "hive", "oracle")
            top_k_tables: 检索 Top-K 个相关表
            execute: 是否执行 SQL
            
        Returns:
            Dict[str, Any]: {
                "thought_process": str,
                "is_executable": bool,
                "sql": str or None,
                "dataframe": pd.DataFrame or None,
                "error": str or None
            }
        """
        logger.info(f"NL2SQL 查询：{natural_language}")
        result = {
            "thought_process": "",
            "is_executable": False,
            "sql": None,
            "dataframe": None,
            "error": None
        }
        
        try:
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
            graph_schema = self._build_graph_schema(list(related_tables))
            
            # 4. 构建提示词
            prompt = self.prompt_template.format(
                dialect=dialect,
                graph_schema=graph_schema,
                user_query=natural_language
            )
            
            # 5. 调用 LLM 生成 SQL (JSON 格式)
            llm_response = self._call_llm(prompt)
            
            # 6. 解析 JSON 输出
            agent_result = self._parse_json_response(llm_response)
            
            if not agent_result:
                result["error"] = "LLM 返回格式错误，无法解析 JSON"
                return result
            
            result["thought_process"] = agent_result.get("thought_process", "")
            result["is_executable"] = agent_result.get("is_executable", False)
            result["sql"] = agent_result.get("sql")
            
            logger.info(f"Agent 思考：{result['thought_process']}")
            logger.info(f"SQL 可执行：{result['is_executable']}")
            
            # 7. 安全阻断检查
            if not result["is_executable"]:
                logger.warning(f"数据不足，无法生成 SQL: {result['thought_process']}")
                result["error"] = f"数据不足：{result['thought_process']}"
                return result
            
            # 8. SQLFluff 验证
            if result["sql"]:
                is_valid = self._validate_sql(result["sql"], dialect)
                if not is_valid:
                    logger.warning("SQL 验证失败，尝试修复...")
                    result["sql"] = self._fix_sql(result["sql"], dialect)
                
                # 9. 执行 SQL (可选)
                if execute:
                    result["dataframe"] = self._execute_sql(result["sql"])
            
            return result
            
        except Exception as e:
            logger.error(f"NL2SQL 查询失败：{e}")
            result["error"] = str(e)
            return result
    
    def _build_graph_schema(self, table_names: List[str]) -> str:
        """
        构建图谱 Schema 文本 (用于 Prompt 上下文)
        
        Args:
            table_names: 表名列表
            
        Returns:
            str: 结构化的 Schema 文本
        """
        lines = []
        
        for table_name in table_names:
            # 获取表信息
            table_info = self.neo4j.get_table_node(table_name)
            columns = self.neo4j.get_columns(table_name)
            foreign_keys = self.neo4j.get_foreign_keys(table_name)
            
            if not table_info:
                continue
            
            # 表头
            lines.append(f"- 表名：{table_name}")
            if table_info.get("description"):
                lines.append(f"  描述：{table_info['description']}")
            
            # 字段列表
            if columns:
                col_str = ", ".join([f"{c['name']} ({c.get('type', 'UNKNOWN')})" for c in columns])
                lines.append(f"  字段：{col_str}")
            
            # 外键关系
            if foreign_keys:
                lines.append("  关联关系 (JOINs):")
                for fk in foreign_keys:
                    lines.append(f"    {fk['from_table']}.{fk['from_column']} = {fk['to_table']}.{fk['to_column']}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_schema_context(self, table_names: List[str]) -> str:
        """
        构建 Schema 上下文 (兼容旧接口)
        
        Args:
            table_names: 表名列表
            
        Returns:
            str: Schema 文本描述
        """
        return self._build_graph_schema(table_names)
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """
        解析 LLM 的 JSON 响应
        
        Args:
            response: LLM 返回的文本
            
        Returns:
            Optional[Dict]: 解析后的字典，失败返回 None
        """
        import json
        import re
        
        try:
            # 清理 Markdown 标记
            text = response.strip()
            text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^```\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
            
            # 尝试直接解析
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # 尝试提取 JSON 块
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    return json.loads(match.group())
                else:
                    logger.error(f"无法从响应中提取 JSON: {response[:200]}")
                    return None
                    
        except Exception as e:
            logger.error(f"JSON 解析失败：{e}")
            return None
    
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
