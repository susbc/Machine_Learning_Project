"""
Data Retrieval Agent - 数据检索智能体

Phase 1 核心智能体，负责接收用户自然语言查询，检索数据并返回结果。

功能:
- 统一接口
- 错误处理
- 日志记录
- 性能监控
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging
import time
from datetime import datetime

from core.knowledge import Neo4jClient, FAISSClient, SchemaEmbedder, NL2SQLEngine

logger = logging.getLogger(__name__)


class DataRetrievalAgent:
    """数据检索智能体"""
    
    def __init__(self, config: Dict):
        """
        初始化数据检索智能体
        
        Args:
            config: 配置字典
                {
                    "neo4j": {
                        "uri": "bolt://localhost:7687",
                        "user": "neo4j",
                        "password": "password"
                    },
                    "faiss": {
                        "dimension": 384,
                        "index_path": "path/to/index"
                    },
                    "embedder": {
                        "model": "mini"
                    },
                    "database": {
                        "db_type": "mysql",
                        "host": "localhost",
                        "port": 3306,
                        "user": "root",
                        "password": "password",
                        "database": "mydb"
                    },
                    "llm": {
                        "provider": "qwen",
                        "api_key": "xxx"
                    }
                }
        """
        self.config = config
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0.0
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化各组件"""
        try:
            # 1. Neo4j 客户端
            neo4j_config = self.config.get("neo4j", {})
            self.neo4j = Neo4jClient(
                uri=neo4j_config.get("uri", "bolt://localhost:7687"),
                user=neo4j_config.get("user", "neo4j"),
                password=neo4j_config.get("password", "neo4j")
            )
            
            # 2. Schema Embedder
            embedder_config = self.config.get("embedder", {})
            self.embedder = SchemaEmbedder(
                model_name=embedder_config.get("model", "mini")
            )
            
            # 3. FAISS 客户端
            faiss_config = self.config.get("faiss", {})
            dimension = faiss_config.get("dimension", self.embedder.get_dimension())
            self.faiss = FAISSClient(dimension=dimension)
            
            # 加载已有索引
            index_path = faiss_config.get("index_path")
            if index_path:
                try:
                    self.faiss.load(index_path)
                except FileNotFoundError:
                    logger.warning(f"FAISS 索引文件不存在：{index_path}")
            
            # 4. LLM 提供者 (可选)
            self.llm_provider = self._initialize_llm(self.config.get("llm", {}))
            
            # 5. NL2SQL 引擎
            database_config = self.config.get("database", {})
            self.nl2sql = NL2SQLEngine(
                neo4j_client=self.neo4j,
                faiss_client=self.faiss,
                embedder=self.embedder,
                llm_provider=self.llm_provider,
                database_config=database_config
            )
            
            logger.info("DataRetrievalAgent 初始化完成")
        except Exception as e:
            logger.error(f"初始化组件失败：{e}")
            raise
    
    def _initialize_llm(self, llm_config: Dict):
        """
        初始化 LLM 提供者
        
        根据配置创建相应的 LLM 实例
        """
        provider = llm_config.get("provider", "")
        api_key = llm_config.get("api_key", "")
        
        if not provider or not api_key:
            logger.warning("未配置 LLM，将使用演示模式")
            return None
        
        # 根据 provider 创建相应实例
        # 这里可以集成项目现有的 LLM 模块
        try:
            from model_code.llm import get_intelligent_analyzer
            analyzer = get_intelligent_analyzer()
            # 包装为简单的 LLM 接口
            class LLMWrapper:
                def __init__(self, analyzer):
                    self.analyzer = analyzer
                
                def generate(self, prompt: str) -> str:
                    # 使用 AI 分析器的某个方法生成文本
                    return self.analyzer.analyze_dataset.__doc__ or ""
            
            return LLMWrapper(analyzer)
        except Exception as e:
            logger.error(f"LLM 初始化失败：{e}")
            return None
    
    def retrieve(self, query: str, execute: bool = True) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        数据检索主接口
        
        Args:
            query: 自然语言查询
            execute: 是否执行 SQL
            
        Returns:
            Tuple[str, Optional[pd.DataFrame]]: (SQL 语句，查询结果)
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        try:
            logger.info(f"收到查询：{query}")
            
            # 调用 NL2SQL 引擎
            sql, result_df = self.nl2sql.query(query, execute=execute)
            
            # 更新统计
            elapsed = time.time() - start_time
            self.stats["successful_queries"] += 1
            self._update_response_time(elapsed)
            
            logger.info(f"查询成功，耗时：{elapsed:.2f}s")
            
            return sql, result_df
            
        except Exception as e:
            self.stats["failed_queries"] += 1
            logger.error(f"查询失败：{e}")
            raise
    
    def _update_response_time(self, elapsed: float):
        """更新平均响应时间"""
        total = self.stats["successful_queries"]
        avg = self.stats["avg_response_time"]
        self.stats["avg_response_time"] = (avg * (total - 1) + elapsed) / total
    
    def explain_schema(self, table_name: Optional[str] = None) -> str:
        """
        返回数据库 Schema 说明
        
        Args:
            table_name: 表名 (可选)
            
        Returns:
            str: Schema 说明
        """
        return self.nl2sql.explain_schema(table_name)
    
    def sync_schema_from_database(self, db_config: Dict):
        """
        从数据库同步 Schema 到 Neo4j 和 FAISS
        
        Args:
            db_config: 数据库配置
        """
        try:
            from model_code.database import DatabaseConnector
            
            # 连接数据库
            db = DatabaseConnector(db_config)
            db.connect()
            
            # 获取所有表
            tables = db.get_table_names()
            
            # 同步到 Neo4j
            for table in tables:
                # 创建表节点
                self.neo4j.create_table_node(table, f"表：{table}")
                
                # 获取字段信息
                columns = db.get_columns(table)
                for col in columns:
                    self.neo4j.create_column_node(
                        table=table,
                        name=col["name"],
                        col_type=col["type"],
                        description=col.get("comment", ""),
                        is_primary=col.get("is_primary", False),
                        is_nullable=col.get("is_nullable", True)
                    )
                
                # 获取外键
                foreign_keys = db.get_foreign_keys(table)
                for fk in foreign_keys:
                    self.neo4j.create_foreign_key(
                        from_table=table,
                        from_col=fk["column"],
                        to_table=fk["ref_table"],
                        to_col=fk["ref_column"]
                    )
            
            # 生成 Embedding 并保存到 FAISS
            schema_data = self.neo4j.export_schema()
            tables_list = [
                {
                    "table_name": name,
                    "description": info.get("description", ""),
                    "columns": [
                        {
                            "name": col["name"],
                            "type": col["type"],
                            "description": col.get("description", "")
                        }
                        for col in info.get("columns", [])
                    ]
                }
                for name, info in schema_data["tables"].items()
            ]
            
            # 批量生成 Embedding
            embeddings_dict = self.embedder.embed_schema_batch(tables_list, show_progress=True)
            
            # 添加到 FAISS
            keys = list(embeddings_dict.keys())
            embeddings = list(embeddings_dict.values())
            
            metadata = []
            for key in keys:
                parts = key.split(":")
                if parts[0] == "table":
                    metadata.append({"key": key, "type": "table", "table_name": parts[1]})
                elif parts[0] == "column":
                    table_col = parts[1].split(".")
                    metadata.append({
                        "key": key,
                        "type": "column",
                        "table_name": table_col[0],
                        "column_name": table_col[1]
                    })
            
            import numpy as np
            self.faiss.add_embeddings(np.array(embeddings), metadata)
            
            # 保存 FAISS 索引
            index_path = self.config.get("faiss", {}).get("index_path")
            if index_path:
                self.faiss.save(index_path)
            
            db.disconnect()
            
            logger.info(f"Schema 同步完成：{len(tables)} 个表")
            
        except Exception as e:
            logger.error(f"Schema 同步失败：{e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        faiss_stats = self.faiss.get_stats()
        return {
            **self.stats,
            "faiss": faiss_stats,
            "neo4j_tables": len(self.neo4j.list_tables())
        }
    
    def close(self):
        """关闭连接"""
        try:
            self.neo4j.close()
            logger.info("DataRetrievalAgent 已关闭")
        except Exception as e:
            logger.error(f"关闭时出错：{e}")
