# Knowledge Layer - 知识检索模块
"""
知识检索层 - Phase 1 核心组件

包含:
- Neo4jClient: 图数据库连接
- FAISSClient: 向量数据库连接
- SchemaEmbedder: 表/字段 Embedding
- NL2SQLEngine: NL2SQL 引擎
"""

from .neo4j_client import Neo4jClient
from .faiss_client import FAISSClient

try:
    from .schema_embedder import SchemaEmbedder
except ModuleNotFoundError:
    SchemaEmbedder = None

try:
    from .nl2sql_engine import NL2SQLEngine
except ModuleNotFoundError:
    NL2SQLEngine = None

__all__ = [
    "Neo4jClient",
    "FAISSClient",
    "SchemaEmbedder",
    "NL2SQLEngine",
]
