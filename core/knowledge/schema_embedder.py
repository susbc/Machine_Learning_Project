"""
Schema Embedder - 表/字段 Embedding 生成

使用 sentence-transformers 生成表名和字段的向量表示，用于 FAISS 相似度搜索。

功能:
- 使用 sentence-transformers 生成 Embedding
- 表名 Embedding
- 字段名 Embedding
- 字段业务含义 Embedding
- 批量嵌入
"""

from typing import List, Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class SchemaEmbedder:
    """Schema Embedding 生成器"""
    
    # 预定义模型列表 (按性能排序)
    MODEL_REGISTRY = {
        "mini": "all-MiniLM-L6-v2",      # 快速，384 维
        "base": "all-mpnet-base-v2",      # 平衡，768 维
        "large": "multi-qa-mpnet-base-dot-v1",  # 高精度，768 维
    }
    
    def __init__(self, model_name: str = "mini"):
        """
        初始化 Schema Embedder
        
        Args:
            model_name: 模型名称或预定义别名 ("mini", "base", "large")
        """
        # 解析模型名称
        if model_name in self.MODEL_REGISTRY:
            model_name = self.MODEL_REGISTRY[model_name]
        
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()
    
    def _load_model(self):
        """加载 sentence-transformers 模型"""
        try:
            logger.info(f"加载 Embedding 模型：{self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"模型加载完成，维度：{self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"模型加载失败：{e}")
            raise
    
    def get_dimension(self) -> int:
        """获取 Embedding 维度"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 0
    
    def embed_table_name(self, table_name: str) -> np.ndarray:
        """
        生成表名 Embedding
        
        Args:
            table_name: 表名 (e.g., "bank_customer")
            
        Returns:
            np.ndarray: Embedding 向量
        """
        # 添加上下文前缀，提高语义理解
        text = f"Database table: {table_name}"
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding
    
    def embed_column(self, table_name: str, column_name: str,
                     description: str = "", col_type: str = "") -> np.ndarray:
        """
        生成字段 Embedding
        
        Args:
            table_name: 所属表名
            column_name: 字段名
            description: 字段描述 (中文业务含义)
            col_type: 字段类型
            
        Returns:
            np.ndarray: Embedding 向量
        """
        # 构建丰富的文本描述，包含更多信息
        parts = [f"Database column: {column_name}"]
        parts.append(f"from table: {table_name}")
        
        if col_type:
            parts.append(f"type: {col_type}")
        
        if description:
            parts.append(f"description: {description}")
        
        text = ". ".join(parts)
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        生成任意文本的 Embedding
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: Embedding 向量
        """
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding
    
    def embed_batch(self, texts: List[str], 
                    batch_size: int = 32,
                    show_progress: bool = False) -> np.ndarray:
        """
        批量生成 Embedding
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            show_progress: 是否显示进度条
            
        Returns:
            np.ndarray: Embedding 矩阵 (n_samples, dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def embed_schema_batch(self, tables: List[Dict],
                           show_progress: bool = False) -> Dict[str, np.ndarray]:
        """
        批量生成 Schema Embedding
        
        Args:
            tables: 表信息列表
                [
                    {
                        "table_name": "bank_customer",
                        "description": "客户信息表",
                        "columns": [
                            {"name": "customer_id", "type": "VARCHAR", "description": "客户 ID"},
                            ...
                        ]
                    },
                    ...
                ]
            show_progress: 是否显示进度条
            
        Returns:
            Dict[str, np.ndarray]: {
                "table:bank_customer": embedding,
                "column:bank_customer.customer_id": embedding,
                ...
            }
        """
        texts = []
        keys = []
        
        # 收集所有需要 Embedding 的文本
        for table in tables:
            table_name = table["table_name"]
            
            # 表名 Embedding
            texts.append(f"Database table: {table_name}. {table.get('description', '')}")
            keys.append(f"table:{table_name}")
            
            # 字段 Embedding
            for col in table.get("columns", []):
                col_text = f"Database column: {col['name']}. from table: {table_name}"
                if col.get("type"):
                    col_text += f". type: {col['type']}"
                if col.get("description"):
                    col_text += f". description: {col['description']}"
                
                texts.append(col_text)
                keys.append(f"column:{table_name}.{col['name']}")
        
        # 批量生成 Embedding
        logger.info(f"批量生成 {len(texts)} 个 Embedding...")
        embeddings = self.embed_batch(texts, show_progress=show_progress)
        
        # 构建结果字典
        result = {key: emb for key, emb in zip(keys, embeddings)}
        logger.info(f"Embedding 生成完成：{len(result)} 个")
        
        return result
