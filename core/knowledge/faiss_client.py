"""
FAISS 客户端 - 向量数据库连接

用于存储和搜索表/字段的 Embedding 向量，支持相似度检索。

功能:
- FAISS 向量索引创建
- 字段 Embedding 存储
- 相似度搜索 (Top-K)
- 索引持久化 (保存/加载)
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import faiss
import pickle
import os
import logging

logger = logging.getLogger(__name__)


class FAISSClient:
    """FAISS 向量数据库客户端"""
    
    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """
        初始化 FAISS 客户端
        
        Args:
            dimension: Embedding 维度 (默认 384，对应 all-MiniLM-L6-v2)
            index_type: 索引类型 ("flat", "ivf", "hnsw")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []  # 存储元数据 (table_name, column_name 等)
        self._create_index()
    
    def _create_index(self):
        """创建 FAISS 索引"""
        try:
            if self.index_type == "flat":
                # 精确搜索，适合小规模数据 (< 100 万)
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                # IVF 索引，适合中等规模数据
                nlist = 100  # 聚类中心数
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
            elif self.index_type == "hnsw":
                # HNSW 索引，适合大规模数据，支持快速搜索
                M = 32  # 每个节点的邻居数
                self.index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_L2)
            else:
                raise ValueError(f"未知索引类型：{self.index_type}")
            
            logger.info(f"FAISS 索引创建完成：{self.index_type}, 维度={self.dimension}")
        except Exception as e:
            logger.error(f"FAISS 索引创建失败：{e}")
            raise
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]) -> int:
        """
        添加 Embedding 到索引
        
        Args:
            embeddings: Embedding 矩阵 (n_samples, dimension)
            metadata: 元数据列表，每个元素对应一个 Embedding
                     [{"key": "table:bank_customer", "table_name": "bank_customer", ...}, ...]
        
        Returns:
            int: 添加的数量
        """
        if len(embeddings) == 0:
            return 0
        
        # 确保 embeddings 是 float32 类型
        embeddings = embeddings.astype(np.float32)
        
        # 验证维度
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding 维度不匹配：期望 {self.dimension}, 实际 {embeddings.shape[1]}")
        
        # 添加到索引
        self.index.add(embeddings)
        
        # 保存元数据
        self.metadata.extend(metadata)
        
        logger.info(f"添加 {len(embeddings)} 个 Embedding 到 FAISS 索引")
        return len(embeddings)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        相似度搜索
        
        Args:
            query_embedding: 查询 Embedding (dimension,)
            top_k: 返回 Top-K 个结果
        
        Returns:
            List[Tuple[Dict, float]]: [(metadata, distance), ...]
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS 索引为空，无法搜索")
            return []
        
        # 确保 query_embedding 形状正确
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # 搜索
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            distance = float(distances[0][i])
            metadata = self.metadata[idx].copy()
            metadata["distance"] = distance
            
            results.append((metadata, distance))
        
        return results
    
    def search_batch(self, query_embeddings: np.ndarray, 
                     top_k: int = 5) -> List[List[Tuple[Dict, float]]]:
        """
        批量相似度搜索
        
        Args:
            query_embeddings: 查询 Embedding 矩阵 (n_queries, dimension)
            top_k: 每个查询返回 Top-K 个结果
        
        Returns:
            List[List[Tuple[Dict, float]]]: 每个查询的结果列表
        """
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]
        
        query_embeddings = query_embeddings.astype(np.float32)
        
        # 批量搜索
        distances, indices = self.index.search(query_embeddings, top_k)
        
        # 构建结果
        all_results = []
        for q_idx in range(len(query_embeddings)):
            results = []
            for i, idx in enumerate(indices[q_idx]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                
                distance = float(distances[q_idx][i])
                metadata = self.metadata[idx].copy()
                metadata["distance"] = distance
                
                results.append((metadata, distance))
            all_results.append(results)
        
        return all_results
    
    def save(self, path: str):
        """
        保存索引到文件
        
        Args:
            path: 保存路径
        """
        try:
            # 保存 FAISS 索引
            faiss.write_index(self.index, path)
            
            # 保存元数据
            metadata_path = path.replace(".index", ".meta.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"FAISS 索引已保存：{path}")
        except Exception as e:
            logger.error(f"FAISS 索引保存失败：{e}")
            raise
    
    def load(self, path: str):
        """
        从文件加载索引
        
        Args:
            path: 索引文件路径
        """
        try:
            # 加载 FAISS 索引
            self.index = faiss.read_index(path)
            
            # 加载元数据
            metadata_path = path.replace(".index", ".meta.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
            
            logger.info(f"FAISS 索引已加载：{path}, 共 {self.index.ntotal} 个向量")
        except Exception as e:
            logger.error(f"FAISS 索引加载失败：{e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_vectors": self.index.ntotal,
            "total_metadata": len(self.metadata)
        }
    
    def clear(self):
        """清空索引"""
        self._create_index()
        self.metadata = []
        logger.info("FAISS 索引已清空")
