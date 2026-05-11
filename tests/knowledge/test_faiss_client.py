"""
FAISS Client 单元测试
"""

import pytest
import numpy as np
import os
import tempfile
from core.knowledge.faiss_client import FAISSClient


class TestFAISSClient:
    """FAISSClient 测试类"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return FAISSClient(dimension=384, index_type="flat")
    
    @pytest.fixture
    def sample_embeddings(self):
        """生成测试 Embedding"""
        np.random.seed(42)
        return np.random.rand(10, 384).astype(np.float32)
    
    @pytest.fixture
    def sample_metadata(self):
        """生成测试元数据"""
        return [
            {"key": f"table:table{i}", "type": "table", "table_name": f"table{i}"}
            for i in range(5)
        ] + [
            {"key": f"column:table{i}.col{j}", "type": "column", 
             "table_name": f"table{i}", "column_name": f"col{j}"}
            for i in range(5) for j in range(2)
        ]
    
    def test_init(self):
        """测试初始化"""
        client = FAISSClient(dimension=384, index_type="flat")
        
        assert client.dimension == 384
        assert client.index_type == "flat"
        assert client.index.ntotal == 0
    
    def test_add_embeddings(self, client, sample_embeddings, sample_metadata):
        """测试添加 Embedding"""
        count = client.add_embeddings(sample_embeddings, sample_metadata)
        
        assert count == 10
        assert client.index.ntotal == 10
        assert len(client.metadata) == 10
    
    def test_add_empty_embeddings(self, client):
        """测试添加空 Embedding"""
        count = client.add_embeddings(np.array([]), [])
        
        assert count == 0
        assert client.index.ntotal == 0
    
    def test_add_wrong_dimension(self, client, sample_embeddings):
        """测试添加错误维度的 Embedding"""
        wrong_emb = np.random.rand(5, 256).astype(np.float32)  # 维度不匹配
        
        with pytest.raises(ValueError, match="维度不匹配"):
            client.add_embeddings(wrong_emb, [])
    
    def test_search(self, client, sample_embeddings, sample_metadata):
        """测试相似度搜索"""
        # 添加数据
        client.add_embeddings(sample_embeddings, sample_metadata)
        
        # 搜索
        query = sample_embeddings[0]  # 使用第一个向量作为查询
        results = client.search(query, top_k=3)
        
        assert len(results) == 3
        
        # 第一个结果应该是查询向量本身 (距离接近 0)
        metadata, distance = results[0]
        assert "key" in metadata
        assert distance >= 0
    
    def test_search_empty_index(self, client, sample_embeddings):
        """测试空索引搜索"""
        query = sample_embeddings[0]
        results = client.search(query, top_k=5)
        
        assert len(results) == 0
    
    def test_search_batch(self, client, sample_embeddings, sample_metadata):
        """测试批量搜索"""
        # 添加数据
        client.add_embeddings(sample_embeddings, sample_metadata)
        
        # 批量搜索
        queries = sample_embeddings[:3]
        all_results = client.search_batch(queries, top_k=2)
        
        assert len(all_results) == 3
        assert all(len(results) == 2 for results in all_results)
    
    def test_save_and_load(self, client, sample_embeddings, sample_metadata):
        """测试保存和加载"""
        # 添加数据
        client.add_embeddings(sample_embeddings, sample_metadata)
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            temp_path = f.name
        
        try:
            client.save(temp_path)
            
            # 验证文件存在
            assert os.path.exists(temp_path)
            assert os.path.exists(temp_path.replace(".index", ".meta.json"))
            
            # 加载到新客户端
            new_client = FAISSClient(dimension=384)
            new_client.load(temp_path)
            
            # 验证数据
            assert new_client.index.ntotal == 10
            assert len(new_client.metadata) == 10
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            meta_path = temp_path.replace(".index", ".meta.json")
            if os.path.exists(meta_path):
                os.remove(meta_path)
    
    def test_get_stats(self, client, sample_embeddings, sample_metadata):
        """测试获取统计信息"""
        client.add_embeddings(sample_embeddings, sample_metadata)
        
        stats = client.get_stats()
        
        assert stats["dimension"] == 384
        assert stats["index_type"] == "flat"
        assert stats["total_vectors"] == 10
        assert stats["total_metadata"] == 10
    
    def test_clear(self, client, sample_embeddings, sample_metadata):
        """测试清空索引"""
        client.add_embeddings(sample_embeddings, sample_metadata)
        client.clear()
        
        assert client.index.ntotal == 0
        assert len(client.metadata) == 0
    
    def test_different_index_types(self):
        """测试不同索引类型"""
        # Flat 索引
        flat_client = FAISSClient(dimension=384, index_type="flat")
        assert flat_client.index_type == "flat"
        
        # IVF 索引
        ivf_client = FAISSClient(dimension=384, index_type="ivf")
        assert ivf_client.index_type == "ivf"
        
        # HNSW 索引
        hnsw_client = FAISSClient(dimension=384, index_type="hnsw")
        assert hnsw_client.index_type == "hnsw"
    
    def test_invalid_index_type(self):
        """测试无效索引类型"""
        with pytest.raises(ValueError, match="未知索引类型"):
            FAISSClient(dimension=384, index_type="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
