"""
Schema Embedder 单元测试
"""

import pytest
import numpy as np
from unittest.mock import patch
from core.knowledge.schema_embedder import SchemaEmbedder


class TestSchemaEmbedder:
    """SchemaEmbedder 测试类"""
    
    @pytest.fixture(scope="class")
    def embedder(self):
        """创建测试 Embedder (使用 mini 模型)"""
        return SchemaEmbedder(model_name="mini")
    
    def test_init(self):
        """测试初始化"""
        embedder = SchemaEmbedder(model_name="mini")
        
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is not None
    
    @patch("core.knowledge.schema_embedder.SentenceTransformer")
    def test_init_with_alias(self, model_factory):
        """测试使用别名初始化"""
        embedder_mini = SchemaEmbedder(model_name="mini")
        embedder_base = SchemaEmbedder(model_name="base")
        embedder_large = SchemaEmbedder(model_name="large")
        
        assert embedder_mini.model_name == "all-MiniLM-L6-v2"
        assert embedder_base.model_name == "all-mpnet-base-v2"
        assert embedder_large.model_name == "multi-qa-mpnet-base-dot-v1"
        assert [call.args[0] for call in model_factory.call_args_list] == [
            "all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1",
        ]
    
    def test_get_dimension(self, embedder):
        """测试获取维度"""
        dim = embedder.get_dimension()
        
        assert dim > 0
        assert dim == 384  # mini 模型的维度
    
    def test_embed_table_name(self, embedder):
        """测试表名 Embedding"""
        embedding = embedder.embed_table_name("bank_customer")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.get_dimension(),)
        # 检查是否归一化
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
    
    def test_embed_column(self, embedder):
        """测试字段 Embedding"""
        embedding = embedder.embed_column(
            table_name="bank_customer",
            column_name="customer_id",
            description="客户 ID",
            col_type="VARCHAR"
        )
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.get_dimension(),)
    
    def test_embed_column_minimal(self, embedder):
        """测试最小字段 Embedding"""
        embedding = embedder.embed_column(
            table_name="users",
            column_name="id"
        )
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.get_dimension(),)
    
    def test_embed_text(self, embedder):
        """测试任意文本 Embedding"""
        embedding = embedder.embed_text("这是一个测试文本")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.get_dimension(),)
    
    def test_embed_batch(self, embedder):
        """测试批量 Embedding"""
        texts = [
            "文本 1",
            "文本 2",
            "文本 3"
        ]
        
        embeddings = embedder.embed_batch(texts, batch_size=2)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, embedder.get_dimension())
    
    def test_embed_batch_empty(self, embedder):
        """测试空批量 Embedding"""
        embeddings = embedder.embed_batch([])
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0,)
    
    def test_embed_schema_batch(self, embedder):
        """测试 Schema 批量 Embedding"""
        tables = [
            {
                "table_name": "bank_customer",
                "description": "客户信息表",
                "columns": [
                    {
                        "name": "customer_id",
                        "type": "VARCHAR",
                        "description": "客户 ID"
                    },
                    {
                        "name": "customer_name",
                        "type": "VARCHAR",
                        "description": "客户姓名"
                    }
                ]
            },
            {
                "table_name": "bank_account",
                "description": "账户信息表",
                "columns": [
                    {
                        "name": "account_id",
                        "type": "VARCHAR",
                        "description": "账户 ID"
                    }
                ]
            }
        ]
        
        embeddings_dict = embedder.embed_schema_batch(tables, show_progress=False)
        
        assert isinstance(embeddings_dict, dict)
        # 2 个表 + 3 个字段 = 5 个 Embedding
        assert len(embeddings_dict) == 5
        
        # 检查 key 格式
        assert "table:bank_customer" in embeddings_dict
        assert "column:bank_customer.customer_id" in embeddings_dict
        
        # 检查维度
        for key, emb in embeddings_dict.items():
            assert emb.shape == (embedder.get_dimension(),)
    
    def test_embed_schema_batch_empty(self, embedder):
        """测试空 Schema 批量 Embedding"""
        embeddings_dict = embedder.embed_schema_batch([])
        
        assert isinstance(embeddings_dict, dict)
        assert len(embeddings_dict) == 0
    
    def test_semantic_similarity(self, embedder):
        """测试语义相似性"""
        # mini 别名使用英文 MiniLM；用英文 Schema 术语验证语义，
        # 中文输入兼容性由 test_embed_text 单独覆盖。
        emb1 = embedder.embed_text("customer information table")
        emb2 = embedder.embed_text("customer data table")
        emb3 = embedder.embed_text("car prices")
        
        # 计算余弦相似度
        similarity_12 = np.dot(emb1, emb2)
        similarity_13 = np.dot(emb1, emb3)
        
        # 相似文本的相似度应该更高
        assert similarity_12 > similarity_13
    
    def test_table_column_distinction(self, embedder):
        """测试表和字段 Embedding 的区别"""
        table_emb = embedder.embed_table_name("users")
        column_emb = embedder.embed_column("users", "id")
        
        # 表和字段的 Embedding 应该不同
        similarity = np.dot(table_emb, column_emb)
        assert similarity < 1.0  # 不完全相同


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
