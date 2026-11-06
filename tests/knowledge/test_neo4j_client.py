"""
Neo4j Client 单元测试
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

# 测试配置
TEST_NEO4J_URI = os.getenv("TEST_NEO4J_URI", "bolt://localhost:7687")
TEST_NEO4J_USER = os.getenv("TEST_NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.getenv("TEST_NEO4J_PASSWORD", "test123")


class TestNeo4jClient:
    """Neo4jClient 测试类"""
    
    @pytest.fixture
    def mock_driver(self):
        """模拟 Neo4j 驱动"""
        with patch('core.knowledge.neo4j_client.GraphDatabase') as mock:
            yield mock
    
    @pytest.fixture
    def client(self, mock_driver):
        """创建测试客户端"""
        from core.knowledge.neo4j_client import Neo4jClient
        
        # 模拟 session
        mock_session = MagicMock()
        mock_driver.driver.return_value.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.driver.return_value.session.return_value.__exit__ = Mock(return_value=None)
        
        client = Neo4jClient(
            uri=TEST_NEO4J_URI,
            user=TEST_NEO4J_USER,
            password=TEST_NEO4J_PASSWORD
        )
        return client
    
    def test_init(self, mock_driver):
        """测试初始化"""
        from core.knowledge.neo4j_client import Neo4jClient
        
        client = Neo4jClient(
            uri=TEST_NEO4J_URI,
            user=TEST_NEO4J_USER,
            password=TEST_NEO4J_PASSWORD
        )
        
        assert client.uri == TEST_NEO4J_URI
        assert client.user == TEST_NEO4J_USER
        assert client.password == TEST_NEO4J_PASSWORD
        mock_driver.driver.assert_called_once()
    
    def test_create_table_node(self, client, mock_driver):
        """测试创建表节点"""
        result = client.create_table_node("test_table", "测试表")
        
        assert result is True
        mock_driver.driver.return_value.session.assert_called()
    
    def test_get_table_node(self, client, mock_driver):
        """测试获取表节点"""
        # 模拟返回
        mock_session = mock_driver.driver.return_value.session.return_value.__enter__.return_value
        mock_result = MagicMock()
        mock_result.single.return_value = {"name": "test_table", "description": "测试表"}
        mock_session.run.return_value = mock_result
        
        result = client.get_table_node("test_table")
        
        assert result is not None
        assert result["name"] == "test_table"
    
    def test_get_table_node_not_found(self, client, mock_driver):
        """测试获取不存在的表节点"""
        mock_session = mock_driver.driver.return_value.session.return_value.__enter__.return_value
        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result
        
        result = client.get_table_node("nonexistent")
        
        assert result is None
    
    def test_list_tables(self, client, mock_driver):
        """测试列出所有表"""
        mock_session = mock_driver.driver.return_value.session.return_value.__enter__.return_value
        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"name": "table1"},
            {"name": "table2"}
        ]))
        mock_session.run.return_value = mock_result
        
        result = client.list_tables()
        
        assert len(result) == 2
        assert "table1" in result
        assert "table2" in result
    
    def test_create_column_node(self, client, mock_driver):
        """测试创建字段节点"""
        # 先创建表
        client.create_table_node("test_table")
        
        result = client.create_column_node(
            table="test_table",
            name="id",
            col_type="INT",
            description="主键 ID",
            is_primary=True,
            is_nullable=False
        )
        
        assert result is True
    
    def test_get_columns(self, client, mock_driver):
        """测试获取字段列表"""
        mock_session = mock_driver.driver.return_value.session.return_value.__enter__.return_value
        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"name": "id", "type": "INT", "description": "主键"},
            {"name": "name", "type": "VARCHAR", "description": "名称"}
        ]))
        mock_session.run.return_value = mock_result
        
        result = client.get_columns("test_table")
        
        assert len(result) == 2
        assert result[0]["name"] == "id"
    
    def test_create_foreign_key(self, client, mock_driver):
        """测试创建外键关系"""
        result = client.create_foreign_key(
            from_table="orders",
            from_col="customer_id",
            to_table="customers",
            to_col="id",
            constraint_name="fk_orders_customer"
        )
        
        assert result is True
    
    def test_get_foreign_keys(self, client, mock_driver):
        """测试获取外键列表"""
        mock_session = mock_driver.driver.return_value.session.return_value.__enter__.return_value
        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "from_table": "orders",
                "from_column": "customer_id",
                "to_table": "customers",
                "to_column": "id",
                "constraint_name": "fk_orders_customer"
            }
        ]))
        mock_session.run.return_value = mock_result
        
        result = client.get_foreign_keys("orders")
        
        assert len(result) == 1
        assert result[0]["to_table"] == "customers"
    
    def test_get_related_tables(self, client, mock_driver):
        """测试获取相关表"""
        mock_session = mock_driver.driver.return_value.session.return_value.__enter__.return_value
        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"name": "customers", "description": "客户表"}
        ]))
        mock_session.run.return_value = mock_result
        
        result = client.get_related_tables("orders", depth=2)
        
        assert len(result) == 1
        assert result[0]["name"] == "customers"
    
    def test_export_schema(self, client, mock_driver):
        """测试导出 Schema"""
        # 模拟 list_tables
        mock_session = mock_driver.driver.return_value.session.return_value.__enter__.return_value
        mock_session.run.return_value.__iter__ = Mock(return_value=iter([
            {"name": "table1"}
        ]))
        
        schema = client.export_schema()
        
        assert "tables" in schema
        assert "relationships" in schema
    
    def test_get_table_schema_text(self, client, mock_driver):
        """测试获取表 Schema 文本"""
        # 模拟返回
        mock_session = mock_driver.driver.return_value.session.return_value.__enter__.return_value
        
        # get_table_node 返回
        mock_result = MagicMock()
        mock_result.single.return_value = {"name": "users", "description": "用户表"}
        mock_session.run.return_value = mock_result
        
        text = client.get_table_schema_text("users")
        
        assert "表名：users" in text
        assert "用户表" in text
    
    def test_close(self, client, mock_driver):
        """测试关闭连接"""
        client.close()
        
        mock_driver.driver.return_value.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
