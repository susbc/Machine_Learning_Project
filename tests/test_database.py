"""
数据库模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_code.database import DatabaseConnector, QueryBuilder


class TestDatabaseConnector:
    """数据库连接器测试"""
    
    def test_sqlite_connection(self):
        """测试 SQLite 连接"""
        # 创建临时数据库文件
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            config = {
                'db_type': 'sqlite',
                'database': db_path
            }
            
            db = DatabaseConnector(config)
            
            # 测试连接
            assert db.connect() is True
            assert db.test_connection() is True
            
            # 测试查询
            db.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            db.execute("INSERT INTO test VALUES (1, 'test')")
            
            result = db.query("SELECT * FROM test")
            assert len(result) == 1
            assert result.iloc[0]['name'] == 'test'
            
            db.disconnect()
            
        finally:
            # 清理临时文件
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_invalid_db_type(self):
        """测试无效数据库类型"""
        config = {
            'db_type': 'invalid',
            'database': 'test.db'
        }
        
        with pytest.raises(ValueError):
            DatabaseConnector(config)
    
    def test_list_tables(self):
        """测试表列表获取"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            config = {
                'db_type': 'sqlite',
                'database': db_path
            }
            
            db = DatabaseConnector(config)
            db.connect()
            
            # 创建测试表
            db.execute("CREATE TABLE table1 (id INTEGER)")
            db.execute("CREATE TABLE table2 (id INTEGER)")
            
            tables = db.list_tables()
            
            assert 'table1' in tables
            assert 'table2' in tables
            
            db.disconnect()
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestQueryBuilder:
    """查询构建器测试"""
    
    def test_basic_select(self):
        """测试基本 SELECT"""
        qb = QueryBuilder('users')
        sql, params = qb.select('id', 'name', 'email').build()
        
        assert sql == "SELECT id, name, email FROM users"
        assert params == {}
    
    def test_select_all(self):
        """测试 SELECT *"""
        qb = QueryBuilder('users')
        sql, params = qb.build()
        
        assert sql == "SELECT * FROM users"
    
    def test_where_clause(self):
        """测试 WHERE 子句"""
        qb = QueryBuilder('users')
        sql, params = qb.where('age', '>', 18).build()
        
        assert "WHERE age > :param_0" in sql
        assert params['param_0'] == 18
    
    def test_multiple_where(self):
        """测试多个 WHERE 条件"""
        qb = QueryBuilder('users')
        sql, params = (qb
            .where('age', '>', 18)
            .where('status', '=', 'active')
            .build())
        
        assert "WHERE age > :param_0" in sql
        assert "AND status = :param_1" in sql
        assert params['param_0'] == 18
        assert params['param_1'] == 'active'
    
    def test_limit_offset(self):
        """测试 LIMIT 和 OFFSET"""
        qb = QueryBuilder('users')
        sql, params = (qb
            .limit(10)
            .offset(20)
            .build())
        
        assert "LIMIT 10" in sql
        assert "OFFSET 20" in sql
    
    def test_order_by(self):
        """测试 ORDER BY"""
        qb = QueryBuilder('users')
        sql, params = (qb
            .order_by('created_at', 'DESC')
            .build())
        
        assert "ORDER BY created_at DESC" in sql
    
    def test_distinct(self):
        """测试 DISTINCT"""
        qb = QueryBuilder('users')
        sql, params = qb.distinct().select('email').build()
        
        assert "SELECT DISTINCT email FROM users" in sql
    
    def test_in_operator(self):
        """测试 IN 操作符"""
        qb = QueryBuilder('users')
        sql, params = qb.where('id', 'IN', [1, 2, 3]).build()
        
        assert "id IN (:param_0_0, :param_0_1, :param_0_2)" in sql
        assert params['param_0_0'] == 1
        assert params['param_0_1'] == 2
        assert params['param_0_2'] == 3
    
    def test_like_operator(self):
        """测试 LIKE 操作符"""
        qb = QueryBuilder('users')
        sql, params = qb.where('name', 'LIKE', 'john').build()
        
        assert "name LIKE :param_0" in sql
        assert params['param_0'] == '%john%'
    
    def test_join(self):
        """测试 JOIN"""
        qb = QueryBuilder('orders')
        sql, params = (qb
            .select('orders.id', 'users.name')
            .join('users', 'orders.user_id = users.id')
            .build())
        
        assert "INNER JOIN users ON orders.user_id = users.id" in sql
    
    def test_group_by(self):
        """测试 GROUP BY"""
        qb = QueryBuilder('orders')
        sql, params = (qb
            .select('user_id', 'COUNT(*)')
            .group_by('user_id')
            .build())
        
        assert "GROUP BY user_id" in sql

    def test_or_where_uses_explicit_boolean_operator(self):
        qb = QueryBuilder('users')
        sql, _ = qb.where('status', '=', 'active').or_where('role', '=', 'admin').build()

        assert "WHERE status = :param_0 OR role = :param_1" in sql

    def test_count_distinct_expression(self):
        sql, _ = QueryBuilder('users').select('COUNT(DISTINCT email) AS unique_count').build()

        assert sql == "SELECT COUNT(DISTINCT email) AS unique_count FROM users"
    
    def test_reset(self):
        """测试重置"""
        qb = QueryBuilder('users')
        qb.where('age', '>', 18).limit(10)
        qb.reset()
        
        sql, params = qb.build()
        assert sql == "SELECT * FROM users"
        assert params == {}

    @pytest.mark.parametrize(
        ("factory", "message"),
        [
            (lambda: QueryBuilder("users; DROP TABLE users"), "表名"),
            (lambda: QueryBuilder("users").select("name, password"), "SELECT"),
            (lambda: QueryBuilder("users").where("id", "= 1 OR", 1), "操作符"),
            (lambda: QueryBuilder("users").where("id; DROP", "=", 1), "列名"),
            (
                lambda: QueryBuilder("orders").join("users", "orders.user_id = users.id OR 1=1"),
                "JOIN",
            ),
        ],
    )
    def test_rejects_unsafe_sql_fragments(self, factory, message):
        """不可参数化的 SQL 片段必须经过严格白名单。"""
        with pytest.raises(ValueError, match=message):
            factory()

    @pytest.mark.parametrize("method", ["limit", "offset"])
    def test_rejects_invalid_pagination(self, method):
        """分页值只能是非负整数。"""
        with pytest.raises(ValueError, match="非负整数"):
            getattr(QueryBuilder("users"), method)(-1)

    def test_rejects_empty_in_values(self):
        with pytest.raises(ValueError, match="空列表"):
            QueryBuilder("users").where("id", "IN", [])

    def test_rejects_duplicate_parameter_names(self):
        qb = QueryBuilder("users").where("id", "=", 1, param_name="identifier")
        with pytest.raises(ValueError, match="参数名已存在"):
            qb.where("owner_id", "=", 2, param_name="identifier")


class TestAutoTrainer:
    """自动训练器测试"""
    
    def test_auto_trainer_init(self):
        """测试自动训练器初始化"""
        from model_code.auto_train import AutoTrainer
        
        config = {
            'X': pd.DataFrame(np.random.randn(100, 10)),
            'y': pd.Series(np.random.randint(0, 2, 100)),
            'model_params': {'n_estimators': 10}
        }
        
        trainer = AutoTrainer(config, max_iterations=3)
        
        assert trainer.max_iterations == 3
        assert trainer.best_score == 0
    
    def test_single_iteration(self):
        """测试单次迭代"""
        from model_code.auto_train import AutoTrainer
        
        np.random.seed(42)
        config = {
            'X': pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)]),
            'y': pd.Series(np.random.randint(0, 2, 100)),
            'model_params': {'n_estimators': 10, 'verbose': -1},
            'test_size': 0.3,
            'random_state': 42
        }
        
        trainer = AutoTrainer(config, max_iterations=1)
        result = trainer.train_iteration(1)
        
        assert 'auc' in result
        assert 'f1' in result
        assert 0 <= result['auc'] <= 1
        assert 0 <= result['f1'] <= 1


class TestDataGenerator:
    """数据生成器测试"""
    
    def test_generate_basic(self):
        """测试基本数据生成"""
        from scripts.generate_test_data import generate_test_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            X, y, df = generate_test_data(
                n_samples=100,
                n_features=10,
                positive_ratio=0.2,
                missing_rate=0.1,
                outlier_rate=0.05,
                output_dir=Path(tmpdir)
            )
            
            assert len(df) == 100
            assert 'target' in df.columns
            assert 'id' in df.columns
            
            # 检查正样本比例
            actual_ratio = df['target'].mean()
            assert 0.1 <= actual_ratio <= 0.3  # 允许一定波动
    
    def test_generate_no_missing(self):
        """测试无缺失值生成"""
        from scripts.generate_test_data import generate_test_data
        
        X, y, df = generate_test_data(
            n_samples=100,
            n_features=10,
            missing_rate=0
        )
        
        assert df.isnull().sum().sum() == 0
    
    def test_generate_custom_seed(self):
        """测试自定义随机种子"""
        from scripts.generate_test_data import generate_test_data
        
        X1, y1, df1 = generate_test_data(
            n_samples=100,
            n_features=10,
            random_state=42
        )
        
        X2, y2, df2 = generate_test_data(
            n_samples=100,
            n_features=10,
            random_state=42
        )
        
        # 相同种子应该生成相同数据
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
