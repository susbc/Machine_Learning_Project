#!/usr/bin/env python3
"""
功能演示脚本

演示所有新增功能:
1. 数据库连接
2. 模型对比
3. 自动迭代训练
4. 数据生成
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*70)
print("Machine Learning Project - 新增功能演示")
print("="*70)
print()

# ============================================================================
# 1. 数据生成器演示
# ============================================================================

print("1️⃣  数据生成器演示")
print("-"*70)

from scripts.generate_test_data import generate_test_data
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    X, y, df = generate_test_data(
        n_samples=1000,
        n_features=10,
        positive_ratio=0.15,
        missing_rate=0.05,
        outlier_rate=0.02,
        output_dir=Path(tmpdir)
    )
    
    print(f"✅ 生成数据：{len(df)} 行 × {len(df.columns)} 列")
    print(f"✅ 正样本比例：{df['target'].mean():.2%}")
    print(f"✅ 缺失值总数：{df.isnull().sum().sum()}")
    print()

# ============================================================================
# 2. 数据库连接演示
# ============================================================================

print("2️⃣  数据库连接演示")
print("-"*70)

from model_code.database import DatabaseConnector, QueryBuilder
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    db_path = f.name

try:
    # 连接数据库
    config = {'db_type': 'sqlite', 'database': db_path}
    db = DatabaseConnector(config)
    db.connect()
    print(f"✅ 数据库连接成功：{db.connection_string}")
    
    # 创建测试表
    db.execute("CREATE TABLE ml_data (id INTEGER, feature1 REAL, feature2 REAL, target INTEGER)")
    print("✅ 创建表成功")
    
    # 插入数据
    for i in range(10):
        db.execute(f"INSERT INTO ml_data VALUES ({i}, {i*0.1}, {i*0.2}, {i%2})")
    print("✅ 插入 10 条测试数据")
    
    # 查询数据
    df = db.query("SELECT * FROM ml_data LIMIT 5")
    print(f"✅ 查询返回 {len(df)} 行")
    print(df.head())
    
    # 使用查询构建器
    qb = QueryBuilder('ml_data')
    sql, params = qb.select('id', 'feature1').where('target', '=', 1).build()
    print(f"✅ 查询构建器生成：{sql}")
    
    # 表列表
    tables = db.list_tables()
    print(f"✅ 表列表：{tables}")
    
    db.disconnect()
    
finally:
    if os.path.exists(db_path):
        os.unlink(db_path)

print()

# ============================================================================
# 3. 自动迭代训练演示
# ============================================================================

print("3️⃣  自动迭代训练演示")
print("-"*70)

import pandas as pd
import numpy as np

# 生成模拟数据
np.random.seed(42)
n_samples = 500
X = pd.DataFrame(
    np.random.randn(n_samples, 10),
    columns=[f'feature_{i}' for i in range(10)]
)
y = pd.Series(np.random.randint(0, 2, n_samples))

# 跳过 LightGBM 依赖检查
try:
    from model_code.auto_train import AutoTrainer
    has_auto_train = True
except Exception as e:
    print(f"⚠️  自动训练模块跳过 (LightGBM 依赖问题): {e}")
    has_auto_train = False

if has_auto_train:
    config = {
        'X': X,
        'y': y,
        'model_params': {'n_estimators': 10, 'verbose': -1},
        'test_size': 0.3,
        'random_state': 42,
        'skip_hyperopt': True,
        'skip_feature_selection': True
    }

    trainer = AutoTrainer(
        base_config=config,
        max_iterations=3,
        early_stopping_rounds=2,
        min_improvement=0.001
    )

    print(f"开始自动迭代训练 (max_iterations={trainer.max_iterations})...")
    print()

    for iteration in range(1, min(3, trainer.max_iterations) + 1):
        result = trainer.train_iteration(iteration)
        print(f"  迭代 {iteration}: AUC={result['auc']:.4f}, F1={result['f1']:.4f}")

    print()
    print(f"✅ 最佳迭代：{trainer.best_iteration}")
    print(f"✅ 最佳 AUC: {trainer.best_score:.4f}")

    # 获取历史
    history_df = trainer.get_history_dataframe()
    print(f"✅ 训练历史：{len(history_df)} 次迭代")
else:
    print("⚠️  跳过自动迭代训练演示")

print()

# ============================================================================
# 4. 模型对比演示
# ============================================================================

print("4️⃣  模型对比组件演示")
print("-"*70)

from web_app.components.model_comparison import (
    create_metrics_comparison_chart,
    create_model_summary_table
)
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 尝试导入 LightGBM
try:
    import lightgbm as lgb
    has_lgb = True
except:
    has_lgb = False

# 准备测试数据
X_test = X.iloc[:100]
y_test = y.iloc[:100]

# 训练多个模拟模型
models = {
    'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=100),
}

if has_lgb:
    models['LightGBM'] = lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42)
else:
    print("⚠️  LightGBM 不可用，跳过")

print("训练对比模型...")

with tempfile.TemporaryDirectory() as tmpdir:
    model_paths = []
    
    for name, model in models.items():
        try:
            model.fit(X, y)
            model_path = Path(tmpdir) / f"{name}.joblib"
            joblib.dump(model, model_path)
            model_paths.append(str(model_path))
            print(f"  ✅ {name} 训练完成")
        except Exception as e:
            print(f"  ⚠️  {name} 训练失败：{e}")
    
    print()
    
    # 对比模型
    print("执行模型对比...")
    from web_app.components.model_comparison import compare_models
    
    results = compare_models(model_paths, X_test, y_test)
    
    # 显示结果
    summary_df = create_model_summary_table(results)
    print("\n模型对比结果:")
    print(summary_df.to_string(index=False))
    
    # 创建图表
    fig = create_metrics_comparison_chart(results)
    print(f"\n✅ 创建对比图表：{len(results)} 个模型")

print()

# ============================================================================
# 总结
# ============================================================================

print("="*70)
print("✅ 所有功能演示完成!")
print("="*70)
print()
print("新增功能:")
print("  1. ✅ 数据库连接支持 (MySQL/PostgreSQL/SQLite/SQL Server)")
print("  2. ✅ 模型效果对比 (多模型可视化对比)")
print("  3. ✅ 自动迭代训练 (超参数优化 + 特征选择)")
print("  4. ✅ UI/UX 优化 (主题切换 + 通知系统)")
print("  5. ✅ 数据生成器 (测试数据生成)")
print()
print("下一步:")
print("  - 启动 Web 应用：streamlit run web_app/app.py")
print("  - 查看文档：cat NEW_FEATURES.md")
print("="*70)
