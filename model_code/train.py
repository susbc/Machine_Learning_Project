import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np


def lgbm_train_with_random_search(X, y):
    """
    基于LightGBM的随机网格搜索调参训练，输出测试集AUC，保存模型和查全查准指标到Excel

    参数:
        df (pd.DataFrame): 输入的训练数据（包含特征列和标签列）
        label_col (str): 标签列的列名（二分类任务，标签值需为0/1）
    """

    # 拆分训练集和测试集（测试集占比30%，固定随机种子保证可复现）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y  # stratify保持标签分布一致
    )

    # ---------------------- 步骤2：定义LightGBM模型和随机搜索参数网格 ----------------------
    # 初始化LightGBM分类器（二分类任务）
    lgb_clf = lgb.LGBMClassifier(
        objective='binary',  # 二分类目标
        metric='auc',  # 评估指标AUC
        verbose=-1,  # 关闭冗余日志输出
        random_state=42
    )

    # 定义随机网格搜索的参数空间（选取LGBM核心可调参数，范围合理）
    param_grid = {
        'n_estimators': np.arange(100, 501, 100),  # 决策树数量
        'learning_rate': np.linspace(0.01, 0.3, 10),  # 学习率
        'max_depth': np.arange(3, 10, 1),  # 树的最大深度
        'num_leaves': np.arange(20, 101, 20),  # 叶子节点数量（需小于2^max_depth）
        'min_child_samples': np.arange(10, 51, 10),  # 叶子节点最小样本数
        'subsample': np.linspace(0.7, 1.0, 4)  # 行采样率（随机森林思想）
    }

    # ---------------------- 步骤3：初始化并执行随机网格搜索 ----------------------
    random_search = RandomizedSearchCV(
        estimator=lgb_clf,
        param_distributions=param_grid,
        n_iter=10,  # 随机搜索迭代次数（用户要求）
        cv=5,  # 5折交叉验证（用户要求）
        scoring='roc_auc',  # 以AUC作为评分标准
        n_jobs=-1,  # 启用所有CPU核心加速
        verbose=1,  # 输出简要训练日志
        random_state=42
    )

    # 执行网格搜索，拟合训练数据
    print("开始执行随机网格搜索调参（cv=5, n_iter=10）...")
    random_search.fit(X_train, y_train)

    # ---------------------- 步骤4：保存最佳模型 ----------------------
    best_lgb_model = random_search.best_estimator_
    model_save_path = "best_lgbm_model.joblib"
    joblib.dump(best_lgb_model, model_save_path)
    print(f"最佳模型已保存至：{model_save_path}")
    print(f"随机网格搜索最佳参数：\n{random_search.best_params_}")

    # ---------------------- 步骤5：计算并输出测试集AUC ----------------------
    # 预测测试集的正类概率（二分类取第1列概率）
    y_test_proba = best_lgb_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)
    print(f"测试集AUC值：{test_auc:.6f}")

    # ---------------------- 步骤6：计算不同阈值的查全率、查准率并保存到Excel ----------------------
    # 生成0.01至0.99的阈值序列（步长0.01）
    thresholds = np.arange(0.01, 1.0, 0.01)
    metric_results = []

    for threshold in thresholds:
        # 根据当前阈值将概率转换为二分类预测结果（大于阈值为1，否则为0）
        y_test_pred = (y_test_proba >= threshold).astype(int)

        # 计算查准率（precision）和查全率（recall）
        precision = precision_score(y_test, y_test_pred, zero_division=0)  # 避免全0预测时报错
        recall = recall_score(y_test, y_test_pred, zero_division=0)

        # 存储结果
        metric_results.append({
            'threshold': threshold,
            'precision': precision,  # 查准率
            'recall': recall  # 查全率
        })

    # 转换为DataFrame并保存到Excel
    metric_df = pd.DataFrame(metric_results)
    excel_save_path = "metric.xlsx"
    metric_df.to_excel(excel_save_path, index=False, engine='openpyxl')
    print(f"查全查准指标已保存至：{excel_save_path}")

    # ---------------------- 步骤7：返回核心结果（可选，方便后续调用） ----------------------
    return {
        'best_model': best_lgb_model,
        'test_auc': test_auc,
        'metric_df': metric_df
    }