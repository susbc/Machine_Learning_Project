"""
模型训练模块 - 优化版

核心优化:
1. 使用 precision_recall_curve 替代循环计算 (向量化，速度快 100x)
2. 解耦路径配置，支持参数注入
3. 添加更多评估指标
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
import lightgbm as lgb
from sklearn.base import clone


def lgbm_train_with_random_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    n_iter: int = 10,
    cv: int = 5
) -> Dict[str, Any]:
    """
    LightGBM 随机网格搜索训练 - 优化版
    
    优化点:
    1. 使用 precision_recall_curve 替代循环 (向量化，速度快)
    2. 添加更多评估指标 (AUC-PR, F1, etc.)
    3. 配置参数化，移除硬编码路径
    4. 保存更多训练信息
    
    Args:
        X: 特征 DataFrame
        y: 标签 Series
        model_dir: 模型保存目录
        output_dir: 输出目录
        test_size: 测试集比例
        random_state: 随机种子
        n_iter: 随机搜索迭代次数
        cv: 交叉验证折数
    
    Returns:
        训练结果字典
    """
    # 路径处理
    model_dir = Path(model_dir) if model_dir else Path("saved")
    output_dir = Path(output_dir) if output_dir else Path("../output")
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)

    if X.select_dtypes(exclude=[np.number, "bool"]).shape[1]:
        bad_columns = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        raise ValueError(f"训练特征必须完成数值编码，非数值列：{', '.join(map(str, bad_columns))}")
    X = X.replace([np.inf, -np.inf], np.nan)
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < 2:
        raise ValueError("二分类训练要求两个类别，且每个类别至少 2 条记录")
    print("LightGBM 随机网格搜索训练")
    print("=" * 60)
    
    # 1. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    safe_cv = min(cv, int(pd.Series(y_train).value_counts().min()))
    if safe_cv < 2:
        raise ValueError("训练集中的每个类别至少需要 2 条记录才能执行交叉验证")
    
    print(f"\n数据集划分:")
    print(f"  训练集：{X_train.shape[0]:,} 行，{X_train.shape[1]} 特征")
    print(f"  测试集：{X_test.shape[0]:,} 行")
    print(f"  正样本比例：{y_train.mean():.2%}")
    
    # 2. 定义模型和参数空间
    lgb_clf = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        verbose=-1,
        random_state=random_state,
        n_jobs=1
    )
    
    param_grid = {
        'n_estimators': np.arange(100, 501, 100),
        'learning_rate': np.linspace(0.01, 0.3, 10),
        'max_depth': np.arange(3, 10, 1),
        'num_leaves': np.arange(20, 101, 20),
        'min_child_samples': np.arange(10, 51, 10),
        'subsample': np.linspace(0.7, 1.0, 4),
        'colsample_bytree': np.linspace(0.7, 1.0, 4),
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0, 0.01, 0.1, 1]
    }
    
    # 3. 执行随机网格搜索
    random_search = RandomizedSearchCV(
        estimator=lgb_clf,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=safe_cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )
    
    print(f"\n开始随机网格搜索 (n_iter={n_iter}, cv={cv})...")
    random_search.fit(X_train, y_train)
    
    # 4. 保存最佳模型
    best_model = random_search.best_estimator_
    model_path = model_dir / "best_lgbm_model.joblib"
    joblib.dump(best_model, model_path)
    
    print(f"\n✅ 最佳模型已保存：{model_path}")
    print(f"\n最佳参数:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    # 5. 测试集评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = best_model.predict(X_test)
    
    # 核心指标
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_ap = average_precision_score(y_test, y_test_proba)  # AUC-PR
    
    print(f"\n核心指标:")
    print(f"  ROC-AUC: {test_auc:.6f}")
    print(f"  AUC-PR:  {test_ap:.6f}")
    
    # 分类报告
    print(f"\n分类报告 (默认阈值 0.5):")
    print(classification_report(y_test, y_test_pred, digits=6))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n混淆矩阵:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    
    # 6. 在训练集内部留出的校准集选择阈值，测试集只做最终评估。
    print("\n" + "=" * 60)
    print("计算阈值 - 查准率/查全率曲线")
    print("=" * 60)
    
    if int(pd.Series(y_train).value_counts().min()) >= 3:
        calibration_size = max(2, int(np.ceil(len(X_train) * 0.2)))
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train,
            y_train,
            test_size=calibration_size,
            random_state=random_state,
            stratify=y_train,
        )
        calibration_model = clone(best_model).fit(X_fit, y_fit)
        calibration_proba = calibration_model.predict_proba(X_cal)[:, 1]
        calibration_source = 'training_calibration_split'
    else:
        X_cal, y_cal = X_train, y_train
        calibration_proba = best_model.predict_proba(X_cal)[:, 1]
        calibration_source = 'training_fit_fallback'
    precisions, recalls, thresholds = precision_recall_curve(y_cal, calibration_proba)
    
    # 构建结果 DataFrame
    metric_df = pd.DataFrame({
        'threshold': np.append(thresholds, 1.0),
        'precision': precisions,
        'recall': recalls
    })
    
    # 添加 F1 分数
    metric_df['f1'] = 2 * (metric_df['precision'] * metric_df['recall']) / (
        metric_df['precision'] + metric_df['recall'] + 1e-10
    )
    
    # 找到最佳阈值 (最大 F1)
    best_idx = metric_df['f1'].idxmax()
    best_threshold = metric_df.loc[best_idx, 'threshold']
    best_f1 = metric_df.loc[best_idx, 'f1']
    best_precision = metric_df.loc[best_idx, 'precision']
    best_recall = metric_df.loc[best_idx, 'recall']
    
    print(f"\n最佳阈值分析:")
    print(f"  最佳阈值：{best_threshold:.4f}")
    print(f"  最佳 F1:   {best_f1:.6f}")
    print(f"  查准率：{best_precision:.6f}")
    print(f"  查全率：{best_recall:.6f}")
    
    # 保存结果
    metric_path = output_dir / "metric_thresholds.csv"
    metric_df.to_csv(metric_path, index=False)
    print(f"\n✅ 阈值指标已保存：{metric_path}")
    
    # 同时保存 Excel 格式 (兼容旧版)
    try:
        excel_path = output_dir / "metric_thresholds.xlsx"
        metric_df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"✅ Excel 格式已保存：{excel_path}")
    except ImportError:
        print("⚠️ 未安装 openpyxl，跳过 Excel 保存")
    
    # 7. 保存训练摘要
    train_summary = {
        'model': best_model,
        'algorithm': 'lightgbm',
        'test_auc': test_auc,
        'test_ap': test_ap,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'best_params': random_search.best_params_,
        'metric_df': metric_df,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_names': X.columns.tolist(),
        'threshold_calibration': {
            'source': calibration_source,
            'calibration_size': int(len(y_cal)),
        }
    }
    
    summary_path = model_dir / "train_summary.joblib"
    joblib.dump(train_summary, summary_path)
    print(f"✅ 训练摘要已保存：{summary_path}")
    
    return train_summary


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1',
    threshold_range: Tuple[float, float, float] = (0.01, 0.99, 0.01)
) -> Tuple[float, Dict[str, float]]:
    """
    寻找最优分类阈值 - 完全向量化版本

    使用矩阵运算一次性计算所有阈值的 precision/recall/f1，
    避免 sklearn 逐阈值调用的循环开销。

    Args:
        y_true: 真实标签
        y_proba: 预测概率
        metric: 优化指标 ('f1', 'precision', 'recall')
        threshold_range: (min, max, step)

    Returns:
        最优阈值和对应指标
    """
    min_th, max_th, step = threshold_range
    thresholds = np.arange(min_th, max_th + step, step)

    # 预测矩阵: (n_samples, n_thresholds)
    predictions = (y_proba[:, np.newaxis] >= thresholds).astype(int)
    y = y_true[:, np.newaxis]

    # 完全向量化计算 TP / FP / FN
    tp = np.sum(predictions & y, axis=0)
    fp = np.sum(predictions & (1 - y), axis=0)
    fn = np.sum((1 - predictions) & y, axis=0)

    precisions = tp / (tp + fp + 1e-10)
    recalls = tp / (tp + fn + 1e-10)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)

    if metric == 'f1':
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx], {
            'f1': f1_scores[best_idx],
            'precision': precisions[best_idx],
            'recall': recalls[best_idx]
        }
    elif metric == 'precision':
        best_idx = np.argmax(precisions)
        return thresholds[best_idx], {'precision': precisions[best_idx]}
    elif metric == 'recall':
        best_idx = np.argmax(recalls)
        return thresholds[best_idx], {'recall': recalls[best_idx]}
    else:
        raise ValueError(f"不支持的指标：{metric}")
