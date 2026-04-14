"""
智能算法推荐引擎

基于数据特征自动推荐最佳算法。
推荐规则基于启发式经验，可解释、零依赖。

使用方式:
    from model_code.auto_selector import AutoModelSelector
    result = AutoModelSelector.recommend(X_train, y_train)
    print(result["recommended"])  # "catboost"
"""

import numpy as np
from typing import Dict, Any, List


class AutoModelSelector:
    """静态方法智能推荐引擎"""

    @staticmethod
    def recommend(X, y) -> Dict[str, Any]:
        """
        分析数据特征，推荐最佳算法

        Args:
            X: 特征矩阵 (DataFrame 或 ndarray)
            y: 标签 (Series 或 ndarray)

        Returns:
            {
                "recommended": "catboost",
                "reason": "...",
                "alternatives": ["lightgbm", "xgboost"],
                "candidates": [...],
                "analysis": {...}
            }
        """
        analysis = AutoModelSelector._analyze_data(X, y)
        scores = AutoModelSelector._score_algorithms(analysis)

        # 按得分排序
        sorted_algos = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommended = sorted_algos[0][0]
        alternatives = [a for a, _ in sorted_algos[1:4]]

        reasons = AutoModelSelector._explain_reasons(analysis, recommended)

        return {
            "recommended": recommended,
            "reason": reasons,
            "alternatives": alternatives,
            "candidates": [a for a, _ in sorted_algos],
            "analysis": analysis,
        }

    @staticmethod
    def _analyze_data(X, y) -> Dict[str, Any]:
        """分析数据集特征"""
        import pandas as pd

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # 类别特征数
        if hasattr(X, 'columns'):
            n_categorical = sum(1 for c in X.columns if X[c].dtype == 'object' or X[c].dtype.name == 'category')
        else:
            n_categorical = 0

        # 类别不平衡比
        if hasattr(y, 'value_counts'):
            counts = y.value_counts()
        else:
            counts = pd.Series(y).value_counts()
        imbalance_ratio = counts.iloc[-1] / counts.iloc[0] if len(counts) > 1 else 1.0

        # 特征稀疏度
        if hasattr(X, 'values'):
            X_vals = X.values
        else:
            X_vals = X
        sparsity = np.sum(X_vals == 0) / X_vals.size if X_vals.size > 0 else 0.0

        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_categorical": n_categorical,
            "imbalance_ratio": round(imbalance_ratio, 4),
            "sparsity": round(sparsity, 4),
        }

    @staticmethod
    def _score_algorithms(analysis: Dict) -> Dict[str, float]:
        """为每种算法打分 (启发式规则)"""
        scores = {
            "lightgbm": 0.0,
            "xgboost": 0.0,
            "catboost": 0.0,
            "random_forest": 0.0,
            "logistic_regression": 0.0,
        }

        n = analysis["n_samples"]
        n_cat = analysis["n_categorical"]
        imbalance = analysis["imbalance_ratio"]
        sparsity = analysis["sparsity"]

        # 规则 1: 小数据集 (< 1000) — RF / LR 更稳定
        if n < 1000:
            scores["random_forest"] += 3
            scores["logistic_regression"] += 2
            scores["catboost"] -= 1
            scores["xgboost"] -= 1

        # 规则 2: 大数据集 (> 100000) — LGBM / XGB 效率高
        if n > 100000:
            scores["lightgbm"] += 3
            scores["xgboost"] += 2
            scores["random_forest"] -= 1

        # 规则 3: 类别特征多 (> 5) — CatBoost 原生支持
        if n_cat > 5:
            scores["catboost"] += 4
            scores["lightgbm"] += 1

        # 规则 4: 类别不平衡 (< 0.05) — LGBM / XGB 支持 scale_pos_weight
        if imbalance < 0.05:
            scores["lightgbm"] += 2
            scores["xgboost"] += 2
            scores["logistic_regression"] -= 1

        # 规则 5: 高稀疏度 (> 0.5) — LGBM 擅长稀疏特征
        if sparsity > 0.5:
            scores["lightgbm"] += 3
            scores["xgboost"] += 1

        # 中等数据集 — LGBM 是稳健的默认选择
        if 1000 <= n <= 100000:
            scores["lightgbm"] += 2

        return scores

    @staticmethod
    def _explain_reasons(analysis: Dict, recommended: str) -> str:
        """生成推荐理由文本"""
        reasons = []
        n = analysis["n_samples"]
        n_cat = analysis["n_categorical"]
        imbalance = analysis["imbalance_ratio"]
        sparsity = analysis["sparsity"]

        if n < 1000:
            reasons.append(f"数据集较小 ({n:,} 样本)，适合简单模型")
        elif n > 100000:
            reasons.append(f"数据集较大 ({n:,} 样本)，需要高效算法")
        else:
            reasons.append(f"中等规模数据集 ({n:,} 样本)")

        if n_cat > 5:
            reasons.append(f"包含 {n_cat} 个类别特征")

        if imbalance < 0.05:
            reasons.append(f"类别不平衡严重 (比例 {imbalance:.1%})")

        if sparsity > 0.5:
            reasons.append(f"特征稀疏度较高 ({sparsity:.1%})")

        algo_name_map = {
            "lightgbm": "LightGBM 训练速度快、内存占用低，适合大规模数据",
            "xgboost": "XGBoost 精度高、正则化能力强",
            "catboost": "CatBoost 原生支持类别特征，无需独热编码",
            "random_forest": "随机森林对中小数据集稳定、不易过拟合",
            "logistic_regression": "逻辑回归简单可解释，适合线性可分问题",
        }

        reasons.append(algo_name_map.get(recommended, ""))

        return "；".join(reasons)

    @staticmethod
    def quick_score(X, y, algorithm: str, cv: int = 3) -> float:
        """
        快速评估单个算法 (k-fold CV)

        Args:
            X: 特征矩阵
            y: 标签
            algorithm: 算法标识
            cv: 交叉验证折数

        Returns:
            平均 ROC-AUC 分数
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from model_code.model_factory import ModelFactory

        model = ModelFactory.build(algorithm, task_type="binary")
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(model, X, y, cv=cv_split, scoring="roc_auc")
            return float(np.mean(scores))
        except Exception:
            return 0.0
