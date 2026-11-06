"""
统一模型训练编排器

整合多算法训练流程，支持 single / auto / compare 三种模式。

使用方式:
    from model_code.model_trainer import ModelTrainer
    trainer = ModelTrainer(X_train, y_train, X_test, y_test)
    result = trainer.train_single("lightgbm")
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from model_code.model_factory import ModelFactory
from model_code.auto_selector import AutoModelSelector

from model_code.config.config import MODEL_DIR, OUTPUT_DIR


class ModelTrainer:
    """统一训练编排器"""

    def __init__(
        self,
        X_train, y_train,
        X_test, y_test,
        model_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        random_state: int = 42,
    ):
        """
        Args:
            X_train, y_train: 训练集 (已预处理)
            X_test, y_test: 测试集 (已预处理)
            model_dir: 模型保存目录
            output_dir: 输出目录
            random_state: 随机种子
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.random_state = random_state

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_single(
        self,
        algorithm: str,
        n_iter: int = 10,
        cv: int = 5,
    ) -> Dict[str, Any]:
        """
        单模型训练 (RandomizedSearchCV)

        Args:
            algorithm: 算法标识
            n_iter: 随机搜索迭代次数
            cv: 交叉验证折数

        Returns:
            训练结果字典
        """
        print("=" * 60)
        print(f"训练算法: {ModelFactory.get_display_name(algorithm)}")
        print("=" * 60)

        # 获取参数网格
        param_grid = ModelFactory.get_param_grid(algorithm)

        # 构建基础模型
        base_model = ModelFactory.build(algorithm, task_type="binary")

        # 随机网格搜索
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state,
        )

        print(f"开始随机网格搜索 (n_iter={n_iter}, cv={cv})...")
        search.fit(self.X_train, self.y_train)

        best_model = search.best_estimator_
        print(f"最佳参数:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")

        # 测试集评估
        y_test_proba = best_model.predict_proba(self.X_test)[:, 1]
        y_test_pred = best_model.predict(self.X_test)

        test_auc = roc_auc_score(self.y_test, y_test_proba)
        test_ap = average_precision_score(self.y_test, y_test_proba)

        print(f"\n测试集评估:")
        print(f"  ROC-AUC: {test_auc:.6f}")
        print(f"  AUC-PR:  {test_ap:.6f}")
        print(classification_report(self.y_test, y_test_pred, digits=6))

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_test_pred)
        print(f"\n混淆矩阵:")
        print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

        # 阈值分析
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_test_proba)
        metric_df = pd.DataFrame({
            "threshold": np.append(thresholds, 1.0),
            "precision": precisions,
            "recall": recalls,
        })
        metric_df["f1"] = 2 * (metric_df["precision"] * metric_df["recall"]) / (
            metric_df["precision"] + metric_df["recall"] + 1e-10
        )

        best_idx = metric_df["f1"].idxmax()
        best_threshold = float(metric_df.loc[best_idx, "threshold"])
        best_f1 = float(metric_df.loc[best_idx, "f1"])
        best_precision = float(metric_df.loc[best_idx, "precision"])
        best_recall = float(metric_df.loc[best_idx, "recall"])

        print(f"\n最佳阈值: {best_threshold:.4f}, F1: {best_f1:.6f}")

        # 保存模型
        model_path = self.model_dir / f"best_model_{algorithm}.joblib"
        joblib.dump(best_model, model_path)

        # 保存阈值指标
        metric_path = self.output_dir / f"metric_thresholds_{algorithm}.csv"
        metric_df.to_csv(metric_path, index=False)

        # 保存训练摘要
        train_summary = {
            "algorithm": algorithm,
            "model": best_model,
            "test_auc": test_auc,
            "test_ap": test_ap,
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "best_precision": best_precision,
            "best_recall": best_recall,
            "best_params": search.best_params_,
            "metric_df": metric_df,
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "feature_names": (
                list(self.X_train.columns)
                if hasattr(self.X_train, "columns")
                else [f"feature_{i}" for i in range(self.X_train.shape[1])]
            ),
        }
        summary_path = self.model_dir / f"train_summary_{algorithm}.joblib"
        joblib.dump(train_summary, summary_path)

        # 同时保存一份无 algorithm 后缀的摘要 (向后兼容)
        joblib.dump(train_summary, self.model_dir / "train_summary.joblib")

        print(f"\n✅ 模型已保存：{model_path}")
        print(f"✅ 训练摘要已保存：{summary_path}")

        return train_summary

    def train_auto(
        self,
        n_iter: int = 10,
        cv: int = 5,
    ) -> Dict[str, Any]:
        """
        自动推荐模式

        1. AutoModelSelector.recommend() 获取推荐
        2. 调用 train_single(recommended_algorithm)

        Returns:
            train_single 的结果 + 推荐信息
        """
        print("=" * 60)
        print("智能算法推荐分析")
        print("=" * 60)

        recommendation = AutoModelSelector.recommend(self.X_train, self.y_train)
        recommended = recommendation["recommended"]
        reason = recommendation["reason"]

        print(f"推荐算法: {ModelFactory.get_display_name(recommended)}")
        print(f"推荐理由: {reason}")
        print(f"备选算法: {[ModelFactory.get_display_name(a) for a in recommendation['alternatives']]}")
        print()

        result = self.train_single(recommended, n_iter, cv)
        result["recommendation"] = recommendation
        return result

    def train_compare(
        self,
        algorithms: Optional[List[str]] = None,
        n_iter: int = 10,
        cv: int = 5,
    ) -> Dict[str, Any]:
        """
        多模型对比

        Args:
            algorithms: 要对比的算法列表 (默认全部)
            n_iter: 每个算法的搜索迭代次数
            cv: 交叉验证折数

        Returns:
            {results, comparison_df, best_algorithm}
        """
        if algorithms is None:
            algorithms = ModelFactory.get_supported_algorithms()

        results = {}
        for algo in algorithms:
            try:
                result = self.train_single(algo, n_iter, cv)
                results[algo] = result
            except Exception as e:
                print(f"⚠️ {algo} 训练失败: {e}")

        if not results:
            raise RuntimeError("所有算法训练均失败")

        # 构建对比表
        comparison_data = []
        for algo, r in results.items():
            comparison_data.append({
                "algorithm": ModelFactory.get_display_name(algo),
                "ROC-AUC": f"{r['test_auc']:.6f}",
                "AUC-PR": f"{r['test_ap']:.6f}",
                "Best F1": f"{r['best_f1']:.6f}",
                "Best Threshold": f"{r['best_threshold']:.4f}",
            })

        comparison_df = pd.DataFrame(comparison_data)
        best_algorithm = max(results, key=lambda a: results[a]["test_auc"])

        print("\n" + "=" * 60)
        print("算法对比结果")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        print(f"\n🏆 最佳算法: {ModelFactory.get_display_name(best_algorithm)}")

        return {
            "results": results,
            "comparison_df": comparison_df,
            "best_algorithm": best_algorithm,
        }
