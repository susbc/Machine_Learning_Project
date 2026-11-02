"""
MLOps Monitor - 数据漂移监控模块

核心功能:
1. PSI (Population Stability Index) 计算
2. 训练基线数据提取
3. 漂移报告生成
4. 自动重训触发

评判标准:
- PSI < 0.1: 无显著漂移
- 0.1 ≤ PSI < 0.2: 轻微漂移 (需关注)
- PSI ≥ 0.2: 严重漂移 (必须重训)
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """
    数据漂移监控器
    
    使用 PSI (群体稳定性指标) 衡量线上数据与训练基线的分布差异
    """
    
    # PSI 阈值标准
    PSI_LOW = 0.1    # 轻微漂移阈值
    PSI_HIGH = 0.2   # 严重漂移阈值
    
    def __init__(self, bins: int = 10):
        """
        初始化监控器
        
        Args:
            bins: 分箱数量 (用于连续特征 PSI 计算)
        """
        self.bins = bins
    
    def calculate_psi(
        self,
        expected_array: np.ndarray,
        actual_array: np.ndarray,
        bins: Optional[int] = None
    ) -> float:
        """
        计算单个连续型特征的 PSI 值
        
        数学公式:
        PSI = Σ(actual% - expected%) × ln(actual% / expected%)
        
        Args:
            expected_array: 训练集 (基线) 数据
            actual_array: 线上真实数据
            bins: 分箱数量
            
        Returns:
            float: PSI 值
        """
        bins = bins or self.bins

        # 数据清洗：统一经 pandas 去缺失，兼容 list / ndarray / Series 输入。
        # （此前 np.array(...).dropna() 的写法对 Series 输入会抛 AttributeError，
        #   因为 np.array 之后是 ndarray，并没有 dropna 方法。）
        expected_array = np.asarray(pd.Series(expected_array).dropna(), dtype=float)
        actual_array = np.asarray(pd.Series(actual_array).dropna(), dtype=float)

        if len(expected_array) == 0 or len(actual_array) == 0:
            logger.warning("数据为空，无法计算 PSI")
            return 0.0

        # 1. 确定分箱边界 (基于训练集 Expected)
        breakpoints = np.arange(0, bins + 1) / bins * 100
        q = np.percentile(expected_array, breakpoints)

        # 处理边界重复值
        q = np.unique(q)
        if len(q) < 2:
            logger.warning("分箱边界过少，PSI 计算可能不准确")
            return 0.0

        # 首尾边界扩到 ±inf：np.histogram 默认丢弃界外样本，会把
        # "线上取值超出训练范围"这类最典型的漂移静默漏掉
        q[0], q[-1] = -np.inf, np.inf
        
        # 2. 计算 Expected 分布比例
        expected_percents = np.histogram(expected_array, bins=q)[0] / len(expected_array)
        
        # 3. 计算 Actual 分布比例 (使用相同的边界)
        actual_percents = np.histogram(actual_array, bins=q)[0] / len(actual_array)
        
        # 4. 避免除以 0 或 log(0) 的平滑处理
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # 5. 计算 PSI 公式
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return float(psi_value)
    
    def calculate_categorical_psi(
        self,
        expected_array: List[Any],
        actual_array: List[Any]
    ) -> float:
        """
        计算类别型特征的 PSI 值
        
        Args:
            expected_array: 训练集 (基线) 数据
            actual_array: 线上真实数据
            
        Returns:
            float: PSI 值
        """
        # 计算各类别占比
        expected_counts = pd.Series(expected_array).value_counts(normalize=True)
        actual_counts = pd.Series(actual_array).value_counts(normalize=True)
        
        # 合并所有类别
        all_categories = set(expected_counts.index) | set(actual_counts.index)
        
        # 平滑处理
        psi_value = 0.0
        for cat in all_categories:
            expected_pct = expected_counts.get(cat, 0.0001)
            actual_pct = actual_counts.get(cat, 0.0001)
            
            # 避免 0 值
            expected_pct = max(expected_pct, 0.0001)
            actual_pct = max(actual_pct, 0.0001)
            
            psi_value += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        
        return float(psi_value)
    
    def extract_baseline(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        max_samples: int = 10000
    ) -> Dict[str, Any]:
        """
        在训练完成时调用：提取并保存基线分布数据
        
        Args:
            df_train: 训练数据集
            features: 特征列名列表
            max_samples: 每个特征最多保存的样本数 (节省存储空间)
            
        Returns:
            Dict: 基线数据字典 {feature_name: [values...]}
        """
        logger.info(f"开始提取基线分布数据，共 {len(features)} 个特征...")
        
        baseline_data = {}
        
        for col in features:
            if col not in df_train.columns:
                continue
            
            col_data = df_train[col].dropna()
            
            # 数值型特征：保存抽样数据
            if pd.api.types.is_numeric_dtype(col_data):
                if len(col_data) > max_samples:
                    # 随机抽样
                    baseline_data[col] = col_data.sample(n=max_samples, random_state=42).tolist()
                else:
                    baseline_data[col] = col_data.tolist()
                
                # 保存统计信息
                baseline_data[f"{col}_stats"] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "type": "numeric"
                }
            
            # 类别型特征：保存类别分布
            else:
                value_counts = col_data.value_counts(normalize=True).to_dict()
                baseline_data[f"{col}_dist"] = value_counts
                baseline_data[f"{col}_stats"] = {
                    "n_categories": len(value_counts),
                    "top_categories": list(value_counts.keys())[:10],
                    "type": "categorical"
                }
        
        logger.info(f"基线数据提取完成，保存 {len(baseline_data)} 项")
        return baseline_data
    
    def generate_drift_report(
        self,
        baseline_data: Dict[str, Any],
        df_online: pd.DataFrame,
        threshold_low: float = 0.1,
        threshold_high: float = 0.2
    ) -> Dict[str, Any]:
        """
        生成完整的漂移检测报告
        
        Args:
            baseline_data: 基线数据 (从 extract_baseline 获取)
            df_online: 线上真实数据
            threshold_low: 轻微漂移阈值
            threshold_high: 严重漂移阈值
            
        Returns:
            Dict: 漂移报告
        """
        logger.info("开始生成漂移检测报告...")
        
        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "n_samples_online": len(df_online),
            "features": {},
            "summary": {
                "total_features": 0,
                "drifted_features": 0,
                "critical_features": 0,
                "needs_retrain": False
            }
        }
        
        for key, data in baseline_data.items():
            # 统计信息键不构成特征
            if key.endswith("_stats"):
                continue

            # 类别特征的基线只有 {col}_dist（extract_baseline 不保留原始样本），
            # 必须以 _dist 键为入口解析出特征名；若只认裸特征名键，
            # 类别特征会整体被跳过，漂移永远漏检
            if key.endswith("_dist"):
                feature = key[: -len("_dist")]
                baseline_dist = data if isinstance(data, dict) else None
            else:
                feature = key
                baseline_dist = None

            if feature not in df_online.columns:
                continue

            stats = baseline_data.get(f"{feature}_stats", {})
            feature_type = stats.get("type", "unknown")

            # 获取线上数据
            online_data = df_online[feature].dropna()

            if len(online_data) == 0:
                continue

            # 计算 PSI
            if baseline_dist is not None or feature_type == "categorical":
                # 类别型：使用类别 PSI
                if baseline_dist:
                    # 从分布重建数组 (近似)
                    baseline_samples = []
                    for cat, pct in baseline_dist.items():
                        n = int(pct * 1000)
                        baseline_samples.extend([cat] * n)
                    psi = self.calculate_categorical_psi(baseline_samples, online_data.tolist())
                else:
                    psi = 0.0
            else:
                # 数值型：使用分箱 PSI
                psi = self.calculate_psi(data, online_data.values)
            
            # 判断漂移程度
            if psi >= threshold_high:
                severity = "critical"
                drift_report["summary"]["critical_features"] += 1
                drift_report["summary"]["needs_retrain"] = True
            elif psi >= threshold_low:
                severity = "warning"
            else:
                severity = "normal"
            
            if psi >= threshold_low:
                drift_report["summary"]["drifted_features"] += 1
            
            drift_report["summary"]["total_features"] += 1
            
            # 记录特征漂移信息
            drift_report["features"][feature] = {
                "psi": round(psi, 4),
                "severity": severity,
                "baseline_stats": stats,
                "online_stats": {
                    "mean": float(online_data.mean()) if pd.api.types.is_numeric_dtype(online_data) else None,
                    "std": float(online_data.std()) if pd.api.types.is_numeric_dtype(online_data) else None,
                    "n_samples": len(online_data)
                }
            }
        
        logger.info(f"漂移报告生成完成，发现 {drift_report['summary']['drifted_features']} 个漂移特征")
        return drift_report
    
    def save_baseline(self, baseline_data: Dict[str, Any], save_path: str):
        """
        保存基线数据到文件
        
        Args:
            baseline_data: 基线数据
            save_path: 保存路径
        """
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)
        logger.info(f"基线数据已保存：{save_path}")
    
    def load_baseline(self, load_path: str) -> Dict[str, Any]:
        """
        从文件加载基线数据
        
        Args:
            load_path: 加载路径
            
        Returns:
            Dict: 基线数据
        """
        with open(load_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        logger.info(f"基线数据已加载：{load_path}")
        return baseline_data


# ==========================================
# 便捷函数
# ==========================================

def check_drift(
    baseline_path: str,
    df_online: pd.DataFrame,
    threshold_low: float = 0.1,
    threshold_high: float = 0.2
) -> Dict[str, Any]:
    """
    便捷函数：检查数据漂移
    
    Args:
        baseline_path: 基线数据文件路径
        df_online: 线上数据
        threshold_low: 轻微漂移阈值
        threshold_high: 严重漂移阈值
        
    Returns:
        Dict: 漂移报告
    """
    monitor = DataDriftMonitor()
    baseline_data = monitor.load_baseline(baseline_path)
    report = monitor.generate_drift_report(
        baseline_data,
        df_online,
        threshold_low,
        threshold_high
    )
    return report


__all__ = [
    "DataDriftMonitor",
    "check_drift",
]
