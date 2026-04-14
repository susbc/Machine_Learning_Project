"""
模型预测模块 - 优化版

优化点:
1. 解耦路径配置，支持参数注入
2. 添加预测结果分析
3. 支持批量预测
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any
import warnings

warnings.filterwarnings('ignore')


def load_and_predict(
    data: pd.DataFrame,
    key: Union[pd.Series, np.ndarray],
    data_month: str,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    加载模型并预测 - 优化版
    
    Args:
        data: 特征数据
        key: 主键列
        data_month: 数据月份标识
        model_dir: 模型目录
        output_dir: 输出目录
        threshold: 分类阈值 (可选，默认使用模型默认)
    
    Returns:
        预测结果 DataFrame
    """
    # 路径处理
    model_dir = Path(model_dir) if model_dir else Path("saved")
    output_dir = Path(output_dir) if output_dir else Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载训练摘要 (获取算法名)
    summary_path = model_dir / "train_summary.joblib"
    if not summary_path.exists():
        raise FileNotFoundError(f"训练摘要文件不存在：{summary_path}")

    try:
        summary = joblib.load(summary_path)
        algorithm = summary.get("algorithm", "lightgbm")  # 向后兼容默认 LGBM
    except Exception:
        algorithm = "lightgbm"

    # 加载模型 (新命名 + 旧命名回退)
    model_path = model_dir / f"best_model_{algorithm}.joblib"
    if not model_path.exists():
        # 回退到旧命名
        old_path = model_dir / "best_lgbm_model.joblib"
        if old_path.exists():
            model_path = old_path
            algorithm = "lightgbm"
            print(f"⚠️ 使用旧版模型文件：{old_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

    print(f"加载模型：{model_path} (算法: {algorithm})")
    model = joblib.load(model_path)

    # 尝试加载训练摘要 (包含最佳阈值)
    best_threshold = threshold
    if summary_path.exists() and threshold is None:
        try:
            best_threshold = summary.get('best_threshold', 0.5)
            print(f"使用训练时的最佳阈值：{best_threshold:.4f}")
        except Exception:
            best_threshold = 0.5
    
    if best_threshold is None:
        best_threshold = 0.5
    
    # 预测
    print(f"进行预测... ({data.shape[0]:,} 条记录)")
    predictions_proba = model.predict_proba(data)[:, 1]
    
    # 使用阈值生成分类预测
    predictions = (predictions_proba >= best_threshold).astype(int)
    
    # 构建结果 DataFrame
    key_values = key.values if hasattr(key, 'values') else key
    result_df = pd.DataFrame({
        'key': key_values,
        'probability': predictions_proba,
        'prediction': predictions
    })
    
    # 统计预测分布
    pos_count = predictions.sum()
    neg_count = len(predictions) - pos_count
    pos_ratio = pos_count / len(predictions)
    
    print(f"\n预测结果统计:")
    print(f"  总记录数：{len(predictions):,}")
    print(f"  正样本预测：{pos_count:,} ({pos_ratio:.2%})")
    print(f"  负样本预测：{neg_count:,} ({1-pos_ratio:.2%})")
    
    # 保存结果
    output_path = output_dir / f"predict_res_{data_month}.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ 预测结果已保存：{output_path}")
    
    # 保存详细结果 (包含阈值分析)
    detail_path = output_dir / f"predict_res_{data_month}_detail.csv"
    result_df['threshold_used'] = best_threshold
    result_df.to_csv(detail_path, index=False)
    print(f"✅ 详细结果已保存：{detail_path}")
    
    return result_df


def batch_predict(
    data_paths: list,
    config_path: str,
    data_months: list,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    批量预测 - 多个月份数据
    
    Args:
        data_paths: 数据文件路径列表
        config_path: 配置文件路径
        data_months: 月份标识列表
        model_dir: 模型目录
        output_dir: 输出目录
    
    Returns:
        预测结果字典 {month: result_df}
    """
    from model_code.main_model import BuildingModel
    
    results = {}
    
    for data_path, data_month in zip(data_paths, data_months):
        print(f"\n{'='*60}")
        print(f"处理月份：{data_month}")
        print(f"{'='*60}")
        
        model = BuildingModel(
            data_path=data_path,
            config_path=config_path,
            data_month=data_month,
            output_dir=output_dir,
            model_dir=model_dir
        )
        
        result = model.model_predict()
        results[data_month] = result
    
    # 合并所有结果
    if results:
        combined_df = pd.concat(results.values(), ignore_index=True)
        combined_path = output_dir / "predict_res_combined.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\n✅ 合并结果已保存：{combined_path}")
    
    return results


def predict_with_explanation(
    data: pd.DataFrame,
    key: pd.Series,
    model_dir: Optional[Path] = None,
    top_n: int = 5
) -> pd.DataFrame:
    """
    带特征重要性解释的预测
    
    Args:
        data: 特征数据
        key: 主键
        model_dir: 模型目录
        top_n: 显示最重要的 N 个特征
    
    Returns:
        包含解释的预测结果
    """
    model_dir_p = Path(model_dir) if model_dir else Path("saved")
    summary_path = model_dir_p / "train_summary.joblib"
    algorithm = "lightgbm"
    if summary_path.exists():
        try:
            summary = joblib.load(summary_path)
            algorithm = summary.get("algorithm", "lightgbm")
        except Exception:
            pass

    model_path = model_dir_p / f"best_model_{algorithm}.joblib"
    if not model_path.exists():
        old_path = model_dir_p / "best_lgbm_model.joblib"
        if old_path.exists():
            model_path = old_path
        else:
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

    model = joblib.load(model_path)
    
    # 预测
    predictions_proba = model.predict_proba(data)[:, 1]
    
    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': data.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 重要特征:")
        print(feature_importance.head(top_n).to_string(index=False))
    
    # 构建结果
    result_df = pd.DataFrame({
        'key': key.values if hasattr(key, 'values') else key,
        'probability': predictions_proba,
        'prediction': (predictions_proba >= 0.5).astype(int)
    })
    
    return result_df
