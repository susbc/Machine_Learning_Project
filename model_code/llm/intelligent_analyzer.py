"""
AI 智能分析模块

使用大模型对机器学习流程进行智能分析和建议：
- 数据集分析 + 业务场景 → 模型推荐
- 训练结果分析 → 调参建议
- 预测结果分析 → 优化方向
"""

import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None


def _safe_float(value, default=None):
    """Convert scalar-like metric values to float without breaking prompts."""
    if value is None:
        return default
    try:
        if pd is not None and pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _format_metric(value, digits: int = 4) -> str:
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return "未提供"
    return f"{numeric_value:.{digits}f}"


def _json_default(value):
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


class IntelligentAnalyzer:
    """智能分析器"""
    
    def __init__(self, llm_manager=None):
        """
        初始化智能分析器
        
        Args:
            llm_manager: 大模型管理器实例
        """
        from model_code.llm.llm_connector import get_llm_manager
        
        self.llm_manager = llm_manager or get_llm_manager()
        
        # 分析模板
        self.prompts = {
            'data_analysis': self._get_data_analysis_prompt(),
            'model_recommendation': self._get_model_recommendation_prompt(),
            'training_analysis': self._get_training_analysis_prompt(),
            'prediction_analysis': self._get_prediction_analysis_prompt(),
            'hyperparameter_tuning': self._get_hyperparameter_tuning_prompt()
        }
    
    def _get_data_analysis_prompt(self) -> str:
        """数据分析提示词模板"""
        return """
你是一位资深的数据科学家和机器学习专家。请分析以下数据集的统计信息，并给出专业的分析建议。

## 数据集基本信息
- 样本数量：{n_samples}
- 特征数量：{n_features}
- 目标变量：{target_info}

## 特征统计信息
{feature_stats}

## 数据质量信息
- 缺失值比例：{missing_rate}
- 重复值比例：{duplicate_rate}
- 类别不平衡情况：{imbalance_info}

## 业务场景
{business_context}

请从以下角度进行分析：
1. **数据质量评估**：数据是否适合建模？存在哪些潜在问题？
2. **特征工程建议**：哪些特征可能需要特殊处理？是否需要创建新特征？
3. **建模风险提示**：基于数据特征，建模可能遇到哪些挑战？
4. **采样建议**：是否需要特殊采样策略？

请用简洁专业的语言回答，给出可操作的建议。
"""
    
    def _get_model_recommendation_prompt(self) -> str:
        """模型推荐提示词模板"""
        return """
你是一位机器学习架构师。请根据以下数据集特征和业务需求，推荐最适合的模型方案。

## 数据集特征
- 样本量：{n_samples} ({size_category})
- 特征类型：{feature_types}
- 特征数量：{n_features}
- 目标类型：{target_type}
- 类别分布：{class_distribution}

## 业务场景
{business_context}

## 业务需求
- 预测速度要求：{speed_requirement}
- 模型可解释性要求：{interpretability_requirement}
- 准确率要求：{accuracy_requirement}

## 可用模型选项
- LightGBM：快速、高效、适合表格数据
- XGBoost：精度高、可调参空间大
- Random Forest：稳定、可解释性好
- Logistic Regression：基线模型、可解释性最好
- Neural Network：适合复杂非线性关系、需要大量数据

请推荐：
1. **首选模型**及理由
2. **备选模型**及适用场景
3. **不推荐的模型**及原因
4. **模型组合建议**（如适用）

请给出具体、可操作的建议。
"""
    
    def _get_training_analysis_prompt(self) -> str:
        """训练结果分析提示词模板"""
        return """
你是一位机器学习专家。请分析以下模型训练结果，给出专业的诊断和优化建议。

## 模型信息
- 模型类型：{model_type}
- 训练样本数：{train_size}
- 验证样本数：{val_size}
- 特征数量：{n_features}

## 训练结果
- 训练集 AUC：{train_auc}
- 验证集 AUC：{val_auc}
- 训练集 F1：{train_f1}
- 验证集 F1：{val_f1}
- AUC 差距：{auc_gap}
- F1 差距：{f1_gap}

## 模型参数
{model_params}

## 特征重要性（Top 10）
{feature_importance}

请分析：
1. **模型状态诊断**：
   - 是否过拟合？（训练 - 验证差距）
   - 是否欠拟合？（训练集表现）
   - 模型收敛是否正常？

2. **问题定位**：
   - 如果过拟合：可能的原因和建议
   - 如果欠拟合：可能的原因和建议

3. **优化建议**（按优先级排序）：
   - 最应该尝试的 3 个优化方向
   - 每个方向的预期效果和风险

4. **下一步行动**：
   - 具体应该调整哪些参数？
   - 是否需要更多数据或特征？

请给出具体、可量化的建议。
"""
    
    def _get_prediction_analysis_prompt(self) -> str:
        """预测结果分析提示词模板"""
        return """
你是一位数据科学家。请分析以下模型预测结果，给出业务洞察和优化方向。

## 预测结果统计
- 总样本数：{total_samples}
- 预测正例数：{predicted_positive}
- 预测正例比例：{positive_rate}
- 使用阈值：{threshold}

## 预测分布
{prediction_distribution}

## 高置信度预测
- 高置信度正例（>0.8）：{high_conf_positive}
- 高置信度负例（<0.2）：{high_conf_negative}
- 低置信度预测（0.4-0.6）：{low_confidence}

## 与历史模型对比
{historical_comparison}

请分析：
1. **预测质量评估**：
   - 预测分布是否合理？
   - 是否存在预测偏差？

2. **业务洞察**：
   - 高置信度正例的特征模式
   - 低置信度样本的特点
   - 是否需要调整阈值？

3. **优化方向**：
   - 模型层面的改进建议
   - 特征层面的改进建议
   - 数据层面的改进建议

4. **风险提示**：
   - 预测结果中需要人工复核的部分
   - 业务应用中的注意事项

请结合业务场景给出具体建议。
"""
    
    def _get_hyperparameter_tuning_prompt(self) -> str:
        """超参数调优提示词模板"""
        return """
你是一位机器学习调参专家。请根据以下模型训练历史，给出下一步调参建议。

## 当前模型配置
- 模型类型：{model_type}
- 当前参数：{current_params}

## 历史调参记录
{history_records}

## 性能指标变化
{performance_history}

## 资源约束
- 最大训练时间：{max_train_time}
- 可用计算资源：{compute_resources}

请给出：
1. **参数敏感性分析**：
   - 哪些参数对性能影响最大？
   - 哪些参数已经接近最优？

2. **下一步调参建议**（按优先级）：
   - 应该调整哪些参数？
   - 建议的取值范围？
   - 预期提升幅度？

3. **调参策略**：
   - 推荐使用什么搜索方法？（网格/随机/贝叶斯）
   - 建议的搜索空间？

4. **早停建议**：
   - 是否应该停止调参？
   - 当前是否已达到性能瓶颈？

请给出具体、可执行的调参计划。
"""
    
    def analyze_dataset(self, df: 'pd.DataFrame', target_col: str,
                       business_context: str = "") -> str:
        """
        分析数据集并给出建议
        
        Args:
            df: 数据框
            target_col: 目标列名
            business_context: 业务场景描述
        
        Returns:
            AI 分析建议
        """
        if pd is None:
            return "错误：需要安装 pandas"

        if df is None or df.empty:
            return "错误：数据集为空，无法分析"

        if target_col not in df.columns:
            return f"错误：目标列 {target_col} 不存在"
        
        # 收集统计信息
        n_samples, n_columns = df.shape
        feature_cols = [c for c in df.columns if c != target_col]
        n_features = len(feature_cols)
        target_unique = df[target_col].nunique(dropna=True)
        target_info = f"{target_col} (类型：{df[target_col].dtype}, 唯一值：{target_unique})"
        
        # 特征统计
        feature_stats = []
        for col in feature_cols[:12]:
            missing_count = int(df[col].isna().sum())
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_value = _safe_float(df[col].mean())
                std_value = _safe_float(df[col].std())
                stats = (
                    f"{col}: 数值型, 均值={_format_metric(mean_value, 3)}, "
                    f"标准差={_format_metric(std_value, 3)}, 缺失={missing_count}"
                )
            else:
                stats = f"{col}: 类别/文本型, 唯一值={df[col].nunique(dropna=True)}, 缺失={missing_count}"
            feature_stats.append(stats)
        feature_stats = "\n".join(feature_stats) or "无可用特征"
        
        # 数据质量
        missing_rate = df.isna().sum().sum() / max(n_samples * n_columns, 1) * 100
        duplicate_rate = df.duplicated().sum() / max(n_samples, 1) * 100
        
        # 类别不平衡
        if target_unique == 2:
            class_dist = df[target_col].value_counts(normalize=True, dropna=True)
            parts = [f"{label}: {ratio * 100:.1f}%" for label, ratio in class_dist.items()]
            imbalance_info = "；".join(parts)
        elif target_unique <= 20:
            imbalance_info = f"{target_unique} 个类别，最大类占比 {df[target_col].value_counts(normalize=True).max() * 100:.1f}%"
        else:
            imbalance_info = "连续目标或高基数目标，请确认当前任务是否适合分类建模"
        
        # 构建提示词
        prompt = self.prompts['data_analysis'].format(
            n_samples=n_samples,
            n_features=n_features,
            target_info=target_info,
            feature_stats=feature_stats,
            missing_rate=f"{missing_rate:.2f}%",
            duplicate_rate=f"{duplicate_rate:.2f}%",
            imbalance_info=imbalance_info,
            business_context=business_context or "未提供"
        )
        
        # 调用大模型
        messages = [{"role": "user", "content": prompt}]
        return self.llm_manager.chat(messages, temperature=0.3)
    
    def recommend_model(self, df: 'pd.DataFrame', target_col: str,
                       business_context: str = "", requirements: Dict = None) -> str:
        """
        推荐模型
        
        Args:
            df: 数据框
            target_col: 目标列名
            business_context: 业务场景
            requirements: 需求字典
        
        Returns:
            模型推荐建议
        """
        if pd is None:
            return "错误：需要安装 pandas"

        if df is None or df.empty:
            return "错误：数据集为空，无法推荐模型"

        if target_col not in df.columns:
            return f"错误：目标列 {target_col} 不存在"
        
        n_samples = len(df)
        feature_df = df.drop(columns=[target_col])
        n_features = feature_df.shape[1]
        
        # 数据规模分类
        if n_samples < 1000:
            size_category = "小"
        elif n_samples < 100000:
            size_category = "中"
        else:
            size_category = "大"
        
        # 特征类型
        numeric_features = feature_df.select_dtypes(include=['number']).columns.tolist()
        categorical_features = feature_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        feature_types = f"数值特征：{len(numeric_features)}, 类别特征：{len(categorical_features)}"
        
        # 目标类型
        target_unique = df[target_col].nunique(dropna=True)
        if target_unique == 2:
            target_type = "二分类"
        elif target_unique <= 20 or not pd.api.types.is_numeric_dtype(df[target_col]):
            target_type = "多分类"
        else:
            target_type = "连续/高基数目标（当前训练流程以分类为主，需确认是否离散化或改造回归）"
        
        # 类别分布
        if target_unique == 2:
            class_dist = df[target_col].value_counts(normalize=True, dropna=True)
            class_distribution = "；".join([
                f"{label}: {ratio * 100:.1f}%"
                for label, ratio in class_dist.items()
            ])
        else:
            class_distribution = f"{target_unique} 个唯一值"
        
        # 默认需求
        default_requirements = {
            'speed_requirement': '中等',
            'interpretability_requirement': '中等',
            'accuracy_requirement': '高'
        }
        requirements = requirements or default_requirements
        
        # 构建提示词
        prompt = self.prompts['model_recommendation'].format(
            n_samples=n_samples,
            size_category=size_category,
            feature_types=feature_types,
            n_features=n_features,
            target_type=target_type,
            class_distribution=class_distribution,
            business_context=business_context or "未提供",
            speed_requirement=requirements.get('speed_requirement', '中等'),
            interpretability_requirement=requirements.get('interpretability_requirement', '中等'),
            accuracy_requirement=requirements.get('accuracy_requirement', '高')
        )
        
        messages = [{"role": "user", "content": prompt}]
        return self.llm_manager.chat(messages, temperature=0.3)
    
    def analyze_training_results(self, metrics: Dict, params: Dict,
                                 feature_importance: List = None) -> str:
        """
        分析训练结果
        
        Args:
            metrics: 指标字典
            params: 参数字典
            feature_importance: 特征重要性列表
        
        Returns:
            分析建议
        """
        metrics = metrics or {}
        params = params or {}
        train_auc = _safe_float(metrics.get('train_auc'))
        val_auc = _safe_float(metrics.get('val_auc'))
        train_f1 = _safe_float(metrics.get('train_f1'))
        val_f1 = _safe_float(metrics.get('val_f1'))
        
        auc_gap = abs(train_auc - val_auc) if train_auc is not None and val_auc is not None else None
        f1_gap = abs(train_f1 - val_f1) if train_f1 is not None and val_f1 is not None else None
        
        # 特征重要性
        if feature_importance:
            fi_text = "\n".join([f"{i+1}. {f}: {v:.4f}" 
                                for i, (f, v) in enumerate(feature_importance[:10])])
        else:
            fi_text = "未提供"
        
        # 参数字符串
        params_text = json.dumps(params, indent=2, ensure_ascii=False, default=_json_default)
        
        # 构建提示词
        prompt = self.prompts['training_analysis'].format(
            model_type=params.get('model_type', 'Unknown'),
            train_size=metrics.get('train_size', 0),
            val_size=metrics.get('val_size', 0),
            n_features=metrics.get('n_features', 0),
            train_auc=_format_metric(train_auc),
            val_auc=_format_metric(val_auc),
            train_f1=_format_metric(train_f1),
            val_f1=_format_metric(val_f1),
            auc_gap=_format_metric(auc_gap),
            f1_gap=_format_metric(f1_gap),
            model_params=params_text,
            feature_importance=fi_text
        )
        
        messages = [{"role": "user", "content": prompt}]
        return self.llm_manager.chat(messages, temperature=0.3)
    
    def analyze_prediction_results(self, predictions: 'pd.Series',
                                   threshold: float = 0.5,
                                   historical: Dict = None) -> str:
        """
        分析预测结果
        
        Args:
            predictions: 预测概率
            threshold: 阈值
            historical: 历史对比数据
        
        Returns:
            分析建议
        """
        if pd is None:
            return "错误：需要安装 pandas"

        predictions = pd.Series(predictions).dropna()
        if predictions.empty:
            return "错误：预测结果为空，无法分析"
        
        total = len(predictions)
        predicted_positive = (predictions >= threshold).sum()
        positive_rate = predicted_positive / total * 100
        
        # 预测分布
        high_conf_pos = (predictions >= 0.8).sum()
        high_conf_neg = (predictions <= 0.2).sum()
        low_conf = ((predictions >= 0.4) & (predictions <= 0.6)).sum()
        
        # 分布统计
        dist = predictions.describe()
        dist_text = f"均值：{dist['mean']:.3f}, 标准差：{dist['std']:.3f}, " \
                   f"最小：{dist['min']:.3f}, 最大：{dist['max']:.3f}"
        
        # 历史对比
        if historical:
            hist_text = f"历史 AUC: {historical.get('auc', 'N/A')}, " \
                       f"历史 F1: {historical.get('f1', 'N/A')}"
        else:
            hist_text = "无历史数据对比"
        
        # 构建提示词
        prompt = self.prompts['prediction_analysis'].format(
            total_samples=total,
            predicted_positive=predicted_positive,
            positive_rate=f"{positive_rate:.1f}%",
            threshold=threshold,
            prediction_distribution=dist_text,
            high_conf_positive=high_conf_pos,
            high_conf_negative=high_conf_neg,
            low_confidence=low_conf,
            historical_comparison=hist_text
        )
        
        messages = [{"role": "user", "content": prompt}]
        return self.llm_manager.chat(messages, temperature=0.3)
    
    def suggest_hyperparameter_tuning(self, current_params: Dict,
                                      history: List[Dict],
                                      constraints: Dict = None) -> str:
        """
        建议超参数调优
        
        Args:
            current_params: 当前参数
            history: 历史调参记录
            constraints: 约束条件
        
        Returns:
            调优建议
        """
        # 历史记录格式化
        if history:
            history_text = "\n".join([
                f"试验{i+1}: {json.dumps(h.get('params', {}), ensure_ascii=False, default=_json_default)} → AUC: {h.get('auc', 'N/A')}"
                for i, h in enumerate(history[-5:])  # 最近 5 次
            ])
        else:
            history_text = "无历史记录"
        
        # 性能历史
        if history:
            perf_text = "\n".join([
                f"试验{i+1}: AUC={h.get('auc', 0):.4f}, F1={h.get('f1', 0):.4f}"
                for i, h in enumerate(history)
            ])
        else:
            perf_text = "无数据"
        
        # 默认约束
        default_constraints = {
            'max_train_time': '30 分钟',
            'compute_resources': '单 CPU'
        }
        constraints = constraints or default_constraints
        
        # 构建提示词
        prompt = self.prompts['hyperparameter_tuning'].format(
            model_type=current_params.get('model_type', 'Unknown'),
            current_params=json.dumps(current_params, indent=2, ensure_ascii=False, default=_json_default),
            history_records=history_text,
            performance_history=perf_text,
            max_train_time=constraints.get('max_train_time', '30 分钟'),
            compute_resources=constraints.get('compute_resources', '单 CPU')
        )
        
        messages = [{"role": "user", "content": prompt}]
        return self.llm_manager.chat(messages, temperature=0.3)
    
    def save_analysis(self, analysis_type: str, content: str,
                     metadata: Dict = None) -> str:
        """
        保存分析结果
        
        Args:
            analysis_type: 分析类型
            content: 分析内容
            metadata: 元数据
        
        Returns:
            保存的文件路径
        """
        from pathlib import Path
        
        output_dir = Path(__file__).parent.parent / "analysis_reports"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{analysis_type}_{timestamp}.md"
        filepath = output_dir / filename
        
        # 构建报告
        report = f"""# AI 智能分析报告

## 分析类型
{analysis_type}

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 元数据
{json.dumps(metadata or {}, indent=2, ensure_ascii=False, default=_json_default)}

---

## 分析结果

{content}

---

*本报告由 AI 智能分析模块自动生成*
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"分析报告已保存：{filepath}")
        return str(filepath)


    def recommend_hyperparameters(
        self,
        df: 'pd.DataFrame',
        target_col: str,
        model_type: str,
        eda_summary: Optional[Dict] = None,
        last_training: Optional[Dict] = None,
        business_context: str = "",
        param_ranges: Optional[Dict[str, Dict]] = None,
    ) -> str:
        """
        调用 LLM 推荐超参数。

        Args:
            df: 完整数据框
            target_col: 目标列名
            model_type: 算法标识 (lightgbm / xgboost / ...)
            eda_summary: EDA 摘要字典（可选）
            last_training: 上次训练结果（可选）
            business_context: 业务场景描述（可选）
            param_ranges: UI 参数边界信息 {name: {min, max, default, options}}

        Returns:
            LLM 原文（包含 JSON + 解释文本）。
        """
        if pd is None:
            raise RuntimeError("需要安装 pandas")

        n_samples, n_features = df.shape
        n_numeric = len(df.select_dtypes(include=['int64', 'float64']).columns)
        n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        missing_rate = float(df.isna().sum().sum()) / (n_samples * n_features)

        # 类别不平衡
        if df[target_col].nunique() == 2:
            class_dist = df[target_col].value_counts(normalize=True)
            imbalance_ratio = float(class_dist.min() / class_dist.max())
        else:
            imbalance_ratio = 1.0

        high_corr_pairs = []
        if eda_summary and "high_corr_pairs" in eda_summary:
            high_corr_pairs = eda_summary["high_corr_pairs"][:5]
        outlier_summary = {}
        if eda_summary and "outlier_rates" in eda_summary:
            outlier_summary = eda_summary["outlier_rates"]

        # 上次训练结果
        last_params = last_training.get("model_params", {}) if last_training else {}
        last_auc = last_training.get("test_auc") if last_training else None

        current_params_str = (
                json.dumps(last_params, indent=2, ensure_ascii=False, default=_json_default)
            if last_params
            else f"首次训练 {model_type}，无历史参数"
        )
        history_text = f"上次训练 AUC: {last_auc:.4f}" if last_auc else "首次训练，无历史"

        # 参数边界说明
        param_range_text = ""
        if param_ranges:
            range_lines = []
            for name, info in param_ranges.items():
                if info.get("options"):
                    range_lines.append(f"  - {name}: 选项 {info['options']}（默认 {info.get('default')}）")
                else:
                    range_lines.append(
                        f"  - {name}: [{info.get('min')}, {info.get('max')}]（默认 {info.get('default')}）"
                    )
            param_range_text = "\n## UI 参数边界（必须严格遵守）\n" + "\n".join(range_lines)

        prompt = f"""你是一位机器学习调参专家，请根据数据画像为 **{model_type}** 模型推荐合适的超参数。

## 当前数据画像
- 样本量：{n_samples}
- 特征数：{n_features}（数值：{n_numeric}，类别：{n_categorical}）
- 缺失率：{missing_rate:.2%}
- 类别不平衡比（少数/多数）：{imbalance_ratio:.4f}
- 高相关特征对：{high_corr_pairs if high_corr_pairs else '无'}
- 异常值概况：{outlier_summary if outlier_summary else '无'}
- 目标列：{target_col}

## 历史信息
- 当前参数：{current_params_str}
- 训练历史：{history_text}

## 业务场景
{business_context or '通用机器学习建模'}
{param_range_text}

## 输出要求

请输出**严格的 JSON**，必须用 ```json 包围，格式如下：

```json
{{
  "data_analysis": "对数据特点的简短分析（80-150 字），点明样本量级、平衡度、特征分布对调参的影响",
  "strategy": "本次调参的总体策略（80-120 字），例如「降低过拟合风险 + 加速收敛」",
  "params": {{
    "param_name_1": value,
    "param_name_2": value
  }},
  "param_reasons": {{
    "param_name_1": "为什么推荐这个值（30-50 字）",
    "param_name_2": "..."
  }},
  "expected_improvement": "对比默认/上次参数预期的改进点（50-100 字）"
}}
```

约束：
- `params` 中只包含适用于 {model_type} 的参数，参数值必须在「UI 参数边界」内
- 所有文字字段必须使用中文
- 必须给出 `param_reasons`（每个推荐参数都要有理由）
- JSON 之外可附自由文本说明，但 JSON 必须完整有效
"""

        messages = [{"role": "user", "content": prompt}]
        return self.llm_manager.chat(messages, temperature=0.3, max_tokens=2500)


# 单例模式
_analyzer_instance: Optional[IntelligentAnalyzer] = None


def get_intelligent_analyzer() -> IntelligentAnalyzer:
    """获取智能分析器单例"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = IntelligentAnalyzer()
    return _analyzer_instance
