"""
AutoML Agent - 自动机器学习智能体

Phase 3 核心组件：根据数据特征智能推荐模型，生成 Optuna 搜索空间配置。

核心设计理念:
- Agent 负责"智能决策" (选模型 + 定搜索空间)
- Python 负责"执行寻优" (Optuna 贝叶斯优化)
- 输出结构化 JSON 配置 (可追溯/可调整)
"""

from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class AutoMLAgent:
    """
    AutoML 智能体
    
    职责:
    1. 分析数据特征 (样本量/特征数/类别占比/不平衡度)
    2. 推荐候选模型 (LightGBM/XGBoost/CatBoost/RF)
    3. 生成 Optuna 搜索空间配置 (JSON)
    4. 说明推荐理由 (rationale)
    """
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        "lightgbm": {
            "description": "LightGBM - 梯度提升树，训练速度快",
            "best_for": ["大规模数据", "高维特征", "快速训练"],
            "weakness": ["类别特征需预处理", "小样本易过拟合"]
        },
        "xgboost": {
            "description": "XGBoost - 经典梯度提升树，稳定性好",
            "best_for": ["中小规模数据", "需要稳定性", "Kaggle 竞赛"],
            "weakness": ["训练较慢", "内存占用高"]
        },
        "catboost": {
            "description": "CatBoost - 原生支持类别特征",
            "best_for": ["大量类别特征", "不平衡数据", "自动处理缺失值"],
            "weakness": ["训练速度慢", "内存占用高"]
        },
        "random_forest": {
            "description": "Random Forest - 随机森林，鲁棒性强",
            "best_for": ["基线模型", "特征重要性", "小样本"],
            "weakness": ["精度上限较低", "大规模数据慢"]
        },
        "logistic_regression": {
            "description": "Logistic Regression - 逻辑回归，可解释性强",
            "best_for": ["线性可分", "需要可解释性", "基线对比"],
            "weakness": ["非线性关系", "特征工程要求高"]
        }
    }
    
    # 系统提示词模板
    SYSTEM_PROMPT = """
# Role
你是一位资深机器学习算法专家，专注于 AutoML 和模型自动选择。你的任务是根据数据集的特征，推荐最合适的树模型，并为 Optuna 贝叶斯优化生成搜索空间配置。

# Input
数据集特征:
<data_features>
{data_features}
</data_features>

业务场景:
<business_scenario>
{business_scenario}
</business_scenario>

# Task
请根据数据特征和业务场景，完成以下任务:

## 1. 模型推荐
从以下模型中选择 2-3 个最合适的候选模型:
- **lightgbm**: 训练速度快，适合大规模数据
- **xgboost**: 稳定性好，Kaggle 竞赛常用
- **catboost**: 原生支持类别特征，适合不平衡数据
- **random_forest**: 鲁棒性强，适合基线
- **logistic_regression**: 可解释性强，适合线性问题

## 2. 搜索空间配置
为每个推荐模型配置 Optuna 超参数搜索空间:
- **learning_rate**: 学习率 (loguniform, 0.01-0.3)
- **n_estimators**: 树数量 (int, 100-1000)
- **max_depth/num_leaves**: 树深度/叶子数 (int)
- **正则化参数**: l2_leaf_reg, reg_alpha, reg_lambda (loguniform)
- **类别不平衡处理**: scale_pos_weight, auto_class_weights (categorical)

## 3. 输出格式
你必须且只能输出一个严格的 JSON 对象，不要包含 Markdown 标记:

{{
 "task_type": "binary_classification" | "multiclass_classification" | "regression",
 "optimization_metric": "roc_auc" | "accuracy" | "f1" | "rmse",
 "candidate_models": ["lightgbm", "catboost"],
 "n_trials": 30,
 "rationale": "详细说明为什么推荐这些模型，考虑了哪些数据特征",
 "search_spaces": {{
  "lightgbm": {{
   "learning_rate": {{"type": "loguniform", "low": 0.01, "high": 0.2}},
   "num_leaves": {{"type": "int", "low": 15, "high": 127}},
   "scale_pos_weight": {{"type": "categorical", "choices": [1, 10, 50]}}
  }},
  "catboost": {{
   "depth": {{"type": "int", "low": 4, "high": 10}},
   "l2_leaf_reg": {{"type": "int", "low": 1, "high": 10}},
   "auto_class_weights": {{"type": "categorical", "choices": ["Balanced", "SqrtBalanced"]}}
  }}
 }}
}}

# Decision Rules (决策规则)
请严格遵循以下经验法则:

1. **样本量 > 50 万**: 优先 LightGBM (训练速度快)
2. **类别特征占比 > 30%**: 必须包含 CatBoost (原生支持)
3. **正负样本比例 < 1:10**: 添加类别权重参数 (scale_pos_weight)
4. **特征数 > 100**: 增加正则化参数搜索范围
5. **小样本 (< 1 万)**: 包含 Random Forest 作为基线
6. **需要可解释性**: 包含 Logistic Regression 作为对比
7. **Kaggle 竞赛场景**: 优先 XGBoost + LightGBM 组合

# Examples

## 示例 1: 不平衡二分类 (网约车反欺诈)
输入：50 万行，120 维特征，45 个类别特征，正负样本比例 1:99
输出:
{{
 "task_type": "binary_classification",
 "optimization_metric": "roc_auc",
 "candidate_models": ["lightgbm", "catboost"],
 "n_trials": 30,
 "rationale": "样本极度不平衡且存在较多类别特征。CatBoost 原生支持类别特征和不平衡数据处理，LightGBM 训练速度快适合做基线对比。",
 "search_spaces": {{
  "lightgbm": {{
   "learning_rate": {{"type": "loguniform", "low": 0.01, "high": 0.2}},
   "num_leaves": {{"type": "int", "low": 15, "high": 127}},
   "scale_pos_weight": {{"type": "categorical", "choices": [10, 50, 99]}}
  }},
  "catboost": {{
   "depth": {{"type": "int", "low": 4, "high": 10}},
   "l2_leaf_reg": {{"type": "int", "low": 1, "high": 10}},
   "auto_class_weights": {{"type": "categorical", "choices": ["Balanced", "SqrtBalanced"]}}
  }}
 }}
}}

# Output
直接输出 JSON 对象，不要有任何额外文字。
"""
    
    def __init__(self, llm_provider=None):
        """
        初始化 AutoML Agent
        
        Args:
            llm_provider: LLM 提供者
        """
        self.llm_provider = llm_provider
    
    def generate_config(self, 
                        data_features: Dict[str, Any],
                        business_scenario: str = "通用机器学习建模") -> Dict[str, Any]:
        """
        生成 AutoML 配置
        
        Args:
            data_features: 数据特征字典
                {
                    "n_samples": 500000,
                    "n_features": 120,
                    "n_categorical": 45,
                    "categorical_ratio": 0.375,
                    "class_imbalance_ratio": 99,  # 正负样本比例
                    "task_type": "binary_classification",
                    "target_col": "is_fraud"
                }
            business_scenario: 业务场景描述
            
        Returns:
            Dict: AutoML 配置 (JSON Schema)
        """
        logger.info(f"开始生成 AutoML 配置，业务场景：{business_scenario}")
        
        # 1. 构建 Prompt
        prompt = self.SYSTEM_PROMPT.format(
            data_features=self._format_data_features(data_features),
            business_scenario=business_scenario
        )
        
        # 2. 调用 LLM
        llm_response = self._call_llm(prompt)
        
        # 3. 解析 JSON 响应
        config = self._parse_config(llm_response, data_features)
        
        logger.info(f"生成 AutoML 配置：{len(config.get('candidate_models', []))} 个候选模型，"
                   f"{config.get('n_trials', 0)} 次 Trial")
        
        return config
    
    def _format_data_features(self, data_features: Dict) -> str:
        """格式化数据特征为文本"""
        lines = []
        lines.append(f"- 样本量：{data_features.get('n_samples', 0):,} 行")
        lines.append(f"- 特征数：{data_features.get('n_features', 0)} 维")
        lines.append(f"- 类别特征：{data_features.get('n_categorical', 0)} 个 "
                    f"(占比 {data_features.get('categorical_ratio', 0):.1%})")
        
        if data_features.get('class_imbalance_ratio'):
            lines.append(f"- 正负样本比例：1:{data_features.get('class_imbalance_ratio')} "
                        f"({'极度不平衡' if data_features.get('class_imbalance_ratio') > 50 else '轻微不平衡'})")
        
        lines.append(f"- 任务类型：{data_features.get('task_type', 'unknown')}")
        
        return "\n".join(lines)
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 生成配置"""
        if not self.llm_provider:
            logger.warning("未配置 LLM，使用规则引擎生成 AutoML 配置")
            return self._rule_based_config()
        
        try:
            response = self.llm_provider.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM 调用失败：{e}")
            return self._rule_based_config()
    
    def _rule_based_config(self) -> str:
        """基于规则的 AutoML 配置 (LLM 不可用时的降级方案)"""
        # 返回默认配置
        default_config = {
            "task_type": "binary_classification",
            "optimization_metric": "roc_auc",
            "candidate_models": ["lightgbm", "catboost"],
            "n_trials": 30,
            "rationale": "默认配置：LightGBM 训练速度快，CatBoost 处理类别特征效果好",
            "search_spaces": {
                "lightgbm": {
                    "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.2},
                    "num_leaves": {"type": "int", "low": 15, "high": 127},
                    "n_estimators": {"type": "int", "low": 100, "high": 500}
                },
                "catboost": {
                    "depth": {"type": "int", "low": 4, "high": 10},
                    "l2_leaf_reg": {"type": "int", "low": 1, "high": 10},
                    "n_estimators": {"type": "int", "low": 100, "high": 500}
                }
            }
        }
        return json.dumps(default_config)
    
    def _parse_config(self, llm_response: str,
                      data_features: Dict) -> Dict[str, Any]:
        """
        解析 LLM 响应，提取配置
        
        Args:
            llm_response: LLM 生成的文本
            data_features: 数据特征 (用于验证)
            
        Returns:
            Dict: 配置字典
        """
        import re
        
        try:
            # 清理 Markdown 标记
            text = llm_response.strip()
            text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^```\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
            
            # 提取 JSON 对象
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group()
                config = json.loads(json_str)
                
                # 验证配置合法性
                validated_config = self._validate_config(config, data_features)
                return validated_config
            else:
                logger.error(f"无法从响应中提取 JSON: {llm_response[:200]}")
                return json.loads(self._rule_based_config())
                
        except Exception as e:
            logger.error(f"JSON 解析失败：{e}")
            return json.loads(self._rule_based_config())
    
    def _validate_config(self, config: Dict,
                         data_features: Dict) -> Dict:
        """
        验证配置合法性
        
        Args:
            config: 配置字典
            data_features: 数据特征
            
        Returns:
            Dict: 验证后的配置
        """
        # 检查必填字段
        required_fields = ["task_type", "optimization_metric", "candidate_models", 
                          "n_trials", "rationale", "search_spaces"]
        
        for field in required_fields:
            if field not in config:
                logger.warning(f"配置缺少必填字段：{field}，使用默认值")
                if field == "task_type":
                    config[field] = "binary_classification"
                elif field == "optimization_metric":
                    config[field] = "roc_auc"
                elif field == "candidate_models":
                    config[field] = ["lightgbm", "catboost"]
                elif field == "n_trials":
                    config[field] = 30
                elif field == "rationale":
                    config[field] = "默认配置"
                elif field == "search_spaces":
                    config[field] = {}
        
        # 验证模型是否在支持列表中
        valid_models = set(self.SUPPORTED_MODELS.keys())
        for model in config["candidate_models"]:
            if model not in valid_models:
                logger.warning(f"不支持的模型：{model}，已移除")
                config["candidate_models"].remove(model)
        
        # 确保至少有 1 个候选模型
        if not config["candidate_models"]:
            config["candidate_models"] = ["lightgbm"]
        
        # 验证搜索空间
        for model in config["candidate_models"]:
            if model not in config["search_spaces"]:
                logger.warning(f"模型 {model} 缺少搜索空间，使用默认配置")
                config["search_spaces"][model] = self._get_default_search_space(model)
        
        return config
    
    def _get_default_search_space(self, model: str) -> Dict:
        """获取默认搜索空间"""
        defaults = {
            "lightgbm": {
                "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.2},
                "num_leaves": {"type": "int", "low": 15, "high": 127},
                "n_estimators": {"type": "int", "low": 100, "high": 500}
            },
            "catboost": {
                "depth": {"type": "int", "low": 4, "high": 10},
                "l2_leaf_reg": {"type": "int", "low": 1, "high": 10},
                "n_estimators": {"type": "int", "low": 100, "high": 500}
            },
            "xgboost": {
                "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.2},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "n_estimators": {"type": "int", "low": 100, "high": 500}
            },
            "random_forest": {
                "n_estimators": {"type": "int", "low": 100, "high": 500},
                "max_depth": {"type": "int", "low": 3, "high": 15},
                "min_samples_split": {"type": "int", "low": 2, "high": 20}
            }
        }
        return defaults.get(model, defaults["lightgbm"])
    
    def get_supported_models(self) -> Dict[str, Dict]:
        """获取支持的模型列表"""
        return self.SUPPORTED_MODELS.copy()


def extract_data_features(df, target_col: str) -> Dict[str, Any]:
    """
    提取数据特征 (用于 AutoML Agent 输入)
    
    Args:
        df: pandas DataFrame
        target_col: 目标变量列名
        
    Returns:
        Dict: 数据特征字典
    """
    import pandas as pd
    import numpy as np
    
    n_samples, n_features = df.shape
    feature_cols = [col for col in df.columns if col != target_col]
    
    # 推断特征类型
    n_categorical = 0
    for col in feature_cols:
        dtype = str(df[col].dtype)
        if df[col].nunique() < 20 or dtype == "object" or "category" in dtype:
            n_categorical += 1
    
    # 计算类别不平衡比例 (二分类任务)
    class_imbalance_ratio = None
    if target_col in df.columns:
        target_data = df[target_col]
        if target_data.nunique() == 2:  # 二分类
            value_counts = target_data.value_counts()
            if len(value_counts) == 2:
                class_imbalance_ratio = int(value_counts.iloc[0] / value_counts.iloc[1])
    
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_categorical": n_categorical,
        "categorical_ratio": n_categorical / n_features if n_features > 0 else 0,
        "class_imbalance_ratio": class_imbalance_ratio,
        "task_type": _infer_task_type(df, target_col),
        "target_col": target_col
    }


def _infer_task_type(df, target_col: str) -> str:
    """推断任务类型"""
    if target_col not in df.columns:
        return "unknown"
    
    target_data = df[target_col]
    n_unique = target_data.nunique()
    
    if n_unique == 2:
        return "binary_classification"
    elif n_unique < 20:
        return "multiclass_classification"
    else:
        return "regression"
