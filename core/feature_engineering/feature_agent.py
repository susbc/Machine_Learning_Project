"""
Feature Agent - 特征工程智能体

Phase 2 核心组件：扮演"特征工程师"角色，基于统计画像和 EDA 报告，输出特征处理策略 JSON。

核心设计理念:
- LLM 只输出 JSON 策略 (不写代码!)
- 严格 Schema 约束，防止幻觉
- 动作枚举，安全可执行

输入：eda_summary.json + EDA 诊断报告
输出：特征处理策略 JSON 数组
"""

from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class FeatureAgent:
    """
    特征工程智能体
    
    职责:
    1. 阅读统计画像和 EDA 诊断报告
    2. 输出特征处理策略 (JSON Schema)
    3. 严格约束动作枚举 (drop/fillna/log_transform 等)
    4. 禁止生成 Python 代码
    """
    
    # 允许的处理动作枚举
    ALLOWED_ACTIONS = {
        "drop": "删除字段",
        "fillna_mean": "均值填充",
        "fillna_median": "中位数填充",
        "fillna_mode": "众数填充",
        "fillna_constant": "常数填充",
        "log_transform": "对数变换 (log1p)",
        "sqrt_transform": "平方根变换",
        "boxcox_transform": "Box-Cox 变换",
        "onehot_encode": "One-Hot 编码",
        "label_encode": "Label 编码",
        "target_encode": "Target Encoding",
        "frequency_encode": "频率编码",
        "extract_hour": "提取小时",
        "extract_day": "提取星期",
        "extract_month": "提取月份",
        "extract_year": "提取年份",
        "binning": "分箱处理",
        "interaction": "特征交叉",
        "scale_standard": "标准化 (Z-score)",
        "scale_minmax": "Min-Max 归一化",
        "scale_robust": "Robust 标准化"
    }
    
    # 系统提示词模板
    SYSTEM_PROMPT = """
# Role
你是一位高级机器学习算法工程师，专注于自动化特征工程。你需要根据上游传来的《统计画像》和《EDA 诊断报告》，为数据集制定自动化的特征清洗和衍生策略。

# Input
1. 统计画像 (JSON):
<eda_summary>
{eda_summary}
</eda_summary>

2. EDA 诊断报告:
<eda_report>
{eda_report}
</eda_report>

# Strict JSON Output Rules
你必须且只能输出一个严格遵循以下 Schema 的 JSON 数组，严禁生成任何 Python 代码，严禁附带 Markdown 标记 (如 ```json):

[
  {{
    "column": "字段名 (必须存在于输入数据中)",
    "action": "处理动作 (必须是以下枚举值之一): {allowed_actions}",
    "reason": "简述为什么这样做 (20 字以内)",
    "params": {{}}  // 可选参数，如 {{"value": 0}} 用于 fillna_constant
  }}
]

# Constraint (必须严格遵守)
1. **高基数类别特征**: unique_values > 100 的字段，绝对禁止使用 onehot_encode，必须使用 target_encode 或 frequency_encode
2. **严重偏态**: skewness > 3 的数值字段，建议使用 log_transform
3. **高缺失率**: null_rate > 50% 的字段，建议直接 drop
4. **单一值字段**: unique_values = 1 的字段，直接 drop
5. **时间字段**: 建议提取小时/星期/月份等周期性特征
6. **数值字段**: 如果偏度严重，建议 log_transform 或 sqrt_transform
7. **类别字段**: 低基数 (unique_values < 10) 可使用 onehot_encode，高基数使用 target_encode

# Examples (参考示例)
输入统计画像包含:
- driver_age (numeric, null_rate=0.05, skewness=0.2)
- city_code (categorical, unique_values=342)
- registration_time (datetime)
- is_fraud (target)

正确输出:
[
  {{"column": "driver_age", "action": "fillna_median", "reason": "数值型缺失 5%，中位数填充更鲁棒", "params": {{}}}},
  {{"column": "city_code", "action": "target_encode", "reason": "高基数类别特征 (342 种)，OneHot 会维度爆炸", "params": {{}}}},
  {{"column": "registration_time", "action": "extract_hour", "reason": "时间字段提取小时特征", "params": {{}}}}
]

# Output
直接输出 JSON 数组，不要有任何额外文字。
"""
    
    def __init__(self, llm_provider=None):
        """
        初始化 Feature Agent
        
        Args:
            llm_provider: LLM 提供者
        """
        self.llm_provider = llm_provider
        self.allowed_actions = list(self.ALLOWED_ACTIONS.keys())
    
    def generate_strategy(self, eda_summary: Dict[str, Any],
                          eda_report: str = "") -> List[Dict[str, Any]]:
        """
        生成特征工程策略
        
        Args:
            eda_summary: 统计特征画像
            eda_report: EDA 诊断报告文本
            
        Returns:
            List[Dict]: 特征处理策略列表
                [
                    {"column": "driver_age", "action": "fillna_median", "reason": "...", "params": {}},
                    ...
                ]
        """
        logger.info("开始生成特征工程策略")
        
        # 1. 构建 Prompt
        prompt = self.SYSTEM_PROMPT.format(
            eda_summary=json.dumps(eda_summary, indent=2, ensure_ascii=False),
            eda_report=eda_report[:2000] if eda_report else "",  # 限制长度
            allowed_actions=", ".join(self.allowed_actions)
        )
        
        # 2. 调用 LLM
        llm_response = self._call_llm(prompt)
        
        # 3. 解析 JSON 响应
        strategy = self._parse_strategy(llm_response, eda_summary)
        
        logger.info(f"生成 {len(strategy)} 条特征处理策略")
        
        return strategy
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 生成策略"""
        if not self.llm_provider:
            logger.warning("未配置 LLM，使用规则引擎生成特征策略")
            return self._rule_based_strategy()
        
        try:
            response = self.llm_provider.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM 调用失败：{e}")
            return self._rule_based_strategy()
    
    def _rule_based_strategy(self) -> str:
        """基于规则的特征策略 (LLM 不可用时的降级方案)"""
        # 返回空 JSON 数组
        return "[]"
    
    def _parse_strategy(self, llm_response: str,
                        eda_summary: Dict) -> List[Dict]:
        """
        解析 LLM 响应，提取特征策略
        
        Args:
            llm_response: LLM 生成的文本
            eda_summary: 统计画像
            
        Returns:
            List[Dict]: 策略列表
        """
        import re
        
        try:
            # 清理 Markdown 标记
            text = llm_response.strip()
            text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^```\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
            
            # 提取 JSON 数组
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                json_str = match.group()
                strategy = json.loads(json_str)
                
                # 验证和清理策略
                validated_strategy = self._validate_strategy(strategy, eda_summary)
                return validated_strategy
            else:
                logger.error(f"无法从响应中提取 JSON 数组：{llm_response[:200]}")
                return []
                
        except Exception as e:
            logger.error(f"JSON 解析失败：{e}")
            return []
    
    def _validate_strategy(self, strategy: List[Dict],
                           eda_summary: Dict) -> List[Dict]:
        """
        验证特征策略的合法性
        
        Args:
            strategy: 策略列表
            eda_summary: 统计画像
            
        Returns:
            List[Dict]: 验证后的策略
        """
        validated = []
        features = eda_summary.get("features", {})
        valid_columns = set(features.keys())
        
        for rule in strategy:
            # 检查必填字段
            if "column" not in rule or "action" not in rule:
                logger.warning(f"策略缺少必填字段：{rule}")
                continue
            
            col = rule["column"]
            action = rule["action"]
            
            # 检查列名是否存在 (防幻觉)
            if col not in valid_columns:
                logger.warning(f"策略引用了不存在的列：{col}，已跳过")
                continue
            
            # 检查动作是否合法
            if action not in self.allowed_actions:
                logger.warning(f"策略使用了非法动作：{action}，已跳过")
                continue
            
            # 检查高基数类别特征是否错误使用 OneHot
            feature_stats = features.get(col, {})
            if feature_stats.get("type") == "categorical":
                unique_vals = feature_stats.get("unique_values", 0)
                if unique_vals > 100 and action == "onehot_encode":
                    logger.warning(f"高基数特征 {col} ({unique_vals} 个值) 错误使用 OneHot，已改为 Target Encoding")
                    rule["action"] = "target_encode"
                    rule["reason"] = "自动修正：高基数类别特征使用 Target Encoding"
            
            # 添加默认参数
            if "params" not in rule:
                rule["params"] = {}
            
            validated.append(rule)
        
        return validated
    
    def get_allowed_actions(self) -> Dict[str, str]:
        """获取允许的动作字典"""
        return self.ALLOWED_ACTIONS.copy()


# ==================== 特征工程执行器 ====================

class FeatureEngineeringExecutor:
    """
    特征工程执行器
    
    安全执行 Feature Agent 生成的 JSON 策略，映射到 Pandas/Scikit-Learn 操作。
    
    核心设计:
    - 不执行任何 LLM 生成的代码
    - 只执行预定义的算子函数
    - 列名不存在时自动跳过 (防幻觉)
    """
    
    def __init__(self, df, target_col: str = None):
        """
        初始化执行器
        
        Args:
            df: pandas DataFrame
            target_col: 目标变量列名
        """
        import pandas as pd
        self.df = df.copy()
        self.target_col = target_col
        self.pd = pd
    
    def execute_strategy(self, strategy: List[Dict]) -> pd.DataFrame:
        """
        执行特征工程策略
        
        Args:
            strategy: Feature Agent 生成的策略列表
            
        Returns:
            pd.DataFrame: 处理后的 DataFrame
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"开始执行 {len(strategy)} 条特征处理策略")
        
        for i, rule in enumerate(strategy):
            col = rule.get("column")
            action = rule.get("action")
            params = rule.get("params", {})
            
            # 安全防线：列名不存在直接跳过
            if col not in self.df.columns:
                logger.warning(f"策略引用了不存在的列：{col}，已跳过")
                continue
            
            try:
                # 执行对应操作
                if action == "drop":
                    self._drop(col)
                elif action.startswith("fillna_"):
                    self._fillna(col, action, params)
                elif action.endswith("_transform"):
                    self._transform(col, action, params)
                elif action.endswith("_encode"):
                    self._encode(col, action, params)
                elif action.startswith("extract_"):
                    self._extract_time_feature(col, action, params)
                elif action.startswith("scale_"):
                    self._scale(col, action, params)
                elif action == "binning":
                    self._binning(col, params)
                elif action == "interaction":
                    self._interaction(col, params)
                else:
                    logger.warning(f"未知动作：{action}，已跳过")
                
                logger.debug(f"执行策略 {i+1}/{len(strategy)}: {col}.{action}")
                
            except Exception as e:
                logger.error(f"执行策略失败 {col}.{action}: {e}")
                # 继续执行其他策略，不中断
        
        logger.info(f"特征工程执行完成，最终形状：{self.df.shape}")
        return self.df
    
    def _drop(self, col: str):
        """删除字段"""
        self.df.drop(columns=[col], inplace=True)
    
    def _fillna(self, col: str, action: str, params: Dict):
        """填充缺失值"""
        if action == "fillna_mean":
            self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif action == "fillna_median":
            self.df[col].fillna(self.df[col].median(), inplace=True)
        elif action == "fillna_mode":
            self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
        elif action == "fillna_constant":
            value = params.get("value", 0)
            self.df[col].fillna(value, inplace=True)
    
    def _transform(self, col: str, action: str, params: Dict):
        """数值变换"""
        import numpy as np
        
        if action == "log_transform":
            self.df[col] = np.log1p(self.df[col])
        elif action == "sqrt_transform":
            self.df[col] = np.sqrt(self.df[col])
        elif action == "boxcox_transform":
            from scipy import stats
            self.df[col], _ = stats.boxcox(self.df[col] + 1)
    
    def _encode(self, col: str, action: str, params: Dict):
        """类别编码"""
        if action == "onehot_encode":
            dummies = self.pd.get_dummies(self.df[col], prefix=col)
            self.df = self.pd.concat([self.df, dummies], axis=1)
            self.df.drop(columns=[col], inplace=True)
        elif action == "label_encode":
            self.df[col] = self.df[col].astype("category").cat.codes
        elif action == "target_encode":
            self._target_encode(col)
        elif action == "frequency_encode":
            freq = self.df[col].value_counts(normalize=True).to_dict()
            self.df[col] = self.df[col].map(freq)
    
    def _target_encode(self, col: str):
        """Target Encoding (平滑)"""
        if not self.target_col or self.target_col not in self.df.columns:
            logger.warning("Target Encoding 需要目标变量，已跳过")
            return
        
        # 计算全局均值
        global_mean = self.df[self.target_col].mean()
        
        # 计算类别均值
        agg = self.df.groupby(col)[self.target_col].agg(["mean", "count"])
        
        # 平滑
        smooth = 10
        self.df[col] = (
            agg["mean"][self.df[col]].fillna(global_mean) * agg["count"][self.df[col]] + 
            global_mean * smooth
        ) / (agg["count"][self.df[col]] + smooth)
    
    def _extract_time_feature(self, col: str, action: str, params: Dict):
        """时间特征提取"""
        self.df[col] = self.pd.to_datetime(self.df[col])
        
        if action == "extract_hour":
            self.df[col] = self.df[col].dt.hour
        elif action == "extract_day":
            self.df[col] = self.df[col].dt.dayofweek
        elif action == "extract_month":
            self.df[col] = self.df[col].dt.month
        elif action == "extract_year":
            self.df[col] = self.df[col].dt.year
    
    def _scale(self, col: str, action: str, params: Dict):
        """特征缩放"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        if action == "scale_standard":
            scaler = StandardScaler()
        elif action == "scale_minmax":
            scaler = MinMaxScaler()
        elif action == "scale_robust":
            scaler = RobustScaler()
        else:
            return
        
        self.df[col] = scaler.fit_transform(self.df[[col]])
    
    def _binning(self, col: str, params: Dict):
        """分箱处理"""
        bins = params.get("bins", 5)
        self.df[col] = self.pd.qcut(self.df[col], q=bins, labels=False, duplicates="drop")
    
    def _interaction(self, col: str, params: Dict):
        """特征交叉"""
        with_cols = params.get("with", [])
        for other_col in with_cols:
            if other_col in self.df.columns:
                new_col = f"{col}_x_{other_col}"
                self.df[new_col] = self.df[col] * self.df[other_col]
