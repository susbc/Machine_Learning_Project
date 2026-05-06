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
