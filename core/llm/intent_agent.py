"""
Intent Recognition Agent - 意图识别 Agent

通过 LLM 解析用户自然语言，识别业务场景、目标变量、评估指标、
约束条件等意图信息，为后续的数据提取和模型训练提供结构化输入。

设计要点:
- temperature=0.2（适度创造性，保持工程确定性）
- Pydantic 严格校验防止幻觉场景
- 低置信度时返回追问建议列表
- 规则降级：无法调用 LLM 时使用关键词匹配降级
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .intent_schemas import IntentResult

logger = logging.getLogger(__name__)


# ==========================================
# 规则降级：关键词匹配 fallback
# ==========================================

_KEYWORD_SCENARIO_MAP: Dict[str, str] = {
    "欺诈": "binary_classification",
    "流失": "binary_classification",
    "违约": "binary_classification",
    "逾期": "binary_classification",
    "预测.*会.*不.*会": "binary_classification",
    "分类": "binary_classification",
    "回归": "regression",
    "销量": "regression",
    "预测.*金额": "regression",
    "多分类": "multiclass",
    "异常": "anomaly_detection",
    "检测.*异常": "anomaly_detection",
}

_DEFAULT_METRIC_MAP: Dict[str, str] = {
    "binary_classification": "roc_auc",
    "multiclass": "accuracy",
    "regression": "rmse",
    "anomaly_detection": "f1",
}


def _rule_based_intent(user_input: str) -> IntentResult:
    """
    基于关键词的规则降级意图识别

    当 LLM 不可用时使用，提供最低可用的意图解析能力。
    """
    scenario = "binary_classification"  # 默认
    for pattern, sc in _KEYWORD_SCENARIO_MAP.items():
        import re
        if re.search(pattern, user_input):
            scenario = sc
            break

    return IntentResult(
        scenario=scenario,
        business_description=f"基于规则降级识别的意图：{user_input[:200]}",
        target_col_hint=None,
        candidate_tables=[],
        constraints=[],
        preferred_metric=_DEFAULT_METRIC_MAP.get(scenario, "auto"),
        max_training_time_minutes=30,
        deployment_mode="batch",
        confidence=0.4,  # 低置信度标记
    )


# ==========================================
# IntentRecognitionAgent
# ==========================================

class IntentRecognitionAgent:
    """
    意图识别 Agent

    使用 LLM + Pydantic 结构化输出，将用户自然语言解析为
    IntentResult 对象，驱动后续自动化训练流程。
    """

    SYSTEM_PROMPT = """你是一位资深机器学习产品经理和数据科学家。你的任务是分析用户的自然语言需求，并将其转化为结构化的机器学习任务意图。

# 你需要识别以下信息：

1. **场景分类** (scenario)：从以下四种中选择
   - binary_classification：二分类（如欺诈检测、客户流失、违约预测）
   - multiclass：多分类（如客户分群、风险等级分类）
   - regression：回归（如销量预测、金额预测）
   - anomaly_detection：异常检测（如异常订单检测、故障检测）

2. **业务目标描述** (business_description)：用一句话准确描述建模目标

3. **目标列提示** (target_col_hint)：如果用户明确或暗示了目标变量名，提取出来

4. **候选表** (candidate_tables)：用户提到的可能涉及的数据库表名

5. **约束条件** (constraints)：用户提出的限制条件（如时间范围、排除某些字段等）

6. **偏好评估指标** (preferred_metric)：如果用户指定了评估指标则提取，否则设为 "auto"

7. **最大训练时间** (max_training_time_minutes)：如果用户提到时间约束则提取，默认 30

8. **部署模式** (deployment_mode)：online（需要在线实时预测）或 batch（批量离线预测）

9. **置信度** (confidence)：你对本次意图识别的信心 (0.0-1.0)。如果用户需求模糊或缺少关键信息，请给低分

# 示例 1：
用户："我要预测近3个月哪些订单可能会欺诈"
→ scenario=binary_classification, target_col_hint=is_fraud, candidate_tables=[orders], constraints=[近3个月]

# 示例 2：
用户："帮我建个模型预测每个月的销售额"
→ scenario=regression, target_col_hint=sales_amount, preferred_metric=rmse

# 示例 3：
用户："找出数据中的异常订单"
→ scenario=anomaly_detection, deployment_mode=batch

# 约束：
- 输出必须是严格的 JSON 对象，符合提供的 Schema
- scenario 和 deployment_mode 只能从预定义的枚举值中选择
- confidence 要如实反映需求的清晰程度"""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "qwen-plus",
    ):
        """
        初始化意图识别 Agent

        Args:
            llm: 已有的 ChatOpenAI 实例（优先使用）
            api_key: API Key（llm 未提供时使用）
            base_url: API Base URL
            model_name: 模型名称
        """
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                temperature=0.2,
                model=model_name,
                api_key=api_key or "",
                base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        self._structured_llm = self.llm.with_structured_output(IntentResult)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{user_input}"),
        ])
        self._chain = self._prompt | self._structured_llm

        logger.info(f"IntentRecognitionAgent 初始化完成：{model_name} (temperature=0.2)")

    def recognize(self, user_input: str) -> IntentResult:
        """
        识别用户意图

        Args:
            user_input: 用户自然语言输入

        Returns:
            IntentResult: 结构化意图识别结果

        Raises:
            RuntimeError: LLM 调用失败且规则降级也失败
        """
        logger.info(f"IntentRecognitionAgent 正在分析意图...")

        try:
            result: IntentResult = self._chain.invoke({"user_input": user_input})
            logger.info(
                f"🎉 意图识别完成：scenario={result.scenario}, "
                f"confidence={result.confidence:.2f}"
            )
            return result

        except Exception as e:
            logger.warning(f"LLM 意图识别失败，降级到规则匹配：{e}")
            try:
                result = _rule_based_intent(user_input)
                logger.info(f"规则降级意图识别完成：scenario={result.scenario}")
                return result
            except Exception as e2:
                logger.error(f"规则降级也失败：{e2}")
                raise RuntimeError(f"意图识别完全失败：{e2}") from e2

    def get_follow_up_questions(self, intent: IntentResult) -> List[str]:
        """
        当置信度低时，生成追问建议列表

        Args:
            intent: 意图识别结果

        Returns:
            追问问题列表
        """
        questions = []

        if not intent.target_col_hint:
            questions.append("你想预测的目标变量是什么？（例如：是否欺诈、销售额、客户等级）")

        if not intent.candidate_tables:
            questions.append("数据存储在哪些表中？")

        if not intent.constraints:
            questions.append("是否有时间范围或其他筛选条件？（例如：仅近3个月数据）")

        if intent.scenario == "anomaly_detection" and not intent.target_col_hint:
            questions.append("什么情况下算异常？有没有已知的异常样本可以作为标注数据？")

        return questions


# ==========================================
# 便捷函数
# ==========================================

def recognize_intent(
    user_input: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "qwen-plus",
) -> IntentResult:
    """
    便捷函数：一键识别意图

    Args:
        user_input: 用户自然语言输入
        api_key: LLM API Key
        base_url: API Base URL
        model_name: 模型名称

    Returns:
        IntentResult: 结构化意图识别结果
    """
    agent = IntentRecognitionAgent(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )
    return agent.recognize(user_input)


__all__ = ["IntentRecognitionAgent", "recognize_intent", "_rule_based_intent"]
