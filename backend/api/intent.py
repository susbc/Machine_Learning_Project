"""
Intent Recognition API - FastAPI 意图识别接口

提供用户意图识别能力，将自然语言需求转化为结构化的
机器学习任务意图，驱动后续自动化训练流程。

Endpoints:
- POST /api/intent/recognize - 识别用户意图
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.llm.intent_agent import IntentRecognitionAgent
from core.llm.intent_schemas import IntentResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/intent", tags=["Intent Recognition"])

# 全局 Agent 实例（延迟初始化，首次请求时创建）
_intent_agent: Optional[IntentRecognitionAgent] = None


def _get_agent() -> IntentRecognitionAgent:
    """获取或创建全局 IntentRecognitionAgent 实例"""
    global _intent_agent
    if _intent_agent is None:
        _intent_agent = IntentRecognitionAgent(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv(
                "LLM_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            model_name=os.getenv("INTENT_MODEL", "qwen-plus"),
        )
    return _intent_agent


# ==========================================
# Request / Response Models
# ==========================================

class IntentRequest(BaseModel):
    """意图识别请求"""
    user_input: str = Field(
        ...,
        description="用户自然语言输入",
        examples=["我要预测近3个月哪些订单可能会欺诈"]
    )


class IntentResponse(BaseModel):
    """意图识别响应"""
    status: str = Field(..., examples=["success", "error"])
    intent: Optional[IntentResult] = None
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="当置信度低时的追问建议"
    )
    msg: Optional[str] = None


# ==========================================
# Endpoints
# ==========================================

@router.post("/recognize", response_model=IntentResponse)
async def recognize_intent(req: IntentRequest):
    """
    识别用户意图

    将用户的自然语言需求解析为结构化的机器学习任务意图，
    包括场景分类、目标变量、评估指标、约束条件等。

    当识别置信度 < 0.6 时，会返回追问建议列表。
    """
    try:
        agent = _get_agent()

        intent = agent.recognize(req.user_input)

        # 低置信度时生成追问
        follow_up = []
        if intent.confidence < 0.6:
            follow_up = agent.get_follow_up_questions(intent)

        return IntentResponse(
            status="success",
            intent=intent,
            follow_up_questions=follow_up,
        )

    except Exception as e:
        logger.error(f"意图识别失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))
