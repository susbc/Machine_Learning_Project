"""
Intent Recognition Schemas - 意图识别 Pydantic Schema

用户自然语言输入的意图解析结果，包含场景分类、目标变量提取、
评估指标推荐、约束条件等。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class IntentResult(BaseModel):
    """
    意图识别结果

    由 LLM 解析用户自然语言后生成，用于驱动后续的数据提取、
    质量评估、模型训练等流程。
    """
    scenario: str = Field(
        ...,
        description="任务场景",
        examples=["binary_classification", "multiclass", "regression", "anomaly_detection"]
    )
    business_description: str = Field(
        ...,
        description="用一句话描述业务目标",
        max_length=500
    )
    target_col_hint: Optional[str] = Field(
        None,
        description="用户暗示或明确指定的目标列名"
    )
    candidate_tables: List[str] = Field(
        default_factory=list,
        description="用户提到的可能表名"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="约束条件（时间范围、排除字段等）"
    )
    preferred_metric: str = Field(
        "auto",
        description="偏好评估指标，'auto' 表示由系统自动推荐"
    )
    max_training_time_minutes: int = Field(
        30,
        description="最大训练时间（分钟）",
        ge=1,
        le=480
    )
    deployment_mode: str = Field(
        "batch",
        description="部署模式：online（在线实时）或 batch（批量离线）"
    )
    confidence: float = Field(
        0.0,
        description="意图识别置信度 (0.0 - 1.0)",
        ge=0.0,
        le=1.0
    )

    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, v: str) -> str:
        allowed = {"binary_classification", "multiclass", "regression", "anomaly_detection"}
        if v not in allowed:
            raise ValueError(
                f"无效的场景 '{v}'，必须是 {', '.join(sorted(allowed))} 之一"
            )
        return v

    @field_validator("deployment_mode")
    @classmethod
    def validate_deployment_mode(cls, v: str) -> str:
        allowed = {"online", "batch"}
        if v not in allowed:
            raise ValueError(f"无效的部署模式 '{v}'，必须是 {', '.join(sorted(allowed))} 之一")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于序列化）"""
        return self.model_dump()

    def to_prompt_context(self) -> str:
        """
        生成用于下游 Agent 的 Prompt 上下文字符串
        """
        lines = [
            f"业务场景：{self.scenario}",
            f"业务目标：{self.business_description}",
        ]
        if self.target_col_hint:
            lines.append(f"目标列（用户指定）：{self.target_col_hint}")
        if self.candidate_tables:
            lines.append(f"候选表：{', '.join(self.candidate_tables)}")
        if self.constraints:
            lines.append(f"约束条件：{'；'.join(self.constraints)}")
        if self.preferred_metric != "auto":
            lines.append(f"偏好指标：{self.preferred_metric}")
        lines.append(f"最大训练时间：{self.max_training_time_minutes} 分钟")
        lines.append(f"部署模式：{self.deployment_mode}")
        return "\n".join(lines)


__all__ = ["IntentResult"]
