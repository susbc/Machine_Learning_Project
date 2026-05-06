"""
Agent Orchestrator - LangChain 编排器

利用 LangChain LCEL + Pydantic 结构化输出，打造防崩溃 (Crash-proof) 的 Agent 核心。

核心特性:
1. .with_structured_output() - 强制 LLM 输出合规 JSON
2. Pydantic 严格校验 - 枚举约束防止幻觉
3. Retry 自愈机制 - 自动重试处理偶发错误
4. 对象化流转 - 返回 Pydantic 对象而非字符串
"""

import logging
from typing import Optional, Type
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from .agent_schemas import (
    SQLGenerationResult,
    FeatureStrategy,
    AutoMLConfig,
    EDAReport,
)

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Agent 编排器
    
    统一管理所有 LLM Agent 的调用，提供结构化输出和防崩机制。
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: str = "qwen-plus",
        temperature: float = 0.0,
        max_retries: int = 3
    ):
        """
        初始化编排器
        
        Args:
            api_key: LLM API Key
            base_url: API 基础 URL (可选，用于兼容通义千问等)
            model_name: 模型名称
            temperature: 温度 (0.0 保证工程输出确定性)
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model_name,
            api_key=api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        logger.info(f"AgentOrchestrator 初始化完成：{model_name} (temperature={temperature})")
    
    def _create_structured_llm(self, output_schema: Type[BaseModel]):
        """
        创建结构化 LLM
        
        Args:
            output_schema: Pydantic Schema 类
            
        Returns:
            绑定结构化输出的 LLM
        """
        return self.llm.with_structured_output(output_schema)
    
    def run_feature_agent(
        self,
        eda_summary_json: str,
        eda_report_text: str,
        max_retries: Optional[int] = None
    ) -> FeatureStrategy:
        """
        特征工程 Agent (自带 Retry 防崩机制)
        
        Args:
            eda_summary_json: 统计画像 JSON
            eda_report_text: EDA 诊断报告文本
            max_retries: 最大重试次数
            
        Returns:
            FeatureStrategy: 严格校验后的特征策略对象
            
        Raises:
            RuntimeError: 超过最大重试次数仍失败
        """
        retries = max_retries or self.max_retries
        
        # 1. 绑定 Pydantic 结构化输出
        structured_llm = self._create_structured_llm(FeatureStrategy)
        
        # 2. 定义 Prompt 模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位高级机器学习算法工程师。你需要根据上游传来的《统计画像》和《EDA 诊断报告》，为数据集制定自动化的特征清洗和衍生策略。

# 约束条件：
- 对于高基数类别特征 (>100 种)，必须使用 target_encode。
- 对于缺失率 > 50% 的特征，强制执行 drop。
- 对于偏度 (skewness) > 3 的数值特征，建议使用 log_transform。
- 你的输出必须严格符合预期的 JSON Schema，严禁任何代码和 Markdown。
- action 字段只能从预定义的枚举值中选择，严禁自创方法。
- reason 字段必须使用中文，且不超过 50 字，务必精简！"""),
            
            ("human", """统计画像：
{eda_summary}

EDA 诊断报告：
{eda_report}

请生成特征处理策略。""")
        ])
        
        # 3. 组装 Chain
        chain = prompt | structured_llm
        
        # 4. 执行与自愈循环 (Auto-Healing Loop)
        last_error = None
        for attempt in range(retries):
            try:
                logger.info(f"Feature_Agent 正在生成特征策略... (尝试 {attempt + 1}/{retries})")
                
                # 调用大模型并自动反序列化为 Pydantic 对象
                result: FeatureStrategy = chain.invoke({
                    "eda_summary": eda_summary_json,
                    "eda_report": eda_report_text
                })
                
                logger.info(f"🎉 成功生成合规的特征策略！共 {len(result.actions)} 个动作")
                return result
            
            except Exception as e:
                last_error = e
                logger.warning(
                    f"大模型输出格式违规或产生幻觉，被 Pydantic 拦截 (尝试 {attempt + 1}/{retries}): {str(e)}"
                )
                
                # 最后一次重试失败
                if attempt == retries - 1:
                    logger.error(f"达到最大重试次数，Feature_Agent 执行失败：{e}")
                    raise RuntimeError(
                        f"无法生成符合规范的特征策略，请检查输入数据或降低要求。"
                        f"最后错误：{str(e)}"
                    ) from e
        
        # 理论上不会到这里
        raise RuntimeError(f"Feature_Agent 执行异常：{last_error}")
    
    def run_data_retrieval_agent(
        self,
        graph_schema: str,
        user_query: str
    ) -> SQLGenerationResult:
        """
        数据拉取 Agent (NL2SQL)
        
        Args:
            graph_schema: 图谱 Schema 文本
            user_query: 用户自然语言查询
            
        Returns:
            SQLGenerationResult: 严格校验后的 SQL 生成结果
        """
        # 1. 绑定结构化输出
        structured_llm = self._create_structured_llm(SQLGenerationResult)
        
        # 2. 定义 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位顶级的数据库架构师和高级数据开发工程师。你的核心任务是根据用户提供的业务需求，结合图数据库检索出的"局部表结构与血缘关联"，编写出精准、高效、无语法的 SQL 查询语句。

# 严格约束：
1. 【禁止幻觉】只能使用 <graph_schema> 中明确提供的表名和字段名！
2. 【严格关联】多表查询必须严格按照 <graph_schema> 中的关联关系进行 JOIN！
3. 【方言适配】SQL 必须符合指定数据库的语法规范！
4. 【精准选取】禁止 `SELECT *`，明确 SELECT 所需字段！
5. 【安全阻断】如果需求无法实现，设置 is_executable=false 并说明缺失！

# 输出格式：
你必须输出严格的 JSON 对象，包含 thought_process, is_executable, sql 三个字段。"""),
            
            ("human", """图谱信息：
{graph_schema}

用户需求：
{user_query}

请生成 SQL 查询。""")
        ])
        
        # 3. 组装 Chain
        chain = prompt | structured_llm
        
        # 4. 执行 (单次调用，无需重试)
        logger.info("DataRetrieval_Agent 正在生成 SQL...")
        result: SQLGenerationResult = chain.invoke({
            "graph_schema": graph_schema,
            "user_query": user_query
        })
        
        logger.info(f"🎉 SQL 生成完成，可执行：{result.is_executable}")
        return result
    
    def run_automl_agent(
        self,
        data_features: str,
        business_scenario: str = "通用机器学习建模"
    ) -> AutoMLConfig:
        """
        AutoML 智能体 (模型推荐 + 搜索空间生成)
        
        Args:
            data_features: 数据特征描述
            business_scenario: 业务场景
            
        Returns:
            AutoMLConfig: 严格校验后的 AutoML 配置
        """
        # 1. 绑定结构化输出
        structured_llm = self._create_structured_llm(AutoMLConfig)
        
        # 2. 定义 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位资深机器学习算法专家，专注于 AutoML 和模型自动选择。你的任务是根据数据集的特征，推荐最合适的树模型，并为 Optuna 贝叶斯优化生成搜索空间配置。

# 决策规则：
1. 样本量 > 50 万：优先 LightGBM (训练速度快)
2. 类别特征占比 > 30%：必须包含 CatBoost (原生支持)
3. 正负样本比例 < 1:10：添加类别权重参数
4. 特征数 > 100：增加正则化参数搜索范围
5. 小样本 (< 1 万)：包含 Random Forest 作为基线

# 输出格式：
你必须输出严格的 JSON 对象，包含 task_type, optimization_metric, candidate_models, n_trials, rationale, search_spaces 字段。"""),
            
            ("human", """数据特征：
{data_features}

业务场景：
{business_scenario}

请生成 AutoML 配置。""")
        ])
        
        # 3. 组装 Chain
        chain = prompt | structured_llm
        
        # 4. 执行
        logger.info("AutoML_Agent 正在生成配置...")
        result: AutoMLConfig = chain.invoke({
            "data_features": data_features,
            "business_scenario": business_scenario
        })
        
        logger.info(f"🎉 AutoML 配置生成完成，推荐 {len(result.candidate_models)} 个模型")
        return result
    
    def run_eda_agent(
        self,
        eda_summary_json: str,
        business_scenario: str = "通用机器学习建模"
    ) -> EDAReport:
        """
        EDA 分析 Agent
        
        Args:
            eda_summary_json: 统计特征画像 JSON
            business_scenario: 业务场景
            
        Returns:
            EDAReport: 严格校验后的 EDA 报告
        """
        # 1. 绑定结构化输出
        structured_llm = self._create_structured_llm(EDAReport)
        
        # 2. 定义 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位资深数据科学家，专注于机器学习数据质量诊断和探索性分析 (EDA)。你的任务是阅读数据集的统计特征画像，并针对特定业务场景输出专业的 EDA 洞察报告。

# 必须包含的部分：
1. 数据概览 (样本量、特征数、目标变量分布)
2. 数据质量告警 (高缺失率、严重偏态、单一值过多)
3. 业务逻辑冲突 (违背业务常识的数据)
4. 数据穿越风险 (与目标变量相关性过高)
5. 特征处理建议

# 输出格式：
你必须输出严格的 JSON 对象，包含 overview, quality_alerts, risk_flags, recommendations 字段。"""),
            
            ("human", """统计特征画像：
{eda_summary}

业务场景：
{business_scenario}

请生成 EDA 分析报告。""")
        ])
        
        # 3. 组装 Chain
        chain = prompt | structured_llm
        
        # 4. 执行
        logger.info("EDA_Agent 正在生成报告...")
        result: EDAReport = chain.invoke({
            "eda_summary": eda_summary_json,
            "business_scenario": business_scenario
        })
        
        logger.info(f"🎉 EDA 报告生成完成，发现 {len(result.risk_flags)} 个风险点")
        return result


# ==========================================
# 便捷工厂函数
# ==========================================

def create_orchestrator(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "qwen-plus",
    **kwargs
) -> AgentOrchestrator:
    """
    创建编排器的便捷函数
    
    自动从环境变量或配置文件读取 API Key
    
    Args:
        api_key: API Key (可选，默认从环境变量读取)
        base_url: API 基础 URL
        model_name: 模型名称
        **kwargs: 其他参数
        
    Returns:
        AgentOrchestrator 实例
    """
    import os
    
    # 从环境变量读取 API Key
    if not api_key:
        api_key = os.getenv("LLM_API_KEY", "")
    
    if not base_url:
        base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    return AgentOrchestrator(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        **kwargs
    )


__all__ = [
    "AgentOrchestrator",
    "create_orchestrator",
]
