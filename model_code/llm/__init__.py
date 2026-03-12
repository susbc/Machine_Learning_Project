"""
大模型智能分析模块

提供：
- 多模型 API 支持（OpenAI/Claude/Qwen/Zhipu/Baidu）
- 数据集智能分析
- 模型推荐
- 训练结果分析
- 预测结果分析
- 超参数调优建议
"""

from .llm_connector import (
    LLMManager,
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    QwenProvider,
    ZhipuProvider,
    BaiduProvider,
    get_llm_manager
)

from .intelligent_analyzer import (
    IntelligentAnalyzer,
    get_intelligent_analyzer
)

__all__ = [
    # 连接器
    'LLMManager',
    'LLMProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'QwenProvider',
    'ZhipuProvider',
    'BaiduProvider',
    'get_llm_manager',
    
    # 分析器
    'IntelligentAnalyzer',
    'get_intelligent_analyzer'
]
