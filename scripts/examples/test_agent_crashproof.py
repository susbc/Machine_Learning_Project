"""
Agent Crash-Proof 防崩测试

测试 LangChain + Pydantic 结构化输出的防幻觉能力。

运行方法:
    python scripts/examples/test_agent_crashproof.py

前置条件:
    pip install langchain langchain-openai pydantic
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.llm.agent_orchestrator import AgentOrchestrator, create_orchestrator


# ==========================================
# 模拟测试数据
# ==========================================

MOCK_EDA_SUMMARY = """
{
 "dataset_meta": {
  "rows": 100000,
  "columns": 15,
  "target_col": "is_fraud"
 },
 "features": {
  "driver_age": {
   "type": "numeric",
   "null_rate": 0.05,
   "mean": 35.2,
   "std": 8.1,
   "skewness": 0.2
  },
  "city_code": {
   "type": "categorical",
   "null_rate": 0.0,
   "unique_values": 342,
   "top_1_ratio": 0.45
  },
  "bad_column": {
   "type": "numeric",
   "null_rate": 0.85,
   "mean": 0.0
  },
  "income": {
   "type": "numeric",
   "null_rate": 0.02,
   "skewness": 4.5
  }
 }
}
"""

MOCK_EDA_REPORT = """
## 数据质量告警
- driver_age 有少量缺失 (5%)
- city_code 基数极高 (342 个唯一值)
- bad_column 基本全是空值 (85%)
- income 严重右偏 (skewness=4.5)

## 业务逻辑冲突
- 无明显冲突

## 数据穿越风险
- 无明显高风险特征
"""


# ==========================================
# 测试函数
# ==========================================

def test_feature_agent():
    """测试 Feature_Agent 防崩能力"""
    print("=" * 60)
    print("🧪 测试 Feature_Agent (特征工程)")
    print("=" * 60)
    
    # 创建编排器
    orchestrator = create_orchestrator(
        model_name="qwen-plus",
        max_retries=3
    )
    
    print("\n📥 输入数据:")
    print(f"  - 特征数：15")
    print(f"  - 高基数特征：city_code (342 个值)")
    print(f"  - 高缺失特征：bad_column (85%)")
    print(f"  - 严重偏态：income (skewness=4.5)")
    
    print("\n🚀 开始调用 Feature Agent...")
    
    try:
        # 调用 Agent (返回严格 Pydantic 对象)
        strategy_obj = orchestrator.run_feature_agent(
            MOCK_EDA_SUMMARY,
            MOCK_EDA_REPORT
        )
        
        print("\n✅ 测试通过！成功生成合规策略：")
        print(f"   共 {len(strategy_obj.actions)} 个处理动作\n")
        
        for i, action in enumerate(strategy_obj.actions, 1):
            print(f"   {i}. 处理列：{action.column}")
            print(f"      动作：{action.action}")
            print(f"      理由：{action.reason}")
            if action.params:
                print(f"      参数：{action.params}")
            print()
        
        # 验证关键规则是否被遵守
        print("📋 规则验证:")
        
        # 规则 1: 高基数类别特征应使用 target_encode
        city_action = next((a for a in strategy_obj.actions if a.column == "city_code"), None)
        if city_action:
            if city_action.action == "target_encode":
                print("   ✅ city_code 正确使用 target_encode (高基数处理)")
            else:
                print(f"   ⚠️ city_code 使用 {city_action.action}，建议 target_encode")
        
        # 规则 2: 高缺失特征应 drop
        bad_action = next((a for a in strategy_obj.actions if a.column == "bad_column"), None)
        if bad_action:
            if bad_action.action == "drop":
                print("   ✅ bad_column 正确执行 drop (高缺失处理)")
            else:
                print(f"   ⚠️ bad_column 使用 {bad_action.action}，建议 drop")
        
        # 规则 3: 严重偏态应 log_transform
        income_action = next((a for a in strategy_obj.actions if a.column == "income"), None)
        if income_action:
            if income_action.action == "log_transform":
                print("   ✅ income 正确执行 log_transform (偏态处理)")
            else:
                print(f"   ⚠️ income 使用 {income_action.action}，建议 log_transform")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败（或者被安全拦截）: {e}")
        return False


def test_data_retrieval_agent():
    """测试 DataRetrieval_Agent (NL2SQL)"""
    print("\n" + "=" * 60)
    print("🧪 测试 DataRetrieval_Agent (NL2SQL)")
    print("=" * 60)
    
    orchestrator = create_orchestrator(model_name="qwen-plus")
    
    graph_schema = """
- 表名：orders (订单表)
  字段：order_id (string), driver_id (bigint), order_fee (decimal), create_time (datetime)
  关联关系 (JOINs):
    orders.driver_id = driver_info.driver_id

- 表名：driver_info (司机信息表)
  字段：driver_id (bigint), city_code (string), register_time (datetime)
"""
    
    user_query = "查询北京地区司机上周的完单金额"
    
    print(f"\n📥 图谱信息：2 张表 (orders, driver_info)")
    print(f"📥 用户查询：{user_query}")
    
    print("\n🚀 开始生成 SQL...")
    
    try:
        result = orchestrator.run_data_retrieval_agent(graph_schema, user_query)
        
        print(f"\n✅ SQL 生成完成！")
        print(f"   可执行：{result.is_executable}")
        print(f"   思考过程：{result.thought_process[:100]}...")
        
        if result.sql:
            print(f"\n📝 生成的 SQL:")
            print(f"   {result.sql}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ SQL 生成失败：{e}")
        return False


def test_automl_agent():
    """测试 AutoML_Agent"""
    print("\n" + "=" * 60)
    print("🧪 测试 AutoML_Agent (模型推荐)")
    print("=" * 60)
    
    orchestrator = create_orchestrator(model_name="qwen-plus")
    
    data_features = """
- 样本量：500,000 行
- 特征数：120 维
- 类别特征：45 个 (占比 37.5%)
- 正负样本比例：1:99 (极度不平衡)
- 任务类型：二分类
"""
    
    business_scenario = "网约车反欺诈建模"
    
    print(f"\n📥 数据特征：50 万行 × 120 维，类别特征 37.5%，不平衡 1:99")
    print(f"📥 业务场景：{business_scenario}")
    
    print("\n🚀 开始生成 AutoML 配置...")
    
    try:
        config = orchestrator.run_automl_agent(data_features, business_scenario)
        
        print(f"\n✅ AutoML 配置生成完成！")
        print(f"   任务类型：{config.task_type}")
        print(f"   优化指标：{config.optimization_metric}")
        print(f"   候选模型：{config.candidate_models}")
        print(f"   Trial 次数：{config.n_trials}")
        print(f"\n📋 推荐理由：{config.rationale}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ AutoML 配置生成失败：{e}")
        return False


def test_invalid_action_rejection():
    """测试 Pydantic 对非法 action 的拦截能力"""
    print("\n" + "=" * 60)
    print("🧪 测试 Pydantic 非法动作拦截")
    print("=" * 60)
    
    from core.llm.agent_schemas import FeatureAction
    from pydantic import ValidationError
    
    print("\n📥 测试用例：LLM 自创 'fillna_magic' 动作")
    
    try:
        # 尝试创建非法 action
        action = FeatureAction(
            column="test_col",
            action="fillna_magic",  # ❌ 非法动作！
            reason="测试"
        )
        print("   ❌ 测试失败：非法动作未被拦截！")
        return False
        
    except ValidationError as e:
        print(f"   ✅ 测试通过：非法动作被 Pydantic 拦截！")
        print(f"   错误信息：{str(e)[:200]}...")
        return True


# ==========================================
# 主入口
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Agent Crash-Proof 防崩测试套件")
    print("=" * 60)
    print()
    print("测试目标:")
    print("  1. Feature_Agent - 特征工程策略生成")
    print("  2. DataRetrieval_Agent - NL2SQL 生成")
    print("  3. AutoML_Agent - 模型推荐配置")
    print("  4. Pydantic 非法动作拦截")
    print()
    print("前置条件:")
    print("  - 配置 LLM_API_KEY 环境变量")
    print("  - 安装：pip install langchain langchain-openai pydantic")
    print()
    
    # 检查 API Key
    if not os.getenv("LLM_API_KEY"):
        print("⚠️  警告：未配置 LLM_API_KEY 环境变量")
        print("   请设置：export LLM_API_KEY='your-api-key'")
        print()
        response = input("是否继续测试 (跳过 LLM 调用测试)？[y/N] ")
        if response.lower() != 'y':
            print("测试取消。")
            sys.exit(0)
    
    # 运行测试
    results = []
    
    # 测试 1: Pydantic 拦截 (无需 LLM)
    results.append(("Pydantic 拦截测试", test_invalid_action_rejection()))
    
    # 测试 2-4: 需要 LLM
    if os.getenv("LLM_API_KEY"):
        results.append(("Feature_Agent", test_feature_agent()))
        results.append(("DataRetrieval_Agent", test_data_retrieval_agent()))
        results.append(("AutoML_Agent", test_automl_agent()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print()
    print(f"总计：{passed}/{total} 通过 ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    sys.exit(0 if passed == total else 1)
