# 🧠 AI 智能分析模块

> **历史快照说明（2026-11-02 补注）**
>
> 本文是 AI 智能分析模块上线时的功能说明，接口与页面形态可能已演进；
> 当前功能与启动方式请以 [README.md](README.md) 为准。

使用大模型（LLM）为机器学习流程提供智能分析和建议。

## 📋 功能概述

### 1. 大模型连接器
支持多种大模型 API：
- 🟢 **OpenAI** (GPT-4, GPT-3.5)
- 🟣 **Anthropic** (Claude 3)
- 🔵 **阿里云通义千问** (Qwen)
- 🟡 **智谱 AI** (GLM-4)
- 🔴 **百度文心一言** (Ernie)

### 2. 智能分析功能
- 📊 **数据集分析** + 业务场景 → 数据质量评估 + 建模建议
- 🎯 **模型推荐** → 根据数据特征和业务需求推荐最佳模型
- 📈 **训练结果分析** → 诊断过拟合/欠拟合 + 优化建议
- 🔮 **预测结果分析** → 业务洞察 + 风险提示
- ⚙️ **超参数调优** → 智能调参建议 + 搜索策略

## 🗂️ 文件结构

```
model_code/llm/
├── __init__.py                 # 模块导出
├── llm_connector.py            # 大模型连接器 (17KB)
└── intelligent_analyzer.py     # 智能分析器 (15KB)

web_app/pages/
├── 12_大模型配置管理.py        # 大模型配置页面 (15KB)
└── 13_AI_智能分析.py           # AI 智能分析页面 (12KB)
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /Users/rgj/PycharmProjects/Machine_Learning_Project
pip install -r requirements.txt
```

### 2. 配置大模型 API

启动 Web 应用，访问 **12_大模型配置管理** 页面：

```bash
.venv/bin/streamlit run web_app/app.py
```

在页面中添加你的 API Key：
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/
- 通义千问：https://dashscope.console.aliyun.com/
- 智谱 AI: https://open.bigmodel.cn/
- 百度：https://console.bce.baidu.com/qianfan/

### 3. 使用智能分析

访问 **13_AI 智能分析** 页面，选择功能标签：

#### 📊 数据集分析
```python
from model_code.llm import get_intelligent_analyzer

analyzer = get_intelligent_analyzer()

# 分析数据集
analysis = analyzer.analyze_dataset(
    df=df,
    target_col='label',
    business_context="金融风控场景，需要识别欺诈交易"
)
print(analysis)
```

#### 🎯 模型推荐
```python
recommendation = analyzer.recommend_model(
    df=df,
    target_col='label',
    business_context="电商用户购买预测",
    requirements={
        'speed_requirement': '高',
        'interpretability_requirement': '中',
        'accuracy_requirement': '高'
    }
)
print(recommendation)
```

#### 📈 训练结果分析
```python
metrics = {
    'train_auc': 0.95,
    'val_auc': 0.88,
    'train_f1': 0.92,
    'val_f1': 0.85,
    'train_size': 10000,
    'val_size': 2000,
    'n_features': 50
}

params = {
    'model_type': 'LightGBM',
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1
}

analysis = analyzer.analyze_training_results(metrics, params)
print(analysis)
```

#### ⚙️ 超参数调优
```python
current_params = {
    'model_type': 'LightGBM',
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1
}

history = [
    {'params': {'n_estimators': 50}, 'auc': 0.82},
    {'params': {'n_estimators': 100}, 'auc': 0.85},
    {'params': {'n_estimators': 200}, 'auc': 0.86}
]

suggestion = analyzer.suggest_hyperparameter_tuning(
    current_params,
    history,
    constraints={'max_train_time': '30 分钟', 'compute_resources': '单 CPU'}
)
print(suggestion)
```

## 💡 使用场景

### 场景 1: 新项目启动
1. 上传数据集
2. 使用 **数据集分析** 了解数据质量
3. 使用 **模型推荐** 选择合适的模型
4. 开始建模

### 场景 2: 模型调优
1. 训练初始模型
2. 使用 **训练结果分析** 诊断问题
3. 使用 **超参数调优** 获取调参建议
4. 迭代优化

### 场景 3: 预测部署
1. 进行预测
2. 使用 **预测结果分析** 评估质量
3. 根据建议调整阈值或模型
4. 部署到生产

## 📊 分析示例

### 数据集分析输出示例

```markdown
## 数据质量评估
1. **整体质量**: 良好，缺失值比例 2.3%，适合建模
2. **潜在问题**: 
   - 特征"age"存在异常值（>100）
   - 特征"income"右偏严重，建议对数变换

## 特征工程建议
1. **数值特征**: 
   - "income"建议对数变换
   - "age"需要截断或分箱处理
2. **类别特征**:
   - "city"类别过多（>100），建议目标编码
   - "occupation"可合并稀有类别

## 建模风险提示
1. 类别不平衡（正例仅 5%），需要使用采样或加权
2. 部分特征相关性高，注意多重共线性

## 采样建议
1. 使用 SMOTE 或 ADASYN 进行过采样
2. 或使用类别权重 class_weight='balanced'
```

### 训练结果分析输出示例

```markdown
## 模型状态诊断
1. **过拟合判断**: 是（训练 AUC 0.95 vs 验证 AUC 0.88，差距 0.07）
2. **欠拟合判断**: 否（训练集表现良好）
3. **收敛情况**: 正常

## 问题定位
过拟合可能原因：
1. max_depth=6 可能过大
2. n_estimators=100 可能过多
3. 缺少正则化

## 优化建议（按优先级）
1. **降低 max_depth**: 从 6 降至 4，预期 AUC 提升 0.01-0.02
2. **增加 min_child_samples**: 从 20 增至 50，减少过拟合
3. **添加正则化**: 设置 reg_alpha=0.1, reg_lambda=0.1

## 下一步行动
1. 调整 max_depth=4, min_child_samples=50
2. 重新训练并验证
3. 如仍过拟合，考虑 early_stopping
```

## 🔧 配置说明

### 环境变量（可选）

```bash
# 默认大模型
export DEFAULT_LLM_PROVIDER=openai

# API Keys（也可在 Web UI 配置）
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DASHSCOPE_API_KEY="sk-..."  # 通义千问
export ZHIPU_API_KEY="..."  # 智谱 AI
```

### 配置文件

配置保存在：
```
Machine_Learning_Project/config/llm_providers.json
```

## 📝 分析报告

分析结果可保存为 Markdown 报告：
```
model_code/analysis_reports/
├── data_analysis_20260309_163000.md
├── training_analysis_20260309_164500.md
└── prediction_analysis_20260309_170000.md
```

## ⚠️ 注意事项

1. **API 费用**: 大模型调用会产生费用，请注意用量
2. **响应时间**: 分析可能需要 10-30 秒，请耐心等待
3. **结果参考**: AI 建议仅供参考，请结合实际情况判断
4. **数据安全**: 敏感数据请勿上传到云端 API

## 🎯 最佳实践

1. **配置多个模型**: 建议配置 2-3 个大模型作为备份
2. **选择合适模型**: 
   - 快速分析 → GPT-3.5 / Qwen-Turbo
   - 深度分析 → GPT-4 / Claude-3-Opus
3. **保存分析结果**: 重要分析结果及时保存报告
4. **迭代优化**: 多次分析对比，获取更全面的建议

## 📚 相关文档

- [大模型配置管理页面](../../web_app/pages/12_大模型配置管理.py)
- [AI 智能分析页面](../../web_app/pages/13_AI_智能分析.py)
- [项目主 README](../../README.md)

---

*最后更新：2026-03-09*
