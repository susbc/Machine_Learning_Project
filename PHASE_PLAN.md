# Machine Learning Project - 向 Agentic Workflow 驱动的 AutoML 平台升级

**版本**: v2.3.2 → v3.0.0  
**制定时间**: 2026-03-23  
**状态**: 🟡 规划中  

---

## 📋 目录

1. [当前状态总结](#当前状态总结)
2. [架构演进路线图](#架构演进路线图)
3. [Phase 1: 知识检索层改造](#phase-1-知识检索层改造)
4. [Phase 2: 智能体编排与结构化执行](#phase-2-智能体编排与结构化执行)
5. [Phase 3: 模型工厂与自动化寻优](#phase-3-模型工厂与自动化寻优)
6. [Phase 4: 线上监控与自愈机制](#phase-4-线上监控与自愈机制)
7. [实施计划](#实施计划)

---

## 当前状态总结

### v2.3.2 核心能力

| 维度 | 已实现功能 | 状态 |
|------|----------|------|
| **工程化底座** | 双环境架构 + Streamlit/Vue 3 + FastAPI | ✅ 完成 |
| **数据接入** | DatabaseConnector + smart_read_csv | ✅ 完成 |
| **建模流水线** | BuildingModel + LightGBM + 自动迭代 | ✅ 完成 |
| **LLM 分析** | IntelligentAnalyzer (静态建议) | ✅ 完成 |

### 待升级痛点

| 痛点 | 当前方案 | 目标方案 |
|------|----------|----------|
| 数据查找 | 手动编写 SQL | NL2SQL 自动检索 |
| 特征工程 | 人工配置 | Agent 自动决策 |
| 模型选择 | 单一 LightGBM | 多模型工厂 + 自动推荐 |
| 模型维护 | 手动重训 | 自动监控 + 自愈重训 |

---

## 架构演进路线图

### 当前架构 (v2.3.2)

```
用户 → Streamlit UI → BuildingModel → LightGBM → 预测结果
              ↓
       IntelligentAnalyzer (静态建议)
```

### 目标架构 (v3.0.0)

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic Workflow Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ DataRetrieval│  │ Feature      │  │ AutoML           │  │
│  │ Agent        │  │ Agent        │  │ Agent            │  │
│  │ (NL2SQL)     │  │ (JSON 约束)   │  │ (模型选择 + 调优) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                              ↓                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              MLOps Monitor Agent                      │   │
│  │         (漂移检测 + 自动告警 + 自愈重训)               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Core Engine Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Model        │  │ Optuna       │  │ Database         │  │
│  │ Factory      │  │ Optimizer    │  │ Connector        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Layer                           │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Neo4j        │  │ FAISS        │                        │
│  │ (表血缘/关系)│  │ (字段 Embedding)│                     │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: 知识检索层改造

### 目标
实现 NL2SQL (自然语言转 SQL)，解决"找数据"的痛点

### 新增组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **Neo4jClient** | `core/knowledge/neo4j_client.py` | 图数据库连接 |
| **FAISSClient** | `core/knowledge/faiss_client.py` | 向量数据库连接 |
| **SchemaEmbedder** | `core/knowledge/schema_embedder.py` | 表/字段 Embedding |
| **NL2SQLEngine** | `core/knowledge/nl2sql_engine.py` | NL2SQL 引擎 |
| **DataRetrievalAgent** | `agents/data_retrieval_agent.py` | 数据检索智能体 |

### 数据流转

```
用户自然语言查询
    ↓
DataRetrievalAgent 接收
    ↓
┌───────────────────────────────────────┐
│ 1. 解析用户意图 (LLM)                  │
│ 2. 检索 Neo4j 获取表关系               │
│ 3. 检索 FAISS 获取相关字段             │
│ 4. 构建提示词 + 调用 LLM 生成 SQL       │
│ 5. SQL 验证 (SQLFluff)                │
│ 6. 执行 SQL 返回 DataFrame             │
└───────────────────────────────────────┘
    ↓
返回 Pandas DataFrame
```

### Neo4j Schema 设计

```cypher
// 表节点
(:Table {name: "bank_customer", description: "银行客户信息表"})

// 字段节点
(:Column {name: "customer_id", type: "VARCHAR", description: "客户 ID"})

// 关系
(:Table)-[:HAS_COLUMN]->(:Column)
(:Table)-[:FOREIGN_KEY {ref_table: "xxx"}]->(:Table)
(:Column)-[:REFERENCES]->(:Column)
```

### FAISS 索引设计

```python
# 索引内容
{
    "table_name": "bank_customer",
    "column_name": "customer_id",
    "embedding": [0.1, 0.2, ...],  # 768 维
    "business_meaning": "客户唯一标识"
}
```

### API 设计

```python
class DataRetrievalAgent:
    def __init__(self, neo4j_uri, faiss_index_path, llm_provider)
    
    async def retrieve(self, natural_language_query: str) -> pd.DataFrame:
        """
        接收自然语言查询，返回 DataFrame
        
        Args:
            natural_language_query: "查询所有 VIP 客户的交易记录"
            
        Returns:
            pd.DataFrame: 查询结果
        """
        pass
    
    async def explain_schema(self) -> str:
        """返回数据库 Schema 说明"""
        pass
```

### 验收标准

- [ ] Neo4j 图数据库连接正常
- [ ] FAISS 向量索引创建完成
- [ ] 10 个 NL2SQL 测试用例通过率 ≥ 90%
- [ ] SQL 生成响应时间 < 5 秒

---

## Phase 2: 智能体编排与结构化执行

### 目标
将静态 `IntelligentAnalyzer` 升级为动态 Agent 工作流，确保 LLM 输出安全

### 核心改造：JSON 约束协议

#### LLM 输出格式规范

**EDA Agent 输出**:
```json
{
    "analysis_type": "data_quality",
    "findings": [
        {
            "column": "age",
            "issue": "missing_values",
            "severity": "medium",
            "missing_rate": 0.15,
            "suggestion": "fillna_median"
        }
    ],
    "statistics": {
        "total_rows": 10000,
        "total_columns": 20,
        "duplicate_rows": 50
    }
}
```

**Feature Agent 输出**:
```json
{
    "feature_engineering_plan": [
        {
            "col": "age",
            "action": "fillna",
            "method": "median",
            "params": {}
        },
        {
            "col": "income",
            "action": "transform",
            "method": "log1p",
            "params": {}
        },
        {
            "col": "city",
            "action": "encode",
            "method": "target_encoding",
            "params": {"smooth": 10}
        },
        {
            "col": ["age", "income"],
            "action": "create_interaction",
            "method": "multiply",
            "params": {"new_col_name": "age_income"}
        }
    ]
}
```

### 新增组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **EDAAgent** | `agents/eda_agent.py` | 探索性数据分析 |
| **FeatureAgent** | `agents/feature_agent.py` | 特征工程决策 |
| **JSONParser** | `core/parsers/json_parser.py` | JSON 解析与验证 |
| **ActionExecutor** | `core/executors/action_executor.py` | 执行 JSON 定义的操作 |
| **SafetyValidator** | `core/validators/safety_validator.py` | 安全校验 (禁止 eval) |

### 解析器设计

```python
class FeatureActionExecutor:
    """执行特征工程 JSON 策略"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.history = []  # 操作历史
    
    def execute(self, plan: Dict) -> pd.DataFrame:
        """
        执行特征工程计划
        
        Args:
            plan: FeatureAgent 输出的 JSON
        """
        for action in plan["feature_engineering_plan"]:
            self._execute_action(action)
        return self.df
    
    def _execute_action(self, action: Dict):
        """执行单个操作"""
        action_type = action["action"]
        
        if action_type == "fillna":
            self._fillna(action)
        elif action_type == "transform":
            self._transform(action)
        elif action_type == "encode":
            self._encode(action)
        elif action_type == "create_interaction":
            self._create_interaction(action)
        # ... 更多操作类型
        
        self.history.append(action)
    
    def _fillna(self, action: Dict):
        """安全填充缺失值"""
        col = action["col"]
        method = action["method"]
        
        if method == "median":
            self.df[col] = self.df[col].fillna(self.df[col].median())
        elif method == "mean":
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif method == "mode":
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        # 禁止执行任意代码
    
    def _transform(self, action: Dict):
        """安全数据变换"""
        col = action["col"]
        method = action["method"]
        
        if method == "log1p":
            self.df[col] = np.log1p(self.df[col])
        elif method == "sqrt":
            self.df[col] = np.sqrt(self.df[col])
        # 预定义变换方法白名单
```

### 安全校验规则

```python
class SafetyValidator:
    """LLM 输出安全校验"""
    
    FORBIDDEN_KEYS = ["__class__", "__import__", "__subclasses__", "eval", "exec"]
    ALLOWED_ACTIONS = ["fillna", "transform", "encode", "drop", "create_interaction"]
    ALLOWED_METHODS = {
        "fillna": ["median", "mean", "mode", "constant"],
        "transform": ["log1p", "sqrt", "square", "boxcox"],
        "encode": ["onehot", "label", "target_encoding"]
    }
    
    @classmethod
    def validate(cls, json_data: Dict) -> Tuple[bool, str]:
        """
        验证 JSON 是否安全
        
        Returns:
            (is_valid, error_message)
        """
        # 检查 forbidden keys
        # 检查 action 是否在白名单
        # 检查 method 是否在白名单
        # 检查参数类型
        pass
```

### 验收标准

- [ ] EDA Agent 输出 JSON 通过率 ≥ 95%
- [ ] Feature Agent 输出 JSON 通过率 ≥ 90%
- [ ] 安全校验拦截率 100% (无 eval/exec 执行)
- [ ] 特征工程执行时间 < 30 秒 (10 万行数据)

---

## Phase 3: 模型工厂与自动化寻优

### 目标
打破单一 LightGBM 局限，实现智能模型选择 + 自动寻优

### 新增组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **ModelFactory** | `core/models/model_factory.py` | 模型工厂 |
| **BaseModel** | `core/models/base_model.py` | 模型基类 |
| **XGBoostModel** | `core/models/xgboost_model.py` | XGBoost 封装 |
| **CatBoostModel** | `core/models/catboost_model.py` | CatBoost 封装 |
| **RandomForestModel** | `core/models/rf_model.py` | 随机森林封装 |
| **AutoMLAgent** | `agents/automl_agent.py` | AutoML 智能体 |
| **OptunaOptimizer** | `core/optimizers/optuna_optimizer.py` | Optuna 优化器 |

### 模型工厂设计

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """模型基类"""
    
    @abstractmethod
    def fit(self, X, y) -> "BaseModel":
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict:
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        pass


class ModelFactory:
    """模型工厂"""
    
    MODEL_REGISTRY = {
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
        "catboost": CatBoostModel,
        "random_forest": RandomForestModel,
        "logistic_regression": LogisticRegressionModel
    }
    
    @classmethod
    def create(cls, model_type: str, **params) -> BaseModel:
        """创建模型实例"""
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"未知模型类型：{model_type}")
        return cls.MODEL_REGISTRY[model_type](**params)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """返回支持的模型列表"""
        return list(cls.MODEL_REGISTRY.keys())
```

### AutoML Agent 决策流程

```
数据特征分析
    ↓
┌─────────────────────────────────────┐
│ 1. 分析数据特征                      │
│    - 样本量、维度、稀疏性            │
│    - 类别不平衡程度                  │
│    - 特征类型分布                    │
│ 2. 推荐候选模型                      │
│    - 小样本 → RF/LightGBM           │
│    - 高维稀疏 → XGBoost             │
│    - 类别特征多 → CatBoost          │
│ 3. 生成 Optuna 搜索空间              │
│    - 模型特定参数                    │
│    - 搜索范围和策略                  │
└─────────────────────────────────────┘
    ↓
返回模型推荐 + 搜索空间配置
```

### Optuna 搜索空间示例

```python
def suggest_lightgbm_params(trial):
    """LightGBM 搜索空间"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0, log=True)
    }


def suggest_xgboost_params(trial):
    """XGBoost 搜索空间"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0, log=True)
    }
```

### API 设计

```python
class AutoMLAgent:
    def __init__(self, model_factory, optuna_optimizer):
        self.model_factory = model_factory
        self.optimizer = optuna_optimizer
    
    async def recommend_models(self, df: pd.DataFrame, 
                               target_col: str,
                               constraints: Dict = None) -> Dict:
        """
        推荐候选模型
        
        Returns:
            {
                "recommended_models": [
                    {"model": "lightgbm", "reason": "...", "priority": 1},
                    {"model": "xgboost", "reason": "...", "priority": 2}
                ],
                "search_spaces": {...}
            }
        """
        pass
    
    async def auto_train(self, df: pd.DataFrame,
                         target_col: str,
                         time_budget: int = 3600) -> Dict:
        """
        自动训练和寻优
        
        Returns:
            {
                "best_model": {...},
                "best_params": {...},
                "best_score": 0.95,
                "trial_history": [...]
            }
        """
        pass
```

### 验收标准

- [ ] 支持 ≥ 5 种模型
- [ ] 模型推荐准确率 ≥ 80%
- [ ] Optuna 寻优时间 < 1 小时 (100 次 trial)
- [ ] 最佳模型 AUC 提升 ≥ 5% (vs 默认参数)

---

## Phase 4: 线上监控与自愈机制

### 目标
实现无人值守的持续迭代，解决模型衰减问题

### 新增组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **PSICalculator** | `core/monitoring/psi_calculator.py` | PSI 计算 |
| **DriftDetector** | `core/monitoring/drift_detector.py` | 漂移检测 |
| **ModelMonitor** | `core/monitoring/model_monitor.py` | 模型监控 |
| **AlertManager** | `core/monitoring/alert_manager.py` | 告警管理 |
| **SelfHealingAgent** | `agents/self_healing_agent.py` | 自愈智能体 |
| **Scheduler** | `core/scheduler/scheduler.py` | 定时调度 |

### PSI (群体稳定性指标) 计算

```python
class PSICalculator:
    """PSI 计算"""
    
    @staticmethod
    def calculate_psi(expected: pd.Series, 
                      actual: pd.Series, 
                      buckets: int = 10) -> float:
        """
        计算 PSI
        
        PSI < 0.1: 稳定性好
        0.1 ≤ PSI < 0.2: 轻微变化
        PSI ≥ 0.2: 显著变化 (需要告警)
        """
        # 分箱
        # 计算期望和实际分布
        # PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        pass
    
    @staticmethod
    def calculate_feature_psi(df_expected: pd.DataFrame,
                              df_actual: pd.DataFrame,
                              feature_cols: List[str]) -> Dict[str, float]:
        """计算所有特征的 PSI"""
        pass
```

### 监控架构

```
APScheduler 定时调度 (每日 02:00)
    ↓
ModelMonitor 唤醒
    ↓
┌───────────────────────────────────────┐
│ 1. 拉取最新预测数据                    │
│ 2. 计算特征 PSI                        │
│ 3. 计算预测分布 PSI                    │
│ 4. 检查阈值                            │
│    - 核心特征 PSI > 0.2 → 告警        │
│    - 预测分布 PSI > 0.25 → 告警       │
│ 5. 触发自愈流程 (如需要)               │
└───────────────────────────────────────┘
    ↓
告警 → SelfHealingAgent → 自动重训
```

### 自愈流程

```python
class SelfHealingAgent:
    """自愈智能体"""
    
    def __init__(self, auto_train_pipeline, alert_manager):
        self.auto_train = auto_train_pipeline
        self.alert_manager = alert_manager
    
    async def on_drift_detected(self, 
                                 drift_report: Dict,
                                 threshold: float = 0.2):
        """
        漂移检测触发后的自愈流程
        
        1. 评估漂移严重程度
        2. 检查是否有足够新数据
        3. 自动重训模型
        4. 验证新模型效果
        5. 部署新模型 (或回滚)
        6. 发送通知
        """
        pass
```

### 告警配置

```yaml
# config/alert_config.yaml
alerts:
  psi_threshold:
    feature: 0.2      # 特征 PSI 阈值
    prediction: 0.25  # 预测 PSI 阈值
  
  channels:
    - type: feishu
      webhook: ${FEISHU_WEBHOOK}
    - type: email
      recipients: [admin@example.com]
  
  actions:
    on_feature_drift:
      - notify
      - trigger_retrain
    on_prediction_drift:
      - notify
      - trigger_retrain
      - rollback_model
```

### 验收标准

- [ ] PSI 计算准确率 100%
- [ ] 漂移检测响应时间 < 5 分钟
- [ ] 自动重训成功率 ≥ 95%
- [ ] 告警通知到达率 100%

---

## 实施计划

### 时间规划

| Phase | 预计时间 | 里程碑 |
|-------|----------|--------|
| **Phase 1** | 2 周 | NL2SQL 引擎完成 |
| **Phase 2** | 2 周 | Agent 工作流上线 |
| **Phase 3** | 2 周 | AutoML 工厂完成 |
| **Phase 4** | 2 周 | MLOps 监控上线 |
| **总计** | 8 周 | v3.0.0 发布 |

### 优先级

| 优先级 | Phase | 原因 |
|--------|-------|------|
| 🔴 P0 | Phase 1 | 数据检索是基础能力 |
| 🔴 P0 | Phase 2 | 安全执行是核心保障 |
| 🟡 P1 | Phase 3 | 模型多样性提升效果 |
| 🟡 P1 | Phase 4 | 监控自愈是进阶能力 |

### 依赖关系

```
Phase 1 (知识检索) → Phase 2 (Agent 编排)
                           ↓
Phase 3 (模型工厂) ← Phase 2 (Agent 编排)
                           ↓
                      Phase 4 (监控自愈)
```

### 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Neo4j 学习曲线 | 中 | 提前调研 + PoC 验证 |
| LLM JSON 输出不稳定 | 中 | 增加重试机制 + 降级策略 |
| Optuna 调优时间长 | 低 | 设置时间预算 + 早停 |
| 生产环境监控复杂 | 中 | 分阶段上线 + 人工审核 |

---

## 下一步行动

1. **立即行动**: 创建项目目录结构
2. **本周完成**: Phase 1 技术选型和 PoC
3. **依赖安装**: `pip install neo4j faiss-cpu optuna apscheduler`

---

*文档版本：v1.0*  
*创建时间：2026-03-23 14:20*  
*状态：规划中*
