# Machine Learning Project

Agentic AutoML Platform 是一个面向表格数据建模的机器学习平台，包含 Streamlit Web UI、FastAPI 工作流服务、传统训练/预测流水线、AutoML 调参、LLM Agent、数据库连接、知识检索和 MLOps 漂移监控模块。

## 当前状态

- 主版本：v3.0.0
- 主要入口：Streamlit Web UI + FastAPI API
- 核心任务：数据上传/数据库取数、特征工程、模型训练、AutoML 调参、预测、模型监控
- 推荐运行环境：Python 3.12 + 项目本地虚拟环境 `.venv`

## 项目结构

```text
.
├── backend/                 # FastAPI 服务、工作流 API、MLOps API、调度器
├── core/
│   ├── automl/              # Optuna AutoML 优化器
│   ├── feature_engineering/ # EDA 与特征工程 Agent
│   ├── knowledge/           # Neo4j/FAISS/Schema Embedding 知识检索
│   └── llm/                 # LLM Agent 编排与 Pydantic Schema
├── model_code/              # 训练、预测、预处理、模型工厂、数据库连接
├── web_app/                 # Streamlit 应用和页面
├── frontend/                # Vue 组件原型
├── tests/                   # 单元测试和集成测试
├── scripts/                 # 数据生成、演示、辅助脚本
├── input_data/              # 本地输入数据目录
├── output/                  # 工作流中间产物和预测输出
├── saved/                   # FastAPI 工作流模型产物
└── model_code/saved/        # 传统训练流水线模型产物
```

## 快速开始

### 1. 创建并安装依赖

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

### 2. 启动 Streamlit Web UI

```bash
.venv/bin/python -m streamlit run web_app/app.py --server.port 8501
```

访问：http://localhost:8501

### 3. 启动 FastAPI 服务

```bash
.venv/bin/python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

访问：

- API 首页：http://localhost:8000
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## 主要功能

- 模型训练：支持 LightGBM、XGBoost、CatBoost、随机森林、逻辑回归。
- 预测输出：加载训练摘要和最佳模型，使用训练阈值生成预测结果。
- AutoML：基于 Optuna TPE 进行贝叶斯调参和模型选择。
- Agent 工作流：自然语言取数、结构化特征策略生成、SSE 训练进度推送、在线预测。
- 数据库连接：支持 SQLite、MySQL、PostgreSQL 等常见数据库连接与查询构建。
- 知识检索：提供 Neo4j 图谱、FAISS 向量索引和 Schema Embedding 组件。
- MLOps：支持 PSI 数据漂移检测、基线保存、巡检日志和重训触发接口。
- Web UI：提供手动训练、AutoML、数据库连接、LLM 配置、Agent 工作流、MLOps 监控等页面。

## 传统训练流水线

`model_code.main_model.run()` 是传统训练/预测入口。数据文件默认使用竖线分隔 CSV，字段规则来自 Excel 配置表。

```python
from model_code.main_model import run

run(
    data_month="202605",
    data_path="input_data/train_data.csv",
    config_path="input_data/field_config.xlsx",
    is_train=True,
    is_predict=True,
    output_dir="output",
    model_dir="model_code/saved",
    algorithm="auto",
)
```

字段配置表至少需要包含：

- `field_name`：字段名
- `field_type`：字段类型，必须有一个 `key` 和一个 `label`
- 可选预处理规则：缺失值、异常值、编码、标准化等

## FastAPI Agent 工作流

典型调用顺序：

1. `POST /api/workflow/step1_retrieve_data`：自然语言转 SQL 并生成任务数据。
2. `POST /api/workflow/step2_generate_features`：生成结构化特征工程策略。
3. `GET /api/workflow/step3_train_stream`：执行特征工程和 AutoML 训练，通过 SSE 推送进度。
4. `POST /api/workflow/predict`：加载模型产物并在线预测。

注意：当前 Step 1 仍保留 demo 数据路径，接入真实数据库时需要替换 `backend/api/workflow.py` 中的 DEMO MODE 段落。

## 环境变量

常用配置：

```bash
export ML_DATA_PATH=input_data/train_data.csv
export ML_CONFIG_PATH=input_data/field_config.xlsx
export ML_OUTPUT_DIR=output
export ML_MODEL_DIR=model_code/saved
export LLM_API_KEY=your_api_key
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 测试

运行全部测试：

```bash
.venv/bin/python -m pytest
```

运行局部测试：

```bash
.venv/bin/python -m pytest tests/test_database.py::TestQueryBuilder -q
```

当前测试套件包含需要外部服务或本地模型缓存的用例，例如 Neo4j、FAISS、HuggingFace sentence-transformers 和 LLM 相关模块。没有对应服务或离线模型缓存时，这些测试应作为集成测试单独处理。

## Docker

```bash
docker compose up web
```

默认暴露 Streamlit：`http://localhost:8501`

后台训练服务可通过 profile 启动：

```bash
docker compose --profile training up trainer
```

## 已知注意事项

- 请优先使用 `.venv/bin/python` 启动服务和测试，避免系统 Python 与项目依赖混用。
- `requirements.txt` 按 Python 3.12 依赖栈更新，并对 LangChain/Pydantic、requests/chardet 等已知冲突使用兼容约束；如果遇到 LightGBM、FAISS、sentence-transformers 等二进制依赖问题，建议重建虚拟环境。
- 知识检索和 LLM 测试需要外部服务或可用 API Key；CI 中建议 mock 或用 pytest marker 隔离。
- `backend/api/workflow.py` 的 Step 1 目前仍是 demo 数据，生产使用前需要接入真实数据库执行链路。

---

Last Updated: 2026-05-10
