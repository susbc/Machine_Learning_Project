# Machine Learning Project

Agentic AutoML Platform 是一个面向表格数据建模的机器学习平台，包含 Streamlit Web UI、FastAPI 工作流服务、传统训练/预测流水线、AutoML 调参、LLM Agent、数据库连接、知识检索和 MLOps 漂移监控模块。

## 当前状态

- 主版本：v3.0.0
- 主要入口：Vue + TypeScript 前端（FastAPI 托管或 Vite 开发模式）+ FastAPI API；Streamlit Web UI 暂作旧功能回退入口
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
├── web_app/                 # Streamlit 应用和页面（迁移期 fallback）
├── frontend/                # Vue + TypeScript 前端工作台
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

### 2. 启动 Vue + FastAPI 工作台

开发模式：

```bash
.venv/bin/python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
cd frontend
npm install
npm run dev
```

访问：http://localhost:5173

生产托管模式：

```bash
cd frontend
npm install
npm run build
cd ..
.venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

访问：http://localhost:8000

FastAPI 会优先托管 `frontend/dist`，并支持 Vue history 路由刷新。详见 `docs/frontend-replacement-runbook.md`。

手动训练和 AutoML 的数据集、任务状态及结果会持久化到
`saved/runtime_state/`。后台任务默认使用 2 个工作线程、最多接收 32 个
执行中或排队任务，可通过以下环境变量调整：

```bash
export ML_BACKGROUND_WORKERS=2
export ML_MAX_PENDING_JOBS=32
# 自动清理超过指定天数的数据集和已结束任务，默认 30 天
export ML_RUNTIME_RETENTION_DAYS=30
# 可选：修改 SQLite 元数据与运行时对象的保存位置
export ML_RUNTIME_STATE_DIR=/path/to/runtime_state
```

本地持久化执行器适用于单实例部署。多实例或多机部署时，应将
`backend/runtime_state.py` 的执行器替换为 Redis/Celery 等共享队列；HTTP
接口和任务状态结构不需要随之改变。

> **安全提示**：将服务绑定到 `0.0.0.0` 等公开地址前，请设置 `API_KEY` 环境变量启用接口鉴权（所有 `/api/*` 请求需携带 `X-API-Key` 头）；仅本机访问时建议改用 `--host 127.0.0.1`。
>
> ```bash
> export API_KEY=$(openssl rand -hex 32)   # 后端鉴权密钥
> # 前端构建期注入：VITE_API_KEY=$API_KEY npm run build
> ```

### 3. 启动 Streamlit Web UI 回退入口

```bash
.venv/bin/python -m streamlit run web_app/app.py --server.port 8501
```

访问：http://localhost:8501

### 4. 单独启动 FastAPI API

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
- Web UI：Vue 工作台提供数据管理、手动训练、AutoML、自动迭代兼容入口、AI 智能分析、数据库连接、LLM 配置、Agent 工作流、MLOps 监控、模型管理与多模型对比、预测实验室（单条和 CSV 批量预测）等主线页面；Streamlit 暂时保留为回退和历史功能对照入口。

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

测试分为两个独立任务运行。`tests/knowledge` 会导入 sentence-transformers（加载 torch），
与 lightgbm/xgboost 的原生库在同一进程内会触发段错误，因此通过 `pytest.ini` 的
`norecursedirs` 从主任务中排除，必须单独运行：

```bash
# 主任务（不含 tests/knowledge）
.venv/bin/python -m pytest

# 独立任务：knowledge 测试（需要本地 Neo4j，如 bolt://localhost:7687）
.venv/bin/python -m pytest tests/knowledge
```

运行局部测试：

```bash
.venv/bin/python -m pytest tests/test_database.py::TestQueryBuilder -q
```

当前测试套件包含需要外部服务或本地模型缓存的用例，例如 Neo4j、FAISS、HuggingFace sentence-transformers 和 LLM 相关模块。没有对应服务或离线模型缓存时，这些测试会失败（CI 的 `test-knowledge` 任务通过 Neo4j service 容器提供依赖，并标记为允许失败）。

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

## 开源协议

本项目采用 [MIT License](LICENSE)。本地 LLM Provider 配置不会进入版本控制；
首次配置可复制 `config/llm_providers.example.json` 为
`config/llm_providers.json`，再通过界面或本地文件填写密钥。

---

Last Updated: 2026-09-08
