# Frontend Replacement Runbook

日期：2026-06-05

## 当前结论

Vue + TypeScript 前端已经覆盖当前主线工作台：数据管理、训练、分析、监控、配置、Agent 工作流、模型管理和预测实验室都已接入 FastAPI API。Streamlit 暂时保留为回退和历史功能对照入口。

## 启动方式

### 开发模式

启动 FastAPI：

```bash
.venv/bin/python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

启动 Vue：

```bash
cd frontend
npm install
npm run dev
```

访问：http://localhost:5173

### 生产托管模式

构建 Vue：

```bash
cd frontend
npm install
npm run build
```

启动 FastAPI：

```bash
.venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

访问：http://localhost:8000

FastAPI 会优先托管 `frontend/dist/index.html` 和 `frontend/dist/assets/*`。刷新 `/automl`、`/ai-analysis`、`/agent-workflow` 等 Vue history 路由时，会回退到 Vue entrypoint；未知 `/api/*` 仍返回 API 404，不会被前端页面吞掉。

如需指定构建产物目录：

```bash
export FRONTEND_DIST_PATH=/absolute/path/to/frontend/dist
```

## 替代状态

灰度可用：

- 大模型配置
- 数据库连接
- AutoML 训练
- 手动训练
- 数据管理
- AI 智能分析
- MLOps 监控
- Agent 工作流
- 模型管理（统一 registry、详情、多模型指标对比）
- 预测实验室（单条 JSON 预测、CSV 批量预测、结果摘要）

已下线：

- 自动迭代训练入口（与 AutoML 训练渲染同一页面，属于冗余路由，已移除；Streamlit 侧的自动迭代功能保留在回退入口中）

待迁移或待明确下线：

- 暂无主线阻塞项；后续只保留增强项和历史兼容项评估。

## 每个模块替换验收

1. 前端 API 测试通过。
2. 前端组件交互回归测试通过。
3. 后端 API 测试通过。
4. 浏览器打开模块页面，完成关键按钮冒烟。
5. 敏感字段不明文回显。
6. 错误、空状态、加载状态可见。
7. Streamlit 回退路径保留或已明确下线。

## 当前验证命令

```bash
cd frontend
npm test
npm run build
npm audit --audit-level=high

cd ..
PYTEST_ADDOPTS='--no-cov' .venv/bin/python -m pytest \
  tests/test_frontend_static.py \
  tests/test_model_registry_api.py \
  tests/test_automl_api.py \
  tests/test_training_api.py \
  tests/test_llm_config_api.py \
  tests/test_database_api.py \
  tests/test_analysis_api.py \
  tests/test_mlops_api.py \
  -q

git diff --check
```
