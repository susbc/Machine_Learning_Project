"""
Agentic ML Studio - FastAPI 主应用

启动方式:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

访问:
    API 文档：http://localhost:8000/docs
    健康检查：http://localhost:8000/health
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from backend.api.workflow import router as workflow_router
from backend.api.mlops import router as mlops_router
from backend.scheduler import start_scheduler, stop_scheduler
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理
    
    启动前：开启 MLOps 监控定时任务
    关闭后：优雅结束调度器
    """
    # 启动前
    print("\n" + "=" * 60)
    print("🚀 启动 MLOps 后台监控任务...")
    print("=" * 60)
    start_scheduler()
    print()
    
    yield
    
    # 关闭后
    print("\n" + "=" * 60)
    print("🛑 正在关闭 MLOps 后台监控任务...")
    print("=" * 60)
    stop_scheduler()


app = FastAPI(
    title="Machine Learning Project - Agentic ML Studio",
    description="""
## 功能特性

### 🤖 Agentic AutoML 工作流
- **Step 1**: 自然语言转 SQL 并提取数据
- **Step 2**: 生成特征工程策略 (Pydantic 严格校验)
- **Step 3**: 数据清洗 + AutoML 调参 + SSE 实时推送
- **Step 4**: 线上推理预测 (有状态特征工程)

### 📊 MLOps 监控
- **PSI 漂移检测**: 每日自动巡检
- **自动重训**: PSI ≥ 0.2 自动触发
- **告警通知**: 飞书/邮件实时告警

### 🛡️ 防崩设计
- **Pydantic 枚举约束**: LLM 无法自创动作
- **防御性编程**: 列不存在自动跳过
- **有状态执行**: fit_transform/transform 分离
    """,
    version="3.0.0",
    lifespan=lifespan
)

# 挂载静态文件
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# 挂载业务路由
app.include_router(workflow_router)
app.include_router(mlops_router)

# TODO: AutoML 路由 (SSE 流式 Optuna 优化)
# 前端 OptunaDashboard.vue 已对接，但后端仍使用 mock 数据，
# 需接入真实训练流程后再启用
# from backend.api.automl import router as automl_router
# app.include_router(automl_router, prefix="/api")


@app.get("/")
async def root():
    """根路径 - 返回美观的欢迎页面"""
    index_path = os.path.join(static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "欢迎使用 Agentic ML Studio v3.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    from backend.scheduler import scheduler
    return {
        "status": "healthy",
        "scheduler_running": scheduler.running,
        "version": "3.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
