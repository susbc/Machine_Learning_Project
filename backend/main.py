"""
Agentic ML Studio - FastAPI 主应用

启动方式:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

访问:
    API 文档：http://localhost:8000/docs
    健康检查：http://localhost:8000/health
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.workflow import router as workflow_router
from scheduler import start_scheduler, stop_scheduler


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

# 挂载业务路由
app.include_router(workflow_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用 Agentic ML Studio v3.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    from scheduler import scheduler
    return {
        "status": "healthy",
        "scheduler_running": scheduler.running,
        "version": "3.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
