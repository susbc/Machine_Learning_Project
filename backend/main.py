"""
Agentic ML Studio - FastAPI 主应用

启动方式:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

访问:
    API 文档：http://localhost:8000/docs
    健康检查：http://localhost:8000/health

安全说明:
    所有 /api/* 管理端点在未设置 API_KEY 时仅允许本机（回环地址）访问；
    绑定 0.0.0.0 等对外地址时必须设置 API_KEY 环境变量，否则远程请求会被拒绝。
    反向代理等合法场景可设置 ALLOW_NETWORK_API_WITHOUT_AUTH=1 显式放开。
"""

from dotenv import load_dotenv
load_dotenv()

import hmac
import ipaddress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from backend.api.workflow import router as workflow_router
from backend.api.mlops import router as mlops_router
from backend.api.automl import router as automl_router
from backend.api.analysis import router as analysis_router
from backend.api.database import router as database_router
from backend.api.llm import router as llm_router
from backend.api.models import router as models_router
from backend.api.training import router as training_router
from backend.api.intent import router as intent_router
from backend.api.automl import prune_automl_runtime, recover_interrupted_automl_jobs
from backend.api.training import prune_training_runtime, recover_interrupted_training_jobs
from backend.api._workflow_artifacts import iter_visible_workflow_artifacts
from backend.scheduler import start_scheduler, stop_scheduler
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_STATIC_PATH = Path(__file__).resolve().parent / "static"
FRONTEND_DIST_PATH = Path(os.getenv("FRONTEND_DIST_PATH", PROJECT_ROOT / "frontend" / "dist"))
RESERVED_FRONTEND_PREFIXES = {
    "api",
    "assets",
    "docs",
    "health",
    "openapi.json",
    "redoc",
    "static",
}
_runtime_recovery_done = False


def frontend_index_path() -> Path | None:
    """Return the preferred frontend entrypoint when a built UI is available."""
    for candidate in (FRONTEND_DIST_PATH / "index.html", BACKEND_STATIC_PATH / "index.html"):
        if candidate.exists():
            return candidate
    return None


def frontend_assets_path() -> Path | None:
    assets_path = FRONTEND_DIST_PATH / "assets"
    return assets_path if assets_path.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理
    
    启动前：开启 MLOps 监控定时任务
    关闭后：优雅结束调度器
    """
    # 启动前：将上次异常退出遗留的执行中任务转换为可解释的失败状态。
    global _runtime_recovery_done
    if not _runtime_recovery_done:
        interrupted = recover_interrupted_training_jobs() + recover_interrupted_automl_jobs()
        # Reconcile any Agent workflow artifact transaction that was interrupted
        # between durable publication and its completed-state commit.
        list(iter_visible_workflow_artifacts(PROJECT_ROOT))
        if interrupted:
            print(f"已恢复 {interrupted} 个中断任务的持久化状态")
        try:
            retention_days = max(1, min(int(os.getenv("ML_RUNTIME_RETENTION_DAYS", "30")), 3650))
        except ValueError:
            retention_days = 30
        pruned = prune_training_runtime(retention_days) + prune_automl_runtime(retention_days)
        if pruned:
            print(f"已清理 {pruned} 条超过 {retention_days} 天的运行状态")
        _runtime_recovery_done = True

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

def _client_is_local(request: Request) -> bool:
    """判断请求是否来自本机（回环地址）。

    ASGI 服务器提供的真实客户端地址总是合法 IP；无法解析的值只会出现在
    测试夹具（如 starlette TestClient 的 "testclient"）中，按本机处理以
    保持开发/测试体验一致。
    """
    client = request.client
    host = client.host if client else None
    if not host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """API 密钥鉴权 + 未鉴权场景的网络暴露兜底。

    设置环境变量 ``API_KEY`` 后，所有 ``/api/*`` 请求必须携带匹配的
    ``X-API-Key`` 头（或 ``Authorization: Bearer <key>``）。

    未设置 ``API_KEY`` 时，仅允许来自回环地址的 ``/api/*`` 请求——
    避免服务被绑定到 0.0.0.0 等对外地址后，数据库连接管理、训练任务、
    LLM 配置等管理端点被匿名访问（安全审查：未鉴权远程访问为高危暴露面）。
    反向代理等合法部署请设置 ``API_KEY``，或用
    ``ALLOW_NETWORK_API_WITHOUT_AUTH=1`` 显式承担风险放开。
    """
    if request.url.path.startswith("/api"):
        expected = os.getenv("API_KEY")
        if expected:
            provided = request.headers.get("X-API-Key")
            if not provided:
                authorization = request.headers.get("Authorization", "")
                if authorization.startswith("Bearer "):
                    provided = authorization[len("Bearer "):]
            if not provided or not hmac.compare_digest(provided, expected):
                return JSONResponse(status_code=401, content={"detail": "无效或缺失的 API 密钥"})
        elif os.getenv("ALLOW_NETWORK_API_WITHOUT_AUTH", "").lower() not in {"1", "true", "yes"}:
            if not _client_is_local(request):
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "未配置 API_KEY，拒绝非本机访问。请设置 API_KEY 环境变量，"
                        "或通过 ALLOW_NETWORK_API_WITHOUT_AUTH=1 显式放开。"
                    },
                )
    return await call_next(request)


# 挂载静态文件
BACKEND_STATIC_PATH.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BACKEND_STATIC_PATH)), name="static")
if frontend_assets_path():
    app.mount("/assets", StaticFiles(directory=str(frontend_assets_path())), name="frontend-assets")

# 挂载业务路由
app.include_router(workflow_router)
app.include_router(mlops_router)
app.include_router(automl_router)
app.include_router(analysis_router)
app.include_router(llm_router)
app.include_router(models_router)
app.include_router(database_router)
app.include_router(training_router)
app.include_router(intent_router)


@app.get("/")
async def root():
    """根路径 - 返回美观的欢迎页面"""
    index_path = frontend_index_path()
    if index_path:
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


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_history_fallback(full_path: str):
    """Serve the Vue entrypoint for frontend history routes."""
    first_segment = full_path.split("/", 1)[0]
    if first_segment in RESERVED_FRONTEND_PREFIXES:
        raise HTTPException(status_code=404, detail="Not Found")
    index_path = frontend_index_path()
    if index_path:
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend build not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=8000)
