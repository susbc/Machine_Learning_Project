"""
AutoML Optuna SSE 测试脚本

独立可运行的 FastAPI 测试，验证 SSE 流式推送逻辑。

核心机制:
1. CORS 跨域处理 - 允许 Vue3 前端访问
2. 线程安全数据投递 - asyncio.run_coroutine_threadsafe
3. 模拟训练任务 - 带 time.sleep 的 mock_objective
4. curl + 前端双验证

使用方法:
    python scripts/examples/test_optuna_sse.py

访问:
    http://127.0.0.1:8000/api/optimize/stream?total_trials=50
"""

import asyncio
import json
import time
import random
import optuna
from fastapi import FastAPI, APIRouter
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor

# ==================== 1. 初始化 FastAPI 与跨域中间件 ====================

app = FastAPI(title="AutoML Optuna SSE Test")

# CORS 跨域配置 (允许 Vue 开发环境跨域请求)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

# 线程池：专门用来跑阻塞的 ML 训练任务
executor = ThreadPoolExecutor(max_workers=3)


# ==================== 2. 模拟机器学习目标函数 (Objective) ====================

def mock_objective(trial):
    """
    模拟一个机器学习训练任务
    
    包含:
    - 模拟训练耗时 (time.sleep)
    - 模拟超参数采样
    - 模拟 AUC 指标收敛
    """
    # 模拟模型训练耗时 (0.2 秒)
    time.sleep(0.2)
    
    # 模拟 Optuna 超参数采样
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    
    # 模拟 AUC 指标 (随 trial 逐渐收敛)
    # 算法：基础值 + 改进值 + 随机噪声
    base_auc = 0.65
    improvement = (0.3 / 10) * min(max_depth, 10) + (0.05 / 0.2) * learning_rate
    noise = random.uniform(-0.02, 0.02)
    
    final_auc = min(base_auc + improvement + noise, 0.99)
    
    return final_auc


# ==================== 3. SSE 异步生成器 ====================

async def sse_generator(queue: asyncio.Queue, total_trials: int):
    """
    异步生成器：持续从队列中读取数据并推送给客户端
    
    Args:
        queue: 异步队列 (存放 Trial 结果)
        total_trials: 总 Trial 次数
    """
    try:
        while True:
            # 阻塞等待队列中的新数据
            data = await queue.get()
            
            # 收到 None 代表训练彻底结束
            if data is None:
                yield f"data: {json.dumps({'status': 'DONE', 'progress': 100})}\n\n"
                break
            
            # 计算当前进度
            progress = int((data['trial_number'] + 1) / total_trials * 100)
            data['progress'] = progress
            
            # SSE 协议格式：data: {json}\n\n
            yield f"data: {json.dumps(data)}\n\n"
    
    except asyncio.CancelledError:
        print("⚠️ 警告：客户端断开了连接 (例如关闭了浏览器页面)")
        # 实际项目中，这里可以触发 Event 去终止 Optuna 运行
        raise


# ==================== 4. SSE 核心路由 ====================

@router.get("/api/optimize/stream")
async def stream_optimization(total_trials: int = 50):
    """
    SSE 流式 Optuna 优化接口
    
    Args:
        total_trials: Trial 次数
        
    Returns:
        StreamingResponse: SSE 数据流
    """
    # 创建异步队列
    queue = asyncio.Queue()
    
    # 获取当前 FastAPI 事件循环
    loop = asyncio.get_running_loop()
    
    # ==================== Optuna 回调函数 ====================
    
    def optuna_callback(study, trial):
        """
        Optuna 回调：每次 Trial 结束时触发，将数据投递到队列
        
        【关键】使用 asyncio.run_coroutine_threadsafe 保证线程安全
        """
        payload = {
            "status": "RUNNING",
            "trial_number": trial.number,
            "value": float(trial.value) if trial.value else None,
            "best_value": float(study.best_value) if study.best_value else None,
            "params": trial.params
        }
        
        # 【关键】将数据从同步线程安全地送入异步队列
        asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
    
    # ==================== Optuna 运行函数 ====================
    
    def run_optuna():
        """
        在线程池中运行 Optuna 优化
        
        无论成功失败，最后都发送结束信号 (None)
        """
        try:
            print(f"🚀 开始执行 Optuna 寻优，总计 {total_trials} 次 Trial...")
            study = optuna.create_study(direction="maximize", study_name="sse_test")
            study.optimize(
                mock_objective,
                n_trials=total_trials,
                callbacks=[optuna_callback]
            )
            print(f"✅ 寻优完成！")
            print(f"   最佳参数：{study.best_params}")
            print(f"   最佳分数：{study.best_value:.4f}")
        
        except Exception as e:
            print(f"❌ Optuna 运行报错：{e}")
        
        finally:
            # 无论成功失败，都发送结束信号
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
    
    # ==================== 启动线程 ====================
    
    # 【关键】把阻塞的 CPU 计算任务扔到线程池，不卡死 FastAPI 主事件循环
    loop.run_in_executor(executor, run_optuna)
    
    # 返回流式响应 (Content-Type: text/event-stream)
    return StreamingResponse(
        sse_generator(queue, total_trials),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Nginx 禁用缓冲
        }
    )


# ==================== 5. 应用注册 ====================

app.include_router(router)


# ==================== 6. 主入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 AutoML Optuna SSE 测试服务启动")
    print("=" * 60)
    print()
    print("📡 SSE 接口地址:")
    print("   http://127.0.0.1:8000/api/optimize/stream?total_trials=50")
    print()
    print("🧪 测试方法:")
    print("   1. curl 测试:")
    print("      curl -N http://127.0.0.1:8000/api/optimize/stream?total_trials=10")
    print()
    print("   2. 前端联调:")
    print("      EventSource('http://127.0.0.1:8000/api/optimize/stream?total_trials=50')")
    print()
    print("📊 Vue 3 前端组件位置:")
    print("   frontend/src/components/OptunaDashboard.vue")
    print()
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
