"""
日志配置模块

提供结构化日志 (JSON 格式)、日志轮转、训练进度持久化等功能

使用示例:
    from logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("训练开始", extra={"stage": "train", "progress": 0})
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import os

# 第三方库
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


# ============================================================================
# JSON 格式化器
# ============================================================================

class JSONFormatter(logging.Formatter):
    """
    JSON 格式的日志格式化器
    
    输出示例:
    {
        "timestamp": "2024-01-15T10:30:00.123456",
        "level": "INFO",
        "logger": "model_code.train",
        "message": "训练开始",
        "stage": "train",
        "progress": 0,
        "module": "train",
        "function": "lgbm_train_with_random_search",
        "line": 45
    }
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化为 JSON 字符串"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # 添加额外字段
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text',
                              'filename', 'funcName', 'id', 'levelname', 'levelno',
                              'lineno', 'module', 'msecs', 'message', 'msg', 'name',
                              'pathname', 'process', 'processName', 'relativeCreated',
                              'stack_info', 'thread', 'threadName']:
                    try:
                        # 尝试序列化
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """
    控制台日志格式化器 (带颜色)
    """
    
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 添加颜色
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # 格式化时间
        record.asctime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 添加额外信息
        extra_info = ""
        if hasattr(record, 'stage'):
            extra_info += f" [{record.stage}]"
        if hasattr(record, 'progress'):
            extra_info += f" {record.progress}%"
        
        return f"{record.asctime} | {record.levelname:8} | {record.name:30} | {record.getMessage()}{extra_info}"


# ============================================================================
# 日志配置类
# ============================================================================

class LoggingConfig:
    """
    日志配置管理器
    
    特性:
    - 结构化日志 (JSON 格式)
    - 日志轮转 (按大小和时间)
    - 多级别日志输出
    - 训练进度持久化
    - 错误日志告警
    """
    
    _initialized = False
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def initialize(
        cls,
        log_dir: Optional[Path] = None,
        log_level: str = "INFO",
        log_format: str = "json",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 10,
        console_output: bool = True,
    ) -> None:
        """
        初始化日志系统
        
        Args:
            log_dir: 日志目录路径
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: 日志格式 (json, text)
            max_bytes: 单个日志文件最大大小 (字节)
            backup_count: 保留的备份文件数量
            console_output: 是否输出到控制台
        """
        if cls._initialized:
            return
        
        # 默认日志目录
        if log_dir is None:
            log_dir = Path(__file__).parent / "logs"
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 文件处理器 (JSON 格式)
        if log_format == "json":
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # 按大小轮转的文件处理器
        rotating_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        rotating_handler.setFormatter(file_formatter)
        rotating_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(rotating_handler)
        
        # 按时间轮转的文件处理器 (每天)
        timed_handler = TimedRotatingFileHandler(
            log_dir / "app_daily.log",
            when='D',
            interval=1,
            backupCount=backup_count,
            encoding='utf-8'
        )
        timed_handler.setFormatter(file_formatter)
        timed_handler.setLevel(logging.INFO)
        root_logger.addHandler(timed_handler)
        
        # 错误日志专用处理器
        error_handler = RotatingFileHandler(
            log_dir / "error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setFormatter(file_formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        
        # 训练进度日志处理器
        progress_handler = RotatingFileHandler(
            log_dir / "training_progress.log",
            maxBytes=max_bytes,
            backupCount=5,
            encoding='utf-8'
        )
        progress_handler.setFormatter(file_formatter)
        progress_handler.setLevel(logging.INFO)
        progress_handler.addFilter(lambda record: hasattr(record, 'stage'))
        root_logger.addHandler(progress_handler)
        
        # 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleFormatter())
            console_handler.setLevel(logging.INFO)
            root_logger.addHandler(console_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取命名日志记录器
        
        Args:
            name: 日志记录器名称 (通常为 __name__)
        
        Returns:
            配置好的日志记录器
        """
        if not cls._initialized:
            cls.initialize()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]


# ============================================================================
# 便捷函数
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
    
    Returns:
        配置好的日志记录器
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Hello, World!")
    """
    return LoggingConfig.get_logger(name)


def log_training_progress(
    logger: logging.Logger,
    stage: str,
    progress: float,
    message: str,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    记录训练进度日志
    
    Args:
        logger: 日志记录器
        stage: 训练阶段 (data_load, preprocess, train, evaluate, etc.)
        progress: 进度百分比 (0-100)
        message: 日志消息
        metrics: 额外指标字典
    """
    extra = {
        'stage': stage,
        'progress': round(progress, 2)
    }
    
    if metrics:
        extra.update(metrics)
    
    logger.info(message, extra=extra)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    记录带上下文的错误日志
    
    Args:
        logger: 日志记录器
        error: 异常对象
        context: 上下文信息字典
    """
    extra = {
        'error_type': type(error).__name__,
        'error_message': str(error)
    }
    
    if context:
        extra.update(context)
    
    logger.error(f"错误：{error}", exc_info=True, extra=extra)


# ============================================================================
# 自动初始化
# ============================================================================

# 检查是否通过环境变量配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.getenv('LOG_FORMAT', 'json')
LOG_DIR = os.getenv('LOG_DIR', None)

if LOG_DIR:
    LoggingConfig.initialize(
        log_dir=Path(LOG_DIR),
        log_level=LOG_LEVEL,
        log_format=LOG_FORMAT
    )
