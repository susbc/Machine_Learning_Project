"""
配置文件 - 优化版

优化点:
1. 使用 Path 对象替代字符串路径
2. 添加配置验证
3. 支持环境变量覆盖
4. 添加配置说明
"""

from pathlib import Path
import os
from typing import Optional


# ============================================================================
# 基础路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 输入输出目录
INPUT_DIR = PROJECT_ROOT / "input_data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "model_code" / "saved"
CONFIG_DIR = PROJECT_ROOT / "model_code" / "config"

# 确保目录存在
for dir_path in [INPUT_DIR, OUTPUT_DIR, MODEL_DIR, CONFIG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 数据配置
# ============================================================================

# 数据文件路径 (支持环境变量覆盖)
DATA_PATH = os.getenv(
    "ML_DATA_PATH",
    str(INPUT_DIR / "train_data.csv")
)

# 配置表路径
CONFIG_PATH = os.getenv(
    "ML_CONFIG_PATH",
    str(INPUT_DIR / "field_config.xlsx")
)


# ============================================================================
# 训练配置
# ============================================================================

# 是否执行训练
IS_TRAIN = os.getenv("ML_IS_TRAIN", "true").lower() == "true"

# 是否执行预测
IS_PREDICT = os.getenv("ML_IS_PREDICT", "true").lower() == "true"

# 测试集比例
TEST_SIZE = float(os.getenv("ML_TEST_SIZE", "0.3"))

# 随机种子
RANDOM_STATE = int(os.getenv("ML_RANDOM_STATE", "42"))

# 交叉验证折数
CV_FOLDS = int(os.getenv("ML_CV_FOLDS", "5"))

# 随机搜索迭代次数
N_ITER = int(os.getenv("ML_N_ITER", "10"))


# ============================================================================
# 模型配置
# ============================================================================

# 模型保存路径
MODEL_SAVE_PATH = MODEL_DIR / "best_lgbm_model.joblib"

# 训练摘要保存路径
TRAIN_SUMMARY_PATH = MODEL_DIR / "train_summary.joblib"


# ============================================================================
# 预测配置
# ============================================================================

# 默认分类阈值 (可在训练后更新)
DEFAULT_THRESHOLD = float(os.getenv("ML_THRESHOLD", "0.5"))

# 预测结果前缀
PREDICT_PREFIX = os.getenv("ML_PREDICT_PREFIX", "predict_res")


# ============================================================================
# 日志配置
# ============================================================================

# 日志级别
LOG_LEVEL = os.getenv("ML_LOG_LEVEL", "INFO")

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 配置验证
# ============================================================================

def validate_config() -> bool:
    """
    验证配置是否有效
    
    Returns:
        配置是否有效
    """
    errors = []
    
    # 检查输入目录
    if not INPUT_DIR.exists():
        errors.append(f"输入目录不存在：{INPUT_DIR}")
    
    # 检查配置文件
    if not Path(CONFIG_PATH).exists():
        errors.append(f"配置文件不存在：{CONFIG_PATH}")
    
    # 检查数据文件
    if not Path(DATA_PATH).exists():
        errors.append(f"数据文件不存在：{DATA_PATH}")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    
    print("配置验证通过 ✅")
    return True


# ============================================================================
# 配置打印
# ============================================================================

def print_config() -> None:
    """打印当前配置"""
    print("\n" + "=" * 60)
    print("当前配置")
    print("=" * 60)
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"输入目录：{INPUT_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"模型目录：{MODEL_DIR}")
    print(f"数据路径：{DATA_PATH}")
    print(f"配置路径：{CONFIG_PATH}")
    print(f"训练：{IS_TRAIN}, 预测：{IS_PREDICT}")
    print(f"测试集比例：{TEST_SIZE}, 随机种子：{RANDOM_STATE}")
    print(f"交叉验证：{CV_FOLDS}折，搜索迭代：{N_ITER}次")
    print(f"分类阈值：{DEFAULT_THRESHOLD}")
    print("=" * 60 + "\n")


# 向后兼容的旧配置 (保留但标记为废弃)
model_path = str(MODEL_SAVE_PATH)
input_path = str(INPUT_DIR)
output_path = str(OUTPUT_DIR)
data_path = DATA_PATH
config_path = CONFIG_PATH
is_train = IS_TRAIN
is_pretrain = IS_PREDICT
