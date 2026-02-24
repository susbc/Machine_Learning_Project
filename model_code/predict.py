import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
import os

from config.config import *


def load_and_predict(data, key,data_month):
    """
    加载已保存的模型并进行预测
    """
    # 加载模型
    model = joblib.load(model_path)

    # 进行预测
    predictions = model.predict(data)
    predictions_proba = model.predict_proba(data)

    # 修正：使用np.column_stack而不是np._c
    # 修正：只取正类的概率（第二列）
    # 修正：确保key是数组形式
    pred_re = np.column_stack([key.values if hasattr(key, 'values') else key,
                               predictions_proba[:, 1]])

    # 创建DataFrame
    res_df = pd.DataFrame(pred_re, columns=['key', 'probability'])

    # 修正：使用to_csv而不是to，并确保路径正确
    # 确保目录存在
    os.makedirs("../output", exist_ok=True)
    res_df.to_csv(f"../output/predict_res_{data_month}.csv", index=False)

    return res_df