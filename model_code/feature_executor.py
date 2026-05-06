"""
Feature Engineering Executor - 特征工程执行器

MLOps 核心概念：有状态执行 (Stateful Execution)

核心设计:
1. fit_transform (训练阶段) - 计算统计量并保存状态
2. transform (推理阶段) - 只应用训练时保存的状态
3. 防止数据穿越 - 验证集/线上推理不使用全量统计量
4. 可持久化 - 与模型一起打包保存，供线上 API 调用
"""

import pandas as pd
import numpy as np
import category_encoders as ce
from typing import Dict, Any, Optional, List
import logging
import joblib

from core.llm.agent_schemas import FeatureStrategy, FeatureAction

logger = logging.getLogger(__name__)


class FeatureEngineeringExecutor:
    """
    特征工程执行器 (有状态)
    
    核心设计:
    - 训练阶段：fit_transform() 计算统计量并保存
    - 推理阶段：transform() 只应用保存的状态
    - 防止数据穿越：验证集/线上推理不使用全量统计量
    - 可持久化：与模型一起打包保存
    """
    
    def __init__(self, target_col: Optional[str] = None):
        """
        初始化执行器
        
        Args:
            target_col: 目标变量列名
        """
        self.target_col = target_col
        self.strategy: Optional[FeatureStrategy] = None
        
        # 【核心设计】：状态字典，用于保存训练集上的统计量或拟合好的编码器
        # 保证模型上线后的 /predict 接口能 100% 还原训练时的特征处理环境
        self.state: Dict[str, Any] = {
            'fill_values': {},      # 保存填充的均值/中位数/众数
            'encoders': {},         # 保存 fit 好的 TargetEncoder 或 OneHotEncoder
            'scalers': {},          # 保存 fit 好的 StandardScaler 等
            'feature_columns': [],  # 保存处理后的特征列顺序
            'fitted': False         # 是否已 fit
        }
        
        logger.info("FeatureEngineeringExecutor 初始化完成")
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        strategy_obj: FeatureStrategy,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        训练阶段：根据 LLM 的策略清洗数据，并记住所有状态
        
        Args:
            df: 输入 DataFrame
            strategy_obj: FeatureAgent 生成的策略 (Pydantic 对象)
            y: 目标变量 (用于 Target Encoding)
            
        Returns:
            pd.DataFrame: 处理后的 DataFrame
        """
        logger.info(f"开始 fit_transform，共 {len(strategy_obj.actions)} 个处理动作")
        
        self.strategy = strategy_obj
        df_out = df.copy()
        
        # 如果没有单独传入 y，尝试从 df 中提取
        if y is None and self.target_col and self.target_col in df_out.columns:
            y = df_out[self.target_col]
        
        # 按顺序执行每个处理动作
        for action_item in self.strategy.actions:
            df_out = self._execute_action(df_out, action_item, y, fit=True)
        
        # 保存处理后的特征列顺序 (用于推理时验证)
        self.state['feature_columns'] = list(df_out.columns)
        self.state['fitted'] = True
        
        logger.info(f"fit_transform 完成，最终形状：{df_out.shape}")
        return df_out
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        推理阶段：只应用训练时保存的状态，不计算新统计量
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            pd.DataFrame: 处理后的 DataFrame
            
        Raises:
            RuntimeError: 如果执行器尚未 fit
        """
        if not self.state['fitted']:
            raise RuntimeError("执行器尚未 fit，无法进行 transform！")
        
        if not self.strategy:
            raise RuntimeError("执行器策略为空，无法进行 transform！")
        
        logger.info(f"开始 transform，共 {len(self.strategy.actions)} 个处理动作")
        
        df_out = df.copy()
        
        # 按顺序执行每个处理动作
        for action_item in self.strategy.actions:
            df_out = self._execute_action(df_out, action_item, y=None, fit=False)
        
        # 验证特征列顺序
        if list(df_out.columns) != self.state['feature_columns']:
            logger.warning("特征列顺序与训练时不一致，可能导致模型预测异常")
        
        logger.info(f"transform 完成，最终形状：{df_out.shape}")
        return df_out
    
    def _execute_action(
        self,
        df: pd.DataFrame,
        action_item: FeatureAction,
        y: Optional[pd.Series],
        fit: bool
    ) -> pd.DataFrame:
        """
        执行单个处理动作
        
        Args:
            df: 输入 DataFrame
            action_item: 处理动作 (Pydantic 对象)
            y: 目标变量 (仅 fit 时需要)
            fit: 是否为训练阶段
            
        Returns:
            pd.DataFrame: 处理后的 DataFrame
        """
        col = action_item.column
        action = action_item.action
        
        # 防御性编程：保护目标列不被删除
        if action == "drop" and col == self.target_col:
            logger.info(f"⚠️ 跳过对目标列 [{col}] 的 drop 操作，训练需要保留目标变量")
            return df

        # 防御性编程：无视 LLM 产生的列名幻觉
        if col not in df.columns:
            logger.warning(f"⚠️ 列 [{col}] 不存在，已安全跳过操作：{action}")
            return df
        
        logger.debug(f"⚙️ 执行算子：{col} -> {action} (原因：{action_item.reason})")
        
        try:
            if action == "drop":
                return self._drop(df, col)
            
            elif action.startswith("fillna_"):
                return self._fillna(df, col, action, fit)
            
            elif action.endswith("_transform"):
                return self._transform_numeric(df, col, action)
            
            elif action.endswith("_encode"):
                return self._encode(df, col, action, y, fit)
            
            elif action.startswith("extract_"):
                return self._extract_time_feature(df, col, action)
            
            elif action.startswith("scale_"):
                return self._scale(df, col, action, fit)
            
            elif action == "binning":
                return self._binning(df, col, action_item.params)
            
            elif action == "interaction":
                return self._interaction(df, col, action_item.params)
            
            else:
                logger.warning(f"未知动作：{action}，已跳过")
                return df
                
        except Exception as e:
            logger.error(f"执行动作 {col}.{action} 失败：{e}")
            # 失败时返回原 DataFrame，不中断流程
            return df
    
    # ==================== 具体算子实现 ====================
    
    def _drop(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """删除字段"""
        df.drop(columns=[col], inplace=True, errors='ignore')
        return df
    
    def _fillna(
        self,
        df: pd.DataFrame,
        col: str,
        action: str,
        fit: bool
    ) -> pd.DataFrame:
        """填充缺失值"""
        if fit:
            # 训练阶段：计算并保存统计量
            if action == "fillna_mean":
                val = df[col].mean()
            elif action == "fillna_median":
                val = df[col].median()
            elif action == "fillna_mode":
                val = df[col].mode()
                val = val[0] if len(val) > 0 else 0
            elif action == "fillna_constant":
                val = 0  # 可从 params 获取
            else:
                val = 0
            
            self.state['fill_values'][col] = val
            logger.debug(f"保存填充值：{col} = {val}")
        
        # 应用填充 (训练和推理都用保存的值)
        saved_val = self.state['fill_values'].get(col, 0)
        df = df.copy()
        df[col] = df[col].fillna(saved_val)
        return df
    
    def _transform_numeric(
        self,
        df: pd.DataFrame,
        col: str,
        action: str
    ) -> pd.DataFrame:
        """数值变换"""
        df = df.copy()
        
        if action == "log_transform":
            # np.log1p 防止 log(0) 报错，clip(lower=0) 防止负数报错
            df[col] = np.log1p(df[col].clip(lower=0))
        
        elif action == "sqrt_transform":
            df[col] = np.sqrt(df[col].clip(lower=0))
        
        elif action == "boxcox_transform":
            from scipy import stats
            # Box-Cox 需要正值
            df[col], _ = stats.boxcox(df[col].clip(lower=1e-6))
        
        return df
    
    def _encode(
        self,
        df: pd.DataFrame,
        col: str,
        action: str,
        y: Optional[pd.Series],
        fit: bool
    ) -> pd.DataFrame:
        """类别编码"""
        df = df.copy()
        
        if action == "target_encode":
            if fit:
                # 训练阶段：fit 并保存编码器
                encoder = ce.TargetEncoder(cols=[col])
                df[col] = encoder.fit_transform(df[col], y)
                self.state['encoders'][col] = encoder
                logger.debug(f"保存 TargetEncoder：{col}")
            else:
                # 推理阶段：使用保存的编码器
                encoder = self.state['encoders'].get(col)
                if encoder:
                    df[col] = encoder.transform(df[col])
                else:
                    logger.warning(f"未找到保存的编码器：{col}")
        
        elif action == "onehot_encode":
            if fit:
                encoder = ce.OneHotEncoder(cols=[col], use_cat_names=True)
                df = encoder.fit_transform(df)
                self.state['encoders'][col] = encoder
                logger.debug(f"保存 OneHotEncoder：{col}")
            else:
                encoder = self.state['encoders'].get(col)
                if encoder:
                    df = encoder.transform(df)
                else:
                    logger.warning(f"未找到保存的编码器：{col}")
        
        elif action == "label_encode":
            if fit:
                # Label Encoding 需要保存映射关系
                unique_vals = df[col].astype('category').cat.categories
                self.state['encoders'][col] = {'type': 'label', 'categories': unique_vals}
                df[col] = df[col].astype('category').cat.codes
            else:
                saved = self.state['encoders'].get(col)
                if saved and saved.get('type') == 'label':
                    categories = saved['categories']
                    df[col] = df[col].astype('category').cat.set_categories(categories).cat.codes
        
        elif action == "frequency_encode":
            if fit:
                freq = df[col].value_counts(normalize=True).to_dict()
                self.state['encoders'][col] = {'type': 'frequency', 'mapping': freq}
                df[col] = df[col].map(freq)
            else:
                saved = self.state['encoders'].get(col)
                if saved and saved.get('type') == 'frequency':
                    df[col] = df[col].map(saved['mapping'])
        
        return df
    
    def _extract_time_feature(
        self,
        df: pd.DataFrame,
        col: str,
        action: str
    ) -> pd.DataFrame:
        """时间特征提取"""
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
        if action == "extract_hour":
            df[col] = df[col].dt.hour
        elif action == "extract_day":
            df[col] = df[col].dt.dayofweek
        elif action == "extract_month":
            df[col] = df[col].dt.month
        elif action == "extract_year":
            df[col] = df[col].dt.year
        
        # 无法解析的时间补 -1
        df[col] = df[col].fillna(-1)
        return df
    
    def _scale(
        self,
        df: pd.DataFrame,
        col: str,
        action: str,
        fit: bool
    ) -> pd.DataFrame:
        """特征缩放"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        df = df.copy()
        
        if action == "scale_standard":
            if fit:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
                self.state['scalers'][col] = scaler
            else:
                scaler = self.state['scalers'].get(col)
                if scaler:
                    df[col] = scaler.transform(df[[col]])
        
        elif action == "scale_minmax":
            if fit:
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(df[[col]])
                self.state['scalers'][col] = scaler
            else:
                scaler = self.state['scalers'].get(col)
                if scaler:
                    df[col] = scaler.transform(df[[col]])
        
        elif action == "scale_robust":
            if fit:
                scaler = RobustScaler()
                df[col] = scaler.fit_transform(df[[col]])
                self.state['scalers'][col] = scaler
            else:
                scaler = self.state['scalers'].get(col)
                if scaler:
                    df[col] = scaler.transform(df[[col]])
        
        return df
    
    def _binning(
        self,
        df: pd.DataFrame,
        col: str,
        params: Optional[Dict]
    ) -> pd.DataFrame:
        """分箱处理"""
        df = df.copy()
        bins = params.get('bins', 5) if params else 5
        
        try:
            df[col] = pd.qcut(df[col], q=bins, labels=False, duplicates='drop')
        except Exception as e:
            logger.warning(f"分箱失败：{e}，使用等宽分箱")
            df[col] = pd.cut(df[col], bins=bins, labels=False)
        
        return df
    
    def _interaction(
        self,
        df: pd.DataFrame,
        col: str,
        params: Optional[Dict]
    ) -> pd.DataFrame:
        """特征交叉"""
        df = df.copy()
        with_cols = params.get('with', []) if params else []
        
        for other_col in with_cols:
            if other_col in df.columns:
                new_col = f"{col}_x_{other_col}"
                df[new_col] = df[col] * df[other_col]
        
        return df
    
    # ==================== 持久化方法 ====================
    
    def save(self, path: str):
        """
        保存执行器状态
        
        Args:
            path: 保存路径
        """
        joblib.dump({
            'target_col': self.target_col,
            'strategy': self.strategy,
            'state': self.state
        }, path)
        logger.info(f"执行器已保存：{path}")
    
    def load(self, path: str):
        """
        加载执行器状态
        
        Args:
            path: 加载路径
        """
        data = joblib.load(path)
        self.target_col = data['target_col']
        self.strategy = data['strategy']
        self.state = data['state']
        logger.info(f"执行器已加载：{path}")
    
    def get_state(self) -> Dict[str, Any]:
        """获取状态字典"""
        return self.state.copy()
    
    def is_fitted(self) -> bool:
        """检查是否已 fit"""
        return self.state['fitted']


# ==========================================
# 便捷函数
# ==========================================

def create_feature_pipeline(
    df: pd.DataFrame,
    strategy_obj: FeatureStrategy,
    target_col: Optional[str] = None,
    y: Optional[pd.Series] = None,
    save_path: Optional[str] = None
) -> tuple:
    """
    创建特征工程流水线的便捷函数
    
    Args:
        df: 训练数据
        strategy_obj: FeatureAgent 生成的策略
        target_col: 目标变量列名
        y: 目标变量 Series
        save_path: 保存路径 (可选)
        
    Returns:
        tuple: (executor, df_clean)
    """
    executor = FeatureEngineeringExecutor(target_col=target_col)
    df_clean = executor.fit_transform(df, strategy_obj, y)
    
    if save_path:
        executor.save(save_path)
    
    return executor, df_clean
