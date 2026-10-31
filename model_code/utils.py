"""
机器学习特征工程工具库 - 优化版

优化点:
1. 移除不必要的 .copy()，使用 inplace 操作减少内存占用
2. 使用向量化操作替代 iterrows() 循环
3. 重构采样逻辑，使用更高效的算法
4. 添加 sklearn Pipeline 兼容的预处理器类
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# ============================================================================
# 智能采样模块
# ============================================================================

def smart_sampling(df: pd.DataFrame, label_col: str = 'label', positive_label: int = 1, 
                   negative_ratio: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """
    智能抽样函数 - 优化版
    
    优化: 使用向量化操作，减少内存拷贝
    """
    np.random.seed(random_state)
    
    # 向量化分离样本
    positive_mask = df[label_col] == positive_label
    positive_df = df[positive_mask].copy()
    negative_df = df[~positive_mask]
    
    print(f"原始数据：{len(df):,} 条记录")
    print(f"正样本 ({positive_label}): {len(positive_df):,} 条")
    print(f"负样本：{len(negative_df):,} 条")
    
    # 负样本抽样
    negative_sampled = negative_df.sample(frac=negative_ratio, random_state=random_state)
    
    # 合并并打乱
    df_sampled = pd.concat([positive_df, negative_sampled], ignore_index=True)
    df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n抽样后数据：{len(df_sampled):,} 条记录")
    print(f"正样本：{len(df_sampled[df_sampled[label_col] == positive_label]):,} 条")
    print(f"负样本抽样比例：{negative_ratio:.0%}")
    
    return df_sampled


def neg_data_undersampling(neg_df: pd.DataFrame, target_stats: np.ndarray, 
                           step_size: int, max_iter: int, error_rate: float, 
                           max_samples: int = 50000) -> pd.Index:
    """
    通过迭代寻找符合统计特征的最优负样本子集 - 优化版
    
    优化:
    1. 向量化统计计算
    2. 提前终止条件优化
    3. 使用 numpy 向量化比较
    """
    allowed_error = np.abs(target_stats * error_rate)
    best_match_index = None
    min_overall_error = float('inf')
    
    # 预计算负样本统计量，避免重复计算
    neg_describe = neg_df.describe().T[["mean", "std"]]
    
    for i in range(1, max_iter + 1):
        n = i * step_size
        
        if n > max_samples or n > len(neg_df):
            break
        
        # 随机采样
        sample_indices = np.random.choice(neg_df.index, size=n, replace=False)
        sample_subset = neg_df.loc[sample_indices]
        
        # 向量化计算统计量
        current_stats = sample_subset.describe().T[["mean", "std"]].values
        
        # 向量化误差计算
        diff = np.abs(current_stats - target_stats)
        total_error = np.sum(diff)
        
        # 更新最优解
        if total_error < min_overall_error:
            min_overall_error = total_error
            best_match_index = sample_indices
        
        # 提前终止
        if np.all(diff <= allowed_error):
            print(f"✅ 在步长 {i} 找到理想样本，规模：{n:,}")
            return sample_indices
    
    print(f"⚠️ 未找到完全匹配样本。返回误差最小的样本子集，累计误差：{min_overall_error:.4f}")
    return best_match_index


# ============================================================================
# 类型转换模块 - 优化版
# ============================================================================

def convert_dataframe_types_vectorized(data: pd.DataFrame, df_field: pd.DataFrame) -> pd.DataFrame:
    """
    向量化数值类型转换 - 替代 iterrows()
    
    优化:
    1. 使用字典映射替代 iterrows 循环
    2. 批量转换数值列
    """
    # 构建字段类型映射字典
    field_mapping = df_field.set_index('field_name')['field_type'].to_dict()
    
    # 批量处理数值字段
    num_fields = [f for f, t in field_mapping.items() if t == 'num' and f in data.columns]
    
    if not num_fields:
        print("未找到 field_type 标记为'num'的字段")
        return data
    
    print(f"批量转换 {len(num_fields)} 个数值字段...")
    
    for field in num_fields:
        try:
            # 向量化转换
            data[field] = pd.to_numeric(data[field], errors='coerce')
        except Exception as e:
            print(f"字段 '{field}' 转换失败：{e}")
    
    return data


# ============================================================================
# 异常值处理模块 - 优化版
# ============================================================================

def handle_dataframe_outliers(data: pd.DataFrame, df_field: pd.DataFrame) -> pd.DataFrame:
    """
    异常值处理 - 优化版

    ⚠️ DEPRECATED: 此函数会在调用时对传入数据计算统计量，
    如果在 train/test split 之前调用会导致数据穿越。
    推荐使用 ConfigDrivenPreprocessor (fit 在训练集, transform 在测试集)。

    优化:
    1. 移除 .copy()，直接修改原数据
    2. 使用字典映射替代 iterrows
    3. 向量化异常值检测
    """
    warnings.warn(
        "handle_dataframe_outliers 存在数据穿越风险，推荐使用 ConfigDrivenPreprocessor",
        DeprecationWarning, stacklevel=2
    )
    # 构建异常值处理映射
    outlier_mapping = df_field.set_index('field_name')['outlier_value'].to_dict()
    
    for field_name, outlier_value in outlier_mapping.items():
        if field_name not in data.columns or pd.isna(outlier_value):
            continue
        
        # 确保数值类型
        if not np.issubdtype(data[field_name].dtype, np.number):
            data[field_name] = pd.to_numeric(data[field_name], errors='coerce')
        
        # 向量化处理
        if outlier_value == '3std':
            _handle_std_outliers_inplace(data, field_name, 3)
        elif outlier_value == '5std':
            _handle_std_outliers_inplace(data, field_name, 5)
        elif isinstance(outlier_value, str) and outlier_value.startswith('(') and outlier_value.endswith(')'):
            _handle_range_outliers_inplace(data, field_name, outlier_value)
    
    return data


def _handle_std_outliers_inplace(data: pd.DataFrame, field_name: str, std_multiplier: float) -> None:
    """就地处理基于标准差的异常值"""
    mean_val = data[field_name].mean()
    std_val = data[field_name].std()
    lower_bound = mean_val - std_multiplier * std_val
    upper_bound = mean_val + std_multiplier * std_val
    
    # 向量化裁剪
    data[field_name] = data[field_name].clip(lower_bound, upper_bound)
    
    print(f"字段 '{field_name}' 使用 {std_multiplier}σ 方法处理，边界 [{lower_bound:.4f}, {upper_bound:.4f}]")


def _handle_range_outliers_inplace(data: pd.DataFrame, field_name: str, range_str: str) -> None:
    """就地处理基于固定范围的异常值"""
    try:
        min_val, max_val = map(float, range_str.strip('()').split(','))
        data[field_name] = data[field_name].clip(min_val, max_val)
        print(f"字段 '{field_name}' 使用范围 [{min_val}, {max_val}] 处理异常值")
    except Exception as e:
        print(f"解析范围 '{range_str}' 失败：{e}")


# ============================================================================
# 缺失值处理模块 - 优化版
# ============================================================================

def handle_dataframe_missing_values(data: pd.DataFrame, df_field: pd.DataFrame) -> pd.DataFrame:
    """
    缺失值处理 - 优化版

    ⚠️ DEPRECATED: 此函数会在调用时对传入数据计算统计量，
    如果在 train/test split 之前调用会导致数据穿越。
    推荐使用 ConfigDrivenPreprocessor (fit 在训练集, transform 在测试集)。

    优化:
    1. 移除 .copy()，使用 inplace 操作
    2. 构建填充字典，批量 fillna
    3. 向量化统计计算
    """
    warnings.warn(
        "handle_dataframe_missing_values 存在数据穿越风险，推荐使用 ConfigDrivenPreprocessor",
        DeprecationWarning, stacklevel=2
    )
    # 构建缺失值处理映射
    missing_mapping = df_field.set_index('field_name')['missing_value'].to_dict()
    
    # 按填充方法分组
    fill_by_mean = []
    fill_by_median = []
    fill_by_mode = []
    fill_by_value = {}
    
    for field_name, missing_value in missing_mapping.items():
        if field_name not in data.columns or pd.isna(missing_value):
            continue
        
        if data[field_name].isna().sum() == 0:
            continue
        
        if missing_value == 'mean':
            fill_by_mean.append(field_name)
        elif missing_value == 'median':
            fill_by_median.append(field_name)
        elif missing_value == 'mode':
            fill_by_mode.append(field_name)
        else:
            fill_by_value[field_name] = missing_value
    
    # 批量处理
    if fill_by_mean:
        fill_dict = {f: data[f].mean() for f in fill_by_mean}
        data.fillna(fill_dict, inplace=True)
        print(f"使用均值填充字段：{fill_by_mean}")
    
    if fill_by_median:
        fill_dict = {f: data[f].median() for f in fill_by_median}
        data.fillna(fill_dict, inplace=True)
        print(f"使用中位数填充字段：{fill_by_median}")
    
    if fill_by_mode:
        for f in fill_by_mode:
            mode_val = data[f].mode()
            if len(mode_val) > 0:
                data[f] = data[f].fillna(mode_val.iloc[0])
        print(f"使用众数填充字段：{fill_by_mode}")
    
    if fill_by_value:
        data.fillna(fill_by_value, inplace=True)
        print(f"使用固定值填充字段：{list(fill_by_value.keys())}")
    
    return data


# ============================================================================
# 编码模块 - 优化版
# ============================================================================

def encode_dataframe(data: pd.DataFrame, df_field: pd.DataFrame) -> pd.DataFrame:
    """
    编码处理 - 优化版

    ⚠️ DEPRECATED: 此函数会对传入数据 fit 编码器，
    如果在 train/test split 之前调用会导致数据穿越。
    推荐使用 ConfigDrivenPreprocessor (fit 在训练集, transform 在测试集)。

    优化:
    1. 移除 .copy()
    2. 批量独热编码
    3. 向量化顺序编码
    """
    warnings.warn(
        "encode_dataframe 存在数据穿越风险，推荐使用 ConfigDrivenPreprocessor",
        DeprecationWarning, stacklevel=2
    )
    # 筛选字符串字段
    str_fields = df_field[df_field['field_type'] == 'str'][['field_name', 'ecode_type']]
    
    for _, row in str_fields.iterrows():
        field_name = row['field_name']
        ecode_type = row['ecode_type']
        
        if field_name not in data.columns:
            continue
        
        if ecode_type == 'one-hot':
            # 批量独热编码
            one_hot = pd.get_dummies(data[field_name], prefix=field_name)
            data = pd.concat([data, one_hot], axis=1)
            data.drop(field_name, axis=1, inplace=True)
            print(f"字段 '{field_name}' 独热编码，生成 {one_hot.shape[1]} 列")
        else:
            # 顺序编码
            encoder = OrdinalEncoder()
            data[field_name] = encoder.fit_transform(data[[field_name]]).flatten()
            print(f"字段 '{field_name}' 顺序编码")
    
    return data


# ============================================================================
# 标准化模块 - 优化版
# ============================================================================

def standardize_dataframe(data: pd.DataFrame, df_field: pd.DataFrame) -> pd.DataFrame:
    """
    标准化处理 - 优化版

    ⚠️ DEPRECATED: 此函数会对传入数据 fit 标准化器，
    如果在 train/test split 之前调用会导致数据穿越。
    推荐使用 ConfigDrivenPreprocessor (fit 在训练集, transform 在测试集)。

    优化:
    1. 移除 .copy()
    2. 批量标准化
    """
    warnings.warn(
        "standardize_dataframe 存在数据穿越风险，推荐使用 ConfigDrivenPreprocessor",
        DeprecationWarning, stacklevel=2
    )
    # 筛选需要标准化的字段
    std_fields = df_field[df_field['standardize_type'].notna()][['field_name', 'standardize_type']]
    
    scalers = {}
    
    for _, row in std_fields.iterrows():
        field_name = row['field_name']
        standardize_type = row['standardize_type']
        
        if field_name not in data.columns:
            continue
        
        # 确保数值类型
        if not np.issubdtype(data[field_name].dtype, np.number):
            data[field_name] = pd.to_numeric(data[field_name], errors='coerce')
        
        # 选择标准化器
        if standardize_type == 'z-score':
            scaler = StandardScaler()
        elif standardize_type == 'min-max':
            scaler = MinMaxScaler()
        elif standardize_type == 'robust':
            scaler = RobustScaler()
        else:
            continue
        
        data[field_name] = scaler.fit_transform(data[[field_name]]).flatten()
        scalers[field_name] = scaler
        print(f"字段 '{field_name}' 使用 {standardize_type} 标准化")
    
    return data


# ============================================================================
# sklearn Pipeline 兼容的预处理器类
# ============================================================================

class ConfigDrivenPreprocessor(BaseEstimator, TransformerMixin):
    """
    配置驱动的预处理器 - sklearn Pipeline 兼容
    
    优势:
    1. 防止数据穿越 (fit 在训练集，transform 在测试集)
    2. 支持 Pipeline 串联
    3. 可保存/加载配置
    """
    
    def __init__(
        self,
        df_field: pd.DataFrame,
        enable_missing: bool = True,
        enable_outlier: bool = True,
        enable_encoding: bool = True,
        enable_standardization: bool = True,
    ):
        self.df_field = df_field
        self.enable_missing = enable_missing
        self.enable_outlier = enable_outlier
        self.enable_encoding = enable_encoding
        self.enable_standardization = enable_standardization
        self.scalers_ = {}
        self.encoders_ = {}
        self.fill_values_ = {}
        self.outlier_bounds_ = {}
        self.feature_names_in_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """仅在训练集上拟合统计量"""
        self.feature_names_in_ = X.columns.tolist()

        if self.enable_missing:
            self._fit_missing_values(X)

        if self.enable_outlier:
            self._fit_outlier_bounds(X)

        if self.enable_standardization:
            self._fit_scalers(X)

        if self.enable_encoding:
            self._fit_encoders(X)

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用预处理规则"""
        X = X.copy()

        if self.enable_missing:
            X.fillna(self.fill_values_, inplace=True)

        if self.enable_outlier:
            for field, (lower, upper) in self.outlier_bounds_.items():
                if field in X.columns:
                    X[field] = X[field].clip(lower, upper)

        if self.enable_standardization:
            for field, scaler in self.scalers_.items():
                if field in X.columns:
                    X[field] = scaler.transform(X[[field]]).flatten()

        if self.enable_encoding:
            for field, encoder in self.encoders_.items():
                if field in X.columns:
                    encoded = encoder.transform(X[[field]])
                    if isinstance(encoder, OneHotEncoder):
                        encoded_df = pd.DataFrame(
                            encoded,
                            columns=encoder.get_feature_names_out([field]),
                            index=X.index,
                        )
                        X = pd.concat([X.drop(columns=[field]), encoded_df], axis=1)
                    else:
                        X[field] = encoded.flatten()

        return X
    
    def _fit_missing_values(self, X: pd.DataFrame):
        """拟合缺失值填充统计量"""
        missing_mapping = self.df_field.set_index('field_name')['missing_value'].to_dict()
        
        for field, method in missing_mapping.items():
            if field not in X.columns or pd.isna(method):
                continue
            
            if method == 'mean':
                self.fill_values_[field] = X[field].mean()
            elif method == 'median':
                self.fill_values_[field] = X[field].median()
            elif method == 'mode':
                mode = X[field].mode()
                self.fill_values_[field] = mode.iloc[0] if len(mode) > 0 else 0
            else:
                self.fill_values_[field] = method
    
    def _fit_outlier_bounds(self, X: pd.DataFrame):
        """拟合异常值边界"""
        outlier_mapping = self.df_field.set_index('field_name')['outlier_value'].to_dict()
        
        for field, method in outlier_mapping.items():
            if field not in X.columns or pd.isna(method):
                continue
            
            if method == '3std':
                mean, std = X[field].mean(), X[field].std()
                self.outlier_bounds_[field] = (mean - 3*std, mean + 3*std)
            elif method == '5std':
                mean, std = X[field].mean(), X[field].std()
                self.outlier_bounds_[field] = (mean - 5*std, mean + 5*std)
            elif isinstance(method, str) and method.startswith('('):
                bounds = method.strip('()').split(',')
                self.outlier_bounds_[field] = (float(bounds[0]), float(bounds[1]))
    
    def _fit_scalers(self, X: pd.DataFrame):
        """拟合标准化器"""
        std_fields = self.df_field[self.df_field['standardize_type'].notna()]
        
        for _, row in std_fields.iterrows():
            field = row['field_name']
            std_type = row['standardize_type']
            
            if field not in X.columns:
                continue
            
            scaler = None
            if std_type == 'z-score':
                scaler = StandardScaler()
            elif std_type == 'min-max':
                scaler = MinMaxScaler()
            elif std_type == 'robust':
                scaler = RobustScaler()

            if scaler is None:
                continue

            self.scalers_[field] = scaler
            self.scalers_[field].fit(X[[field]])
    
    def _fit_encoders(self, X: pd.DataFrame):
        """拟合编码器"""
        str_fields = self.df_field[self.df_field['field_type'] == 'str']
        
        for _, row in str_fields.iterrows():
            field = row['field_name']
            ecode_type = row['ecode_type']
            
            if field not in X.columns:
                continue
            
            if ecode_type == 'one-hot':
                self.encoders_[field] = OneHotEncoder(
                    handle_unknown='ignore',
                    min_frequency=2,
                    max_categories=100,
                    sparse_output=False,
                )
                self.encoders_[field].fit(X[[field]])
            else:
                self.encoders_[field] = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                )
                self.encoders_[field].fit(X[[field]])

    def save(self, path: str) -> None:
        """Save preprocessor state to joblib file."""
        import joblib
        joblib.dump({
            'df_field': self.df_field,
            'fill_values_': self.fill_values_,
            'outlier_bounds_': self.outlier_bounds_,
            'scalers_': self.scalers_,
            'encoders_': self.encoders_,
            'feature_names_in_': self.feature_names_in_,
            'enable_missing': self.enable_missing,
            'enable_outlier': self.enable_outlier,
            'enable_encoding': self.enable_encoding,
            'enable_standardization': self.enable_standardization,
        }, path)

    @classmethod
    def load(cls, path: str) -> "ConfigDrivenPreprocessor":
        """Load preprocessor state from joblib file."""
        import joblib
        data = joblib.load(path)
        instance = cls(
            df_field=data['df_field'],
            enable_missing=data['enable_missing'],
            enable_outlier=data['enable_outlier'],
            enable_encoding=data['enable_encoding'],
            enable_standardization=data['enable_standardization'],
        )
        instance.fill_values_ = data['fill_values_']
        instance.outlier_bounds_ = data['outlier_bounds_']
        instance.scalers_ = data['scalers_']
        instance.encoders_ = data['encoders_']
        instance.feature_names_in_ = data['feature_names_in_']
        return instance


# ============================================================================
# 数据文件读取与字段配置推断
# ============================================================================
# 这几个函数刻意不依赖 streamlit：无头的 FastAPI / 训练流水线也要能读文件、推断
# 字段类型。web_app.utils.file_handler 从此处再导出，Streamlit 侧导入路径不变。

def read_file_from_path(file_path: str) -> Optional[pd.DataFrame]:
    """
    从路径读取文件

    Args:
        file_path: 文件路径

    Returns:
        DataFrame 或 None
    """
    path = Path(file_path)

    if not path.exists():
        return None

    try:
        file_extension = path.suffix.lower()

        if file_extension == '.csv':
            for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
                try:
                    return pd.read_csv(path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(path, engine='openpyxl')
        else:
            return None

    except Exception as e:
        logger.error(f"读取文件失败：{str(e)}")
        return None


def infer_field_type(series: pd.Series) -> str:
    """
    根据数据内容推断字段类型

    Args:
        series: 数据列

    Returns:
        推断的类型 ('num', 'str', 'key', 'label')
    """
    # 字段名和唯一率优先识别 ID；纯数字 ID 不能先被误判成普通数值特征。
    name = str(series.name or '').strip().lower()
    unique_ratio = series.nunique(dropna=True) / max(int(series.notna().sum()), 1)
    if unique_ratio > 0.9 and any(token in name for token in ('id', 'uuid', 'key', '编号', '序号')):
        return "key"

    # 检查是否全为数值
    if pd.api.types.is_numeric_dtype(series):
        return "num"

    # 尝试转换为数值
    try:
        converted = pd.to_numeric(series, errors='coerce')
        if converted.notna().mean() > 0.9:  # 90% 以上能转换
            return "num"
    except:
        pass

    # 检查是否为 ID 类（唯一值比例高）
    if unique_ratio > 0.9:
        return "key"

    # 默认为字符串
    return "str"


def auto_generate_field_config(
    df: pd.DataFrame,
    label_col: str,
    key_col: Optional[str] = None
) -> pd.DataFrame:
    """
    根据 DataFrame 自动生成字段配置

    使用 infer_field_type 自动推断字段类型，并设置合理的默认处理规则。

    Args:
        df: 输入 DataFrame
        label_col: 目标变量列名
        key_col: 可选，ID/键列名

    Returns:
        配置 DataFrame，包含 field_name, field_type, outlier_value,
        missing_value, ecode_type, standardize_type 列
    """
    config_data = []

    for col in df.columns:
        if col == label_col:
            field_type = "label"
        elif col == key_col:
            field_type = "key"
        else:
            field_type = infer_field_type(df[col])

        # 缺失值处理默认规则
        missing_value = np.nan
        if df[col].isna().sum() > 0:
            if field_type == 'num':
                missing_value = 'mean'
            else:
                missing_value = 'mode'

        config_data.append({
            "field_name": col,
            "field_type": field_type,
            "outlier_value": np.nan,
            "missing_value": missing_value,
            "ecode_type": "one-hot" if field_type == "str" else np.nan,
            "standardize_type": "z-score" if field_type == "num" else np.nan,
        })

    return pd.DataFrame(config_data)
