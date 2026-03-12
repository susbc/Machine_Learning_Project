"""
文件处理工具模块

提供文件上传、读取、保存等功能
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import io


# ============================================================================
# 文件读取
# ============================================================================

def read_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    读取上传的文件，支持 CSV 和 Excel 格式
    
    Args:
        uploaded_file: Streamlit UploadedFile 对象
    
    Returns:
        DataFrame 或 None
    """
    if uploaded_file is None:
        return None
    
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.csv':
            # 尝试不同的编码
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
            # 如果都失败，尝试自动检测
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding='utf-8-sig')
        
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(uploaded_file, engine='openpyxl')
        
        else:
            st.error(f"不支持的文件格式：{file_extension}")
            return None
    
    except Exception as e:
        st.error(f"读取文件失败：{str(e)}")
        return None


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
        st.error(f"读取文件失败：{str(e)}")
        return None


# ============================================================================
# 文件保存
# ============================================================================

def save_dataframe_to_csv(df: pd.DataFrame, file_path: str) -> bool:
    """
    保存 DataFrame 为 CSV 文件
    
    Args:
        df: 要保存的 DataFrame
        file_path: 保存路径
    
    Returns:
        是否成功
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        st.error(f"保存 CSV 失败：{str(e)}")
        return False


def save_dataframe_to_excel(df: pd.DataFrame, file_path: str) -> bool:
    """
    保存 DataFrame 为 Excel 文件
    
    Args:
        df: 要保存的 DataFrame
        file_path: 保存路径
    
    Returns:
        是否成功
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, index=False, engine='openpyxl')
        return True
    except Exception as e:
        st.error(f"保存 Excel 失败：{str(e)}")
        return False


def save_config_to_excel(df: pd.DataFrame, file_path: str) -> bool:
    """
    保存配置表为 Excel 文件（带格式）
    
    Args:
        df: 配置 DataFrame
        file_path: 保存路径
    
    Returns:
        是否成功
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='字段配置')
            
            # 获取 worksheet 并设置格式
            worksheet = writer.sheets['字段配置']
            
            # 设置列宽
            column_widths = {
                'A': 20,  # field_name
                'B': 15,  # field_type
                'C': 15,  # outlier_value
                'D': 15,  # missing_value
                'E': 15,  # ecode_type
                'F': 15,  # standardize_type
            }
            
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width
        
        return True
    except Exception as e:
        st.error(f"保存配置文件失败：{str(e)}")
        return False


# ============================================================================
# 数据统计分析
# ============================================================================

def calculate_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    计算数据集的统计信息
    
    Args:
        df: 输入 DataFrame
    
    Returns:
        统计信息字典
    """
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024 ** 2,  # MB
        "columns_info": [],
        "missing_summary": {},
        "unique_summary": {},
        "dtype_summary": {},
    }
    
    # 每列的详细信息
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
            "null_rate": float(df[col].isna().mean() * 100),
            "unique_count": int(df[col].nunique()),
            "unique_rate": float(df[col].nunique() / len(df) * 100),
        }
        stats["columns_info"].append(col_info)
        stats["missing_summary"][col] = col_info["null_rate"]
        stats["unique_summary"][col] = col_info["unique_count"]
    
    # 数据类型汇总
    dtype_counts = df.dtypes.value_counts()
    stats["dtype_summary"] = {str(k): int(v) for k, v in dtype_counts.items()}
    
    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    # 分类列统计
    object_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(object_cols) > 0:
        stats["category_stats"] = {}
        for col in object_cols:
            stats["category_stats"][col] = df[col].value_counts().head(10).to_dict()
    
    return stats


# ============================================================================
# 配置管理
# ============================================================================

def create_default_config(columns: list) -> pd.DataFrame:
    """
    创建默认配置表
    
    Args:
        columns: 数据列名列表
    
    Returns:
        配置 DataFrame
    """
    config_data = []
    
    for col in columns:
        config_data.append({
            "field_name": col,
            "field_type": "num",  # 默认为数值型
            "outlier_value": np.nan,  # 不处理异常值
            "missing_value": "mean",  # 默认用均值填充
            "ecode_type": np.nan,  # 不需要编码
            "standardize_type": np.nan,  # 不需要标准化
        })
    
    return pd.DataFrame(config_data)


def load_config_from_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    从文件加载配置
    
    Args:
        file_path: 配置文件路径
    
    Returns:
        配置 DataFrame 或 None
    """
    return read_file_from_path(file_path)


def infer_field_type(series: pd.Series) -> str:
    """
    根据数据内容推断字段类型
    
    Args:
        series: 数据列
    
    Returns:
        推断的类型 ('num', 'str', 'key', 'label')
    """
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
    if series.nunique() / len(series) > 0.9:
        return "key"
    
    # 默认为字符串
    return "str"


# ============================================================================
# 文件下载
# ============================================================================

def create_download_button(df: pd.DataFrame, filename: str, label: str = "下载数据") -> None:
    """
    创建数据下载按钮
    
    Args:
        df: 要下载的 DataFrame
        filename: 文件名
        label: 按钮文本
    """
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def create_excel_download_button(df: pd.DataFrame, filename: str, label: str = "下载 Excel") -> None:
    """
    创建 Excel 下载按钮
    
    Args:
        df: 要下载的 DataFrame
        filename: 文件名
        label: 按钮文本
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    
    st.download_button(
        label=label,
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
