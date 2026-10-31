"""
文件处理工具模块

提供文件上传、读取、保存等功能
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import io
import csv
import chardet

# 以下三个函数的实现已下沉到 model_code.utils（不依赖 streamlit），此处再导出
# 以保持 Streamlit 侧既有导入路径不变。
from model_code.utils import (
    auto_generate_field_config,
    infer_field_type,
    read_file_from_path,
)


# ============================================================================
# 智能文件识别
# ============================================================================

def detect_delimiter(file_content: bytes, sample_lines: int = 10) -> str:
    """
    自动检测 CSV 文件的分隔符
    
    Args:
        file_content: 文件内容（字节）
        sample_lines: 采样行数
    
    Returns:
        检测到的分隔符
    """
    # 常见分隔符
    delimiters = [',', '\t', ';', '|', ' ']
    
    try:
        # 解码文件内容
        text = file_content.decode('utf-8-sig', errors='ignore')
        lines = text.split('\n')[:sample_lines]
        
        if not lines:
            return ','
        
        # 统计每种分隔符的计数
        delimiter_counts = {}
        for delim in delimiters:
            counts = [line.count(delim) for line in lines if line.strip()]
            if counts:
                # 检查是否每行分隔符数量一致
                if len(set(counts)) == 1 and counts[0] > 0:
                    delimiter_counts[delim] = counts[0]
                elif counts:
                    # 取中位数
                    median_count = np.median(counts)
                    if median_count > 0:
                        delimiter_counts[delim] = median_count
        
        # 返回计数最多的分隔符
        if delimiter_counts:
            return max(delimiter_counts, key=delimiter_counts.get)
        
        return ','
    
    except Exception as e:
        st.warning(f"分隔符检测失败，使用默认分隔符 (,)：{str(e)}")
        return ','


def detect_encoding(file_content: bytes) -> str:
    """
    自动检测文件编码
    
    Args:
        file_content: 文件内容（字节）
    
    Returns:
        检测到的编码
    """
    try:
        # 使用 chardet 检测编码
        result = chardet.detect(file_content)
        encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0)
        
        # 如果置信度低，尝试常见编码
        if confidence < 0.7:
            common_encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
            for enc in common_encodings:
                try:
                    file_content.decode(enc)
                    return enc
                except:
                    continue
        
        return encoding if encoding else 'utf-8'
    
    except Exception as e:
        st.warning(f"编码检测失败，使用 UTF-8：{str(e)}")
        return 'utf-8'


def infer_column_names(df: pd.DataFrame, sample_rows: int = 5) -> Dict[str, str]:
    """
    推断列名（业务含义）
    
    Args:
        df: DataFrame
        sample_rows: 采样行数
    
    Returns:
        列名推断结果字典 {原列名：推断的业务名称}
    """
    inferred_names = {}
    
    # 常见列名映射（中文）
    column_mappings = {
        # 用户相关
        'user': '用户',
        'usr': '用户',
        'customer': '客户',
        'client': '客户',
        'buyer': '买家',
        
        # 订单相关
        'order': '订单',
        'ord': '订单',
        'purchase': '购买',
        
        # 金额相关
        'amount': '金额',
        'money': '金额',
        'price': '价格',
        'cost': '成本',
        'fee': '费用',
        'total': '总计',
        'sum': '合计',
        
        # 时间相关
        'time': '时间',
        'date': '日期',
        'create': '创建',
        'update': '更新',
        'start': '开始',
        'end': '结束',
        
        # 状态相关
        'status': '状态',
        'state': '状态',
        'flag': '标志',
        
        # 数量相关
        'count': '数量',
        'num': '数量',
        'quantity': '数量',
        'qty': '数量',
        
        # ID 相关
        'id': '编号',
        'code': '代码',
        'no': '编号',
        'number': '号码',
        
        # 其他
        'name': '名称',
        'type': '类型',
        'category': '类别',
        'level': '等级',
        'score': '分数',
        'rate': '比率',
        'ratio': '比例',
        'age': '年龄',
        'sex': '性别',
        'gender': '性别',
        'phone': '电话',
        'mobile': '手机',
        'email': '邮箱',
        'address': '地址',
        'city': '城市',
        'region': '地区',
        'area': '区域',
    }
    
    for col in df.columns:
        col_lower = col.lower()
        inferred_name = None
        
        # 尝试匹配映射
        for key, value in column_mappings.items():
            if key in col_lower:
                inferred_name = value
                break
        
        # 如果没有匹配，根据数据类型推断
        if not inferred_name:
            if 'id' in col_lower or 'no' in col_lower or col_lower.endswith('_id'):
                inferred_name = '编号'
            elif 'time' in col_lower or 'date' in col_lower:
                inferred_name = '时间'
            elif 'amount' in col_lower or 'price' in col_lower or 'money' in col_lower:
                inferred_name = '金额'
            elif 'name' in col_lower:
                inferred_name = '名称'
            elif 'type' in col_lower or 'category' in col_lower:
                inferred_name = '类型'
            elif 'status' in col_lower or 'state' in col_lower:
                inferred_name = '状态'
        
        # 如果还是没推断出来，使用原列名
        inferred_names[col] = inferred_name if inferred_name else col
    
    return inferred_names


def smart_read_csv(file_content: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    智能读取 CSV 文件
    
    Args:
        file_content: 文件内容（字节）
    
    Returns:
        (DataFrame, 识别信息字典)
    """
    # 检测编码
    encoding = detect_encoding(file_content)
    
    # 检测分隔符
    delimiter = detect_delimiter(file_content)
    
    # 读取数据
    try:
        df = pd.read_csv(
            io.BytesIO(file_content),
            encoding=encoding,
            delimiter=delimiter,
            skipinitialspace=True,  # 跳过空格
            on_bad_lines='skip' if pd.__version__ >= '1.3.0' else 'warn'
        )
        
        # 推断列名
        inferred_names = infer_column_names(df)
        
        # 识别信息
        info = {
            'encoding': encoding,
            'delimiter': delimiter,
            'delimiter_name': {
                ',': '逗号',
                '\t': '制表符',
                ';': '分号',
                '|': '竖线',
                ' ': '空格'
            }.get(delimiter, delimiter),
            'inferred_names': inferred_names,
            'rows': len(df),
            'columns': len(df.columns)
        }
        
        return df, info
    
    except Exception as e:
        st.error(f"读取 CSV 失败：{str(e)}")
        raise


# ============================================================================
# 文件读取
# ============================================================================

def read_uploaded_file(uploaded_file, return_info: bool = False) -> Optional[pd.DataFrame]:
    """
    读取上传的文件，支持 CSV 和 Excel 格式
    
    Args:
        uploaded_file: Streamlit UploadedFile 对象
        return_info: 是否返回识别信息
    
    Returns:
        DataFrame 或 None（return_info=True 时返回元组）
    """
    if uploaded_file is None:
        return None if not return_info else (None, {})
    
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        
        if file_extension == '.csv':
            # 使用智能读取
            df, info = smart_read_csv(file_content)
            
            if return_info:
                return df, info
            return df
        
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Excel 文件也推断列名
            inferred_names = infer_column_names(df)
            info = {
                'encoding': 'N/A (Excel)',
                'delimiter': 'N/A (Excel)',
                'delimiter_name': 'N/A',
                'inferred_names': inferred_names,
                'rows': len(df),
                'columns': len(df.columns)
            }
            
            if return_info:
                return df, info
            return df
        
        else:
            st.error(f"不支持的文件格式：{file_extension}")
            return None if not return_info else (None, {})
    
    except Exception as e:
        st.error(f"读取文件失败：{str(e)}")
        return None if not return_info else (None, {})


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
