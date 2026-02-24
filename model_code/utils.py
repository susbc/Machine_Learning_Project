import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def smart_sampling(df, label_col='label', positive_label=1, negative_ratio=0.1, random_state=42):
    """
    智能抽样函数

    参数:
    - df: 原始数据框
    - label_col: 标签列名
    - positive_label: 正样本标签值
    - negative_ratio: 负样本抽样比例
    - random_state: 随机种子
    """
    np.random.seed(random_state)

    # 分离样本
    positive_mask = df[label_col] == positive_label
    positive_df = df[positive_mask]
    negative_df = df[~positive_mask]

    print(f"原始数据: {len(df)} 条记录")
    print(f"正样本({positive_label}): {len(positive_df)} 条")
    print(f"负样本: {len(negative_df)} 条")

    # 负样本抽样
    negative_sampled = negative_df.sample(frac=negative_ratio, random_state=random_state)

    # 合并
    df_sampled = pd.concat([positive_df, negative_sampled], ignore_index=True)

    # 打乱顺序（可选）
    df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\n抽样后数据: {len(df_sampled)} 条记录")
    print(f"正样本: {len(df_sampled[df_sampled[label_col] == positive_label])} 条")
    print(f"负样本: {len(df_sampled[df_sampled[label_col] != positive_label])} 条")
    print(f"负样本抽样比例: {negative_ratio:.0%}")

    return df_sampled


def neg_data_undersampling(neg_df, target_stats, step_size, max_iter, error_rate, max_samples=50000):
    """
    通过迭代寻找符合统计特征的最优负样本子集

    :param neg_df: 原始负样本 DataFrame
    :param target_stats: 目标特征矩阵 (n_features, 2)，包含期望的 [mean, std]
    :param step_size: 每次迭代增加的样本数 (z_num)
    :param max_iter: 最大迭代次数 (weight)
    :param error_rate: 允许的误差比例 (如 0.05)
    :param max_samples: 采样上限
    """
    allowed_error = np.abs(target_stats * error_rate)
    best_match_index = None
    min_overall_error = float('inf')

    for i in range(1, max_iter + 1):
        n = i * step_size

        if n > max_samples or n > len(neg_df):
            break

        # 随机采样
        sample_subset = neg_df.sample(n=n, random_state=i)  # 改变随机种子以遍历不同组合

        # 计算当前子集的统计量
        current_stats = sample_subset.describe().T[["mean", "std"]].values

        # 计算绝对误差
        diff = np.abs(current_stats - target_stats)

        # 记录史上最接近的样本（以防循环结束也没找到完全符合要求的）
        total_error = np.sum(diff)
        if total_error < min_overall_error:
            min_overall_error = total_error
            best_match_index = sample_subset.index

        # 检查是否满足停止条件 (所有维度的误差都在允许范围内)
        if np.all(diff <= allowed_error):
            print(f"✅ 在步长 {i} 找到理想样本，规模: {n}")
            return sample_subset.index

    print(f"⚠️ 未找到完全匹配样本。返回误差最小的样本子集，累计误差: {min_overall_error:.4f}")
    return best_match_index

# 如果担心抽样后负样本太少，可以设置最小样本量
def sampling_with_min_samples(df, negative_ratio=0.1, min_negative_samples=10):
    positive_samples = df[df['label'] == 1]
    negative_samples = df[df['label'] == 0]

    # 计算抽样数量
    sample_size = max(int(len(negative_samples) * negative_ratio), min_negative_samples)

    # 如果计算的数量超过实际数量，则取全部
    sample_size = min(sample_size, len(negative_samples))

    negative_sampled = negative_samples.sample(n=sample_size, random_state=42)

    df_sampled = pd.concat([positive_samples, negative_sampled], ignore_index=True)

    print(f"负样本抽样: {sample_size} 条 (确保至少 {min_negative_samples} 条)")

    return df_sampled



def convert_dataframe_types_strict(data, df_field):
    """
    更严格的数值类型转换版本
    """
    result_data = data.copy()

    for _, row in df_field.iterrows():
        field_name = row['field_name']
        field_type = row['field_type']

        if field_name not in result_data.columns:
            continue

        if field_type == 'num':
            # 先转换为字符串，然后移除可能的空格和逗号
            result_data[field_name] = result_data[field_name].astype(str).str.replace(',', '').str.strip()

            # 尝试转换
            original_non_null = result_data[field_name].notna().sum()
            result_data[field_name] = pd.to_numeric(result_data[field_name], errors='coerce')
            converted_non_null = result_data[field_name].notna().sum()

            if converted_non_null < original_non_null:
                print(f"警告: 字段 '{field_name}' 有 {original_non_null - converted_non_null} 个值无法转换为数值")

    return result_data

def handle_dataframe_outliers(data, df_field):
    """
    根据字段定义对DataFrame进行异常值处理

    参数:
    data: pandas DataFrame, 原始数据
    df_field: pandas DataFrame, 包含字段名和异常值处理方式，
              必须有'field_name'和'outlier_value'列

    返回:
    pandas DataFrame: 异常值处理后的数据
    """
    # 创建数据的副本，避免修改原始数据
    result_data = data.copy()

    # 遍历字段定义表
    for _, row in df_field.iterrows():
        field_name = row['field_name']
        outlier_value = row['outlier_value']

        # 检查字段是否存在于数据中
        if field_name not in result_data.columns:
            print(f"警告: 字段 '{field_name}' 在数据中不存在，跳过处理")
            continue

        # 跳过空值处理方式
        if pd.isna(outlier_value):
            continue

        # 确保字段是数值类型
        if not np.issubdtype(result_data[field_name].dtype, np.number):
            try:
                result_data[field_name] = pd.to_numeric(result_data[field_name], errors='coerce')
                print(f"信息: 字段 '{field_name}' 已转换为数值类型")
            except Exception as e:
                print(f"错误: 字段 '{field_name}' 无法转换为数值类型: {str(e)}")
                continue

        # 处理不同的异常值检测方法
        if outlier_value == '3std':
            result_data = _handle_std_outliers(result_data, field_name, 3)
        elif outlier_value == '5std':
            result_data = _handle_std_outliers(result_data, field_name, 5)
        elif isinstance(outlier_value, str) and outlier_value.startswith('(') and outlier_value.endswith(')'):
            result_data = _handle_range_outliers(result_data, field_name, outlier_value)
        else:
            print(f"警告: 字段 '{field_name}' 的异常值处理方式 '{outlier_value}' 不被支持，跳过处理")

    return result_data


def _handle_std_outliers(data, field_name, std_multiplier):
    """
    处理基于标准差的异常值
    """
    # 计算均值和标准差
    mean_val = data[field_name].mean()
    std_val = data[field_name].std()

    # 计算边界
    lower_bound = mean_val - std_multiplier * std_val
    upper_bound = mean_val + std_multiplier * std_val

    # 记录异常值数量
    lower_outliers = (data[field_name] < lower_bound).sum()
    upper_outliers = (data[field_name] > upper_bound).sum()

    # 替换异常值
    data[field_name] = np.where(data[field_name] < lower_bound, lower_bound, data[field_name])
    data[field_name] = np.where(data[field_name] > upper_bound, upper_bound, data[field_name])

    print(f"信息: 字段 '{field_name}' 使用 {std_multiplier}σ 方法处理异常值，"
          f"边界 [{lower_bound:.4f}, {upper_bound:.4f}]，"
          f"处理了 {lower_outliers} 个低于下限和 {upper_outliers} 个高于上限的异常值")

    return data


def _handle_range_outliers(data, field_name, range_str):
    """
    处理基于固定范围的异常值
    """
    try:
        # 解析范围字符串，例如 "(0, 100)"
        range_str = range_str.strip('()')
        parts = range_str.split(',')

        if len(parts) != 2:
            raise ValueError("范围格式不正确")

        min_val = float(parts[0].strip())
        max_val = float(parts[1].strip())

        if min_val >= max_val:
            raise ValueError("最小值必须小于最大值")

        # 记录异常值数量
        lower_outliers = (data[field_name] < min_val).sum()
        upper_outliers = (data[field_name] > max_val).sum()

        # 替换异常值
        data[field_name] = np.where(data[field_name] < min_val, min_val, data[field_name])
        data[field_name] = np.where(data[field_name] > max_val, max_val, data[field_name])

        print(f"信息: 字段 '{field_name}' 使用范围 [{min_val}, {max_val}] 处理异常值，"
              f"处理了 {lower_outliers} 个低于下限和 {upper_outliers} 个高于上限的异常值")

    except Exception as e:
        print(f"错误: 解析字段 '{field_name}' 的范围 '{range_str}' 时出错: {str(e)}")

    return data


# 可选：添加更灵活的范围解析
def parse_range_string(range_str):
    """
    更灵活的范围字符串解析，支持多种格式
    """
    range_str = range_str.strip()

    # 移除括号
    if range_str.startswith('(') and range_str.endswith(')'):
        range_str = range_str[1:-1]
    elif range_str.startswith('[') and range_str.endswith(']'):
        range_str = range_str[1:-1]

    # 分割最小值和最大值
    parts = [part.strip() for part in range_str.split(',')]

    if len(parts) != 2:
        raise ValueError("范围格式不正确，应为 'min, max'")

    return float(parts[0]), float(parts[1])

# 在 _handle_range_outliers 函数中使用这个更灵活的解析器


def handle_dataframe_missing_values(data, df_field):
    """
    根据字段定义对DataFrame进行缺失值处理

    参数:
    data: pandas DataFrame, 原始数据
    df_field: pandas DataFrame, 包含字段名和缺失值处理方式，
              必须有'field_name'和'missing_value'列

    返回:
    pandas DataFrame: 缺失值处理后的数据
    """
    # 创建数据的副本，避免修改原始数据
    result_data = data.copy()

    # 遍历字段定义表
    for _, row in df_field.iterrows():
        field_name = row['field_name']
        missing_value = row['missing_value']

        # 检查字段是否存在于数据中
        if field_name not in result_data.columns:
            print(f"警告: 字段 '{field_name}' 在数据中不存在，跳过处理")
            continue

        # 跳过空值处理方式
        if pd.isna(missing_value):
            continue

        # 检查字段中是否有缺失值
        missing_count = result_data[field_name].isna().sum()
        if missing_count == 0:
            print(f"信息: 字段 '{field_name}' 没有缺失值，跳过处理")
            continue

        print(f"信息: 字段 '{field_name}' 有 {missing_count} 个缺失值，使用 '{missing_value}' 方法填充")

        # 处理不同的缺失值填充方法
        if missing_value == 'mean':
            result_data = _fill_with_mean(result_data, field_name)
        elif missing_value == 'median':
            result_data = _fill_with_median(result_data, field_name)
        elif missing_value == 'mode':
            result_data = _fill_with_mode(result_data, field_name)
        else:
            # 尝试作为固定值处理
            result_data = _fill_with_value(result_data, field_name, missing_value)

    return result_data


def _fill_with_mean(data, field_name):
    """
    使用均值填充缺失值
    """
    try:
        # 确保字段是数值类型
        if not np.issubdtype(data[field_name].dtype, np.number):
            data[field_name] = pd.to_numeric(data[field_name], errors='coerce')
            print(f"信息: 字段 '{field_name}' 已转换为数值类型以计算均值")

        mean_val = data[field_name].mean()
        data[field_name].fillna(mean_val, inplace=True)
        print(f"信息: 字段 '{field_name}' 使用均值 {mean_val:.4f} 填充缺失值")

    except Exception as e:
        print(f"错误: 计算字段 '{field_name}' 的均值时出错: {str(e)}")

    return data


def _fill_with_median(data, field_name):
    """
    使用中位数填充缺失值
    """
    try:
        # 确保字段是数值类型
        if not np.issubdtype(data[field_name].dtype, np.number):
            data[field_name] = pd.to_numeric(data[field_name], errors='coerce')
            print(f"信息: 字段 '{field_name}' 已转换为数值类型以计算中位数")

        median_val = data[field_name].median()
        data[field_name].fillna(median_val, inplace=True)
        print(f"信息: 字段 '{field_name}' 使用中位数 {median_val:.4f} 填充缺失值")

    except Exception as e:
        print(f"错误: 计算字段 '{field_name}' 的中位数时出错: {str(e)}")

    return data


def _fill_with_mode(data, field_name):
    """
    使用众数填充缺失值
    """
    try:
        # 计算众数（可能有多个，取第一个）
        mode_vals = data[field_name].mode()
        if len(mode_vals) > 0:
            mode_val = mode_vals[0]
            data[field_name].fillna(mode_val, inplace=True)
            print(f"信息: 字段 '{field_name}' 使用众数 '{mode_val}' 填充缺失值")
        else:
            print(f"警告: 字段 '{field_name}' 无法计算众数，跳过填充")

    except Exception as e:
        print(f"错误: 计算字段 '{field_name}' 的众数时出错: {str(e)}")

    return data


def _fill_with_value(data, field_name, value):
    """
    使用指定值填充缺失值
    """
    try:
        # 尝试将值转换为与字段相同的数据类型
        if np.issubdtype(data[field_name].dtype, np.number):
            # 如果是数值类型，尝试转换为数值
            try:
                value = float(value)
            except ValueError:
                print(f"警告: 字段 '{field_name}' 是数值类型，但填充值 '{value}' 无法转换为数值")

        data[field_name].fillna(value, inplace=True)
        print(f"信息: 字段 '{field_name}' 使用值 '{value}' 填充缺失值")

    except Exception as e:
        print(f"错误: 使用值 '{value}' 填充字段 '{field_name}' 时出错: {str(e)}")

    return data


# 可选：添加分组填充功能
def _fill_with_group_mean(data, field_name, group_by):
    """
    使用分组均值填充缺失值
    """
    try:
        data[field_name] = data.groupby(group_by)[field_name].transform(
            lambda x: x.fillna(x.mean())
        )
        print(f"信息: 字段 '{field_name}' 使用分组均值填充缺失值")
    except Exception as e:
        print(f"错误: 使用分组均值填充字段 '{field_name}' 时出错: {str(e)}")

    return data


# 可选：添加前向/后向填充
def _fill_with_forward(data, field_name):
    """
    使用前向填充
    """
    try:
        data[field_name].fillna(method='ffill', inplace=True)
        print(f"信息: 字段 '{field_name}' 使用前向填充缺失值")
    except Exception as e:
        print(f"错误: 使用前向填充字段 '{field_name}' 时出错: {str(e)}")

    return data


# 在主要函数中添加这些选项
def handle_dataframe_missing_values_advanced(data, df_field, group_by_fields=None):
    """
    高级版本的缺失值处理，支持更多填充方法
    """
    result_data = data.copy()

    for _, row in df_field.iterrows():
        field_name = row['field_name']
        missing_value = row['missing_value']

        if field_name not in result_data.columns or pd.isna(missing_value):
            continue

        missing_count = result_data[field_name].isna().sum()
        if missing_count == 0:
            continue

        # 扩展的填充方法
        if missing_value == 'group_mean' and group_by_fields:
            result_data = _fill_with_group_mean(result_data, field_name, group_by_fields)
        elif missing_value == 'forward':
            result_data = _fill_with_forward(result_data, field_name)
        # ... 其他高级方法

    return result_data


def encode_dataframe(data, df_field):
    """
    根据字段定义对DataFrame进行编码处理

    参数:
    data: pandas DataFrame, 原始数据
    df_field: pandas DataFrame, 包含字段名、类型和编码方式，
              必须有'field_name'、'field_type'和'ecode_type'列

    返回:
    pandas DataFrame: 编码处理后的数据
    """
    # 创建数据的副本，避免修改原始数据
    result_data = data.copy()

    # 遍历字段定义表
    for _, row in df_field.iterrows():
        field_name = row['field_name']
        field_type = row['field_type']
        ecode_type = row['ecode_type']

        # 检查字段是否存在于数据中
        if field_name not in result_data.columns:
            print(f"警告: 字段 '{field_name}' 在数据中不存在，跳过处理")
            continue

        # 只处理字符串类型字段
        if field_type != 'str':
            continue

        # 检查字段是否包含字符串数据
        if result_data[field_name].dtype != 'object' and not pd.api.types.is_string_dtype(result_data[field_name]):
            print(f"警告: 字段 '{field_name}' 不是字符串类型，尝试转换为字符串")
            result_data[field_name] = result_data[field_name].astype(str)

        # 处理不同的编码方式
        if ecode_type == 'one-hot':
            result_data = _apply_one_hot_encoding(result_data, field_name)
        elif pd.isna(ecode_type):
            result_data = _apply_ordinal_encoding(result_data, field_name)
        else:
            print(f"警告: 字段 '{field_name}' 的编码方式 '{ecode_type}' 不被支持，跳过处理")

    return result_data


def _apply_one_hot_encoding(data, field_name):
    """
    应用独热编码
    """
    try:
        # 获取唯一值的数量
        unique_count = data[field_name].nunique()

        # 如果唯一值太多，发出警告
        if unique_count > 50:
            print(f"警告: 字段 '{field_name}' 有 {unique_count} 个唯一值，独热编码可能产生大量列")

        # 应用独热编码
        one_hot_encoded = pd.get_dummies(data[field_name], prefix=field_name)

        # 将编码后的列添加到原始数据中
        data = pd.concat([data, one_hot_encoded], axis=1)

        # 删除原始列
        data.drop(field_name, axis=1, inplace=True)

        print(f"信息: 字段 '{field_name}' 已进行独热编码，生成了 {unique_count} 个新列")

    except Exception as e:
        print(f"错误: 对字段 '{field_name}' 进行独热编码时出错: {str(e)}")

    return data


def _apply_ordinal_encoding(data, field_name):
    """
    应用顺序编码
    """
    try:
        # 创建编码器
        encoder = OrdinalEncoder()

        # 应用编码
        encoded_values = encoder.fit_transform(data[[field_name]])

        # 替换原始列
        data[field_name] = encoded_values

        # 获取类别映射
        categories = encoder.categories_[0]
        category_mapping = {i: category for i, category in enumerate(categories)}

        print(f"信息: 字段 '{field_name}' 已进行顺序编码，映射关系: {category_mapping}")

    except Exception as e:
        print(f"错误: 对字段 '{field_name}' 进行顺序编码时出错: {str(e)}")
        # 如果使用sklearn失败，回退到pandas的factorize方法
        try:
            data[field_name], unique_categories = pd.factorize(data[field_name])
            category_mapping = {i: category for i, category in enumerate(unique_categories)}
            print(f"信息: 字段 '{field_name}' 已使用备用方法进行顺序编码，映射关系: {category_mapping}")
        except Exception as e2:
            print(f"错误: 备用编码方法也失败: {str(e2)}")

    return data


# 可选：添加自定义顺序编码
def _apply_custom_ordinal_encoding(data, field_name, custom_order):
    """
    应用自定义顺序编码
    """
    try:
        # 创建映射字典
        mapping = {category: i for i, category in enumerate(custom_order)}

        # 应用映射
        data[field_name] = data[field_name].map(mapping)

        # 处理未在自定义顺序中出现的类别
        if data[field_name].isna().any():
            print(f"警告: 字段 '{field_name}' 包含未在自定义顺序中定义的类别")
            # 使用-1填充未定义的类别
            data[field_name].fillna(-1, inplace=True)

        print(f"信息: 字段 '{field_name}' 已进行自定义顺序编码，顺序: {custom_order}")

    except Exception as e:
        print(f"错误: 对字段 '{field_name}' 进行自定义顺序编码时出错: {str(e)}")

    return data


# 可选：添加频率编码
def _apply_frequency_encoding(data, field_name):
    """
    应用频率编码（按类别出现频率编码）
    """
    try:
        # 计算每个类别的频率
        frequency_map = data[field_name].value_counts().to_dict()

        # 应用频率编码
        data[field_name] = data[field_name].map(frequency_map)

        print(f"信息: 字段 '{field_name}' 已进行频率编码")

    except Exception as e:
        print(f"错误: 对字段 '{field_name}' 进行频率编码时出错: {str(e)}")

    return data


# 在主要函数中添加这些选项
def encode_dataframe_advanced(data, df_field, custom_orders=None):
    """
    高级版本的编码处理，支持更多编码方法
    """
    result_data = data.copy()

    # 初始化自定义顺序字典
    if custom_orders is None:
        custom_orders = {}

    for _, row in df_field.iterrows():
        field_name = row['field_name']
        field_type = row['field_type']
        ecode_type = row['ecode_type']

        if field_name not in result_data.columns or field_type != 'str':
            continue

        # 扩展的编码方法
        if ecode_type == 'custom-ordinal' and field_name in custom_orders:
            result_data = _apply_custom_ordinal_encoding(result_data, field_name, custom_orders[field_name])
        elif ecode_type == 'frequency':
            result_data = _apply_frequency_encoding(result_data, field_name)
        # ... 其他编码方法

    return result_data



def standardize_dataframe(data, df_field):
    """
    根据字段定义对DataFrame进行标准化处理

    参数:
    data: pandas DataFrame, 原始数据
    df_field: pandas DataFrame, 包含字段名和标准化方式，
              必须有'field_name'和'standardize_type'列

    返回:
    pandas DataFrame: 标准化处理后的数据
    """
    # 创建数据的副本，避免修改原始数据
    result_data = data.copy()

    # 遍历字段定义表
    for _, row in df_field.iterrows():
        field_name = row['field_name']
        standardize_type = row['standardize_type']

        # 检查字段是否存在于数据中
        if field_name not in result_data.columns:
            print(f"警告: 字段 '{field_name}' 在数据中不存在，跳过处理")
            continue

        # 跳过空值处理方式
        if pd.isna(standardize_type):
            continue

        # 确保字段是数值类型
        if not np.issubdtype(result_data[field_name].dtype, np.number):
            try:
                result_data[field_name] = pd.to_numeric(result_data[field_name], errors='coerce')
                print(f"信息: 字段 '{field_name}' 已转换为数值类型")
            except Exception as e:
                print(f"错误: 字段 '{field_name}' 无法转换为数值类型: {str(e)}")
                continue

        # 处理不同的标准化方法
        if standardize_type == 'z-score':
            result_data = _apply_z_score_standardization(result_data, field_name)
        elif standardize_type == 'min-max':
            result_data = _apply_min_max_standardization(result_data, field_name)
        elif standardize_type == 'robust':
            result_data = _apply_robust_standardization(result_data, field_name)
        else:
            print(f"警告: 字段 '{field_name}' 的标准化方式 '{standardize_type}' 不被支持，跳过处理")

    return result_data


def _apply_z_score_standardization(data, field_name):
    """
    应用Z-score标准化（均值0，标准差1）
    """
    try:
        # 创建标准化器
        scaler = StandardScaler()

        # 应用标准化
        data[field_name] = scaler.fit_transform(data[[field_name]])

        print(f"信息: 字段 '{field_name}' 已进行Z-score标准化，均值≈0，标准差≈1")

    except Exception as e:
        print(f"错误: 对字段 '{field_name}' 进行Z-score标准化时出错: {str(e)}")

    return data


def _apply_min_max_standardization(data, field_name):
    """
    应用最小-最大标准化（将值缩放到[0,1]范围）
    """
    try:
        # 创建标准化器
        scaler = MinMaxScaler()

        # 应用标准化
        data[field_name] = scaler.fit_transform(data[[field_name]])

        print(f"信息: 字段 '{field_name}' 已进行最小-最大标准化，值范围[0,1]")

    except Exception as e:
        print(f"错误: 对字段 '{field_name}' 进行最小-最大标准化时出错: {str(e)}")

    return data


def _apply_robust_standardization(data, field_name):
    """
    应用鲁棒标准化（基于中位数和四分位数，对异常值不敏感）
    """
    try:
        # 创建标准化器
        scaler = RobustScaler()

        # 应用标准化
        data[field_name] = scaler.fit_transform(data[[field_name]])

        print(f"信息: 字段 '{field_name}' 已进行鲁棒标准化")

    except Exception as e:
        print(f"错误: 对字段 '{field_name}' 进行鲁棒标准化时出错: {str(e)}")

    return data


# 可选：添加自定义标准化方法
def _apply_custom_standardization(data, field_name, method='z-score'):
    """
    应用自定义标准化方法（不使用sklearn）
    """
    try:
        if method == 'z-score':
            # 手动计算Z-score
            mean_val = data[field_name].mean()
            std_val = data[field_name].std()
            data[field_name] = (data[field_name] - mean_val) / std_val
            print(f"信息: 字段 '{field_name}' 已进行手动Z-score标准化")

        elif method == 'min-max':
            # 手动计算最小-最大标准化
            min_val = data[field_name].min()
            max_val = data[field_name].max()
            data[field_name] = (data[field_name] - min_val) / (max_val - min_val)
            print(f"信息: 字段 '{field_name}' 已进行手动最小-最大标准化")

        elif method == 'decimal':
            # 小数定标标准化
            max_abs_val = data[field_name].abs().max()
            j = len(str(int(max_abs_val)))
            data[field_name] = data[field_name] / (10 ** j)
            print(f"信息: 字段 '{field_name}' 已进行小数定标标准化")

    except Exception as e:
        print(f"错误: 对字段 '{field_name}' 进行自定义标准化时出错: {str(e)}")

    return data


# 可选：添加保存和加载标准化器功能
def standardize_dataframe_with_save(data, df_field, save_scalers=False, scalers_path=None):
    """
    高级版本的标准化处理，可以保存和加载标准化器
    """
    result_data = data.copy()
    scalers = {}

    for _, row in df_field.iterrows():
        field_name = row['field_name']
        standardize_type = row['standardize_type']

        if field_name not in result_data.columns or pd.isna(standardize_type):
            continue

        # 确保字段是数值类型
        if not np.issubdtype(result_data[field_name].dtype, np.number):
            try:
                result_data[field_name] = pd.to_numeric(result_data[field_name], errors='coerce')
            except Exception as e:
                print(f"错误: 字段 '{field_name}' 无法转换为数值类型: {str(e)}")
                continue

        # 根据标准化类型创建标准化器
        if standardize_type == 'z-score':
            scaler = StandardScaler()
        elif standardize_type == 'min-max':
            scaler = MinMaxScaler()
        elif standardize_type == 'robust':
            scaler = RobustScaler()
        else:
            continue

        # 应用标准化
        result_data[field_name] = scaler.fit_transform(result_data[[field_name]]).flatten()
        scalers[field_name] = scaler

        print(f"信息: 字段 '{field_name}' 已进行{standardize_type}标准化")

    # 保存标准化器（如果需要）
    if save_scalers and scalers_path:
        import joblib
        joblib.dump(scalers, scalers_path)
        print(f"信息: 标准化器已保存到 {scalers_path}")

    return result_data, scalers


# 可选：添加逆标准化功能
def inverse_standardize_dataframe(data, scalers):
    """
    对标准化后的数据进行逆标准化
    """
    result_data = data.copy()

    for field_name, scaler in scalers.items():
        if field_name in result_data.columns:
            try:
                result_data[field_name] = scaler.inverse_transform(result_data[[field_name]]).flatten()
                print(f"信息: 字段 '{field_name}' 已进行逆标准化")
            except Exception as e:
                print(f"错误: 对字段 '{field_name}' 进行逆标准化时出错: {str(e)}")

    return result_data


