import pandas as pd
import numpy as np
from scipy import stats


def analyze_dataframe(df):
    """
    分析DataFrame的统计特征

    参数:
    df -- 输入的pandas DataFrame

    返回:
    result_df -- 包含各字段统计信息的DataFrame
    """

    # 初始化结果字典
    results = {
        '字段名': [],
        '空值率': [],
        '空值数量': [],
        '非空值数量': [],
        '数据类型': [],
        '唯一值数量': [],
        '唯一值比例': [],
        '方差': [],
        '均值': [],
        '中位数': [],
        '众数': [],
        '众数出现次数': [],
        '标准差': [],
        '变异系数': [],
        '最小值': [],
        '最大值': [],
        '范围': [],
        '偏度': [],
        '峰度': []
    }

    for column in df.columns:
        # 基础信息
        results['字段名'].append(column)
        total_count = len(df)
        null_count = df[column].isnull().sum()
        non_null_count = total_count - null_count

        results['空值数量'].append(null_count)
        results['非空值数量'].append(non_null_count)
        results['空值率'].append(null_count / total_count if total_count > 0 else 0)
        results['数据类型'].append(str(df[column].dtype))

        # 唯一值统计
        unique_count = df[column].nunique()
        results['唯一值数量'].append(unique_count)
        results['唯一值比例'].append(unique_count / non_null_count if non_null_count > 0 else 0)

        # 数值型统计（仅对数值型数据计算）
        if pd.api.types.is_numeric_dtype(df[column]):
            # 移除空值后的数据
            clean_data = df[column].dropna()

            if len(clean_data) > 0:
                # 基本统计量
                results['方差'].append(clean_data.var())
                results['均值'].append(clean_data.mean())
                results['中位数'].append(clean_data.median())
                results['标准差'].append(clean_data.std())

                # 变异系数（标准差/均值）
                mean_val = clean_data.mean()
                std_val = clean_data.std()
                results['变异系数'].append(std_val / mean_val if mean_val != 0 else np.inf)

                # 范围统计
                results['最小值'].append(clean_data.min())
                results['最大值'].append(clean_data.max())
                results['范围'].append(clean_data.max() - clean_data.min())

                # 高阶统计量
                results['偏度'].append(clean_data.skew())
                results['峰度'].append(clean_data.kurtosis())

                # 众数（可能有多个，取第一个）
                mode_result = clean_data.mode()
                if len(mode_result) > 0:
                    results['众数'].append(mode_result.iloc[0])
                    # 计算众数出现次数
                    mode_count = (clean_data == mode_result.iloc[0]).sum()
                    results['众数出现次数'].append(mode_count)
                else:
                    results['众数'].append(np.nan)
                    results['众数出现次数'].append(0)
            else:
                # 如果所有值都是空值，填充NaN
                numeric_fields = ['方差', '均值', '中位数', '标准差', '变异系数',
                                  '最小值', '最大值', '范围', '偏度', '峰度', '众数']
                for field in numeric_fields:
                    results[field].append(np.nan)
                results['众数出现次数'].append(0)
        else:
            # 非数值型数据的处理
            clean_data = df[column].dropna()

            if len(clean_data) > 0:
                # 对于非数值型数据，只能计算众数
                mode_result = clean_data.mode()
                if len(mode_result) > 0:
                    results['众数'].append(mode_result.iloc[0])
                    mode_count = (clean_data == mode_result.iloc[0]).sum()
                    results['众数出现次数'].append(mode_count)
                else:
                    results['众数'].append(np.nan)
                    results['众数出现次数'].append(0)
            else:
                results['众数'].append(np.nan)
                results['众数出现次数'].append(0)

            # 非数值型数据的数值统计量设为NaN
            numeric_fields = ['方差', '均值', '中位数', '标准差', '变异系数',
                              '最小值', '最大值', '范围', '偏度', '峰度']
            for field in numeric_fields:
                results[field].append(np.nan)

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    # 设置显示格式
    float_columns = ['空值率', '唯一值比例', '方差', '均值', '中位数', '标准差',
                     '变异系数', '偏度', '峰度']
    for col in float_columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].round(4)

    return result_df


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    example_df = pd.DataFrame({
        '数值列1': np.random.normal(0, 1, 100),
        '数值列2': np.random.randint(1, 10, 100),
        '类别列': np.random.choice(['A', 'B', 'C'], 100),
        '含空值列': [None if i % 10 == 0 else f'value_{i}' for i in range(100)],
        '含空值数值列': [np.nan if i % 5 == 0 else float(i) for i in range(100)]
    })

    # 添加一些空值
    example_df.loc[10:15, '数值列1'] = np.nan

    print("原始DataFrame:")
    print(example_df.head())
    print("\n数据概况:")
    print(example_df.info())

    # 进行分析
    analysis_result = analyze_dataframe(example_df)

    print("\n统计分析结果:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(analysis_result)