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

if __name__ == '__main__':
    # 生成更大的数据集以提高找到合适样本的概率
    neg_data_df = pd.DataFrame({
        'A': np.random.randn(100000),
        'B': np.random.uniform(0, 1, 100000),
        'C': np.random.randint(0, 100, 100000),
        'D': np.random.randn(100000),
        'E': np.random.exponential(1, 100000),
        'G': np.random.lognormal(0, 1, 100000),
        'H': np.random.lognormal(0, 1, 100000),
        'I': np.random.lognormal(0, 1, 100000),
        'J': np.random.lognormal(0, 1, 100000)
    })
    print(neg_data_df.head())
    # 计算整个数据集的描述性统计
    describe = neg_data_df.describe().T
    describe_values = describe[["mean", "std"]].values

    z_num = 300
    weight = 900
    error_rate = 0.1

    # 获取满足条件的样本索引
    index_result = neg_data_undersampling(neg_data_df, describe_values, z_num, weight, error_rate)

    if index_result is not None:
        # 使用满足条件的索引创建新DataFrame
        df_new = neg_data_df.loc[index_result]
        print(f"New DataFrame shape: {df_new.shape}")
        print("New DataFrame description:")
        print(df_new.describe())
    else:
        print("No suitable sample found")

    # # 创建示例数据
    # np.random.seed(42)
    # example_df = pd.DataFrame({
    #     '数值列1': np.random.normal(0, 1, 100),
    #     '数值列2': np.random.randint(1, 10, 100),
    #     '类别列': np.random.choice(['A', 'B', 'C'], 100),
    #     '含空值列': [None if i % 10 == 0 else f'value_{i}' for i in range(100)],
    #     '含空值数值列': [np.nan if i % 5 == 0 else float(i) for i in range(100)]
    # })
    #
    # # 添加一些空值
    # example_df.loc[10:15, '数值列1'] = np.nan
    #
    # print("原始DataFrame:")
    # print(example_df.head())
    # print("\n数据概况:")
    # print(example_df.info())
    #
    # # 进行分析
    # analysis_result = analyze_dataframe(example_df)
    #
    # print("\n统计分析结果:")
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # print(analysis_result)