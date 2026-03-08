#!/usr/bin/env python3
"""
测试数据生成器

用于生成机器学习测试数据集，支持:
- 生成分类数据集
- 可配置样本数、特征数
- 可配置正负样本比例
- 添加噪声/异常值
- 添加缺失值
- 导出为 CSV/Excel
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from datetime import datetime
import logging
from typing import Optional, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_data(
    n_samples: int = 10000,
    n_features: int = 20,
    n_informative: int = 15,
    n_redundant: int = 5,
    positive_ratio: float = 0.1,
    missing_rate: float = 0.05,
    outlier_rate: float = 0.02,
    noise_rate: float = 0.01,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
    include_id: bool = True,
    include_categorical: bool = True,
    n_categorical: int = 3
) -> tuple:
    """
    生成测试数据集
    
    Args:
        n_samples: 样本数量
        n_features: 特征总数
        n_informative: 信息性特征数
        n_redundant: 冗余特征数
        positive_ratio: 正样本比例
        missing_rate: 缺失值比例
        outlier_rate: 异常值比例
        noise_rate: 噪声比例
        random_state: 随机种子
        output_dir: 输出目录
        include_id: 是否包含 ID 列
        include_categorical: 是否包含分类特征
        n_categorical: 分类特征数量
    
    Returns:
        (X, y, df) 元组
    """
    logger.info(f"生成测试数据:")
    logger.info(f"  样本数：{n_samples:,}")
    logger.info(f"  特征数：{n_features}")
    logger.info(f"  正样本比例：{positive_ratio:.2%}")
    logger.info(f"  缺失值比例：{missing_rate:.2%}")
    logger.info(f"  异常值比例：{outlier_rate:.2%}")
    
    # 调整信息性和冗余特征数
    n_clusters_per_class = 2
    actual_features = n_features
    if include_categorical:
        actual_features = n_features - n_categorical
    
    # 生成基础数据
    X, y = make_classification(
        n_samples=n_samples,
        n_features=actual_features,
        n_informative=min(n_informative, actual_features),
        n_redundant=min(n_redundant, actual_features - min(n_informative, actual_features)),
        n_clusters_per_class=n_clusters_per_class,
        weights=[1 - positive_ratio, positive_ratio],
        flip_y=noise_rate,
        random_state=random_state
    )
    
    # 转换为 DataFrame
    feature_names = [f'feature_{i+1:02d}' for i in range(actual_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # 添加分类特征
    if include_categorical:
        cat_categories = ['A', 'B', 'C', 'D']
        for i in range(n_categorical):
            cat_col = f'cat_feature_{i+1}'
            df[cat_col] = np.random.choice(
                cat_categories,
                size=n_samples,
                p=[0.4, 0.3, 0.2, 0.1]
            )
    
    # 添加 ID 列
    if include_id:
        df.insert(0, 'id', range(1, n_samples + 1))
    
    # 添加目标列
    df['target'] = y
    
    # 添加缺失值
    if missing_rate > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
        
        n_missing = int(len(df) * missing_rate)
        missing_rows = np.random.choice(len(df), n_missing, replace=False)
        missing_cols = np.random.choice(numeric_cols, n_missing)
        
        for row, col in zip(missing_rows, missing_cols):
            df.loc[row, col] = np.nan
        
        logger.info(f"  添加了 {n_missing} 个缺失值")
    
    # 添加异常值
    if outlier_rate > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
        
        n_outliers = int(len(df) * outlier_rate)
        outlier_rows = np.random.choice(len(df), n_outliers, replace=False)
        outlier_cols = np.random.choice(numeric_cols, n_outliers)
        
        for row, col in zip(outlier_rows, outlier_cols):
            # 添加 3-5 倍标准差的异常值
            std = df[col].std()
            mean = df[col].mean()
            direction = np.random.choice([-1, 1])
            df.loc[row, col] = mean + direction * np.random.uniform(3, 5) * std
        
        logger.info(f"  添加了 {n_outliers} 个异常值")
    
    # 保存数据
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存 CSV
        csv_path = output_dir / f"test_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  已保存 CSV: {csv_path}")
        
        # 保存 Excel
        try:
            excel_path = output_dir / f"test_data_{timestamp}.xlsx"
            df.to_excel(excel_path, index=False, engine='openpyxl')
            logger.info(f"  已保存 Excel: {excel_path}")
        except ImportError:
            logger.warning("  openpyxl 未安装，跳过 Excel 保存")
        
        # 保存配置信息
        config = {
            'n_samples': n_samples,
            'n_features': len(df.columns) - 2,  # 排除 id 和 target
            'positive_ratio': positive_ratio,
            'missing_rate': missing_rate,
            'outlier_rate': outlier_rate,
            'noise_rate': noise_rate,
            'random_state': random_state,
            'generated_at': datetime.now().isoformat(),
            'columns': list(df.columns)
        }
        
        config_path = output_dir / f"test_data_config_{timestamp}.json"
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存配置：{config_path}")
    
    return X, y, df


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='生成机器学习测试数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_test_data.py --samples 10000 --features 20
  python generate_test_data.py --samples 5000 --positive-ratio 0.15 --missing 0.1
  python generate_test_data.py --output ./data --samples 20000
        """
    )
    
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=10000,
        help='样本数量 (默认：10000)'
    )
    
    parser.add_argument(
        '--features', '-f',
        type=int,
        default=20,
        help='特征数量 (默认：20)'
    )
    
    parser.add_argument(
        '--positive-ratio', '-p',
        type=float,
        default=0.1,
        help='正样本比例 (默认：0.1)'
    )
    
    parser.add_argument(
        '--missing', '-m',
        type=float,
        default=0.05,
        help='缺失值比例 (默认：0.05)'
    )
    
    parser.add_argument(
        '--outliers', '-o',
        type=float,
        default=0.02,
        help='异常值比例 (默认：0.02)'
    )
    
    parser.add_argument(
        '--noise',
        type=float,
        default=0.01,
        help='噪声比例 (默认：0.01)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='随机种子 (默认：42)'
    )
    
    parser.add_argument(
        '--output', '-O',
        type=str,
        default=None,
        help='输出目录 (默认：./generated_data)'
    )
    
    parser.add_argument(
        '--no-id',
        action='store_true',
        help='不生成 ID 列'
    )
    
    parser.add_argument(
        '--no-categorical',
        action='store_true',
        help='不生成分类特征'
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output) if args.output else Path('./generated_data')
    
    # 生成数据
    X, y, df = generate_test_data(
        n_samples=args.samples,
        n_features=args.features,
        positive_ratio=args.positive_ratio,
        missing_rate=args.missing,
        outlier_rate=args.outliers,
        noise_rate=args.noise,
        random_state=args.seed,
        output_dir=output_dir,
        include_id=not args.no_id,
        include_categorical=not args.no_categorical
    )
    
    # 显示数据统计
    print("\n" + "="*60)
    print("数据生成完成!")
    print("="*60)
    print(f"\n数据统计:")
    print(f"  总行数：{len(df):,}")
    print(f"  总列数：{len(df.columns)}")
    print(f"  正样本数：{y.sum():,} ({y.mean():.2%})")
    print(f"  负样本数：{(y==0).sum():,} ({(y==0).mean():.2%})")
    print(f"\n列信息:")
    print(f"  数值列：{len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  分类列：{len(df.select_dtypes(include=['object']).columns)}")
    print(f"\n缺失值统计:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"  {col}: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        print("  无缺失值")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    from typing import Optional, Tuple
    main()
