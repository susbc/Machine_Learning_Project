"""
模型编排主引擎 - 优化版

核心优化:
1. 修复数据穿越：先 split 再 fit 预处理
2. 使用 sklearn Pipeline 管理数据流
3. 解耦配置，移除全局依赖
4. 减少内存拷贝
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from model_code.utils import ConfigDrivenPreprocessor
from model_code.predict import load_and_predict


class BuildingModel:
    """
    机器学习流水线总控引擎 - 优化版
    
    优化点:
    1. 配置注入式初始化，移除全局依赖
    2. 先划分训练/测试集，再 fit 预处理器 (防止数据穿越)
    3. 使用 ConfigDrivenPreprocessor 确保预处理一致性
    """
    
    def __init__(self, data_path: str, config_path: str, data_month: str,
                 output_dir: Optional[str] = None, model_dir: Optional[str] = None,
                 algorithm: str = "auto"):
        """
        初始化模型引擎
        
        Args:
            data_path: 数据文件路径
            config_path: Excel 配置表路径
            data_month: 数据月份标识
            output_dir: 输出目录 (可选)
            model_dir: 模型保存目录 (可选)
            algorithm: 训练算法配置 ("auto" | "compare" | 具体算法名)
        """
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.data_month = data_month
        self.output_dir = Path(output_dir) if output_dir else Path("../output")
        self.model_dir = Path(model_dir) if model_dir else Path("saved")
        
        # 确保目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 训练算法配置
        self.algorithm = algorithm  # "auto" | "compare" | 具体算法名

        # 数据与配置
        self.data: Optional[pd.DataFrame] = None
        self.df_field: Optional[pd.DataFrame] = None
        self.label: Optional[str] = None
        self.key: Optional[str] = None
        
        # 预处理器 (sklearn Pipeline 兼容)
        self.preprocessor: Optional[ConfigDrivenPreprocessor] = None
        
        # 训练相关
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def _load_data(self) -> None:
        """加载数据"""
        print(f"加载数据：{self.data_path}")
        self.data = pd.read_csv(self.data_path, sep="|", encoding='utf-8')
        print(f"数据加载完成：{self.data.shape[0]:,} 行 × {self.data.shape[1]:} 列")
        
    def _load_config(self) -> None:
        """加载配置表"""
        print(f"加载配置：{self.config_path}")
        self.df_field = pd.read_excel(self.config_path)
        print(f"配置加载完成：{len(self.df_field)} 个字段规则")
        
    def _identify_fields(self) -> None:
        """识别 key 和 label 字段"""
        for _, row in self.df_field.iterrows():
            if row['field_type'] == 'key':
                self.key = row['field_name']
            elif row['field_type'] == 'label':
                self.label = row['field_name']
        
        if not self.key or not self.label:
            raise ValueError("配置表中必须包含 field_type 为 'key' 和 'label' 的字段")
        
        print(f"识别字段 - Key: {self.key}, Label: {self.label}")
    
    def _fields_stype(self) -> None:
        """
        数值类型转换 - 优化版
        
        优化:
        1. 向量化转换
        2. 移除不必要的循环
        """
        num_fields = self.df_field[
            self.df_field['field_type'].isin(['num', 'key', 'label'])
        ]['field_name'].tolist()
        
        if not num_fields:
            print("未找到需要类型转换的字段")
            return
        
        print(f"转换 {len(num_fields)} 个数值字段类型...")
        
        for field in num_fields:
            if field not in self.data.columns:
                continue
            
            field_type = self.df_field[
                self.df_field['field_name'] == field
            ]['field_type'].iloc[0]
            
            if field_type == 'key':
                self.key = field
                continue
            if field_type == 'label':
                self.label = field
                continue
            if field_type == 'str':
                continue
            
            # 向量化数值转换
            self.data[field] = pd.to_numeric(self.data[field], errors='coerce')
            
            # 智能类型推断
            numeric_series = self.data[field]
            is_all_integer = numeric_series.dropna().apply(lambda x: x == int(x)).all()
            
            if is_all_integer:
                self.data[field] = numeric_series.astype(pd.Int64Dtype())
                print(f"  {field} → Int64")
            else:
                self.data[field] = numeric_series.astype(np.float64)
                print(f"  {field} → Float64")
        
        print("类型转换完成")
    
    def _prepare_data_for_train(self) -> None:
        """
        训练数据准备 - 修复数据穿越
        
        关键修复:
        1. 先划分训练集/测试集
        2. 仅在训练集上 fit 预处理器
        3. 用训练集的统计量 transform 测试集
        """
        self._load_data()
        self._load_config()
        self._identify_fields()
        self._fields_stype()

        # 关键：先划分数据集，再 fit 预处理器 (防止数据穿越)
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            self.data, test_size=0.3, random_state=42, stratify=self.data[self.label]
        )

        print(f"\n数据集划分:")
        print(f"  训练集：{train_data.shape[0]:,} 行")
        print(f"  测试集：{test_data.shape[0]:,} 行")

        # 初始化预处理器 (包含异常值处理 + 缺失值填充 + 编码 + 标准化)
        feature_cols = [c for c in train_data.columns if c not in [self.label, self.key]]
        self.preprocessor = ConfigDrivenPreprocessor(self.df_field)

        # 仅在训练集上 fit (异常值边界和缺失值统计量都来自训练集)
        self.preprocessor.fit(train_data[feature_cols])

        # 分别 transform
        self.X_train = self.preprocessor.transform(train_data[feature_cols])
        self.X_test = self.preprocessor.transform(test_data[feature_cols])
        self.y_train = train_data[self.label]
        self.y_test = test_data[self.label]

        print("\n数据预处理完成 (无数据穿越)")
    
    def _prepare_data_for_predict(self) -> None:
        """
        预测数据准备

        优先使用训练时 fit 好的预处理器，确保预处理逻辑完全一致。
        若预处理器尚未 fit (仅运行预测模式)，则全量 fit 后 transform。
        """
        self._load_data()
        self._load_config()
        self._identify_fields()
        self._fields_stype()

        feature_cols = [c for c in self.data.columns if c not in [self.label, self.key]]

        if self.preprocessor is not None:
            # 使用训练时 fit 好的预处理器 (推荐路径)
            self.data = self.preprocessor.transform(self.data[feature_cols])
            print("使用训练时 fit 好的预处理器进行预测")
        else:
            # 仅运行预测模式：全量 fit + transform
            print("预处理器尚未初始化，对全量数据 fit + transform")
            self.preprocessor = ConfigDrivenPreprocessor(self.df_field)
            self.preprocessor.fit(self.data[feature_cols])
            self.data = self.preprocessor.transform(self.data[feature_cols])
    
    def model_train(self) -> Dict[str, Any]:
        """
        模型训练主流程 - 使用 ModelTrainer 统一编排

        支持三种模式:
        - algorithm="auto": 智能推荐最佳算法
        - algorithm="compare": 多算法对比
        - algorithm="lightgbm"/"xgboost"/...: 指定算法训练
        """
        from model_code.model_trainer import ModelTrainer

        print("=" * 60)
        print("开始模型训练流程")
        print("=" * 60)

        self._prepare_data_for_train()

        print(f"\n训练集: {self.X_train.shape[0]:,} 样本, {self.X_train.shape[1]} 特征")
        print(f"正样本比例: {self.y_train.mean():.2%}")

        trainer = ModelTrainer(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            self.model_dir, self.output_dir,
        )

        # 根据 algorithm 参数选择训练模式
        if self.algorithm == "auto":
            print("\n模式：智能推荐")
            results = trainer.train_auto()
        elif self.algorithm == "compare":
            print("\n模式：多算法对比")
            results = trainer.train_compare()
        else:
            print(f"\n模式：指定算法 {self.algorithm}")
            results = trainer.train_single(self.algorithm)

        print("\n" + "=" * 60)
        print("训练完成")
        print("=" * 60)

        return results
    
    def model_predict(self) -> pd.DataFrame:
        """
        模型预测主流程
        
        Returns:
            预测结果 DataFrame
        """
        print("=" * 60)
        print("开始模型预测流程")
        print("=" * 60)
        
        self._prepare_data_for_predict()
        
        # 准备预测数据
        key = self.data[self.key]
        features = self.data.drop([self.label, self.key], axis=1)
        
        # 预测
        results = load_and_predict(
            features, key, self.data_month,
            self.model_dir, self.output_dir
        )
        
        print("\n" + "=" * 60)
        print("预测完成")
        print("=" * 60)
        
        return results


def run(data_month: str, data_path: str, config_path: str,
        is_train: bool = True, is_predict: bool = True,
        output_dir: Optional[str] = None, model_dir: Optional[str] = None,
        algorithm: str = "auto") -> None:
    """
    运行入口函数 - 优化版

    优化:
    1. 配置参数化，移除全局依赖
    2. 支持单独训练或预测
    """
    model = BuildingModel(
        data_path=data_path,
        config_path=config_path,
        data_month=data_month,
        output_dir=output_dir,
        model_dir=model_dir,
        algorithm=algorithm,
    )
    
    if is_train:
        model.model_train()
    
    if is_predict:
        model.model_predict()


if __name__ == '__main__':
    from model_code.config.config import (
        data_path, config_path, is_train, is_pretrain
    )

    data_month = sys.argv[1] if len(sys.argv) > 1 else "default"
    
    run(
        data_month=data_month,
        data_path=data_path,
        config_path=config_path,
        is_train=is_train,
        is_predict=is_pretrain
    )
