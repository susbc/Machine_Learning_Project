import sys

import pandas as pd

from config.config import *
from model_code.predict import load_and_predict
from model_code.train import lgbm_train_with_random_search
from  utils import *

class BuildingModel:

    def __init__(self,data_path,config_path,data_month):
        self.data_path = data_path
        self.data = None
        self.df_field = pd.read_excel(config_path)
        self.label = None
        self.key =  None
        self.data_month = data_month

    def _load_data(self):
        self.data = pd.read_csv(self.data_path,sep="|",encoding='utf-8')

    def _prepare_data(self):
        self._load_data()
        self._fields_stype()
        self._handle_outliers()
        self._handle_missing_value()
        self._field_encode()
        self._field_standardize()

    def model_train(self):
        self._prepare_data()
        self._train()

    def model_predict(self):
        self._prepare_data()
        self._predict()


    def _train(self):
        neg_data = self.data[self.data[self.label]==0]
        pos_data = self.data[self.data[self.label]==1]
        describe_values = neg_data.describe().T[["mean", "std"]].values

        pos_data_num = pos_data.shape[0]
        weight = pos_data_num/neg_data.shape[0]
        error_rate = 0.1
        index_result = neg_data_undersampling(neg_data, describe_values, pos_data_num, weight, error_rate)
        neg_data_sample = neg_data.loc[index_result]
        self.data = pd.concat([neg_data_sample,pos_data],axis=0)
        y=self.data[self.label]
        X = self.data.drop([self.label,self.key],axis=1)
        lgbm_train_with_random_search(X,y)

    def _predict(self):
        key = self.data[self.key]
        data = self.data.drop([self.label,self.key],axis=1)
        load_and_predict(data, key,self.data_month)

    def _fields_stype(self):
        """
        公有方法：核心功能 - 仅处理field_type为'num'的字段，实现数值类型自动转换
        逻辑：
            1. 提取所有field_type='num'的列名
            2. 遍历对应self.data的列，先转为可计算的数值类型（无法转换的设为NaN）
            3. 区分整数和浮点数，分别转换为对应类型（支持缺失值兼容）
        """
        # 步骤1：提取所有需要处理的数值型列名（field_type严格等于'num'）
        num_field_names = self.df_field[self.df_field['field_type'] is not None]['field_name'].tolist()
        if not num_field_names:
            print("未找到field_type标记为'num'的字段，无需进行数值类型转换。")
            return

        print(f"开始处理数值类型字段，共 {len(num_field_names)} 个字段：{num_field_names}")

        # 步骤2：遍历每个数值字段，进行类型转换
        for field in num_field_names:
            # 检查该字段是否存在于self.data中
            if field not in self.data.columns:
                print(f"警告：self.data中不存在字段 '{field}'，跳过该字段转换。")
                continue
            field_type = self.df_field[self.df_field['field_name'] == field]['field_type']
            # 步骤3：先将列转为数值类型（无法转换的值设为NaN，避免整体转换失败）
            try:
                # 转换为数值列（coerce：无法转换的转为NaN）
                if field_type == 'key':
                    self.key = field
                    continue
                if field_type == 'label':
                    self.label = field
                    continue
                if field_type == 'str':
                    continue
                numeric_series = pd.to_numeric(self.data[field], errors='coerce')
            except Exception as e:
                print(f"错误：字段 '{field}' 转换为基础数值类型失败，报错信息：{str(e)}，跳过该字段。")
                continue

            # 步骤4：区分整数和浮点数，进行精准类型转换
            # 判断：非NaN值是否全部为整数（即数值等于其整数形式）
            is_all_integer = np.all(numeric_series.dropna().apply(lambda x: x == int(x)))

            if is_all_integer:
                # 整数类型转换：使用pd.Int64（支持NaN缺失值的可空整型，优于原生int）
                self.data[field] = numeric_series.astype(pd.Int64Dtype())
                print(f"字段 '{field}' 已转换为【可空整型（pd.Int64）】")
            else:
                # 浮点数类型转换：转为标准64位浮点数
                self.data[field] = numeric_series.astype(np.float64)
                print(f"字段 '{field}' 已转换为【64位浮点数（np.float64）】")

        print("数值类型字段转换全部完成！")

    def _handle_outliers(self):
        self.data = handle_dataframe_outliers(self.data, self.df_field)
    def _handle_missing_value(self):
        self.data = handle_dataframe_missing_values(self.data, self.df_field)
    def _field_encode(self):
        self.data = encode_dataframe(self.data, self.df_field)
    def _field_standardize(self):
        self.data = standardize_dataframe(self.data, self.df_field)


def run(data_month):
    model = BuildingModel(data_path,config_path,data_month)
    if is_train:
        model.model_train()
    if is_pretrain:
        model.model_predict()

if __name__ == '__main__':
    data_month = sys.argv[0]
    run(data_month)
