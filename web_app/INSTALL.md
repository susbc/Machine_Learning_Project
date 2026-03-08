# Web UI 安装和启动指南

## 📦 安装依赖

### 方式 1：使用 requirements.txt（推荐）

```bash
cd /Users/rgj/PycharmProjects/Machine_Learning_Project
pip install -r requirements.txt
```

### 方式 2：手动安装

```bash
pip install streamlit plotly pandas numpy scikit-learn lightgbm openpyxl
```

## 🚀 启动 Web 应用

### macOS / Linux

```bash
cd web_app
streamlit run app.py
```

或使用启动脚本：

```bash
./run_web.sh
```

### Windows

```bash
cd web_app
streamlit run app.py
```

或双击运行：

```
run_web.bat
```

## 🌐 访问应用

启动后在浏览器访问：**http://localhost:8501**

## 📋 功能模块

### 1. 📊 数据管理
- 上传 CSV/Excel 数据文件
- 数据预览（表格展示）
- 字段名自动读取
- 数据统计信息（空值率、唯一值、分布等）

### 2. ⚙️ 配置管理
- field_config.csv 在线编辑
- 字段类型配置
- 异常值处理设置
- 缺失值填充配置
- 编码方式选择
- 标准化方法配置
- 配置保存/加载/导出/导入

### 3. 📈 数据分析
- EDA 可视化（分布图）
- 相关性热力图
- 缺失值热力图
- 异常值检测箱线图
- HTML 分析报告生成

### 4. 🚀 模型训练
- LightGBM 参数调整
  - n_estimators, learning_rate, max_depth
  - num_leaves, min_child_samples
  - subsample, colsample_bytree
  - reg_alpha, reg_lambda
- 训练进度实时展示
- 日志流输出
- 交叉验证结果
- 模型评估指标（AUC、Precision、Recall、F1）
- ROC 曲线、PR 曲线、混淆矩阵
- 特征重要性可视化

### 5. 📦 模型管理
- 已训练模型列表
- 模型详情查看
- 模型参数展示
- 模型指标查看
- 特征重要性分析
- 模型下载
- 模型删除

### 6. 🔮 模型预测
- 上传预测数据
- 选择已训练模型
- 分类阈值调整
- 预测进度展示
- 预测结果下载（CSV/Excel）

### 7. 📋 结果分析
- 预测结果分布可视化
- 阈值分析（如有真实标签）
- 与训练集对比
- HTML 分析报告导出

## 🎯 使用流程

1. **启动应用**
   ```bash
   cd web_app
   streamlit run app.py
   ```

2. **上传数据**
   - 访问 http://localhost:8501
   - 点击左侧导航【📊 数据管理】
   - 上传 CSV 或 Excel 文件

3. **配置字段**
   - 点击【⚙️ 配置管理】
   - 编辑字段类型和处理规则
   - 保存配置

4. **分析数据**
   - 点击【📈 数据分析】
   - 查看分布、相关性、异常值

5. **训练模型**
   - 点击【🚀 模型训练】
   - 调整模型参数
   - 点击【开始训练】
   - 等待训练完成

6. **管理模型**
   - 点击【📦 模型管理】
   - 查看已训练模型
   - 下载或删除模型

7. **进行预测**
   - 点击【🔮 模型预测】
   - 选择模型
   - 上传预测数据
   - 查看和下载结果

8. **分析结果**
   - 点击【📋 结果分析】
   - 查看预测分布
   - 导出分析报告

## 💡 提示

- **数据格式**: 支持 CSV (UTF-8/GBK 编码) 和 Excel (.xlsx, .xls)
- **大文件**: 大文件上传可能需要一些时间，请耐心等待
- **训练时间**: 根据数据量大小，训练可能需要几分钟
- **浏览器**: 推荐使用 Chrome、Firefox、Safari 等现代浏览器
- **端口**: 如果 8501 端口被占用，Streamlit 会自动使用其他端口

## 🔧 配置选项

### Streamlit 配置

创建 `~/.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "localhost"
headless = false

[theme]
primaryColor = "#3498db"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### 环境变量

```bash
# 设置项目路径
export ML_PROJECT_ROOT="/Users/rgj/PycharmProjects/Machine_Learning_Project"
```

## 🐛 常见问题

### 1. Streamlit 未找到

```bash
pip install streamlit
```

### 2. 端口被占用

```bash
# 查找占用端口的进程
lsof -i :8501

# 或使用其他端口
streamlit run app.py --server.port 8502
```

### 3. 依赖冲突

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 4. 图表不显示

确保安装了 Plotly:

```bash
pip install plotly
```

## 📚 技术栈

- **前端框架**: Streamlit
- **图表库**: Plotly
- **数据处理**: Pandas, NumPy
- **机器学习**: scikit-learn, LightGBM
- **文件处理**: openpyxl

## 📝 注意事项

- 所有数据保存在会话状态中，刷新页面会丢失
- 训练完成的模型会保存到 `model_code/saved/` 目录
- 预测结果可以下载为 CSV 或 Excel 格式
- 建议定期备份重要模型文件

---

**版本**: 1.0.0  
**最后更新**: 2026-03-08
