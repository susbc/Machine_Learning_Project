# Machine Learning Project

基于 Streamlit 的机器学习模型训练和预测 Web 应用。

## 项目结构

```
machine-learning-project/
├── backend/           # 后端模型代码
├── frontend/          # 前端界面
├── web_app/          # Streamlit 应用
├── tests/            # 测试文件
├── scripts/          # 工具脚本
├── docs/             # 文档
└── requirements.txt  # Python 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动应用

```bash
cd web_app
streamlit run app.py --server.port 8503
```

### 3. 访问界面

打开浏览器：**http://localhost:8503**

## 功能特点

- 📊 模型训练（支持 LightGBM）
- 🔮 在线预测
- 📈 可视化分析
- 📥 数据上传
- 📤 模型导出

## 技术栈

- **前端**: Streamlit
- **后端**: Python + LightGBM
- **数据处理**: Pandas + NumPy

## 训练数据

- 位置：`/Users/rgj/Downloads/train.csv`
- 规格：22500 行 × 22 列

## 已知问题

- Streamlit 必须使用 `.venv/bin/python` 启动，避免使用系统 Python

## 最近更新

- 2026-03-11: 修复 LightGBM 网格搜索缺少 fit() 问题
- 2026-03-11: 修复 Streamlit 使用系统 Python 问题

---

*Last Updated: 2026-03-12*
