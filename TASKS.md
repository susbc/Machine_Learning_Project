# Machine Learning Project - Phase 1 开发任务

> **历史快照说明（2026-11-02 补注）**
>
> 本文是 2026-03-23 的 Phase 1 任务清单，所列事项多数已完成或已变更；
> 当前迭代计划以 [PHASE_PLAN.md](PHASE_PLAN.md)、功能与启动方式以
> [README.md](README.md) 为准。

**创建时间**: 2026-03-23 14:25  
**状态**: 🟡 待开发  
**优先级**: 🔴 P0  

---

## 任务概述

实现 Phase 1: 知识检索层改造 (NL2SQL 闭环)

参考文档：`PHASE_PLAN.md`

---

## 需要创建的组件

### 1. Neo4j 客户端
**文件**: `core/knowledge/neo4j_client.py`

**功能**:
- [ ] Neo4j 数据库连接 (使用 neo4j Python 驱动)
- [ ] 表节点 CRUD (Create/Read/Update/Delete)
- [ ] 字段节点 CRUD
- [ ] 关系查询 (FOREIGN_KEY, HAS_COLUMN, REFERENCES)
- [ ] Schema 导出

**依赖**: `pip install neo4j`

**示例代码结构**:
```python
from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def create_table_node(self, name: str, description: str):
        pass
    
    def create_column_node(self, table: str, name: str, 
                           col_type: str, description: str):
        pass
    
    def create_foreign_key(self, from_table: str, from_col: str,
                           to_table: str, to_col: str):
        pass
    
    def get_table_schema(self, table_name: str) -> Dict:
        pass
    
    def find_related_tables(self, table_name: str, depth: int = 2) -> List[str]:
        pass
    
    def close(self):
        self.driver.close()
```

---

### 2. FAISS 客户端
**文件**: `core/knowledge/faiss_client.py`

**功能**:
- [ ] FAISS 向量索引创建 (使用 IndexFlatL2 或 IndexIVFFlat)
- [ ] 字段 Embedding 存储
- [ ] 相似度搜索 (Top-K)
- [ ] 索引持久化 (保存/加载)

**依赖**: `pip install faiss-cpu`

**示例代码结构**:
```python
import faiss
import numpy as np
import pickle

class FAISSClient:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []  # 存储元数据 (table_name, column_name)
    
    def add_embeddings(self, embeddings: np.ndarray, 
                       metadata: List[Dict]):
        """添加 Embedding 到索引"""
        pass
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 5) -> List[Dict]:
        """相似度搜索"""
        pass
    
    def save(self, path: str):
        """保存索引"""
        pass
    
    def load(self, path: str):
        """加载索引"""
        pass
```

---

### 3. Schema Embedder
**文件**: `core/knowledge/schema_embedder.py`

**功能**:
- [ ] 使用 sentence-transformers 生成 Embedding
- [ ] 表名 Embedding
- [ ] 字段名 Embedding
- [ ] 字段业务含义 Embedding
- [ ] 批量嵌入

**依赖**: `pip install sentence-transformers`

**示例代码结构**:
```python
from sentence_transformers import SentenceTransformer

class SchemaEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_table_name(self, table_name: str) -> np.ndarray:
        """生成表名 Embedding"""
        pass
    
    def embed_column(self, table_name: str, 
                     column_name: str, 
                     description: str = "") -> np.ndarray:
        """生成字段 Embedding"""
        pass
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量生成 Embedding"""
        pass
```

---

### 4. NL2SQL 引擎
**文件**: `core/knowledge/nl2sql_engine.py`

**功能**:
- [ ] 自然语言解析
- [ ] 检索 Neo4j 获取表关系
- [ ] 检索 FAISS 获取相关字段
- [ ] 构建提示词调用 LLM 生成 SQL
- [ ] SQL 验证 (SQLFluff)
- [ ] 执行 SQL 返回 DataFrame

**依赖**: `pip install sqlfluff` + 现有 LLM 依赖

**示例代码结构**:
```python
class NL2SQLEngine:
    def __init__(self, neo4j_client: Neo4jClient,
                 faiss_client: FAISSClient,
                 embedder: SchemaEmbedder,
                 llm_provider):
        pass
    
    async def query(self, natural_language: str) -> pd.DataFrame:
        """
        自然语言查询
        
        Args:
            natural_language: "查询所有 VIP 客户的交易记录"
        
        Returns:
            pd.DataFrame: 查询结果
        """
        # 1. 生成查询 Embedding
        # 2. FAISS 检索相关字段
        # 3. Neo4j 检索表关系
        # 4. 构建提示词
        # 5. 调用 LLM 生成 SQL
        # 6. SQLFluff 验证
        # 7. 执行 SQL
        pass
    
    def _build_prompt(self, nl_query: str, 
                      schema_context: str) -> str:
        """构建 LLM 提示词"""
        pass
    
    def _validate_sql(self, sql: str, dialect: str = "mysql") -> bool:
        """使用 SQLFluff 验证 SQL"""
        pass
```

---

### 5. 数据检索智能体
**文件**: `agents/data_retrieval_agent.py`

**功能**:
- [ ] 统一接口
- [ ] 错误处理
- [ ] 日志记录
- [ ] 性能监控

**示例代码结构**:
```python
class DataRetrievalAgent:
    def __init__(self, config: Dict):
        self.neo4j = Neo4jClient(...)
        self.faiss = FAISSClient(...)
        self.embedder = SchemaEmbedder(...)
        self.nl2sql = NL2SQLEngine(...)
    
    async def retrieve(self, query: str) -> pd.DataFrame:
        """数据检索主接口"""
        try:
            return await self.nl2sql.query(query)
        except Exception as e:
            logger.error(f"检索失败：{e}")
            raise
    
    async def explain_schema(self) -> str:
        """返回数据库 Schema 说明"""
        pass
    
    def close(self):
        """关闭连接"""
        self.neo4j.close()
```

---

## 测试计划

### 单元测试
- [x] `tests/knowledge/test_neo4j_client.py` (7KB)
- [x] `tests/knowledge/test_faiss_client.py` (6KB)
- [x] `tests/knowledge/test_schema_embedder.py` (6KB)
- [ ] `tests/knowledge/test_nl2sql_engine.py`

### 集成测试
- [ ] `tests/integration/test_data_retrieval.py`

### 测试用例 (10 个 NL2SQL 场景)
1. 简单单表查询
2. 多表 JOIN 查询
3. 聚合函数查询
4. 条件过滤查询
5. 排序查询
6. 分组查询
7. 子查询
8. 日期范围查询
9. 模糊匹配查询
10. 复杂业务查询

**验收标准**: 通过率 ≥ 90%

---

## 实施步骤

### Step 1: 环境准备
```bash
cd /Users/rgj/PycharmProjects/Machine_Learning_Project
source .venv/bin/activate
pip install neo4j faiss-cpu sentence-transformers sqlfluff
```

### Step 2: 创建目录结构
```bash
mkdir -p core/knowledge
mkdir -p agents
mkdir -p tests/knowledge
```

### Step 3: 实现组件 (按顺序)
1. Neo4jClient (基础依赖)
2. SchemaEmbedder (独立)
3. FAISSClient (依赖 Embedder)
4. NL2SQLEngine (依赖以上所有)
5. DataRetrievalAgent (封装)

### Step 4: 编写测试
- 每个组件完成后立即编写测试
- 运行测试验证

### Step 5: 集成验证
- 使用真实数据库测试
- 10 个 NL2SQL 测试用例

---

## 配置示例

### Neo4j 配置 (.env)
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j@123
```

### FAISS 索引路径
```
FAISS_INDEX_PATH=/path/to/faiss_index.bin
```

---

## 参考资源

- Neo4j Python Driver: https://neo4j.com/docs/python-manual/current/
- FAISS 文档：https://github.com/facebookresearch/faiss
- sentence-transformers: https://www.sbert.net/
- SQLFluff: https://docs.sqlfluff.com/

---

*创建时间：2026-03-23 14:25*  
*状态：待开发*
esearch/faiss
- sentence-transformers: https://www.sbert.net/
- SQLFluff: https://docs.sqlfluff.com/

---

*创建时间：2026-03-23 14:25*  
*状态：待开发*
