"""
查询构建器模块

提供安全的 SQL 查询构建功能，防止 SQL 注入
"""

from typing import Optional, List, Dict, Any, Union
from enum import Enum


class JoinType(Enum):
    """连接类型枚举"""
    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    RIGHT = "RIGHT JOIN"
    FULL = "FULL JOIN"
    CROSS = "CROSS JOIN"


class OrderDirection(Enum):
    """排序方向枚举"""
    ASC = "ASC"
    DESC = "DESC"


class QueryBuilder:
    """
    SQL 查询构建器
    
    提供链式调用的 SQL 构建接口，确保 SQL 安全
    """
    
    def __init__(self, table: str):
        """
        初始化查询构建器
        
        Args:
            table: 主表名
        """
        self.table = table
        self._columns: List[str] = ["*"]
        self._joins: List[str] = []
        self._where_clauses: List[str] = []
        self._params: Dict[str, Any] = {}
        self._group_by_cols: List[str] = []
        self._order_by_cols: List[str] = []
        self._limit_value: Optional[int] = None
        self._offset_value: Optional[int] = None
        self._distinct = False
    
    def select(self, *columns: str) -> 'QueryBuilder':
        """
        指定要查询的列
        
        Args:
            columns: 列名列表
        
        Returns:
            self (支持链式调用)
        """
        self._columns = list(columns) if columns else ["*"]
        return self
    
    def distinct(self) -> 'QueryBuilder':
        """
        添加 DISTINCT 关键字
        
        Returns:
            self
        """
        self._distinct = True
        return self
    
    def join(
        self,
        table: str,
        condition: str,
        join_type: JoinType = JoinType.INNER
    ) -> 'QueryBuilder':
        """
        添加 JOIN 子句
        
        Args:
            table: 要连接的表
            condition: 连接条件
            join_type: 连接类型
        
        Returns:
            self
        """
        join_clause = f"{join_type.value} {table} ON {condition}"
        self._joins.append(join_clause)
        return self
    
    def where(
        self,
        column: str,
        operator: str,
        value: Any,
        param_name: Optional[str] = None
    ) -> 'QueryBuilder':
        """
        添加 WHERE 条件
        
        Args:
            column: 列名
            operator: 操作符 (=, >, <, >=, <=, !=, LIKE, IN, etc.)
            value: 值
            param_name: 参数名 (可选，自动生成)
        
        Returns:
            self
        """
        if param_name is None:
            param_name = f"param_{len(self._params)}"
        
        # 处理不同的操作符
        if operator.upper() == 'IN':
            if not isinstance(value, (list, tuple)):
                raise ValueError("IN 操作符需要列表或元组")
            placeholders = ", ".join(f":{param_name}_{i}" for i in range(len(value)))
            self._where_clauses.append(f"{column} IN ({placeholders})")
            for i, v in enumerate(value):
                self._params[f"{param_name}_{i}"] = v
        elif operator.upper() == 'LIKE':
            self._where_clauses.append(f"{column} LIKE :{param_name}")
            self._params[param_name] = f"%{value}%"
        elif operator.upper() == 'IS NULL':
            self._where_clauses.append(f"{column} IS NULL")
        elif operator.upper() == 'IS NOT NULL':
            self._where_clauses.append(f"{column} IS NOT NULL")
        else:
            self._where_clauses.append(f"{column} {operator} :{param_name}")
            self._params[param_name] = value
        
        return self
    
    def and_where(
        self,
        column: str,
        operator: str,
        value: Any,
        param_name: Optional[str] = None
    ) -> 'QueryBuilder':
        """添加 AND 条件 (与 where 相同，语义更清晰)"""
        return self.where(column, operator, value, param_name)
    
    def or_where(
        self,
        column: str,
        operator: str,
        value: Any,
        param_name: Optional[str] = None
    ) -> 'QueryBuilder':
        """
        添加 OR 条件
        
        Args:
            column: 列名
            operator: 操作符
            value: 值
            param_name: 参数名
        
        Returns:
            self
        """
        if param_name is None:
            param_name = f"param_{len(self._params)}"
        
        # 添加 OR 条件
        if operator.upper() == 'IN':
            if not isinstance(value, (list, tuple)):
                raise ValueError("IN 操作符需要列表或元组")
            placeholders = ", ".join(f":{param_name}_{i}" for i in range(len(value)))
            self._where_clauses.append(f"OR {column} IN ({placeholders})")
            for i, v in enumerate(value):
                self._params[f"{param_name}_{i}"] = v
        elif operator.upper() == 'LIKE':
            self._where_clauses.append(f"OR {column} LIKE :{param_name}")
            self._params[param_name] = f"%{value}%"
        elif operator.upper() == 'IS NULL':
            self._where_clauses.append(f"OR {column} IS NULL")
        elif operator.upper() == 'IS NOT NULL':
            self._where_clauses.append(f"OR {column} IS NOT NULL")
        else:
            self._where_clauses.append(f"OR {column} {operator} :{param_name}")
            self._params[param_name] = value
        
        return self
    
    def group_by(self, *columns: str) -> 'QueryBuilder':
        """
        添加 GROUP BY 子句
        
        Args:
            columns: 分组列名
        
        Returns:
            self
        """
        self._group_by_cols = list(columns)
        return self
    
    def order_by(
        self,
        column: str,
        direction: Union[OrderDirection, str] = OrderDirection.ASC
    ) -> 'QueryBuilder':
        """
        添加 ORDER BY 子句
        
        Args:
            column: 排序列
            direction: 排序方向，支持 OrderDirection 枚举或 "ASC"/"DESC" 字符串
        
        Returns:
            self
        """
        if isinstance(direction, OrderDirection):
            direction_value = direction.value
        elif isinstance(direction, str):
            normalized = direction.upper()
            try:
                direction_value = OrderDirection[normalized].value
            except KeyError as exc:
                valid = ", ".join(item.name for item in OrderDirection)
                raise ValueError(f"不支持的排序方向：{direction}，支持：{valid}") from exc
        else:
            raise TypeError("direction 必须是 OrderDirection 或字符串")

        self._order_by_cols.append(f"{column} {direction_value}")
        return self
    
    def limit(self, limit: int) -> 'QueryBuilder':
        """
        添加 LIMIT 子句
        
        Args:
            limit: 限制行数
        
        Returns:
            self
        """
        self._limit_value = limit
        return self
    
    def offset(self, offset: int) -> 'QueryBuilder':
        """
        添加 OFFSET 子句
        
        Args:
            offset: 偏移量
        
        Returns:
            self
        """
        self._offset_value = offset
        return self
    
    def build(self) -> tuple:
        """
        构建 SQL 查询
        
        Returns:
            (sql, params) 元组
        """
        # SELECT 子句
        distinct = "DISTINCT " if self._distinct else ""
        columns = ", ".join(self._columns)
        sql = f"SELECT {distinct}{columns} FROM {self.table}"
        
        # JOIN 子句
        if self._joins:
            sql += " " + " ".join(self._joins)
        
        # WHERE 子句
        if self._where_clauses:
            # 处理第一个条件前的 AND/OR
            where_sql = []
            for i, clause in enumerate(self._where_clauses):
                if i == 0 and clause.startswith("OR "):
                    clause = clause[3:]  # 移除第一个 OR
                where_sql.append(clause)
            sql += " WHERE " + " ".join(where_sql)
        
        # GROUP BY 子句
        if self._group_by_cols:
            sql += " GROUP BY " + ", ".join(self._group_by_cols)
        
        # ORDER BY 子句
        if self._order_by_cols:
            sql += " ORDER BY " + ", ".join(self._order_by_cols)
        
        # LIMIT 子句
        if self._limit_value is not None:
            sql += f" LIMIT {self._limit_value}"
        
        # OFFSET 子句
        if self._offset_value is not None:
            sql += f" OFFSET {self._offset_value}"
        
        return sql, self._params
    
    def to_sql(self) -> str:
        """
        获取 SQL 字符串 (用于调试，不包含参数)
        
        Returns:
            SQL 字符串
        """
        sql, _ = self.build()
        return sql
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取查询参数
        
        Returns:
            参数字典
        """
        return self._params.copy()
    
    def reset(self) -> 'QueryBuilder':
        """
        重置查询构建器
        
        Returns:
            self
        """
        self._columns = ["*"]
        self._joins = []
        self._where_clauses = []
        self._params = {}
        self._group_by_cols = []
        self._order_by_cols = []
        self._limit_value = None
        self._offset_value = None
        self._distinct = False
        return self


class TableInspector:
    """
    表结构检查器
    
    提供表结构信息查询功能
    """
    
    def __init__(self, db_connector):
        """
        初始化表检查器
        
        Args:
            db_connector: DatabaseConnector 实例
        """
        self.db = db_connector
    
    def get_table_info(self, table: str) -> Dict[str, Any]:
        """
        获取表的详细信息
        
        Args:
            table: 表名
        
        Returns:
            表信息字典
        """
        columns = self.db.get_table_columns(table)
        
        # 获取行数
        sql, params = QueryBuilder(table).select("COUNT(*)").build()
        sql = sql.replace("SELECT *", "SELECT COUNT(*)")
        count_df = self.db.query(sql.replace("SELECT *", "SELECT COUNT(*)"), params)
        row_count = count_df.iloc[0, 0] if len(count_df) > 0 else 0
        
        return {
            'table_name': table,
            'column_count': len(columns),
            'row_count': row_count,
            'columns': columns
        }
    
    def get_column_stats(self, table: str, column: str) -> Dict[str, Any]:
        """
        获取列的统计信息
        
        Args:
            table: 表名
            column: 列名
        
        Returns:
            统计信息字典
        """
        # 获取唯一值数量
        sql = f"""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT "{column}") as unique_count,
                COUNT(CASE WHEN "{column}" IS NULL THEN 1 END) as null_count
            FROM {table}
        """
        df = self.db.query(sql)
        
        if len(df) == 0:
            return {}
        
        return {
            'column_name': column,
            'total_rows': int(df['total'].iloc[0]),
            'unique_values': int(df['unique_count'].iloc[0]),
            'null_count': int(df['null_count'].iloc[0]),
            'null_percentage': float(df['null_count'].iloc[0]) / float(df['total'].iloc[0]) * 100
        }
    
    def sample_data(self, table: str, limit: int = 10) -> Dict[str, Any]:
        """
        获取表的数据样本
        
        Args:
            table: 表名
            limit: 样本数量
        
        Returns:
            样本数据
        """
        df = self.db.read_table(table, limit=limit)
        
        return {
            'table_name': table,
            'sample_size': len(df),
            'columns': list(df.columns),
            'data': df.to_dict('records')
        }
