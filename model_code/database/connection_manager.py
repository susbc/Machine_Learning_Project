"""
数据库连接管理器

功能：
- 保存历史连接配置
- 测试连接
- 加载/切换连接
- 删除连接
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    数据库连接管理器
    
    管理多个数据库连接配置，支持保存、加载、切换和删除
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化连接管理器
        
        Args:
            storage_path: 存储文件路径，默认在项目根目录的 config/saved_connections.json
        """
        if storage_path:
            self.storage_file = Path(storage_path)
        else:
            # 默认存储路径
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
            config_dir.mkdir(exist_ok=True)
            self.storage_file = config_dir / "saved_connections.json"
        
        self.connections: Dict[str, Dict] = {}
        self.current_connection: Optional[str] = None  # 当前激活的连接名称
        
        # 加载已保存的连接
        self._load_connections()
    
    def _load_connections(self) -> None:
        """从文件加载连接配置"""
        if not self.storage_file.exists():
            logger.info("连接配置文件不存在，创建新文件")
            self._save_connections()
            return
        
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.connections = data.get('connections', {})
                self.current_connection = data.get('current_connection')
            logger.info(f"已加载 {len(self.connections)} 个保存的连接")
        except Exception as e:
            logger.error(f"加载连接配置失败：{e}")
            self.connections = {}
            self.current_connection = None
    
    def _save_connections(self) -> bool:
        """保存连接配置到文件"""
        try:
            # 确保目录存在
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'connections': self.connections,
                'current_connection': self.current_connection,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"连接配置已保存到 {self.storage_file}")
            return True
        except Exception as e:
            logger.error(f"保存连接配置失败：{e}")
            return False
    
    def _generate_connection_id(self, config: Dict[str, Any]) -> str:
        """
        根据连接配置生成唯一 ID
        
        Args:
            config: 数据库配置
        
        Returns:
            连接 ID (哈希值前 8 位)
        """
        # 使用关键信息生成哈希
        key_parts = [
            config.get('db_type', ''),
            config.get('host', ''),
            str(config.get('port', '')),
            config.get('database', ''),
            config.get('user', '')
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:8]
    
    def add_connection(self, name: str, config: Dict[str, Any], 
                       description: str = "", tags: Optional[List[str]] = None) -> str:
        """
        添加新的连接配置
        
        Args:
            name: 连接名称（用户自定义）
            config: 数据库配置
            description: 连接描述
            tags: 标签列表
        
        Returns:
            连接 ID
        """
        connection_id = self._generate_connection_id(config)
        
        # 检查是否已存在相同配置
        if connection_id in self.connections:
            logger.warning(f"连接配置已存在：{connection_id}")
            # 更新现有连接
            self.connections[connection_id].update({
                'name': name,
                'description': description,
                'tags': tags or [],
                'last_used': datetime.now().isoformat(),
                'use_count': self.connections[connection_id].get('use_count', 0) + 1
            })
        else:
            # 创建新连接
            self.connections[connection_id] = {
                'id': connection_id,
                'name': name,
                'config': config,
                'description': description,
                'tags': tags or [],
                'db_type': config.get('db_type', 'mysql'),
                'created_at': datetime.now().isoformat(),
                'last_used': datetime.now().isoformat(),
                'use_count': 1,
                'is_active': False
            }
        
        # 保存
        self._save_connections()
        
        logger.info(f"添加连接：{name} ({connection_id})")
        return connection_id
    
    def get_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定连接配置
        
        Args:
            connection_id: 连接 ID
        
        Returns:
            连接配置或 None
        """
        return self.connections.get(connection_id)
    
    def get_all_connections(self) -> List[Dict[str, Any]]:
        """
        获取所有连接配置
        
        Returns:
            连接配置列表
        """
        return list(self.connections.values())
    
    def get_active_connection(self) -> Optional[Dict[str, Any]]:
        """
        获取当前激活的连接
        
        Returns:
            激活的连接配置或 None
        """
        if self.current_connection:
            return self.connections.get(self.current_connection)
        return None
    
    def set_active_connection(self, connection_id: str) -> bool:
        """
        设置当前激活的连接
        
        Args:
            connection_id: 连接 ID
        
        Returns:
            是否成功
        """
        if connection_id not in self.connections:
            logger.error(f"连接不存在：{connection_id}")
            return False
        
        # 取消所有连接的激活状态
        for cid in self.connections:
            self.connections[cid]['is_active'] = False
        
        # 激活指定连接
        self.connections[connection_id]['is_active'] = True
        self.connections[connection_id]['last_used'] = datetime.now().isoformat()
        self.connections[connection_id]['use_count'] = \
            self.connections[connection_id].get('use_count', 0) + 1
        
        self.current_connection = connection_id
        self._save_connections()
        
        logger.info(f"已激活连接：{self.connections[connection_id]['name']}")
        return True
    
    def delete_connection(self, connection_id: str) -> bool:
        """
        删除连接配置
        
        Args:
            connection_id: 连接 ID
        
        Returns:
            是否成功
        """
        if connection_id not in self.connections:
            logger.error(f"连接不存在：{connection_id}")
            return False
        
        # 如果删除的是当前激活连接，清除激活状态
        if self.current_connection == connection_id:
            self.current_connection = None
        
        conn_name = self.connections[connection_id]['name']
        del self.connections[connection_id]
        self._save_connections()
        
        logger.info(f"已删除连接：{conn_name}")
        return True
    
    def update_connection(self, connection_id: str, **kwargs) -> bool:
        """
        更新连接配置
        
        Args:
            connection_id: 连接 ID
            **kwargs: 要更新的字段
        
        Returns:
            是否成功
        """
        if connection_id not in self.connections:
            logger.error(f"连接不存在：{connection_id}")
            return False
        
        # 不允许更新 id 和 created_at
        protected_fields = ['id', 'created_at']
        for key, value in kwargs.items():
            if key not in protected_fields:
                self.connections[connection_id][key] = value
        
        self.connections[connection_id]['last_updated'] = datetime.now().isoformat()
        self._save_connections()
        
        logger.info(f"已更新连接：{connection_id}")
        return True
    
    def search_connections(self, keyword: str) -> List[Dict[str, Any]]:
        """
        搜索连接
        
        Args:
            keyword: 搜索关键词
        
        Returns:
            匹配的连接列表
        """
        results = []
        keyword = keyword.lower()
        
        for conn in self.connections.values():
            # 在名称、描述、标签、数据库类型中搜索
            if (keyword in conn['name'].lower() or
                keyword in conn.get('description', '').lower() or
                keyword in conn.get('db_type', '').lower() or
                any(keyword in tag.lower() for tag in conn.get('tags', []))):
                results.append(conn)
        
        logger.info(f"搜索 '{keyword}' 找到 {len(results)} 个连接")
        return results
    
    def get_connections_by_type(self, db_type: str) -> List[Dict[str, Any]]:
        """
        按数据库类型获取连接
        
        Args:
            db_type: 数据库类型
        
        Returns:
            连接列表
        """
        return [
            conn for conn in self.connections.values()
            if conn.get('db_type') == db_type
        ]
    
    def export_connections(self, output_path: str) -> bool:
        """
        导出连接配置到文件
        
        Args:
            output_path: 输出文件路径
        
        Returns:
            是否成功
        """
        try:
            # 导出时不包含密码
            export_data = []
            for conn in self.connections.values():
                export_conn = conn.copy()
                if 'config' in export_conn:
                    # 隐藏密码
                    export_conn['config'] = conn['config'].copy()
                    export_conn['config']['password'] = '***'
                export_data.append(export_conn)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已导出 {len(export_data)} 个连接到 {output_path}")
            return True
        except Exception as e:
            logger.error(f"导出连接失败：{e}")
            return False
    
    def import_connections(self, input_path: str, merge: bool = True) -> int:
        """
        从文件导入连接配置
        
        Args:
            input_path: 输入文件路径
            merge: 是否合并（True=合并，False=覆盖）
        
        Returns:
            导入的连接数量
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if isinstance(import_data, list):
                # 列表格式
                imported = 0
                for conn in import_data:
                    conn_id = conn.get('id')
                    if conn_id:
                        if merge:
                            self.connections[conn_id] = conn
                        imported += 1
                
                if not merge:
                    self.connections = {c['id']: c for c in import_data if 'id' in c}
                    imported = len(self.connections)
                
                self._save_connections()
                logger.info(f"已导入 {imported} 个连接")
                return imported
            else:
                logger.error("无效的导入文件格式")
                return 0
                
        except Exception as e:
            logger.error(f"导入连接失败：{e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取连接统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_connections': len(self.connections),
            'by_type': {},
            'active_connection': None,
            'most_used': None,
            'recently_used': None
        }
        
        if not self.connections:
            return stats
        
        # 按类型统计
        for conn in self.connections.values():
            db_type = conn.get('db_type', 'unknown')
            stats['by_type'][db_type] = stats['by_type'].get(db_type, 0) + 1
        
        # 当前激活
        if self.current_connection:
            stats['active_connection'] = self.connections[self.current_connection]['name']
        
        # 最常用
        most_used = max(self.connections.values(), key=lambda x: x.get('use_count', 0))
        stats['most_used'] = {
            'name': most_used['name'],
            'count': most_used.get('use_count', 0)
        }
        
        # 最近使用
        recently_used = max(
            self.connections.values(),
            key=lambda x: x.get('last_used', '')
        )
        stats['recently_used'] = {
            'name': recently_used['name'],
            'time': recently_used.get('last_used', '')
        }
        
        return stats
    
    def __len__(self) -> int:
        """返回连接数量"""
        return len(self.connections)
    
    def __repr__(self) -> str:
        return f"ConnectionManager(connections={len(self)}, active={self.current_connection})"


# 单例模式
_manager_instance: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """获取连接管理器单例"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ConnectionManager()
    return _manager_instance
