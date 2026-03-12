"""
大模型连接器模块

支持多种大模型 API：
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- 阿里云通义千问 (Qwen)
- 智谱 AI (GLM)
- 百度文心一言
"""

import json
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """大模型提供商基类"""
    
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """发送对话请求"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """测试连接"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """提供商名称"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI 提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.model = config.get('model', 'gpt-3.5-turbo')
        self._client = None
    
    @property
    def name(self) -> str:
        return "OpenAI"
    
    def _get_client(self):
        """懒加载客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                logger.error("请安装 openai 库：pip install openai")
                raise
        return self._client
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """发送对话请求"""
        client = self._get_client()
        
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2000)
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI 请求失败：{e}")
            raise
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            messages = [{"role": "user", "content": "Hello"}]
            self.chat(messages, max_tokens=10)
            return True
        except Exception as e:
            logger.error(f"OpenAI 连接测试失败：{e}")
            return False


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) 提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'claude-3-sonnet-20240229')
        self._client = None
    
    @property
    def name(self) -> str:
        return "Anthropic"
    
    def _get_client(self):
        """懒加载客户端"""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("请安装 anthropic 库：pip install anthropic")
                raise
        return self._client
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """发送对话请求"""
        client = self._get_client()
        
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2000)
        
        try:
            # 转换消息格式
            system_message = ""
            chat_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    chat_messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
            
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_message,
                messages=chat_messages
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic 请求失败：{e}")
            raise
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            messages = [{"role": "user", "content": "Hello"}]
            self.chat(messages, max_tokens=10)
            return True
        except Exception as e:
            logger.error(f"Anthropic 连接测试失败：{e}")
            return False


class QwenProvider(LLMProvider):
    """阿里云通义千问提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'qwen-plus')
        self._client = None
    
    @property
    def name(self) -> str:
        return "Qwen"
    
    def _get_client(self):
        """懒加载客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                # 通义千问使用 OpenAI 兼容接口
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
            except ImportError:
                logger.error("请安装 openai 库：pip install openai")
                raise
        return self._client
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """发送对话请求"""
        client = self._get_client()
        
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2000)
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Qwen 请求失败：{e}")
            raise
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            messages = [{"role": "user", "content": "Hello"}]
            self.chat(messages, max_tokens=10)
            return True
        except Exception as e:
            logger.error(f"Qwen 连接测试失败：{e}")
            return False


class ZhipuProvider(LLMProvider):
    """智谱 AI (GLM) 提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'glm-4')
        self._client = None
    
    @property
    def name(self) -> str:
        return "Zhipu"
    
    def _get_client(self):
        """懒加载客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                # 智谱 AI 使用 OpenAI 兼容接口
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://open.bigmodel.cn/api/paas/v4/"
                )
            except ImportError:
                logger.error("请安装 openai 库：pip install openai")
                raise
        return self._client
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """发送对话请求"""
        client = self._get_client()
        
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2000)
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Zhipu 请求失败：{e}")
            raise
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            messages = [{"role": "user", "content": "Hello"}]
            self.chat(messages, max_tokens=10)
            return True
        except Exception as e:
            logger.error(f"Zhipu 连接测试失败：{e}")
            return False


class BaiduProvider(LLMProvider):
    """百度文心一言提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key', '')
        self.secret_key = config.get('secret_key', '')
        self.model = config.get('model', 'ernie-bot-4')
        self._access_token = None
    
    @property
    def name(self) -> str:
        return "Baidu"
    
    def _get_access_token(self) -> str:
        """获取访问令牌"""
        import requests
        
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        
        response = requests.post(url, params=params)
        data = response.json()
        return data.get("access_token")
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """发送对话请求"""
        import requests
        
        access_token = self._get_access_token()
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.model}?access_token={access_token}"
        
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2000)
        
        try:
            # 转换消息格式
            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
            
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=payload, headers=headers)
            result = response.json()
            return result.get('result', '')
        except Exception as e:
            logger.error(f"Baidu 请求失败：{e}")
            raise
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            messages = [{"role": "user", "content": "Hello"}]
            self.chat(messages, max_tokens=10)
            return True
        except Exception as e:
            logger.error(f"Baidu 连接测试失败：{e}")
            return False


class LLMManager:
    """大模型管理器"""
    
    PROVIDERS = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'qwen': QwenProvider,
        'zhipu': ZhipuProvider,
        'baidu': BaiduProvider
    }
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化大模型管理器
        
        Args:
            storage_path: 配置文件存储路径
        """
        from pathlib import Path
        
        if storage_path:
            self.config_file = Path(storage_path)
        else:
            # 默认存储路径
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
            config_dir.mkdir(exist_ok=True)
            self.config_file = config_dir / "llm_providers.json"
        
        self.providers: Dict[str, LLMProvider] = {}
        self.active_provider: Optional[str] = None
        self.configs: Dict[str, Dict] = {}
        
        # 加载配置
        self._load_configs()
    
    def _load_configs(self) -> None:
        """加载配置"""
        if not self.config_file.exists():
            self._save_configs()
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.configs = data.get('providers', {})
                self.active_provider = data.get('active_provider')
            
            # 初始化提供商
            for name, config in self.configs.items():
                if config.get('enabled', True):
                    self._init_provider(name, config)
            
            logger.info(f"已加载 {len(self.providers)} 个大模型提供商")
        except Exception as e:
            logger.error(f"加载配置失败：{e}")
            self.configs = {}
    
    def _save_configs(self) -> bool:
        """保存配置"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'providers': self.configs,
                'active_provider': self.active_provider
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"保存配置失败：{e}")
            return False
    
    def _init_provider(self, name: str, config: Dict) -> None:
        """初始化提供商"""
        provider_type = config.get('type', '').lower()
        
        if provider_type not in self.PROVIDERS:
            logger.warning(f"未知的大模型类型：{provider_type}")
            return
        
        try:
            provider_class = self.PROVIDERS[provider_type]
            self.providers[name] = provider_class(config)
            logger.info(f"初始化 {name} ({provider_type}) 成功")
        except Exception as e:
            logger.error(f"初始化 {name} 失败：{e}")
    
    def add_provider(self, name: str, provider_type: str, config: Dict,
                     description: str = "", enabled: bool = True) -> str:
        """
        添加大模型提供商
        
        Args:
            name: 配置名称
            provider_type: 提供商类型 (openai/anthropic/qwen/zhipu/baidu)
            config: 配置信息
            description: 描述
            enabled: 是否启用
        
        Returns:
            配置 ID
        """
        config_id = hashlib.md5(f"{name}_{provider_type}".encode()).hexdigest()[:8]
        
        self.configs[config_id] = {
            'id': config_id,
            'name': name,
            'type': provider_type,
            'config': config,
            'description': description,
            'enabled': enabled,
            'created_at': str(pd.Timestamp.now())
        }
        
        # 初始化提供商
        if enabled:
            self._init_provider(config_id, self.configs[config_id])
        
        self._save_configs()
        logger.info(f"添加大模型配置：{name} ({config_id})")
        return config_id
    
    def get_provider(self, config_id: str) -> Optional[LLMProvider]:
        """获取提供商实例"""
        return self.providers.get(config_id)
    
    def get_all_providers(self) -> List[Dict]:
        """获取所有提供商配置"""
        return list(self.configs.values())
    
    def get_active_provider(self) -> Optional[LLMProvider]:
        """获取当前激活的提供商"""
        if self.active_provider:
            return self.providers.get(self.active_provider)
        return None
    
    def set_active_provider(self, config_id: str) -> bool:
        """设置激活的提供商"""
        if config_id not in self.configs:
            logger.error(f"配置不存在：{config_id}")
            return False
        
        self.active_provider = config_id
        self._save_configs()
        logger.info(f"已激活大模型：{self.configs[config_id]['name']}")
        return True
    
    def delete_provider(self, config_id: str) -> bool:
        """删除提供商配置"""
        if config_id not in self.configs:
            return False
        
        if self.active_provider == config_id:
            self.active_provider = None
        
        # 移除实例
        if config_id in self.providers:
            del self.providers[config_id]
        
        del self.configs[config_id]
        self._save_configs()
        return True
    
    def update_provider(self, config_id: str, **kwargs) -> bool:
        """更新提供商配置"""
        if config_id not in self.configs:
            return False
        
        for key, value in kwargs.items():
            if key not in ['id', 'created_at']:
                self.configs[config_id][key] = value
        
        self._save_configs()
        return True
    
    def chat(self, messages: List[Dict], provider_id: Optional[str] = None,
             **kwargs) -> str:
        """
        发送对话请求
        
        Args:
            messages: 消息列表
            provider_id: 指定提供商 ID（不指定则使用激活的）
            **kwargs: 其他参数
        
        Returns:
            回复内容
        """
        # 选择提供商
        if provider_id:
            provider = self.providers.get(provider_id)
        else:
            provider = self.get_active_provider()
        
        if not provider:
            raise ValueError("没有可用的大模型提供商")
        
        return provider.chat(messages, **kwargs)
    
    def test_provider(self, config_id: str) -> bool:
        """测试提供商连接"""
        provider = self.providers.get(config_id)
        if not provider:
            return False
        
        return provider.test_connection()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_providers': len(self.configs),
            'enabled_providers': len(self.providers),
            'active_provider': self.configs.get(self.active_provider, {}).get('name'),
            'provider_types': list(set(c['type'] for c in self.configs.values()))
        }
    
    def __len__(self) -> int:
        return len(self.configs)


# 单例模式
_manager_instance: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """获取大模型管理器单例"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = LLMManager()
    return _manager_instance


# 延迟导入 pandas
try:
    import pandas as pd
except ImportError:
    pd = None
