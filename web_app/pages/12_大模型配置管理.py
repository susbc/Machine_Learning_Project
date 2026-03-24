"""
大模型配置管理页面

功能：
- 配置各大模型 API 接口
- 测试连接
- 切换激活模型
- 查看使用统计
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_code.llm import get_llm_manager, LLMManager


# 页面配置
st.set_page_config(
    page_title="大模型配置管理",
    page_icon="🤖",
    layout="wide"
)

# 标题
st.title("🤖 大模型配置管理")
st.markdown("---")


def init_llm_manager():
    """初始化大模型管理器"""
    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = get_llm_manager()
    return st.session_state.llm_manager


def get_provider_config(provider_type: str) -> Dict:
    """获取提供商配置模板"""
    templates = {
        'openai': {
            'name': 'OpenAI',
            'fields': {
                'api_key': {'type': 'password', 'label': 'API Key', 'help': 'OpenAI API Key'},
                'base_url': {'type': 'text', 'label': 'Base URL', 'default': 'https://api.openai.com/v1'},
                'model': {'type': 'text', 'label': '默认模型', 'default': 'gpt-3.5-turbo', 'customizable': True}
            },
            'models': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4o', 'gpt-4o-mini']
        },
        'anthropic': {
            'name': 'Anthropic (Claude)',
            'fields': {
                'api_key': {'type': 'password', 'label': 'API Key', 'help': 'Anthropic API Key'},
                'model': {'type': 'text', 'label': '默认模型', 'default': 'claude-3-sonnet-20240229', 'customizable': True}
            },
            'models': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'claude-3-5-sonnet-20240620']
        },
        'qwen': {
            'name': '阿里云通义千问',
            'fields': {
                'api_key': {'type': 'password', 'label': 'API Key', 'help': '阿里云 DashScope API Key'},
                'model': {'type': 'text', 'label': '默认模型', 'default': 'qwen-plus', 'customizable': True}
            },
            'models': ['qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-max-longcontext', 'qwen2.5-72b-instruct']
        },
        'zhipu': {
            'name': '智谱 AI (GLM)',
            'fields': {
                'api_key': {'type': 'password', 'label': 'API Key', 'help': '智谱 AI API Key'},
                'model': {'type': 'text', 'label': '默认模型', 'default': 'glm-4', 'customizable': True}
            },
            'models': ['glm-4', 'glm-3-turbo', 'glm-4-flash', 'glm-4-air', 'glm-4-long']
        },
        'baidu': {
            'name': '百度文心一言',
            'fields': {
                'api_key': {'type': 'password', 'label': 'API Key', 'help': '百度 API Key'},
                'secret_key': {'type': 'password', 'label': 'Secret Key', 'help': '百度 Secret Key'},
                'model': {'type': 'text', 'label': '默认模型', 'default': 'ernie-bot-4'}
            },
            'models': ['ernie-bot-4', 'ernie-bot', 'ernie-bot-turbo', 'ernie-speed']
        }
    }
    return templates.get(provider_type, {})


def render_provider_form(manager: LLMManager, edit_id: Optional[str] = None):
    """渲染配置表单"""
    st.subheader("➕ 添加大模型配置" if not edit_id else "✏️ 编辑配置")

    edit_data = None
    if edit_id:
        edit_data = manager.configs.get(edit_id)

    # 使用唯一的 form key
    form_key = f"llm_provider_form_{edit_id}" if edit_id else "llm_provider_form_add"

    with st.form(form_key):
        col1, col2 = st.columns(2)
        
        with col1:
            # 配置名称
            config_name = st.text_input(
                "配置名称 *",
                value=edit_data['name'] if edit_data else "",
                help="自定义配置名称，便于识别"
            )
            
            # 提供商类型
            provider_type = st.selectbox(
                "大模型提供商 *",
                ["openai", "anthropic", "qwen", "zhipu", "baidu"],
                format_func=lambda x: get_provider_config(x)['name'],
                index=0 if not edit_data else ["openai", "anthropic", "qwen", "zhipu", "baidu"].index(edit_data['type'])
            )
            
            # 描述
            description = st.text_area(
                "描述",
                value=edit_data.get('description', '') if edit_data else "",
                help="可选的配置描述"
            )
        
        with col2:
            # 启用状态
            enabled = st.checkbox(
                "启用此配置",
                value=edit_data.get('enabled', True) if edit_data else True
            )
            
            # 是否设为默认
            set_default = st.checkbox(
                "设为默认模型",
                value=edit_data.get('id') == manager.active_provider if edit_data else False
            )
        
        st.markdown("---")
        st.markdown("### 🔑 API 配置")
        
        template = get_provider_config(provider_type)
        config_fields = {}
        
        # 根据类型显示不同字段
        if provider_type == 'openai':
            col1, col2 = st.columns(2)
            with col1:
                config_fields['api_key'] = st.text_input("API Key", type="password",
                    value=edit_data['config'].get('api_key', '') if edit_data else "",
                    help="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            with col2:
                config_fields['base_url'] = st.text_input("Base URL",
                    value=edit_data['config'].get('base_url', 'https://api.openai.com/v1') if edit_data else "https://api.openai.com/v1")
            # 模型选择 - 支持预设 + 自定义输入
            saved_model = edit_data['config'].get('model', 'gpt-3.5-turbo') if edit_data else 'gpt-3.5-turbo'
            preset_index = template['models'].index(saved_model) if saved_model in template['models'] else 0
            preset_model = st.selectbox("预设模型",
                template['models'],
                index=preset_index)

            custom_model = st.text_input("或输入自定义模型名称",
                value="" if preset_model in template['models'] else preset_model,
                placeholder="例如：gpt-4o-2024-08-06",
                help="留空则使用上方预设模型")

            config_fields['model'] = custom_model if custom_model.strip() else preset_model

        elif provider_type == 'anthropic':
            config_fields['api_key'] = st.text_input("API Key", type="password",
                value=edit_data['config'].get('api_key', '') if edit_data else "",
                help="sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            saved_model = edit_data['config'].get('model', 'claude-3-sonnet-20240229') if edit_data else 'claude-3-sonnet-20240229'
            preset_index = template['models'].index(saved_model) if saved_model in template['models'] else 0
            preset_model = st.selectbox("预设模型", template['models'],
                index=preset_index)

            custom_model = st.text_input("或输入自定义模型名称",
                value="" if preset_model in template['models'] else preset_model,
                placeholder="例如：claude-3-5-sonnet-20241022",
                help="留空则使用上方预设模型")

            config_fields['model'] = custom_model if custom_model.strip() else preset_model

        elif provider_type == 'qwen':
            config_fields['api_key'] = st.text_input("API Key", type="password",
                value=edit_data['config'].get('api_key', '') if edit_data else "",
                help="阿里云 DashScope API Key")

            saved_model = edit_data['config'].get('model', 'qwen-plus') if edit_data else 'qwen-plus'
            preset_index = template['models'].index(saved_model) if saved_model in template['models'] else 0
            preset_model = st.selectbox("预设模型", template['models'],
                index=preset_index)

            custom_model = st.text_input("或输入自定义模型名称",
                value="" if preset_model in template['models'] else preset_model,
                placeholder="例如：qwen2.5-vl-72b-instruct",
                help="留空则使用上方预设模型")

            config_fields['model'] = custom_model if custom_model.strip() else preset_model

        elif provider_type == 'zhipu':
            config_fields['api_key'] = st.text_input("API Key", type="password",
                value=edit_data['config'].get('api_key', '') if edit_data else "",
                help="智谱 AI API Key")

            saved_model = edit_data['config'].get('model', 'glm-4') if edit_data else 'glm-4'
            preset_index = template['models'].index(saved_model) if saved_model in template['models'] else 0
            preset_model = st.selectbox("预设模型", template['models'],
                index=preset_index)

            custom_model = st.text_input("或输入自定义模型名称",
                value="" if preset_model in template['models'] else preset_model,
                placeholder="例如：glm-4-airx",
                help="留空则使用上方预设模型")

            config_fields['model'] = custom_model if custom_model.strip() else preset_model

        elif provider_type == 'baidu':
            col1, col2 = st.columns(2)
            with col1:
                config_fields['api_key'] = st.text_input("API Key", type="password",
                    value=edit_data['config'].get('api_key', '') if edit_data else "")
            with col2:
                config_fields['secret_key'] = st.text_input("Secret Key", type="password",
                    value=edit_data['config'].get('secret_key', '') if edit_data else "")

            saved_model = edit_data['config'].get('model', 'ernie-bot-4') if edit_data else 'ernie-bot-4'
            preset_index = template['models'].index(saved_model) if saved_model in template['models'] else 0
            preset_model = st.selectbox("预设模型", template['models'],
                index=preset_index)
            
            custom_model = st.text_input("或输入自定义模型名称",
                value="" if preset_model in template['models'] else preset_model,
                placeholder="例如：ernie-4.0-turbo-8k",
                help="留空则使用上方预设模型")
            
            config_fields['model'] = custom_model if custom_model.strip() else preset_model
        
        # 表单按钮
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            submit_button = st.form_submit_button("💾 保存配置", use_container_width=True)
        
        with col2:
            test_button = st.form_submit_button("🧪 测试连接", use_container_width=True)
        
        with col3:
            if edit_id:
                cancel_button = st.form_submit_button("❌ 取消编辑", use_container_width=True)
            else:
                cancel_button = False
        
        # 处理表单提交
        if submit_button:
            if not config_name:
                st.error("❌ 配置名称不能为空")
                return
            
            # 保存配置
            if edit_id:
                manager.update_provider(edit_id, 
                    config=config_fields,
                    name=config_name,
                    description=description,
                    enabled=enabled)
                st.success(f"✅ 配置 '{config_name}' 已更新")
            else:
                config_id = manager.add_provider(
                    name=config_name,
                    provider_type=provider_type,
                    config=config_fields,
                    description=description,
                    enabled=enabled
                )
                st.success(f"✅ 配置 '{config_name}' 已保存 (ID: {config_id})")
            
            # 设为默认
            if set_default and not edit_id:
                manager.set_active_provider(config_id)
                st.success("✅ 已设为默认模型")
            
            st.rerun()
        
        # 测试连接
        if test_button:
            if not edit_data and not config_fields.get('api_key'):
                st.error("❌ 请先填写 API Key")
            else:
                with st.spinner("🔄 正在测试连接..."):
                    # 创建临时配置进行测试
                    temp_config = {
                        'type': provider_type,
                        'config': config_fields if config_fields else edit_data['config']
                    }
                    
                    from model_code.llm.llm_connector import LLMManager
                    temp_manager = LLMManager.PROVIDERS.get(provider_type)
                    if temp_manager:
                        try:
                            provider = temp_manager(temp_config['config'])
                            success = provider.test_connection()
                            
                            if success:
                                st.success("✅ 连接测试成功！")
                            else:
                                st.error("❌ 连接测试失败，请检查 API Key")
                        except Exception as e:
                            st.error(f"❌ 测试失败：{str(e)}")
                    else:
                        st.error("❌ 未知的大模型类型")
        
        # 取消编辑
        if cancel_button:
            st.session_state.edit_mode = False
            st.session_state.edit_id = None
            st.rerun()


def render_provider_list(manager: LLMManager):
    """渲染配置列表"""
    st.subheader(f"📚 已配置模型 ({len(manager)})")
    
    providers = manager.get_all_providers()
    
    if not providers:
        st.info("📭 暂无配置，请添加大模型配置")
        return
    
    # 统计
    stats = manager.get_statistics()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("总配置数", stats['total_providers'])
    
    with col2:
        st.metric("已启用", stats['enabled_providers'])
    
    with col3:
        active = stats['active_provider'] or "无"
        st.metric("当前激活", active)
    
    st.markdown("---")
    
    # 按类型分组
    by_type = {}
    for prov in providers:
        ptype = prov.get('type', 'unknown')
        if ptype not in by_type:
            by_type[ptype] = []
        by_type[ptype].append(prov)
    
    # 显示各组
    type_names = {
        'openai': '🟢 OpenAI',
        'anthropic': '🟣 Anthropic',
        'qwen': '🔵 通义千问',
        'zhipu': '🟡 智谱 AI',
        'baidu': '🔴 百度'
    }
    
    for ptype, provs in by_type.items():
        with st.expander(f"{type_names.get(ptype, ptype)} ({len(provs)})"):
            for prov in provs:
                render_provider_card(manager, prov)


def render_provider_card(manager: LLMManager, prov: Dict):
    """渲染单个配置卡片"""
    prov_id = prov['id']
    name = prov['name']
    ptype = prov.get('type', 'unknown')
    is_active = prov_id == manager.active_provider
    is_enabled = prov.get('enabled', True)
    description = prov.get('description', '')
    
    col1, col2, col3 = st.columns([4, 1, 2])
    
    with col1:
        if is_active:
            st.markdown(f"### ✅ {name} (当前激活)")
        elif is_enabled:
            st.markdown(f"### {name}")
        else:
            st.markdown(f"### {name} ⏸️")
        
        if description:
            st.caption(description)
        
        # 模型信息
        config = prov.get('config', {})
        model = config.get('model', 'N/A')
        st.caption(f"📦 模型：{model}")
        
        # 类型标签
        type_names = {
            'openai': 'OpenAI',
            'anthropic': 'Anthropic',
            'qwen': '通义千问',
            'zhipu': '智谱 AI',
            'baidu': '百度'
        }
        st.caption(f"🏷️ 类型：{type_names.get(ptype, ptype)}")
    
    with col2:
        # 状态
        if is_enabled:
            st.success("已启用") if is_enabled else st.warning("已禁用")
    
    with col3:
        # 操作按钮
        col_a, col_b = st.columns(2)
        
        with col_a:
            if not is_active:
                if st.button("🔌 激活", key=f"activate_{prov_id}", use_container_width=True):
                    manager.set_active_provider(prov_id)
                    st.success(f"已激活：{name}")
                    st.rerun()
            else:
                st.button("✅ 已激活", disabled=True, use_container_width=True)
        
        with col_b:
            if st.button("✏️ 编辑", key=f"edit_{prov_id}", use_container_width=True):
                st.session_state.edit_mode = True
                st.session_state.edit_id = prov_id
                st.rerun()
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            if st.button("🧪 测试", key=f"test_{prov_id}", use_container_width=True):
                with st.spinner("测试中..."):
                    success = manager.test_provider(prov_id)
                
                if success:
                    st.success("连接成功！")
                else:
                    st.error("连接失败！")
        
        with col_d:
            if st.button("🗑️ 删除", key=f"delete_{prov_id}", use_container_width=True):
                manager.delete_provider(prov_id)
                st.warning(f"已删除：{name}")
                st.rerun()
    
    st.markdown("---")


def main():
    """主函数"""
    # 初始化
    manager = init_llm_manager()
    
    # 检查编辑模式
    if not hasattr(st.session_state, 'edit_mode'):
        st.session_state.edit_mode = False
        st.session_state.edit_id = None
    
    # 侧边栏
    with st.sidebar:
        st.header("🔧 操作")
        
        if st.button("➕ 新建配置", use_container_width=True):
            st.session_state.edit_mode = False
            st.session_state.edit_id = None
        
        st.markdown("---")
        
        # 统计信息
        st.subheader("📊 统计")
        stats = manager.get_statistics()
        st.json(stats)
        
        st.markdown("---")
        
        # 帮助
        st.subheader("❓ 帮助")
        st.markdown("""
        ### API Key 获取
        
        **OpenAI**: https://platform.openai.com/api-keys
        
        **Anthropic**: https://console.anthropic.com/
        
        **通义千问**: https://dashscope.console.aliyun.com/
        
        **智谱 AI**: https://open.bigmodel.cn/
        
        **百度**: https://console.bce.baidu.com/qianfan/
        """)
    
    # 主内容区
    if st.session_state.edit_mode:
        # 编辑模式：只显示编辑表单
        render_provider_form(manager, st.session_state.edit_id)
    else:
        # 普通模式：显示添加表单
        render_provider_form(manager)

    st.markdown("---")
    
    render_provider_list(manager)


if __name__ == "__main__":
    main()
