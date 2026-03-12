"""
Web 应用快速测试脚本

用于验证 Web 应用的基本功能
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        from utils.session_state import init_session_state
        print("✅ utils.session_state")
    except Exception as e:
        print(f"❌ utils.session_state: {e}")
        return False
    
    try:
        from utils.file_handler import read_uploaded_file, calculate_data_statistics
        print("✅ utils.file_handler")
    except Exception as e:
        print(f"❌ utils.file_handler: {e}")
        return False
    
    try:
        from components.data_upload import render_data_upload_section
        print("✅ components.data_upload")
    except Exception as e:
        print(f"❌ components.data_upload: {e}")
        return False
    
    try:
        from components.config_editor import render_config_editor
        print("✅ components.config_editor")
    except Exception as e:
        print(f"❌ components.config_editor: {e}")
        return False
    
    try:
        from components.charts import render_distribution_chart
        print("✅ components.charts")
    except Exception as e:
        print(f"❌ components.charts: {e}")
        return False
    
    try:
        from components.progress import render_progress_bar
        print("✅ components.progress")
    except Exception as e:
        print(f"❌ components.progress: {e}")
        return False
    
    return True


def test_streamlit():
    """测试 Streamlit 是否可用"""
    print("\n🔍 测试 Streamlit...")
    
    try:
        import streamlit as st
        print(f"✅ Streamlit 版本：{st.__version__}")
        return True
    except ImportError:
        print("❌ Streamlit 未安装")
        print("💡 请运行：pip install streamlit plotly")
        return False


def test_plotly():
    """测试 Plotly 是否可用"""
    print("\n🔍 测试 Plotly...")
    
    try:
        import plotly
        print(f"✅ Plotly 版本：{plotly.__version__}")
        return True
    except ImportError:
        print("❌ Plotly 未安装")
        print("💡 请运行：pip install plotly")
        return False


def test_model_code():
    """测试 model_code 模块是否可用"""
    print("\n🔍 测试 model_code 模块...")
    
    try:
        from model_code.utils import ConfigDrivenPreprocessor
        print("✅ model_code.utils")
    except Exception as e:
        print(f"❌ model_code.utils: {e}")
        return False
    
    try:
        from model_code.train import lgbm_train_with_random_search
        print("✅ model_code.train")
    except Exception as e:
        print(f"❌ model_code.train: {e}")
        return False
    
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 机器学习 Web UI 测试")
    print("=" * 60)
    
    results = []
    
    results.append(("Streamlit", test_streamlit()))
    results.append(("Plotly", test_plotly()))
    results.append(("模块导入", test_imports()))
    results.append(("Model Code", test_model_code()))
    
    print("\n" + "=" * 60)
    print("📊 测试结果")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ 所有测试通过！")
        print("\n🚀 启动 Web 应用:")
        print("   cd web_app")
        print("   streamlit run app.py")
        print("\n🌐 访问地址：http://localhost:8501")
        return 0
    else:
        print("\n❌ 部分测试失败，请修复后重试")
        return 1


if __name__ == "__main__":
    sys.exit(main())
