"""
tab_model_mgmt.py — 模型管理与对比 Tab

合并自：
  - web_app/pages/5_模型管理.py（已训练模型列表 / 详情 / 删除 / 导出）
  - web_app/pages/9_模型对比.py（多模型并排评估）
"""

import streamlit as st


def render() -> None:
    """渲染模型管理与对比 Tab。"""
    sub_list, sub_cmp = st.tabs(["📦 模型列表", "⚖️ 模型对比"])
    with sub_list:
        _render_model_list()
    with sub_cmp:
        _render_model_compare()


# ===== 子分区 1：模型列表（来自原 5_模型管理.py） =====

def _scan_saved_models() -> list:
    """扫描模型目录中的所有模型文件。"""
    import joblib

    model_dir = st.session_state.model_dir
    models = []

    if model_dir.exists():
        for file_path in model_dir.glob("*.joblib"):
            try:
                model_data = joblib.load(file_path)
                if isinstance(model_data, dict):
                    model_info = {
                        "name": file_path.stem,
                        "path": str(file_path),
                        "type": "summary",
                        "timestamp": file_path.stat().st_mtime,
                        "size": file_path.stat().st_size / 1024 / 1024,
                    }
                    if "test_auc" in model_data:
                        model_info["auc"] = model_data["test_auc"]
                    if "best_f1" in model_data:
                        model_info["f1"] = model_data["best_f1"]
                    if "model_params" in model_data:
                        model_info["params"] = model_data["model_params"]
                    models.append(model_info)
                else:
                    models.append(
                        {
                            "name": file_path.stem,
                            "path": str(file_path),
                            "type": "model",
                            "timestamp": file_path.stat().st_mtime,
                            "size": file_path.stat().st_size / 1024 / 1024,
                        }
                    )
            except Exception as e:
                models.append(
                    {
                        "name": file_path.stem,
                        "path": str(file_path),
                        "type": "unknown",
                        "timestamp": file_path.stat().st_mtime,
                        "size": file_path.stat().st_size / 1024 / 1024,
                        "error": str(e),
                    }
                )

    models.sort(key=lambda x: x["timestamp"], reverse=True)
    return models


def _render_model_list() -> None:
    import pandas as pd
    import joblib
    from datetime import datetime
    from pathlib import Path
    from web_app.components.charts import render_feature_importance

    st.subheader("📦 已训练模型")

    models = _scan_saved_models()

    if not models:
        st.info("📭 暂无已训练的模型，请先在「4️⃣ 模型训练」Tab 训练模型")
        return

    model_df = pd.DataFrame(models)
    display_df = model_df.copy()
    display_df["训练时间"] = pd.to_datetime(display_df["timestamp"], unit="s").dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    display_df["文件大小"] = display_df["size"].apply(lambda x: f"{x:.2f} MB")

    show_cols = ["name", "训练时间", "文件大小"]
    if "auc" in display_df.columns:
        display_df["AUC"] = display_df["auc"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "-"
        )
        show_cols.append("AUC")
    if "f1" in display_df.columns:
        display_df["F1"] = display_df["f1"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "-"
        )
        show_cols.append("F1")

    st.dataframe(
        display_df[show_cols], use_container_width=True, hide_index=True, height=300
    )

    st.markdown("---")
    st.subheader("🔍 模型详情")

    selected_model_name = st.selectbox(
        "选择模型",
        [m["name"] for m in models],
        format_func=lambda x: f"{x}",
        key="mm_list_select",
    )

    if not selected_model_name:
        return

    selected_model = next(m for m in models if m["name"] == selected_model_name)
    st.session_state.selected_model = selected_model["path"]

    try:
        model_data = joblib.load(selected_model["path"])

        sub_metric, sub_param, sub_imp, sub_op = st.tabs(
            ["📊 模型指标", "⚙️ 模型参数", "📈 特征重要性", "💾 操作"]
        )

        with sub_metric:
            st.markdown("**📊 模型评估指标**")
            if isinstance(model_data, dict):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if "test_auc" in model_data:
                        st.metric("ROC-AUC", f"{model_data['test_auc']:.4f}")
                with col2:
                    if "best_f1" in model_data:
                        st.metric("最佳 F1", f"{model_data['best_f1']:.4f}")
                with col3:
                    if "best_threshold" in model_data:
                        st.metric("最佳阈值", f"{model_data['best_threshold']:.4f}")
                with col4:
                    if "test_ap" in model_data:
                        st.metric("AUC-PR", f"{model_data['test_ap']:.4f}")

                st.markdown("**📋 训练信息**")
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.metric("训练集大小", f"{model_data.get('train_size', '-'):,}")
                with info_col2:
                    st.metric("测试集大小", f"{model_data.get('test_size', '-'):,}")

                if "y_test" in model_data and "y_pred" in model_data:
                    from sklearn.metrics import classification_report

                    report = classification_report(
                        model_data["y_test"], model_data["y_pred"], output_dict=True
                    )
                    st.markdown("**📋 分类报告**")
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)
            else:
                st.info("这是模型对象文件，不包含评估指标")

        with sub_param:
            st.markdown("**⚙️ 模型参数**")
            if isinstance(model_data, dict) and "model_params" in model_data:
                params_df = pd.DataFrame(
                    {
                        "参数": list(model_data["model_params"].keys()),
                        "值": list(model_data["model_params"].values()),
                    }
                )
                st.dataframe(params_df, use_container_width=True, hide_index=True)
            elif hasattr(model_data, "get_params"):
                params = model_data.get_params()
                params_df = pd.DataFrame(
                    {"参数": list(params.keys()), "值": list(params.values())}
                )
                st.dataframe(params_df, use_container_width=True, hide_index=True)
            else:
                st.info("无参数信息")

        with sub_imp:
            st.markdown("**📈 特征重要性**")
            if isinstance(model_data, dict):
                if "feature_names" in model_data and "feature_importances" in model_data:
                    fig = render_feature_importance(
                        model_data["feature_names"],
                        model_data["feature_importances"],
                        top_n=20,
                    )
                    st.plotly_chart(
                        fig, use_container_width=True, key="mm_list_feature_imp_chart"
                    )

                    importance_df = pd.DataFrame(
                        {
                            "特征名": model_data["feature_names"],
                            "重要性": model_data["feature_importances"],
                        }
                    ).sort_values("重要性", ascending=False)
                    csv = importance_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载特征重要性 (CSV)",
                        data=csv,
                        file_name=f"feature_importance_{selected_model_name}.csv",
                        mime="text/csv",
                        key="mm_list_dl_imp_btn",
                    )
                else:
                    st.info("无特征重要性信息")
            elif hasattr(model_data, "feature_importances_"):
                if "feature_names" in st.session_state:
                    fig = render_feature_importance(
                        st.session_state.feature_names,
                        model_data.feature_importances_,
                        top_n=20,
                    )
                    st.plotly_chart(
                        fig, use_container_width=True, key="mm_list_feature_imp_chart_obj"
                    )
                else:
                    st.info("需要特征名来显示特征重要性")
            else:
                st.info("无特征重要性信息")

        with sub_op:
            st.markdown("**💾 模型操作**")

            st.info(
                f"""
            **模型名称**: {selected_model['name']}
            **文件路径**: {selected_model['path']}
            **文件大小**: {selected_model['size']:.2f} MB
            **创建时间**: {datetime.fromtimestamp(selected_model['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
            """
            )

            col1, col2 = st.columns(2)
            with col1:
                with open(selected_model["path"], "rb") as f:
                    model_bytes = f.read()
                st.download_button(
                    label="📥 下载模型文件",
                    data=model_bytes,
                    file_name=f"{selected_model_name}.joblib",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="mm_list_dl_model_btn",
                )

            with col2:
                if st.button(
                    "🗑️ 删除模型",
                    use_container_width=True,
                    type="secondary",
                    key="mm_list_delete_btn",
                ):
                    try:
                        Path(selected_model["path"]).unlink()
                        st.success(f"模型 {selected_model_name} 已删除")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败：{str(e)}")

            st.markdown("---")
            st.markdown("**🎯 设为当前模型**")

            if st.button(
                "✅ 设为当前使用模型",
                type="primary",
                use_container_width=True,
                key="mm_list_set_current_btn",
            ):
                st.session_state.current_model_path = selected_model["path"]
                st.session_state.current_model_data = model_data
                st.success(f"已将 {selected_model_name} 设为当前模型")
                st.info("💡 现在可以在「6️⃣ 预测与结果」Tab 使用此模型进行预测")

    except Exception as e:
        st.error(f"加载模型失败：{str(e)}")
        import traceback
        st.code(traceback.format_exc())

    # ===== 会话中的模型 =====
    st.markdown("---")
    st.subheader("🔄 会话中的模型")

    if st.session_state.get("training_results") is not None:
        st.success("✅ 当前会话中有刚训练的模型")
        results = st.session_state.training_results
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                f"""
            **ROC-AUC**: {results['test_auc']:.4f}
            **最佳 F1**: {results['best_f1']:.4f}
            **最佳阈值**: {results['best_threshold']:.4f}
            """
            )
        with col2:
            if st.button("💾 保存当前模型到列表", key="mm_list_save_session_btn"):
                model_path = (
                    st.session_state.model_dir
                    / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                )
                joblib.dump(results["model"], model_path)
                st.success(f"模型已保存：{model_path.name}")
                st.rerun()


# ===== 子分区 2：模型对比（来自原 9_模型对比.py） =====


def _encode_test_data(X_test):
    """对测试数据进行编码转换（数值化）。"""
    from sklearn.preprocessing import LabelEncoder

    X_encoded = X_test.copy()
    for col in X_encoded.columns:
        if X_encoded[col].dtype == "object":
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    return X_encoded


def _render_model_compare() -> None:
    import pandas as pd
    import numpy as np
    import joblib
    from datetime import datetime
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from web_app.components.model_comparison import render_model_comparison_section

    st.subheader("⚖️ 模型对比")

    if st.session_state.get("uploaded_data") is None:
        st.warning("⚠️ 请先在「1️⃣ 数据准备」Tab 上传数据")
        return

    if st.session_state.get("field_config") is None:
        st.warning("⚠️ 请先在「2️⃣ 字段配置」Tab 配置字段")
        return

    st.markdown("**📦 已训练模型**")

    model_dir = st.session_state.get("model_dir", Path("model_code/saved"))
    model_files = list(model_dir.glob("*.joblib"))
    model_files = [f for f in model_files if "train_summary" not in f.name]

    if not model_files:
        st.warning("未找到已训练的模型，请先在「4️⃣ 模型训练」Tab 训练模型")
        return

    available_models = []
    for model_path in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            summary_name = f"train_summary_{model_path.stem.split('_')[-1]}.joblib"
            summary_path = model_dir / summary_name

            model_info = {
                "name": model_path.stem,
                "path": str(model_path),
                "modified": datetime.fromtimestamp(model_path.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                ),
            }

            if summary_path.exists():
                summary = joblib.load(summary_path)
                model_info["auc"] = summary.get("test_auc", 0)
                model_info["f1"] = summary.get("best_f1", 0)
                model_info["params"] = summary.get("best_params", {})
            else:
                model_info["auc"] = 0
                model_info["f1"] = 0
                model_info["params"] = {}
            available_models.append(model_info)
        except Exception as e:
            st.warning(f"加载模型信息失败 {model_path.name}: {str(e)}")

    st.markdown(f"**找到 {len(available_models)} 个模型**")

    model_df = pd.DataFrame(available_models)
    if len(model_df) > 0:
        display_cols = ["name", "modified"]
        if "auc" in model_df.columns:
            display_cols.append("auc")
        if "f1" in model_df.columns:
            display_cols.append("f1")
        st.dataframe(
            model_df[display_cols].rename(
                columns={
                    "name": "模型名称",
                    "modified": "修改时间",
                    "auc": "ROC-AUC",
                    "f1": "F1 分数",
                }
            ),
            use_container_width=True,
            height=200,
        )

    st.markdown("---")
    st.markdown("**📊 测试数据准备**")

    if st.session_state.get("training_results") is not None:
        training_results = st.session_state.training_results

        if "X_test" in training_results and "y_test" in training_results:
            st.info("✅ 使用训练时保存的测试集（已预处理）")
            X_test = training_results["X_test"]
            y_test = training_results["y_test"]
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            col1, col2 = st.columns(2)
            with col1:
                st.metric("测试集样本数", len(y_test))
            with col2:
                y_mean = y_test.mean() if hasattr(y_test, "mean") else np.mean(y_test)
                st.metric("正样本比例", f"{y_mean:.2%}")
        else:
            st.warning("训练结果中未找到测试集数据，将使用当前数据重新划分")
            df = st.session_state.uploaded_data
            config_df = st.session_state.field_config
            label_col = config_df[config_df["field_type"] == "label"]["field_name"].values[0]
            key_col = config_df[config_df["field_type"] == "key"]["field_name"].values

            if len(key_col) > 0:
                key_col = key_col[0]
                feature_cols = [c for c in df.columns if c not in [label_col, key_col]]
            else:
                feature_cols = [c for c in df.columns if c != label_col]

            X = df[feature_cols]
            y = df[label_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            col1, col2 = st.columns(2)
            with col1:
                st.metric("测试集样本数", len(y_test))
            with col2:
                y_mean = y_test.mean() if hasattr(y_test, "mean") else np.mean(y_test)
                st.metric("正样本比例", f"{y_mean:.2%}")
    else:
        st.info("使用当前上传的数据划分测试集")
        df = st.session_state.uploaded_data
        config_df = st.session_state.field_config
        label_col = config_df[config_df["field_type"] == "label"]["field_name"].values[0]
        key_col = config_df[config_df["field_type"] == "key"]["field_name"].values

        if len(key_col) > 0:
            key_col = key_col[0]
            feature_cols = [c for c in df.columns if c not in [label_col, key_col]]
        else:
            feature_cols = [c for c in df.columns if c != label_col]

        X = df[feature_cols]
        y = df[label_col]
        test_size = st.slider(
            "测试集比例", 0.1, 0.5, 0.3, 0.05, key="mm_cmp_test_size"
        )
        random_state = st.number_input(
            "随机种子", 0, 10000, 42, key="mm_cmp_random_state"
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        col1, col2 = st.columns(2)
        with col1:
            st.metric("测试集样本数", len(y_test))
        with col2:
            y_mean = y_test.mean() if hasattr(y_test, "mean") else np.mean(y_test)
            st.metric("正样本比例", f"{y_mean:.2%}")

    st.markdown("---")

    if st.session_state.get("X_test") is not None and st.session_state.get("y_test") is not None:
        X_test_for_compare = st.session_state.X_test
        y_test_for_compare = st.session_state.y_test

        if isinstance(X_test_for_compare, pd.DataFrame):
            X_test_for_compare = _encode_test_data(X_test_for_compare)

        if isinstance(y_test_for_compare, pd.Series) and y_test_for_compare.dtype == "object":
            y_test_for_compare = LabelEncoder().fit_transform(y_test_for_compare)

        render_model_comparison_section(
            available_models, X_test_for_compare, y_test_for_compare
        )

    if st.session_state.get("comparison_results") is not None:
        st.markdown("---")
        st.subheader("📋 对比详情")

        results = st.session_state.comparison_results
        st.markdown("**详细性能指标**")

        for model_name, result in results.items():
            with st.expander(f"📊 {model_name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ROC-AUC", f"{result['auc']:.6f}")
                    st.metric("精确率", f"{result['precision']:.6f}")
                with col2:
                    st.metric("F1 分数", f"{result['f1']:.6f}")
                    st.metric("召回率", f"{result['recall']:.6f}")
                with col3:
                    st.metric("准确率", f"{result['accuracy']:.6f}")
                    st.metric("AUC-PR", f"{result['auc_pr']:.6f}")

                st.write("**混淆矩阵:**")
                cm_df = pd.DataFrame(
                    result["confusion_matrix"],
                    index=["真实负", "真实正"],
                    columns=["预测负", "预测正"],
                )
                st.dataframe(cm_df, use_container_width=True)

    st.markdown("---")
    st.markdown("**⚡ 快捷操作**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "🔄 刷新模型列表", use_container_width=True, key="mm_cmp_refresh_btn"
        ):
            if "comparison_results" in st.session_state:
                del st.session_state.comparison_results
            st.rerun()
    with col2:
        st.caption("💡 训练新模型请切换到「4️⃣ 模型训练」Tab")
