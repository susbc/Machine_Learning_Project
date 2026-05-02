"""
tab_train.py — 模型训练 Tab

迁移自：web_app/pages/4_模型训练.py（639 行）
本次仅做机械迁移：
  - 删除 set_page_config / init_session_state / sys.path / st.title
  - 包进 def render()
  - widget key 加 train_ 前缀
  - 跨页面跳转改为同页面 Tab 提示
不做行为/算法/参数变更。
"""

import json

import streamlit as st


def render() -> None:
    """渲染模型训练 Tab。"""
    import pandas as pd
    import numpy as np
    import joblib
    from datetime import datetime
    from web_app.utils.session_state import add_training_log, get_training_logs
    from web_app.components.progress import render_training_progress
    from web_app.components.charts import (
        render_roc_curve,
        render_precision_recall_curve,
        render_confusion_matrix,
        render_feature_importance,
    )

    # LLM 推荐参数「应用」前置：必须在 widget 渲染之前 pop
    if "train_llm_pending_apply" in st.session_state:
        pending = st.session_state.pop("train_llm_pending_apply")
        for k, v in pending.items():
            st.session_state[f"train_{k}"] = v

    # 算法推荐「采纳」前置：必须在 selectbox 渲染之前写 widget key
    if "train_algo_pending_adopt" in st.session_state:
        st.session_state.train_algo_selectbox = st.session_state.pop("train_algo_pending_adopt")

    st.header("🚀 模型训练")
    st.markdown("---")

    # ====== 检查前置条件 ======
    if st.session_state.get("uploaded_data") is None:
        st.warning("⚠️ 请先在「1️⃣ 数据准备」Tab 上传数据")
        return

    if st.session_state.get("field_config") is None:
        st.warning("⚠️ 请先在「2️⃣ 字段配置」Tab 配置字段")
        return

    # 提前取数据/配置，供算法推荐 + LLM 调参 + 训练共用
    df = st.session_state.uploaded_data
    config_df = st.session_state.field_config

    # ====== 算法选择 ======
    st.subheader("🎯 算法选择")

    from web_app.manual_training.param_specs import get_specs, get_supported_algorithms
    from model_code.model_factory import ModelFactory
    from model_code.auto_selector import AutoModelSelector

    algo_options = get_supported_algorithms()
    algo_labels = [ModelFactory.get_display_name(a) for a in algo_options]

    # 默认值（仅在 widget 首次渲染时生效）
    if "train_algo_selectbox" not in st.session_state:
        st.session_state.train_algo_selectbox = algo_labels[0]

    col_algo, col_btn = st.columns([3, 1])
    with col_algo:
        selected_algo_display = st.selectbox(
            "选择算法",
            options=algo_labels,
            key="train_algo_selectbox",
        )
    selected_algo = algo_options[algo_labels.index(selected_algo_display)]
    st.session_state.train_algorithm = selected_algo

    # 智能推荐按钮
    with col_btn:
        st.write("")  # 占位对齐
        recommend_clicked = st.button(
            "🧠 智能推荐",
            use_container_width=True,
            key="train_algo_recommend_btn",
            help="基于数据特征启发式推荐最适合的算法",
        )

    if recommend_clicked:
        with st.spinner("正在分析数据特征并推荐算法..."):
            label_rows = config_df[config_df["field_type"] == "label"]
            if len(label_rows) == 0:
                st.error("⚠️ 配置表中未指定 label 字段，请先在「2️⃣ 字段配置」Tab 配置")
                st.stop()
            label_col = label_rows["field_name"].values[0]
            feature_cols_raw = [c for c in df.columns if c != label_col]
            feature_df = df[feature_cols_raw]
            rec = AutoModelSelector.recommend(feature_df, df[label_col])
            st.session_state.train_algo_recommendation = rec
            st.session_state.train_algo_show_recommendation = True

    if st.session_state.get("train_algo_show_recommendation"):
        rec = st.session_state.train_algo_recommendation
        analysis = rec.get("analysis", {})
        with st.expander("🧠 算法推荐结果", expanded=True):
            # 顶部：推荐算法 + 主理由
            st.success(
                f"### 🏆 推荐算法：**{ModelFactory.get_display_name(rec['recommended'])}**\n\n"
                f"**推荐理由**：{rec['reason']}"
            )

            # 中部：数据画像
            st.markdown("#### 📊 数据画像")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("样本数", f"{analysis.get('n_samples', 0):,}")
            with col_b:
                st.metric("特征数", f"{analysis.get('n_features', 0)}")
            with col_c:
                st.metric("类别特征", f"{analysis.get('n_categorical', 0)}")
            with col_d:
                imb = analysis.get("imbalance_ratio", 1.0)
                st.metric("不平衡比", f"{imb:.3f}", help="少数类/多数类，越小越不平衡")

            spar = analysis.get("sparsity", 0.0)
            st.caption(f"特征稀疏度（零值占比）：{spar:.1%}")

            # 候选算法排行
            if rec.get("candidates"):
                st.markdown("#### 📋 候选算法排行")
                candidate_explanations = {
                    "lightgbm": "训练速度快、内存占用低，适合大规模数据与稀疏特征",
                    "xgboost": "精度高、正则化能力强，可调参空间大",
                    "catboost": "原生支持类别特征，无需独热编码，类别多时优势明显",
                    "random_forest": "稳定不易过拟合，可解释性好，适合中小数据集",
                    "logistic_regression": "简单可解释，适合线性可分问题，可作为基线模型",
                }
                rank_rows = []
                for i, algo in enumerate(rec["candidates"], 1):
                    medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"{i}."
                    rank_rows.append({
                        "排名": medal,
                        "算法": ModelFactory.get_display_name(algo),
                        "适用性说明": candidate_explanations.get(algo, ""),
                    })
                import pandas as _pd
                st.dataframe(_pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)

            # 操作按钮
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                if st.button("✅ 采纳推荐", use_container_width=True, type="primary", key="train_algo_adopt_btn"):
                    # 通过 pending key 在下一轮 rerun 顶部写入 selectbox（widget 已渲染，不能直接改）
                    st.session_state.train_algo_pending_adopt = ModelFactory.get_display_name(rec["recommended"])
                    st.session_state.train_algorithm = rec["recommended"]
                    st.session_state.train_algo_show_recommendation = False
                    st.rerun()
            with col_rec2:
                if st.button("❌ 关闭", use_container_width=True, key="train_algo_close_btn"):
                    st.session_state.train_algo_show_recommendation = False
                    st.rerun()

    st.markdown("---")

    # ====== 动态参数 widgets ======
    st.subheader("⚙️ 模型参数配置")

    with st.expander(f"{ModelFactory.get_display_name(selected_algo)} 参数设置", expanded=True):
        specs = get_specs(selected_algo)
        collected_params = {}
        cols = st.columns(3)
        for i, spec in enumerate(specs):
            with cols[i % 3]:
                widget_key = f"train_{spec.name}"
                current = st.session_state.model_params.get(spec.name, spec.default)
                if spec.widget == "slider":
                    v = st.slider(
                        spec.label,
                        min_value=spec.min_value,
                        max_value=spec.max_value,
                        value=current,
                        step=spec.step,
                        format=spec.fmt,
                        help=spec.help,
                        key=widget_key,
                    )
                elif spec.widget == "select":
                    idx = spec.options.index(current) if current in spec.options else 0
                    v = st.selectbox(
                        spec.label,
                        options=spec.options,
                        index=idx,
                        help=spec.help,
                        key=widget_key,
                    )
                elif spec.widget == "number_input":
                    v = st.number_input(
                        spec.label,
                        value=current,
                        min_value=spec.min_value or 0,
                        max_value=spec.max_value,
                        step=spec.step,
                        help=spec.help,
                        key=widget_key,
                    )
                elif spec.widget == "checkbox":
                    v = st.checkbox(
                        spec.label,
                        value=current,
                        help=spec.help,
                        key=widget_key,
                    )
                collected_params[spec.name] = v

    # 训练配置
    st.markdown("**📊 训练配置**")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider(
            "测试集比例",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            format="%.2f",
            key="train_test_size",
        )
    with col2:
        cv_folds = st.slider(
            "交叉验证折数",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            key="train_cv_folds",
        )
    with col3:
        random_state = st.number_input(
            "随机种子",
            min_value=0,
            max_value=10000,
            value=42,
            step=1,
            key="train_random_state",
        )

    use_grid_search = st.checkbox(
        "启用随机网格搜索调参",
        value=True,
        help="启用后会尝试不同的参数组合，寻找最优参数",
        key="train_use_grid_search",
    )

    if use_grid_search:
        n_iter = st.slider(
            "随机搜索迭代次数",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            key="train_n_iter",
        )
    else:
        n_iter = 1

    st.session_state.model_params = collected_params

    # ====== LLM 一键调参区块 ======
    st.markdown("---")
    st.subheader("🤖 LLM 一键调参")

    from web_app.manual_training.llm_tuning import recommend_params, LLMTuningError

    llm_state = st.session_state.get("train_llm_state", "idle")
    llm_rec = st.session_state.get("train_llm_recommendation")

    if llm_state == "idle" or llm_state == "error":
        if st.button("🤖 LLM 推荐参数", use_container_width=True, key="train_llm_request_btn"):
            from model_code.llm.llm_connector import get_llm_manager
            llm_mgr = get_llm_manager()
            if not llm_mgr.providers:
                st.warning("⚠️ 请先在「12_大模型配置管理」页面配置 LLM 提供商")
            else:
                st.session_state.train_llm_state = "loading"
                st.rerun()

    if llm_state == "loading":
        with st.spinner("正在调用 LLM 分析并推荐参数..."):
            try:
                label_col_llm = st.session_state.field_config[
                    st.session_state.field_config["field_type"] == "label"
                ]["field_name"].values[0]
                result = recommend_params(
                    df=df,
                    target_col=label_col_llm,
                    model_type=selected_algo,
                    eda_summary=st.session_state.get("eda_summary_cache"),
                    last_training=st.session_state.get("training_results"),
                )
                st.session_state.train_llm_recommendation = result
                st.session_state.train_llm_state = "has_recommendation"
            except LLMTuningError as e:
                st.session_state.train_llm_error = str(e)
                st.session_state.train_llm_raw_backup = ""
                st.session_state.train_llm_state = "error"
            except Exception as e:
                st.session_state.train_llm_error = f"调用 LLM 失败：{e}"
                st.session_state.train_llm_state = "error"
            st.rerun()

    if llm_state == "has_recommendation" and llm_rec:
        with st.expander("🤖 LLM 推荐结果", expanded=True):
            st.info(f"**推荐理由**\n\n{llm_rec['reasoning']}")
            st.code(json.dumps(llm_rec["params"], indent=2, ensure_ascii=False), language="json")
            if llm_rec.get("warnings"):
                for w in llm_rec["warnings"]:
                    st.warning(f"⚠️ {w}")
            col_llm1, col_llm2 = st.columns(2)
            with col_llm1:
                if st.button("✅ 应用到参数控件", use_container_width=True, key="train_llm_apply_btn"):
                    st.session_state.train_llm_pending_apply = llm_rec["params"]
                    st.session_state.train_llm_state = "idle"
                    st.session_state.train_llm_recommendation = None
                    st.rerun()
            with col_llm2:
                if st.button("🗑️ 清除推荐", use_container_width=True, key="train_llm_clear_btn"):
                    st.session_state.train_llm_state = "idle"
                    st.session_state.train_llm_recommendation = None
                    st.rerun()

    # ====== 训练控制区域 ======
    st.markdown("---")
    st.subheader("🚀 训练控制")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.info(
            f"""
        📊 训练数据：{len(df):,} 行 × {len(df.columns)} 列
        ⚙️ 配置字段：{len(config_df)} 个
        🎯 标签字段：{config_df[config_df['field_type']=='label']['field_name'].values[0] if 'label' in config_df['field_type'].values else '未指定'}
        """
        )

    with col2:
        start_training = st.button(
            "🚀 开始训练",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("training_in_progress", False),
            key="train_start_btn",
        )

    with col3:
        if st.button(
            "🗑️ 清除训练结果",
            use_container_width=True,
            disabled=st.session_state.get("training_results") is None,
            key="train_clear_btn",
        ):
            st.session_state.training_results = None
            st.session_state.training_logs = []
            st.session_state.training_progress = 0
            st.rerun()

    # ====== 训练执行 ======
    if start_training:
        st.session_state.training_in_progress = True
        st.session_state.training_logs = []
        st.session_state.training_progress = 0

        try:
            add_training_log("=" * 60)
            add_training_log("开始模型训练流程")
            add_training_log("=" * 60)
            add_training_log(f"训练样本数：{len(df):,}")
            add_training_log(f"特征数：{len(df.columns) - 1}")
            add_training_log(f"参数配置：{st.session_state.model_params}")

            add_training_log("\n正在准备训练数据...")
            st.session_state.training_progress = 0.1

            label_col = config_df[config_df["field_type"] == "label"]["field_name"].values
            key_col = config_df[config_df["field_type"] == "key"]["field_name"].values

            if len(label_col) == 0:
                raise ValueError("配置表中未指定 label 字段")

            label_col = label_col[0]
            key_col = key_col[0] if len(key_col) > 0 else None

            add_training_log(f"标签字段：{label_col}")
            add_training_log(f"ID 字段：{key_col or '未指定'}")

            add_training_log("\n正在执行数据预处理...")
            st.session_state.training_progress = 0.2

            from model_code.utils import (
                handle_dataframe_outliers,
                handle_dataframe_missing_values,
                encode_dataframe,
                standardize_dataframe,
            )

            train_data = df.copy()

            train_data = handle_dataframe_outliers(train_data, config_df)
            st.session_state.training_progress = 0.3

            train_data = handle_dataframe_missing_values(train_data, config_df)
            st.session_state.training_progress = 0.4

            train_data = encode_dataframe(train_data, config_df)
            st.session_state.training_progress = 0.5

            train_data = standardize_dataframe(train_data, config_df)
            st.session_state.training_progress = 0.6

            feature_cols = [c for c in train_data.columns if c not in [label_col, key_col] if key_col]
            if key_col is None:
                feature_cols = [c for c in train_data.columns if c != label_col]

            X = train_data[feature_cols]
            y = train_data[label_col]

            if y.dtype == "object":
                y = (y == y.mode()[0]).astype(int)
                add_training_log("Label 已编码为数值")

            add_training_log(f"特征矩阵：{X.shape}")
            add_training_log(f"标签向量：{y.shape}")

            if y.dtype in ["int64", "float64"]:
                add_training_log(f"正样本比例：{y.mean():.2%}")
            else:
                add_training_log(f"Label 唯一值：{y.unique()}")

            st.session_state.training_progress = 0.7

            add_training_log(f"\n使用 {ModelFactory.get_display_name(selected_algo)} 模型训练...")

            try:
                base_model = ModelFactory.build(
                    algorithm=selected_algo,
                    task_type="binary",
                    params=st.session_state.model_params,
                )
            except ImportError as e:
                add_training_log(f"\n❌ {selected_algo} 未安装：{e}")
                st.error(f"{ModelFactory.get_display_name(selected_algo)} 未安装")
                if selected_algo == "xgboost":
                    st.code("pip install xgboost")
                elif selected_algo == "catboost":
                    st.code("pip install catboost")
                st.session_state.training_in_progress = False
                return
            except Exception as e:
                add_training_log(f"\n❌ 模型构建失败：{e}")
                st.error(f"模型构建失败：{e}")
                st.session_state.training_in_progress = False
                return

            from sklearn.model_selection import train_test_split, RandomizedSearchCV
            from sklearn.metrics import roc_auc_score, precision_recall_curve

            stratify_col = y if y.dtype in ["int64", "float64"] and y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_col,
            )

            add_training_log(f"训练集：{X_train.shape[0]:,} 样本")
            add_training_log(f"测试集：{X_test.shape[0]:,} 样本")

            if use_grid_search and n_iter > 1:
                add_training_log(f"\n执行随机网格搜索 (n_iter={n_iter}, cv={cv_folds})...")
                param_grid = ModelFactory.get_param_grid(selected_algo)
                random_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=0,
                    random_state=random_state,
                )
                random_search.fit(X_train, y_train)
                model = random_search.best_estimator_
                add_training_log(f"最佳参数：{random_search.best_params_}")
                st.session_state.cv_results = random_search.cv_results_
            else:
                add_training_log("\n使用固定参数训练...")
                model = base_model
                model.fit(X_train, y_train)

            st.session_state.training_progress = 0.85
            add_training_log("\n模型训练完成！")

            add_training_log("\n正在评估模型...")
            st.session_state.training_progress = 0.9

            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            test_auc = roc_auc_score(y_test, y_pred_proba)

            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores[:-1])
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]

            add_training_log(f"\n测试集 ROC-AUC: {test_auc:.6f}")
            add_training_log(f"最佳阈值：{best_threshold:.4f}")
            add_training_log(f"最佳 F1 分数：{best_f1:.6f}")

            st.session_state.training_results = {
                "model": model,
                "algorithm": selected_algo,
                "model_params": st.session_state.model_params,
                "test_auc": test_auc,
                "best_threshold": best_threshold,
                "best_f1": best_f1,
                "feature_names": feature_cols,
                "feature_importances": model.feature_importances_,
                "X_test": X_test,
                "y_test": y_test.values,
                "y_pred_proba": y_pred_proba,
                "y_pred": y_pred,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "timestamp": datetime.now().isoformat(),
            }

            model_path = (
                st.session_state.model_dir
                / f"model_{selected_algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
            joblib.dump(model, model_path)
            add_training_log(f"\n✅ 模型已保存：{model_path}")

            summary_path = (
                st.session_state.model_dir
                / f"train_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
            joblib.dump(st.session_state.training_results, summary_path)
            add_training_log(f"✅ 训练摘要已保存：{summary_path}")

            st.session_state.training_progress = 1.0
            add_training_log("\n" + "=" * 60)
            add_training_log("训练流程完成！")
            add_training_log("=" * 60)

            if "trained_models" not in st.session_state:
                st.session_state.trained_models = []

            st.session_state.trained_models.append(
                {
                    "name": model_path.stem,
                    "path": str(model_path),
                    "auc": test_auc,
                    "f1": best_f1,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            st.session_state.training_in_progress = False
            st.rerun()

        except Exception as e:
            st.session_state.training_in_progress = False
            add_training_log(f"\n❌ 训练失败：{str(e)}")
            st.error(f"训练出错：{str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # ====== 显示训练进度和结果 ======
    if st.session_state.get("training_in_progress", False):
        render_training_progress(
            st.session_state.get("training_progress", 0),
            get_training_logs(),
        )

    if st.session_state.get("training_results") is not None:
        st.markdown("---")
        st.subheader("📊 训练结果")

        results = st.session_state.training_results

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROC-AUC", f"{results['test_auc']:.4f}")
        with col2:
            st.metric("最佳 F1", f"{results['best_f1']:.4f}")
        with col3:
            st.metric("最佳阈值", f"{results['best_threshold']:.4f}")
        with col4:
            st.metric("测试集大小", f"{results['test_size']:,}")

        st.markdown("**📈 模型评估**")

        train_res_t1, train_res_t2, train_res_t3, train_res_t4 = st.tabs(
            ["ROC 曲线", "PR 曲线", "混淆矩阵", "特征重要性"]
        )

        with train_res_t1:
            fig = render_roc_curve(results["y_test"], results["y_pred_proba"])
            st.plotly_chart(fig, use_container_width=True, key="train_plot_roc")

        with train_res_t2:
            fig = render_precision_recall_curve(results["y_test"], results["y_pred_proba"])
            st.plotly_chart(fig, use_container_width=True, key="train_plot_pr")

        with train_res_t3:
            fig = render_confusion_matrix(results["y_test"], results["y_pred"])
            st.plotly_chart(fig, use_container_width=True, key="train_plot_cm")

        with train_res_t4:
            fig = render_feature_importance(
                results["feature_names"], results["feature_importances"]
            )
            st.plotly_chart(fig, use_container_width=True, key="train_plot_fi")

        st.markdown("**📋 分类报告**")
        from sklearn.metrics import classification_report

        report = classification_report(
            results["y_test"], results["y_pred"], output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)

        st.markdown("**💾 导出结果**")
        col1, col2 = st.columns(2)

        with col1:
            pred_df = pd.DataFrame(
                {
                    "真实标签": results["y_test"],
                    "预测概率": results["y_pred_proba"],
                    "预测标签": results["y_pred"],
                }
            )
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 下载预测结果 (CSV)",
                data=csv,
                file_name=f"train_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="train_dl_predictions_btn",
            )

        with col2:
            importance_df = pd.DataFrame(
                {
                    "特征名": results["feature_names"],
                    "重要性": results["feature_importances"],
                }
            ).sort_values("重要性", ascending=False)
            csv = importance_df.to_csv(index=False)
            st.download_button(
                label="📥 下载特征重要性 (CSV)",
                data=csv,
                file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="train_dl_importance_btn",
            )
