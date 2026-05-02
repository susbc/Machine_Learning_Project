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

    st.header("🚀 模型训练")
    st.markdown("---")

    # ====== 检查前置条件 ======
    if st.session_state.get("uploaded_data") is None:
        st.warning("⚠️ 请先在「1️⃣ 数据准备」Tab 上传数据")
        return

    if st.session_state.get("field_config") is None:
        st.warning("⚠️ 请先在「2️⃣ 字段配置」Tab 配置字段")
        return

    # ====== 参数配置区域 ======
    st.subheader("🎯 模型参数配置")

    with st.expander("⚙️ LightGBM 参数设置", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            n_estimators = st.slider(
                "n_estimators (树的数量)",
                min_value=50,
                max_value=1000,
                value=st.session_state.model_params.get("n_estimators", 200),
                step=50,
                key="train_n_estimators",
            )

            learning_rate = st.slider(
                "learning_rate (学习率)",
                min_value=0.001,
                max_value=0.3,
                value=st.session_state.model_params.get("learning_rate", 0.1),
                step=0.01,
                format="%.3f",
                key="train_learning_rate",
            )

            max_depth = st.slider(
                "max_depth (树的最大深度)",
                min_value=3,
                max_value=15,
                value=st.session_state.model_params.get("max_depth", 6),
                step=1,
                key="train_max_depth",
            )

        with col2:
            num_leaves = st.slider(
                "num_leaves (叶子节点数)",
                min_value=10,
                max_value=200,
                value=st.session_state.model_params.get("num_leaves", 50),
                step=10,
                key="train_num_leaves",
            )

            min_child_samples = st.slider(
                "min_child_samples (叶子最小样本数)",
                min_value=5,
                max_value=100,
                value=st.session_state.model_params.get("min_child_samples", 20),
                step=5,
                key="train_min_child_samples",
            )

            subsample = st.slider(
                "subsample (样本采样比例)",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.model_params.get("subsample", 0.9),
                step=0.05,
                format="%.2f",
                key="train_subsample",
            )

        with col3:
            colsample_bytree = st.slider(
                "colsample_bytree (特征采样比例)",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.model_params.get("colsample_bytree", 0.9),
                step=0.05,
                format="%.2f",
                key="train_colsample_bytree",
            )

            reg_alpha = st.slider(
                "reg_alpha (L1 正则化)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.model_params.get("reg_alpha", 0.1),
                step=0.1,
                format="%.1f",
                key="train_reg_alpha",
            )

            reg_lambda = st.slider(
                "reg_lambda (L2 正则化)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.model_params.get("reg_lambda", 0.1),
                step=0.1,
                format="%.1f",
                key="train_reg_lambda",
            )

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

    st.session_state.model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
    }

    # ====== 训练控制区域 ======
    st.markdown("---")
    st.subheader("🚀 训练控制")

    col1, col2, col3 = st.columns([2, 1, 1])

    df = st.session_state.uploaded_data
    config_df = st.session_state.field_config

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

            try:
                import lightgbm as lgb

                use_lgbm = True
                add_training_log("\n使用 LightGBM 模型训练...")
            except Exception:
                use_lgbm = False
                add_training_log("\n⚠️ LightGBM 不可用，使用 RandomForest 模型...")
                add_training_log("💡 提示：安装 LightGBM 需要系统库支持，或使用 RandomForest")

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

            if use_lgbm:
                model = lgb.LGBMClassifier(
                    objective="binary",
                    metric="auc",
                    verbose=-1,
                    random_state=random_state,
                    n_jobs=-1,
                    **st.session_state.model_params,
                )

                if use_grid_search and n_iter > 1:
                    add_training_log(f"\n执行随机网格搜索 (n_iter={n_iter}, cv={cv_folds})...")
                    param_grid = {
                        "n_estimators": [100, 200, 300],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "max_depth": [5, 6, 7, 8],
                        "num_leaves": [30, 50, 70],
                        "min_child_samples": [10, 20, 30],
                    }
                    random_search = RandomizedSearchCV(
                        estimator=model,
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
                    model.fit(X_train, y_train)
            else:
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=random_state,
                    n_jobs=-1,
                )

                if use_grid_search and n_iter > 1:
                    add_training_log(f"\n执行随机网格搜索 (n_iter={n_iter}, cv={cv_folds})...")
                    param_grid = {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [5, 10, 15, None],
                        "min_samples_split": [2, 5, 10],
                    }
                    random_search = RandomizedSearchCV(
                        estimator=model,
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
                / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
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
