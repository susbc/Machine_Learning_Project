"""
Web 应用组件包
"""

from .data_upload import (
    render_data_upload_section,
    render_data_preview,
    render_data_statistics,
    render_field_names,
    render_data_management_page,
)

from .config_editor import (
    render_config_editor,
    render_config_actions,
    render_field_type_distribution,
    render_config_summary,
    auto_infer_config,
    render_config_management_page,
)

from .charts import (
    render_distribution_chart,
    render_correlation_heatmap,
    render_missing_value_heatmap,
    render_outlier_boxplot,
    render_roc_curve,
    render_precision_recall_curve,
    render_confusion_matrix,
    render_feature_importance,
    render_prediction_distribution,
    render_threshold_analysis,
)

from .progress import (
    render_progress_bar,
    render_training_progress,
    StreamlitLogger,
    render_cv_results,
    render_training_summary,
    render_prediction_progress,
)

__all__ = [
    # Data upload
    'render_data_upload_section',
    'render_data_preview',
    'render_data_statistics',
    'render_field_names',
    'render_data_management_page',
    
    # Config editor
    'render_config_editor',
    'render_config_actions',
    'render_field_type_distribution',
    'render_config_summary',
    'auto_infer_config',
    'render_config_management_page',
    
    # Charts
    'render_distribution_chart',
    'render_correlation_heatmap',
    'render_missing_value_heatmap',
    'render_outlier_boxplot',
    'render_roc_curve',
    'render_precision_recall_curve',
    'render_confusion_matrix',
    'render_feature_importance',
    'render_prediction_distribution',
    'render_threshold_analysis',
    
    # Progress
    'render_progress_bar',
    'render_training_progress',
    'StreamlitLogger',
    'render_cv_results',
    'render_training_summary',
    'render_prediction_progress',
]
