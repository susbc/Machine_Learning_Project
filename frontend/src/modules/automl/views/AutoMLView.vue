<template>
  <section class="automl-workspace">
    <div class="section-heading automl-heading">
      <div>
        <p class="eyebrow">Phase 1</p>
        <h2>AutoML 训练</h2>
        <p>上传数据后自动完成目标识别、模型推荐、Optuna 参数搜索、评估和模型保存。</p>
      </div>
      <span class="status-chip" :class="`status-chip--${statusTagType}`">
        <Clock :size="13" :stroke-width="2" />{{ statusLabel }}
      </span>
    </div>

    <div class="automl-grid">
      <section class="setup-panel">
        <div class="panel-title">
          <h3><Folder class="panel-title-icon" :size="17" :stroke-width="1.9" />训练输入</h3>
          <span>{{ datasetFile?.name ?? '未选择数据集' }}</span>
        </div>

        <el-upload
          class="dataset-upload"
          drag
          accept=".csv,.xlsx,.xls"
          :auto-upload="false"
          :limit="1"
          :on-change="handleFileChange"
          :on-remove="handleFileRemove"
        >
          <div class="upload-copy">
            <CloudUpload class="upload-icon" :size="44" :stroke-width="1.4" />
            <strong>拖拽文件到此处，或 <em>点击上传</em></strong>
            <span>支持 CSV / Excel 数据集</span>
            <small>上传后将自动完成特征编码和目标检查</small>
          </div>
        </el-upload>
        <div class="file-bar">
          <el-button class="file-bar-button" plain @click="triggerFilePicker">
            <Folder :size="15" :stroke-width="1.9" />选择文件
          </el-button>
          <span class="file-bar-name">{{ datasetFile?.name ?? '未选择任何文件' }}</span>
        </div>
        <input
          ref="nativeFileInput"
          data-testid="automl-file-input"
          class="native-file-input native-file-input--hidden"
          type="file"
          accept=".csv,.xlsx,.xls"
          @change="handleNativeFileChange"
        />

        <div v-if="capabilityStatusMessage" class="capability-alert" role="alert">
          <span>{{ capabilityStatusMessage }}</span>
          <el-button text type="primary" :loading="capabilitiesLoading" @click="loadCapabilities">重试加载能力</el-button>
        </div>

        <div class="form-grid">
          <label>
            <span class="field-label"><Crosshair :size="14" :stroke-width="1.9" />问题类型</span>
            <el-select v-model="problemType" aria-label="问题类型" :loading="capabilitiesLoading" @change="applyProblemType">
              <el-option label="自动推断" value="auto" />
              <el-option label="二分类" value="binary_classification" />
              <el-option label="多分类" value="multiclass_classification" />
              <el-option label="回归" value="regression" />
            </el-select>
          </label>

          <label>
            <span class="field-label"><Target :size="14" :stroke-width="1.9" />目标列</span>
            <el-input v-model="targetCol" placeholder="留空则自动识别 label / target / 最后一列" clearable />
          </label>

          <label>
            <span class="field-label"><ChartColumn :size="14" :stroke-width="1.9" />优化指标</span>
            <el-select v-model="optimizationMetric" aria-label="优化指标" :disabled="problemType === 'auto' || explicitActionsBlocked">
              <el-option v-if="problemType === 'auto'" label="自动选择（后端）" value="auto" />
              <el-option
                v-for="metric in compatibleMetrics"
                :key="metric"
                :label="metricOptionLabel(metric)"
                :value="metric"
              />
            </el-select>
          </label>

          <label>
            <span class="field-label"><Layers :size="14" :stroke-width="1.9" />Trial 次数</span>
            <el-input-number v-model="nTrials" :min="1" :max="100" :step="1" />
          </label>

          <label>
            <span class="field-label"><Grid2x2 :size="14" :stroke-width="1.9" />交叉验证</span>
            <el-input-number v-model="cvFolds" :min="2" :max="10" :step="1" />
          </label>
        </div>

        <label class="full-field">
          <span class="field-label"><Box :size="14" :stroke-width="1.9" />候选模型</span>
          <el-select
            v-model="candidateModels"
            aria-label="候选模型"
            multiple
            collapse-tags
            collapse-tags-tooltip
            :disabled="problemType === 'auto' || explicitActionsBlocked"
            :placeholder="problemType === 'auto' ? '自动模式由后端推荐' : '选择兼容模型'"
          >
            <el-option
              v-for="model in compatibleModels"
              :key="model"
              :label="modelLabel(model)"
              :value="model"
            />
          </el-select>
        </label>
        <el-alert v-if="selectionResetMessage" type="warning" :title="selectionResetMessage" show-icon :closable="false" />

        <div class="full-field advanced-options">
          <span class="advanced-title"><Zap :size="14" :stroke-width="1.9" />自动迭代增强（可选）</span>

          <div class="option-row">
            <span class="option-label"><SlidersHorizontal :size="14" :stroke-width="1.9" /><b>特征选择</b><small>随机森林重要性 + 中位数阈值，逐折重新拟合</small></span>
            <el-switch v-model="enableFeatureSelection" data-testid="automl-feature-selection" />
          </div>

          <div class="option-row">
            <span class="option-label"><ChartPie :size="14" :stroke-width="1.9" /><b>智能采样</b><small>二分类负样本降采样，仅作用训练集</small></span>
            <el-switch v-model="enableSampling" data-testid="automl-sampling" />
          </div>

          <div v-if="enableSampling" class="form-grid">
            <label>
              <span class="field-label"><ChartPie :size="14" :stroke-width="1.9" />负样本保留比例</span>
              <el-input-number
                v-model="samplingNegativeRatio"
                :min="0.01"
                :max="1"
                :step="0.05"
                :precision="2"
              />
            </label>
          </div>

          <div class="form-grid">
            <label>
              <span class="field-label"><Timer :size="14" :stroke-width="1.9" />早停轮数（0 为关闭）</span>
              <el-input-number v-model="earlyStoppingRounds" :min="0" :max="50" :step="1" />
            </label>
            <label>
              <span class="field-label"><Gauge :size="14" :stroke-width="1.9" />最小改进幅度</span>
              <el-input-number
                v-model="minImprovement"
                :min="0"
                :max="0.5"
                :step="0.001"
                :precision="4"
                :disabled="earlyStoppingRounds === 0"
              />
            </label>
          </div>
        </div>

      </section>

      <section class="run-panel">
        <div class="panel-title">
          <h3><ChartColumn class="panel-title-icon" :size="17" :stroke-width="1.9" />训练进度</h3>
          <span>{{ job?.job_id ?? '等待创建任务' }}</span>
        </div>

        <div class="metric-grid">
          <div class="metric metric--blue">
            <span class="metric-icon"><Target :size="18" :stroke-width="1.9" /></span>
            <div class="metric-body">
              <span>进度</span>
              <strong>{{ job?.progress ?? 0 }}%</strong>
            </div>
          </div>
          <div class="metric metric--teal">
            <span class="metric-icon"><Layers :size="18" :stroke-width="1.9" /></span>
            <div class="metric-body">
              <span>Trial</span>
              <strong>{{ job?.current_trial ?? 0 }}/{{ job?.n_trials ?? nTrials }}</strong>
            </div>
          </div>
          <div class="metric metric--amber">
            <span class="metric-icon"><Trophy :size="18" :stroke-width="1.9" /></span>
            <div class="metric-body">
              <span>最佳分数</span>
              <strong>{{ bestValueLabel }}</strong>
            </div>
          </div>
          <div class="metric metric--purple">
            <span class="metric-icon"><Box :size="18" :stroke-width="1.9" /></span>
            <div class="metric-body">
              <span>最佳模型</span>
              <strong class="metric-model">{{ result ? modelLabel(result.best_model) : '-' }}</strong>
            </div>
          </div>
        </div>

        <div class="progress-row">
          <el-progress
            class="progress-bar"
            :percentage="job?.progress ?? 0"
            :show-text="false"
            :status="job?.status === 'completed' ? 'success' : job?.status === 'failed' ? 'exception' : undefined"
            :stroke-width="12"
            striped
            striped-flow
          />
          <span class="progress-value">{{ job?.progress ?? 0 }}%</span>
        </div>

        <div class="chart-card">
          <div class="chart-head">
            <span class="chart-title">{{ chartMetricLabel }}</span>
            <div class="chart-legend" aria-label="图例">
              <span><i class="legend-dot"></i>Trial 分数</span>
              <span><i class="legend-line"></i>历史最佳</span>
            </div>
          </div>
          <div ref="chartRef" class="automl-chart"></div>
        </div>

        <div v-if="activeProblemSpec" class="problem-spec-card" aria-live="polite">
          <strong>已解析任务：{{ problemTypeLabel(activeProblemSpec.resolved_type) }}</strong>
          <span>来源：{{ problemSourceLabel(activeProblemSpec.source) }}</span>
          <span>依据：{{ activeProblemSpec.reason }}</span>
          <span>说明：{{ activeProblemSpec.explanation }}</span>
          <span>目标列 {{ activeProblemSpec.target.column }}（{{ activeProblemSpec.target.dtype }}）</span>
          <span>有效行 {{ activeProblemSpec.target.valid_rows }}</span>
          <span>唯一值 {{ activeProblemSpec.target.unique_count }}</span>
          <span>唯一值比例 {{ percentLabel(activeProblemSpec.target.unique_ratio) }}</span>
          <span>整数样式 {{ activeProblemSpec.target.integer_like ? '是' : '否' }}</span>
        </div>

        <el-alert
          v-if="errorMessage || job?.error"
          class="status-alert"
          type="error"
          :title="errorMessage || job?.error || ''"
          show-icon
          :closable="false"
        />

        <div v-if="result" class="result-box">
          <div class="panel-title">
            <h3>最佳结果</h3>
            <el-button type="success" plain :loading="savingResult" @click="saveResult">保存模型</el-button>
          </div>
          <div class="result-grid">
            <template v-if="isRegressionResult">
              <span>RMSE：{{ metricLabel('rmse') }}</span>
              <span>MAE：{{ metricLabel('mae') }}</span>
              <span>R²：{{ metricLabel('r2') }}</span>
            </template>
            <template v-else>
              <span>ROC-AUC：{{ metricLabel('roc_auc') }}</span>
              <span>Accuracy：{{ metricLabel('accuracy') }}</span>
              <span>F1：{{ metricLabel('f1') }}</span>
            </template>
          </div>
          <div
            v-if="result.early_stopped || result.feature_selection?.applied || result.sampling?.applied"
            class="enhancement-summary"
          >
            <el-tag v-if="result.early_stopped" type="warning" size="small">已触发早停</el-tag>
            <el-tag v-if="result.feature_selection?.applied" type="success" size="small">
              特征选择：{{ result.feature_selection.original_feature_count }} →
              {{ result.feature_selection.selected_feature_count }}
            </el-tag>
            <el-tag v-if="result.sampling?.applied" type="success" size="small">
              智能采样：训练集 {{ result.sampling.original_rows }} → {{ result.sampling.sampled_rows }} 条
            </el-tag>
          </div>
          <pre>{{ JSON.stringify(result.best_params, null, 2) }}</pre>
          <el-alert
            v-if="savedArtifactPath"
            type="success"
            :title="`模型已保存：${savedArtifactPath}`"
            show-icon
            :closable="false"
          />
        </div>
      </section>
    </div>

    <section class="execution-panel">
      <div class="panel-title"><h3><FileText class="panel-title-icon" :size="18" />训练设置与操作</h3></div>
      <div class="execution-fields">
        <label class="full-field">
          <span class="field-label"><FileText :size="14" :stroke-width="1.9" />业务场景</span>
          <el-input
            v-model="businessContext"
            type="textarea"
            :rows="2"
            placeholder="例如：信贷风控、流失预测、质量检测"
          />
        </label>
        <div class="option-row">
          <span>调用 LLM 决策</span>
          <el-switch v-model="useLlm" active-text="启用" inactive-text="禁用" />
        </div>
      </div>
      <div class="actions">
        <el-button type="primary" :loading="creatingJob" :disabled="!canStart" @click="startJob">
          开始全自动训练
        </el-button>
        <el-button plain :disabled="!currentJobId" @click="refreshJob">刷新状态</el-button>
        <el-button type="danger" plain :disabled="!canCancel" @click="cancelJob">取消任务</el-button>
      </div>
    </section>

    <section class="history-panel">
      <div class="panel-title">
        <h3>Trial 历史</h3>
        <span>{{ trialHistory.length }} 条记录</span>
      </div>
      <el-table :data="trialHistory" class="history-table" height="340">
        <el-table-column prop="trial_number" label="Trial" width="90" />
        <el-table-column label="模型" width="150">
          <template #default="{ row }">{{ modelLabel(row.model_type) }}</template>
        </el-table-column>
        <el-table-column label="分数" width="120">
          <template #default="{ row }">{{ typeof row.value === 'number' ? row.value.toFixed(5) : '-' }}</template>
        </el-table-column>
        <el-table-column label="历史最佳" width="120">
          <template #default="{ row }">{{ typeof row.best_value === 'number' ? row.best_value.toFixed(5) : '-' }}</template>
        </el-table-column>
        <el-table-column label="参数">
          <template #default="{ row }">
            <el-tag v-for="(value, key) in row.params" :key="key" size="small" class="param-tag">
              {{ key }}: {{ value }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </section>

    <section v-if="job?.decision_summary" class="decision-panel">
      <div class="panel-title">
        <h3>自动决策摘要</h3>
        <span>{{ job.decision_summary.recommendation_source ?? 'auto' }}</span>
      </div>
      <div class="decision-grid">
        <span>目标列：{{ job.decision_summary.target_col }}</span>
        <span>样本数：{{ job.decision_summary.rows }}</span>
        <span>特征数：{{ job.decision_summary.feature_count }}</span>
        <span>LLM：{{ job.decision_summary.llm_status }}</span>
      </div>
      <p>{{ job.decision_summary.recommendation_reason }}</p>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { UploadFile } from 'element-plus'
import { ElMessage } from 'element-plus/es/components/message/index.mjs'
import {
  Box,
  ChartColumn,
  ChartPie,
  Clock,
  CloudUpload,
  Crosshair,
  FileText,
  Folder,
  Gauge,
  Grid2x2,
  Layers,
  SlidersHorizontal,
  Target,
  Timer,
  Trophy,
  Zap,
} from 'lucide-vue-next'
import { init, type ECharts } from '@/shared/charts/echarts'

import {
  automlApi,
  type AutoMLJob,
  type AutoMLMetric,
  type AutoMLModel,
  type AutoMLResult,
  type AutoMLStatus,
  type AutoMLTrialEvent,
} from '@/api/automl'
import { trainingApi } from '@/api/training'
import type { MetricDirection, ProblemCapabilities, ProblemCapabilitiesResponse, ProblemSpec, ProblemType } from '@/api/problem'

const STORAGE_KEY = 'agentic-automl.currentJobId'

const datasetFile = ref<File | null>(null)
const targetCol = ref('')
const problemType = ref<ProblemType>('auto')
const optimizationMetric = ref<AutoMLMetric>('auto')
const nTrials = ref(10)
const cvFolds = ref(3)
const candidateModels = ref<AutoMLModel[]>([])
const businessContext = ref('通用机器学习建模')
const useLlm = ref(true)
const enableFeatureSelection = ref(false)
const enableSampling = ref(false)
const samplingNegativeRatio = ref(0.1)
const earlyStoppingRounds = ref(0)
const minImprovement = ref(0.001)

const job = ref<AutoMLJob | null>(null)
const result = ref<AutoMLResult | null>(null)
const trialHistory = ref<AutoMLTrialEvent[]>([])
const creatingJob = ref(false)
const savingResult = ref(false)
const cancellingRequest = ref(false)
const errorMessage = ref('')
const savedArtifactPath = ref('')
const capabilities = ref<ProblemCapabilitiesResponse | null>(null)
const capabilitiesLoading = ref(false)
const capabilitiesUnavailable = ref(false)
const capabilitiesError = ref('')
const selectionResetMessage = ref('')
const streamController = ref<AbortController | null>(null)
const chartRef = ref<HTMLElement | null>(null)
const nativeFileInput = ref<HTMLInputElement | null>(null)
let chartInstance: ECharts | null = null

function triggerFilePicker() {
  nativeFileInput.value?.click()
}

const currentJobId = computed(() => job.value?.job_id ?? localStorage.getItem(STORAGE_KEY) ?? '')
const explicitCapabilities = computed<ProblemCapabilities | null>(() => {
  if (problemType.value === 'auto') return null
  return capabilities.value?.problem_types?.[problemType.value] ?? null
})
const explicitCapabilitiesValidation = computed(() => validateProblemCapabilities(explicitCapabilities.value))
const compatibleModels = computed<AutoMLModel[]>(() => explicitCapabilitiesValidation.value.valid
  ? explicitCapabilities.value?.algorithms.filter(isAutoMLModel) ?? []
  : [])
const compatibleMetrics = computed<Exclude<AutoMLMetric, 'auto'>[]>(() => explicitCapabilitiesValidation.value.valid
  ? explicitCapabilities.value?.metrics.filter(isExplicitAutoMLMetric) ?? []
  : [])
const explicitSelectionsValid = computed(() => problemType.value === 'auto' || (
  candidateModels.value.length > 0
  && candidateModels.value.every((model) => compatibleModels.value.includes(model))
  && compatibleMetrics.value.includes(optimizationMetric.value as Exclude<AutoMLMetric, 'auto'>)
))
const explicitCapabilitiesReady = computed(() => problemType.value === 'auto' || (
  explicitCapabilitiesValidation.value.valid && explicitSelectionsValid.value
))
const explicitActionsBlocked = computed(() => problemType.value !== 'auto' && (capabilitiesLoading.value || !explicitCapabilitiesReady.value))
const capabilityStatusMessage = computed(() => {
  if (capabilitiesUnavailable.value && capabilitiesError.value) return capabilitiesError.value
  if (problemType.value !== 'auto' && !capabilitiesLoading.value && !explicitCapabilitiesReady.value) {
    return `无法验证显式任务能力：${problemTypeLabel(problemType.value)}${explicitCapabilitiesValidation.value.message}。请重试加载能力。`
  }
  return ''
})
const activeProblemSpec = computed<ProblemSpec | null>(() => result.value?.problem_spec ?? job.value?.problem_spec ?? null)
const isRegressionResult = computed(() => result.value?.problem_spec.resolved_type === 'regression')
const resolvedMetric = computed<AutoMLMetric>(() => {
  const metric = job.value?.optimization_metric ?? result.value?.problem_spec.capabilities.default_metric
  if (isAutoMLMetric(metric)) return metric
  if (problemType.value !== 'auto' && optimizationMetric.value !== 'auto') return optimizationMetric.value
  return 'auto'
})
const resolvedMetricDirection = computed<MetricDirection | null>(() => {
  if (resolvedMetric.value === 'auto') return null
  return activeProblemSpec.value?.capabilities.metric_directions[resolvedMetric.value]
    ?? explicitCapabilities.value?.metric_directions[resolvedMetric.value]
    ?? null
})
const chartMetricLabel = computed(() => metricOptionLabel(resolvedMetric.value, resolvedMetricDirection.value))
const canStart = computed(() => Boolean(datasetFile.value) && !creatingJob.value && !isRunningStatus(job.value?.status) && !explicitActionsBlocked.value)
const canCancel = computed(() => Boolean(job.value && (job.value.status === 'queued' || job.value.status === 'running') && !cancellingRequest.value))
const statusLabel = computed(() => statusText(job.value?.status))
const statusTagType = computed(() => {
  if (job.value?.status === 'completed') return 'success'
  if (job.value?.status === 'failed') return 'danger'
  if (isRunningStatus(job.value?.status)) return 'warning'
  return 'info'
})
const bestValueLabel = computed(() => {
  const value = result.value?.best_value ?? trialHistory.value.at(-1)?.best_value
  return typeof value === 'number' ? value.toFixed(5) : '-'
})

function isRunningStatus(status?: AutoMLStatus): boolean {
  return status === 'queued' || status === 'running' || status === 'cancelling'
}

function statusText(status?: AutoMLStatus): string {
  const labels: Record<AutoMLStatus | 'idle', string> = {
    idle: '待启动',
    queued: '排队中',
    running: '训练中',
    cancelling: '取消中',
    cancelled: '已取消',
    completed: '已完成',
    failed: '失败',
  }
  return labels[status ?? 'idle']
}

function modelLabel(model?: AutoMLModel): string {
  const labels: Record<AutoMLModel, string> = {
    lightgbm: 'LightGBM',
    xgboost: 'XGBoost',
    catboost: 'CatBoost',
    random_forest: '随机森林',
    logistic_regression: '逻辑回归',
    linear_regression: '线性回归',
  }
  return model ? labels[model] : '-'
}

function metricOptionLabel(metric: AutoMLMetric, direction?: MetricDirection | null): string {
  const labels: Record<AutoMLMetric, string> = {
    auto: '自动选择（后端）',
    roc_auc: 'ROC-AUC',
    accuracy: 'Accuracy',
    f1: 'F1',
    rmse: 'RMSE',
    mae: 'MAE',
    r2: 'R²',
  }
  if (!direction) return labels[metric]
  return `${labels[metric]}（${direction === 'minimize' ? '越小越好' : '越大越好'}）`
}

function metricLabel(name: string): string {
  const value = result.value?.metrics?.[name]
  return typeof value === 'number' ? value.toFixed(5) : '-'
}

function isAutoMLModel(value: string): value is AutoMLModel {
  return [
    'lightgbm',
    'xgboost',
    'catboost',
    'random_forest',
    'logistic_regression',
    'linear_regression',
  ].includes(value)
}

function isAutoMLMetric(value: string | undefined): value is AutoMLMetric {
  return value === 'auto' || value === 'roc_auc' || value === 'accuracy' || value === 'f1'
    || value === 'rmse' || value === 'mae' || value === 'r2'
}

function isExplicitAutoMLMetric(value: string): value is Exclude<AutoMLMetric, 'auto'> {
  return value !== 'auto' && isAutoMLMetric(value)
}

function isMetricDirection(value: unknown): value is MetricDirection {
  return value === 'minimize' || value === 'maximize'
}

function problemTypeLabel(value: ProblemType): string {
  const labels: Record<ProblemType, string> = {
    auto: '自动推断',
    binary_classification: '二分类',
    multiclass_classification: '多分类',
    regression: '回归',
  }
  return labels[value]
}

function problemSourceLabel(source: ProblemSpec['source']): string {
  const labels: Record<ProblemSpec['source'], string> = {
    auto_inferred: '自动推断',
    user_override: '用户指定',
    legacy_compatibility: '旧模型兼容',
  }
  return labels[source]
}

function validateProblemCapabilities(capability: ProblemCapabilities | null | undefined): { valid: boolean; message: string } {
  if (!capability) return { valid: false, message: '未返回该任务类型的能力' }
  if (!Array.isArray(capability.algorithms) || capability.algorithms.length === 0 || !capability.algorithms.every(isAutoMLModel)) {
    return { valid: false, message: '未返回可用候选模型' }
  }
  if (!Array.isArray(capability.metrics) || capability.metrics.length === 0 || !capability.metrics.every(isExplicitAutoMLMetric)) {
    return { valid: false, message: '未返回可用优化指标' }
  }
  if (!isExplicitAutoMLMetric(capability.default_metric) || !capability.metrics.includes(capability.default_metric)) {
    return { valid: false, message: '默认优化指标不在可用指标中' }
  }
  if (!capability.metric_directions || capability.metrics.some((metric) => !isMetricDirection(capability.metric_directions[metric]))) {
    return { valid: false, message: '优化指标方向不完整或无效' }
  }
  return { valid: true, message: '' }
}

function hasUsableCapabilities(response: ProblemCapabilitiesResponse | undefined | null): response is ProblemCapabilitiesResponse {
  return Boolean(response && Object.values(response.problem_types ?? {}).some((capability) => validateProblemCapabilities(capability).valid))
}

function applyProblemType() {
  selectionResetMessage.value = ''
  if (problemType.value === 'auto') {
    const hadExplicitSelection = candidateModels.value.length > 0 || optimizationMetric.value !== 'auto'
    candidateModels.value = []
    optimizationMetric.value = 'auto'
    if (hadExplicitSelection) selectionResetMessage.value = '任务类型切换为自动推断后，候选模型和优化指标已重置为后端自动选择。'
    return
  }

  if (!explicitCapabilitiesValidation.value.valid) return
  const models = compatibleModels.value
  const metrics = compatibleMetrics.value
  const retainedModels = candidateModels.value.filter((model) => models.includes(model))
  const nextModels = retainedModels.length ? retainedModels : [models[0]]
  const nextMetric = metrics.includes(optimizationMetric.value as Exclude<AutoMLMetric, 'auto'>)
    ? optimizationMetric.value
    : metrics.includes(explicitCapabilities.value?.default_metric as Exclude<AutoMLMetric, 'auto'>)
      ? explicitCapabilities.value?.default_metric as Exclude<AutoMLMetric, 'auto'>
      : metrics[0]
  const removedModels = candidateModels.value.filter((model) => !nextModels.includes(model))
  const modelsChanged = nextModels.length !== candidateModels.value.length
    || nextModels.some((model, index) => model !== candidateModels.value[index])
  const metricChanged = nextMetric !== optimizationMetric.value
  const messages: string[] = []
  if (modelsChanged) {
    if (removedModels.length) {
      messages.push(`候选模型已移除不兼容项：${removedModels.map(modelLabel).join('、')}`)
    } else if (candidateModels.value.length) {
      messages.push(`候选模型已重置为${nextModels.map(modelLabel).join('、')}`)
    } else {
      messages.push(`候选模型已选择${nextModels.map(modelLabel).join('、')}`)
    }
  }
  if (metricChanged) {
    messages.push(optimizationMetric.value === 'auto'
      ? `优化指标已选择${metricOptionLabel(nextMetric)}`
      : `优化指标已从${metricOptionLabel(optimizationMetric.value)}重置为${metricOptionLabel(nextMetric)}`)
  }
  candidateModels.value = nextModels
  optimizationMetric.value = nextMetric
  selectionResetMessage.value = messages.join('；')
}

async function loadCapabilities() {
  capabilitiesLoading.value = true
  capabilitiesUnavailable.value = false
  capabilitiesError.value = ''
  try {
    const response = await trainingApi.getCapabilities()
    if (!hasUsableCapabilities(response)) {
      capabilities.value = null
      capabilitiesUnavailable.value = true
      capabilitiesError.value = '无法验证显式任务能力：能力接口未返回可用模型和指标。请重试加载能力。'
      return
    }
    capabilities.value = response
    applyProblemType()
  } catch {
    capabilities.value = null
    capabilitiesUnavailable.value = true
    capabilitiesError.value = '无法验证显式任务能力：加载失败。请重试加载能力。'
  } finally {
    capabilitiesLoading.value = false
  }
}

function handleFileChange(uploadFile: UploadFile) {
  datasetFile.value = uploadFile.raw ?? null
}

function handleNativeFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  datasetFile.value = input.files?.[0] ?? null
}

function handleFileRemove() {
  datasetFile.value = null
}

async function startJob() {
  if (!datasetFile.value || explicitActionsBlocked.value) return
  creatingJob.value = true
  errorMessage.value = ''
  result.value = null
  savedArtifactPath.value = ''
  trialHistory.value = []
  closeStream()
  try {
    job.value = await automlApi.createJob({
      file: datasetFile.value,
      targetCol: targetCol.value,
      problemType: problemType.value,
      nTrials: nTrials.value,
      cvFolds: cvFolds.value,
      ...(problemType.value === 'auto' ? {
        candidateModels: undefined,
        optimizationMetric: 'auto' as const,
      } : {
        candidateModels: candidateModels.value,
        optimizationMetric: optimizationMetric.value,
      }),
      useLlm: useLlm.value,
      businessContext: businessContext.value,
      enableFeatureSelection: enableFeatureSelection.value,
      enableSampling: enableSampling.value,
      samplingNegativeRatio: samplingNegativeRatio.value,
      earlyStoppingRounds: earlyStoppingRounds.value,
      minImprovement: minImprovement.value,
    })
    localStorage.setItem(STORAGE_KEY, job.value.job_id)
    connectStream(job.value.job_id)
    ElMessage.success('AutoML 任务已创建')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'AutoML 任务创建失败'
  } finally {
    creatingJob.value = false
  }
}

async function refreshJob() {
  const jobId = currentJobId.value
  if (!jobId) return
  try {
    job.value = await automlApi.getJob(jobId)
    if (job.value.status === 'completed') {
      await loadResult(jobId)
    }
    if (isRunningStatus(job.value.status) && !streamController.value) {
      connectStream(jobId)
    } else if (!isRunningStatus(job.value.status)) {
      closeStream()
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '刷新 AutoML 状态失败'
  }
}

async function cancelJob() {
  if (!job.value || !canCancel.value) return
  cancellingRequest.value = true
  try {
    job.value = await automlApi.cancelJob(job.value.job_id)
    ElMessage.info('已发送取消请求')
    if (isRunningStatus(job.value.status) && !streamController.value) {
      connectStream(job.value.job_id)
    }
    if (!isRunningStatus(job.value.status)) {
      closeStream()
      await refreshJob()
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '取消任务失败'
  } finally {
    cancellingRequest.value = false
  }
}

async function loadResult(jobId: string) {
  result.value = await automlApi.getResult(jobId)
  trialHistory.value = result.value.trial_history ?? trialHistory.value
}

async function saveResult() {
  if (!job.value) return
  savingResult.value = true
  try {
    const saved = await automlApi.saveResult(job.value.job_id)
    savedArtifactPath.value = saved.artifact_path
    ElMessage.success('模型保存成功')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '模型保存失败'
  } finally {
    savingResult.value = false
  }
}

function connectStream(jobId: string) {
  closeStream()
  const controller = new AbortController()
  streamController.value = controller
  void automlApi.streamJob(jobId, async (event: AutoMLTrialEvent) => {
    if (event.problem_spec && job.value) job.value.problem_spec = event.problem_spec
    if (event.event === 'trial') {
      upsertTrial(event)
      if (typeof event.progress === 'number' && job.value) {
        job.value.progress = event.progress
        job.value.current_trial = (event.trial_number ?? 0) + 1
      }
    }
    if (event.event === 'done') {
      closeStream()
      await refreshJob()
    }
    if (event.event === 'error') {
      closeStream()
      errorMessage.value = event.message ?? 'AutoML 训练失败'
      await refreshJob()
    }
  }, controller.signal).then(() => {
    if (controller.signal.aborted || streamController.value !== controller) return
    streamController.value = null
    if (job.value && isRunningStatus(job.value.status)) void refreshJob()
  }).catch((error: unknown) => {
    if (controller.signal.aborted) return
    if (streamController.value === controller) streamController.value = null
    errorMessage.value = error instanceof Error ? error.message : 'AutoML 训练流连接中断'
    void refreshJob()
  })
}

function closeStream() {
  streamController.value?.abort()
  streamController.value = null
}

function upsertTrial(event: AutoMLTrialEvent) {
  const index = trialHistory.value.findIndex((item) => item.trial_number === event.trial_number)
  if (index >= 0) {
    trialHistory.value[index] = event
  } else {
    trialHistory.value.push(event)
  }
}

function initChart() {
  if (!chartRef.value || chartInstance) return
  chartInstance = init(chartRef.value)
  updateChart()
}

function updateChart() {
  if (!chartInstance) return
  const trialData = trialHistory.value
    .filter((item) => typeof item.value === 'number')
    .map((item) => [item.trial_number ?? 0, item.value])
  const bestData = trialHistory.value
    .filter((item) => typeof item.best_value === 'number')
    .map((item) => [item.trial_number ?? 0, item.best_value])
  const hasData = trialData.length > 0 || bestData.length > 0
  const startsAtZero = [...trialData, ...bestData].some(([trial]) => trial === 0)
  chartInstance.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      valueFormatter: (value: number | string) => `${value}${metricDirectionHint(resolvedMetricDirection.value)}`,
    },
    legend: { show: false, data: ['Trial 分数', '历史最佳'] },
    grid: { left: 38, right: 8, top: 21, bottom: 24, containLabel: true },
    xAxis: {
      type: 'value', name: 'Trial', nameLocation: 'middle', nameGap: 28,
      min: startsAtZero ? 0 : 1, max: Math.max(startsAtZero ? nTrials.value - 1 : nTrials.value, startsAtZero ? 9 : 10), interval: 1,
      axisLine: { lineStyle: { color: '#9cb0cf' } },
      axisTick: { show: false },
      axisLabel: { color: '#405886', fontSize: 12 },
      splitLine: { show: true, lineStyle: { color: '#dce8ff', type: 'dashed', width: 1 } },
      nameTextStyle: { color: '#405886', fontSize: 13 },
    },
    yAxis: {
      type: 'value', name: resolvedMetric.value === 'auto' ? '分数' : chartMetricLabel.value, nameLocation: 'middle', nameGap: 34,
      ...(hasData ? {} : { min: 0, max: 1, interval: .1 }),
      axisLine: { show: true, lineStyle: { color: '#9cb0cf' } },
      axisTick: { show: false },
      axisLabel: { color: '#405886', fontSize: 12, formatter: (value: number) => value.toFixed(1) },
      splitLine: { show: true, lineStyle: { color: '#dce8ff', type: 'dashed', width: 1 } },
      nameTextStyle: { color: '#405886', fontSize: 13 },
    },
    series: [
      { name: 'Trial 分数', type: 'scatter', data: trialData, symbolSize: 10, itemStyle: { color: '#3484ff' } },
      {
        name: '历史最佳',
        type: 'line',
        data: bestData,
        step: 'start',
        symbol: 'circle',
        symbolSize: 7,
        itemStyle: { color: '#fff', borderColor: '#2374ff', borderWidth: 2 },
        lineStyle: { color: '#2374ff', width: 2 },
      },
    ],
  })
}

function metricDirectionHint(direction: MetricDirection | null): string {
  if (direction === 'minimize') return '（越小越好）'
  if (direction === 'maximize') return '（越大越好）'
  return '（优化方向未验证）'
}

function percentLabel(value: number): string {
  return `${(value * 100).toFixed(2)}%`
}

watch([trialHistory, chartMetricLabel], updateChart, { deep: true })

const resizeChart = () => chartInstance?.resize()

onMounted(async () => {
  await nextTick()
  initChart()
  window.addEventListener('resize', resizeChart)
  void loadCapabilities()
  await refreshJob()
})

onBeforeUnmount(() => {
  closeStream()
  window.removeEventListener('resize', resizeChart)
  chartInstance?.dispose()
  chartInstance = null
})
</script>

<style scoped>
.automl-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.automl-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  min-height: 92px;
  max-width: none;
  width: 100%;
  padding-left: 15px;
}

.automl-heading .eyebrow {
  display: table;
  border-radius: 999px;
  background: #c7e7ff;
  color: #075cf2;
  padding: 1px 9px;
  font-size: 11px;
  line-height: 18px;
  margin-bottom: 2px;
}

.automl-grid {
  display: grid;
  grid-template-columns: minmax(500px, .735fr) minmax(0, 1fr);
  gap: 18px;
  align-items: stretch;
}

.setup-panel,
.run-panel,
.history-panel,
.decision-panel,
.execution-panel {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(199, 221, 255, .96);
  border-radius: var(--workbench-radius);
  background: var(--workbench-surface);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 22px 24px;
  box-shadow: var(--workbench-shadow);
}

.setup-panel::after,
.run-panel::after {
  content: '';
  position: absolute;
  width: 520px;
  height: 190px;
  right: -80px;
  bottom: -50px;
  pointer-events: none;
  opacity: .82;
  background: url('@/assets/card-waves.svg') no-repeat center / contain, radial-gradient(ellipse at 60% 75%, rgba(69,218,232,.22), transparent 60%), radial-gradient(ellipse at 75% 90%, rgba(94,153,255,.25), transparent 62%);
}

.setup-panel > *, .run-panel > * { position: relative; z-index: 1; }
.setup-panel, .run-panel { padding-bottom: 12px; }

.panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 18px;
}

.panel-title h3 {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0;
  color: #08143b;
  font-size: 20px;
  font-weight: 750;
}

.panel-title-icon {
  flex: 0 0 auto;
  color: #287dff;
  width: 23px;
  height: 23px;
}

.panel-title span {
  color: #51658e;
  font-size: 14px;
}

.run-panel > .panel-title { margin-bottom: 22px; }

.status-chip {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
  border: 1px solid #fff;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.75);
  padding: 5px 13px;
  color: #3f5891;
  font-size: 13px;
  font-weight: 600;
  margin: 2px 10px 0 0;
}

.status-chip--success {
  border-color: #bbf7d0;
  background: #f0fdf4;
  color: #15803d;
}

.status-chip--warning {
  border-color: #fde68a;
  background: #fffbeb;
  color: #b45309;
}

.status-chip--danger {
  border-color: #fecaca;
  background: #fef2f2;
  color: #b91c1c;
}

.dataset-upload {
  margin-bottom: 16px;
}

.dataset-upload :deep(.el-upload-dragger) {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 181px;
  border: 1.5px dashed #7eafff;
  border-radius: 13px;
  background: radial-gradient(ellipse at 96% 85%, rgba(137,184,255,.20), transparent 40%), linear-gradient(150deg, #eef5ff 0%, #f7faff 55%, #eaf2ff 100%);
  padding: 16px;
  transition: border-color 0.2s ease, background 0.2s ease;
}

.dataset-upload :deep(.el-upload-dragger:hover) {
  border-color: #3b82f6;
  background: linear-gradient(180deg, #f2f7ff 0%, #e6f0ff 100%);
}

.upload-copy {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
  color: #334155;
}

.upload-icon {
  color: #448eff;
  margin-bottom: 7px;
}

.upload-copy strong {
  color: #0b1941;
  font-size: 16px;
}

.upload-copy em {
  color: #2563eb;
  font-style: normal;
}

.upload-copy span {
  color: #536791;
  font-size: 14px;
}

.upload-copy small {
  margin-top: 9px;
  color: #8296bc;
  font-size: 13px;
}

.file-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  border: 1px solid #dbe5f5;
  border-radius: 10px;
  background: #f6f9fd;
  margin-bottom: 16px;
  min-height: 44px;
  padding: 4px 9px;
}

.file-bar-button {
  border-color: #bfdbfe;
  color: #2563eb;
  font-weight: 600;
}

.file-bar-button :deep(span) {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.file-bar-name {
  color: #65789e;
  font-size: 14px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.native-file-input--hidden {
  display: none;
}

.field-label,
.option-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.field-label svg,
.option-label svg,
.advanced-title svg {
  flex: 0 0 auto;
  color: #3b82f6;
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  column-gap: 26px;
  row-gap: 12px;
}

label,
.full-field {
  display: flex;
  flex-direction: column;
  gap: 5px;
  color: #2d416b;
  font-size: 14px;
}

.full-field {
  margin-top: 12px;
}

.option-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 16px;
  color: #475569;
  font-size: 13px;
}

.advanced-options {
  padding: 10px 10px 12px 20px;
  border: 1px solid #e2ebfb;
  border-radius: 12px;
  background: linear-gradient(155deg, #f0f6ff, #f9fbff 74%);
  box-shadow: 0 7px 18px rgba(50, 100, 183, .06);
}

.advanced-title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #132450;
  font-size: 14px;
  font-weight: 700;
}

.advanced-options .option-row {
  margin-top: 10px;
  min-height: 23px;
}

.advanced-options :deep(.el-switch) { height: 22px; }
.advanced-options :deep(.el-switch__core) { min-height: 20px; height: 20px; }
.advanced-options :deep(.el-switch__action) { width: 16px; height: 16px; }

.option-label b {
  color: #23385e;
  font-weight: 600;
}

.advanced-options .option-row small {
  margin-left: 2px;
  color: #899bbd;
  font-weight: 400;
}

.advanced-options .form-grid {
  margin-top: 9px;
  column-gap: 15px;
}

.advanced-options .form-grid label {
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  white-space: nowrap;
  font-size: 12px;
}

.advanced-options .form-grid .field-label { white-space: nowrap; }
/* Compact steppers share the same geometry as the main form. */
.advanced-options .form-grid :deep(.el-input-number) {
  --stepper-button-width: 19px;
  --stepper-input-padding: 4px;
  width: 85px;
  flex: 0 0 85px;
}
.advanced-options .form-grid :deep(.el-input__inner) { font-size: 12px; }

.enhancement-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 17px;
}

.metric {
  display: block;
  min-width: 0;
  min-height: 101px;
  border: 1px solid #bfd8ff;
  border-radius: 15px;
  padding: 15px 20px 10px;
  background: #f4f8ff;
  box-shadow: 0 9px 18px rgba(46, 111, 228, .09), inset 0 1px rgba(255,255,255,.9);
}

.metric--blue {
  background: linear-gradient(130deg, #f7fbff, #edf5ff);
}

.metric--teal {
  border-color: #a9ebff;
  background: linear-gradient(130deg, #f5fcff, #def7ff);
}

.metric--amber {
  border-color: #cfd5ff;
  background: linear-gradient(130deg, #f8f9ff, #e9edff);
}

.metric--purple {
  border-color: #c7d7ff;
  background: linear-gradient(130deg, #f8fbff, #e7efff);
}

.metric-icon {
  display: inline-flex;
  vertical-align: middle;
  align-items: center;
  width: 30px;
  height: 27px;
}

.metric--blue .metric-icon {
  color: #36a6ff;
}

.metric--teal .metric-icon {
  color: #08c8e4;
}

.metric--amber .metric-icon {
  color: #7072ff;
}

.metric--purple .metric-icon {
  color: #5a7dff;
}

.metric-body { display: inline; }

.metric-body span {
  display: inline;
  color: #415684;
  font-size: 14px;
  font-weight: 600;
}

.metric-body strong {
  display: block;
  margin: 2px 0 0 40px;
  color: #071747;
  font-size: 28px;
  line-height: 1.1;
}

.metric-model {
  font-size: 22px !important;
}

.progress-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 19px 0 27px;
}

.progress-row :deep(.el-progress-bar__outer) { background: #d5e2ff; }

.progress-bar {
  flex: 1;
  margin: 0;
}

.progress-value {
  min-width: 32px;
  color: #405582;
  font-size: 18px;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.chart-card {
  height: 438px;
  border: 1px solid #dce9ff;
  border-radius: 15px;
  background: linear-gradient(145deg, rgba(255,255,255,.9), rgba(247,250,255,.94));
  padding: 18px 17px 12px;
  box-shadow: 0 9px 24px rgba(51, 110, 207, .06);
}

.chart-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.chart-title {
  color: #263c6d;
  font-size: 16px;
  font-weight: 650;
}

.chart-legend {
  display: flex;
  align-items: center;
  gap: 22px;
  color: #435783;
  font-size: 14px;
  white-space: nowrap;
}

.chart-legend span { display: inline-flex; align-items: center; gap: 8px; }
.legend-dot { width: 15px; height: 15px; border-radius: 50%; background: linear-gradient(#67abff,#1866fa); }
.legend-line { width: 28px; height: 2px; background: #2374ff; position: relative; }
.legend-line::after { content: ''; position: absolute; top: -6px; left: 10px; width: 12px; height: 12px; border: 2px solid #2374ff; border-radius: 50%; background: #fff; }

.automl-chart {
  width: 100%;
  height: 378px;
}

.execution-fields { display: grid; grid-template-columns: minmax(250px, 1fr) auto; align-items: center; gap: 25px; }
.execution-fields .full-field, .execution-fields .option-row { margin: 0; }

.status-alert,
.result-box {
  margin-top: 14px;
}

.result-grid,
.decision-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  color: #334155;
  font-size: 13px;
}

pre {
  max-height: 220px;
  overflow: auto;
  margin: 14px 0;
  padding: 12px;
  border-radius: 8px;
  background: #0f172a;
  color: #e2e8f0;
  font-size: 12px;
}

.param-tag {
  margin: 0 6px 6px 0;
}

.decision-panel p {
  margin: 10px 0 0;
  color: #475569;
}

.setup-panel :deep(.el-input__wrapper),
.setup-panel :deep(.el-select__wrapper) {
  min-height: 33px;
  border-radius: 7px;
  background: linear-gradient(#fff, #f8faff);
  box-shadow: 0 0 0 1px #d0dced inset, 0 2px 5px rgba(33,77,160,.04);
}

.setup-panel :deep(.el-input__wrapper.is-focus),
.setup-panel :deep(.el-select__wrapper.is-focused) {
  box-shadow: 0 0 0 1px #3b82f6 inset;
}

.setup-panel :deep(.el-input-number) {
  --stepper-button-width: 35px;
  --stepper-input-padding: 10px;
  width: 100%;
  height: 33px;
}

.setup-panel :deep(.el-input-number .el-input) { height: 100%; }
.setup-panel :deep(.el-input-number .el-input__wrapper) {
  padding-left: var(--stepper-input-padding);
  padding-right: calc(2 * var(--stepper-button-width) + var(--stepper-input-padding));
}
.setup-panel :deep(.el-input-number__decrease),
.setup-panel :deep(.el-input-number__increase) {
  top: 1px;
  bottom: 1px;
  left: auto;
  width: var(--stepper-button-width);
  height: auto;
  border: 0;
  border-left: 1px solid #dae5f6;
  background: #f3f7ff;
  color: #56729d;
}
.setup-panel :deep(.el-input-number__decrease) {
  right: calc(var(--stepper-button-width) + 1px);
  border-radius: 0;
}
.setup-panel :deep(.el-input-number__increase) {
  right: 1px;
  border-radius: 0 6px 6px 0;
}
.setup-panel :deep(.el-input-number__decrease.is-disabled),
.setup-panel :deep(.el-input-number__increase.is-disabled) { color: #b2bfd3; }
.setup-panel :deep(.el-switch__core) { min-width: 43px; }

.capability-alert {
  display: flex;
  align-items: center;
  gap: 10px;
  border: 1px solid #fde68a;
  border-radius: 10px;
  background: #fffbeb;
  margin-bottom: 14px;
  padding: 10px 12px;
  color: #b45309;
  font-size: 13px;
}

.actions :deep(.el-button--primary) {
  border: 0;
  border-radius: 8px;
  background: linear-gradient(90deg, #3b82f6, #2563eb);
  font-weight: 600;
}

@media (max-width: 1250px) {
  .automl-grid,
  .result-grid,
  .decision-grid {
    grid-template-columns: 1fr;
  }

  .metric-grid { grid-template-columns: repeat(4, minmax(0,1fr)); }

  .automl-grid { align-items: start; }

  .setup-panel, .run-panel { min-width: 0; }
}

@media (max-width: 700px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .form-grid {
    grid-template-columns: 1fr;
  }

  .execution-fields { grid-template-columns: 1fr; }
  .chart-legend { gap: 10px; font-size: 11px; }
  .chart-title { font-size: 13px; }
}
</style>
