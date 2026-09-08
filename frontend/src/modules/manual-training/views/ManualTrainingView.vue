<template>
  <section class="training-workspace">
    <div class="section-heading training-heading">
      <div>
        <p class="eyebrow">Phase 4</p>
        <h2>手动训练</h2>
        <p>按步骤完成数据上传、字段配置检查、EDA、单模型训练、指标查看和模型保存。</p>
      </div>
      <el-tag :type="job?.status === 'completed' ? 'success' : job?.status === 'failed' ? 'danger' : 'info'" effect="light">
        {{ job ? statusLabel(job.status) : '待上传' }}
      </el-tag>
    </div>

    <el-alert v-if="visibleError" type="error" :title="visibleError" show-icon :closable="false" />

    <el-steps :active="activeStep" finish-status="success" class="training-steps">
      <el-step title="数据" />
      <el-step title="字段 / EDA" />
      <el-step title="训练" />
      <el-step title="结果" />
    </el-steps>

    <div class="training-grid">
      <section class="panel">
        <div class="panel-title">
          <h3>1. 数据上传</h3>
          <span>{{ datasetFile?.name ?? dataset?.filename ?? '未选择数据' }}</span>
        </div>
        <el-upload
          drag
          accept=".csv,.xlsx,.xls"
          :auto-upload="false"
          :limit="1"
          :on-change="handleFileChange"
          :on-remove="handleFileRemove"
        >
          <div class="upload-copy">
            <strong>选择训练数据</strong>
            <span>支持 CSV / Excel，目标列可留空自动识别</span>
          </div>
        </el-upload>
        <input
          data-testid="manual-training-file-input"
          class="native-file-input"
          type="file"
          accept=".csv,.xlsx,.xls"
          @change="handleNativeFileChange"
        />
        <div class="inline-form">
          <el-input v-model="targetCol" placeholder="目标列，例如 label" clearable />
          <el-button type="primary" :loading="uploading" :disabled="!datasetFile" @click="uploadDataset">上传</el-button>
        </div>

        <div v-if="dataset" class="dataset-summary">
          <span>行数 {{ dataset.rows }}</span>
          <span>列数 {{ dataset.columns }}</span>
          <span>目标列 {{ dataset.target_col }}</span>
        </div>
      </section>

      <section class="panel">
        <div class="panel-title">
          <h3>2. 字段配置 / EDA</h3>
          <el-button plain :disabled="!dataset" :loading="edaLoading" @click="runEda">运行 EDA</el-button>
        </div>
        <div v-if="eda" class="eda-summary">
          <span>缺失率 {{ percentLabel(eda.summary.missing_rate) }}</span>
          <span>数值特征 {{ eda.summary.numeric_features }}</span>
          <span>类别特征 {{ eda.summary.categorical_features }}</span>
          <span>目标唯一值 {{ eda.summary.target_unique }}</span>
        </div>
        <el-table :data="fields" height="280">
          <el-table-column prop="field_name" label="字段" />
          <el-table-column prop="field_type" label="类型" width="90" />
          <el-table-column prop="dtype" label="dtype" width="120" />
          <el-table-column label="缺失率" width="110">
            <template #default="{ row }">{{ percentLabel(row.missing_rate) }}</template>
          </el-table-column>
        </el-table>
      </section>
    </div>

    <div class="training-grid">
      <section class="panel">
        <div class="panel-title">
          <h3>3. 模型训练</h3>
          <span>{{ algorithmLabel(algorithm) }}</span>
        </div>
        <div class="form-grid">
          <label>
            <span>算法</span>
            <el-select v-model="algorithm" @change="applyAlgorithmPreset">
              <el-option label="LightGBM" value="lightgbm" />
              <el-option label="XGBoost" value="xgboost" />
              <el-option label="CatBoost" value="catboost" />
              <el-option label="随机森林" value="random_forest" />
              <el-option label="逻辑回归" value="logistic_regression" />
            </el-select>
          </label>
          <label>
            <span>测试集比例</span>
            <el-input-number v-model="testSize" :min="0.1" :max="0.5" :step="0.05" />
          </label>
          <label>
            <span>随机种子</span>
            <el-input-number v-model="randomState" :min="0" :max="10000" />
          </label>
        </div>
        <label class="params-field">
          <span>模型参数 JSON</span>
          <el-input v-model="paramsJson" type="textarea" :rows="8" />
          <small v-if="paramsError" class="field-error">{{ paramsError }}</small>
        </label>
        <div class="actions">
          <el-button type="primary" :disabled="!dataset || isRunning || Boolean(paramsError)" :loading="creatingJob" @click="createJob">
            开始训练
          </el-button>
          <el-button plain :disabled="!job" @click="refreshJob">刷新任务</el-button>
          <el-button type="danger" plain :disabled="!isRunning" @click="cancelJob">取消任务</el-button>
        </div>
        <el-progress
          class="progress"
          :percentage="job?.progress ?? 0"
          :status="job?.status === 'completed' ? 'success' : job?.status === 'failed' ? 'exception' : undefined"
          :stroke-width="12"
          striped
          striped-flow
        />
      </section>

      <section class="panel">
        <div class="panel-title">
          <h3>4. 训练结果</h3>
          <el-button type="success" plain :disabled="!result" :loading="saving" @click="saveResult">保存模型</el-button>
        </div>
        <div class="metric-grid">
          <div class="metric">
            <span>ROC-AUC</span>
            <strong>{{ metricLabel('roc_auc') }}</strong>
          </div>
          <div class="metric">
            <span>Accuracy</span>
            <strong>{{ metricLabel('accuracy') }}</strong>
          </div>
          <div class="metric">
            <span>F1</span>
            <strong>{{ metricLabel('f1') }}</strong>
          </div>
          <div class="metric">
            <span>阈值</span>
            <strong>{{ metricLabel('best_threshold') }}</strong>
          </div>
        </div>
        <p v-if="result?.threshold_calibration" class="calibration-note">
          阈值来源：{{ result.threshold_calibration.source === 'training_calibration_split' ? '训练集内部校准' : '默认阈值' }}
          <template v-if="result.threshold_calibration.calibration_size">
            （{{ result.threshold_calibration.calibration_size }} 条校准样本）
          </template>
        </p>
        <el-table :data="result?.feature_importance ?? []" height="240">
          <el-table-column prop="feature" label="特征" />
          <el-table-column label="重要性" width="130">
            <template #default="{ row }">{{ row.importance.toFixed(5) }}</template>
          </el-table-column>
        </el-table>
        <el-alert
          v-if="savedArtifactPath"
          class="saved-alert"
          type="success"
          :title="`模型已保存：${savedArtifactPath}`"
          show-icon
          :closable="false"
        />
      </section>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import type { UploadFile } from 'element-plus'
import { ElMessage } from 'element-plus/es/components/message/index.mjs'

import {
  trainingApi,
  type TrainingAlgorithm,
  type TrainingDataset,
  type TrainingEda,
  type TrainingField,
  type TrainingJob,
  type TrainingResult,
  type TrainingStatus,
} from '@/api/training'

const datasetFile = ref<File | null>(null)
const targetCol = ref('')
const dataset = ref<TrainingDataset | null>(null)
const fields = ref<TrainingField[]>([])
const eda = ref<TrainingEda | null>(null)
const job = ref<TrainingJob | null>(null)
const result = ref<TrainingResult | null>(null)
const savedArtifactPath = ref('')
const errorMessage = ref('')

const uploading = ref(false)
const edaLoading = ref(false)
const creatingJob = ref(false)
const saving = ref(false)

const algorithm = ref<TrainingAlgorithm>('logistic_regression')
const testSize = ref(0.3)
const randomState = ref(42)
const paramsJson = ref(JSON.stringify({ C: 1.0, solver: 'liblinear', max_iter: 1000 }, null, 2))
let pollGeneration = 0

const PARAM_PRESETS: Record<TrainingAlgorithm, Record<string, unknown>> = {
  lightgbm: { n_estimators: 200, learning_rate: 0.05, num_leaves: 31 },
  xgboost: { n_estimators: 200, learning_rate: 0.05, max_depth: 6 },
  catboost: { iterations: 200, learning_rate: 0.05, depth: 6 },
  random_forest: { n_estimators: 300, max_depth: 12, min_samples_leaf: 2 },
  logistic_regression: { C: 1.0, solver: 'liblinear', max_iter: 1000 },
  linear_regression: {},
}

const activeStep = computed(() => {
  if (result.value) return 4
  if (job.value) return 3
  if (eda.value || fields.value.length) return 2
  if (dataset.value) return 1
  return 0
})
const isRunning = computed(() => ['queued', 'running', 'cancelling'].includes(job.value?.status ?? ''))
const visibleError = computed(() => errorMessage.value || (job.value?.status === 'failed' ? job.value.error || '训练任务失败' : ''))
const paramsError = computed(() => {
  try {
    const parsed = JSON.parse(paramsJson.value || '{}')
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? '' : '模型参数必须是 JSON 对象'
  } catch {
    return '模型参数 JSON 格式不正确'
  }
})

onBeforeUnmount(() => {
  pollGeneration += 1
})

function handleFileChange(uploadFile: UploadFile) {
  setDatasetFile(uploadFile.raw ?? null)
}

function handleNativeFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  setDatasetFile(input.files?.[0] ?? null)
}

function handleFileRemove() {
  datasetFile.value = null
}

function setDatasetFile(file: File | null) {
  if (file && file.size > 50 * 1024 * 1024) {
    datasetFile.value = null
    errorMessage.value = '文件超过 50 MB 上传限制'
    return
  }
  datasetFile.value = file
  errorMessage.value = ''
}

function applyAlgorithmPreset() {
  paramsJson.value = JSON.stringify(PARAM_PRESETS[algorithm.value], null, 2)
}

async function uploadDataset() {
  if (!datasetFile.value) return
  uploading.value = true
  errorMessage.value = ''
  try {
    dataset.value = await trainingApi.uploadDataset({ file: datasetFile.value, targetCol: targetCol.value })
    const config = await trainingApi.getFieldConfig(dataset.value.dataset_id)
    fields.value = config.fields
    result.value = null
    job.value = null
    eda.value = null
    pollGeneration += 1
    savedArtifactPath.value = ''
    ElMessage.success('数据上传完成')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '数据上传失败'
  } finally {
    uploading.value = false
  }
}

async function runEda() {
  if (!dataset.value) return
  edaLoading.value = true
  errorMessage.value = ''
  try {
    eda.value = await trainingApi.runEda(dataset.value.dataset_id)
    fields.value = eda.value.columns
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'EDA 失败'
  } finally {
    edaLoading.value = false
  }
}

async function createJob() {
  if (!dataset.value || paramsError.value) return
  creatingJob.value = true
  errorMessage.value = ''
  result.value = null
  savedArtifactPath.value = ''
  try {
    const params = JSON.parse(paramsJson.value || '{}') as Record<string, unknown>
    job.value = await trainingApi.createJob({
      dataset_id: dataset.value.dataset_id,
      algorithm: algorithm.value,
      params,
      test_size: testSize.value,
      random_state: randomState.value,
    })
    ElMessage.success('训练任务已创建')
    void pollJob(job.value.job_id)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '训练任务创建失败'
  } finally {
    creatingJob.value = false
  }
}

async function pollJob(jobId: string) {
  const generation = ++pollGeneration
  const deadline = Date.now() + 10 * 60 * 1000
  let delay = 500
  while (generation === pollGeneration && Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, delay))
    if (generation !== pollGeneration) return
    let current: TrainingJob
    try {
      current = await trainingApi.getJob(jobId)
    } catch (error) {
      if (generation === pollGeneration) {
        errorMessage.value = error instanceof Error ? error.message : '任务状态轮询失败'
      }
      return
    }
    if (generation !== pollGeneration) return
    job.value = current
    if (current.status === 'completed') {
      result.value = await trainingApi.getResult(jobId)
      return
    }
    if (current.status === 'failed' || current.status === 'cancelled') return
    delay = Math.min(Math.round(delay * 1.35), 2000)
  }
  if (generation === pollGeneration) {
    errorMessage.value = '训练仍在后台运行，请稍后点击“刷新任务”查看结果'
  }
}

async function refreshJob() {
  if (!job.value) return
  try {
    job.value = await trainingApi.getJob(job.value.job_id)
    if (job.value.status === 'completed' && !result.value) {
      result.value = await trainingApi.getResult(job.value.job_id)
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '任务刷新失败'
  }
}

async function cancelJob() {
  if (!job.value || !isRunning.value) return
  errorMessage.value = ''
  try {
    pollGeneration += 1
    job.value = await trainingApi.cancelJob(job.value.job_id)
    ElMessage.info('已发送取消请求')
    if (isRunning.value) void pollJob(job.value.job_id)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '取消任务失败'
  }
}

async function saveResult() {
  if (!job.value) return
  saving.value = true
  errorMessage.value = ''
  try {
    const saved = await trainingApi.saveResult(job.value.job_id)
    savedArtifactPath.value = saved.artifact_path
    ElMessage.success('模型保存成功')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '模型保存失败'
  } finally {
    saving.value = false
  }
}

function metricLabel(name: string): string {
  const value = result.value?.metrics?.[name]
  return typeof value === 'number' ? value.toFixed(5) : '-'
}

function percentLabel(value: unknown): string {
  const numeric = Number(value ?? 0)
  return `${(numeric * 100).toFixed(2)}%`
}

function statusLabel(status: TrainingStatus): string {
  const labels: Record<TrainingStatus, string> = {
    queued: '排队中',
    running: '训练中',
    cancelling: '取消中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }
  return labels[status]
}

function algorithmLabel(value: TrainingAlgorithm): string {
  const labels: Record<TrainingAlgorithm, string> = {
    lightgbm: 'LightGBM',
    xgboost: 'XGBoost',
    catboost: 'CatBoost',
    random_forest: '随机森林',
    logistic_regression: '逻辑回归',
    linear_regression: '线性回归',
  }
  return labels[value]
}
</script>

<style scoped>
.training-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.training-heading,
.panel-title {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.training-steps {
  padding: 18px;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #fff;
}

.training-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.panel {
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #ffffff;
  padding: 18px;
  box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
}

.panel-title {
  align-items: center;
  margin-bottom: 14px;
}

.panel-title h3 {
  margin: 0;
  color: #111827;
  font-size: 16px;
}

.panel-title span {
  color: #64748b;
  font-size: 12px;
}

.upload-copy {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #334155;
}

.upload-copy span {
  color: #64748b;
  font-size: 13px;
}

.inline-form,
.actions {
  display: flex;
  gap: 10px;
  margin-top: 14px;
}

.dataset-summary,
.eda-summary,
.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 14px 0;
}

.eda-summary {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.dataset-summary span,
.eda-summary span {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px;
  color: #334155;
  background: #f8fafc;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

label,
.params-field {
  display: flex;
  flex-direction: column;
  gap: 7px;
  color: #475569;
  font-size: 13px;
}

.field-error {
  color: #dc2626;
}

.calibration-note {
  margin: -4px 0 12px;
  color: #64748b;
  font-size: 12px;
}

.params-field {
  margin-top: 14px;
}

.progress,
.saved-alert {
  margin-top: 14px;
}

.metric {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px;
  background: #f8fafc;
}

.metric span {
  color: #64748b;
  font-size: 12px;
}

.metric strong {
  display: block;
  margin-top: 4px;
  color: #0f172a;
  font-size: 20px;
}

@media (max-width: 1100px) {
  .training-grid,
  .form-grid,
  .dataset-summary,
  .eda-summary,
  .metric-grid {
    grid-template-columns: 1fr;
  }

  .inline-form,
  .actions {
    flex-direction: column;
  }
}
</style>
