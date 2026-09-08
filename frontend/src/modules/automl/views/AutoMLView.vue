<template>
  <section class="automl-workspace">
    <div class="section-heading automl-heading">
      <div>
        <p class="eyebrow">Phase 1</p>
        <h2>AutoML 训练</h2>
        <p>上传数据后自动完成目标识别、模型推荐、Optuna 参数搜索、评估和模型保存。</p>
      </div>
      <el-tag :type="statusTagType" effect="light">{{ statusLabel }}</el-tag>
    </div>

    <div class="automl-grid">
      <section class="setup-panel">
        <div class="panel-title">
          <h3>训练输入</h3>
          <span>{{ selectedFile?.name ?? '未选择数据' }}</span>
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
            <strong>选择 CSV / Excel 数据集</strong>
            <span>后端会自动完成特征编码和目标检查</span>
          </div>
        </el-upload>
        <input
          data-testid="automl-file-input"
          class="native-file-input"
          type="file"
          accept=".csv,.xlsx,.xls"
          @change="handleNativeFileChange"
        />

        <div class="form-grid">
          <label>
            <span>目标列</span>
            <el-input v-model="targetCol" placeholder="留空则自动识别 label / target / 最后一列" clearable />
          </label>

          <label>
            <span>优化指标</span>
            <el-select v-model="optimizationMetric">
              <el-option label="ROC-AUC" value="roc_auc" />
              <el-option label="Accuracy" value="accuracy" />
              <el-option label="F1" value="f1" />
            </el-select>
          </label>

          <label>
            <span>Trial 次数</span>
            <el-input-number v-model="nTrials" :min="1" :max="100" :step="1" controls-position="right" />
          </label>

          <label>
            <span>交叉验证</span>
            <el-input-number v-model="cvFolds" :min="2" :max="10" :step="1" controls-position="right" />
          </label>
        </div>

        <label class="full-field">
          <span>候选模型</span>
          <el-select v-model="candidateModels" multiple collapse-tags collapse-tags-tooltip placeholder="留空则后端自动推荐">
            <el-option label="LightGBM" value="lightgbm" />
            <el-option label="XGBoost" value="xgboost" />
            <el-option label="CatBoost" value="catboost" />
            <el-option label="随机森林" value="random_forest" />
            <el-option label="逻辑回归" value="logistic_regression" />
          </el-select>
        </label>

        <label class="full-field">
          <span>业务场景</span>
          <el-input
            v-model="businessContext"
            type="textarea"
            :rows="3"
            placeholder="例如：信贷风控、流失预测、质量检测"
          />
        </label>

        <div class="option-row">
          <span>调用 LLM 决策</span>
          <el-switch v-model="useLlm" active-text="启用" inactive-text="禁用" />
        </div>

        <div class="actions">
          <el-button type="primary" :loading="creatingJob" :disabled="!canStart" @click="startJob">
            开始全自动训练
          </el-button>
          <el-button plain :disabled="!currentJobId" @click="refreshJob">刷新状态</el-button>
          <el-button type="danger" plain :disabled="!canCancel" @click="cancelJob">取消任务</el-button>
        </div>
      </section>

      <section class="run-panel">
        <div class="panel-title">
          <h3>训练进度</h3>
          <span>{{ job?.job_id ?? '等待创建任务' }}</span>
        </div>

        <div class="metric-grid">
          <div class="metric">
            <span>进度</span>
            <strong>{{ job?.progress ?? 0 }}%</strong>
          </div>
          <div class="metric">
            <span>Trial</span>
            <strong>{{ job?.current_trial ?? 0 }}/{{ job?.n_trials ?? nTrials }}</strong>
          </div>
          <div class="metric">
            <span>最佳分数</span>
            <strong>{{ bestValueLabel }}</strong>
          </div>
          <div class="metric">
            <span>最佳模型</span>
            <strong>{{ result ? modelLabel(result.best_model) : '-' }}</strong>
          </div>
        </div>

        <el-progress
          class="progress-bar"
          :percentage="job?.progress ?? 0"
          :status="job?.status === 'completed' ? 'success' : job?.status === 'failed' ? 'exception' : undefined"
          :stroke-width="12"
          striped
          striped-flow
        />

        <div ref="chartRef" class="automl-chart"></div>

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
            <span>ROC-AUC：{{ metricLabel('roc_auc') }}</span>
            <span>Accuracy：{{ metricLabel('accuracy') }}</span>
            <span>F1：{{ metricLabel('f1') }}</span>
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

const STORAGE_KEY = 'agentic-automl.currentJobId'

const selectedFile = ref<File | null>(null)
const targetCol = ref('')
const optimizationMetric = ref<AutoMLMetric>('roc_auc')
const nTrials = ref(10)
const cvFolds = ref(3)
const candidateModels = ref<AutoMLModel[]>([])
const businessContext = ref('通用机器学习建模')
const useLlm = ref(true)

const job = ref<AutoMLJob | null>(null)
const result = ref<AutoMLResult | null>(null)
const trialHistory = ref<AutoMLTrialEvent[]>([])
const creatingJob = ref(false)
const savingResult = ref(false)
const errorMessage = ref('')
const savedArtifactPath = ref('')
const streamController = ref<AbortController | null>(null)
const chartRef = ref<HTMLElement | null>(null)
let chartInstance: ECharts | null = null

const currentJobId = computed(() => job.value?.job_id ?? localStorage.getItem(STORAGE_KEY) ?? '')
const canStart = computed(() => Boolean(selectedFile.value) && !creatingJob.value && !isRunningStatus(job.value?.status))
const canCancel = computed(() => Boolean(job.value && isRunningStatus(job.value.status)))
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
  }
  return model ? labels[model] : '-'
}

function metricLabel(name: string): string {
  const value = result.value?.metrics?.[name]
  return typeof value === 'number' ? value.toFixed(5) : '-'
}

function handleFileChange(uploadFile: UploadFile) {
  selectedFile.value = uploadFile.raw ?? null
}

function handleNativeFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  selectedFile.value = input.files?.[0] ?? null
}

function handleFileRemove() {
  selectedFile.value = null
}

async function startJob() {
  if (!selectedFile.value) return
  creatingJob.value = true
  errorMessage.value = ''
  result.value = null
  savedArtifactPath.value = ''
  trialHistory.value = []
  closeStream()
  try {
    job.value = await automlApi.createJob({
      file: selectedFile.value,
      targetCol: targetCol.value,
      nTrials: nTrials.value,
      cvFolds: cvFolds.value,
      candidateModels: candidateModels.value,
      optimizationMetric: optimizationMetric.value,
      useLlm: useLlm.value,
      businessContext: businessContext.value,
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
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '刷新 AutoML 状态失败'
  }
}

async function cancelJob() {
  if (!job.value) return
  try {
    job.value = await automlApi.cancelJob(job.value.job_id)
    closeStream()
    ElMessage.info('已发送取消请求')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '取消任务失败'
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
  }, controller.signal).catch((error: unknown) => {
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
  chartInstance.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { top: 8, right: 12, data: ['Trial 分数', '历史最佳'] },
    grid: { left: 44, right: 24, top: 56, bottom: 36, containLabel: true },
    xAxis: { type: 'value', name: 'Trial', minInterval: 1 },
    yAxis: { type: 'value', name: optimizationMetric.value },
    series: [
      { name: 'Trial 分数', type: 'scatter', data: trialData, symbolSize: 10, itemStyle: { color: '#2563eb' } },
      { name: '历史最佳', type: 'line', data: bestData, step: 'start', lineStyle: { color: '#dc2626', width: 3 } },
    ],
  })
}

watch(trialHistory, updateChart, { deep: true })

const resizeChart = () => chartInstance?.resize()

onMounted(async () => {
  await nextTick()
  initChart()
  window.addEventListener('resize', resizeChart)
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
  gap: 20px;
}

.automl-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
}

.automl-grid {
  display: grid;
  grid-template-columns: minmax(320px, 440px) minmax(0, 1fr);
  gap: 18px;
}

.setup-panel,
.run-panel,
.history-panel,
.decision-panel {
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #ffffff;
  padding: 18px;
  box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
}

.panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
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

.dataset-upload {
  margin-bottom: 16px;
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

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

label,
.full-field {
  display: flex;
  flex-direction: column;
  gap: 7px;
  color: #475569;
  font-size: 13px;
}

.full-field {
  margin-top: 14px;
}

.option-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 16px;
  color: #475569;
  font-size: 13px;
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
  gap: 12px;
}

.metric {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px;
  background: #f8fafc;
}

.metric span {
  display: block;
  color: #64748b;
  font-size: 12px;
}

.metric strong {
  display: block;
  margin-top: 5px;
  color: #0f172a;
  font-size: 20px;
}

.progress-bar {
  margin: 16px 0;
}

.automl-chart {
  width: 100%;
  height: 320px;
}

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

@media (max-width: 1100px) {
  .automl-grid,
  .metric-grid,
  .result-grid,
  .decision-grid {
    grid-template-columns: 1fr;
  }

  .form-grid {
    grid-template-columns: 1fr;
  }
}
</style>
