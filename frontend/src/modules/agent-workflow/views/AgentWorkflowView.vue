<template>
  <section class="workspace-theme workflow-workspace">
    <div class="section-heading workflow-heading">
      <div>
        <p class="eyebrow">Phase 6</p>
        <h2>Agent 工作流</h2>
        <p>按步骤执行自然语言取数、特征策略生成、AutoML 训练和在线预测，后续步骤会受前置状态约束。</p>
      </div>
      <div class="heading-actions">
        <el-button
          data-testid="workflow-refresh-status-button"
          :icon="Refresh"
          :loading="refreshingStatus"
          @click="refreshTask()"
        >
          刷新状态
        </el-button>
        <el-button data-testid="workflow-model-list-button" :icon="List" :loading="loadingModels" @click="openModelList">
          模型列表
        </el-button>
      </div>
    </div>

    <el-alert v-if="infoMessage" type="info" :title="infoMessage" show-icon :closable="true" @close="infoMessage = ''" />
    <el-alert
      v-if="errorMessage"
      data-testid="workflow-error-alert"
      type="error"
      :title="errorMessage"
      show-icon
      :closable="false"
    />

    <section class="workflow-status-strip">
      <div class="status-item">
        <span>Task ID</span>
        <strong>{{ taskId || '-' }}</strong>
      </div>
      <div class="status-item">
        <span>Step 1</span>
        <strong>{{ stepState.retrieve }}</strong>
      </div>
      <div class="status-item">
        <span>Step 2</span>
        <strong>{{ stepState.features }}</strong>
      </div>
      <div class="status-item">
        <span>{{ metricLabel(taskMetricName) }}</span>
        <strong>{{ formatMetric(taskMetricValue) }}</strong>
      </div>
    </section>

    <section v-if="taskStatus?.problem_spec" class="panel problem-spec-panel" aria-live="polite">
      <div class="panel-title">
        <h3>任务类型解析</h3>
        <el-tag :type="taskStatus.problem_spec.resolved_type === 'regression' ? 'warning' : 'primary'" effect="light">
          最终：{{ problemTypeLabel(taskStatus.problem_spec.resolved_type) }}
        </el-tag>
      </div>
      <div class="problem-spec-summary">
        <strong v-if="taskStatus.suggested_type">建议：{{ problemTypeLabel(taskStatus.suggested_type) }}</strong>
        <span>最终：{{ problemTypeLabel(taskStatus.problem_spec.resolved_type) }}</span>
        <span>来源：{{ problemSourceLabel(taskStatus.problem_spec.source) }}</span>
        <span>依据：{{ taskStatus.problem_spec.reason }}</span>
        <span>{{ taskStatus.problem_spec.explanation }}</span>
        <span>
          目标列 {{ taskStatus.problem_spec.target.column }}（{{ taskStatus.problem_spec.target.dtype }}），
          有效行 {{ taskStatus.problem_spec.target.valid_rows }}，唯一值 {{ taskStatus.problem_spec.target.unique_count }}
        </span>
      </div>
      <div v-if="taskStatus.agent_corrections?.length" class="correction-list">
        <h4>Agent 配置修正</h4>
        <div v-for="(correction, index) in taskStatus.agent_corrections" :key="`${correction.field}-${index}`" class="correction-item">
          <strong>{{ correction.field }}</strong>
          <span>{{ formatCorrectionValue(correction.suggested) }} → {{ formatCorrectionValue(correction.resolved) }}</span>
          <small>{{ correction.reason }}</small>
        </div>
      </div>
    </section>

    <section v-if="modelListVisible" class="panel model-list-panel" data-testid="workflow-model-list-panel">
      <div class="panel-title">
        <h3>模型列表</h3>
        <span>{{ modelRecords.length }} 个模型</span>
      </div>
      <el-table :data="modelRecords" max-height="260">
        <el-table-column prop="task_id" label="模型版本" width="180" />
        <el-table-column prop="model_path" label="模型文件" />
        <el-table-column label="操作" width="110">
          <template #default="{ row }">
            <el-button link type="primary" @click="selectModel(row)">选择</el-button>
          </template>
        </el-table-column>
      </el-table>
    </section>

    <div class="workflow-grid">
      <section class="panel">
        <div class="panel-title">
          <h3>1. 自然语言取数</h3>
          <span>{{ retrieveResult?.data_shape?.join(' × ') ?? '待执行' }}</span>
        </div>
        <el-form label-position="top" @submit.prevent>
          <el-form-item label="业务查询">
            <el-input v-model="retrieveForm.user_query" type="textarea" :rows="4" :disabled="training" />
          </el-form-item>
          <el-form-item label="目标列">
            <el-input v-model="retrieveForm.target_col" :disabled="training" />
          </el-form-item>
          <el-button
            data-testid="workflow-retrieve-button"
            type="primary"
            :loading="retrieving"
            :disabled="training"
            @click="runRetrieve"
          >
            生成 SQL 并提取数据
          </el-button>
        </el-form>
      </section>

      <section class="panel">
        <div class="panel-title">
          <h3>2. 特征策略生成</h3>
          <span>{{ featureResult?.actions_count ?? 0 }} 个动作</span>
        </div>
        <el-input v-model="featureMemo" type="textarea" :rows="5" placeholder="可选：补充业务先验或字段处理偏好" />
        <el-button class="step-button" type="primary" :loading="generatingFeatures" :disabled="!canGenerateFeatures" @click="runFeatureGeneration">
          生成特征策略
        </el-button>
        <el-table :data="featureActions" height="220" class="feature-table">
          <el-table-column prop="action" label="动作" width="140" />
          <el-table-column prop="column" label="字段" width="140" />
          <el-table-column prop="reason" label="理由" />
        </el-table>
      </section>
    </div>

    <div class="workflow-grid">
      <section class="panel">
        <div class="panel-title">
          <h3>3. AutoML 训练</h3>
          <span>{{ trainEvents.length }} 条事件</span>
        </div>
        <div class="train-controls">
          <label class="train-field">
            <span>问题类型</span>
            <el-select
              v-model="problemType"
              data-testid="workflow-problem-type-select"
              aria-label="工作流问题类型"
              :disabled="training"
              @change="handleProblemTypeChange"
            >
              <el-option label="自动识别" value="auto" />
              <el-option label="二分类" value="binary_classification" />
              <el-option label="多分类" value="multiclass_classification" />
              <el-option label="回归" value="regression" />
            </el-select>
          </label>
          <el-input-number v-model="trainConfig.nTrials" :min="10" :max="100" controls-position="right" :disabled="training" />
          <el-input-number v-model="trainConfig.cvFolds" :min="3" :max="10" controls-position="right" :disabled="training" />
          <el-button type="warning" :loading="training" :disabled="!canTrain" @click="startTraining">开始训练</el-button>
        </div>
        <el-progress :percentage="trainingProgress" :status="trainingProgress === 100 ? 'success' : undefined" />
        <el-table :data="trainEvents" height="240" class="event-table">
          <el-table-column prop="status" label="状态" width="120" />
          <el-table-column prop="trial" label="Trial" width="90" />
          <el-table-column label="最佳指标" width="160">
            <template #default="{ row }">{{ metricLabel(row.metric ?? row.best_metric) }} {{ formatMetric(row.value ?? row.best_value) }}</template>
          </el-table-column>
          <el-table-column prop="error" label="消息" />
        </el-table>
      </section>

      <section class="panel">
        <div class="panel-title">
          <h3>4. 在线预测</h3>
          <span>{{ modelRecords.length }} 个模型</span>
        </div>
        <el-select v-model="predictForm.model_version" aria-label="工作流预测模型" placeholder="选择模型版本">
          <el-option v-for="model in modelRecords" :key="model.task_id" :label="model.task_id" :value="model.task_id" />
        </el-select>
        <el-input v-model="predictFeatureJson" class="predict-json" type="textarea" :rows="7" />
        <el-button type="primary" :loading="predicting" :disabled="!canPredict" @click="runPrediction">执行预测</el-button>
        <div v-if="predictionResult" class="prediction-result" aria-live="polite">
          <span>预测值</span>
          <strong>{{ predictionResult.prediction }}</strong>
          <template v-if="predictionResult.encoded_prediction != null">
            <span>类别编码</span>
            <strong>{{ predictionResult.encoded_prediction }}</strong>
          </template>
          <template v-if="predictionResult.probability != null">
            <span>预测概率</span>
            <strong>{{ formatMetric(predictionResult.probability) }}</strong>
          </template>
          <div v-if="hasPredictionDiagnostics" class="prediction-diagnostics" role="status">
            <span v-if="predictionResult.diagnostics?.unknown_input_keys.length">
              未知输入字段：{{ predictionResult.diagnostics.unknown_input_keys.join('、') }}
            </span>
            <span v-if="predictionResult.diagnostics?.missing_columns_imputed.length">
              已补齐缺失字段：{{ predictionResult.diagnostics.missing_columns_imputed.join('、') }}
            </span>
          </div>
        </div>
      </section>
    </div>

    <section class="panel">
      <div class="panel-title">
        <h3>任务结果</h3>
        <span>{{ taskStatus?.status ?? 'no task' }}</span>
      </div>
      <div class="result-grid">
        <pre>{{ prettyRetrieve }}</pre>
        <pre>{{ prettyTaskStatus }}</pre>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { List, Refresh } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus/es/components/message/index.mjs'

import {
  workflowApi,
  type AgentCorrectionValue,
  type WorkflowFeatureAction,
  type WorkflowFeatureResponse,
  type WorkflowModelRecord,
  type WorkflowPredictResponse,
  type WorkflowRetrieveResponse,
  type WorkflowTaskStatus,
  type WorkflowTrainEvent,
} from '@/api/workflow'
import type { ProblemSpec, ProblemType } from '@/api/problem'

const retrieveForm = reactive({
  user_query: '查询所有司机订单数据，并包含司机年龄和城市信息',
  target_col: 'is_fraud',
})
const trainConfig = reactive({
  nTrials: 10,
  cvFolds: 3,
})
const predictForm = reactive({
  model_version: '',
})
const featureMemo = ref('')
const problemType = ref<ProblemType>('auto')
const predictFeatureJson = ref('{\n  "order_fee": 128.5,\n  "age": 35,\n  "city_code": "北京"\n}')
const taskId = ref('')
const retrieveResult = ref<WorkflowRetrieveResponse | null>(null)
const featureResult = ref<WorkflowFeatureResponse | null>(null)
const taskStatus = ref<WorkflowTaskStatus | null>(null)
const liveMetric = ref<{ name?: string; value?: number | null } | null>(null)
const predictionResult = ref<WorkflowPredictResponse | null>(null)
const modelRecords = ref<WorkflowModelRecord[]>([])
const trainEvents = ref<WorkflowTrainEvent[]>([])
const retrieving = ref(false)
const generatingFeatures = ref(false)
const training = ref(false)
const predicting = ref(false)
const refreshingStatus = ref(false)
const loadingModels = ref(false)
const errorMessage = ref('')
const infoMessage = ref('')
const modelListVisible = ref(false)
let streamController: AbortController | null = null
let contextGeneration = 0
let streamGeneration = 0
let predictionGeneration = 0

const stepState = computed(() => ({
  retrieve: taskId.value ? '完成' : '待执行',
  features: featureResult.value ? '完成' : '待执行',
}))
const canGenerateFeatures = computed(() => Boolean(taskId.value) && !generatingFeatures.value)
const canTrain = computed(() => Boolean(taskId.value && featureResult.value) && !training.value)
const canPredict = computed(() => Boolean(
  predictForm.model_version && predictFeatureJson.value.trim() && !predicting.value,
))
const featureActions = computed<WorkflowFeatureAction[]>(() => featureResult.value?.strategy?.actions ?? [])
const taskMetricName = computed(() => liveMetric.value?.name ?? taskStatus.value?.best_metric ?? (typeof taskStatus.value?.best_auc === 'number' ? 'roc_auc' : undefined))
const taskMetricValue = computed(() => liveMetric.value?.value ?? taskStatus.value?.best_value ?? taskStatus.value?.best_auc)
const hasPredictionDiagnostics = computed(() => Boolean(
  predictionResult.value?.diagnostics?.unknown_input_keys.length
  || predictionResult.value?.diagnostics?.missing_columns_imputed.length,
))
const trainingProgress = computed(() => {
  if (trainEvents.value.some((event) => event.status === 'DONE')) return 100
  const latestTrial = trainEvents.value.reduce((max, event) => Math.max(max, Number(event.trial ?? 0)), 0)
  return Math.min(95, Math.round((latestTrial / Math.max(trainConfig.nTrials, 1)) * 100))
})
const prettyRetrieve = computed(() => JSON.stringify(retrieveResult.value ?? {}, null, 2))
const prettyTaskStatus = computed(() => JSON.stringify(taskStatus.value ?? {}, null, 2))

async function runRetrieve() {
  retrieving.value = true
  errorMessage.value = ''
  infoMessage.value = ''
  try {
    const result = await workflowApi.retrieveData({ ...retrieveForm })
    if (result.status !== 'success' || !result.task_id) {
      throw new Error(result.msg ?? 'Step 1 未成功返回 task_id')
    }
    contextGeneration += 1
    invalidateTrainingStream()
    invalidatePrediction()
    const generation = contextGeneration
    retrieveResult.value = result
    taskId.value = result.task_id
    featureResult.value = null
    taskStatus.value = null
    liveMetric.value = null
    trainEvents.value = []
    ElMessage.success('数据提取完成')
    await refreshTask(result.task_id, generation)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '自然语言取数失败'
  } finally {
    retrieving.value = false
  }
}

async function runFeatureGeneration() {
  if (!taskId.value) return
  const expectedTaskId = taskId.value
  const generation = contextGeneration
  generatingFeatures.value = true
  errorMessage.value = ''
  infoMessage.value = ''
  try {
    const result = await workflowApi.generateFeatures({
      task_id: expectedTaskId,
      eda_report_text: featureMemo.value,
    })
    if (!isCurrentContext(expectedTaskId, generation)) return
    featureResult.value = result
    ElMessage.success('特征策略已生成')
    await refreshTask(expectedTaskId, generation)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '特征策略生成失败'
  } finally {
    generatingFeatures.value = false
  }
}

function startTraining() {
  if (!taskId.value || training.value) return
  closeTrainingStream()
  const expectedTaskId = taskId.value
  const generation = contextGeneration
  const activeStreamGeneration = ++streamGeneration
  trainEvents.value = []
  liveMetric.value = null
  training.value = true
  errorMessage.value = ''
  infoMessage.value = ''
  const controller = new AbortController()
  streamController = controller
  void workflowApi.streamTraining(expectedTaskId, trainConfig.nTrials, trainConfig.cvFolds, problemType.value, async (payload: WorkflowTrainEvent) => {
    if (!isCurrentStream(expectedTaskId, generation, activeStreamGeneration)) return
    trainEvents.value.unshift(payload)
    applyTrainingMetadata(payload)
    if (payload.status === 'DONE') {
      training.value = false
      closeTrainingStream(controller)
      await refreshTask(expectedTaskId, generation)
      if (!isCurrentStream(expectedTaskId, generation, activeStreamGeneration)) return
      await loadModels(generation)
      if (!isCurrentStream(expectedTaskId, generation, activeStreamGeneration)) return
      predictForm.model_version = expectedTaskId
    }
    if (payload.status === 'ERROR') {
      training.value = false
      errorMessage.value = payload.error ?? '训练失败'
      closeTrainingStream(controller)
    }
    if (payload.status === 'CANCELLED') {
      training.value = false
      errorMessage.value = payload.error ?? '训练已取消'
      closeTrainingStream(controller)
    }
  }, controller.signal).catch((error: unknown) => {
    if (controller.signal.aborted) return
    training.value = false
    errorMessage.value = error instanceof Error ? error.message : '训练流连接中断'
    if (streamController === controller) streamController = null
  })
}

async function refreshTask(expectedTaskId = taskId.value, generation = contextGeneration) {
  errorMessage.value = ''
  if (!expectedTaskId) {
    taskStatus.value = null
    infoMessage.value = '暂无任务可刷新，请先完成自然语言取数'
    return
  }
  refreshingStatus.value = true
  try {
    const status = await workflowApi.getTaskStatus(expectedTaskId)
    if (!isCurrentContext(expectedTaskId, generation)) return
    taskStatus.value = status
    if ((status.status === 'failed' || status.status === 'cancelled') && status.error) {
      errorMessage.value = status.error
    }
    infoMessage.value = '任务状态已刷新'
  } catch (error) {
    if (!isCurrentContext(expectedTaskId, generation)) return
    errorMessage.value = error instanceof Error ? error.message : '任务状态刷新失败'
  } finally {
    refreshingStatus.value = false
  }
}

async function openModelList() {
  modelListVisible.value = true
  await loadModels()
}

async function loadModels(generation = contextGeneration) {
  loadingModels.value = true
  errorMessage.value = ''
  try {
    const models = (await workflowApi.listModels()).models
    if (generation !== contextGeneration) return
    modelRecords.value = models
    if (!predictForm.model_version && modelRecords.value[0]) {
      predictForm.model_version = modelRecords.value[0].task_id
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '模型列表加载失败'
  } finally {
    loadingModels.value = false
  }
}

function selectModel(model: WorkflowModelRecord) {
  predictForm.model_version = model.task_id
  infoMessage.value = `已选择模型 ${model.task_id}`
}

async function runPrediction() {
  if (!canPredict.value) return
  const expectedModelId = predictForm.model_version
  const jsonSnapshot = predictFeatureJson.value
  const expectedTaskId = taskId.value
  const expectedContextGeneration = contextGeneration
  const activePredictionGeneration = ++predictionGeneration
  predicting.value = true
  errorMessage.value = ''
  infoMessage.value = ''
  predictionResult.value = null
  try {
    const response = await workflowApi.predict({
      model_version: expectedModelId,
      features: JSON.parse(jsonSnapshot) as Record<string, unknown>,
    })
    if (!isCurrentPrediction(
      activePredictionGeneration,
      expectedModelId,
      jsonSnapshot,
      expectedTaskId,
      expectedContextGeneration,
    )) return
    predictionResult.value = response
  } catch (error) {
    if (!isCurrentPrediction(
      activePredictionGeneration,
      expectedModelId,
      jsonSnapshot,
      expectedTaskId,
      expectedContextGeneration,
    )) return
    errorMessage.value = error instanceof Error ? error.message : '预测失败，请检查 JSON 和模型版本'
  } finally {
    if (isCurrentPrediction(
      activePredictionGeneration,
      expectedModelId,
      jsonSnapshot,
      expectedTaskId,
      expectedContextGeneration,
    )) predicting.value = false
  }
}

function formatMetric(value: number | null | undefined) {
  return typeof value === 'number' ? value.toFixed(4) : '-'
}

function applyTrainingMetadata(payload: WorkflowTrainEvent) {
  if (payload.metric !== undefined || payload.value !== undefined) {
    liveMetric.value = { name: payload.metric, value: payload.value }
  } else if (payload.best_metric !== undefined || payload.best_value !== undefined) {
    liveMetric.value = { name: payload.best_metric, value: payload.best_value }
  }
  if (!payload.problem_spec && !payload.best_metric && payload.best_value == null) return
  const current = taskStatus.value ?? { task_id: taskId.value, status: 'running' }
  taskStatus.value = {
    ...current,
    ...(payload.problem_spec ? { problem_spec: payload.problem_spec } : {}),
    ...(payload.suggested_type !== undefined ? { suggested_type: payload.suggested_type } : {}),
    ...(payload.agent_corrections ? { agent_corrections: payload.agent_corrections } : {}),
    ...(payload.best_metric ? { best_metric: payload.best_metric } : {}),
    ...(payload.best_value != null ? { best_value: payload.best_value } : {}),
  }
}

function metricLabel(metric: string | undefined) {
  const labels: Record<string, string> = {
    roc_auc: 'ROC-AUC',
    val_auc: 'ROC-AUC',
    accuracy: 'Accuracy',
    f1: 'F1',
    rmse: 'RMSE',
    mae: 'MAE',
    r2: 'R²',
  }
  return metric ? (labels[metric] ?? metric.toUpperCase()) : '最佳指标'
}

function problemTypeLabel(value: string) {
  const labels: Record<string, string> = {
    auto: '自动识别',
    binary_classification: '二分类',
    multiclass_classification: '多分类',
    regression: '回归',
  }
  return labels[value] ?? value
}

function problemSourceLabel(source: ProblemSpec['source']) {
  const labels: Record<ProblemSpec['source'], string> = {
    auto_inferred: '自动推断',
    user_override: '用户指定',
    legacy_compatibility: '旧模型兼容',
  }
  return labels[source]
}

function formatCorrectionValue(value: AgentCorrectionValue) {
  if (value === null) return '无'
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

function isCurrentContext(expectedTaskId: string, generation: number) {
  return taskId.value === expectedTaskId && contextGeneration === generation
}

function isCurrentStream(expectedTaskId: string, generation: number, expectedStreamGeneration: number) {
  return isCurrentContext(expectedTaskId, generation) && streamGeneration === expectedStreamGeneration
}

function isCurrentPrediction(
  expectedPredictionGeneration: number,
  expectedModelId: string,
  expectedJson: string,
  expectedTaskId: string,
  expectedContextGeneration: number,
) {
  return predictionGeneration === expectedPredictionGeneration
    && predictForm.model_version === expectedModelId
    && predictFeatureJson.value === expectedJson
    && taskId.value === expectedTaskId
    && contextGeneration === expectedContextGeneration
}

function invalidatePrediction() {
  predictionGeneration += 1
  predictionResult.value = null
  predicting.value = false
  errorMessage.value = ''
}

function closeTrainingStream(expectedController?: AbortController) {
  if (streamController && (!expectedController || streamController === expectedController)) {
    streamController.abort()
    streamController = null
  }
}

function invalidateTrainingStream() {
  streamGeneration += 1
  closeTrainingStream()
  training.value = false
}

function handleProblemTypeChange() {
  contextGeneration += 1
  invalidateTrainingStream()
  invalidatePrediction()
  taskStatus.value = null
  liveMetric.value = null
  trainEvents.value = []
  predictForm.model_version = ''
  infoMessage.value = ''
}

onMounted(loadModels)
onBeforeUnmount(closeTrainingStream)

watch([() => predictForm.model_version, predictFeatureJson], invalidatePrediction)
</script>

<style scoped>
.workflow-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.workflow-heading,
.panel-title,
.heading-actions,
.train-controls,
.prediction-result {
  display: flex;
  align-items: center;
  gap: 12px;
}

.workflow-heading,
.panel-title {
  justify-content: space-between;
}

.heading-actions {
  flex-wrap: wrap;
  justify-content: flex-end;
}

.workflow-status-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.status-item,
.panel {
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
}

.status-item {
  min-width: 0;
  padding: 16px;
}

.status-item span,
.panel-title span {
  color: #64748b;
  font-size: 12px;
}

.status-item strong {
  display: block;
  overflow: hidden;
  margin-top: 5px;
  color: #0f172a;
  font-size: 19px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.workflow-grid,
.result-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.panel {
  min-width: 0;
  padding: 18px;
}

.panel-title {
  margin-bottom: 14px;
}

.panel-title h3 {
  margin: 0;
  color: #111827;
  font-size: 16px;
}

.step-button,
.feature-table,
.event-table,
.predict-json {
  margin-top: 12px;
}

.train-controls {
  flex-wrap: wrap;
  margin-bottom: 14px;
}

.train-field {
  display: grid;
  min-width: 160px;
  gap: 4px;
}

.train-field > span {
  color: #64748b;
  font-size: 12px;
}

.problem-spec-summary,
.correction-list,
.correction-item,
.prediction-diagnostics {
  display: grid;
  gap: 7px;
}

.problem-spec-summary span,
.correction-item span,
.correction-item small,
.prediction-diagnostics span {
  color: #475569;
}

.correction-list {
  margin-top: 16px;
}

.correction-list h4 {
  margin: 0;
  color: #0f172a;
}

.correction-item {
  border-left: 3px solid #f59e0b;
  background: #fffbeb;
  padding: 10px 12px;
}

.prediction-diagnostics {
  margin-top: 8px;
  border-top: 1px solid #e2e8f0;
  padding-top: 10px;
}

.prediction-result {
  justify-content: space-between;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  margin-top: 14px;
  padding: 12px;
}

.prediction-result span {
  color: #64748b;
}

pre {
  overflow: auto;
  max-height: 300px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
  margin: 0;
  padding: 12px;
  color: #334155;
}

@media (max-width: 1080px) {
  .workflow-grid,
  .workflow-status-strip,
  .result-grid {
    grid-template-columns: 1fr;
  }

  .workflow-heading,
  .panel-title {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
