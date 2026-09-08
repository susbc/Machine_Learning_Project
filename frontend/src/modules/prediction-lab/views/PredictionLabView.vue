<template>
  <section class="prediction-workspace">
    <div class="section-heading prediction-heading">
      <div>
        <p class="eyebrow">Phase 8</p>
        <h2>预测实验室</h2>
        <p>选择已保存模型进行单条 JSON 预测，后续可扩展批量预测、阈值分析和结果导出。</p>
      </div>
      <el-button
        data-testid="prediction-refresh-models"
        aria-label="刷新预测模型"
        plain
        :loading="loadingModels"
        @click="loadModels"
      >
        刷新模型
      </el-button>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" />

    <div class="prediction-grid">
      <section class="panel">
        <div class="panel-title">
          <h3>输入</h3>
          <span>{{ selectedModelId || '未选择模型' }}</span>
        </div>
        <label class="field-row">
          <span>模型版本</span>
          <el-select v-model="selectedModelId" aria-label="预测模型" filterable placeholder="选择模型">
            <el-option v-for="model in models" :key="model.model_id" :label="model.model_id" :value="model.model_id" />
          </el-select>
        </label>
        <el-tabs v-model="inputMode" class="prediction-tabs">
          <el-tab-pane label="单条预测" name="single">
            <label class="field-row">
              <span>特征 JSON</span>
              <el-input v-model="featureJson" aria-label="特征 JSON" type="textarea" :rows="8" />
            </label>
            <el-button type="primary" :loading="predicting" :disabled="!canPredict" @click="runPrediction">执行预测</el-button>
          </el-tab-pane>
          <el-tab-pane label="批量预测" name="batch">
            <label class="field-row">
              <span>CSV 特征表</span>
              <input
                data-testid="prediction-csv-input"
                aria-label="CSV 特征表"
                class="file-input"
                type="file"
                accept=".csv,text/csv"
                @change="onCsvSelected"
              />
            </label>
            <div class="batch-preview">
              <span>已解析 {{ batchRows.length }} 行</span>
              <el-button type="primary" :loading="batchPredicting" :disabled="!canBatchPredict" @click="runBatchPrediction">
                执行批量预测
              </el-button>
            </div>
          </el-tab-pane>
        </el-tabs>
      </section>

      <section class="panel">
        <div class="panel-title">
          <h3>预测结果</h3>
          <span>{{ predictionResult?.source ?? '等待预测' }}</span>
        </div>
        <div v-if="predictionResult" class="result-card" aria-live="polite">
          <span>预测值</span>
          <strong>{{ predictionResult.prediction }}</strong>
          <template v-if="!isRegressionModel && predictionResult.encoded_prediction != null">
            <span>类别编码</span>
            <strong>{{ predictionResult.encoded_prediction }}</strong>
          </template>
          <template v-if="!isRegressionModel && predictionResult.probability != null">
            <span>预测概率</span>
            <strong>{{ formatMetric(predictionResult.probability) }}</strong>
          </template>
          <div v-if="hasDiagnostics(predictionResult.diagnostics)" class="diagnostics" role="status">
            <span v-if="predictionResult.diagnostics.unknown_input_keys.length">
              未知输入字段：{{ predictionResult.diagnostics.unknown_input_keys.join('、') }}
            </span>
            <span v-if="predictionResult.diagnostics.missing_columns_imputed.length">
              已补齐缺失字段：{{ predictionResult.diagnostics.missing_columns_imputed.join('、') }}
            </span>
          </div>
        </div>
        <el-empty v-else description="暂无预测结果" />
      </section>
    </div>

    <section v-if="batchResult" class="panel">
      <div class="panel-title">
        <h3>批量结果</h3>
        <span>{{ batchResult.summary.count }} 行</span>
      </div>
      <div v-if="!isRegressionModel" class="summary-grid">
        <div>
          <span>正例占比</span>
          <strong>{{ formatMetric(batchResult.summary.positive_rate) }}</strong>
        </div>
        <div>
          <span>平均概率</span>
          <strong>{{ formatMetric(batchResult.summary.average_probability) }}</strong>
        </div>
      </div>
      <el-table :data="batchResult.rows" height="360">
        <el-table-column prop="index" label="#" width="80" />
        <el-table-column prop="prediction" label="预测值" width="120" />
        <el-table-column v-if="!isRegressionModel" prop="encoded_prediction" label="类别编码" width="110" />
        <el-table-column v-if="!isRegressionModel" label="预测概率" width="140">
          <template #default="{ row }">{{ formatMetric(row.probability) }}</template>
        </el-table-column>
        <el-table-column label="输入诊断" min-width="180">
          <template #default="{ row }">
            <div v-if="hasDiagnostics(row.diagnostics)" class="row-diagnostics">
              <span v-if="row.diagnostics.unknown_input_keys.length">未知：{{ row.diagnostics.unknown_input_keys.join('、') }}</span>
              <span v-if="row.diagnostics.missing_columns_imputed.length">缺失：{{ row.diagnostics.missing_columns_imputed.join('、') }}</span>
            </div>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column label="输入特征">
          <template #default="{ row }">{{ JSON.stringify(row.features) }}</template>
        </el-table-column>
      </el-table>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import {
  modelsApi,
  type ModelBatchPredictionResponse,
  type ModelPredictionResponse,
  type ModelRegistryRecord,
  type PredictionDiagnostics,
} from '@/api/models'

const models = ref<ModelRegistryRecord[]>([])
const selectedModelId = ref('')
const inputMode = ref<'single' | 'batch'>('single')
const featureJson = ref('{\n  "age": 35,\n  "order_fee": 128.5,\n  "city_code": "北京"\n}')
const predictionResult = ref<ModelPredictionResponse | null>(null)
const batchRows = ref<Record<string, unknown>[]>([])
const batchResult = ref<ModelBatchPredictionResponse | null>(null)
const loadingModels = ref(false)
const predicting = ref(false)
const batchPredicting = ref(false)
const errorMessage = ref('')
let predictionRequestGeneration = 0
let batchRequestGeneration = 0
let csvReadGeneration = 0

const canPredict = computed(() => Boolean(selectedModelId.value && featureJson.value.trim()))
const canBatchPredict = computed(() => Boolean(selectedModelId.value && batchRows.value.length > 0))
const selectedModel = computed(() => models.value.find((model) => model.model_id === selectedModelId.value) ?? null)
const isRegressionModel = computed(() => selectedModel.value?.problem_type === 'regression')

async function loadModels() {
  loadingModels.value = true
  errorMessage.value = ''
  try {
    const records = (await modelsApi.listModels()).models
      .filter((model) => !model.metadata_error)
    models.value = records
    selectedModelId.value = records.some((model) => model.model_id === selectedModelId.value)
      ? selectedModelId.value
      : (records[0]?.model_id ?? '')
    clearResults()
  } catch (error) {
    models.value = []
    selectedModelId.value = ''
    clearResults()
    errorMessage.value = error instanceof Error ? error.message : '模型列表加载失败'
  } finally {
    loadingModels.value = false
  }
}

async function runPrediction() {
  const requestGeneration = ++predictionRequestGeneration
  const requestedModelId = selectedModelId.value
  const requestedFeatureJson = featureJson.value
  predicting.value = true
  errorMessage.value = ''
  predictionResult.value = null
  try {
    const result = await modelsApi.predict(requestedModelId, {
      features: JSON.parse(requestedFeatureJson) as Record<string, unknown>,
    })
    if (
      requestGeneration !== predictionRequestGeneration
      || requestedModelId !== selectedModelId.value
      || requestedFeatureJson !== featureJson.value
    ) return
    predictionResult.value = result
  } catch (error) {
    if (requestGeneration !== predictionRequestGeneration) return
    errorMessage.value = error instanceof Error ? error.message : '预测失败，请检查模型和 JSON'
  } finally {
    if (requestGeneration === predictionRequestGeneration) predicting.value = false
  }
}

function parseCsvLine(line: string): string[] {
  const cells: string[] = []
  let current = ''
  let inQuotes = false
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index]
    const next = line[index + 1]
    if (char === '"' && inQuotes && next === '"') {
      current += '"'
      index += 1
    } else if (char === '"') {
      inQuotes = !inQuotes
    } else if (char === ',' && !inQuotes) {
      cells.push(current.trim())
      current = ''
    } else {
      current += char
    }
  }
  cells.push(current.trim())
  return cells
}

function coerceCsvValue(value: string): unknown {
  if (value === '') return ''
  const numeric = Number(value)
  if (Number.isFinite(numeric)) return numeric
  if (value.toLowerCase() === 'true') return true
  if (value.toLowerCase() === 'false') return false
  return value
}

function parseCsvText(text: string) {
  invalidateBatchResult()
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
  const [headerLine, ...dataLines] = lines
  if (!headerLine || dataLines.length === 0) {
    batchRows.value = []
    return
  }
  const headers = parseCsvLine(headerLine)
  batchRows.value = dataLines.map((line) => {
    const values = parseCsvLine(line)
    return headers.reduce<Record<string, unknown>>((row, header, index) => {
      row[header] = coerceCsvValue(values[index] ?? '')
      return row
    }, {})
  })
}

async function onCsvSelected(event: Event) {
  const readGeneration = ++csvReadGeneration
  batchRows.value = []
  invalidateBatchResult()
  errorMessage.value = ''
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  try {
    const text = await file.text()
    if (readGeneration !== csvReadGeneration) return
    parseCsvText(text)
  } catch (error) {
    if (readGeneration !== csvReadGeneration) return
    errorMessage.value = error instanceof Error ? error.message : 'CSV 文件读取失败'
  }
}

async function runBatchPrediction() {
  const requestGeneration = ++batchRequestGeneration
  const requestedModelId = selectedModelId.value
  const requestedRows = batchRows.value
  batchPredicting.value = true
  errorMessage.value = ''
  batchResult.value = null
  try {
    const result = await modelsApi.batchPredict(requestedModelId, { rows: requestedRows })
    if (
      requestGeneration !== batchRequestGeneration
      || requestedModelId !== selectedModelId.value
      || requestedRows !== batchRows.value
    ) return
    batchResult.value = result
  } catch (error) {
    if (requestGeneration !== batchRequestGeneration) return
    errorMessage.value = error instanceof Error ? error.message : '批量预测失败，请检查模型和 CSV'
  } finally {
    if (requestGeneration === batchRequestGeneration) batchPredicting.value = false
  }
}

function formatMetric(value: number | null | undefined) {
  return typeof value === 'number' ? value.toFixed(4) : '-'
}

function hasDiagnostics(diagnostics: PredictionDiagnostics | null | undefined) {
  return Boolean(diagnostics?.unknown_input_keys.length || diagnostics?.missing_columns_imputed.length)
}

function clearResults() {
  predictionRequestGeneration += 1
  batchRequestGeneration += 1
  predictionResult.value = null
  batchResult.value = null
  predicting.value = false
  batchPredicting.value = false
}

function invalidateSingleResult() {
  predictionRequestGeneration += 1
  predictionResult.value = null
  predicting.value = false
}

function invalidateBatchResult() {
  batchRequestGeneration += 1
  batchResult.value = null
  batchPredicting.value = false
}

watch(selectedModelId, clearResults)
watch(featureJson, invalidateSingleResult)

onMounted(loadModels)
</script>

<style scoped>
.prediction-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.prediction-heading,
.panel-title {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.prediction-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 0.7fr);
  gap: 18px;
}

.panel {
  min-width: 0;
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

.panel-title span,
.field-row > span,
.result-card span {
  color: #64748b;
  font-size: 12px;
}

.field-row {
  display: grid;
  gap: 6px;
  margin-bottom: 14px;
}

.field-row :deep(.el-select),
.field-row :deep(.el-textarea) {
  width: 100%;
}

.prediction-tabs {
  margin-top: 4px;
}

.file-input {
  width: 100%;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  padding: 9px 10px;
  color: #334155;
  background: #f8fafc;
}

.batch-preview {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: #64748b;
  font-size: 13px;
}

.result-card {
  display: grid;
  gap: 8px;
}

.result-card strong {
  color: #0f172a;
  font-size: 28px;
}

.diagnostics,
.row-diagnostics {
  display: grid;
  gap: 6px;
}

.diagnostics {
  margin-top: 8px;
  border-top: 1px solid #e2e8f0;
  padding-top: 12px;
}

.diagnostics span,
.row-diagnostics span {
  color: #475569;
  font-size: 12px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.summary-grid > div {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 14px;
}

.summary-grid span {
  display: block;
  color: #64748b;
  font-size: 12px;
  margin-bottom: 8px;
}

.summary-grid strong {
  color: #0f172a;
  font-size: 22px;
}

@media (max-width: 1100px) {
  .prediction-grid {
    grid-template-columns: 1fr;
  }

  .prediction-heading {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
