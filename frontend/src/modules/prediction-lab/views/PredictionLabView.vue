<template>
  <section class="workspace-theme prediction-workspace">
    <div class="section-heading prediction-heading">
      <div>
        <p class="eyebrow">Phase 8</p>
        <h2>预测实验室</h2>
        <p>选择已保存模型进行单条 JSON 预测，支持 CSV 批量预测和结果导出。</p>
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
              <span>特征来源</span>
              <el-radio-group v-model="batchSource" aria-label="批量特征来源">
                <el-radio-button value="csv">CSV 文件</el-radio-button>
                <el-radio-button value="database">数据库表</el-radio-button>
              </el-radio-group>
            </label>
            <label v-if="batchSource === 'csv'" class="field-row">
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
            <template v-else>
              <label class="field-row">
                <span>数据库连接（默认使用激活连接）</span>
                <el-select
                  v-model="batchConnectionId"
                  aria-label="批量数据库连接"
                  filterable
                  placeholder="选择数据库连接"
                  :loading="loadingConnections"
                >
                  <el-option
                    v-for="connection in connections"
                    :key="connection.id"
                    :label="`${connection.name}（${connection.db_type}）`"
                    :value="connection.id"
                  />
                </el-select>
              </label>
              <label class="field-row">
                <span>特征表</span>
                <el-select
                  v-model="batchTableName"
                  aria-label="批量特征表"
                  filterable
                  placeholder="选择数据表"
                  :loading="loadingTables"
                >
                  <el-option v-for="table in tableOptions" :key="table" :label="table" :value="table" />
                </el-select>
              </label>
              <div class="batch-preview import-controls">
                <span>导入行数上限</span>
                <el-input-number v-model="batchImportLimit" aria-label="导入行数上限" :min="1" :max="10000" :step="100" />
                <el-button plain :loading="importingRows" :disabled="!canImportRows" @click="importBatchRowsFromDatabase">
                  从数据库导入
                </el-button>
              </div>
            </template>
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
        <span>{{ batchResult.summary.count }} 行（预览前 200 行）</span>
        <el-button plain @click="exportBatchResults">导出全部结果</el-button>
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
      <el-table :data="batchResult.rows.slice(0, 200)" height="360">
        <el-table-column prop="index" label="#" width="80" />
        <el-table-column prop="prediction" label="预测值" width="150" show-overflow-tooltip />
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
        <el-table-column label="输入特征" min-width="260" show-overflow-tooltip>
          <template #default="{ row }">{{ JSON.stringify(row.features) }}</template>
        </el-table-column>
      </el-table>
    </section>

    <section v-if="hasExportableResults" class="panel">
      <div class="panel-title">
        <h3>结果导出</h3>
        <span>{{ exportScope === 'batch' ? `批量结果 ${exportRows.length} 行` : '单条预测 1 行' }}</span>
      </div>
      <label class="field-row">
        <span>导出内容</span>
        <el-radio-group v-model="exportScope" aria-label="导出内容">
          <el-radio-button v-if="batchResult" value="batch">批量结果</el-radio-button>
          <el-radio-button v-if="predictionResult" value="single">单条预测</el-radio-button>
        </el-radio-group>
      </label>
      <label class="field-row">
        <span>导出目标</span>
        <el-radio-group v-model="exportTarget" aria-label="导出目标">
          <el-radio-button value="file">服务器文件</el-radio-button>
          <el-radio-button value="database">数据库</el-radio-button>
        </el-radio-group>
      </label>
      <label v-if="exportTarget === 'file'" class="field-row">
        <span>导出目录（相对项目根目录，不存在时自动创建）</span>
        <el-input v-model="exportDir" aria-label="导出目录" placeholder="默认 output" clearable />
      </label>
      <template v-else>
        <label class="field-row">
          <span>数据库连接（默认使用激活连接）</span>
          <el-select
            v-model="exportConnectionId"
            aria-label="导出数据库连接"
            filterable
            placeholder="选择数据库连接"
            :loading="loadingConnections"
          >
            <el-option
              v-for="connection in connections"
              :key="connection.id"
              :label="`${connection.name}（${connection.db_type}）`"
              :value="connection.id"
            />
          </el-select>
        </label>
        <label class="field-row">
          <span>表名（表不存在时按默认结构自动创建）</span>
          <el-input v-model="exportTableName" aria-label="导出表名" placeholder="默认 prediction_results" clearable />
        </label>
      </template>
      <div class="batch-preview">
        <span>{{ exportTarget === 'file' ? '以 CSV 文件写入服务器指定目录' : '将预测结果写入指定数据库表' }}</span>
        <el-button type="primary" :loading="exporting" :disabled="!canExport" @click="runExport">导出结果</el-button>
      </div>
      <el-alert v-if="exportNotice" :type="exportNotice.type" :title="exportNotice.text" show-icon :closable="false" />
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { parseFeatureCsv } from '../csv'

import { databaseApi, type DatabaseConnection } from '@/api/database'
import {
  modelsApi,
  type ModelBatchPredictionResponse,
  type ModelPredictionResponse,
  type ModelRegistryRecord,
  type PredictionDiagnostics,
  type PredictionExportResponse,
  type PredictionExportRow,
  type PredictionExportTarget,
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
const connections = ref<DatabaseConnection[]>([])
const exportScope = ref<'single' | 'batch'>('batch')
const exportTarget = ref<PredictionExportTarget>('file')
const exportDir = ref('')
const exportConnectionId = ref('')
const exportTableName = ref('')
const exporting = ref(false)
const loadingConnections = ref(false)
const exportNotice = ref<{ type: 'success' | 'warning'; text: string } | null>(null)
const batchSource = ref<'csv' | 'database'>('csv')
const batchConnectionId = ref('')
const batchTableName = ref('')
const batchImportLimit = ref(1000)
const tableOptions = ref<string[]>([])
const importingRows = ref(false)
const loadingTables = ref(false)
let predictionRequestGeneration = 0
let batchRequestGeneration = 0
let csvReadGeneration = 0

const canPredict = computed(() => Boolean(selectedModelId.value && featureJson.value.trim()))
const canBatchPredict = computed(() => Boolean(selectedModelId.value && batchRows.value.length > 0))
const selectedModel = computed(() => models.value.find((model) => model.model_id === selectedModelId.value) ?? null)
const isRegressionModel = computed(() => selectedModel.value?.problem_type === 'regression')
const hasExportableResults = computed(() => Boolean(predictionResult.value || batchResult.value))
const exportRows = computed<PredictionExportRow[]>(() => {
  if (exportScope.value === 'batch' && batchResult.value) {
    return batchResult.value.rows.map((row) => ({
      index: row.index,
      features: row.features,
      prediction: row.prediction,
      encoded_prediction: row.encoded_prediction,
      probability: row.probability,
      diagnostics: row.diagnostics,
    }))
  }
  if (predictionResult.value) {
    const result = predictionResult.value
    return [{
      index: 0,
      features: result.input_features,
      prediction: result.prediction,
      encoded_prediction: result.encoded_prediction,
      probability: result.probability,
      diagnostics: result.diagnostics,
    }]
  }
  return []
})
const canExport = computed(
  () => exportRows.value.length > 0 && (exportTarget.value === 'file' || Boolean(exportConnectionId.value)),
)
const canImportRows = computed(() => Boolean(batchConnectionId.value && batchTableName.value))

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

function parseCsvText(text: string) {
  invalidateBatchResult()
  batchRows.value = parseFeatureCsv(text)
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

function exportBatchResults() {
  if (!batchResult.value) return
  const rows = batchResult.value.rows
  const featureNames = Object.keys(rows[0]?.features ?? {})
  const escape = (value: unknown) => `"${String(value ?? '').replaceAll('"', '""')}"`
  const csv = [
    [...featureNames, 'prediction', 'probability'].map(escape).join(','),
    ...rows.map((row) => [...featureNames.map((name) => row.features[name]), row.prediction, row.probability].map(escape).join(',')),
  ].join('\r\n')
  const url = URL.createObjectURL(new Blob(['\uFEFF', csv], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = 'predictions.csv'
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function formatMetric(value: number | null | undefined) {
  return typeof value === 'number' ? value.toFixed(4) : '-'
}

async function loadConnections() {
  if (connections.value.length) return
  loadingConnections.value = true
  try {
    const payload = await databaseApi.listConnections()
    connections.value = payload.connections.filter((connection) => connection.is_active)
    const activeId = payload.active_connection?.id ?? connections.value[0]?.id ?? ''
    if (!exportConnectionId.value) exportConnectionId.value = activeId
    if (!batchConnectionId.value) batchConnectionId.value = activeId
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '数据库连接列表加载失败'
  } finally {
    loadingConnections.value = false
  }
}

async function loadTableOptions() {
  tableOptions.value = []
  batchTableName.value = ''
  if (!batchConnectionId.value) return
  loadingTables.value = true
  try {
    const schema = await databaseApi.getSchema(batchConnectionId.value, 1)
    tableOptions.value = schema.tables.map((table) => table.name)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '数据表列表加载失败'
  } finally {
    loadingTables.value = false
  }
}

async function importBatchRowsFromDatabase() {
  if (!canImportRows.value) return
  importingRows.value = true
  errorMessage.value = ''
  batchRows.value = []
  invalidateBatchResult()
  try {
    const payload = await databaseApi.getTableRows(batchConnectionId.value, batchTableName.value, batchImportLimit.value)
    batchRows.value = payload.rows
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '数据库表数据导入失败'
  } finally {
    importingRows.value = false
  }
}

function describeExportSuccess(result: PredictionExportResponse) {
  if (result.target === 'file') return `已导出 ${result.row_count} 行到 ${result.path}`
  const created = result.table_created ? '（表不存在，已自动创建）' : '（追加写入已有表）'
  return `已导出 ${result.row_count} 行到数据库表 ${result.table_name}${created}`
}

async function runExport() {
  const rows = exportRows.value
  if (!rows.length) return
  exporting.value = true
  exportNotice.value = null
  errorMessage.value = ''
  try {
    const result = await modelsApi.exportPredictions({
      model_id: selectedModelId.value,
      target: exportTarget.value,
      rows,
      output_dir: exportTarget.value === 'file' ? (exportDir.value.trim() || undefined) : undefined,
      connection_id: exportTarget.value === 'database' ? exportConnectionId.value : undefined,
      table_name: exportTarget.value === 'database' ? (exportTableName.value.trim() || undefined) : undefined,
    })
    exportNotice.value = result.exported
      ? { type: 'success', text: describeExportSuccess(result) }
      : { type: 'warning', text: result.warnings.join('；') || '导出未完成' }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '导出失败'
  } finally {
    exporting.value = false
  }
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
  exportNotice.value = null
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
watch([predictionResult, batchResult], () => {
  exportNotice.value = null
  if (exportScope.value === 'batch' && !batchResult.value) exportScope.value = 'single'
  if (exportScope.value === 'single' && !predictionResult.value) exportScope.value = 'batch'
})
watch(exportTarget, (target) => {
  if (target === 'database') void loadConnections()
})
watch(batchSource, (source) => {
  if (source === 'database') void loadConnections()
})
watch(batchConnectionId, () => {
  void loadTableOptions()
})

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

.import-controls {
  justify-content: flex-start;
  margin-bottom: 14px;
}

.import-controls :deep(.el-input-number) {
  width: 150px;
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
