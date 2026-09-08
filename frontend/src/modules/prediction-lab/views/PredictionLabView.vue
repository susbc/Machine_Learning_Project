<template>
  <section class="prediction-workspace">
    <div class="section-heading prediction-heading">
      <div>
        <p class="eyebrow">Phase 8</p>
        <h2>预测实验室</h2>
        <p>选择已保存模型进行单条 JSON 预测，后续可扩展批量预测、阈值分析和结果导出。</p>
      </div>
      <el-button plain :loading="loadingModels" @click="loadModels">刷新模型</el-button>
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
          <el-select v-model="selectedModelId" filterable placeholder="选择模型">
            <el-option v-for="model in models" :key="model.model_id" :label="model.model_id" :value="model.model_id" />
          </el-select>
        </label>
        <el-tabs v-model="inputMode" class="prediction-tabs">
          <el-tab-pane label="单条预测" name="single">
            <label class="field-row">
              <span>特征 JSON</span>
              <el-input v-model="featureJson" type="textarea" :rows="8" />
            </label>
            <el-button type="primary" :loading="predicting" :disabled="!canPredict" @click="runPrediction">执行预测</el-button>
          </el-tab-pane>
          <el-tab-pane label="批量预测" name="batch">
            <label class="field-row">
              <span>CSV 特征表</span>
              <input class="file-input" type="file" accept=".csv,text/csv" @change="onCsvSelected" />
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
        <div v-if="predictionResult" class="result-card">
          <span>预测值</span>
          <strong>{{ predictionResult.prediction }}</strong>
          <span>概率</span>
          <strong>{{ predictionResult.probability == null ? '-' : predictionResult.probability.toFixed(4) }}</strong>
        </div>
        <el-empty v-else description="暂无预测结果" />
      </section>
    </div>

    <section v-if="batchResult" class="panel">
      <div class="panel-title">
        <h3>批量结果</h3>
        <span>{{ batchResult.summary.count }} 行</span>
      </div>
      <div class="summary-grid">
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
        <el-table-column label="概率" width="140">
          <template #default="{ row }">{{ row.probability == null ? '-' : row.probability.toFixed(4) }}</template>
        </el-table-column>
        <el-table-column label="输入特征">
          <template #default="{ row }">{{ JSON.stringify(row.features) }}</template>
        </el-table-column>
      </el-table>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

import { modelsApi, type ModelBatchPredictionResponse, type ModelPredictionResponse, type ModelRegistryRecord } from '@/api/models'

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

const canPredict = computed(() => Boolean(selectedModelId.value && featureJson.value.trim()))
const canBatchPredict = computed(() => Boolean(selectedModelId.value && batchRows.value.length > 0))

async function loadModels() {
  loadingModels.value = true
  errorMessage.value = ''
  try {
    models.value = (await modelsApi.listModels()).models.filter((model) => model.source !== 'legacy')
    if (!selectedModelId.value && models.value[0]) {
      selectedModelId.value = models.value[0].model_id
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '模型列表加载失败'
  } finally {
    loadingModels.value = false
  }
}

async function runPrediction() {
  predicting.value = true
  errorMessage.value = ''
  try {
    predictionResult.value = await modelsApi.predict(selectedModelId.value, {
      features: JSON.parse(featureJson.value) as Record<string, unknown>,
    })
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '预测失败，请检查模型和 JSON'
  } finally {
    predicting.value = false
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
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  parseCsvText(await file.text())
}

async function runBatchPrediction() {
  batchPredicting.value = true
  errorMessage.value = ''
  try {
    batchResult.value = await modelsApi.batchPredict(selectedModelId.value, { rows: batchRows.value })
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '批量预测失败，请检查模型和 CSV'
  } finally {
    batchPredicting.value = false
  }
}

function formatMetric(value: number | null) {
  return typeof value === 'number' ? value.toFixed(4) : '-'
}

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
