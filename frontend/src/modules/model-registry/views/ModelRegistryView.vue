<template>
  <section class="workspace-theme model-registry-workspace">
    <div class="section-heading registry-heading">
      <div>
        <h2>模型管理</h2>
        <p>统一查看 AutoML、手动训练、Agent 工作流和 legacy 训练产物，为模型对比和预测实验室提供入口。</p>
      </div>
      <el-button
        data-testid="registry-refresh-button"
        aria-label="刷新模型注册表"
        type="primary"
        plain
        :loading="loading"
        @click="loadModels"
      >
        刷新模型
      </el-button>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" />

    <section v-if="detailModel" class="panel detail-panel">
      <div class="panel-title">
        <h3>模型详情 · {{ detailModel.model_id }}</h3>
        <el-tag :type="sourceTagType(detailModel.source)" effect="light">{{ sourceLabel(detailModel.source) }}</el-tag>
      </div>
      <div class="detail-actions">
        <el-button plain @click="backToList">
          <ArrowLeft :size="15" :stroke-width="1.9" />返回列表
        </el-button>
        <el-button plain :loading="weightsLoading" @click="loadWeights(detailModel)">
          <RefreshCw :size="15" :stroke-width="1.9" />刷新权重
        </el-button>
        <el-button
          type="danger"
          plain
          :loading="deletingId === detailModel.model_id"
          @click="confirmDelete(detailModel)"
        >
          <Trash2 :size="15" :stroke-width="1.9" />删除模型
        </el-button>
      </div>
      <div class="detail-info-strip">
        <div class="info-card">
          <span class="info-icon"><Box :size="16" :stroke-width="1.9" /></span>
          <div>
            <span>模型 ID</span>
            <strong>{{ detailModel.model_id }}</strong>
          </div>
        </div>
        <div class="info-card">
          <span class="info-icon"><FolderOpen :size="16" :stroke-width="1.9" /></span>
          <div>
            <span>产物路径</span>
            <strong>{{ detailModel.artifact_path }}</strong>
          </div>
        </div>
        <div class="info-card">
          <span class="info-icon"><Crosshair :size="16" :stroke-width="1.9" /></span>
          <div>
            <span>任务类型</span>
            <strong>{{ registryProblemTypeLabel(detailModel) }}</strong>
          </div>
        </div>
        <div class="info-card">
          <span class="info-icon"><Target :size="16" :stroke-width="1.9" /></span>
          <div>
            <span>主指标</span>
            <strong>{{ metricText(detailModel) }}</strong>
          </div>
        </div>
        <div class="info-card">
          <span class="info-icon"><Clock :size="16" :stroke-width="1.9" /></span>
          <div>
            <span>更新时间</span>
            <strong>{{ new Date(detailModel.updated_at * 1000).toLocaleString() }}</strong>
          </div>
        </div>
      </div>
      <el-alert
        v-if="detailModel.metadata_error"
        class="metadata-alert"
        type="error"
        :title="detailModel.metadata_error.message"
        :description="`${detailModel.metadata_error.code} · ${detailModel.metadata_error.field}`"
        show-icon
        :closable="false"
      />
      <section class="detail-block">
        <div class="block-head">
          <h4><ChartColumn :size="15" :stroke-width="1.9" />评估指标</h4>
          <span>基于测试集评估的核心指标</span>
        </div>
        <div class="metric-cards">
          <div
            v-for="entry in metricEntries"
            :key="entry.name"
            class="metric-card"
            :class="{ 'metric-card--primary': entry.primary }"
          >
            <div class="metric-card-head">
              <component :is="metricIcon(entry.name)" :size="16" :stroke-width="1.9" />
              <span>{{ metricLabel(entry.name) }}</span>
              <el-tag v-if="entry.primary" size="small" effect="dark" round>主指标</el-tag>
            </div>
            <strong>{{ entry.value.toFixed(4) }}</strong>
          </div>
        </div>
      </section>
      <section class="detail-block">
        <div class="block-head">
          <h4><Layers :size="15" :stroke-width="1.9" />特征重要性</h4>
          <span>基于模型权重的特征重要性（数值越大，特征越重要）· {{ weightsKindLabel }}</span>
        </div>
        <div class="importance-grid">
          <div class="importance-chart-box">
            <h5>特征重要性（Top {{ topWeights.length }}）</h5>
            <div ref="importanceChartElement" class="importance-chart" aria-label="特征重要性条形图" />
          </div>
          <el-table :data="importanceRows" height="300" empty-text="暂无权重数据，点击「刷新权重」加载">
            <el-table-column label="#" width="56">
              <template #default="{ row }">
                <span class="rank-badge" :class="`rank-badge--${Math.min(row.rank, 4)}`">{{ row.rank }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="feature" label="特征" min-width="150" show-overflow-tooltip />
            <el-table-column label="权重" width="110">
              <template #default="{ row }">{{ row.importance.toFixed(4) }}</template>
            </el-table-column>
            <el-table-column label="相对重要性" min-width="180">
              <template #default="{ row }">
                <div class="share-cell">
                  <span>{{ (row.share * 100).toFixed(2) }}%</span>
                  <el-progress :percentage="Number((row.share * 100).toFixed(1))" :show-text="false" :stroke-width="8" />
                </div>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </section>
      <section class="detail-block">
        <div class="block-head">
          <h4><Braces :size="15" :stroke-width="1.9" />指标详情（JSON）</h4>
          <div class="json-actions">
            <el-button text size="small" @click="copyJson">复制</el-button>
            <el-button text size="small" @click="showJson = !showJson">{{ showJson ? '收起' : '展开' }}</el-button>
          </div>
        </div>
        <pre v-show="showJson" class="json-pre">{{ prettyMetrics }}</pre>
      </section>
    </section>

    <section v-else-if="detailNotFound" class="panel">
      <el-alert
        type="warning"
        title="模型不存在或已删除"
        :description="`未找到 ${detailModelId}，可能已被删除或链接失效`"
        show-icon
        :closable="false"
      />
      <div class="detail-actions">
        <el-button plain @click="backToList">← 返回列表</el-button>
      </div>
    </section>

    <template v-else>
      <section class="registry-status-strip">
        <div class="status-item">
          <span>模型总数</span>
          <strong>{{ models.length }}</strong>
        </div>
        <div class="status-item">
          <span>AutoML</span>
          <strong>{{ countBySource('automl') }}</strong>
        </div>
        <div class="status-item">
          <span>手动训练</span>
          <strong>{{ countBySource('manual_training') }}</strong>
        </div>
        <div class="status-item">
          <span>Agent 工作流</span>
          <strong>{{ countBySource('agent_workflow') }}</strong>
        </div>
      </section>

      <section class="panel">
        <div class="panel-title">
          <h3>模型列表</h3>
          <span>{{ models.length }} 个模型 · 点击模型名称查看详情</span>
        </div>
        <el-table :data="models" height="420">
          <el-table-column label="模型" min-width="210">
            <template #default="{ row }">
              <div class="model-name-cell">
                <span
                  class="model-link"
                  role="link"
                  tabindex="0"
                  :aria-label="`查看 ${row.model_id} 详情`"
                  @click="openDetail(row)"
                  @keydown.enter="openDetail(row)"
                >{{ row.name }}</span>
                <el-tag v-if="row.legacy" size="small" type="info" effect="plain">Legacy</el-tag>
                <el-tag v-if="row.metadata_error" size="small" type="danger" effect="plain">元数据错误</el-tag>
                <small class="model-id-sub">{{ row.model_id }}</small>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="来源" width="130">
            <template #default="{ row }">
              <el-tag :type="sourceTagType(row.source)" effect="light">{{ sourceLabel(row.source) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="主指标" width="180">
            <template #default="{ row }">{{ metricText(row) }}</template>
          </el-table-column>
          <el-table-column label="任务类型" width="120">
            <template #default="{ row }">{{ registryProblemTypeLabel(row) }}</template>
          </el-table-column>
          <el-table-column label="元数据状态" min-width="200">
            <template #default="{ row }">
              <span v-if="row.metadata_error" class="metadata-error-text">
                {{ row.metadata_error.field }}：{{ row.metadata_error.message }}
              </span>
              <span v-else>{{ row.legacy ? '旧模型兼容模式' : '有效' }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="artifact_path" label="产物路径" min-width="260" show-overflow-tooltip />
          <el-table-column label="操作" width="110">
            <template #default="{ row }">
              <el-button text type="primary" @click="openDetail(row)">详情</el-button>
            </template>
          </el-table-column>
        </el-table>
      </section>

      <section class="panel compare-panel">
        <div class="panel-title">
          <h3>模型对比</h3>
          <span>已选择 {{ compareRows.length }} 个模型</span>
        </div>
        <el-checkbox-group v-model="compareModelIds" class="compare-selector">
          <el-checkbox
            v-for="model in models"
            :key="model.model_id"
            :value="model.model_id"
            :disabled="Boolean(model.metadata_error)"
          >
            {{ model.name }}
          </el-checkbox>
        </el-checkbox-group>
        <el-table :data="compareRows" height="320">
          <el-table-column prop="name" label="模型" min-width="180" />
          <el-table-column prop="source" label="来源" width="120" />
          <el-table-column prop="primaryMetric" label="主指标" width="120" />
          <el-table-column prop="rocAuc" label="ROC-AUC" width="120" />
          <el-table-column prop="accuracy" label="Accuracy" width="120" />
          <el-table-column prop="r2" label="R²" width="120" />
          <el-table-column prop="rmse" label="RMSE" width="120" />
          <el-table-column prop="mae" label="MAE" width="120" />
          <el-table-column prop="updatedAt" label="更新时间" min-width="160" />
        </el-table>
      </section>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch, type Component } from 'vue'
import { ElMessage } from 'element-plus/es/components/message/index.mjs'
import { ElMessageBox } from 'element-plus/es/components/message-box/index.mjs'
import {
  ArrowLeft,
  Award,
  Box,
  Braces,
  ChartColumn,
  Clock,
  Crosshair,
  FolderOpen,
  Layers,
  RefreshCw,
  RotateCcw,
  Scale,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  Trash2,
  TrendingDown,
  TrendingUp,
} from 'lucide-vue-next'
import { useRoute, useRouter } from 'vue-router'

import { modelsApi, type ModelRegistryRecord, type ModelWeightsResponse } from '@/api/models'
import { init, type ECharts } from '@/shared/charts/echarts'

const route = useRoute()
const router = useRouter()

const models = ref<ModelRegistryRecord[]>([])
const compareModelIds = ref<string[]>([])
const loading = ref(false)
const errorMessage = ref('')
const weightsResult = ref<ModelWeightsResponse | null>(null)
const weightsModelId = ref('')
const weightsLoading = ref(false)
const deletingId = ref('')
const showJson = ref(true)
const importanceChartElement = ref<HTMLDivElement | null>(null)
let importanceChart: ECharts | null = null

const METRIC_ORDER = [
  'roc_auc', 'val_auc', 'best_auc', 'accuracy', 'precision', 'recall', 'f1', 'macro_f1',
  'balanced_accuracy', 'best_threshold', 'r2', 'rmse', 'mae',
]
const METRIC_ICONS: Record<string, Component> = {
  roc_auc: Target,
  val_auc: Target,
  best_auc: Target,
  accuracy: ShieldCheck,
  precision: Crosshair,
  recall: RotateCcw,
  f1: Layers,
  macro_f1: Layers,
  balanced_accuracy: Scale,
  best_threshold: SlidersHorizontal,
  r2: TrendingUp,
  rmse: TrendingDown,
  mae: TrendingDown,
}

function metricIcon(name: string): Component {
  return METRIC_ICONS[name] ?? Award
}

const detailModelId = computed(() => {
  const value = route.query.model
  return typeof value === 'string' && value ? value : null
})
const detailModel = computed(() => models.value.find((model) => model.model_id === detailModelId.value) ?? null)
const detailNotFound = computed(() => Boolean(detailModelId.value) && !loading.value && !detailModel.value)
const prettyMetrics = computed(() => JSON.stringify(detailModel.value?.metrics ?? {}, null, 2))
const weightsKindLabel = computed(() =>
  weightsResult.value?.kind === 'coefficients' ? '线性模型系数（绝对值）' : '树模型特征重要性',
)
const metricEntries = computed(() => {
  const record = detailModel.value
  if (!record) return []
  const metrics = record.metrics ?? {}
  const primary = primaryMetricName(record)
  const names = Object.keys(metrics).filter((key) => typeof metrics[key] === 'number')
  names.sort((a, b) => {
    if (a === primary) return -1
    if (b === primary) return 1
    const ia = METRIC_ORDER.indexOf(a)
    const ib = METRIC_ORDER.indexOf(b)
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
  })
  return names.map((name) => ({ name, value: metrics[name] as number, primary: name === primary }))
})
const importanceRows = computed(() => {
  const weights = weightsResult.value?.weights ?? []
  const total = weights.reduce((sum, item) => sum + Math.abs(item.importance), 0) || 1
  return weights.slice(0, 50).map((item, index) => ({
    rank: index + 1,
    feature: item.feature,
    importance: item.importance,
    share: Math.abs(item.importance) / total,
  }))
})
const topWeights = computed(() => importanceRows.value.slice(0, 6))
const compareRows = computed(() =>
  models.value
    .filter((model) => compareModelIds.value.includes(model.model_id))
    .map((model) => ({
      model_id: model.model_id,
      name: model.name,
      source: sourceLabel(model.source),
      primaryMetric: metricText(model),
      rocAuc: formatFirstMetricWithDirection(model, ['roc_auc', 'val_auc']),
      accuracy: formatMetricWithDirection(model, 'accuracy', model.metrics.accuracy),
      r2: model.problem_type === 'regression' ? formatMetricWithDirection(model, 'r2', model.metrics.r2) : '-',
      rmse: model.problem_type === 'regression' ? formatMetricWithDirection(model, 'rmse', model.metrics.rmse) : '-',
      mae: model.problem_type === 'regression' ? formatMetricWithDirection(model, 'mae', model.metrics.mae) : '-',
      updatedAt: new Date(model.updated_at * 1000).toLocaleString(),
    })),
)

async function loadModels() {
  loading.value = true
  errorMessage.value = ''
  try {
    models.value = (await modelsApi.listModels()).models
    compareModelIds.value = models.value.filter((model) => !model.metadata_error).slice(0, 4).map((model) => model.model_id)
  } catch (error) {
    models.value = []
    compareModelIds.value = []
    errorMessage.value = error instanceof Error ? error.message : '模型列表加载失败'
  } finally {
    loading.value = false
  }
}

function openDetail(row: ModelRegistryRecord) {
  void router.push({ query: { ...route.query, model: row.model_id } })
}

function backToList() {
  const query = { ...route.query }
  delete query.model
  void router.push({ query })
}

watch(detailModel, (model) => {
  weightsResult.value = null
  if (!model) {
    if (importanceChart) {
      importanceChart.dispose()
      importanceChart = null
    }
    return
  }
  void loadWeights(model)
})

function renderImportanceChart() {
  const element = importanceChartElement.value
  if (!element) return
  // 返回再进入时详情区会被 v-if 重建，旧实例绑定的 DOM 已脱离文档，必须重新 init
  if (importanceChart && (importanceChart.isDisposed() || importanceChart.getDom() !== element)) {
    importanceChart.dispose()
    importanceChart = null
  }
  if (!importanceChart) importanceChart = init(element)
  const rows = [...topWeights.value].reverse()
  importanceChart.setOption({
    grid: { left: 100, right: 80, top: 8, bottom: 28 },
    xAxis: { type: 'value', name: '特征权重' },
    yAxis: { type: 'category', data: rows.map((row) => row.feature) },
    series: [
      {
        type: 'bar',
        data: rows.map((row) => Number(row.importance.toFixed(2))),
        barWidth: 14,
        itemStyle: { color: '#3b82f6', borderRadius: 4 },
        label: { show: true, position: 'right', color: '#334155' },
      },
    ],
  })
}

watch(weightsResult, async () => {
  await nextTick()
  renderImportanceChart()
})

onBeforeUnmount(() => {
  importanceChart?.dispose()
  importanceChart = null
})

async function copyJson() {
  try {
    await navigator.clipboard.writeText(prettyMetrics.value)
    ElMessage.success('指标 JSON 已复制')
  } catch {
    ElMessage.warning('当前环境不支持剪贴板复制')
  }
}

async function loadWeights(row: ModelRegistryRecord) {
  weightsLoading.value = true
  weightsModelId.value = row.model_id
  errorMessage.value = ''
  try {
    weightsResult.value = await modelsApi.getModelWeights(row.model_id)
  } catch (error) {
    weightsResult.value = null
    errorMessage.value = error instanceof Error ? error.message : '字段权重加载失败'
  } finally {
    weightsLoading.value = false
  }
}

async function confirmDelete(row: ModelRegistryRecord) {
  try {
    await ElMessageBox.confirm(
      `确认删除模型 ${row.model_id}？将删除产物文件与监控基线，不可恢复。`,
      '删除模型',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  deletingId.value = row.model_id
  errorMessage.value = ''
  try {
    await modelsApi.deleteModel(row.model_id)
    ElMessage.success(`模型 ${row.model_id} 已删除`)
    compareModelIds.value = compareModelIds.value.filter((id) => id !== row.model_id)
    if (weightsResult.value?.model_id === row.model_id) weightsResult.value = null
    const wasDetail = detailModelId.value === row.model_id
    await loadModels()
    if (wasDetail) backToList()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '模型删除失败'
  } finally {
    deletingId.value = ''
  }
}

function countBySource(source: string) {
  return models.value.filter((model) => model.source === source).length
}

function sourceLabel(source: string) {
  const labels: Record<string, string> = {
    automl: 'AutoML',
    manual_training: '手动训练',
    agent_workflow: 'Agent',
    legacy: 'Legacy',
  }
  return labels[source] ?? source
}

function sourceTagType(source: string) {
  if (source === 'automl') return 'success'
  if (source === 'manual_training') return 'warning'
  if (source === 'agent_workflow') return 'primary'
  return 'info'
}

function primaryMetricName(record: ModelRegistryRecord): string | null {
  if (record.metadata_error) return null
  const persistedMetric = record.optimization_metric ?? record.best_metric
  if (persistedMetric && typeof record.metrics[persistedMetric] === 'number') return persistedMetric
  const fallback = record.problem_type === 'regression' && !record.legacy
    ? ['r2', 'rmse', 'mae']
    : ['best_auc', 'roc_auc', 'val_auc', 'accuracy', 'best_value']
  return fallback
    .find((metric) => typeof record.metrics[metric] === 'number') ?? null
}

function metricText(record: ModelRegistryRecord) {
  if (record.metadata_error) return '元数据错误'
  const metric = primaryMetricName(record)
  return metric ? `${metricLabel(metric)} ${formatMetricWithDirection(record, metric, record.metrics[metric])}` : '-'
}

function metricLabel(metric: string) {
  const labels: Record<string, string> = {
    roc_auc: 'ROC-AUC',
    val_auc: 'ROC-AUC',
    best_auc: 'ROC-AUC',
    accuracy: 'Accuracy',
    precision: 'Precision',
    recall: 'Recall',
    f1: 'F1',
    macro_f1: 'Macro-F1',
    balanced_accuracy: 'Balanced Accuracy',
    best_threshold: 'Best Threshold',
    best_value: 'Best',
    rmse: 'RMSE',
    mae: 'MAE',
    r2: 'R²',
  }
  return labels[metric] ?? metric.toUpperCase()
}

function registryProblemTypeLabel(record: ModelRegistryRecord) {
  if (record.metadata_error) return '不可用'
  if (record.legacy) return 'Legacy 分类'
  const labels: Record<string, string> = {
    binary_classification: '二分类',
    multiclass_classification: '多分类',
    regression: '回归',
  }
  return record.problem_type ? (labels[record.problem_type] ?? record.problem_type) : '-'
}

function formatMetric(value: unknown) {
  return typeof value === 'number' ? value.toFixed(4) : '-'
}

function formatMetricWithDirection(record: ModelRegistryRecord, metric: string, value: unknown) {
  const formatted = formatMetric(value)
  if (formatted === '-') return formatted
  const direction = record.problem_spec?.capabilities.metric_directions[metric]
  if (direction === 'minimize') return `${formatted}（越小越好）`
  if (direction === 'maximize') return `${formatted}（越大越好）`
  return formatted
}

function formatFirstMetricWithDirection(record: ModelRegistryRecord, metrics: string[]) {
  const metric = metrics.find((candidate) => typeof record.metrics[candidate] === 'number')
  return metric ? formatMetricWithDirection(record, metric, record.metrics[metric]) : '-'
}

onMounted(loadModels)
</script>

<style scoped>
.model-registry-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.registry-heading,
.panel-title {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.registry-status-strip {
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
  margin-top: 5px;
  color: #0f172a;
  font-size: 19px;
}

.panel {
  min-width: 0;
  padding: 18px;
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

.model-name-cell {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}

.model-link {
  color: #2563eb;
  cursor: pointer;
}

.model-link:hover {
  text-decoration: underline;
}

.model-link:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 2px;
  border-radius: 4px;
}

.model-id-sub {
  flex-basis: 100%;
  color: #64748b;
  font-size: 11px;
}

.metadata-error-text {
  color: #b91c1c;
}

.detail-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 16px;
}

.detail-actions :deep(.el-button) {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.detail-info-strip {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
}

.info-card {
  display: flex;
  gap: 10px;
  min-width: 0;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 12px 14px;
  background: #f8fafc;
}

.info-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border-radius: 8px;
  background: #dbeafe;
  color: #2563eb;
  flex: none;
}

.info-card > div {
  min-width: 0;
  display: grid;
  gap: 4px;
}

.info-card span {
  color: #64748b;
  font-size: 12px;
}

.info-card strong {
  color: #0f172a;
  font-size: 13px;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.detail-block {
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 14px 16px;
  margin-top: 14px;
  background: #fbfdff;
}

.block-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.block-head h4 {
  margin: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  color: #111827;
  font-size: 14px;
}

.block-head h4 :deep(svg) {
  color: #2563eb;
}

.block-head > span {
  color: #64748b;
  font-size: 12px;
}

.metric-cards {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.metric-card {
  border: 1px solid #dbe7fb;
  border-radius: 10px;
  padding: 12px 14px;
  background: #ffffff;
  display: grid;
  gap: 8px;
}

.metric-card--primary {
  border-color: #93c5fd;
  background: linear-gradient(135deg, #eff6ff, #dbeafe);
}

.metric-card-head {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #334155;
  font-size: 13px;
  font-weight: 600;
  min-width: 0;
}

.metric-card-head :deep(svg) {
  color: #2563eb;
  flex: none;
}

.metric-card-head span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.metric-card strong {
  color: #2563eb;
  font-size: 24px;
  line-height: 1.15;
}

.importance-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr);
  gap: 16px;
}

.importance-chart-box h5 {
  margin: 0 0 8px;
  color: #334155;
  font-size: 13px;
}

.importance-chart {
  width: 100%;
  height: 280px;
}

.rank-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: #e2e8f0;
  color: #334155;
  font-size: 12px;
  font-weight: 700;
}

.rank-badge--1 { background: #fcd34d; color: #78350f; }
.rank-badge--2 { background: #cbd5e1; color: #334155; }
.rank-badge--3 { background: #fdba74; color: #7c2d12; }

.share-cell {
  display: grid;
  gap: 4px;
}

.share-cell span {
  color: #334155;
  font-size: 12px;
}

.json-actions {
  display: flex;
  gap: 4px;
}

.json-pre {
  overflow: auto;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
  margin: 0;
  padding: 12px;
}

.metadata-alert {
  margin-top: 14px;
}

.compare-selector {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  margin-bottom: 14px;
}

.compare-selector :deep(.el-checkbox) {
  margin-right: 0;
}

@media (max-width: 1100px) {
  .registry-status-strip,
  .detail-info-strip,
  .metric-cards,
  .importance-grid {
    grid-template-columns: 1fr;
  }

  .detail-info-strip,
  .metric-cards {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .registry-heading {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
