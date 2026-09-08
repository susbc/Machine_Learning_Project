<template>
  <section class="model-registry-workspace">
    <div class="section-heading registry-heading">
      <div>
        <p class="eyebrow">Phase 7</p>
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
        <span>{{ selectedModel?.model_id ?? '选择模型查看详情' }}</span>
      </div>
      <el-table :data="models" height="420" highlight-current-row @current-change="selectModel">
        <el-table-column label="模型" min-width="210">
          <template #default="{ row }">
            <div class="model-name-cell">
              <span>{{ row.name }}</span>
              <el-tag v-if="row.legacy" size="small" type="info" effect="plain">Legacy</el-tag>
              <el-tag v-if="row.metadata_error" size="small" type="danger" effect="plain">元数据错误</el-tag>
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
        <el-table-column label="元数据状态" min-width="260">
          <template #default="{ row }">
            <span v-if="row.metadata_error" class="metadata-error-text">
              {{ row.metadata_error.field }}：{{ row.metadata_error.message }}
            </span>
            <span v-else>{{ row.legacy ? '旧模型兼容模式' : '有效' }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="artifact_path" label="产物路径" min-width="260" />
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

    <section v-if="selectedModel" class="panel detail-panel">
      <div class="panel-title">
        <h3>模型详情</h3>
        <div class="detail-tags">
          <el-tag :type="sourceTagType(selectedModel.source)" effect="light">{{ sourceLabel(selectedModel.source) }}</el-tag>
          <el-tag v-if="selectedModel.legacy" type="info" effect="plain">Legacy</el-tag>
          <el-tag v-if="selectedModel.metadata_error" type="danger" effect="plain">元数据错误</el-tag>
        </div>
      </div>
      <div class="detail-grid">
        <div>
          <span>模型 ID</span>
          <strong>{{ selectedModel.model_id }}</strong>
        </div>
        <div>
          <span>产物</span>
          <strong>{{ selectedModel.artifact_path }}</strong>
        </div>
      </div>
      <el-alert
        v-if="selectedModel.metadata_error"
        class="metadata-alert"
        type="error"
        :title="selectedModel.metadata_error.message"
        :description="`${selectedModel.metadata_error.code} · ${selectedModel.metadata_error.field}`"
        show-icon
        :closable="false"
      />
      <pre>{{ prettyMetrics }}</pre>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

import { modelsApi, type ModelRegistryRecord } from '@/api/models'

const models = ref<ModelRegistryRecord[]>([])
const selectedModel = ref<ModelRegistryRecord | null>(null)
const compareModelIds = ref<string[]>([])
const loading = ref(false)
const errorMessage = ref('')

const prettyMetrics = computed(() => JSON.stringify(selectedModel.value?.metrics ?? {}, null, 2))
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
    selectedModel.value = models.value[0] ?? null
    compareModelIds.value = models.value.filter((model) => !model.metadata_error).slice(0, 4).map((model) => model.model_id)
  } catch (error) {
    models.value = []
    selectedModel.value = null
    compareModelIds.value = []
    errorMessage.value = error instanceof Error ? error.message : '模型列表加载失败'
  } finally {
    loading.value = false
  }
}

function selectModel(row: ModelRegistryRecord | undefined) {
  selectedModel.value = row ?? null
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
  if (!record.legacy) return null
  return ['best_auc', 'roc_auc', 'val_auc', 'accuracy', 'best_value']
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
.panel-title span,
.detail-grid span {
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

.model-name-cell,
.detail-tags {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}

.metadata-error-text {
  color: #b91c1c;
}

.metadata-alert {
  margin-top: 14px;
}

.detail-grid {
  display: grid;
  grid-template-columns: minmax(0, 0.4fr) minmax(0, 1fr);
  gap: 12px;
}

.detail-grid strong {
  display: block;
  overflow-wrap: anywhere;
  margin-top: 4px;
  color: #0f172a;
}

.detail-panel pre {
  overflow: auto;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
  margin: 14px 0 0;
  padding: 12px;
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
  .detail-grid {
    grid-template-columns: 1fr;
  }

  .registry-heading {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
