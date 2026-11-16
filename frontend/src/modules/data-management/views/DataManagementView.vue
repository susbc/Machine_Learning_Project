<template>
  <section class="workspace-theme data-workspace">
    <div class="section-heading data-heading">
      <div>
        <h2>数据管理</h2>
        <p>独立完成数据上传、样本预览、字段概览和基础统计；上传后的 dataset 可继续用于手动训练。</p>
      </div>
      <el-tag :type="dataset ? 'success' : 'info'" effect="light">{{ dataset ? '已加载数据' : '待上传' }}</el-tag>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" />

    <section class="panel">
      <div class="panel-title">
        <h3>数据上传</h3>
        <span>{{ sourceLabel }}</span>
      </div>
      <label class="source-row">
        <span>数据来源</span>
        <el-radio-group v-model="source" aria-label="数据来源">
          <el-radio-button value="file">CSV / Excel 文件</el-radio-button>
          <el-radio-button value="database">数据库表</el-radio-button>
        </el-radio-group>
      </label>
      <div v-if="source === 'file'" class="upload-form">
        <label class="field field--file">
          <span>数据文件（CSV / Excel，≤100MB）</span>
          <input class="file-input" type="file" accept=".csv,.xlsx,.xls" @change="handleFileChange" />
        </label>
        <label class="field">
          <span>目标列（可留空自动识别）</span>
          <el-input v-model="targetCol" placeholder="如 label / target" clearable />
        </label>
        <div class="field field--action">
          <el-button type="primary" :loading="uploading" :disabled="!datasetFile" @click="uploadDataset">上传并分析</el-button>
        </div>
      </div>
      <div v-else class="db-form">
        <label class="field">
          <span>数据库连接（默认使用激活连接）</span>
          <el-select
            v-model="connectionId"
            aria-label="导入数据库连接"
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
        <label class="field">
          <span>数据表</span>
          <el-select
            v-model="tableName"
            aria-label="导入数据表"
            filterable
            placeholder="选择数据表"
            :loading="loadingTables"
          >
            <el-option v-for="table in tableOptions" :key="table" :label="table" :value="table" />
          </el-select>
        </label>
        <label class="field field--limit">
          <span>导入行数上限</span>
          <el-input-number v-model="importLimit" aria-label="导入行数上限" :min="1" :max="10000" :step="100" controls-position="right" />
        </label>
        <label class="field field--target">
          <span>目标列（可留空自动识别）</span>
          <el-input v-model="targetCol" placeholder="如 label / target" clearable />
        </label>
        <div class="field field--action">
          <el-button type="primary" :loading="uploading" :disabled="!canImportTable" @click="uploadDataset">导入并分析</el-button>
        </div>
      </div>
    </section>

    <section v-if="dataset" class="status-strip">
      <div>
        <span>数据集</span>
        <strong>{{ dataset.filename }}</strong>
      </div>
      <div>
        <span>行数</span>
        <strong>{{ dataset.rows }}</strong>
      </div>
      <div>
        <span>列数</span>
        <strong>{{ dataset.columns }}</strong>
      </div>
      <div>
        <span>目标列</span>
        <strong>{{ dataset.target_col }}</strong>
      </div>
    </section>

    <section v-if="dataset?.quality" class="quality-strip" aria-label="数据质量摘要">
      <el-alert
        :type="qualityAlertType"
        :title="qualitySummary"
        show-icon
        :closable="false"
      />
    </section>

    <div class="data-grid">
      <section class="panel">
        <div class="panel-title">
          <h3>字段概览</h3>
          <span>{{ fields.length }} 个字段</span>
        </div>
        <el-table :data="fields" height="360">
          <el-table-column prop="field_name" label="字段" min-width="140" />
          <el-table-column prop="field_type" label="类型" width="100" />
          <el-table-column prop="dtype" label="dtype" width="120" />
          <el-table-column prop="unique_count" label="唯一值" width="100" />
          <el-table-column label="缺失率" width="110">
            <template #default="{ row }">{{ percentLabel(row.missing_rate) }}</template>
          </el-table-column>
        </el-table>
      </section>

      <section class="panel">
        <div class="panel-title">
          <h3>基础统计</h3>
          <span>{{ eda?.dataset_id ?? '等待上传' }}</span>
        </div>
        <div class="summary-grid">
          <div>
            <span>缺失率</span>
            <strong>{{ percentLabel(eda?.summary.missing_rate) }}</strong>
          </div>
          <div>
            <span>数值特征</span>
            <strong>{{ eda?.summary.numeric_features ?? '-' }}</strong>
          </div>
          <div>
            <span>类别特征</span>
            <strong>{{ eda?.summary.categorical_features ?? '-' }}</strong>
          </div>
          <div>
            <span>目标唯一值</span>
            <strong>{{ eda?.summary.target_unique ?? '-' }}</strong>
          </div>
        </div>
      </section>
    </div>

    <section class="panel">
      <div class="panel-title">
        <h3>数据预览</h3>
        <span>最多显示前 20 行</span>
      </div>
      <el-table :data="dataset?.preview ?? []" height="360">
        <el-table-column v-for="column in previewColumns" :key="column" :prop="column" :label="column" min-width="140" />
      </el-table>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import { trainingApi, type TrainingDataset, type TrainingEda, type TrainingField } from '@/api/training'
import { useDatabaseTableSource } from '@/shared/composables/useDatabaseTableSource'

const datasetFile = ref<File | null>(null)
const targetCol = ref('')
const dataset = ref<TrainingDataset | null>(null)
const fields = ref<TrainingField[]>([])
const eda = ref<TrainingEda | null>(null)
const uploading = ref(false)
const errorMessage = ref('')
const source = ref<'file' | 'database'>('file')
const {
  connections,
  connectionId,
  tableName,
  tableOptions,
  importLimit,
  loadingConnections,
  loadingTables,
  canImportTable,
  loadConnections,
  importTableAsFile,
} = useDatabaseTableSource()

const previewColumns = computed(() => Object.keys(dataset.value?.preview?.[0] ?? {}))
const sourceLabel = computed(() => {
  if (source.value === 'database') return tableName.value ? `${tableName.value}（数据库表）` : '未选择数据表'
  return datasetFile.value?.name ?? dataset.value?.filename ?? '未选择文件'
})
const qualityIssueCount = computed(() => {
  const quality = dataset.value?.quality
  if (!quality) return 0
  return quality.duplicate_rows + quality.infinite_values + quality.constant_columns.length + quality.high_missing_columns.length
})
const qualityAlertType = computed(() => (qualityIssueCount.value ? 'warning' : 'success'))
const qualitySummary = computed(() => {
  const quality = dataset.value?.quality
  if (!quality) return '数据质量信息不可用'
  if (!qualityIssueCount.value) return `基础质量检查通过；缺失单元格 ${quality.missing_cells} 个`
  return `发现重复行 ${quality.duplicate_rows}、无穷值 ${quality.infinite_values}、常量列 ${quality.constant_columns.length}、高缺失列 ${quality.high_missing_columns.length}`
})

function handleFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0] ?? null
  if (file && file.size > 100 * 1024 * 1024) {
    datasetFile.value = null
    errorMessage.value = '文件超过 100 MB 上传限制'
    return
  }
  datasetFile.value = file
  errorMessage.value = ''
}

async function uploadDataset() {
  uploading.value = true
  errorMessage.value = ''
  try {
    let file = datasetFile.value
    if (source.value === 'database') {
      const imported = await importTableAsFile()
      if (imported.error) {
        errorMessage.value = imported.error
        return
      }
      file = imported.file ?? null
    }
    if (!file) return
    dataset.value = await trainingApi.uploadDataset({ file, targetCol: targetCol.value })
    const [fieldConfig, edaResult] = await Promise.all([
      trainingApi.getFieldConfig(dataset.value.dataset_id),
      trainingApi.runEda(dataset.value.dataset_id),
    ])
    fields.value = fieldConfig.fields
    eda.value = edaResult
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '数据上传或分析失败'
  } finally {
    uploading.value = false
  }
}

watch(source, async (value) => {
  if (value !== 'database') return
  const message = await loadConnections()
  if (message) errorMessage.value = message
})

function percentLabel(value: unknown) {
  return typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '-'
}
</script>

<style scoped>
.data-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.data-heading,
.panel-title {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.panel,
.status-strip > div {
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
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

.panel-title span,
.status-strip span,
.summary-grid span {
  color: #64748b;
  font-size: 12px;
}

.upload-form,
.db-form {
  display: grid;
  gap: 14px 16px;
  align-items: end;
}

.upload-form {
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) auto;
}

.db-form {
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 150px;
}

.data-grid,
.status-strip {
  display: grid;
  gap: 12px;
}

.field {
  display: grid;
  gap: 6px;
  min-width: 0;
}

.field > span {
  color: #64748b;
  font-size: 12px;
}

.field :deep(.el-select),
.field :deep(.el-input-number) {
  width: 100%;
}

.field--target {
  grid-column: 1 / 3;
}

.field--action {
  display: flex;
  justify-content: flex-end;
}

.source-row {
  display: grid;
  gap: 6px;
  margin-bottom: 14px;
}

.source-row > span {
  color: #64748b;
  font-size: 12px;
}

.data-grid {
  grid-template-columns: minmax(0, 1fr) minmax(320px, 0.65fr);
}

.status-strip {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.status-strip > div,
.summary-grid > div {
  min-width: 0;
  padding: 16px;
}

.quality-strip {
  min-width: 0;
}

.status-strip strong,
.summary-grid strong {
  display: block;
  overflow-wrap: anywhere;
  margin-top: 6px;
  color: #0f172a;
  font-size: 20px;
}

.file-input {
  width: 100%;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  padding: 9px 10px;
  color: #334155;
  background: #f8fafc;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.summary-grid > div {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
}

@media (max-width: 1100px) {
  .upload-form,
  .db-form,
  .data-grid,
  .status-strip {
    grid-template-columns: 1fr;
  }

  .field--target {
    grid-column: auto;
  }

  .field--action {
    justify-content: stretch;
  }

  .field--action :deep(.el-button) {
    width: 100%;
  }

  .data-heading {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
