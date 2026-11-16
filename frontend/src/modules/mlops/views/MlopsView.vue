<template>
  <section class="workspace-theme mlops-workspace">
    <div class="section-heading mlops-heading">
      <div>
        <h2>MLOps 监控</h2>
        <p>展示调度器状态、PSI 漂移报告、模型健康、巡检日志，并支持触发重训任务。</p>
      </div>
      <el-button type="primary" :loading="loading" @click="loadAll">刷新监控</el-button>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" />

    <section class="status-strip">
      <div class="status-item">
        <span>调度器</span>
        <strong>{{ status?.scheduler_running ? '运行中' : '未运行' }}</strong>
      </div>
      <div class="status-item">
        <span>下次巡检</span>
        <strong>{{ nextRunLabel }}</strong>
      </div>
      <div class="status-item">
        <span>任务数</span>
        <strong>{{ status?.jobs?.length ?? 0 }}</strong>
      </div>
      <div class="status-item">
        <span>重训建议</span>
        <strong>{{ driftReport?.summary?.needs_retrain ? '需要' : '暂无' }}</strong>
      </div>
    </section>

    <div class="mlops-grid">
      <section class="panel">
        <div class="panel-title">
          <h3>PSI 漂移检测</h3>
          <span>{{ driftReport?.message ?? 'model_version 查询' }}</span>
        </div>
        <div class="inline-form">
          <el-input
            v-model="modelVersion"
            placeholder="模型 ID（留空巡检全部模型）"
            aria-label="巡检模型版本"
            clearable
          />
          <el-button plain @click="loadDrift">查询</el-button>
          <el-button type="primary" plain :loading="inspecting" @click="runInspect">立即巡检</el-button>
          <el-button type="warning" plain :loading="retraining" @click="triggerRetrain">触发重训</el-button>
        </div>
        <div class="inline-form bind-form">
          <el-select
            v-model="connectionId"
            aria-label="巡检源连接"
            filterable
            placeholder="巡检源连接（启用）"
            :loading="loadingConnections"
          >
            <el-option
              v-for="connection in connections"
              :key="connection.id"
              :label="`${connection.name}（${connection.db_type}）`"
              :value="connection.id"
            />
          </el-select>
          <el-select
            v-model="tableName"
            aria-label="巡检源表"
            filterable
            placeholder="巡检源表"
            :loading="loadingTables"
          >
            <el-option v-for="table in tableOptions" :key="table" :label="table" :value="table" />
          </el-select>
          <el-input v-model="bindTimeColumn" placeholder="时间列（可选）" aria-label="巡检源时间列" clearable />
          <el-button plain :disabled="!canBindSource" @click="bindSource">绑定为巡检源</el-button>
          <el-button plain :disabled="!modelVersion.trim()" @click="unbindSource">解除绑定</el-button>
        </div>
        <p class="bind-hint">当前模型巡检源：{{ currentBindingLabel }}</p>
        <div class="inline-form threshold-form">
          <span class="threshold-label">PSI 阈值（轻微 / 严重）</span>
          <el-input-number
            v-model="thresholdLow"
            aria-label="轻微漂移阈值"
            :min="0.01"
            :max="0.99"
            :step="0.05"
            :precision="2"
          />
          <el-input-number
            v-model="thresholdHigh"
            aria-label="严重漂移阈值"
            :min="0.02"
            :max="5"
            :step="0.05"
            :precision="2"
          />
          <el-button plain @click="saveThresholds">保存阈值</el-button>
        </div>
        <div class="summary-grid">
          <span>特征 {{ driftReport?.summary.total_features ?? 0 }}</span>
          <span>漂移 {{ driftReport?.summary.drifted_features ?? 0 }}</span>
          <span>严重 {{ driftReport?.summary.critical_features ?? 0 }}</span>
        </div>
        <el-table :data="psiRows" height="300">
          <el-table-column prop="feature" label="特征" />
          <el-table-column label="PSI" width="120">
            <template #default="{ row }">{{ row.psi.toFixed(4) }}</template>
          </el-table-column>
          <el-table-column label="状态" width="120">
            <template #default="{ row }">
              <el-tag :type="psiTagType(row.psi)" effect="light">{{ psiStatus(row.psi) }}</el-tag>
            </template>
          </el-table-column>
        </el-table>
      </section>

      <section class="panel">
        <div class="panel-title">
          <h3>模型健康</h3>
          <span>{{ healthRows.length }} 个模型</span>
        </div>
        <el-table :data="healthRows" height="386">
          <el-table-column label="模型" min-width="120">
            <template #default="{ row }">
              <span class="cell-ellipsis" :title="row.模型">{{ row.模型 }}</span>
            </template>
          </el-table-column>
          <el-table-column label="最佳指标" width="110">
            <template #default="{ row }">
              <span class="metric-value">{{ row.最佳指标 ?? '-' }}</span>
              <small v-if="row.指标名" class="metric-name">{{ row.指标名 }}</small>
            </template>
          </el-table-column>
          <el-table-column prop="健康度" label="健康度" width="90" />
          <el-table-column prop="调用次数" label="调用次数" width="80" />
        </el-table>
      </section>
    </div>

    <section class="panel">
      <div class="panel-title">
        <h3>巡检日志</h3>
        <div class="log-actions">
          <el-input-number v-model="logDays" :min="1" :max="30" controls-position="right" />
          <el-button plain @click="loadLogs">刷新日志</el-button>
        </div>
      </div>
      <el-table :data="logs" height="320">
        <el-table-column prop="时间" label="时间" width="190" />
        <el-table-column prop="事件" label="事件" width="140" />
        <el-table-column label="结果" width="120">
          <template #default="{ row }">
            <el-tag :type="logTagType(row.结果)" effect="light">{{ row.结果 }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="详情" label="详情" />
      </el-table>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus/es/components/message/index.mjs'

import {
  mlopsApi,
  type DriftReport,
  type InspectionLogRow,
  type MlopsSettings,
  type MlopsStatus,
  type ModelHealthRow,
  type OnlineSourceBinding,
} from '@/api/mlops'
import { useDatabaseTableSource } from '@/shared/composables/useDatabaseTableSource'

interface PsiRow {
  feature: string
  psi: number
}

const loading = ref(false)
const retraining = ref(false)
const inspecting = ref(false)
const errorMessage = ref('')
const modelVersion = ref('')
const logDays = ref(7)
const status = ref<MlopsStatus | null>(null)
const driftReport = ref<DriftReport | null>(null)
const healthRows = ref<ModelHealthRow[]>([])
const logs = ref<InspectionLogRow[]>([])
const onlineSources = ref<Record<string, OnlineSourceBinding>>({})
const bindTimeColumn = ref('')
const settings = ref<MlopsSettings | null>(null)
const thresholdLow = ref(0.1)
const thresholdHigh = ref(0.2)
const {
  connections,
  connectionId,
  tableName,
  tableOptions,
  loadingConnections,
  loadingTables,
  canImportTable,
  loadConnections,
} = useDatabaseTableSource()

const canBindSource = computed(() => Boolean(modelVersion.value.trim()) && canImportTable.value)
const currentBindingLabel = computed(() => {
  const model = modelVersion.value.trim()
  const binding = model ? onlineSources.value[model] : undefined
  if (!binding) return '未绑定（使用预测日志）'
  const timePart = binding.time_column ? `，时间列 ${binding.time_column}` : ''
  return `${binding.table_name}（连接 ${binding.connection_id}${timePart}）`
})

const nextRunLabel = computed(() => {
  const value = status.value?.next_run
  if (!value) return '-'
  const parts = value.match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/)
  return parts ? `${parts[1]} ${parts[2]}` : value
})

const psiRows = computed<PsiRow[]>(() => {
  const features = driftReport.value?.features ?? {}
  return Object.entries(features).map(([feature, info]) => ({
    feature,
    psi: Number(info.psi ?? 0),
  }))
})

async function loadAll() {
  loading.value = true
  errorMessage.value = ''
  try {
    await Promise.all([loadStatus(), loadDrift(), loadHealth(), loadLogs(), loadOnlineSources(), loadSettings()])
    const message = await loadConnections()
    if (message) errorMessage.value = message
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'MLOps 数据加载失败'
  } finally {
    loading.value = false
  }
}

async function loadSettings() {
  try {
    settings.value = await mlopsApi.getSettings()
    thresholdLow.value = settings.value.psi_threshold_low
    thresholdHigh.value = settings.value.psi_threshold_high
  } catch {
    settings.value = null
  }
}

async function saveThresholds() {
  errorMessage.value = ''
  try {
    settings.value = await mlopsApi.updateSettings({
      psi_threshold_low: thresholdLow.value,
      psi_threshold_high: thresholdHigh.value,
    })
    ElMessage.success('PSI 阈值已更新，下次巡检生效')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '阈值保存失败'
  }
}

async function loadOnlineSources() {
  try {
    onlineSources.value = (await mlopsApi.getOnlineSources()).sources
  } catch {
    onlineSources.value = {}
  }
}

async function bindSource() {
  const model = modelVersion.value.trim()
  if (!model || !canImportTable.value) return
  errorMessage.value = ''
  try {
    await mlopsApi.bindOnlineSource(model, {
      connection_id: connectionId.value,
      table_name: tableName.value,
      ...(bindTimeColumn.value.trim() ? { time_column: bindTimeColumn.value.trim() } : {}),
    })
    await loadOnlineSources()
    ElMessage.success(`已绑定 ${tableName.value} 为 ${model} 的巡检源`)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '绑定巡检源失败'
  }
}

async function unbindSource() {
  const model = modelVersion.value.trim()
  if (!model) return
  errorMessage.value = ''
  try {
    await mlopsApi.unbindOnlineSource(model)
    await loadOnlineSources()
    ElMessage.success(`已解除 ${model} 的巡检源绑定`)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '解除绑定失败'
  }
}

async function loadStatus() {
  status.value = await mlopsApi.getStatus()
}

async function loadDrift() {
  driftReport.value = await mlopsApi.getDriftReport(modelVersion.value || 'default')
}

async function loadHealth() {
  healthRows.value = (await mlopsApi.getModelHealth()).models
}

async function loadLogs() {
  logs.value = (await mlopsApi.getLogs(logDays.value)).logs
}

async function triggerRetrain() {
  retraining.value = true
  errorMessage.value = ''
  try {
    const response = await mlopsApi.triggerRetrain(modelVersion.value || 'default', true)
    ElMessage.success(response.message ?? '重训任务已触发')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '触发重训失败'
  } finally {
    retraining.value = false
  }
}

async function runInspect() {
  inspecting.value = true
  errorMessage.value = ''
  try {
    const requested = modelVersion.value.trim()
    const input = requested ? { model_id: requested, days: logDays.value } : { days: logDays.value }
    const response = await mlopsApi.runInspect(input)
    if (response.reports > 0) {
      ElMessage.success(`巡检 ${response.inspected} 个模型，产出 ${response.reports} 份报告`)
    }
    const skipped = response.results.filter((item) => item.status !== 'ok')
    if (skipped.length) {
      ElMessage.warning(skipped.map((item) => `${item.model_id}：${item.message}`).join('；'))
    }
    if (!response.reports && !skipped.length) {
      ElMessage.info('暂无可巡检模型')
    }
    await Promise.all([loadDrift(), loadHealth(), loadLogs()])
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '巡检失败'
  } finally {
    inspecting.value = false
  }
}

function psiStatus(value: number): string {
  const high = settings.value?.psi_threshold_high ?? 0.2
  const low = settings.value?.psi_threshold_low ?? 0.1
  if (value >= high) return '严重漂移'
  if (value >= low) return '轻微漂移'
  return '稳定'
}

function psiTagType(value: number) {
  const high = settings.value?.psi_threshold_high ?? 0.2
  const low = settings.value?.psi_threshold_low ?? 0.1
  if (value >= high) return 'danger'
  if (value >= low) return 'warning'
  return 'success'
}

function logTagType(value: string) {
  if (value === '重训') return 'danger'
  if (value === '告警') return 'warning'
  return 'success'
}

onMounted(loadAll)
</script>

<style scoped>
.mlops-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.mlops-heading,
.panel-title {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.status-strip {
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
  padding: 16px;
}

.status-item span {
  display: block;
  color: #64748b;
  font-size: 12px;
}

.status-item strong {
  display: block;
  margin-top: 5px;
  color: #0f172a;
  font-size: 20px;
}

.mlops-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
  gap: 18px;
}

.panel {
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

.panel-title span {
  color: #64748b;
  font-size: 12px;
}

.inline-form,
.log-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.bind-form {
  margin-top: 10px;
  flex-wrap: wrap;
}

.bind-form :deep(.el-select) {
  min-width: 170px;
}

.bind-hint {
  margin: 8px 0 0;
  color: #64748b;
  font-size: 12px;
}

.threshold-form {
  margin-top: 10px;
}

.threshold-label {
  color: #64748b;
  font-size: 12px;
}

.cell-ellipsis {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.metric-value {
  display: block;
}

.metric-name {
  display: block;
  color: #64748b;
  font-size: 11px;
  line-height: 1.4;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 14px 0;
}

.summary-grid span {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px;
  color: #334155;
  background: #f8fafc;
}

@media (max-width: 1100px) {
  .status-strip,
  .mlops-grid,
  .summary-grid {
    grid-template-columns: 1fr;
  }

  .inline-form,
  .log-actions {
    align-items: stretch;
    flex-direction: column;
  }
}
</style>
