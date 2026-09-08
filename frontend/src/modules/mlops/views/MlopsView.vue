<template>
  <section class="mlops-workspace">
    <div class="section-heading mlops-heading">
      <div>
        <p class="eyebrow">Phase 3</p>
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
        <strong>{{ status?.next_run ?? '-' }}</strong>
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
          <el-input v-model="modelVersion" placeholder="模型版本 / task_id" />
          <el-button plain @click="loadDrift">查询</el-button>
          <el-button type="warning" plain :loading="retraining" @click="triggerRetrain">触发重训</el-button>
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
          <el-table-column prop="模型" label="模型" />
          <el-table-column prop="AUC" label="AUC" width="100" />
          <el-table-column prop="健康度" label="健康度" width="120" />
          <el-table-column prop="调用次数" label="调用次数" width="120" />
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

import { mlopsApi, type DriftReport, type InspectionLogRow, type MlopsStatus, type ModelHealthRow } from '@/api/mlops'

interface PsiRow {
  feature: string
  psi: number
}

const loading = ref(false)
const retraining = ref(false)
const errorMessage = ref('')
const modelVersion = ref('default')
const logDays = ref(7)
const status = ref<MlopsStatus | null>(null)
const driftReport = ref<DriftReport | null>(null)
const healthRows = ref<ModelHealthRow[]>([])
const logs = ref<InspectionLogRow[]>([])

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
    await Promise.all([loadStatus(), loadDrift(), loadHealth(), loadLogs()])
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'MLOps 数据加载失败'
  } finally {
    loading.value = false
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

function psiStatus(value: number): string {
  if (value >= 0.2) return '严重漂移'
  if (value >= 0.1) return '轻微漂移'
  return '稳定'
}

function psiTagType(value: number) {
  if (value >= 0.2) return 'danger'
  if (value >= 0.1) return 'warning'
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
