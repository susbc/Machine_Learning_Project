<template>
  <section class="workspace-theme overview-workspace">
    <div class="section-heading overview-heading">
      <div>
        <h2>项目总览</h2>
        <p>面向表格数据建模的智能机器学习平台，集数据、训练、评估、部署与监控于一体。</p>
      </div>
      <div class="overview-refresh">
        <span>最近更新：{{ summary?.generated_at ?? '-' }}</span>
        <el-button aria-label="刷新总览" :loading="loading" circle @click="loadSummary">
          <RefreshCw :size="15" :stroke-width="1.9" />
        </el-button>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" />

    <div class="stat-grid">
      <article v-for="card in statCards" :key="card.label" class="stat-card">
        <span class="stat-icon" :class="`stat-icon--${card.tone}`">
          <component :is="card.icon" :size="22" :stroke-width="1.9" />
        </span>
        <div class="stat-body">
          <span>{{ card.label }}</span>
          <div class="stat-value-row">
            <strong>{{ card.value }}</strong>
            <em v-if="card.delta" class="stat-delta">↑ +{{ card.delta }}</em>
            <span v-if="card.badge" class="stat-badge" :class="`stat-badge--${card.badgeTone}`">{{ card.badge }}</span>
            <small v-if="card.sub" class="stat-sub">{{ card.sub }}</small>
          </div>
        </div>
      </article>
    </div>

    <div class="chart-grid">
      <section class="panel">
        <h3>模型类型分布</h3>
        <div v-if="distribution.length" class="donut-row">
          <div ref="donutElement" class="donut-chart" aria-label="模型类型分布" />
          <ul class="donut-legend">
            <li v-for="(item, index) in distribution" :key="item.name">
              <i :style="{ background: DONUT_COLORS[index % DONUT_COLORS.length] }"></i>
              <span>{{ item.name }}</span>
              <strong>{{ item.count }}</strong>
              <em>({{ percentOf(item.count) }})</em>
            </li>
          </ul>
        </div>
        <el-empty v-else description="暂无模型" />
      </section>
      <section class="panel">
        <h3>训练任务趋势（近7天）</h3>
        <div ref="trendElement" class="trend-chart" aria-label="训练任务趋势" />
      </section>
    </div>

    <section class="panel">
      <h3>近期运行任务</h3>
      <el-table :data="summary?.recent_tasks ?? []" empty-text="暂无训练任务">
        <el-table-column prop="name" label="任务名称" min-width="220" show-overflow-tooltip />
        <el-table-column prop="source" label="来源" width="120" />
        <el-table-column prop="type" label="类型" width="90" />
        <el-table-column prop="model" label="模型" width="160" show-overflow-tooltip />
        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.status)" effect="light" size="small">{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="submitted_at" label="提交时间" width="180" />
        <el-table-column label="进度" min-width="180">
          <template #default="{ row }">
            <div class="task-progress">
              <el-progress :percentage="row.progress" :show-text="false" :stroke-width="8" />
              <span>{{ row.progress }}%</span>
            </div>
          </template>
        </el-table-column>
      </el-table>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { Box, Database, Layers, RefreshCw, Sparkles, Workflow, Zap } from 'lucide-vue-next'

import { overviewApi, type OverviewSummary } from '@/api/overview'
import { init, type ECharts } from '@/shared/charts/echarts'

const DONUT_COLORS = ['#3b82f6', '#2dd4bf', '#a78bfa', '#fbbf24']

interface StatCard {
  label: string
  value: number
  icon: typeof Box
  tone: string
  delta?: number | null
  badge?: string | null
  badgeTone?: string
  sub?: string | null
}

const summary = ref<OverviewSummary | null>(null)
const loading = ref(false)
const errorMessage = ref('')
const donutElement = ref<HTMLDivElement | null>(null)
const trendElement = ref<HTMLDivElement | null>(null)
let donutChart: ECharts | null = null
let trendChart: ECharts | null = null

const distribution = computed(() => summary.value?.models.distribution ?? [])
const distributionTotal = computed(() => distribution.value.reduce((total, item) => total + item.count, 0))

const statCards = computed<StatCard[]>(() => {
  const data = summary.value
  const dbTypes = data?.connections.db_types ?? []
  const providerNames = data?.llm.names ?? []
  return [
    {
      label: '模型总数',
      value: data?.models.total ?? 0,
      delta: data?.models.recent_7d || null,
      icon: Box,
      tone: 'blue',
    },
    {
      label: '训练中模型',
      value: data?.training.running ?? 0,
      badge: data?.training.queued ? `排队 ${data.training.queued}` : null,
      badgeTone: 'blue',
      icon: Zap,
      tone: 'teal',
    },
    {
      label: '已完成任务',
      value: data?.training.completed ?? 0,
      icon: Layers,
      tone: 'indigo',
    },
    {
      label: '接入数据库',
      value: data?.connections.total ?? 0,
      sub: dbTypes.join(' / ') || null,
      icon: Database,
      tone: 'sky',
    },
    {
      label: '大模型数量',
      value: data?.llm.total ?? 0,
      sub: providerNames.slice(0, 3).join(' / ') || null,
      icon: Sparkles,
      tone: 'purple',
    },
    {
      label: 'Agent 工作流',
      value: data?.workflow.total ?? 0,
      badge: data?.workflow.running ? `运行中 ${data.workflow.running}` : null,
      badgeTone: 'green',
      icon: Workflow,
      tone: 'blue',
    },
  ]
})

function percentOf(count: number) {
  return distributionTotal.value ? `${((count / distributionTotal.value) * 100).toFixed(1)}%` : '0%'
}

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    queued: '排队中',
    running: '训练中',
    cancelling: '取消中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }
  return labels[status] ?? status
}

function statusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' {
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'danger'
  if (status === 'queued') return 'warning'
  if (status === 'running' || status === 'cancelling') return 'primary'
  return 'info'
}

function renderCharts() {
  const data = summary.value
  if (!data) return
  if (!donutChart && donutElement.value) donutChart = init(donutElement.value)
  if (!trendChart && trendElement.value) trendChart = init(trendElement.value)
  if (donutChart) {
    donutChart.setOption({
      color: DONUT_COLORS,
      title: {
        text: String(distributionTotal.value),
        subtext: '模型总数',
        left: 'center',
        top: '40%',
        textStyle: { fontSize: 26, fontWeight: 700, color: '#071747' },
        subtextStyle: { fontSize: 12, color: '#64748b' },
      },
      tooltip: { trigger: 'item' },
      series: [
        {
          type: 'pie',
          radius: ['58%', '80%'],
          center: ['50%', '50%'],
          label: { show: false },
          labelLine: { show: false },
          data: distribution.value.map((item) => ({ name: item.name, value: item.count })),
        },
      ],
    })
  }
  if (trendChart) {
    trendChart.setOption({
      color: ['#3b82f6', '#2dd4bf'],
      grid: { left: 34, right: 12, top: 32, bottom: 26 },
      legend: { top: 0, data: ['提交任务数', '完成任务数'] },
      xAxis: { type: 'category', data: data.trend.map((point) => point.label) },
      yAxis: { type: 'value', minInterval: 1 },
      series: [
        { name: '提交任务数', type: 'bar', barWidth: 14, data: data.trend.map((point) => point.submitted) },
        { name: '完成任务数', type: 'bar', barWidth: 14, data: data.trend.map((point) => point.completed) },
      ],
    })
  }
}

async function loadSummary() {
  loading.value = true
  errorMessage.value = ''
  try {
    summary.value = await overviewApi.getSummary()
    await nextTick()
    renderCharts()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '总览数据加载失败'
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  void loadSummary()
})

onBeforeUnmount(() => {
  donutChart?.dispose()
  trendChart?.dispose()
  donutChart = null
  trendChart = null
})
</script>

<style scoped>
.overview-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.overview-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.overview-refresh {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #64748b;
  font-size: 12px;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 17px;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 16px;
  min-width: 0;
  border: 1px solid #dbe7fb;
  border-radius: 12px;
  padding: 20px 22px;
  background: linear-gradient(135deg, #fbfdff, #f3f8ff);
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
}

.stat-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 52px;
  height: 52px;
  border-radius: 14px;
  flex: none;
}

.stat-icon--blue { background: #dbeafe; color: #2563eb; }
.stat-icon--teal { background: #cffafe; color: #0891b2; }
.stat-icon--indigo { background: #e0e7ff; color: #4f46e5; }
.stat-icon--sky { background: #dbeafe; color: #0284c7; }
.stat-icon--purple { background: #ede9fe; color: #7c3aed; }

.stat-body {
  min-width: 0;
  display: grid;
  gap: 6px;
}

.stat-body > span {
  color: #475569;
  font-size: 14px;
  font-weight: 600;
}

.stat-value-row {
  display: flex;
  align-items: baseline;
  gap: 10px;
  min-width: 0;
}

.stat-value-row strong {
  color: #0f172a;
  font-size: 30px;
  line-height: 1.1;
}

.stat-delta {
  color: #16a34a;
  font-size: 13px;
  font-style: normal;
  font-weight: 700;
}

.stat-badge {
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 12px;
  font-weight: 600;
}

.stat-badge--blue { background: #dbeafe; color: #1d4ed8; }
.stat-badge--green { background: #dcfce7; color: #15803d; }

.stat-sub {
  color: #64748b;
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chart-grid {
  display: grid;
  grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
  gap: 17px;
}

.panel {
  min-width: 0;
  border: 1px solid #dbe3ef;
  border-radius: 12px;
  background: #ffffff;
  padding: 18px 20px;
  box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
}

.panel h3 {
  margin: 0 0 16px;
  color: #111827;
  font-size: 16px;
}

.donut-row {
  display: flex;
  align-items: center;
  gap: 22px;
}

.donut-chart {
  width: 210px;
  height: 210px;
  flex: none;
}

.donut-legend {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 14px;
  min-width: 0;
}

.donut-legend li {
  display: grid;
  grid-template-columns: 12px minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 10px;
  color: #334155;
  font-size: 14px;
}

.donut-legend i {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.donut-legend strong {
  color: #0f172a;
}

.donut-legend em {
  color: #64748b;
  font-style: normal;
}

.trend-chart {
  width: 100%;
  height: 250px;
}

.task-progress {
  display: flex;
  align-items: center;
  gap: 10px;
}

.task-progress :deep(.el-progress) {
  flex: 1;
}

.task-progress span {
  width: 42px;
  text-align: right;
  color: #334155;
  font-size: 13px;
}

@media (max-width: 1100px) {
  .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .chart-grid { grid-template-columns: 1fr; }
}

@media (max-width: 700px) {
  .stat-grid { grid-template-columns: 1fr; }
}
</style>
