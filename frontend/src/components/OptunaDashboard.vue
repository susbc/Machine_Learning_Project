<!--
Optuna Dashboard - Vue 3 组件

实时显示 Optuna 优化过程，支持 SSE 数据流接收和 ECharts 动态渲染
-->

<template>
  <div class="optuna-dashboard">
    <header class="dashboard-header">
      <div>
        <p class="eyebrow">AutoML Optimization</p>
        <h2>Optuna 智能调参</h2>
      </div>
      <el-tag :type="isRunning ? 'warning' : progress === 100 ? 'success' : 'info'" effect="light">
        {{ statusLabel }}
      </el-tag>
    </header>

    <!-- 控制面板 -->
    <div class="control-panel">
      <div class="control-actions">
        <el-button
          type="primary"
          :loading="isRunning"
          @click="startOptimization"
          :disabled="isRunning"
        >
          {{ isRunning ? '优化进行中' : '开始调参' }}
        </el-button>

        <el-button
          type="danger"
          plain
          @click="stopOptimization"
          :disabled="!isRunning"
        >
          停止
        </el-button>
      </div>

      <el-select
        v-model="nTrials"
        placeholder="Trial 次数"
        :disabled="isRunning"
        class="trial-select"
      >
        <el-option label="20 次" :value="20" />
        <el-option label="30 次" :value="30" />
        <el-option label="50 次" :value="50" />
        <el-option label="100 次" :value="100" />
      </el-select>
    </div>
    
    <!-- 进度条 -->
    <section class="progress-section">
      <div class="metric-grid">
        <div class="metric">
          <span class="metric-label">最佳 AUC</span>
          <strong>{{ bestValue?.toFixed(4) || '-' }}</strong>
        </div>
        <div class="metric">
          <span class="metric-label">当前 Trial</span>
          <strong>{{ currentTrial }}/{{ nTrials }}</strong>
        </div>
        <div class="metric">
          <span class="metric-label">最近分数</span>
          <strong>{{ latestValue?.toFixed(4) || '-' }}</strong>
        </div>
      </div>

      <el-progress
        class="progress-bar"
        :percentage="progress"
        :status="progress === 100 ? 'success' : ''"
        :format="formatProgress"
        :stroke-width="12"
        striped
        striped-flow
      />
    </section>
    
    <!-- ECharts 图表 -->
    <section class="chart-section">
      <div ref="chartRef" class="optuna-chart"></div>
      <div v-if="trialHistory.length === 0" class="empty-state">
        <span>等待 Trial 数据</span>
      </div>
    </section>
    
    <!-- Trial 历史表格 -->
    <section class="trial-table" v-if="trialHistory.length > 0">
      <div class="section-title">
        <h3>Trial 历史记录</h3>
        <span>最近 {{ trialHistory.length }} 条</span>
      </div>
      <el-table :data="trialHistory" class="history-table" max-height="320">
        <el-table-column prop="number" label="Trial #" width="80" />
        <el-table-column prop="value" label="AUC 分数" width="120">
          <template #default="{ row }">
            <span :class="{ 'best-value': row.isBest }">
              {{ row.value?.toFixed(4) || '-' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="params" label="主要参数">
          <template #default="{ row }">
            <el-tag
              v-for="(value, key) in row.displayParams"
              :key="key"
              size="small"
              class="param-tag"
            >
              {{ key }}: {{ value }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="datetime" label="完成时间" width="180" />
      </el-table>
    </section>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, computed } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'

const props = defineProps({
  apiBaseUrl: {
    type: String,
    default: 'http://localhost:8000/api'
  }
})

// 响应式状态
const isRunning = ref(false)
const progress = ref(0)
const currentTrial = ref(0)
const bestValue = ref(null)
const latestValue = ref(null)
const trialHistory = ref([])
const nTrials = ref(30)
const studyId = ref('')
const eventSource = ref(null)

// ECharts 实例
const chartRef = ref(null)
let chartInstance = null

const normalizedApiBaseUrl = computed(() => props.apiBaseUrl.replace(/\/$/, ''))

const statusLabel = computed(() => {
  if (isRunning.value) return '运行中'
  if (progress.value === 100) return '已完成'
  return '待启动'
})

// 格式化进度显示
const formatProgress = (percentage) => {
  return `${percentage}% (${currentTrial.value}/${nTrials.value})`
}

// 启动优化
const startOptimization = () => {
  isRunning.value = true
  progress.value = 0
  currentTrial.value = 0
  bestValue.value = null
  latestValue.value = null
  trialHistory.value = []
  studyId.value = `study_${Date.now()}`
  
  // 连接 FastAPI 的 SSE 接口
  const url = `${normalizedApiBaseUrl.value}/automl/optimize/stream?study_id=${studyId.value}&n_trials=${nTrials.value}`
  
  eventSource.value = new EventSource(url)
  
  eventSource.value.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      
      if (data.status === 'DONE') {
        // 优化完成
        completeOptimization()
        return
      }
      
      if (data.status === 'ERROR') {
        // 发生错误
        handleError(data.message)
        return
      }
      
      if (data.status === 'HEARTBEAT') {
        // 心跳，忽略
        return
      }
      
      // 更新进度
      progress.value = data.progress || 0
      currentTrial.value = data.trial_number + 1
      
      // 更新最佳值
      if (data.best_value !== null && data.best_value !== undefined) {
        bestValue.value = data.best_value
      }

      if (data.value !== null && data.value !== undefined) {
        latestValue.value = data.value
      }
      
      // 添加到历史记录
      addTrialToHistory(data)
      
      // 更新图表
      updateChart()
      
    } catch (error) {
      console.error('解析 SSE 数据失败:', error)
    }
  }
  
  eventSource.value.onerror = (error) => {
    console.error('SSE 连接错误:', error)
    handleError('连接服务器失败')
  }
  
  ElMessage.success('优化任务已启动')
}

// 停止优化
const stopOptimization = () => {
  if (eventSource.value) {
    eventSource.value.close()
    eventSource.value = null
  }
  isRunning.value = false
  ElMessage.info('优化任务已停止')
}

// 完成优化
const completeOptimization = () => {
  isRunning.value = false
  progress.value = 100
  
  if (eventSource.value) {
    eventSource.value.close()
    eventSource.value = null
  }
  
  ElMessage.success(`优化完成！最佳 AUC: ${bestValue.value?.toFixed(4) || '-'}`)
}

// 处理错误
const handleError = (message) => {
  isRunning.value = false
  ElMessage.error(`优化失败：${message}`)
  
  if (eventSource.value) {
    eventSource.value.close()
    eventSource.value = null
  }
}

// 添加 Trial 到历史记录
const addTrialToHistory = (data) => {
  // 判断是否是新的最佳值
  const isBest = !bestValue.value || (data.value && data.value >= bestValue.value)
  
  // 提取主要参数用于显示
  const displayParams = {}
  if (data.params) {
    const importantParams = ['learning_rate', 'num_leaves', 'depth', 'n_estimators', 'model_type']
    for (const key of importantParams) {
      if (key in data.params) {
        displayParams[key] = data.params[key]
      }
    }
  }
  
  trialHistory.value.push({
    number: data.trial_number,
    value: data.value,
    isBest: isBest,
    params: data.params,
    displayParams: displayParams,
    datetime: data.datetime_complete || new Date().toISOString()
  })
  
  // 保持表格只显示最近的 20 条
  if (trialHistory.value.length > 20) {
    trialHistory.value.shift()
  }
}

// 更新 ECharts 图表
const updateChart = () => {
  if (!chartInstance) return
  
  // 分离散点数据和最佳线数据
  const scatterData = []
  const bestLineData = []
  let currentBest = -Infinity
  
  trialHistory.value.forEach((trial) => {
    if (trial.value !== null && trial.value !== undefined) {
      scatterData.push([trial.number, trial.value])
      
      if (trial.value > currentBest) {
        currentBest = trial.value
      }
      bestLineData.push([trial.number, currentBest])
    }
  })
  
  // 增量更新图表
  chartInstance.setOption({
    series: [
      { data: scatterData },
      { data: bestLineData }
    ]
  })
}

// 初始化 ECharts
const initChart = () => {
  if (!chartRef.value) return
  
  chartInstance = echarts.init(chartRef.value)
  
  const option = {
    backgroundColor: 'transparent',
    title: {
      text: 'Optuna 寻优历史 (Optimization History)',
      left: 12,
      top: 8,
      textStyle: {
        color: '#1f2937',
        fontSize: 15,
        fontWeight: 700
      }
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      },
      formatter: (params) => {
        if (Array.isArray(params) && params.length > 0) {
          const trialNum = params[0].data[0]
          const value = params[0].data[1]
          return `Trial ${trialNum}<br/>AUC: ${value.toFixed(4)}`
        }
        return ''
      }
    },
    legend: {
      data: ['当前 Trial 值', '历史最佳值'],
      top: 40,
      right: 12,
      textStyle: {
        color: '#475569'
      }
    },
    grid: {
      left: 48,
      right: 28,
      bottom: 42,
      top: 92,
      containLabel: true
    },
    xAxis: {
      type: 'value',
      name: 'Trial',
      min: 0,
      max: nTrials.value - 1,
      interval: 1,
      axisLine: { lineStyle: { color: '#cbd5e1' } },
      splitLine: { lineStyle: { color: '#eef2f7' } }
    },
    yAxis: {
      type: 'value',
      name: 'Objective Value (AUC)',
      min: 0.5,
      max: 1.0,
      axisLine: { lineStyle: { color: '#cbd5e1' } },
      splitLine: { lineStyle: { color: '#eef2f7' } }
    },
    series: [
      {
        name: '当前 Trial 值',
        type: 'scatter',
        data: [],
        symbolSize: 11,
        itemStyle: {
          color: '#2563eb',
          borderColor: '#ffffff',
          borderWidth: 2
        }
      },
      {
        name: '历史最佳值',
        type: 'line',
        step: 'start',
        data: [],
        itemStyle: {
          color: '#ef4444'
        },
        lineStyle: {
          width: 3,
          shadowBlur: 8,
          shadowColor: 'rgba(239, 68, 68, 0.18)'
        }
      }
    ]
  }
  
  chartInstance.setOption(option)
}

// 窗口大小变化时重新渲染图表
const handleResize = () => {
  if (chartInstance) {
    chartInstance.resize()
  }
}

onMounted(() => {
  initChart()
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  if (eventSource.value) {
    eventSource.value.close()
  }
  if (chartInstance) {
    chartInstance.dispose()
  }
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.optuna-dashboard {
  width: 100%;
  padding: 24px;
  color: #111827;
  background:
    linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 0.98)),
    #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}

.dashboard-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}

.eyebrow {
  margin: 0 0 6px;
  color: #64748b;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

.dashboard-header h2 {
  margin: 0;
  color: #0f172a;
  font-size: 22px;
  line-height: 1.25;
}

.control-panel {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px;
  margin-bottom: 18px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}

.control-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.trial-select {
  width: 132px;
}

.progress-section {
  margin-bottom: 18px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}

.metric {
  min-width: 0;
  padding: 14px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}

.metric-label {
  display: block;
  margin-bottom: 5px;
  color: #64748b;
  font-size: 12px;
}

.metric strong {
  display: block;
  overflow: hidden;
  color: #111827;
  font-size: 21px;
  line-height: 1.15;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.progress-bar {
  width: 100%;
}

.chart-section {
  position: relative;
  min-height: 420px;
  margin: 20px 0;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}

.optuna-chart {
  width: 100%;
  height: 400px;
}

.empty-state {
  position: absolute;
  inset: 92px 28px 42px 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #94a3b8;
  font-size: 14px;
  pointer-events: none;
  border: 1px dashed #dbe3ef;
  border-radius: 8px;
  background: rgba(248, 250, 252, 0.72);
}

.trial-table {
  margin-top: 22px;
}

.section-title {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.trial-table h3 {
  margin: 0;
  color: #1f2937;
  font-size: 16px;
}

.section-title span {
  color: #64748b;
  font-size: 12px;
}

.history-table {
  width: 100%;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}

.param-tag {
  margin: 2px 6px 2px 0;
}

.best-value {
  color: #16a34a;
  font-weight: bold;
}

@media (max-width: 720px) {
  .optuna-dashboard {
    padding: 16px;
  }

  .dashboard-header,
  .control-panel,
  .section-title {
    align-items: stretch;
    flex-direction: column;
  }

  .metric-grid {
    grid-template-columns: 1fr;
  }

  .trial-select {
    width: 100%;
  }

  .control-actions :deep(.el-button) {
    flex: 1;
  }
}
</style>
