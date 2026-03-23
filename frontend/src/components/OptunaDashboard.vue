<!--
Optuna Dashboard - Vue 3 组件

实时显示 Optuna 优化过程，支持 SSE 数据流接收和 ECharts 动态渲染
-->

<template>
  <div class="optuna-dashboard">
    <!-- 控制面板 -->
    <div class="control-panel">
      <el-button
        type="primary"
        :loading="isRunning"
        @click="startOptimization"
        :disabled="isRunning"
      >
        {{ isRunning ? '优化进行中...' : '开始智能调参' }}
      </el-button>
      
      <el-button
        type="danger"
        @click="stopOptimization"
        :disabled="!isRunning"
      >
        停止
      </el-button>
      
      <el-select
        v-model="nTrials"
        placeholder="Trial 次数"
        :disabled="isRunning"
        style="width: 120px; margin-left: 10px;"
      >
        <el-option label="20 次" :value="20" />
        <el-option label="30 次" :value="30" />
        <el-option label="50 次" :value="50" />
        <el-option label="100 次" :value="100" />
      </el-select>
    </div>
    
    <!-- 进度条 -->
    <div v-if="progress > 0" class="progress-section">
      <el-progress
        :percentage="progress"
        :status="progress === 100 ? 'success' : ''"
        :format="formatProgress"
        striped
        striped-flow
      />
      
      <div class="stats">
        <el-tag type="success" size="small">
          最佳 AUC: {{ bestValue?.toFixed(4) || '-' }}
        </el-tag>
        <el-tag type="info" size="small" style="margin-left: 10px;">
          当前 Trial: {{ currentTrial }}
        </el-tag>
      </div>
    </div>
    
    <!-- ECharts 图表 -->
    <div ref="chartRef" class="optuna-chart"></div>
    
    <!-- Trial 历史表格 -->
    <div class="trial-table" v-if="trialHistory.length > 0">
      <h3>Trial 历史记录</h3>
      <el-table :data="trialHistory" style="width: 100%" max-height="300">
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
              style="margin-right: 5px;"
            >
              {{ key }}: {{ value }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="datetime" label="完成时间" width="180" />
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, computed } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'

// 响应式状态
const isRunning = ref(false)
const progress = ref(0)
const currentTrial = ref(0)
const bestValue = ref(null)
const trialHistory = ref([])
const nTrials = ref(30)
const studyId = ref('')
const eventSource = ref(null)

// ECharts 实例
const chartRef = ref(null)
let chartInstance = null

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
  trialHistory.value = []
  studyId.value = `study_${Date.now()}`
  
  // 连接 FastAPI 的 SSE 接口
  const url = `http://localhost:8000/api/automl/optimize/stream?study_id=${studyId.value}&n_trials=${nTrials.value}`
  
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
    title: {
      text: 'Optuna 寻优历史 (Optimization History)',
      left: 'center'
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
      top: 30
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: 80,
      containLabel: true
    },
    xAxis: {
      type: 'value',
      name: 'Trial',
      min: 0,
      max: nTrials.value - 1,
      interval: 1
    },
    yAxis: {
      type: 'value',
      name: 'Objective Value (AUC)',
      min: 0.5,
      max: 1.0
    },
    series: [
      {
        name: '当前 Trial 值',
        type: 'scatter',
        data: [],
        symbolSize: 10,
        itemStyle: {
          color: '#409EFF'
        }
      },
      {
        name: '历史最佳值',
        type: 'line',
        step: 'start',
        data: [],
        itemStyle: {
          color: '#F56C6C'
        },
        lineStyle: {
          width: 3
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
  padding: 20px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.control-panel {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.progress-section {
  margin-bottom: 20px;
}

.stats {
  margin-top: 10px;
  display: flex;
  gap: 10px;
}

.optuna-chart {
  width: 100%;
  height: 400px;
  margin: 20px 0;
}

.trial-table {
  margin-top: 30px;
}

.trial-table h3 {
  margin-bottom: 15px;
  color: #303133;
}

.best-value {
  color: #67C23A;
  font-weight: bold;
}
</style>
