<template>
  <div ref="chartElement" class="readiness-chart" aria-label="Phase 0 readiness chart" />
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { init, type ECharts } from './echarts'

const chartElement = ref<HTMLDivElement | null>(null)
let chart: ECharts | null = null

onMounted(() => {
  if (!chartElement.value) {
    return
  }

  chart = init(chartElement.value)
  chart.setOption({
    grid: { left: 28, right: 12, top: 18, bottom: 24 },
    xAxis: {
      type: 'category',
      data: ['工程', '路由', '状态', '图表', 'API'],
    },
    yAxis: {
      type: 'value',
      max: 100,
    },
    series: [
      {
        type: 'bar',
        data: [100, 100, 85, 80, 70],
        itemStyle: { color: '#2f6f73' },
        barWidth: 22,
      },
    ],
  })
})

onBeforeUnmount(() => {
  chart?.dispose()
  chart = null
})
</script>
