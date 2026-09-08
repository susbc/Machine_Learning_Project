<template>
  <section class="page-section">
    <div class="section-heading">
      <p class="eyebrow">Migration Status</p>
      <h2>前端替代状态看板</h2>
      <p>Vue + TypeScript 已覆盖主线工作台；生产托管、模块等价和旧功能迁移按阶段推进。</p>
    </div>

    <div class="overview-grid">
      <div class="panel">
        <h3>替代底座</h3>
        <ul class="check-list">
          <li>Vue 3 + TypeScript + Vite</li>
          <li>Vue Router + Pinia</li>
          <li>Element Plus + ECharts</li>
          <li>FastAPI health API 接入</li>
          <li>生产托管：FastAPI + frontend/dist</li>
          <li>Streamlit fallback：保留</li>
        </ul>
      </div>

      <div class="panel">
        <h3>替代进度</h3>
        <ReadinessChart />
      </div>
    </div>

    <div class="module-table">
      <div class="module-row" v-for="module in modules" :key="module.name">
        <div>
          <strong>{{ module.name }}</strong>
          <small>{{ module.scope }}</small>
        </div>
        <span :class="['module-status', module.tone]">{{ module.status }}</span>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import ReadinessChart from '@/shared/charts/ReadinessChart.vue'

const modules = [
  { name: '大模型配置', scope: 'Provider 管理、连接测试、激活、脱敏', status: '灰度可用', tone: 'ready' },
  { name: '数据库连接', scope: '连接管理、测试、Schema、只读预览', status: '灰度可用', tone: 'ready' },
  { name: 'AutoML 训练', scope: '上传数据、自动推荐、SSE 进度、保存模型', status: '灰度可用', tone: 'ready' },
  { name: '手动训练', scope: '数据上传、字段配置、EDA、训练、保存', status: '灰度可用', tone: 'ready' },
  { name: '数据管理', scope: '独立数据上传、预览、统计、字段概览', status: '灰度可用', tone: 'ready' },
  { name: 'AI 智能分析', scope: '规则诊断、跨模块上下文、报告历史', status: '灰度可用', tone: 'ready' },
  { name: 'MLOps 监控', scope: '状态、PSI、模型健康、日志、重训触发', status: '灰度可用', tone: 'ready' },
  { name: 'Agent 工作流', scope: '取数、特征策略、训练、预测、模型列表', status: '灰度可用', tone: 'ready' },
  { name: '自动迭代训练', scope: '兼容入口，合并至 AutoML/Optuna 流程', status: '灰度可用', tone: 'ready' },
  { name: '模型管理 / 模型对比', scope: '统一 registry、来源、指标、详情、多模型对比', status: '灰度可用', tone: 'ready' },
  { name: '预测实验室 / 结果分析', scope: '选择模型、单条 JSON 预测、CSV 批量预测、结果摘要', status: '灰度可用', tone: 'ready' },
]
</script>
