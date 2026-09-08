import type { RouteRecordRaw } from 'vue-router'
import { createRouter, createWebHistory } from 'vue-router'

import OverviewView from '@/modules/overview/views/OverviewView.vue'

// 首页保持同步加载，其余业务模块按路由拆包，避免 ECharts 和所有工作台逻辑
// 在首次打开页面时一次性下载。
const AutoMLView = () => import('@/modules/automl/views/AutoMLView.vue')

export const moduleRoutes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'overview',
    component: OverviewView,
    meta: { label: '总览', navGroup: 'core' },
  },
  {
    path: '/automl',
    name: 'automl',
    component: AutoMLView,
    meta: { label: 'AutoML 训练', navGroup: 'core' },
  },
  {
    path: '/auto-iteration',
    name: 'auto-iteration',
    component: AutoMLView,
    meta: { label: '自动迭代', navGroup: 'core' },
  },
  {
    path: '/ai-analysis',
    name: 'ai-analysis',
    component: () => import('@/modules/ai-analysis/views/AiAnalysisView.vue'),
    meta: { label: 'AI 智能分析', navGroup: 'core' },
  },
  {
    path: '/mlops',
    name: 'mlops',
    component: () => import('@/modules/mlops/views/MlopsView.vue'),
    meta: { label: 'MLOps 监控', navGroup: 'core' },
  },
  {
    path: '/data-management',
    name: 'data-management',
    component: () => import('@/modules/data-management/views/DataManagementView.vue'),
    meta: { label: '数据管理', navGroup: 'training' },
  },
  {
    path: '/manual-training',
    name: 'manual-training',
    component: () => import('@/modules/manual-training/views/ManualTrainingView.vue'),
    meta: { label: '手动训练', navGroup: 'training' },
  },
  {
    path: '/llm-config',
    name: 'llm-config',
    component: () => import('@/modules/llm-config/views/LlmConfigView.vue'),
    meta: { label: '大模型配置', navGroup: 'settings' },
  },
  {
    path: '/database',
    name: 'database',
    component: () => import('@/modules/database/views/DatabaseView.vue'),
    meta: { label: '数据库连接', navGroup: 'settings' },
  },
  {
    path: '/agent-workflow',
    name: 'agent-workflow',
    component: () => import('@/modules/agent-workflow/views/AgentWorkflowView.vue'),
    meta: { label: 'Agent 工作流', navGroup: 'workflow' },
  },
  {
    path: '/model-registry',
    name: 'model-registry',
    component: () => import('@/modules/model-registry/views/ModelRegistryView.vue'),
    meta: { label: '模型管理', navGroup: 'models' },
  },
  {
    path: '/prediction-lab',
    name: 'prediction-lab',
    component: () => import('@/modules/prediction-lab/views/PredictionLabView.vue'),
    meta: { label: '预测实验室', navGroup: 'models' },
  },
]

export function createAppRouter() {
  return createRouter({
    history: createWebHistory(),
    routes: moduleRoutes,
  })
}
