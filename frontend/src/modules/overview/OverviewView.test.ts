import { flushPromises, mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import OverviewView from './views/OverviewView.vue'

const overviewMocks = vi.hoisted(() => ({
  getSummary: vi.fn(),
}))

vi.mock('@/api/overview', () => ({
  overviewApi: overviewMocks,
}))

const chartMocks = vi.hoisted(() => ({
  setOption: vi.fn(),
  dispose: vi.fn(),
}))

vi.mock('@/shared/charts/echarts', () => ({
  init: vi.fn(() => chartMocks),
}))

const summary = {
  generated_at: '2026-11-11T10:00:00',
  models: {
    total: 5,
    recent_7d: 2,
    distribution: [
      { name: '分类模型', count: 3 },
      { name: '回归模型', count: 2 },
    ],
  },
  training: { running: 1, queued: 2, completed: 4 },
  connections: { total: 3, db_types: ['mysql', 'sqlite'] },
  llm: { total: 2, names: ['DeepSeek', 'Qwen'] },
  workflow: { total: 6, running: 1 },
  trend: [
    { label: '周二', submitted: 2, completed: 1 },
    { label: '周三', submitted: 0, completed: 0 },
    { label: '周四', submitted: 1, completed: 1 },
    { label: '周五', submitted: 3, completed: 2 },
    { label: '周六', submitted: 0, completed: 0 },
    { label: '周日', submitted: 1, completed: 0 },
    { label: '周一', submitted: 2, completed: 2 },
  ],
  recent_tasks: [
    {
      name: 'train.csv 训练',
      source: '手动训练',
      type: '分类',
      model: 'lightgbm',
      status: 'running',
      submitted_at: '2026-11-11T09:00:00',
      progress: 45,
    },
    {
      name: 'task_ab12',
      source: 'Agent 工作流',
      type: '回归',
      model: 'xgboost',
      status: 'completed',
      submitted_at: '2026-11-10T09:00:00',
      progress: 100,
    },
  ],
} as const

describe('OverviewView', () => {
  beforeEach(() => {
    overviewMocks.getSummary.mockReset()
    chartMocks.setOption.mockReset()
    overviewMocks.getSummary.mockResolvedValue(summary)
  })

  it('renders real platform metrics from the overview summary', async () => {
    const wrapper = mount(OverviewView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(overviewMocks.getSummary).toHaveBeenCalled()
    const text = wrapper.text()
    expect(text).toContain('项目总览')
    expect(text).toContain('模型总数')
    expect(text).toContain('↑ +2')
    expect(text).toContain('训练中模型')
    expect(text).toContain('排队 2')
    expect(text).toContain('已完成任务')
    expect(text).toContain('接入数据库')
    expect(text).toContain('mysql / sqlite')
    expect(text).toContain('大模型数量')
    expect(text).toContain('DeepSeek / Qwen')
    expect(text).toContain('Agent 工作流')
    expect(text).toContain('运行中 1')
    expect(text).toContain('模型类型分布')
    expect(text).toContain('分类模型')
    expect(text).toContain('60.0%')
    expect(text).toContain('训练任务趋势（近7天）')
    expect(text).toContain('近期运行任务')
    expect(text).toContain('train.csv 训练')
    expect(text).toContain('训练中')
    expect(text).toContain('45%')
    expect(chartMocks.setOption).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })

  it('refetches the summary when the refresh button is clicked', async () => {
    const wrapper = mount(OverviewView, { global: { plugins: [ElementPlus] } })
    await flushPromises()
    expect(overviewMocks.getSummary).toHaveBeenCalledTimes(1)

    await wrapper.get('[aria-label="刷新总览"]').trigger('click')
    await flushPromises()
    expect(overviewMocks.getSummary).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })

  it('shows an empty model distribution state without charts', async () => {
    overviewMocks.getSummary.mockResolvedValue({
      ...summary,
      models: { total: 0, recent_7d: 0, distribution: [] },
    })
    const wrapper = mount(OverviewView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(wrapper.text()).toContain('暂无模型')
    expect(chartMocks.setOption).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })
})
