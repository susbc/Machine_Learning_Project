import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import MlopsView from './views/MlopsView.vue'

const mlopsMocks = vi.hoisted(() => ({
  getStatus: vi.fn(),
  getDriftReport: vi.fn(),
  getModelHealth: vi.fn(),
  getLogs: vi.fn(),
  runInspect: vi.fn(),
  triggerRetrain: vi.fn(),
  getOnlineSources: vi.fn(),
  bindOnlineSource: vi.fn(),
  unbindOnlineSource: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
}))

vi.mock('@/api/mlops', () => ({
  mlopsApi: mlopsMocks,
}))

const dbMocks = vi.hoisted(() => ({
  listConnections: vi.fn(),
  getSchema: vi.fn(),
  getTableRows: vi.fn(),
}))

vi.mock('@/api/database', () => ({
  databaseApi: dbMocks,
}))

const sqliteConnection = {
  id: 'conn1',
  name: '本地 SQLite',
  db_type: 'sqlite',
  is_active: true,
  config: { db_type: 'sqlite', database: 'local.db' },
} as const

const driftReport = {
  features: { age: { psi: 0.42, severity: 'critical' } },
  summary: { total_features: 1, drifted_features: 1, critical_features: 1, needs_retrain: true },
} as const

const statusPayload = {
  scheduler_running: true,
  active_task_id: null,
  next_run: '2026-11-13 03:00:00',
  jobs: [{ id: 'daily_drift_check', name: '每日漂移巡检', next_run: '2026-11-13 03:00:00' }],
} as const

const healthPayload = {
  models: [
    {
      模型: 'manual:m1',
      最佳指标: 0.9123,
      指标名: 'roc_auc',
      健康度: '需重训',
      调用次数: 35,
      '预测延迟 (ms)': null,
    },
  ],
} as const

const logsPayload = {
  logs: [
    {
      时间: '2026-11-12 20:00:00',
      事件: '手动巡检',
      结果: '重训',
      详情: 'manual:m1：发现 1 个特征严重漂移',
    },
  ],
} as const

function actionButton(wrapper: VueWrapper, label: string) {
  const button = wrapper.findAll('button').find((candidate) => candidate.text().trim() === label)
  if (!button) throw new Error(`Expected ${label} action`)
  return button
}

async function selectByLabel(wrapper: VueWrapper, ariaLabel: string, optionLabel: string) {
  await wrapper.get(`[aria-label="${ariaLabel}"]`).trigger('click')
  await flushPromises()
  const option = Array.from(document.querySelectorAll<HTMLElement>('.el-select-dropdown__item'))
    .reverse()
    .find((candidate) => candidate.textContent?.trim() === optionLabel)
  if (!option) throw new Error(`Expected ${optionLabel} option`)
  option.click()
  await flushPromises()
}

describe('MlopsView', () => {
  beforeEach(() => {
    Object.values(mlopsMocks).forEach((mock) => mock.mockReset())
    dbMocks.listConnections.mockReset()
    dbMocks.getSchema.mockReset()
    dbMocks.getTableRows.mockReset()
    mlopsMocks.getStatus.mockResolvedValue(statusPayload)
    mlopsMocks.getDriftReport.mockResolvedValue(driftReport)
    mlopsMocks.getModelHealth.mockResolvedValue(healthPayload)
    mlopsMocks.getLogs.mockResolvedValue(logsPayload)
    mlopsMocks.getOnlineSources.mockResolvedValue({ sources: {} })
    mlopsMocks.getSettings.mockResolvedValue({ psi_threshold_low: 0.1, psi_threshold_high: 0.2 })
    mlopsMocks.updateSettings.mockResolvedValue({ psi_threshold_low: 0.35, psi_threshold_high: 0.6 })
    mlopsMocks.bindOnlineSource.mockResolvedValue({
      model_id: 'manual:m1',
      source: { connection_id: 'conn1', table_name: 'customers' },
    })
    mlopsMocks.unbindOnlineSource.mockResolvedValue(undefined)
    dbMocks.listConnections.mockResolvedValue({ connections: [sqliteConnection], active_connection: sqliteConnection })
    dbMocks.getSchema.mockResolvedValue({
      connection: sqliteConnection,
      tables: [{ name: 'customers', columns: [], sample: { columns: [], rows: [], row_count: 0 } }],
    })
  })

  it('renders real monitoring data across all panels', async () => {
    const wrapper = mount(MlopsView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    const text = wrapper.text()
    expect(text).toContain('运行中')
    expect(text).toContain('2026-11-13 03:00')
    expect(text).toContain('需要')
    expect(text).toContain('特征 1')
    expect(text).toContain('漂移 1')
    expect(text).toContain('严重 1')
    expect(text).toContain('age')
    expect(text).toContain('0.4200')
    expect(text).toContain('manual:m1')
    expect(text).toContain('0.9123')
    expect(text).toContain('roc_auc')
    expect(text).toContain('需重训')
    expect(text).toContain('35')
    expect(text).toContain('手动巡检')
    wrapper.unmount()
  })

  it('triggers inspection for all models and refreshes panels', async () => {
    mlopsMocks.runInspect.mockResolvedValue({
      inspected: 1,
      reports: 1,
      results: [{ model_id: 'manual:m1', status: 'ok', message: '巡检完成', summary: driftReport.summary }],
    })
    const wrapper = mount(MlopsView, { global: { plugins: [ElementPlus] } })
    await flushPromises()
    expect(mlopsMocks.getDriftReport).toHaveBeenCalledTimes(1)

    await actionButton(wrapper, '立即巡检').trigger('click')
    await flushPromises()

    expect(mlopsMocks.runInspect).toHaveBeenCalledWith({ days: 7 })
    expect(mlopsMocks.getDriftReport).toHaveBeenCalledTimes(2)
    expect(mlopsMocks.getModelHealth).toHaveBeenCalledTimes(2)
    expect(mlopsMocks.getLogs).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })

  it('inspects a specific model when the version input is filled', async () => {
    mlopsMocks.runInspect.mockResolvedValue({
      inspected: 1,
      reports: 0,
      results: [{ model_id: 'manual:m1', status: 'insufficient_online', message: '近 7 天线上预测样本不足（0/100），跳过巡检' }],
    })
    const wrapper = mount(MlopsView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    await wrapper.get('[aria-label="巡检模型版本"]').setValue('manual:m1')
    await actionButton(wrapper, '立即巡检').trigger('click')
    await flushPromises()

    expect(mlopsMocks.runInspect).toHaveBeenCalledWith({ model_id: 'manual:m1', days: 7 })
    wrapper.unmount()
  })

  it('binds a database table as the inspection source for the selected model', async () => {
    const wrapper = mount(MlopsView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(wrapper.text()).toContain('未绑定（使用预测日志）')
    await wrapper.get('[aria-label="巡检模型版本"]').setValue('manual:m1')
    await selectByLabel(wrapper, '巡检源连接', '本地 SQLite（sqlite）')
    await selectByLabel(wrapper, '巡检源表', 'customers')
    await actionButton(wrapper, '绑定为巡检源').trigger('click')
    await flushPromises()

    expect(mlopsMocks.bindOnlineSource).toHaveBeenCalledWith('manual:m1', {
      connection_id: 'conn1',
      table_name: 'customers',
    })
    wrapper.unmount()
  })

  it('classifies PSI rows with configured thresholds and saves updates', async () => {
    mlopsMocks.getSettings.mockResolvedValue({ psi_threshold_low: 0.3, psi_threshold_high: 0.6 })
    const wrapper = mount(MlopsView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    // psi 0.42 在 0.3/0.6 阈值下是轻微漂移而非严重漂移（PSI 行标签紧随数值）
    expect(wrapper.text()).toContain('age0.4200轻微漂移')

    await wrapper.get('[aria-label="轻微漂移阈值"]').setValue(0.35)
    await actionButton(wrapper, '保存阈值').trigger('click')
    await flushPromises()
    expect(mlopsMocks.updateSettings).toHaveBeenCalledWith({ psi_threshold_low: 0.35, psi_threshold_high: 0.6 })
    wrapper.unmount()
  })
})
