import { mount, flushPromises, type VueWrapper } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import DataManagementView from './views/DataManagementView.vue'

const trainingMocks = vi.hoisted(() => ({
  uploadDataset: vi.fn(async (_input: { file: File; targetCol?: string }) => ({
    dataset_id: 'dataset_1',
    filename: 'train.csv',
    rows: 2,
    columns: 3,
    target_col: 'label',
    created_at: '2026-06-05T17:00:00',
    quality: {
      duplicate_rows: 1,
      missing_cells: 2,
      missing_rate: 0.1,
      infinite_values: 0,
      constant_columns: ['income'],
      high_missing_columns: [],
    },
    preview: [
      { age: 30, income: 100, label: 0 },
      { age: 70, income: 200, label: 1 },
    ],
  })),
  getFieldConfig: vi.fn(async () => ({
    dataset_id: 'dataset_1',
    fields: [
      { field_name: 'age', field_type: 'num', dtype: 'int64', missing_rate: 0, unique_count: 2 },
      { field_name: 'label', field_type: 'label', dtype: 'int64', missing_rate: 0, unique_count: 2 },
    ],
  })),
  runEda: vi.fn(async () => ({
    dataset_id: 'dataset_1',
    summary: {
      rows: 2,
      columns: 3,
      target_col: 'label',
      missing_rate: 0,
      numeric_features: 2,
      categorical_features: 0,
      target_unique: 2,
    },
    columns: [],
    correlations: {},
  })),
}))

vi.mock('@/api/training', () => ({
  trainingApi: {
    uploadDataset: trainingMocks.uploadDataset,
    getFieldConfig: trainingMocks.getFieldConfig,
    runEda: trainingMocks.runEda,
  },
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

async function chooseRadio(wrapper: VueWrapper, groupLabel: string, value: string) {
  const input = wrapper.get(`[aria-label="${groupLabel}"] input[value="${value}"]`)
  ;(input.element as HTMLInputElement).checked = true
  await input.trigger('change')
  await flushPromises()
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

function actionButton(wrapper: VueWrapper, label: string) {
  const button = wrapper.findAll('button').find((candidate) => candidate.text().trim() === label)
  if (!button) throw new Error(`Expected ${label} action`)
  return button
}

describe('DataManagementView', () => {
  beforeEach(() => {
    trainingMocks.uploadDataset.mockClear()
    trainingMocks.getFieldConfig.mockClear()
    trainingMocks.runEda.mockClear()
    dbMocks.listConnections.mockReset()
    dbMocks.getSchema.mockReset()
    dbMocks.getTableRows.mockReset()
    dbMocks.listConnections.mockResolvedValue({ connections: [sqliteConnection], active_connection: sqliteConnection })
    dbMocks.getSchema.mockResolvedValue({
      connection: sqliteConnection,
      tables: [{ name: 'customers', columns: [], sample: { columns: [], rows: [], row_count: 0 } }],
    })
    dbMocks.getTableRows.mockResolvedValue({
      columns: ['age', 'income', 'label'],
      rows: [
        { age: 30, income: 100, label: 0 },
        { age: 70, income: 200, label: 1 },
      ],
      row_count: 2,
    })
  })

  it('uploads a dataset and shows preview, fields and statistics', async () => {
    const wrapper = mount(DataManagementView, { global: { plugins: [ElementPlus] } })

    const vm = wrapper.vm as unknown as { datasetFile: File; targetCol: string; uploadDataset: () => Promise<void> }
    vm.datasetFile = new File(['age,income,label\n30,100,0'], 'train.csv')
    vm.targetCol = 'label'
    await vm.uploadDataset()
    await flushPromises()

    expect(trainingMocks.uploadDataset).toHaveBeenCalledWith({ file: vm.datasetFile, targetCol: 'label' })
    expect(wrapper.text()).toContain('数据管理')
    expect(wrapper.text()).toContain('train.csv')
    expect(wrapper.text()).toContain('字段概览')
    expect(wrapper.text()).toContain('age')
    expect(wrapper.text()).toContain('数据预览')
    expect(wrapper.text()).toContain('缺失率')
    expect(wrapper.text()).toContain('发现重复行 1')
    expect(wrapper.text()).toContain('常量列 1')
  })

  it('imports a database table through the active connection as a dataset', async () => {
    const wrapper = mount(DataManagementView, { global: { plugins: [ElementPlus] } })

    await chooseRadio(wrapper, '数据来源', 'database')
    expect(dbMocks.listConnections).toHaveBeenCalled()
    await flushPromises()
    expect(dbMocks.getSchema).toHaveBeenCalledWith('conn1', 1)

    await selectByLabel(wrapper, '导入数据表', 'customers')
    await actionButton(wrapper, '导入并分析').trigger('click')
    await flushPromises()

    expect(dbMocks.getTableRows).toHaveBeenCalledWith('conn1', 'customers', 1000)
    const call = trainingMocks.uploadDataset.mock.calls.at(-1)![0]
    expect(call.file.name).toBe('customers.csv')
    const csv = await new Promise<string>((resolve) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result))
      reader.readAsText(call.file)
    })
    expect(csv.split('\r\n')[0]).toBe('age,income,label')
    expect(csv).toContain('30,100,0')
    expect(wrapper.text()).toContain('customers（数据库表）')
    expect(wrapper.text()).toContain('字段概览')
    wrapper.unmount()
  })

  it('reports an empty table instead of uploading', async () => {
    dbMocks.getTableRows.mockResolvedValue({ columns: ['age'], rows: [], row_count: 0 })
    const wrapper = mount(DataManagementView, { global: { plugins: [ElementPlus] } })

    await chooseRadio(wrapper, '数据来源', 'database')
    await flushPromises()
    await selectByLabel(wrapper, '导入数据表', 'customers')
    await actionButton(wrapper, '导入并分析').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('数据表没有可导入的行')
    expect(trainingMocks.uploadDataset).not.toHaveBeenCalled()
    wrapper.unmount()
  })
})
