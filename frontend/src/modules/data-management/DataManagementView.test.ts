import { mount, flushPromises } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it, vi } from 'vitest'

import DataManagementView from './views/DataManagementView.vue'

const trainingMocks = vi.hoisted(() => ({
  uploadDataset: vi.fn(async () => ({
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

describe('DataManagementView', () => {
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
})
