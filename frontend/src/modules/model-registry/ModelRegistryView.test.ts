import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ModelRegistryView from './views/ModelRegistryView.vue'

const registryMocks = vi.hoisted(() => ({
  listModels: vi.fn(),
}))

vi.mock('@/api/models', () => ({
  modelsApi: { listModels: registryMocks.listModels },
}))

const regressionSpec = {
  schema_version: 1,
  requested_type: 'auto',
  resolved_type: 'regression',
  source: 'auto_inferred',
  reason: 'numeric_continuous_target',
  explanation: '数值目标具有连续取值，识别为回归任务',
  target: { column: 'price', dtype: 'float64', valid_rows: 80, unique_count: 80, unique_ratio: 1, integer_like: false },
  capabilities: {
    algorithms: ['linear_regression'],
    metrics: ['rmse', 'mae', 'r2'],
    default_metric: 'rmse',
    metric_directions: { rmse: 'minimize', mae: 'minimize', r2: 'maximize' },
  },
} as const

const regressionRecord = {
  model_id: 'manual:reg',
  source: 'manual_training',
  name: 'reg',
  artifact_path: 'saved/manual_training/reg.joblib',
  metrics: { rmse: 0.1, mae: 0.08, r2: 0.99 },
  optimization_metric: 'mae',
  best_metric: null,
  problem_type: 'regression',
  problem_spec: regressionSpec,
  legacy: false,
  metadata_error: null,
  updated_at: 2,
} as const

const classificationRecord = {
  model_id: 'automl:classifier',
  source: 'automl',
  name: 'classifier',
  artifact_path: 'saved/automl/classifier.joblib',
  metrics: { val_auc: 0.7, accuracy: 0.8 },
  optimization_metric: 'val_auc',
  best_metric: null,
  problem_type: 'binary_classification',
  problem_spec: {
    ...regressionSpec,
    resolved_type: 'binary_classification',
    reason: 'binary_target',
    explanation: '目标列具有两个类别，识别为二分类任务',
    capabilities: {
      algorithms: ['logistic_regression'],
      metrics: ['val_auc', 'accuracy', 'f1'],
      default_metric: 'val_auc',
      metric_directions: { val_auc: 'maximize', accuracy: 'maximize', f1: 'maximize' },
    },
  },
  legacy: false,
  metadata_error: null,
  updated_at: 3,
} as const

const legacyRecord = {
  model_id: 'legacy:old',
  source: 'legacy',
  name: 'old',
  artifact_path: 'model_code/saved/old.joblib',
  metrics: { best_auc: 0.77, accuracy: 0.75 },
  optimization_metric: null,
  best_metric: null,
  problem_type: 'binary_classification',
  problem_spec: null,
  legacy: true,
  metadata_error: null,
  updated_at: 1,
} as const

const invalidMetadataRecord = {
  model_id: 'manual:broken',
  source: 'manual_training',
  name: 'broken',
  artifact_path: 'saved/manual_training/broken.joblib',
  metrics: {},
  optimization_metric: null,
  best_metric: null,
  problem_type: null,
  problem_spec: null,
  legacy: false,
  metadata_error: {
    code: 'invalid_artifact_problem_type',
    field: 'problem_type',
    message: 'schema-v3 模型缺少有效的问题类型',
  },
  updated_at: 4,
} as const

const records = [regressionRecord, classificationRecord, legacyRecord, invalidMetadataRecord]

function refreshButton(wrapper: VueWrapper) {
  return wrapper.get('[data-testid="registry-refresh-button"]')
}

function tableRows(wrapper: VueWrapper, tableIndex: number) {
  const body = wrapper.findAll('.el-table__body-wrapper tbody')[tableIndex]
  if (!body) throw new Error(`Expected table body ${tableIndex}`)
  return body.findAll('tr').map((row) => row.findAll('td .cell').map((cell) => cell.text().trim()))
}

describe('ModelRegistryView', () => {
  beforeEach(() => {
    registryMocks.listModels.mockReset()
    registryMocks.listModels.mockResolvedValue({ models: records })
  })

  it('uses task-aware primary metrics and shows all regression comparison metrics', async () => {
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    const mainRows = tableRows(wrapper, 0)
    const compareRows = tableRows(wrapper, 1)

    expect(mainRows[0]?.[2]).toBe('MAE 0.0800（越小越好）')
    expect(mainRows[1]?.[2]).toBe('ROC-AUC 0.7000（越大越好）')
    expect(compareRows[0]?.slice(2, 8)).toEqual([
      'MAE 0.0800（越小越好）',
      '-',
      '-',
      '0.9900（越大越好）',
      '0.1000（越小越好）',
      '0.0800（越小越好）',
    ])
    expect(compareRows[1]?.slice(2, 5)).toEqual([
      'ROC-AUC 0.7000（越大越好）',
      '0.7000（越大越好）',
      '0.8000（越大越好）',
    ])
    wrapper.unmount()
  })

  it('does not invent a primary metric when persisted metric identity is unavailable', async () => {
    registryMocks.listModels.mockResolvedValue({
      models: [{ ...regressionRecord, optimization_metric: null, best_metric: null }],
    })
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(tableRows(wrapper, 0)[0]?.[2]).toBe('-')
    wrapper.unmount()
  })

  it('marks legacy and malformed metadata records without treating either as a valid v3 task', async () => {
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    const mainRows = tableRows(wrapper, 0)
    expect(mainRows[2]).toEqual(expect.arrayContaining(['ROC-AUC 0.7700', 'Legacy 分类', '旧模型兼容模式']))
    expect(mainRows[3]).toEqual(expect.arrayContaining([
      '元数据错误',
      '不可用',
      'problem_type：schema-v3 模型缺少有效的问题类型',
    ]))
    wrapper.unmount()
  })

  it('clears stale records on refresh failure and recovers on a later refresh', async () => {
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()
    expect(wrapper.text()).toContain('reg')

    registryMocks.listModels.mockRejectedValueOnce(new Error('注册表暂不可用'))
    await refreshButton(wrapper).trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('注册表暂不可用')
    expect(wrapper.text()).not.toContain('manual:reg')

    registryMocks.listModels.mockResolvedValueOnce({ models: [regressionRecord] })
    await refreshButton(wrapper).trigger('click')
    await flushPromises()

    expect(wrapper.text()).not.toContain('注册表暂不可用')
    expect(wrapper.text()).toContain('manual:reg')
    wrapper.unmount()
  })
})
