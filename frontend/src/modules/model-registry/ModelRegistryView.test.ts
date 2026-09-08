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
  metrics: { optimization_metric: 'mae', rmse: 0.1, mae: 0.08, r2: 0.99 },
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
  metrics: { roc_auc: 0.72, val_auc: 0.7, accuracy: 0.8 },
  problem_type: 'binary_classification',
  problem_spec: null,
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

describe('ModelRegistryView', () => {
  beforeEach(() => {
    registryMocks.listModels.mockReset()
    registryMocks.listModels.mockResolvedValue({ models: records })
  })

  it('uses task-aware primary metrics and shows all regression comparison metrics', async () => {
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(wrapper.text()).toContain('模型管理')
    expect(wrapper.text()).toContain('MAE 0.0800')
    expect(wrapper.text()).toContain('ROC-AUC 0.7200')
    expect(wrapper.text()).toContain('R²')
    expect(wrapper.text()).toContain('0.9900')
    expect(wrapper.text()).toContain('MAE')
    expect(wrapper.text()).toContain('0.0800')
    expect(wrapper.text()).toContain('RMSE')
    expect(wrapper.text()).toContain('0.1000')
    wrapper.unmount()
  })

  it('marks legacy and malformed metadata records without treating either as a valid v3 task', async () => {
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(wrapper.text()).toContain('old')
    expect(wrapper.text()).toContain('Legacy')
    expect(wrapper.text()).toContain('ROC-AUC 0.7700')
    expect(wrapper.text()).toContain('broken')
    expect(wrapper.text()).toContain('元数据错误')
    expect(wrapper.text()).toContain('problem_type')
    expect(wrapper.text()).toContain('schema-v3 模型缺少有效的问题类型')
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
