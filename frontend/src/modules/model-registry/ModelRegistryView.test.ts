import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ModelRegistryView from './views/ModelRegistryView.vue'

const registryMocks = vi.hoisted(() => ({
  listModels: vi.fn(),
  getModelWeights: vi.fn(),
  deleteModel: vi.fn(),
}))

const messageBoxMocks = vi.hoisted(() => ({
  confirm: vi.fn(),
}))

const routeHolder = vi.hoisted(() => ({
  route: null as { query: Record<string, string> } | null,
}))

const routerMock = vi.hoisted(() => ({
  push: vi.fn(),
}))

const chartMocks = vi.hoisted(() => {
  const instance = {
    setOption: vi.fn(),
    dispose: vi.fn(),
    resize: vi.fn(),
    isDisposed: vi.fn(() => false),
    getDom: vi.fn(() => null as unknown),
  }
  const init = vi.fn((element: unknown) => {
    instance.getDom.mockReturnValue(element)
    return instance
  })
  return { instance, init }
})

vi.mock('@/api/models', () => ({
  modelsApi: registryMocks,
}))

vi.mock('@/shared/charts/echarts', () => ({
  init: chartMocks.init,
}))

vi.mock('element-plus/es/components/message-box/index.mjs', () => ({
  ElMessageBox: messageBoxMocks,
}))

vi.mock('vue-router', async () => {
  const { reactive } = await import('vue')
  routeHolder.route = reactive({ query: {} }) as { query: Record<string, string> }
  return {
    useRoute: () => routeHolder.route,
    useRouter: () => routerMock,
  }
})

function setQuery(query: Record<string, string>) {
  if (!routeHolder.route) throw new Error('route mock not initialized')
  routeHolder.route.query = query
}

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
    registryMocks.getModelWeights.mockReset()
    registryMocks.deleteModel.mockReset()
    messageBoxMocks.confirm.mockReset()
    routerMock.push.mockReset()
    chartMocks.instance.setOption.mockReset()
    chartMocks.instance.dispose.mockReset()
    chartMocks.init.mockReset()
    setQuery({})
    registryMocks.listModels.mockResolvedValue({ models: records })
    registryMocks.getModelWeights.mockResolvedValue({
      model_id: 'manual:reg',
      kind: 'coefficients',
      weights: [{ feature: 'x', importance: 0.75 }],
    })
    registryMocks.deleteModel.mockResolvedValue({ status: 'deleted', model_id: 'manual:reg' })
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

  it('uses the real manual v3 record shape and falls back to r2 without metric identity', async () => {
    registryMocks.listModels.mockResolvedValue({
      models: [{ ...regressionRecord, optimization_metric: null, best_metric: null }],
    })
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(tableRows(wrapper, 0)[0]?.[2]).toBe('R² 0.9900（越大越好）')
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

  it('navigates to the model detail view when a model name is clicked', async () => {
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    const links = wrapper.findAll('.model-link')
    await links[1]!.trigger('click')
    await flushPromises()

    expect(routerMock.push).toHaveBeenCalledWith({ query: { model: 'automl:classifier' } })
    wrapper.unmount()
  })

  it('renders the detail view from the route query with weights and back navigation', async () => {
    setQuery({ model: 'manual:reg' })
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(wrapper.text()).toContain('模型详情 · manual:reg')
    expect(wrapper.get('.detail-panel').text()).toContain('manual:reg')
    expect(registryMocks.getModelWeights).toHaveBeenCalledWith('manual:reg')
    expect(wrapper.text()).toContain('线性模型系数（绝对值）')
    expect(wrapper.text()).toContain('0.7500')
    expect(wrapper.text()).toContain('100.00%')
    expect(wrapper.text()).toContain('评估指标')
    expect(wrapper.text()).toContain('0.9900')
    expect(wrapper.text()).toContain('主指标')
    expect(chartMocks.instance.setOption).toHaveBeenCalled()

    const back = wrapper.findAll('button').find((candidate) => candidate.text().trim() === '返回列表')
    if (!back) throw new Error('Expected 返回列表 action')
    await back.trigger('click')
    await flushPromises()
    expect(routerMock.push).toHaveBeenCalledWith({ query: {} })
    wrapper.unmount()
  })

  it('refreshes weights from the detail view', async () => {
    setQuery({ model: 'manual:reg' })
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()
    expect(registryMocks.getModelWeights).toHaveBeenCalledTimes(1)

    const refresh = wrapper.findAll('button').find((candidate) => candidate.text().trim() === '刷新权重')
    if (!refresh) throw new Error('Expected 刷新权重 action')
    await refresh.trigger('click')
    await flushPromises()
    expect(registryMocks.getModelWeights).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })

  it('deletes a model from the detail view after confirmation and returns to the list', async () => {
    messageBoxMocks.confirm.mockResolvedValue('confirm')
    setQuery({ model: 'manual:reg' })
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    const deleteButton = wrapper.findAll('button').find((candidate) => candidate.text().trim() === '删除模型')
    if (!deleteButton) throw new Error('Expected 删除模型 action')
    await deleteButton.trigger('click')
    await flushPromises()

    expect(messageBoxMocks.confirm).toHaveBeenCalled()
    expect(registryMocks.deleteModel).toHaveBeenCalledWith('manual:reg')
    expect(registryMocks.listModels).toHaveBeenCalledTimes(2)
    expect(routerMock.push).toHaveBeenCalledWith({ query: {} })
    wrapper.unmount()
  })

  it('keeps the model when deletion is cancelled', async () => {
    messageBoxMocks.confirm.mockRejectedValue(new Error('cancel'))
    setQuery({ model: 'manual:reg' })
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    const deleteButton = wrapper.findAll('button').find((candidate) => candidate.text().trim() === '删除模型')
    if (!deleteButton) throw new Error('Expected 删除模型 action')
    await deleteButton.trigger('click')
    await flushPromises()

    expect(registryMocks.deleteModel).not.toHaveBeenCalled()
    expect(registryMocks.listModels).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('rebuilds the importance chart when re-entering the detail view', async () => {
    setQuery({ model: 'manual:reg' })
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(chartMocks.init).toHaveBeenCalledTimes(1)
    const firstElement = chartMocks.init.mock.calls[0]![0]

    setQuery({})
    await flushPromises()
    expect(chartMocks.instance.dispose).toHaveBeenCalled()

    setQuery({ model: 'manual:reg' })
    await flushPromises()

    expect(chartMocks.init).toHaveBeenCalledTimes(2)
    expect(chartMocks.init.mock.calls[1]![0]).not.toBe(firstElement)
    expect(chartMocks.instance.setOption).toHaveBeenCalled()
    wrapper.unmount()
  })
})
