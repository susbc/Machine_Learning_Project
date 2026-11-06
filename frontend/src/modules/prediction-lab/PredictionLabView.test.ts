import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import PredictionLabView from './views/PredictionLabView.vue'

const modelMocks = vi.hoisted(() => ({
  listModels: vi.fn(),
  predict: vi.fn(),
  batchPredict: vi.fn(),
}))

vi.mock('@/api/models', () => ({
  modelsApi: modelMocks,
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
  optimization_metric: 'rmse',
  best_metric: null,
  problem_type: 'regression',
  problem_spec: regressionSpec,
  legacy: false,
  metadata_error: null,
  updated_at: 2,
} as const

const classificationRecord = {
  model_id: 'manual:classifier',
  source: 'manual_training',
  name: 'classifier',
  artifact_path: 'saved/manual_training/classifier.joblib',
  metrics: { roc_auc: 0.91 },
  optimization_metric: 'roc_auc',
  best_metric: null,
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
  metrics: { accuracy: 0.8 },
  optimization_metric: null,
  best_metric: null,
  problem_type: null,
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
  metadata_error: { code: 'invalid_artifact_problem_type', field: 'problem_type', message: 'invalid type' },
  updated_at: 4,
} as const

const regressionPrediction = {
  model_id: 'manual:reg',
  source: 'manual_training',
  input_features: { x: 3 },
  prediction: 7.5,
  encoded_prediction: null,
  probability: null,
  diagnostics: { unknown_input_keys: ['extra'], missing_columns_imputed: ['city'] },
} as const

const regressionBatch = {
  model_id: 'manual:reg',
  source: 'manual_training',
  summary: { count: 2, positive_rate: null, average_probability: null },
  rows: [
    {
      index: 0,
      features: { x: 3 },
      prediction: 7.5,
      encoded_prediction: null,
      probability: null,
      diagnostics: { unknown_input_keys: [], missing_columns_imputed: ['city'] },
    },
    {
      index: 1,
      features: { x: 4 },
      prediction: 9.5,
      encoded_prediction: null,
      probability: null,
      diagnostics: { unknown_input_keys: ['extra'], missing_columns_imputed: [] },
    },
  ],
} as const

const classificationPrediction = {
  model_id: 'manual:classifier',
  source: 'manual_training',
  input_features: { age: 70 },
  prediction: '违约',
  encoded_prediction: 1,
  probability: 0.84,
  diagnostics: { unknown_input_keys: [], missing_columns_imputed: [] },
} as const

const classificationBatch = {
  model_id: 'manual:classifier',
  source: 'manual_training',
  summary: { count: 2, positive_rate: 0.5, average_probability: 0.62 },
  rows: [
    { index: 0, features: { age: 30 }, prediction: '正常', encoded_prediction: 0, probability: 0.4, diagnostics: { unknown_input_keys: [], missing_columns_imputed: [] } },
    { index: 1, features: { age: 70 }, prediction: '违约', encoded_prediction: 1, probability: 0.84, diagnostics: { unknown_input_keys: [], missing_columns_imputed: [] } },
  ],
} as const

function actionButton(wrapper: VueWrapper, label: string) {
  const button = wrapper.findAll('button').find((candidate) => candidate.text().trim() === label)
  if (!button) throw new Error(`Expected ${label} action`)
  return button
}

async function selectOption(wrapper: VueWrapper, optionLabel: string) {
  await wrapper.get('[aria-label="预测模型"]').trigger('click')
  await flushPromises()
  const option = Array.from(document.querySelectorAll<HTMLElement>('.el-select-dropdown__item'))
    .reverse()
    .find((candidate) => candidate.textContent?.trim() === optionLabel)
  if (!option) throw new Error(`Expected ${optionLabel} option`)
  option.click()
  await flushPromises()
}

async function optionLabels(wrapper: VueWrapper) {
  const control = wrapper.get('[aria-label="预测模型"]')
  await control.trigger('click')
  await flushPromises()
  const listbox = document.getElementById(control.attributes('aria-controls')!)
  return Array.from(listbox?.querySelectorAll<HTMLElement>('.el-select-dropdown__item') ?? [])
    .map((option) => option.textContent?.trim())
}

async function chooseCsv(wrapper: VueWrapper, text: string) {
  const input = wrapper.get('[data-testid="prediction-csv-input"]')
  const file = new File([text], 'rows.csv', { type: 'text/csv' })
  Object.defineProperty(file, 'text', { configurable: true, value: async () => text })
  Object.defineProperty(input.element, 'files', { configurable: true, value: [file] })
  await input.trigger('change')
  await flushPromises()
}

async function selectCsvFile(wrapper: VueWrapper, file: File) {
  const input = wrapper.get('[data-testid="prediction-csv-input"]')
  Object.defineProperty(input.element, 'files', { configurable: true, value: [file] })
  await input.trigger('change')
  await flushPromises()
}

describe('PredictionLabView', () => {
  beforeEach(() => {
    modelMocks.listModels.mockReset()
    modelMocks.predict.mockReset()
    modelMocks.batchPredict.mockReset()
    modelMocks.listModels.mockResolvedValue({
      models: [regressionRecord, classificationRecord, legacyRecord, invalidMetadataRecord],
    })
    modelMocks.predict.mockResolvedValue(regressionPrediction)
    modelMocks.batchPredict.mockResolvedValue(regressionBatch)
  })

  it('renders a regression prediction and diagnostics without category or probability fields', async () => {
    const wrapper = mount(PredictionLabView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()

    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()

    expect(modelMocks.predict).toHaveBeenCalledWith('manual:reg', expect.objectContaining({ features: expect.any(Object) }))
    expect(wrapper.text()).toContain('7.5')
    expect(wrapper.text()).toContain('未知输入字段：extra')
    expect(wrapper.text()).toContain('已补齐缺失字段：city')
    expect(wrapper.text()).not.toContain('预测概率')
    expect(wrapper.text()).not.toContain('类别编码')
    wrapper.unmount()
  })

  it('renders regression batch values and row diagnostics without classification summaries or columns', async () => {
    const wrapper = mount(PredictionLabView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await chooseCsv(wrapper, 'x,extra\n3,\n4,surplus')

    await actionButton(wrapper, '执行批量预测').trigger('click')
    await flushPromises()

    expect(modelMocks.batchPredict).toHaveBeenCalledWith('manual:reg', {
      rows: [{ x: 3, extra: null }, { x: 4, extra: 'surplus' }],
    })
    expect(wrapper.text()).toContain('批量结果')
    expect(wrapper.text()).toContain('7.5')
    expect(wrapper.text()).toContain('9.5')
    expect(wrapper.text()).toContain('缺失：city')
    expect(wrapper.text()).toContain('未知：extra')
    expect(wrapper.text()).not.toContain('正例占比')
    expect(wrapper.text()).not.toContain('平均概率')
    expect(wrapper.text()).not.toContain('类别编码')
    expect(wrapper.text()).not.toContain('预测概率')
    wrapper.unmount()
  })

  it('preserves classification fields for single and batch predictions', async () => {
    modelMocks.predict.mockResolvedValue(classificationPrediction)
    modelMocks.batchPredict.mockResolvedValue(classificationBatch)
    const wrapper = mount(PredictionLabView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await selectOption(wrapper, 'manual:classifier')

    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()
    await chooseCsv(wrapper, 'age\n30\n70')
    await actionButton(wrapper, '执行批量预测').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('违约')
    expect(wrapper.text()).toContain('类别编码')
    expect(wrapper.text()).toContain('预测概率')
    expect(wrapper.text()).toContain('0.8400')
    expect(wrapper.text()).toContain('正例占比')
    expect(wrapper.text()).toContain('0.5000')
    expect(wrapper.text()).toContain('平均概率')
    expect(wrapper.text()).toContain('0.6200')
    wrapper.unmount()
  })

  it('clears stale results on model changes and failed predictions, then recovers', async () => {
    const wrapper = mount(PredictionLabView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('7.5')

    await selectOption(wrapper, 'manual:classifier')
    expect(wrapper.text()).not.toContain('7.5')

    modelMocks.predict.mockRejectedValueOnce(new Error('预测服务暂不可用'))
    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('预测服务暂不可用')
    expect(wrapper.text()).toContain('暂无预测结果')

    modelMocks.predict.mockResolvedValueOnce(classificationPrediction)
    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()
    expect(wrapper.text()).not.toContain('预测服务暂不可用')
    expect(wrapper.text()).toContain('违约')
    wrapper.unmount()
  })

  it('ignores a late single prediction response after the selected model changes', async () => {
    let resolvePrediction!: (value: typeof regressionPrediction) => void
    modelMocks.predict.mockReturnValueOnce(new Promise((resolve) => {
      resolvePrediction = resolve
    }))
    const wrapper = mount(PredictionLabView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()

    await actionButton(wrapper, '执行预测').trigger('click')
    await selectOption(wrapper, 'manual:classifier')
    resolvePrediction(regressionPrediction)
    await flushPromises()

    expect(wrapper.text()).toContain('暂无预测结果')
    expect(wrapper.text()).not.toContain('7.5')
    wrapper.unmount()
  })

  it('ignores a late batch response after the selected model changes', async () => {
    let resolveBatch!: (value: typeof regressionBatch) => void
    modelMocks.batchPredict.mockReturnValueOnce(new Promise((resolve) => {
      resolveBatch = resolve
    }))
    const wrapper = mount(PredictionLabView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await chooseCsv(wrapper, 'x\n3\n4')

    await actionButton(wrapper, '执行批量预测').trigger('click')
    await selectOption(wrapper, 'manual:classifier')
    resolveBatch(regressionBatch)
    await flushPromises()

    expect(wrapper.text()).not.toContain('批量结果')
    expect(wrapper.text()).not.toContain('7.5')
    wrapper.unmount()
  })

  it('keeps valid legacy models classification-compatible while excluding malformed metadata', async () => {
    modelMocks.listModels.mockRejectedValueOnce(new Error('模型列表暂不可用')).mockResolvedValue({
      models: [regressionRecord, classificationRecord, legacyRecord, invalidMetadataRecord],
    })
    const wrapper = mount(PredictionLabView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    expect(wrapper.text()).toContain('模型列表暂不可用')

    await wrapper.get('[data-testid="prediction-refresh-models"]').trigger('click')
    await flushPromises()

    expect(wrapper.text()).not.toContain('模型列表暂不可用')
    expect(await optionLabels(wrapper)).toEqual(['manual:reg', 'manual:classifier', 'legacy:old'])

    modelMocks.predict.mockResolvedValue({ ...classificationPrediction, model_id: 'legacy:old', source: 'legacy' })
    await selectOption(wrapper, 'legacy:old')
    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()

    expect(modelMocks.predict).toHaveBeenLastCalledWith('legacy:old', expect.any(Object))
    expect(wrapper.text()).toContain('类别编码')
    expect(wrapper.text()).toContain('预测概率')
    wrapper.unmount()
  })

  it('discards an older CSV read when a newer file finishes first', async () => {
    let resolveFirst!: (text: string) => void
    let resolveSecond!: (text: string) => void
    const first = new File([], 'first.csv', { type: 'text/csv' })
    const second = new File([], 'second.csv', { type: 'text/csv' })
    Object.defineProperty(first, 'text', { value: () => new Promise<string>((resolve) => { resolveFirst = resolve }) })
    Object.defineProperty(second, 'text', { value: () => new Promise<string>((resolve) => { resolveSecond = resolve }) })
    const wrapper = mount(PredictionLabView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()

    await selectCsvFile(wrapper, first)
    await selectCsvFile(wrapper, second)
    resolveSecond('x\n9')
    await flushPromises()
    resolveFirst('x\n1')
    await flushPromises()
    await actionButton(wrapper, '执行批量预测').trigger('click')
    await flushPromises()

    expect(modelMocks.batchPredict).toHaveBeenCalledWith('manual:reg', { rows: [{ x: 9 }] })
    wrapper.unmount()
  })

  it('invalidates batch state and rejects a late response as soon as CSV input changes', async () => {
    let resolveRead!: (text: string) => void
    let resolveBatch!: (value: typeof regressionBatch) => void
    modelMocks.batchPredict.mockReturnValueOnce(new Promise((resolve) => { resolveBatch = resolve }))
    const replacement = new File([], 'replacement.csv', { type: 'text/csv' })
    Object.defineProperty(replacement, 'text', { value: () => new Promise<string>((resolve) => { resolveRead = resolve }) })
    const wrapper = mount(PredictionLabView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await chooseCsv(wrapper, 'x\n3\n4')
    await actionButton(wrapper, '执行批量预测').trigger('click')

    await selectCsvFile(wrapper, replacement)
    resolveBatch(regressionBatch)
    await flushPromises()

    expect(wrapper.text()).not.toContain('批量结果')
    expect(actionButton(wrapper, '执行批量预测').attributes('disabled')).toBeDefined()
    resolveRead('x\n8')
    await flushPromises()
    wrapper.unmount()
  })
})
