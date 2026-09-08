import { flushPromises, mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { AutoMLJob, AutoMLResult, AutoMLTrialEvent } from '@/api/automl'
import type { ProblemCapabilitiesResponse, ProblemSpec } from '@/api/problem'
import AutoMLView from './views/AutoMLView.vue'

const automlMocks = vi.hoisted(() => ({
  createJob: vi.fn(), getJob: vi.fn(), getResult: vi.fn(), cancelJob: vi.fn(), saveResult: vi.fn(), streamJob: vi.fn(),
}))
const trainingApiMocks = vi.hoisted(() => ({ getCapabilities: vi.fn() }))
const chartMocks = vi.hoisted(() => ({ setOption: vi.fn(), resize: vi.fn(), dispose: vi.fn() }))

vi.mock('@/api/automl', () => ({ automlApi: automlMocks }))
vi.mock('@/api/training', () => ({ trainingApi: trainingApiMocks }))
vi.mock('@/shared/charts/echarts', () => ({ init: vi.fn(() => chartMocks) }))

const regressionSpec = {
  schema_version: 1, requested_type: 'auto', resolved_type: 'regression', source: 'auto_inferred',
  reason: 'numeric_continuous_target', explanation: '连续数值目标，识别为回归任务',
  target: { column: 'price', dtype: 'float64', valid_rows: 80, unique_count: 80, unique_ratio: 1, integer_like: false },
  capabilities: {
    algorithms: ['linear_regression'], metrics: ['rmse', 'mae', 'r2'], default_metric: 'rmse',
    metric_directions: { rmse: 'minimize', mae: 'minimize', r2: 'maximize' },
  },
} satisfies ProblemSpec
const classificationSpec = {
  schema_version: 1, requested_type: 'auto', resolved_type: 'binary_classification', source: 'auto_inferred',
  reason: 'string_target', explanation: '字符串目标自动识别为二分类任务',
  target: { column: 'label', dtype: 'object', valid_rows: 80, unique_count: 2, unique_ratio: 0.025, integer_like: false },
  capabilities: {
    algorithms: ['logistic_regression'], metrics: ['roc_auc', 'accuracy', 'f1'], default_metric: 'roc_auc',
    metric_directions: { roc_auc: 'maximize', accuracy: 'maximize', f1: 'maximize' },
  },
} satisfies ProblemSpec
const capabilities: ProblemCapabilitiesResponse = {
  problem_types: {
    binary_classification: classificationSpec.capabilities,
    multiclass_classification: {
      algorithms: ['logistic_regression', 'random_forest'], metrics: ['roc_auc', 'accuracy'], default_metric: 'roc_auc',
      metric_directions: { roc_auc: 'maximize', accuracy: 'maximize' },
    },
    regression: regressionSpec.capabilities,
  },
}
const queuedRegressionJob: AutoMLJob = {
  job_id: 'job_reg', status: 'queued', progress: 0, current_trial: 0, n_trials: 3, cv_folds: 3,
  optimization_metric: 'rmse', target_col: 'price', decision_summary: {}, problem_spec: regressionSpec,
}
const queuedClassificationJob: AutoMLJob = {
  job_id: 'job_cls', status: 'queued', progress: 0, current_trial: 0, n_trials: 3, cv_folds: 3,
  optimization_metric: 'roc_auc', target_col: 'label', decision_summary: {}, problem_spec: classificationSpec,
}
const regressionResult: AutoMLResult = {
  job_id: 'job_reg', status: 'completed', best_model: 'linear_regression', best_params: { model_type: 'linear_regression' },
  best_value: 0.1, metrics: { rmse: 0.1, mae: 0.08, r2: 0.99 },
  trial_history: [{ event: 'trial', trial_number: 0, value: 0.1, best_value: 0.1, model_type: 'linear_regression', params: {} }],
  decision_summary: {}, feature_columns: ['x'], problem_spec: regressionSpec,
}
const classificationResult: AutoMLResult = {
  job_id: 'job_cls', status: 'completed', best_model: 'logistic_regression', best_params: { model_type: 'logistic_regression' },
  best_value: 0.91, metrics: { roc_auc: 0.91, accuracy: 0.88, f1: 0.86 }, trial_history: [],
  decision_summary: {}, feature_columns: ['x'], problem_spec: classificationSpec,
}

function mountView() {
  return mount(AutoMLView, { global: { plugins: [ElementPlus] } })
}
function actionButton(wrapper: ReturnType<typeof mount>, label: string) {
  const button = wrapper.findAll('button').find((candidate) => candidate.text().trim() === label)
  if (!button) throw new Error(`Expected ${label} action`)
  return button
}
async function chooseFile(wrapper: ReturnType<typeof mount>, filename = 'train.csv') {
  const fileInput = wrapper.get('[data-testid="automl-file-input"]').element as HTMLInputElement
  Object.defineProperty(fileInput, 'files', { configurable: true, value: [new File(['x,target\n1,2.5'], filename, { type: 'text/csv' })] })
  await wrapper.get('[data-testid="automl-file-input"]').trigger('change')
}
async function selectOption(wrapper: ReturnType<typeof mount>, controlLabel: string, optionLabel: string) {
  const control = wrapper.get(`[aria-label="${controlLabel}"]`)
  await control.trigger('click')
  const option = Array.from(document.querySelectorAll<HTMLElement>('.el-select-dropdown__item')).reverse()
    .find((candidate) => candidate.textContent?.trim() === optionLabel)
  if (!option) throw new Error(`Expected ${optionLabel} option`)
  option.click()
  await flushPromises()
}
async function start(wrapper: ReturnType<typeof mount>) {
  await actionButton(wrapper, '开始全自动训练').trigger('click')
  await flushPromises()
}

describe('AutoMLView', () => {
  beforeEach(() => {
    localStorage.clear()
    Object.values(automlMocks).forEach((mock) => mock.mockReset())
    trainingApiMocks.getCapabilities.mockReset()
    chartMocks.setOption.mockReset()
    trainingApiMocks.getCapabilities.mockResolvedValue(capabilities)
    automlMocks.streamJob.mockImplementation((_jobId, _onEvent, signal) => new Promise<void>((resolve) => {
      signal?.addEventListener('abort', () => resolve(), { once: true })
    }))
  })

  it('submits automatic problem and metric defaults from the rendered upload flow', async () => {
    automlMocks.createJob.mockResolvedValue(queuedRegressionJob)
    const wrapper = mountView()
    await flushPromises()
    await chooseFile(wrapper, 'reg.csv')
    await start(wrapper)

    expect(automlMocks.createJob).toHaveBeenCalledWith(expect.objectContaining({
      problemType: 'auto', optimizationMetric: 'auto', candidateModels: undefined,
    }))
    wrapper.unmount()
  })

  it('renders every resolved regression target fact and regression-only metrics after the snapshot', async () => {
    automlMocks.createJob.mockResolvedValue(queuedRegressionJob)
    automlMocks.streamJob.mockImplementation(async (_jobId, onEvent) => onEvent({ event: 'done', status: 'completed' }))
    automlMocks.getJob.mockResolvedValue({ ...queuedRegressionJob, status: 'completed', progress: 100 })
    automlMocks.getResult.mockResolvedValue(regressionResult)
    const wrapper = mountView()
    await flushPromises()
    await chooseFile(wrapper, 'reg.csv')
    await start(wrapper)
    await flushPromises()

    expect(wrapper.text()).toContain('连续数值目标')
    expect(wrapper.text()).toContain('目标列 price（float64）')
    expect(wrapper.text()).toContain('有效行 80')
    expect(wrapper.text()).toContain('唯一值 80')
    expect(wrapper.text()).toContain('唯一值比例 100.00%')
    expect(wrapper.text()).toContain('整数样式 否')
    expect(wrapper.text()).toContain('RMSE：0.10000')
    expect(wrapper.text()).toContain('MAE：0.08000')
    expect(wrapper.text()).toContain('R²：0.99000')
    expect(wrapper.text()).not.toContain('ROC-AUC：')
    expect(chartMocks.setOption).toHaveBeenLastCalledWith(expect.objectContaining({
      yAxis: expect.objectContaining({ name: 'RMSE（越小越好）' }),
    }))
    wrapper.unmount()
  })

  it('uses compatible explicit selections in the serialized payload and retains them across a compatible type switch', async () => {
    automlMocks.createJob.mockResolvedValue(queuedClassificationJob)
    const wrapper = mountView()
    await flushPromises()
    await selectOption(wrapper, '问题类型', '二分类')
    await selectOption(wrapper, '问题类型', '多分类')
    await chooseFile(wrapper, 'class.csv')
    await start(wrapper)

    expect(automlMocks.createJob).toHaveBeenCalledWith(expect.objectContaining({
      problemType: 'multiclass_classification', candidateModels: ['logistic_regression'], optimizationMetric: 'roc_auc',
    }))
    expect(wrapper.text()).not.toContain('候选模型已')
    expect(wrapper.text()).not.toContain('优化指标已从')
    wrapper.unmount()
  })

  it('names only the incompatible candidate and metric that a type switch resets', async () => {
    const wrapper = mountView()
    await flushPromises()
    await selectOption(wrapper, '问题类型', '二分类')
    await selectOption(wrapper, '问题类型', '回归')

    expect(wrapper.text()).toContain('候选模型已移除不兼容项：逻辑回归')
    expect(wrapper.text()).toContain('优化指标已从ROC-AUC重置为RMSE')
    expect(wrapper.text()).not.toContain('候选模型和优化指标已重置')
    wrapper.unmount()
  })

  it('blocks malformed selected capabilities until retry supplies a complete entry', async () => {
    const malformed = {
      problem_types: {
        ...capabilities.problem_types,
        regression: { algorithms: ['linear_regression'], metrics: ['rmse'], default_metric: 'rmse', metric_directions: {} },
      },
    }
    trainingApiMocks.getCapabilities.mockResolvedValueOnce(malformed).mockResolvedValueOnce(capabilities)
    const wrapper = mountView()
    await flushPromises()
    await selectOption(wrapper, '问题类型', '回归')
    await chooseFile(wrapper, 'reg.csv')

    expect(wrapper.text()).toContain('无法验证显式任务能力')
    expect(actionButton(wrapper, '开始全自动训练').attributes('disabled')).toBeDefined()
    await actionButton(wrapper, '重试加载能力').trigger('click')
    await flushPromises()

    expect(wrapper.text()).not.toContain('无法验证显式任务能力')
    expect(actionButton(wrapper, '开始全自动训练').attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })

  it('blocks an empty capability response until retry succeeds', async () => {
    trainingApiMocks.getCapabilities.mockResolvedValueOnce({ problem_types: {} }).mockResolvedValueOnce(capabilities)
    const wrapper = mountView()
    await flushPromises()
    await selectOption(wrapper, '问题类型', '回归')
    await chooseFile(wrapper, 'reg.csv')

    expect(wrapper.text()).toContain('无法验证显式任务能力')
    expect(actionButton(wrapper, '开始全自动训练').attributes('disabled')).toBeDefined()
    await actionButton(wrapper, '重试加载能力').trigger('click')
    await flushPromises()

    expect(actionButton(wrapper, '开始全自动训练').attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })

  it('labels maximizing classification metrics without rendering regression cards', async () => {
    automlMocks.createJob.mockResolvedValue(queuedClassificationJob)
    automlMocks.streamJob.mockImplementation(async (_jobId, onEvent) => onEvent({ event: 'done', status: 'completed' }))
    automlMocks.getJob.mockResolvedValue({ ...queuedClassificationJob, status: 'completed', progress: 100 })
    automlMocks.getResult.mockResolvedValue(classificationResult)
    const wrapper = mountView()
    await flushPromises()
    await chooseFile(wrapper, 'class.csv')
    await start(wrapper)
    await flushPromises()

    expect(wrapper.text()).toContain('ROC-AUC（越大越好）')
    expect(wrapper.text()).toContain('ROC-AUC：0.91000')
    expect(wrapper.text()).not.toContain('RMSE：')
    expect(chartMocks.setOption).toHaveBeenLastCalledWith(expect.objectContaining({
      yAxis: expect.objectContaining({ name: 'ROC-AUC（越大越好）' }),
    }))
    wrapper.unmount()
  })

  it('reconnects after an SSE failure and recovers the completed snapshot', async () => {
    automlMocks.createJob.mockResolvedValue(queuedRegressionJob)
    automlMocks.streamJob
      .mockRejectedValueOnce(new Error('connection lost'))
      .mockImplementationOnce(async (_jobId: string, onEvent: (event: AutoMLTrialEvent) => Promise<void>) => onEvent({ event: 'done', status: 'completed' }))
    automlMocks.getJob
      .mockResolvedValueOnce({ ...queuedRegressionJob, status: 'running', progress: 50 })
      .mockResolvedValueOnce({ ...queuedRegressionJob, status: 'completed', progress: 100 })
    automlMocks.getResult.mockResolvedValue(regressionResult)
    const wrapper = mountView()
    await flushPromises()
    await chooseFile(wrapper, 'reg.csv')
    await start(wrapper)
    await flushPromises()

    expect(automlMocks.streamJob).toHaveBeenCalledTimes(2)
    expect(automlMocks.getResult).toHaveBeenCalledWith('job_reg')
    expect(wrapper.text()).toContain('RMSE：0.10000')
    wrapper.unmount()
  })

  it('keeps tracking a cancelling job until the terminal SSE snapshot is refreshed', async () => {
    let onEvent: ((event: AutoMLTrialEvent) => void | Promise<void>) | undefined
    automlMocks.createJob.mockResolvedValue(queuedRegressionJob)
    automlMocks.streamJob.mockImplementation((_jobId, eventHandler, signal) => new Promise<void>((resolve) => {
      onEvent = eventHandler
      signal?.addEventListener('abort', () => resolve(), { once: true })
    }))
    automlMocks.cancelJob.mockResolvedValue({ ...queuedRegressionJob, status: 'cancelling' })
    automlMocks.getJob.mockResolvedValue({ ...queuedRegressionJob, status: 'cancelled' })
    const wrapper = mountView()
    await flushPromises()
    await chooseFile(wrapper, 'reg.csv')
    await start(wrapper)
    await actionButton(wrapper, '取消任务').trigger('click')
    await flushPromises()

    expect(automlMocks.cancelJob).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('取消中')
    expect(actionButton(wrapper, '取消任务').attributes('disabled')).toBeDefined()
    await onEvent?.({ event: 'done', status: 'cancelled' })
    await flushPromises()

    expect(automlMocks.getJob).toHaveBeenCalledWith('job_reg')
    expect(wrapper.text()).toContain('已取消')
    wrapper.unmount()
  })
})
