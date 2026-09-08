import { flushPromises, mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { AutoMLJob, AutoMLResult } from '@/api/automl'
import type { ProblemSpec, ProblemType } from '@/api/problem'
import AutoMLView from './views/AutoMLView.vue'

const automlMocks = vi.hoisted(() => ({
  createJob: vi.fn(),
  getJob: vi.fn(),
  getResult: vi.fn(),
  cancelJob: vi.fn(),
  saveResult: vi.fn(),
  streamJob: vi.fn(),
}))
const trainingApiMocks = vi.hoisted(() => ({ getCapabilities: vi.fn() }))
const chartMocks = vi.hoisted(() => ({
  setOption: vi.fn(),
  resize: vi.fn(),
  dispose: vi.fn(),
}))

vi.mock('@/api/automl', () => ({ automlApi: automlMocks }))
vi.mock('@/api/training', () => ({ trainingApi: trainingApiMocks }))
vi.mock('@/shared/charts/echarts', () => ({ init: vi.fn(() => chartMocks), }))

const regressionSpec = {
  schema_version: 1,
  requested_type: 'auto',
  resolved_type: 'regression',
  source: 'auto_inferred',
  reason: 'numeric_continuous_target',
  explanation: '连续数值目标，识别为回归任务',
  target: { column: 'price', dtype: 'float64', valid_rows: 80, unique_count: 80, unique_ratio: 1, integer_like: false },
  capabilities: {
    algorithms: ['linear_regression'],
    metrics: ['rmse', 'mae', 'r2'],
    default_metric: 'rmse',
    metric_directions: { rmse: 'minimize', mae: 'minimize', r2: 'maximize' },
  },
} satisfies ProblemSpec

const classificationSpec = {
  schema_version: 1,
  requested_type: 'auto',
  resolved_type: 'binary_classification',
  source: 'auto_inferred',
  reason: 'string_target',
  explanation: '字符串目标自动识别为二分类任务',
  target: { column: 'label', dtype: 'object', valid_rows: 80, unique_count: 2, unique_ratio: 0.025, integer_like: false },
  capabilities: {
    algorithms: ['logistic_regression'],
    metrics: ['roc_auc', 'accuracy', 'f1'],
    default_metric: 'roc_auc',
    metric_directions: { roc_auc: 'maximize', accuracy: 'maximize', f1: 'maximize' },
  },
} satisfies ProblemSpec

const capabilities = {
  problem_types: {
    binary_classification: classificationSpec.capabilities,
    multiclass_classification: {
      algorithms: ['random_forest'],
      metrics: ['accuracy'],
      default_metric: 'accuracy',
      metric_directions: { accuracy: 'maximize' },
    },
    regression: regressionSpec.capabilities,
  },
} as const

const regressionResult: AutoMLResult = {
  job_id: 'job_reg',
  status: 'completed',
  best_model: 'linear_regression',
  best_params: { model_type: 'linear_regression' },
  best_value: 0.1,
  metrics: { rmse: 0.1, mae: 0.08, r2: 0.99 },
  trial_history: [{ event: 'trial', trial_number: 0, value: 0.1, best_value: 0.1, model_type: 'linear_regression', params: {} }],
  decision_summary: {},
  feature_columns: ['x'],
  problem_spec: regressionSpec,
}

const classificationResult: AutoMLResult = {
  job_id: 'job_cls',
  status: 'completed',
  best_model: 'logistic_regression',
  best_params: { model_type: 'logistic_regression' },
  best_value: 0.91,
  metrics: { roc_auc: 0.91, accuracy: 0.88, f1: 0.86 },
  trial_history: [],
  decision_summary: {},
  feature_columns: ['x'],
  problem_spec: classificationSpec,
}

const queuedRegressionJob = {
  job_id: 'job_reg',
  status: 'queued',
  progress: 0,
  current_trial: 0,
  n_trials: 3,
  cv_folds: 3,
  optimization_metric: 'rmse',
  target_col: 'price',
  decision_summary: {},
  problem_spec: regressionSpec,
} as const

function mountView() {
  return mount(AutoMLView, { global: { plugins: [ElementPlus] } })
}

describe('AutoMLView', () => {
  beforeEach(() => {
    localStorage.clear()
    automlMocks.createJob.mockReset()
    automlMocks.getJob.mockReset()
    automlMocks.getResult.mockReset()
    automlMocks.cancelJob.mockReset()
    automlMocks.saveResult.mockReset()
    automlMocks.streamJob.mockReset()
    trainingApiMocks.getCapabilities.mockReset()
    chartMocks.setOption.mockReset()
    trainingApiMocks.getCapabilities.mockResolvedValue(capabilities)
    automlMocks.getJob.mockRejectedValue(new Error('no stored job'))
    automlMocks.streamJob.mockResolvedValue(undefined)
  })

  it('submits automatic problem and metric defaults', async () => {
    automlMocks.createJob.mockResolvedValue(queuedRegressionJob)
    const wrapper = mountView()
    await flushPromises()
    const vm = wrapper.vm as unknown as { datasetFile: File; problemType: ProblemType; startJob: () => Promise<void> }
    vm.datasetFile = new File(['x,price\n1,2.5'], 'reg.csv', { type: 'text/csv' })
    vm.problemType = 'auto'

    await vm.startJob()

    expect(automlMocks.createJob).toHaveBeenCalledWith(expect.objectContaining({
      problemType: 'auto',
      optimizationMetric: 'auto',
      candidateModels: undefined,
    }))
    wrapper.unmount()
  })

  it('uses the response problem specification for regression metrics and minimize labels', async () => {
    const wrapper = mountView()
    await flushPromises()
    const vm = wrapper.vm as unknown as { result: AutoMLResult | null }
    vm.result = regressionResult
    await wrapper.vm.$nextTick()

    expect(wrapper.text()).toContain('连续数值目标')
    expect(wrapper.text()).toContain('RMSE：0.10000')
    expect(wrapper.text()).toContain('MAE：0.08000')
    expect(wrapper.text()).toContain('R²：0.99000')
    expect(wrapper.text()).toContain('RMSE（越小越好）')
    expect(wrapper.text()).not.toContain('ROC-AUC：')
    expect(chartMocks.setOption).toHaveBeenLastCalledWith(expect.objectContaining({
      yAxis: expect.objectContaining({ name: 'RMSE（越小越好）' }),
    }))
    wrapper.unmount()
  })

  it('keeps the final explicit metric and raw trial values when its direction minimizes', async () => {
    const wrapper = mountView()
    await flushPromises()
    const vm = wrapper.vm as unknown as {
      job: AutoMLJob | null
      result: AutoMLResult | null
      trialHistory: Array<{ event: 'trial'; trial_number: number; value: number; best_value: number; model_type: 'linear_regression'; params: Record<string, unknown> }>
    }
    vm.job = { ...queuedRegressionJob, optimization_metric: 'mae' }
    vm.result = regressionResult
    vm.trialHistory = [{ event: 'trial', trial_number: 0, value: 0.08, best_value: 0.08, model_type: 'linear_regression', params: {} }]
    await wrapper.vm.$nextTick()

    expect(wrapper.text()).toContain('MAE（越小越好）')
    expect(chartMocks.setOption).toHaveBeenLastCalledWith(expect.objectContaining({
      yAxis: expect.objectContaining({ name: 'MAE（越小越好）' }),
      series: expect.arrayContaining([expect.objectContaining({ data: [[0, 0.08]] })]),
    }))
    wrapper.unmount()
  })

  it('keeps classification result cards free of regression metrics', async () => {
    const wrapper = mountView()
    await flushPromises()
    const vm = wrapper.vm as unknown as { result: AutoMLResult | null }
    vm.result = classificationResult
    await wrapper.vm.$nextTick()

    expect(wrapper.text()).toContain('ROC-AUC：0.91000')
    expect(wrapper.text()).toContain('Accuracy：0.88000')
    expect(wrapper.text()).not.toContain('RMSE：')
    wrapper.unmount()
  })

  it('blocks explicit requests when capabilities cannot be verified', async () => {
    trainingApiMocks.getCapabilities.mockRejectedValueOnce(new Error('offline'))
    const wrapper = mountView()
    await flushPromises()
    const vm = wrapper.vm as unknown as { datasetFile: File; problemType: ProblemType; startJob: () => Promise<void> }
    vm.datasetFile = new File(['x,price\n1,2.5'], 'reg.csv', { type: 'text/csv' })
    vm.problemType = 'regression'
    await wrapper.vm.$nextTick()

    expect(wrapper.text()).toContain('无法验证显式任务能力')
    await vm.startJob()
    expect(automlMocks.createJob).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('resets incompatible explicit controls and tells the user', async () => {
    const wrapper = mountView()
    await flushPromises()
    const vm = wrapper.vm as unknown as {
      problemType: ProblemType
      candidateModels: string[]
      optimizationMetric: string
      applyProblemType: () => void
    }
    vm.problemType = 'binary_classification'
    vm.candidateModels = ['logistic_regression']
    vm.optimizationMetric = 'roc_auc'
    vm.problemType = 'regression'
    vm.applyProblemType()
    await wrapper.vm.$nextTick()

    expect(vm.candidateModels).toEqual(['linear_regression'])
    expect(vm.optimizationMetric).toBe('rmse')
    expect(wrapper.text()).toContain('候选模型和优化指标已重置')
    wrapper.unmount()
  })

  it('refreshes the final snapshot after a done event without changing raw trial values', async () => {
    automlMocks.createJob.mockResolvedValue(queuedRegressionJob)
    automlMocks.streamJob.mockImplementation(async (_jobId, onEvent) => {
      await onEvent({ event: 'trial', trial_number: 0, value: 0.4, best_value: 0.4, model_type: 'linear_regression', params: {} })
      await onEvent({ event: 'done', status: 'completed' })
    })
    automlMocks.getJob.mockResolvedValue({ ...queuedRegressionJob, status: 'completed', progress: 100 })
    automlMocks.getResult.mockResolvedValue(regressionResult)
    const wrapper = mountView()
    await flushPromises()
    const vm = wrapper.vm as unknown as { datasetFile: File; startJob: () => Promise<void> }
    vm.datasetFile = new File(['x,price\n1,2.5'], 'reg.csv', { type: 'text/csv' })

    await vm.startJob()
    await flushPromises()

    expect(automlMocks.getResult).toHaveBeenCalledWith('job_reg')
    expect(wrapper.text()).toContain('0.10000')
    expect(wrapper.text()).not.toContain('-0.10000')
    wrapper.unmount()
  })

  it('keeps cancellation wired to the current job', async () => {
    automlMocks.cancelJob.mockResolvedValue({ ...queuedRegressionJob, status: 'cancelled' })
    const wrapper = mountView()
    await flushPromises()
    const vm = wrapper.vm as unknown as { job: AutoMLJob | null; cancelJob: () => Promise<void> }
    vm.job = queuedRegressionJob

    await vm.cancelJob()

    expect(automlMocks.cancelJob).toHaveBeenCalledWith('job_reg')
    expect(vm.job?.status).toBe('cancelled')
    wrapper.unmount()
  })
})
