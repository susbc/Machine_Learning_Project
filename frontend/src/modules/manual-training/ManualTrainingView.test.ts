import { flushPromises, mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it, vi } from 'vitest'

import ManualTrainingView from './views/ManualTrainingView.vue'

const trainingMocks = vi.hoisted(() => ({
  getCapabilities: vi.fn(),
  uploadDataset: vi.fn(),
  getFieldConfig: vi.fn(),
  createJob: vi.fn(),
  getJob: vi.fn(),
  getResult: vi.fn(),
  cancelJob: vi.fn(async () => ({
    job_id: 'job_1',
    dataset_id: 'dataset_1',
    algorithm: 'logistic_regression',
    status: 'cancelled',
    progress: 60,
  })),
}))

vi.mock('@/api/training', () => ({
  trainingApi: {
    uploadDataset: trainingMocks.uploadDataset,
    getFieldConfig: trainingMocks.getFieldConfig,
    getCapabilities: trainingMocks.getCapabilities,
    runEda: vi.fn(),
    createJob: trainingMocks.createJob,
    getJob: trainingMocks.getJob,
    getResult: trainingMocks.getResult,
    saveResult: vi.fn(),
    cancelJob: trainingMocks.cancelJob,
  },
}))

const regressionSpec = {
  schema_version: 1,
  requested_type: 'regression',
  resolved_type: 'regression',
  source: 'user_override',
  reason: 'user_selected_regression',
  explanation: '用户显式选择了回归任务',
  target: { column: 'price', dtype: 'float64', valid_rows: 80, unique_count: 80, unique_ratio: 1, integer_like: false },
  capabilities: {
    algorithms: ['linear_regression', 'random_forest'],
    metrics: ['rmse', 'mae', 'r2'],
    default_metric: 'rmse',
    metric_directions: { rmse: 'minimize', mae: 'minimize', r2: 'maximize' },
  },
} as const

const capabilities = {
  problem_types: {
    binary_classification: {
      algorithms: ['logistic_regression', 'random_forest'],
      metrics: ['roc_auc', 'accuracy', 'f1'],
      default_metric: 'roc_auc',
      metric_directions: { roc_auc: 'maximize', accuracy: 'maximize', f1: 'maximize' },
    },
    multiclass_classification: {
      algorithms: ['logistic_regression', 'random_forest'],
      metrics: ['accuracy', 'f1', 'roc_auc'],
      default_metric: 'accuracy',
      metric_directions: { accuracy: 'maximize', f1: 'maximize', roc_auc: 'maximize' },
    },
    regression: regressionSpec.capabilities,
  },
} as const

const regressionDataset = {
  dataset_id: 'dataset_reg',
  filename: 'reg.csv',
  rows: 80,
  columns: 2,
  target_col: 'price',
  created_at: '2026-09-09T00:00:00',
  preview: [],
  problem_spec: regressionSpec,
} as const

const regressionResult = {
  job_id: 'job_reg',
  dataset_id: 'dataset_reg',
  algorithm: 'linear_regression',
  params: {},
  metrics: { rmse: 0.1, mae: 0.08, r2: 0.99 },
  confusion_matrix: null,
  threshold_calibration: null,
  feature_importance: [],
  train_size: 60,
  test_size: 20,
  feature_columns: ['x'],
  problem_spec: regressionSpec,
} as const

describe('ManualTrainingView', () => {
  it('uses a regression override for compatible training and regression-only results', async () => {
    trainingMocks.getCapabilities.mockResolvedValue(capabilities)
    trainingMocks.uploadDataset.mockResolvedValue(regressionDataset)
    trainingMocks.getFieldConfig.mockResolvedValue({ dataset_id: 'dataset_reg', fields: [] })
    trainingMocks.createJob.mockResolvedValue({
      job_id: 'job_reg',
      dataset_id: 'dataset_reg',
      algorithm: 'linear_regression',
      status: 'queued',
      progress: 0,
      problem_spec: regressionSpec,
    })
    trainingMocks.getJob.mockResolvedValue({
      job_id: 'job_reg',
      dataset_id: 'dataset_reg',
      algorithm: 'linear_regression',
      status: 'completed',
      progress: 100,
      problem_spec: regressionSpec,
    })
    trainingMocks.getResult.mockResolvedValue(regressionResult)

    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    await wrapper.get('[aria-label="问题类型"]').trigger('click')
    const regressionOption = Array.from(document.querySelectorAll<HTMLElement>('.el-select-dropdown__item'))
      .find((option) => option.textContent?.trim() === '回归')
    expect(regressionOption).toBeDefined()
    regressionOption!.click()
    await wrapper.vm.$nextTick()

    const fileInput = wrapper.get('[data-testid="manual-training-file-input"]').element as HTMLInputElement
    Object.defineProperty(fileInput, 'files', {
      configurable: true,
      value: [new File(['x,price\n1,10'], 'reg.csv', { type: 'text/csv' })],
    })
    await wrapper.get('[data-testid="manual-training-file-input"]').trigger('change')
    await wrapper.findAll('button').find((button) => button.text() === '上传')!.trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('用户显式选择了回归任务')
    expect(wrapper.text()).toContain('唯一值 80')
    expect(wrapper.text()).toContain('线性回归')
    expect(trainingMocks.uploadDataset).toHaveBeenCalledWith(expect.objectContaining({ problemType: 'regression' }))

    await wrapper.findAll('button').find((button) => button.text() === '开始训练')!.trigger('click')
    await flushPromises()
    await wrapper.findAll('button').find((button) => button.text() === '刷新任务')!.trigger('click')
    await flushPromises()

    expect(trainingMocks.createJob).toHaveBeenCalledWith(expect.objectContaining({
      problemType: 'regression',
      algorithm: 'linear_regression',
    }))
    expect(wrapper.text()).toContain('RMSE')
    expect(wrapper.text()).toContain('MAE')
    expect(wrapper.text()).toContain('R²')
    expect(wrapper.text()).not.toContain('ROC-AUC')
    expect(wrapper.text()).not.toContain('最佳阈值')
    wrapper.unmount()
  })

  it('validates parameter JSON before creating a job', async () => {
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    const vm = wrapper.vm as unknown as { paramsJson: string }

    vm.paramsJson = '[]'
    await wrapper.vm.$nextTick()

    expect(wrapper.text()).toContain('模型参数必须是 JSON 对象')
    wrapper.unmount()
  })

  it('sends cancellation for a running job', async () => {
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    const vm = wrapper.vm as unknown as {
      job: Record<string, unknown>
      cancelJob: () => Promise<void>
    }
    vm.job = {
      job_id: 'job_1',
      dataset_id: 'dataset_1',
      algorithm: 'logistic_regression',
      status: 'running',
      progress: 60,
    }

    await vm.cancelJob()

    expect(trainingMocks.cancelJob).toHaveBeenCalledWith('job_1')
    expect(vm.job.status).toBe('cancelled')
    wrapper.unmount()
  })
})
