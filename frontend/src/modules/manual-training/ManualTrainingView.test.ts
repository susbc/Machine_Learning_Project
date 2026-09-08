import { flushPromises, mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ManualTrainingView from './views/ManualTrainingView.vue'

const trainingMocks = vi.hoisted(() => ({
  getCapabilities: vi.fn(),
  uploadDataset: vi.fn(),
  getFieldConfig: vi.fn(),
  runEda: vi.fn(),
  createJob: vi.fn(),
  getJob: vi.fn(),
  getResult: vi.fn(),
  saveResult: vi.fn(),
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
    runEda: trainingMocks.runEda,
    createJob: trainingMocks.createJob,
    getJob: trainingMocks.getJob,
    getResult: trainingMocks.getResult,
    saveResult: trainingMocks.saveResult,
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

const classificationSpec = {
  schema_version: 1,
  requested_type: 'auto',
  resolved_type: 'binary_classification',
  source: 'auto_inferred',
  reason: 'string_target',
  explanation: '字符串目标列自动解析为二分类任务',
  target: { column: 'label', dtype: 'object', valid_rows: 80, unique_count: 2, unique_ratio: 0.025, integer_like: false },
  capabilities: capabilities.problem_types.binary_classification,
} as const

const classificationDataset = {
  dataset_id: 'dataset_cls',
  filename: 'class.csv',
  rows: 80,
  columns: 2,
  target_col: 'label',
  created_at: '2026-09-09T00:00:00',
  preview: [],
  problem_spec: classificationSpec,
} as const

const classificationResult = {
  job_id: 'job_cls',
  dataset_id: 'dataset_cls',
  algorithm: 'logistic_regression',
  params: {},
  metrics: { roc_auc: 0.91, accuracy: 0.88, f1: 0.86, best_threshold: 0.5 },
  confusion_matrix: [[10, 2], [1, 7]],
  feature_importance: [],
  train_size: 60,
  test_size: 20,
  feature_columns: ['x'],
  problem_spec: classificationSpec,
} as const

function actionButton(wrapper: ReturnType<typeof mount>, label: string) {
  const button = wrapper.findAll('button').find((candidate) => candidate.text().trim() === label)
  if (!button) throw new Error(`Expected ${label} action`)
  return button
}

async function selectOption(wrapper: ReturnType<typeof mount>, controlLabel: string, optionLabel: string) {
  await wrapper.get(`[aria-label="${controlLabel}"]`).trigger('click')
  const option = Array.from(document.querySelectorAll<HTMLElement>('.el-select-dropdown__item'))
    .reverse()
    .find((candidate) => candidate.textContent?.trim() === optionLabel)
  if (!option) throw new Error(`Expected ${optionLabel} option`)
  option.click()
  await flushPromises()
}

async function optionLabels(wrapper: ReturnType<typeof mount>, controlLabel: string) {
  const control = wrapper.get(`[aria-label="${controlLabel}"]`)
  await control.trigger('click')
  await flushPromises()
  const listbox = document.getElementById(control.attributes('aria-controls')!)
  return Array.from(listbox?.querySelectorAll<HTMLElement>('.el-select-dropdown__item') ?? [])
    .map((option) => option.textContent?.trim())
}

async function chooseFile(wrapper: ReturnType<typeof mount>, filename = 'train.csv') {
  const fileInput = wrapper.get('[data-testid="manual-training-file-input"]').element as HTMLInputElement
  Object.defineProperty(fileInput, 'files', {
    configurable: true,
    value: [new File(['x,target\n1,0'], filename, { type: 'text/csv' })],
  })
  await wrapper.get('[data-testid="manual-training-file-input"]').trigger('change')
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe('ManualTrainingView', () => {
  beforeEach(() => {
    trainingMocks.getCapabilities.mockReset()
    trainingMocks.uploadDataset.mockReset()
    trainingMocks.getFieldConfig.mockReset()
    trainingMocks.runEda.mockReset()
    trainingMocks.createJob.mockReset()
    trainingMocks.getJob.mockReset()
    trainingMocks.getResult.mockReset()
    trainingMocks.saveResult.mockReset()
    trainingMocks.cancelJob.mockReset()
    trainingMocks.getCapabilities.mockResolvedValue(capabilities)
    trainingMocks.getFieldConfig.mockResolvedValue({ dataset_id: 'dataset_cls', fields: [] })
    trainingMocks.runEda.mockResolvedValue({
      dataset_id: 'dataset_cls',
      summary: { missing_rate: 0, numeric_features: 1, categorical_features: 0, target_unique: 2 },
      columns: [],
      correlations: {},
    })
    trainingMocks.saveResult.mockResolvedValue({
      status: 'saved', job_id: 'job_cls', artifact_path: 'saved/manual_old.pkl',
    })
    trainingMocks.cancelJob.mockResolvedValue({
      job_id: 'job_1',
      dataset_id: 'dataset_1',
      algorithm: 'logistic_regression',
      status: 'cancelled',
      progress: 60,
    })
  })

  it('keeps default auto in the upload and job payloads', async () => {
    trainingMocks.uploadDataset.mockResolvedValue(classificationDataset)
    trainingMocks.createJob.mockResolvedValue({
      job_id: 'job_cls', dataset_id: 'dataset_cls', algorithm: 'logistic_regression', status: 'queued', progress: 0, problem_spec: classificationSpec,
    })
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    await chooseFile(wrapper)
    await actionButton(wrapper, '上传').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()

    expect(trainingMocks.uploadDataset).toHaveBeenCalledWith(expect.objectContaining({ problemType: 'auto' }))
    expect(trainingMocks.createJob).toHaveBeenCalledWith(expect.objectContaining({ problemType: 'auto' }))
    wrapper.unmount()
  })

  it('keeps explicit actions disabled after a capability failure until retry succeeds', async () => {
    trainingMocks.getCapabilities.mockRejectedValueOnce(new Error('capability network failure')).mockResolvedValue(capabilities)
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    await selectOption(wrapper, '问题类型', '回归')
    await chooseFile(wrapper)

    expect(wrapper.text()).toContain('无法验证显式任务能力')
    expect(actionButton(wrapper, '上传').attributes('disabled')).toBeDefined()
    expect(actionButton(wrapper, '开始训练').attributes('disabled')).toBeDefined()

    await actionButton(wrapper, '重试加载能力').trigger('click')
    await flushPromises()

    expect(wrapper.text()).not.toContain('无法验证显式任务能力')
    expect(actionButton(wrapper, '上传').attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })

  it('keeps explicit actions disabled when the capability response is empty until retry succeeds', async () => {
    trainingMocks.getCapabilities.mockResolvedValueOnce({ problem_types: {} }).mockResolvedValue(capabilities)
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    await selectOption(wrapper, '问题类型', '回归')
    await chooseFile(wrapper)

    expect(wrapper.text()).toContain('无法验证显式任务能力')
    expect(actionButton(wrapper, '上传').attributes('disabled')).toBeDefined()

    await actionButton(wrapper, '重试加载能力').trigger('click')
    await flushPromises()

    expect(actionButton(wrapper, '上传').attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })

  it('filters algorithms for each explicit type without overwriting compatible edited params', async () => {
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    await selectOption(wrapper, '问题类型', '二分类')
    expect(await optionLabels(wrapper, '训练算法')).toEqual(['逻辑回归', '随机森林'])
    const paramsInput = wrapper.get('[aria-label="模型参数 JSON"]')
    await paramsInput.setValue('{\n  "C": 7\n}')
    await selectOption(wrapper, '问题类型', '多分类')

    expect(wrapper.text()).toContain('逻辑回归')
    expect((paramsInput.element as HTMLTextAreaElement).value).toBe('{\n  "C": 7\n}')
    expect(wrapper.text()).not.toContain('模型参数 JSON 已重置')

    await selectOption(wrapper, '问题类型', '回归')
    expect(await optionLabels(wrapper, '训练算法')).toEqual(['线性回归', '随机森林'])
    expect(wrapper.text()).toContain('线性回归')
    expect(wrapper.text()).toContain('算法已从逻辑回归重置为线性回归')
    expect(wrapper.text()).toContain('模型参数 JSON 已重置')
    expect((paramsInput.element as HTMLTextAreaElement).value).toBe('{}')
    wrapper.unmount()
  })

  it('keeps classification result metrics when optional calibration fields are absent', async () => {
    trainingMocks.uploadDataset.mockResolvedValue(classificationDataset)
    trainingMocks.createJob.mockResolvedValue({
      job_id: 'job_cls', dataset_id: 'dataset_cls', algorithm: 'logistic_regression', status: 'queued', progress: 0, problem_spec: classificationSpec,
    })
    trainingMocks.getJob.mockResolvedValue({
      job_id: 'job_cls', dataset_id: 'dataset_cls', algorithm: 'logistic_regression', status: 'completed', progress: 100, problem_spec: classificationSpec,
    })
    trainingMocks.getResult.mockResolvedValue(classificationResult)
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    await chooseFile(wrapper, 'class.csv')
    await actionButton(wrapper, '上传').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '刷新任务').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('ROC-AUC')
    expect(wrapper.text()).toContain('Accuracy')
    expect(wrapper.text()).toContain('最佳阈值')
    expect(wrapper.text()).not.toContain('RMSE')
    expect(wrapper.text()).not.toContain('阈值来源')
    wrapper.unmount()
  })

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

    await selectOption(wrapper, '问题类型', '回归')

    await chooseFile(wrapper, 'reg.csv')
    await actionButton(wrapper, '上传').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('用户显式选择了回归任务')
    expect(wrapper.text()).toContain('唯一值 80')
    expect(wrapper.text()).toContain('线性回归')
    expect(trainingMocks.uploadDataset).toHaveBeenCalledWith(expect.objectContaining({ problemType: 'regression' }))

    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '刷新任务').trigger('click')
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
      datasetFile: File
      uploadedDatasetFile: File
      dataset: Record<string, unknown>
      job: Record<string, unknown>
      cancelJob: () => Promise<void>
    }
    const file = new File(['x,label\n1,0'], 'running.csv', { type: 'text/csv' })
    vm.datasetFile = file
    vm.uploadedDatasetFile = file
    vm.dataset = { ...classificationDataset, dataset_id: 'dataset_1' }
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

  it('invalidates every dataset-bound view state immediately when a new file is selected', async () => {
    trainingMocks.uploadDataset.mockResolvedValue(classificationDataset)
    trainingMocks.getFieldConfig.mockResolvedValue({
      dataset_id: 'dataset_cls',
      fields: [{ field_name: 'old_feature', field_type: 'num', dtype: 'float64', missing_rate: 0, unique_count: 80 }],
    })
    trainingMocks.runEda.mockResolvedValue({
      dataset_id: 'dataset_cls',
      summary: { missing_rate: 0, numeric_features: 1, categorical_features: 0, target_unique: 2 },
      columns: [{ field_name: 'old_feature', field_type: 'num', dtype: 'float64', missing_rate: 0, unique_count: 80 }],
      correlations: {},
    })
    trainingMocks.createJob.mockResolvedValue({
      job_id: 'job_cls', dataset_id: 'dataset_cls', algorithm: 'logistic_regression', status: 'queued', progress: 0, problem_spec: classificationSpec,
    })
    trainingMocks.getJob.mockResolvedValue({
      job_id: 'job_cls', dataset_id: 'dataset_cls', algorithm: 'logistic_regression', status: 'completed', progress: 100, problem_spec: classificationSpec,
    })
    trainingMocks.getResult.mockResolvedValue(classificationResult)
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    await flushPromises()
    await chooseFile(wrapper, 'old.csv')
    await actionButton(wrapper, '上传').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '运行 EDA').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '刷新任务').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '保存模型').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('old_feature')
    expect(wrapper.text()).toContain('0.91000')
    expect(wrapper.text()).toContain('saved/manual_old.pkl')

    await chooseFile(wrapper, 'replacement.csv')
    await flushPromises()

    expect(wrapper.text()).toContain('待上传')
    expect(wrapper.text()).not.toContain('行数 80')
    expect(wrapper.text()).not.toContain('old_feature')
    expect(wrapper.text()).not.toContain('0.91000')
    expect(wrapper.text()).not.toContain('saved/manual_old.pkl')
    expect(actionButton(wrapper, '开始训练').attributes('disabled')).toBeDefined()
    await actionButton(wrapper, '开始训练').trigger('click')
    expect(trainingMocks.createJob).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('ignores a deferred field response after its uploaded file is replaced', async () => {
    const fieldResponse = deferred<{ dataset_id: string; fields: [] }>()
    trainingMocks.uploadDataset.mockResolvedValue(classificationDataset)
    trainingMocks.getFieldConfig.mockReturnValue(fieldResponse.promise)
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    await flushPromises()
    await chooseFile(wrapper, 'old.csv')
    await actionButton(wrapper, '上传').trigger('click')
    await flushPromises()

    await chooseFile(wrapper, 'replacement.csv')
    fieldResponse.resolve({ dataset_id: 'dataset_cls', fields: [] })
    await flushPromises()

    expect(wrapper.text()).toContain('待上传')
    expect(wrapper.text()).not.toContain('行数 80')
    expect(actionButton(wrapper, '开始训练').attributes('disabled')).toBeDefined()
    wrapper.unmount()
  })

  it('ignores a deferred job status after its dataset file is replaced', async () => {
    const statusResponse = deferred<typeof classificationResult & { status: 'completed'; progress: number }>()
    trainingMocks.uploadDataset.mockResolvedValue(classificationDataset)
    trainingMocks.createJob.mockResolvedValue({
      job_id: 'job_cls', dataset_id: 'dataset_cls', algorithm: 'logistic_regression', status: 'queued', progress: 0, problem_spec: classificationSpec,
    })
    trainingMocks.getJob.mockReturnValue(statusResponse.promise)
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    await flushPromises()
    await chooseFile(wrapper, 'old.csv')
    await actionButton(wrapper, '上传').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '刷新任务').trigger('click')
    await flushPromises()

    await chooseFile(wrapper, 'replacement.csv')
    statusResponse.resolve({ ...classificationResult, status: 'completed', progress: 100 })
    await flushPromises()

    expect(trainingMocks.getResult).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('待上传')
    expect(wrapper.text()).not.toContain('0.91000')
    wrapper.unmount()
  })

  it('ignores a deferred result after its job and dataset file are replaced', async () => {
    const resultResponse = deferred<typeof classificationResult>()
    trainingMocks.uploadDataset.mockResolvedValue(classificationDataset)
    trainingMocks.createJob.mockResolvedValue({
      job_id: 'job_cls', dataset_id: 'dataset_cls', algorithm: 'logistic_regression', status: 'queued', progress: 0, problem_spec: classificationSpec,
    })
    trainingMocks.getJob.mockResolvedValue({
      job_id: 'job_cls', dataset_id: 'dataset_cls', algorithm: 'logistic_regression', status: 'completed', progress: 100, problem_spec: classificationSpec,
    })
    trainingMocks.getResult.mockReturnValue(resultResponse.promise)
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    await flushPromises()
    await chooseFile(wrapper, 'old.csv')
    await actionButton(wrapper, '上传').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '刷新任务').trigger('click')
    await flushPromises()

    await chooseFile(wrapper, 'replacement.csv')
    resultResponse.resolve(classificationResult)
    await flushPromises()

    expect(wrapper.text()).toContain('待上传')
    expect(wrapper.text()).not.toContain('0.91000')
    wrapper.unmount()
  })
})
