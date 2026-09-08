import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import AgentWorkflowView from './views/AgentWorkflowView.vue'

const workflowMocks = vi.hoisted(() => ({
  getTaskStatus: vi.fn(),
  retrieveData: vi.fn(),
  generateFeatures: vi.fn(),
  streamTraining: vi.fn(),
  getImportance: vi.fn(async () => []),
  listModels: vi.fn(),
  predict: vi.fn(),
  trainStreamUrl: vi.fn(() => '/api/workflow/step3_train_stream'),
}))

vi.mock('@/api/workflow', () => ({ workflowApi: workflowMocks }))

const regressionSpec = {
  schema_version: 1,
  requested_type: 'auto',
  resolved_type: 'regression',
  source: 'auto_inferred',
  reason: 'numeric_continuous_target',
  explanation: '数值目标具有连续取值，识别为回归任务',
  target: {
    column: 'order_fee',
    dtype: 'float64',
    valid_rows: 80,
    unique_count: 80,
    unique_ratio: 1,
    integer_like: false,
  },
  capabilities: {
    algorithms: ['linear_regression'],
    metrics: ['rmse', 'mae', 'r2'],
    default_metric: 'rmse',
    metric_directions: { rmse: 'minimize', mae: 'minimize', r2: 'maximize' },
  },
} as const

const classificationSpec = {
  ...regressionSpec,
  resolved_type: 'binary_classification',
  reason: 'binary_target',
  explanation: '目标列具有两个类别，识别为二分类任务',
  target: { ...regressionSpec.target, column: 'is_fraud', dtype: 'int64', unique_count: 2, unique_ratio: 0.025, integer_like: true },
  capabilities: {
    algorithms: ['logistic_regression'],
    metrics: ['roc_auc', 'accuracy', 'f1'],
    default_metric: 'roc_auc',
    metric_directions: { roc_auc: 'maximize', accuracy: 'maximize', f1: 'maximize' },
  },
} as const

const regressionWorkflowStatus = {
  task_id: 'task_ab12cd34',
  status: 'completed',
  best_metric: 'rmse',
  best_value: 0.12,
  suggested_type: 'binary_classification',
  agent_corrections: [
    {
      field: 'task_type',
      suggested: 'binary_classification',
      resolved: 'regression',
      reason: 'Agent 建议不能覆盖规范任务类型',
    },
    {
      field: 'optimization_metric',
      suggested: 'roc_auc',
      resolved: 'rmse',
      reason: '指标与规范任务类型不兼容',
    },
  ],
  problem_spec: regressionSpec,
} as const

function actionButton(wrapper: VueWrapper, label: string) {
  const button = wrapper.findAll('button').find((candidate) => candidate.text().trim() === label)
  if (!button) throw new Error(`Expected ${label} action`)
  return button
}

async function selectOption(wrapper: VueWrapper, controlLabel: string, optionLabel: string) {
  await wrapper.get(`[aria-label="${controlLabel}"]`).trigger('click')
  await flushPromises()
  const option = Array.from(document.querySelectorAll<HTMLElement>('.el-select-dropdown__item'))
    .reverse()
    .find((candidate) => candidate.textContent?.trim() === optionLabel)
  if (!option) throw new Error(`Expected ${optionLabel} option`)
  option.click()
  await flushPromises()
}

async function createWorkflow(wrapper: VueWrapper) {
  await actionButton(wrapper, '生成 SQL 并提取数据').trigger('click')
  await flushPromises()
  await actionButton(wrapper, '生成特征策略').trigger('click')
  await flushPromises()
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

describe('AgentWorkflowView', () => {
  beforeEach(() => {
    workflowMocks.getTaskStatus.mockReset()
    workflowMocks.retrieveData.mockReset()
    workflowMocks.generateFeatures.mockReset()
    workflowMocks.streamTraining.mockReset()
    workflowMocks.listModels.mockReset()
    workflowMocks.predict.mockReset()

    workflowMocks.retrieveData.mockResolvedValue({
      status: 'success',
      task_id: 'task_ab12cd34',
      data_shape: [80, 3],
    })
    workflowMocks.generateFeatures.mockResolvedValue({
      status: 'success',
      task_id: 'task_ab12cd34',
      strategy: { actions: [] },
      actions_count: 0,
    })
    workflowMocks.getTaskStatus.mockResolvedValue(regressionWorkflowStatus)
    workflowMocks.listModels.mockResolvedValue({
      models: [{ task_id: 'task_ab12cd34', model_path: 'saved/model_task_ab12cd34.pkl' }],
    })
    workflowMocks.predict.mockResolvedValue({
      status: 'success',
      model_version: 'task_ab12cd34',
      prediction: 123.45,
      encoded_prediction: null,
      probability: null,
      diagnostics: { unknown_input_keys: ['aeg'], missing_columns_imputed: ['age'] },
    })
    workflowMocks.streamTraining.mockResolvedValue(undefined)
  })

  it('submits the rendered automatic or user-selected task type to the training stream', async () => {
    workflowMocks.streamTraining.mockImplementation(async (...args: unknown[]) => {
      const onEvent = args[4] as (event: { status: string; error: string }) => Promise<void>
      await onEvent({ status: 'ERROR', error: '测试结束训练流' })
    })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await createWorkflow(wrapper)

    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()
    expect(workflowMocks.streamTraining).toHaveBeenLastCalledWith(
      'task_ab12cd34',
      10,
      3,
      'auto',
      expect.any(Function),
      expect.any(AbortSignal),
    )

    await selectOption(wrapper, '工作流问题类型', '回归')

    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()

    expect(workflowMocks.streamTraining).toHaveBeenCalledWith(
      'task_ab12cd34',
      10,
      3,
      'regression',
      expect.any(Function),
      expect.any(AbortSignal),
    )
    wrapper.unmount()
  })

  it('renders canonical task metadata as soon as the training stream provides it', async () => {
    workflowMocks.getTaskStatus.mockResolvedValue({ task_id: 'task_ab12cd34', status: 'running' })
    workflowMocks.streamTraining.mockImplementation(async (...args: unknown[]) => {
      const onEvent = args[4] as (event: Record<string, unknown>) => Promise<void>
      await onEvent({
        status: 'RUNNING',
        trial: 0,
        metric: 'rmse',
        best_metric: 'rmse',
        best_value: 0.2,
        problem_spec: regressionSpec,
        suggested_type: 'binary_classification',
        agent_corrections: regressionWorkflowStatus.agent_corrections,
      })
    })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await createWorkflow(wrapper)

    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('建议：二分类')
    expect(wrapper.text()).toContain('最终：回归')
    expect(wrapper.text()).toContain('Agent 建议不能覆盖规范任务类型')
    expect(wrapper.text()).toContain('RMSE 0.2000')
    wrapper.unmount()
  })

  it('renders metric/value-only RUNNING and DONE events as the canonical live metric', async () => {
    workflowMocks.getTaskStatus.mockResolvedValue({ task_id: 'task_ab12cd34', status: 'running' })
    workflowMocks.streamTraining.mockImplementation(async (...args: unknown[]) => {
      const onEvent = args[4] as (event: Record<string, unknown>) => Promise<void>
      await onEvent({ status: 'RUNNING', trial: 1, metric: 'mae', value: 0.31 })
      await onEvent({ status: 'DONE', trial: 10, metric: 'mae', value: 0.19 })
    })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await createWorkflow(wrapper)

    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('MAE')
    expect(wrapper.text()).toContain('0.1900')
    expect(wrapper.text()).toContain('0.3100')
    wrapper.unmount()
  })

  it('recovers a failed status refresh and renders final metadata, structured corrections, metrics, and regression diagnostics', async () => {
    workflowMocks.getTaskStatus.mockRejectedValueOnce(new Error('状态服务暂不可用')).mockResolvedValue(regressionWorkflowStatus)
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()

    await actionButton(wrapper, '生成 SQL 并提取数据').trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('状态服务暂不可用')

    await wrapper.get('[data-testid="workflow-refresh-status-button"]').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()

    expect(wrapper.text()).not.toContain('状态服务暂不可用')
    expect(wrapper.text()).toContain('建议：二分类')
    expect(wrapper.text()).toContain('最终：回归')
    expect(wrapper.text()).toContain('来源：自动推断')
    expect(wrapper.text()).toContain('依据：numeric_continuous_target')
    expect(wrapper.text()).toContain('数值目标具有连续取值，识别为回归任务')
    expect(wrapper.text()).toContain('task_type')
    expect(wrapper.text()).toContain('binary_classification → regression')
    expect(wrapper.text()).toContain('Agent 建议不能覆盖规范任务类型')
    expect(wrapper.text()).toContain('RMSE')
    expect(wrapper.text()).toContain('0.1200')
    expect(wrapper.text()).toContain('123.45')
    expect(wrapper.text()).toContain('未知输入字段：aeg')
    expect(wrapper.text()).toContain('已补齐缺失字段：age')
    expect(wrapper.text()).not.toContain('预测概率')
    expect(wrapper.text()).not.toContain('类别编码')
    wrapper.unmount()
  })

  it('preserves classification probability and encoded prediction while using generic metric fields', async () => {
    workflowMocks.getTaskStatus.mockResolvedValue({
      task_id: 'task_ab12cd34',
      status: 'completed',
      best_metric: 'roc_auc',
      best_value: 0.91,
      suggested_type: 'binary_classification',
      agent_corrections: [],
      problem_spec: classificationSpec,
    })
    workflowMocks.predict.mockResolvedValue({
      status: 'success',
      model_version: 'task_ab12cd34',
      prediction: '违约',
      encoded_prediction: 1,
      probability: 0.84,
      diagnostics: { unknown_input_keys: [], missing_columns_imputed: [] },
    })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()

    await actionButton(wrapper, '生成 SQL 并提取数据').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('ROC-AUC')
    expect(wrapper.text()).toContain('0.9100')
    expect(wrapper.text()).toContain('违约')
    expect(wrapper.text()).toContain('类别编码')
    expect(wrapper.text()).toContain('预测概率')
    expect(wrapper.text()).toContain('0.8400')
    wrapper.unmount()
  })

  it('shows classification response fields after selecting a classification model from a regression workflow', async () => {
    workflowMocks.listModels.mockResolvedValue({
      models: [
        { task_id: 'task_ab12cd34', model_path: 'saved/model_task_ab12cd34.pkl' },
        { task_id: 'class_ef56gh78', model_path: 'saved/model_class_ef56gh78.pkl' },
      ],
    })
    workflowMocks.predict.mockResolvedValue({
      status: 'success',
      model_version: 'class_ef56gh78',
      prediction: '违约',
      encoded_prediction: 1,
      probability: 0.84,
      diagnostics: { unknown_input_keys: [], missing_columns_imputed: [] },
    })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await actionButton(wrapper, '生成 SQL 并提取数据').trigger('click')
    await flushPromises()

    await selectOption(wrapper, '工作流预测模型', 'class_ef56gh78')
    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('类别编码')
    expect(wrapper.text()).toContain('预测概率')
    expect(wrapper.text()).toContain('0.8400')
    wrapper.unmount()
  })

  it('locks task/type replacement and rejects duplicate starts while a server run is active', async () => {
    workflowMocks.streamTraining.mockImplementation((...args: unknown[]) => {
      const onEvent = args[4] as (event: Record<string, unknown>) => Promise<void>
      void onEvent({ status: 'RUNNING', trial: 1, metric: 'rmse', value: 0.27, problem_spec: regressionSpec })
      return new Promise<void>(() => undefined)
    })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await createWorkflow(wrapper)
    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('0.2700')
    expect(wrapper.get('[data-testid="workflow-problem-type-select"] input').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-testid="workflow-retrieve-button"]').attributes('disabled')).toBeDefined()
    await actionButton(wrapper, '开始训练').trigger('click')
    await wrapper.get('[data-testid="workflow-retrieve-button"]').trigger('click')
    await flushPromises()
    expect(workflowMocks.streamTraining).toHaveBeenCalledTimes(1)
    expect(workflowMocks.retrieveData).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('renders the canonical safe error from a real backend SSE payload shape', async () => {
    workflowMocks.streamTraining.mockImplementation(async (...args: unknown[]) => {
      const onEvent = args[4] as (event: Record<string, unknown>) => Promise<void>
      await onEvent({ status: 'ERROR', error: '模型训练失败，请检查配置后重试', run_id: 'run_123' })
    })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await createWorkflow(wrapper)

    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="workflow-error-alert"]').text()).toContain('模型训练失败，请检查配置后重试')
    wrapper.unmount()
  })

  it('renders the canonical safe error restored from the backend status snapshot', async () => {
    workflowMocks.getTaskStatus.mockResolvedValue({
      task_id: 'task_ab12cd34',
      status: 'failed',
      error: '模型训练失败，请检查配置后重试',
    })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()

    await actionButton(wrapper, '生成 SQL 并提取数据').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="workflow-error-alert"]').text()).toContain('模型训练失败，请检查配置后重试')
    wrapper.unmount()
  })

  it('ignores late callbacks after the server run ends and the workflow task is replaced', async () => {
    let lateEvent: ((event: Record<string, unknown>) => Promise<void>) | undefined
    let oldSignal: AbortSignal | undefined
    workflowMocks.retrieveData
      .mockResolvedValueOnce({ status: 'success', task_id: 'task_old', data_shape: [80, 3] })
      .mockResolvedValueOnce({ status: 'success', task_id: 'task_new', data_shape: [42, 3] })
    workflowMocks.generateFeatures.mockResolvedValue({
      status: 'success', task_id: 'task_old', strategy: { actions: [] }, actions_count: 0,
    })
    workflowMocks.getTaskStatus
      .mockResolvedValueOnce({ task_id: 'task_old', status: 'running' })
      .mockResolvedValueOnce({ task_id: 'task_old', status: 'running' })
      .mockResolvedValueOnce({ task_id: 'task_new', status: 'created' })
    workflowMocks.streamTraining.mockImplementation((...args: unknown[]) => {
      lateEvent = args[4] as (event: Record<string, unknown>) => Promise<void>
      oldSignal = args[5] as AbortSignal
      return new Promise<void>(() => undefined)
    })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await createWorkflow(wrapper)
    await actionButton(wrapper, '开始训练').trigger('click')
    await flushPromises()
    await lateEvent?.({ status: 'ERROR', error: '训练流已断开' })
    await flushPromises()
    const statusCallsBeforeReplacement = workflowMocks.getTaskStatus.mock.calls.length
    const modelCallsBeforeReplacement = workflowMocks.listModels.mock.calls.length

    await actionButton(wrapper, '生成 SQL 并提取数据').trigger('click')
    await flushPromises()

    expect(oldSignal?.aborted).toBe(true)
    expect(wrapper.text()).toContain('task_new')
    expect(workflowMocks.getTaskStatus).toHaveBeenCalledTimes(statusCallsBeforeReplacement + 1)
    expect(workflowMocks.listModels).toHaveBeenCalledTimes(modelCallsBeforeReplacement)

    await lateEvent?.({
      status: 'DONE', trial: 10, metric: 'mae', value: 0.01, problem_spec: regressionSpec,
    })
    await flushPromises()

    expect(wrapper.text()).toContain('task_new')
    expect(wrapper.text()).not.toContain('0.0100')
    expect(wrapper.text()).not.toContain('任务类型解析')
    expect(workflowMocks.getTaskStatus).toHaveBeenCalledTimes(statusCallsBeforeReplacement + 1)
    expect(workflowMocks.listModels).toHaveBeenCalledTimes(modelCallsBeforeReplacement)
    wrapper.unmount()
  })

  it('falls back to legacy best_auc without showing a fabricated task resolution', async () => {
    workflowMocks.getTaskStatus.mockResolvedValue({
      task_id: 'task_ab12cd34',
      status: 'completed',
      best_auc: 0.81,
    })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()

    await actionButton(wrapper, '生成 SQL 并提取数据').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('ROC-AUC')
    expect(wrapper.text()).toContain('0.8100')
    expect(wrapper.text()).not.toContain('最终：')
    wrapper.unmount()
  })

  it('ignores a deferred prediction result after the JSON input changes', async () => {
    const pending = deferred<{ status: string; prediction: number }>()
    workflowMocks.predict.mockReturnValue(pending.promise)
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()

    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()
    await wrapper.get('.predict-json textarea').setValue('{"order_fee":999,"age":41,"city_code":"上海"}')
    pending.resolve({ status: 'success', prediction: 111.25 })
    await flushPromises()

    expect(wrapper.text()).not.toContain('111.25')
    expect(actionButton(wrapper, '执行预测').classes()).not.toContain('is-loading')
    wrapper.unmount()
  })

  it('keeps the current model prediction when a superseded model request rejects late', async () => {
    const oldRequest = deferred<{ status: string; prediction: number }>()
    workflowMocks.listModels.mockResolvedValue({
      models: [{ task_id: 'model_old' }, { task_id: 'model_new' }],
    })
    workflowMocks.predict
      .mockReturnValueOnce(oldRequest.promise)
      .mockResolvedValueOnce({ status: 'success', model_version: 'model_new', prediction: 222.5 })
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()

    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()
    await selectOption(wrapper, '工作流预测模型', 'model_new')
    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()
    oldRequest.reject(new Error('old model failed'))
    await flushPromises()

    expect(wrapper.text()).toContain('222.5')
    expect(wrapper.text()).not.toContain('old model failed')
    wrapper.unmount()
  })

  it('ignores a deferred prediction when the workflow task changes', async () => {
    const pending = deferred<{ status: string; prediction: number }>()
    workflowMocks.retrieveData
      .mockResolvedValueOnce({ status: 'success', task_id: 'task_old', data_shape: [80, 3] })
      .mockResolvedValueOnce({ status: 'success', task_id: 'task_new', data_shape: [40, 3] })
    workflowMocks.getTaskStatus
      .mockResolvedValueOnce({ task_id: 'task_old', status: 'created' })
      .mockResolvedValueOnce({ task_id: 'task_new', status: 'created' })
    workflowMocks.predict.mockReturnValue(pending.promise)
    const wrapper = mount(AgentWorkflowView, { attachTo: document.body, global: { plugins: [ElementPlus] } })
    await flushPromises()
    await actionButton(wrapper, '生成 SQL 并提取数据').trigger('click')
    await flushPromises()

    await actionButton(wrapper, '执行预测').trigger('click')
    await flushPromises()
    await actionButton(wrapper, '生成 SQL 并提取数据').trigger('click')
    await flushPromises()
    pending.resolve({ status: 'success', prediction: 333.75 })
    await flushPromises()

    expect(wrapper.text()).toContain('task_new')
    expect(wrapper.text()).not.toContain('333.75')
    expect(actionButton(wrapper, '执行预测').classes()).not.toContain('is-loading')
    wrapper.unmount()
  })
})
