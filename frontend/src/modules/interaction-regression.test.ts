import { mount, flushPromises } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it, vi } from 'vitest'

import AutoMLView from './automl/views/AutoMLView.vue'
import AgentWorkflowView from './agent-workflow/views/AgentWorkflowView.vue'
import AiAnalysisView from './ai-analysis/views/AiAnalysisView.vue'
import LlmConfigView from './llm-config/views/LlmConfigView.vue'
import ManualTrainingView from './manual-training/views/ManualTrainingView.vue'

const llmMocks = vi.hoisted(() => ({
  updateProvider: vi.fn(async (providerId: string, input: Record<string, unknown>) => ({
    id: providerId,
    name: input.name,
    type: input.type,
    enabled: input.enabled,
    is_active: false,
    config: { model: 'qwen-plus', api_key_masked: 'sk-...cret', has_api_key: true },
  })),
}))

vi.mock('@/api/llm', () => ({
  providerTypeOptions: [
    { label: 'OpenAI / 兼容接口', value: 'openai', defaultModel: 'gpt-4o-mini', defaultBaseUrl: 'https://api.openai.com/v1' },
    { label: '通义千问 Qwen', value: 'qwen', defaultModel: 'qwen-plus', defaultBaseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1' },
  ],
  llmApi: {
    listProviders: vi.fn(async () => ({
      providers: [
        {
          id: 'p1',
          name: 'Existing',
          type: 'openai',
          enabled: true,
          is_active: true,
          config: { model: 'gpt-4o-mini', api_key_masked: 'sk-...cret', has_api_key: true },
        },
      ],
      active_provider: null,
      statistics: { total_providers: 1, enabled_providers: 1, provider_types: ['openai'] },
    })),
    createProvider: vi.fn(),
    updateProvider: llmMocks.updateProvider,
    deleteProvider: vi.fn(),
    testProvider: vi.fn(),
    activateProvider: vi.fn(),
  },
}))

vi.mock('@/api/workflow', () => ({
  workflowApi: {
    retrieveData: vi.fn(),
    generateFeatures: vi.fn(),
    trainStreamUrl: vi.fn(() => '/api/workflow/step3_train_stream'),
    streamTraining: vi.fn(),
    getTaskStatus: vi.fn(),
    getImportance: vi.fn(),
    listModels: vi.fn(async () => ({
      models: [
        { task_id: 'task_1234abcd', model_path: 'saved/model_task_1234abcd.pkl' },
        { task_id: 'task_5678efab', model_path: 'saved/model_task_5678efab.pkl' },
      ],
    })),
    predict: vi.fn(),
  },
}))

vi.mock('@/api/mlops', () => ({
  mlopsApi: {
    getModelHealth: vi.fn(async () => ({
      models: [{ 模型: 'task_1234abcd', 最佳指标: 0.61, 指标名: 'roc_auc', 健康度: '漂移告警', 调用次数: 12 }],
    })),
  },
}))

vi.mock('@/api/analysis', () => ({
  analysisApi: {
    analyzeDataset: vi.fn(),
    analyzeTraining: vi.fn(),
    analyzePrediction: vi.fn(),
    createCompositeReport: vi.fn(),
    listReports: vi.fn(async () => ({
      reports: [
        {
          analysis_id: 'analysis_1',
          analysis_type: 'training_rules',
          risk_level: 'medium',
          summary: '最近训练诊断',
        },
      ],
    })),
  },
}))

vi.mock('@/shared/charts/echarts', () => ({
  init: vi.fn(() => ({
    setOption: vi.fn(),
    resize: vi.fn(),
    dispose: vi.fn(),
  })),
}))

describe('frontend interaction regressions', () => {
  it('keeps stable native file inputs for AutoML and manual training uploads', () => {
    const automl = mount(AutoMLView, { global: { plugins: [ElementPlus] } })
    const manual = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })

    expect(automl.find('[data-testid="automl-file-input"]').exists()).toBe(true)
    expect(manual.find('[data-testid="manual-training-file-input"]').exists()).toBe(true)
  })

  it('sends edited LLM provider type with the update payload', async () => {
    const wrapper = mount(LlmConfigView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    const vm = wrapper.vm as unknown as {
      form: { name: string; type: string; model: string; baseUrl: string; apiKey: string; description: string; enabled: boolean }
      saveProvider: () => Promise<void>
    }
    vm.form.name = 'Existing tuned'
    vm.form.type = 'qwen'
    vm.form.model = 'qwen-plus'
    vm.form.baseUrl = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    vm.form.apiKey = ''
    vm.form.description = ''
    vm.form.enabled = true
    await vm.saveProvider()

    expect(llmMocks.updateProvider).toHaveBeenCalledWith(
      'p1',
      expect.objectContaining({
        type: 'qwen',
      }),
    )
  })

  it('shows the complete LLM provider create form', async () => {
    const wrapper = mount(LlmConfigView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    const vm = wrapper.vm as unknown as {
      startCreate: () => void
    }
    vm.startCreate()
    await wrapper.vm.$nextTick()

    expect(wrapper.find('[data-testid="llm-provider-name-input"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="llm-provider-type-select"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="llm-provider-model-input"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="llm-provider-base-url-input"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="llm-provider-api-key-input"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="llm-provider-description-input"]').exists()).toBe(true)
  })

  it('makes workflow status refresh and model list actions visible', async () => {
    const wrapper = mount(AgentWorkflowView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    await wrapper.get('[data-testid="workflow-refresh-status-button"]').trigger('click')
    await wrapper.vm.$nextTick()
    expect(wrapper.text()).toContain('暂无任务可刷新')

    await wrapper.get('[data-testid="workflow-model-list-button"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="workflow-model-list-panel"]').text()).toContain('task_1234abcd')
    expect(wrapper.get('[data-testid="workflow-model-list-panel"]').text()).toContain('task_5678efab')
  })

  it('connects AI analysis with model, MLOps, LLM, and report context', async () => {
    const wrapper = mount(AiAnalysisView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(wrapper.get('[data-testid="analysis-context-strip"]').text()).toContain('跨模块上下文')
    expect(wrapper.get('[data-testid="analysis-context-strip"]').text()).toContain('task_1234abcd')
    expect(wrapper.get('[data-testid="analysis-context-strip"]').text()).toContain('Existing')
    expect(wrapper.get('[data-testid="analysis-context-strip"]').text()).toContain('最近训练诊断')

    await wrapper.get('[data-testid="analysis-import-health-button"]').trigger('click')
    await wrapper.vm.$nextTick()

    const vm = wrapper.vm as unknown as { trainingJson: string }
    expect(vm.trainingJson).toContain('"best_value": 0.61')
    expect(vm.trainingJson).toContain('"best_metric": "roc_auc"')
    expect(vm.trainingJson).toContain('"source": "mlops_health"')
  })
})
