import { mount, flushPromises } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it, vi } from 'vitest'

import ModelRegistryView from './views/ModelRegistryView.vue'

vi.mock('@/api/models', () => ({
  modelsApi: {
    listModels: vi.fn(async () => ({
      models: [
        {
          model_id: 'automl:automl_1',
          source: 'automl',
          name: 'automl_1',
          artifact_path: 'saved/automl/automl_1.joblib',
          metrics: { roc_auc: 0.72 },
          updated_at: 1,
        },
        {
          model_id: 'manual:manual_1',
          source: 'manual_training',
          name: 'manual_1',
          artifact_path: 'saved/manual_training/manual_1.joblib',
          metrics: { accuracy: 0.8 },
          updated_at: 2,
        },
      ],
    })),
  },
}))

describe('ModelRegistryView', () => {
  it('shows unified model registry records and selected details', async () => {
    const wrapper = mount(ModelRegistryView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(wrapper.text()).toContain('模型管理')
    expect(wrapper.text()).toContain('AutoML')
    expect(wrapper.text()).toContain('手动训练')
    expect(wrapper.text()).toContain('automl_1')
    expect(wrapper.text()).toContain('0.7200')
    expect(wrapper.text()).toContain('saved/automl/automl_1.joblib')
    expect(wrapper.text()).toContain('模型对比')
    expect(wrapper.text()).toContain('已选择 2 个模型')
    expect(wrapper.text()).toContain('0.8000')
  })
})
