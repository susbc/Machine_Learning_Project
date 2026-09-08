import { mount, flushPromises } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it, vi } from 'vitest'

import PredictionLabView from './views/PredictionLabView.vue'

const modelMocks = vi.hoisted(() => ({
  predict: vi.fn(async () => ({
    model_id: 'manual:manual_1',
    source: 'manual_training',
    input_features: { age: 70 },
    prediction: 1,
    probability: 0.84,
  })),
  batchPredict: vi.fn(async () => ({
    model_id: 'manual:manual_1',
    source: 'manual_training',
    summary: { count: 2, positive_rate: 0.5, average_probability: 0.62 },
    rows: [
      { index: 0, features: { age: 30 }, prediction: 0, probability: 0.4 },
      { index: 1, features: { age: 70 }, prediction: 1, probability: 0.84 },
    ],
  })),
}))

vi.mock('@/api/models', () => ({
  modelsApi: {
    listModels: vi.fn(async () => ({
      models: [
        {
          model_id: 'manual:manual_1',
          source: 'manual_training',
          name: 'manual_1',
          artifact_path: 'saved/manual_training/manual_1.joblib',
          metrics: {},
          updated_at: 1,
        },
      ],
    })),
    predict: modelMocks.predict,
    batchPredict: modelMocks.batchPredict,
  },
}))

describe('PredictionLabView', () => {
  it('loads models and runs a single-row prediction', async () => {
    const wrapper = mount(PredictionLabView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    expect(wrapper.text()).toContain('预测实验室')
    expect(wrapper.text()).toContain('manual:manual_1')

    const vm = wrapper.vm as unknown as { runPrediction: () => Promise<void>; featureJson: string }
    vm.featureJson = '{ "age": 70 }'
    await vm.runPrediction()
    await flushPromises()

    expect(modelMocks.predict).toHaveBeenCalledWith('manual:manual_1', { features: { age: 70 } })
    expect(wrapper.text()).toContain('0.8400')
  })

  it('parses CSV rows and runs batch prediction', async () => {
    const wrapper = mount(PredictionLabView, { global: { plugins: [ElementPlus] } })
    await flushPromises()

    const vm = wrapper.vm as unknown as { parseCsvText: (text: string) => void; runBatchPrediction: () => Promise<void> }
    vm.parseCsvText(`age,income
30,100
70,200`)
    await vm.runBatchPrediction()
    await flushPromises()

    expect(modelMocks.batchPredict).toHaveBeenCalledWith('manual:manual_1', {
      rows: [
        { age: 30, income: 100 },
        { age: 70, income: 200 },
      ],
    })
    expect(wrapper.text()).toContain('批量结果')
    expect(wrapper.text()).toContain('0.6200')
  })
})
