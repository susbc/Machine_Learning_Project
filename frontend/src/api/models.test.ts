import { describe, expect, it, vi } from 'vitest'

import { createModelsApi } from './models'

describe('models api', () => {
  it('lists model registry records and fetches one model', async () => {
    const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/models' && init?.method === 'GET') {
        return new Response(
          JSON.stringify({
            models: [{ model_id: 'automl:automl_1', source: 'automl', artifact_path: 'saved/automl/automl_1.joblib', metrics: {} }],
          }),
          { status: 200 },
        )
      }
      if (url === '/api/models/automl%3Aautoml_1' && init?.method === 'GET') {
        return new Response(
          JSON.stringify({ model_id: 'automl:automl_1', source: 'automl', artifact_path: 'saved/automl/automl_1.joblib', metrics: {} }),
          { status: 200 },
        )
      }
      if (url === '/api/models/automl%3Aautoml_1/predict' && init?.method === 'POST') {
        return new Response(
          JSON.stringify({ model_id: 'automl:automl_1', source: 'automl', input_features: { age: 30 }, prediction: 1, probability: 0.82 }),
          { status: 200 },
        )
      }
      if (url === '/api/models/automl%3Aautoml_1/batch-predict' && init?.method === 'POST') {
        return new Response(
          JSON.stringify({
            model_id: 'automl:automl_1',
            source: 'automl',
            summary: { count: 2, positive_rate: 0.5, average_probability: 0.61 },
            rows: [
              { index: 0, features: { age: 30 }, prediction: 0, probability: 0.4 },
              { index: 1, features: { age: 70 }, prediction: 1, probability: 0.82 },
            ],
          }),
          { status: 200 },
        )
      }
      throw new Error(`unexpected request: ${url}`)
    }) as typeof fetch

    const api = createModelsApi({ baseUrl: '', fetcher })

    const list = await api.listModels()
    const model = await api.getModel('automl:automl_1')
    const prediction = await api.predict('automl:automl_1', { features: { age: 30 } })
    const batch = await api.batchPredict('automl:automl_1', { rows: [{ age: 30 }, { age: 70 }] })

    expect(list.models[0].model_id).toBe('automl:automl_1')
    expect(model.source).toBe('automl')
    expect(prediction.probability).toBe(0.82)
    expect(batch.summary.count).toBe(2)
  })
})
