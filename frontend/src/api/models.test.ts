import { describe, expect, expectTypeOf, it, vi } from 'vitest'

import { createModelsApi, type ModelBatchPredictionRow, type ModelPredictionResponse } from './models'

describe('models api', () => {
  it('types regression-only prediction fields as nullable', () => {
    expectTypeOf<ModelPredictionResponse['encoded_prediction']>().toEqualTypeOf<number | null>()
    expectTypeOf<ModelPredictionResponse['probability']>().toEqualTypeOf<number | null>()
    expectTypeOf<ModelBatchPredictionRow['encoded_prediction']>().toEqualTypeOf<number | null>()
    expectTypeOf<ModelBatchPredictionRow['probability']>().toEqualTypeOf<number | null>()
  })

  it('lists model registry records and fetches one model', async () => {
    const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/models' && init?.method === 'GET') {
        return new Response(
          JSON.stringify({
            models: [
              {
                model_id: 'automl:automl_1',
                source: 'automl',
                artifact_path: 'saved/automl/automl_1.joblib',
                metrics: { rmse: 1.2 },
                problem_type: 'regression',
                problem_spec: null,
                legacy: false,
                metadata_error: null,
              },
            ],
          }),
          { status: 200 },
        )
      }
      if (url === '/api/models/automl%3Aautoml_1' && init?.method === 'GET') {
        return new Response(
          JSON.stringify({
            model_id: 'automl:automl_1',
            source: 'automl',
            artifact_path: 'saved/automl/automl_1.joblib',
            metrics: { rmse: 1.2 },
            problem_type: 'regression',
            problem_spec: null,
            legacy: false,
            metadata_error: null,
          }),
          { status: 200 },
        )
      }
      if (url === '/api/models/automl%3Aautoml_1/predict' && init?.method === 'POST') {
        return new Response(
          JSON.stringify({
            model_id: 'automl:automl_1',
            source: 'automl',
            input_features: { age: 30 },
            prediction: 12.5,
            encoded_prediction: null,
            probability: null,
          }),
          { status: 200 },
        )
      }
      if (url === '/api/models/automl%3Aautoml_1/batch-predict' && init?.method === 'POST') {
        return new Response(
          JSON.stringify({
            model_id: 'automl:automl_1',
            source: 'automl',
            summary: { count: 2, positive_rate: null, average_probability: null },
            rows: [
              { index: 0, features: { age: 30 }, prediction: 12.5, encoded_prediction: null, probability: null },
              { index: 1, features: { age: 70 }, prediction: 13.5, encoded_prediction: null, probability: null },
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
    expect(list.models[0].problem_type).toBe('regression')
    expect(list.models[0].legacy).toBe(false)
    expect(model.source).toBe('automl')
    expect(prediction.encoded_prediction).toBeNull()
    expect(prediction.probability).toBeNull()
    expect(batch.rows[0].probability).toBeNull()
    expect(batch.summary.count).toBe(2)
  })
})
