import { describe, expect, expectTypeOf, it, vi } from 'vitest'

import {
  createModelsApi,
  type ModelBatchPredictionRow,
  type ModelMetadataError,
  type ModelPredictionResponse,
  type ModelRegistryRecord,
} from './models'
import type { ProblemSpec } from './problem'

describe('models api', () => {
  it('types predictions, diagnostics, and malformed-v3 metadata exactly', () => {
    expectTypeOf<ModelPredictionResponse['prediction']>().toEqualTypeOf<string | number>()
    expectTypeOf<ModelPredictionResponse['encoded_prediction']>().toEqualTypeOf<number | null>()
    expectTypeOf<ModelPredictionResponse['probability']>().toEqualTypeOf<number | null>()
    expectTypeOf<ModelPredictionResponse['diagnostics']>().toEqualTypeOf<import('./models').PredictionDiagnostics>()
    expectTypeOf<ModelBatchPredictionRow['encoded_prediction']>().toEqualTypeOf<number | null>()
    expectTypeOf<ModelBatchPredictionRow['probability']>().toEqualTypeOf<number | null>()
    expectTypeOf<ModelMetadataError>().toMatchTypeOf<{ code: string; field: string; message: string }>()
    expectTypeOf<ModelRegistryRecord['optimization_metric']>().toEqualTypeOf<string | null>()
    expectTypeOf<ModelRegistryRecord['best_metric']>().toEqualTypeOf<string | null>()
    expectTypeOf<ProblemSpec['source']>().toEqualTypeOf<
      'auto_inferred' | 'user_override' | 'legacy_compatibility'
    >()
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
                metrics: {},
                optimization_metric: 'mae',
                best_metric: null,
                problem_type: 'regression',
                problem_spec: null,
                legacy: false,
                metadata_error: null,
              },
              {
                model_id: 'manual:broken',
                source: 'manual_training',
                artifact_path: 'saved/manual_training/broken.joblib',
                metrics: {},
                optimization_metric: null,
                best_metric: null,
                problem_type: null,
                problem_spec: null,
                legacy: false,
                metadata_error: {
                  code: 'invalid_artifact_problem_type',
                  field: 'problem_type',
                  message: 'schema-v3 模型产物必须保存已解析的任务类型，不能使用 auto',
                },
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
            metrics: {},
            optimization_metric: 'mae',
            best_metric: null,
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
            prediction: '高风险',
            encoded_prediction: 1,
            probability: 0.82,
            diagnostics: { unknown_input_keys: ['aeg'], missing_columns_imputed: ['age'] },
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
              {
                index: 0,
                features: { age: 30 },
                prediction: 12.5,
                encoded_prediction: null,
                probability: null,
                diagnostics: { unknown_input_keys: [], missing_columns_imputed: [] },
              },
              {
                index: 1,
                features: { age: 70 },
                prediction: 13.5,
                encoded_prediction: null,
                probability: null,
                diagnostics: { unknown_input_keys: [], missing_columns_imputed: [] },
              },
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
    expect(list.models[0].optimization_metric).toBe('mae')
    expect(list.models[0].problem_type).toBe('regression')
    expect(list.models[0].legacy).toBe(false)
    expect(list.models[0].metadata_error).toBeNull()
    expect(list.models[1].optimization_metric).toBeNull()
    expect(list.models[1].metadata_error).toEqual({
      code: 'invalid_artifact_problem_type',
      field: 'problem_type',
      message: 'schema-v3 模型产物必须保存已解析的任务类型，不能使用 auto',
    })
    expect(model.source).toBe('automl')
    expect(model.optimization_metric).toBe('mae')
    expect(prediction.prediction).toBe('高风险')
    expect(prediction.diagnostics.unknown_input_keys).toEqual(['aeg'])
    expect(batch.rows[0].probability).toBeNull()
    expect(batch.summary.count).toBe(2)
  })

  it('posts prediction exports to the export endpoint', async () => {
    const fetcher = vi.fn(async (_url: string, _init?: RequestInit) => new Response(
      JSON.stringify({
        target: 'file',
        exported: true,
        status: 'ok',
        warnings: [],
        row_count: 1,
        path: 'output/predictions_manual_m1.csv',
        connection_id: null,
        table_name: null,
        table_created: null,
      }),
      { status: 200 },
    ))
    const api = createModelsApi({ baseUrl: '', fetcher: fetcher as typeof fetch })
    const result = await api.exportPredictions({
      model_id: 'manual:m1',
      target: 'file',
      rows: [
        { index: 0, features: { age: 30 }, prediction: 1.5, encoded_prediction: null, probability: null, diagnostics: null },
      ],
      output_dir: 'output',
    })

    expect(result.exported).toBe(true)
    expect(result.path).toBe('output/predictions_manual_m1.csv')
    const [url, init] = fetcher.mock.calls[0]!
    expect(url).toBe('/api/prediction-exports')
    expect(init?.method).toBe('POST')
    expect(JSON.parse(init!.body as string)).toEqual({
      model_id: 'manual:m1',
      target: 'file',
      rows: [
        { index: 0, features: { age: 30 }, prediction: 1.5, encoded_prediction: null, probability: null, diagnostics: null },
      ],
      output_dir: 'output',
    })
  })
})

describe('large prediction batches', () => {
  it('chunks to the API limit and merges row order and weighted summaries', async () => {
    const fetcher = vi.fn(async (_url: string, init?: RequestInit) => {
      const { rows } = JSON.parse(init!.body as string) as { rows: { x: number }[] }
      expect(rows.length).toBeLessThanOrEqual(1000)
      const probability = rows[0]!.x === 0 ? 0.2 : 0.8
      return new Response(JSON.stringify({
        model_id: 'manual:test', source: 'manual_training',
        summary: { count: rows.length, positive_rate: probability, average_probability: probability },
        rows: rows.map((features, index) => ({ index, features, prediction: features.x, probability })),
      }))
    })
    const api = createModelsApi({ baseUrl: '', fetcher: fetcher as typeof fetch })
    const rows = Array.from({ length: 2505 }, (_, x) => ({ x }))
    const result = await api.batchPredict('manual:test', { rows })
    expect(fetcher).toHaveBeenCalledTimes(3)
    expect(result.summary.count).toBe(2505)
    expect(result.summary.average_probability).toBeCloseTo((1000 * 0.2 + 1505 * 0.8) / 2505)
    expect(result.rows.map((row) => row.index)).toEqual(rows.map((row) => row.x))
    expect(result.rows.map((row) => row.features)).toEqual(rows)
  })

  it('stops on a failed chunk instead of returning partial success', async () => {
    const fetcher = vi.fn(async () => new Response('{"detail":"failed"}', { status: 500 }))
    const api = createModelsApi({ baseUrl: '', fetcher: fetcher as typeof fetch })
    await expect(api.batchPredict('manual:test', { rows: Array.from({ length: 2001 }, () => ({ x: 1 })) })).rejects.toThrow()
    expect(fetcher).toHaveBeenCalledTimes(1)
  })
})
