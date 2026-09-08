import { describe, expect, it, vi } from 'vitest'

import { buildDatasetAnalysisForm, createAnalysisApi } from './analysis'

describe('analysis api', () => {
  it('builds dataset analysis multipart payloads', () => {
    const file = new File(['x,label\n1,0'], 'sample.csv')

    const form = buildDatasetAnalysisForm({ file, targetCol: 'label' })

    expect(form.get('file')).toBe(file)
    expect(form.get('target_col')).toBe('label')
  })

  it('uses stable analysis endpoints for dataset, training, prediction, composite and reports', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/analysis/dataset' && init?.method === 'POST') {
        expect(init.body).toBeInstanceOf(FormData)
        return new Response(JSON.stringify({ analysis_type: 'dataset_rules', risk_level: 'high' }), { status: 200 })
      }
      if (url === '/api/analysis/training' && init?.method === 'POST') {
        return new Response(JSON.stringify({ analysis_type: 'training_rules', risk_level: 'medium' }), { status: 200 })
      }
      if (url === '/api/analysis/prediction' && init?.method === 'POST') {
        return new Response(JSON.stringify({ analysis_type: 'prediction_rules', risk_level: 'high' }), { status: 200 })
      }
      if (url === '/api/analysis/composite' && init?.method === 'POST') {
        return new Response(JSON.stringify({ result: { analysis_type: 'composite_diagnosis' }, report_path: 'x.md' }), {
          status: 200,
        })
      }
      if (url === '/api/analysis/reports' && init?.method === 'GET') {
        return new Response(JSON.stringify({ reports: [{ analysis_id: 'a1' }] }), { status: 200 })
      }
      throw new Error(`unexpected request: ${url}`)
    })
    const api = createAnalysisApi({ baseUrl: '', fetcher })
    const file = new File(['x,label\n1,0'], 'sample.csv')

    expect((await api.analyzeDataset({ file, targetCol: 'label' })).risk_level).toBe('high')
    expect((await api.analyzeTraining({ metrics: { val_auc: 0.65 } })).analysis_type).toBe('training_rules')
    expect((await api.analyzePrediction({ predictions: [0.99] })).analysis_type).toBe('prediction_rules')
    expect((await api.createCompositeReport({ results: [], save_report: true })).report_path).toBe('x.md')
    expect((await api.listReports()).reports[0].analysis_id).toBe('a1')
  })
})
