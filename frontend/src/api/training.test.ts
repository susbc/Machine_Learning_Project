import { describe, expect, it, vi } from 'vitest'

import { buildTrainingDatasetForm, createTrainingApi } from './training'

describe('training api', () => {
  it('builds dataset upload form data', () => {
    const file = new File(['x,label\n1,0'], 'train.csv')
    const form = buildTrainingDatasetForm({ file, targetCol: 'label' })

    expect(form.get('file')).toBe(file)
    expect(form.get('target_col')).toBe('label')
  })

  it('uses stable manual training endpoints', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/training/datasets' && init?.method === 'POST') {
        expect(init.body).toBeInstanceOf(FormData)
        return new Response(JSON.stringify({ dataset_id: 'd1', rows: 10, columns: 3, target_col: 'label' }), {
          status: 201,
        })
      }
      if (url === '/api/training/datasets/d1/field-config' && init?.method === 'GET') {
        return new Response(JSON.stringify({ fields: [{ field_name: 'label', field_type: 'label' }] }), { status: 200 })
      }
      if (url === '/api/training/eda' && init?.method === 'POST') {
        return new Response(JSON.stringify({ summary: { rows: 10 } }), { status: 200 })
      }
      if (url === '/api/training/jobs' && init?.method === 'POST') {
        return new Response(JSON.stringify({ job_id: 'j1', status: 'queued', progress: 0 }), { status: 201 })
      }
      if (url === '/api/training/jobs/j1' && init?.method === 'GET') {
        return new Response(JSON.stringify({ job_id: 'j1', status: 'completed', progress: 100 }), { status: 200 })
      }
      if (url === '/api/training/jobs/j1/result' && init?.method === 'GET') {
        return new Response(JSON.stringify({ algorithm: 'logistic_regression', metrics: { roc_auc: 0.8 } }), {
          status: 200,
        })
      }
      if (url === '/api/training/jobs/j1/cancel' && init?.method === 'POST') {
        return new Response(JSON.stringify({ job_id: 'j1', status: 'cancelling', progress: 80 }), { status: 200 })
      }
      if (url === '/api/training/jobs/j1/save' && init?.method === 'POST') {
        return new Response(JSON.stringify({ status: 'saved', artifact_path: 'saved/manual/model.joblib' }), {
          status: 200,
        })
      }
      throw new Error(`unexpected request: ${url}`)
    })
    const api = createTrainingApi({ baseUrl: '', fetcher })
    const file = new File(['x,label\n1,0'], 'train.csv')

    expect((await api.uploadDataset({ file, targetCol: 'label' })).dataset_id).toBe('d1')
    expect((await api.getFieldConfig('d1')).fields[0].field_name).toBe('label')
    expect((await api.runEda('d1')).summary.rows).toBe(10)
    expect((await api.createJob({ dataset_id: 'd1', algorithm: 'logistic_regression', params: {} })).job_id).toBe('j1')
    expect((await api.getJob('j1')).progress).toBe(100)
    expect((await api.getResult('j1')).metrics.roc_auc).toBe(0.8)
    expect((await api.cancelJob('j1')).status).toBe('cancelling')
    expect((await api.saveResult('j1')).artifact_path).toContain('manual')
  })
})
