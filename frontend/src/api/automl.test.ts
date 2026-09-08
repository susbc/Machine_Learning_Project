import { describe, expect, it, vi } from 'vitest'

import { buildAutoMLJobForm, createAutoMLApi } from './automl'

describe('automl api', () => {
  it('builds multipart payloads for AutoML job creation', () => {
    const file = new File(['a,b,label\n1,2,0'], 'sample.csv', { type: 'text/csv' })

    const form = buildAutoMLJobForm({
      file,
      targetCol: 'label',
      nTrials: 12,
      cvFolds: 3,
      candidateModels: ['logistic_regression', 'random_forest'],
      optimizationMetric: 'roc_auc',
      useLlm: false,
      businessContext: 'credit risk',
    })

    expect(form.get('file')).toBe(file)
    expect(form.get('target_col')).toBe('label')
    expect(form.get('n_trials')).toBe('12')
    expect(form.get('cv_folds')).toBe('3')
    expect(form.get('candidate_models')).toBe(JSON.stringify(['logistic_regression', 'random_forest']))
    expect(form.get('optimization_metric')).toBe('roc_auc')
    expect(form.get('use_llm')).toBe('false')
    expect(form.get('business_context')).toBe('credit risk')
  })

  it('creates jobs, reads status, reads result, cancels and saves through the stable API paths', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/automl/jobs' && init?.method === 'POST') {
        expect(init.body).toBeInstanceOf(FormData)
        expect(init.headers).toBeUndefined()
        return new Response(JSON.stringify({ job_id: 'job_1', status: 'queued', progress: 0 }), { status: 201 })
      }
      if (url === '/api/automl/jobs/job_1' && init?.method === 'GET') {
        return new Response(JSON.stringify({ job_id: 'job_1', status: 'running', progress: 50 }), { status: 200 })
      }
      if (url === '/api/automl/jobs/job_1/result' && init?.method === 'GET') {
        return new Response(JSON.stringify({ best_model: 'random_forest', metrics: { roc_auc: 0.9 } }), { status: 200 })
      }
      if (url === '/api/automl/jobs/job_1/cancel' && init?.method === 'POST') {
        return new Response(JSON.stringify({ job_id: 'job_1', status: 'cancelling' }), { status: 200 })
      }
      if (url === '/api/automl/jobs/job_1/save' && init?.method === 'POST') {
        return new Response(JSON.stringify({ status: 'saved', artifact_path: 'saved/automl/model.joblib' }), {
          status: 200,
        })
      }
      if (url === '/api/automl/jobs/job_1/stream?cursor=0' && init?.method === 'GET') {
        return new Response('data: {"event":"done","status":"completed"}\n\n', { status: 200 })
      }
      throw new Error(`unexpected request: ${url}`)
    })
    const api = createAutoMLApi({ baseUrl: '', fetcher })
    const file = new File(['x,label\n1,0'], 'sample.csv')

    const created = await api.createJob({ file, nTrials: 2, cvFolds: 2 })
    const status = await api.getJob('job_1')
    const result = await api.getResult('job_1')
    const cancelled = await api.cancelJob('job_1')
    const saved = await api.saveResult('job_1')
    const events: string[] = []
    await api.streamJob('job_1', (event) => {
      events.push(event.event)
    })

    expect(created.job_id).toBe('job_1')
    expect(status.progress).toBe(50)
    expect(result.best_model).toBe('random_forest')
    expect(cancelled.status).toBe('cancelling')
    expect(saved.artifact_path).toBe('saved/automl/model.joblib')
    expect(events).toEqual(['done'])
  })
})
