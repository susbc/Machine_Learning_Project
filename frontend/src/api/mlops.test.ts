import { describe, expect, it, vi } from 'vitest'

import { createMlopsApi } from './mlops'

describe('mlops api', () => {
  it('uses stable MLOps endpoints for status, drift, health, logs and retrain', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/mlops/status' && init?.method === 'GET') {
        return new Response(JSON.stringify({ scheduler_running: true, jobs: [] }), { status: 200 })
      }
      if (url === '/api/mlops/drift-report?model_version=model_a' && init?.method === 'GET') {
        return new Response(JSON.stringify({ features: {}, summary: { needs_retrain: false } }), { status: 200 })
      }
      if (url === '/api/mlops/health' && init?.method === 'GET') {
        return new Response(JSON.stringify({ models: [] }), { status: 200 })
      }
      if (url === '/api/mlops/logs?days=14' && init?.method === 'GET') {
        return new Response(JSON.stringify({ logs: [{ 结果: '正常' }] }), { status: 200 })
      }
      if (url === '/api/mlops/retrain?model_version=model_a&force=true' && init?.method === 'POST') {
        return new Response(JSON.stringify({ status: 'success', scheduled_time: '2026-06-04 04:00:00' }), {
          status: 200,
        })
      }
      throw new Error(`unexpected request: ${url}`)
    })
    const api = createMlopsApi({ baseUrl: '', fetcher })

    expect((await api.getStatus()).scheduler_running).toBe(true)
    expect((await api.getDriftReport('model_a')).summary.needs_retrain).toBe(false)
    expect((await api.getModelHealth()).models).toEqual([])
    expect((await api.getLogs(14)).logs[0].结果).toBe('正常')
    expect((await api.triggerRetrain('model_a', true)).status).toBe('success')
  })
})
