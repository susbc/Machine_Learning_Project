import { describe, expect, it, vi } from 'vitest'

import { createWorkflowApi, parseSseEvent } from './workflow'

describe('workflow api', () => {
  it('uses stable Agent workflow endpoints and parses SSE payloads', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const body = init?.body ? JSON.parse(String(init.body)) : null

      if (url === '/api/workflow/step1_retrieve_data' && init?.method === 'POST') {
        expect(body.user_query).toContain('订单')
        return new Response(JSON.stringify({ status: 'success', task_id: 'task_1234abcd', eda_summary: { rows: 100 } }), {
          status: 200,
        })
      }
      if (url === '/api/workflow/step2_generate_features' && init?.method === 'POST') {
        expect(body.task_id).toBe('task_1234abcd')
        return new Response(JSON.stringify({ status: 'success', actions_count: 2, strategy: { actions: [] } }), { status: 200 })
      }
      if (url === '/api/workflow/task/task_1234abcd/status' && init?.method === 'GET') {
        return new Response(JSON.stringify({ status: 'completed', step1: 'done', step2: 'done', step3: 'done' }), { status: 200 })
      }
      if (url === '/api/workflow/task/task_1234abcd/importance' && init?.method === 'GET') {
        return new Response(JSON.stringify([{ feature: 'age', importance: 0.42 }]), { status: 200 })
      }
      if (url === '/api/workflow/models' && init?.method === 'GET') {
        return new Response(JSON.stringify({ models: [{ task_id: 'task_1234abcd' }] }), { status: 200 })
      }
      if (url === '/api/workflow/predict' && init?.method === 'POST') {
        return new Response(JSON.stringify({ status: 'success', prediction: 1, probability: 0.85 }), { status: 200 })
      }
      if (url.startsWith('/api/workflow/step3_train_stream?') && init?.method === 'GET') {
        return new Response('data: {"status":"DONE"}\n\n', { status: 200 })
      }
      throw new Error(`unexpected request: ${url}`)
    })
    const api = createWorkflowApi({ baseUrl: '', fetcher })

    expect((await api.retrieveData({ user_query: '查询订单', target_col: 'is_fraud' })).task_id).toBe('task_1234abcd')
    expect((await api.generateFeatures({ task_id: 'task_1234abcd' })).actions_count).toBe(2)
    expect((await api.getTaskStatus('task_1234abcd')).status).toBe('completed')
    expect((await api.getImportance('task_1234abcd'))[0].feature).toBe('age')
    expect((await api.listModels()).models[0].task_id).toBe('task_1234abcd')
    expect((await api.predict({ model_version: 'task_1234abcd', features: { age: 35 } })).probability).toBe(0.85)
    const streamEvents: string[] = []
    await api.streamTraining('task_1234abcd', 10, 3, (event) => {
      streamEvents.push(event.status)
    })
    expect(streamEvents).toEqual(['DONE'])

    expect(parseSseEvent('data: {"status":"RUNNING","trial":1}\\n\\n')).toEqual({ status: 'RUNNING', trial: 1 })
    expect(parseSseEvent('event: ping\\ndata: {"status":"DONE"}')).toEqual({ status: 'DONE' })
  })
})
