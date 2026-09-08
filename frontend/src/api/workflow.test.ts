import { describe, expect, expectTypeOf, it, vi } from 'vitest'

import {
  createWorkflowApi,
  parseSseEvent,
  type AgentCorrection,
  type WorkflowPredictResponse,
  type WorkflowTrainEvent,
} from './workflow'
import type { ProblemSpec } from './problem'

describe('workflow api', () => {
  it('types workflow predictions and agent corrections for both backend response branches', () => {
    expectTypeOf<WorkflowPredictResponse['prediction']>().toEqualTypeOf<string | number>()
    expectTypeOf<WorkflowPredictResponse['encoded_prediction']>().toEqualTypeOf<number | null | undefined>()
    expectTypeOf<WorkflowPredictResponse['probability']>().toEqualTypeOf<number | null | undefined>()
    expectTypeOf<WorkflowPredictResponse['diagnostics']>().toEqualTypeOf<
      import('./models').PredictionDiagnostics | undefined
    >()
    expectTypeOf<WorkflowTrainEvent['problem_spec']>().toEqualTypeOf<ProblemSpec | undefined>()
    expectTypeOf<WorkflowTrainEvent['agent_corrections']>().toEqualTypeOf<AgentCorrection[] | undefined>()
  })

  it('adds problem type to workflow training stream URL', () => {
    const api = createWorkflowApi({ baseUrl: '' })

    expect(api.trainStreamUrl('task_1234abcd', 3, 3, 'regression')).toContain('problem_type=regression')
  })

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
        return new Response(
          JSON.stringify({
            status: 'success',
            prediction: '高风险',
            probability: 0.85,
            diagnostics: { unknown_input_keys: ['aeg'], missing_columns_imputed: ['age'] },
          }),
          { status: 200 },
        )
      }
      if (url.startsWith('/api/workflow/step3_train_stream?') && init?.method === 'GET') {
        const parsed = new URL(url, 'http://frontend.test')
        expect(parsed.searchParams.get('task_id')).toBe('task_1234abcd')
        expect(parsed.searchParams.get('n_trials')).toBe('10')
        expect(parsed.searchParams.get('cv_folds')).toBe('3')
        expect(parsed.searchParams.get('problem_type')).toBe('regression')
        return new Response(
          'data: {"status":"DONE","agent_corrections":[{"field":"candidate_models","suggested":["bad_model"],"resolved":["linear_regression"],"reason":"incompatible"}]}\n\n',
          { status: 200 },
        )
      }
      throw new Error(`unexpected request: ${url}`)
    })
    const api = createWorkflowApi({ baseUrl: '', fetcher })

    expect((await api.retrieveData({ user_query: '查询订单', target_col: 'is_fraud' })).task_id).toBe('task_1234abcd')
    expect((await api.generateFeatures({ task_id: 'task_1234abcd' })).actions_count).toBe(2)
    expect((await api.getTaskStatus('task_1234abcd')).status).toBe('completed')
    expect((await api.getImportance('task_1234abcd'))[0].feature).toBe('age')
    expect((await api.listModels()).models[0].task_id).toBe('task_1234abcd')
    const prediction = await api.predict({ model_version: 'task_1234abcd', features: { age: 35 } })
    expect(prediction.prediction).toBe('高风险')
    expect(prediction.diagnostics?.unknown_input_keys).toEqual(['aeg'])
    const streamEvents: WorkflowTrainEvent[] = []
    await api.streamTraining('task_1234abcd', 10, 3, 'regression', (event) => {
      streamEvents.push(event)
    })
    expect(streamEvents.map((event) => event.status)).toEqual(['DONE'])
    expect(streamEvents[0].agent_corrections?.[0]).toEqual({
      field: 'candidate_models',
      suggested: ['bad_model'],
      resolved: ['linear_regression'],
      reason: 'incompatible',
    })

    expect(parseSseEvent('data: {"status":"RUNNING","trial":1}\\n\\n')).toEqual({ status: 'RUNNING', trial: 1 })
    expect(parseSseEvent('event: ping\\ndata: {"status":"DONE"}')).toEqual({ status: 'DONE' })
  })

  it('preserves the legacy workflow call shape by streaming automatic task resolution', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const parsed = new URL(String(input), 'http://frontend.test')
      expect(parsed.searchParams.get('problem_type')).toBe('auto')
      return new Response('data: {"status":"DONE"}\n\n', { status: 200 })
    })
    const api = createWorkflowApi({ baseUrl: '', fetcher })
    const statuses: string[] = []

    await api.streamTraining('task_1234abcd', 3, 3, (event) => {
      statuses.push(event.status)
    })

    expect(statuses).toEqual(['DONE'])
  })
})
