import { describe, expect, it, vi } from 'vitest'

import { createOverviewApi } from './overview'

describe('overview api', () => {
  it('fetches the aggregated overview summary', async () => {
    const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
      expect(url).toBe('/api/overview/summary')
      expect(init?.method).toBe('GET')
      return new Response(
        JSON.stringify({
          generated_at: '2026-11-11T10:00:00',
          models: { total: 4, recent_7d: 1, distribution: [{ name: '分类模型', count: 4 }] },
          training: { running: 0, queued: 0, completed: 3 },
          connections: { total: 1, db_types: ['sqlite'] },
          llm: { total: 1, names: ['Qwen'] },
          workflow: { total: 2, running: 0 },
          trend: [{ label: '周一', submitted: 1, completed: 1 }],
          recent_tasks: [],
        }),
        { status: 200 },
      )
    })
    const api = createOverviewApi({ baseUrl: '', fetcher: fetcher as typeof fetch })

    const summary = await api.getSummary()
    expect(summary.models.total).toBe(4)
    expect(summary.connections.db_types).toEqual(['sqlite'])
    expect(summary.trend[0]?.label).toBe('周一')
  })
})
