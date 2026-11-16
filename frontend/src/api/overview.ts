import { createRequestJson } from './client'

export interface OverviewDistributionItem {
  name: string
  count: number
}

export interface OverviewTrendPoint {
  label: string
  submitted: number
  completed: number
}

export interface OverviewRecentTask {
  name: string
  source: string
  type: string
  model: string
  status: string
  submitted_at: string | null
  progress: number
}

export interface OverviewSummary {
  generated_at: string
  models: {
    total: number
    recent_7d: number
    distribution: OverviewDistributionItem[]
  }
  training: {
    running: number
    queued: number
    completed: number
  }
  connections: {
    total: number
    db_types: string[]
  }
  llm: {
    total: number
    names: string[]
  }
  workflow: {
    total: number
    running: number
  }
  trend: OverviewTrendPoint[]
  recent_tasks: OverviewRecentTask[]
}

interface OverviewApiOptions {
  baseUrl: string
  fetcher?: typeof fetch
}

export function createOverviewApi(options: OverviewApiOptions) {
  const requestJson = createRequestJson({ ...options, label: 'Overview' })

  return {
    getSummary: () => requestJson<OverviewSummary>('/api/overview/summary', { method: 'GET' }),
  }
}

export const overviewApi = createOverviewApi({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
