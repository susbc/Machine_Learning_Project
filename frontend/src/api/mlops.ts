import { createRequestJson } from './client'

export interface MlopsStatus {
  scheduler_running: boolean
  active_task_id?: string | null
  next_run?: string | null
  jobs: Array<{ id: string; name?: string; next_run?: string | null }>
}

export interface DriftFeature {
  psi?: number
  severity?: string
  status?: string
  [key: string]: unknown
}

export interface DriftReport {
  features: Record<string, DriftFeature>
  summary: {
    total_features?: number
    drifted_features?: number
    critical_features?: number
    needs_retrain?: boolean
  }
  message?: string
}

export interface ModelHealthRow {
  模型?: string
  最佳指标?: number | null
  指标名?: string | null
  健康度?: string
  调用次数?: number | null
  '预测延迟 (ms)'?: number | null
}

export interface InspectResultItem {
  model_id: string
  status: 'ok' | 'no_baseline' | 'insufficient_online' | string
  message: string
  summary?: DriftReport['summary'] | null
}

export interface InspectResponse {
  inspected: number
  reports: number
  results: InspectResultItem[]
}

export interface OnlineSourceBinding {
  connection_id: string
  table_name: string
  time_column?: string
  limit?: number
}

export interface MlopsSettings {
  psi_threshold_low: number
  psi_threshold_high: number
}

export interface InspectionLogRow {
  时间: string
  事件: string
  结果: string
  详情: string
}

export interface RetrainResponse {
  status: string
  message?: string
  job_id?: string
  scheduled_time?: string
}

interface MlopsApiOptions {
  baseUrl: string
  fetcher?: typeof fetch
}

export function createMlopsApi(options: MlopsApiOptions) {
  const requestJson = createRequestJson({ ...options, label: 'MLOps' })

  return {
    getStatus: () => requestJson<MlopsStatus>('/api/mlops/status', { method: 'GET' }),
    getDriftReport: (modelVersion: string) =>
      requestJson<DriftReport>(`/api/mlops/drift-report?model_version=${encodeURIComponent(modelVersion)}`, {
        method: 'GET',
      }),
    getModelHealth: () => requestJson<{ models: ModelHealthRow[] }>('/api/mlops/health', { method: 'GET' }),
    getLogs: (days = 7) => requestJson<{ logs: InspectionLogRow[] }>(`/api/mlops/logs?days=${days}`, { method: 'GET' }),
    runInspect: (input: { model_id?: string; days?: number } = {}) =>
      requestJson<InspectResponse>('/api/mlops/inspect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    getOnlineSources: () =>
      requestJson<{ sources: Record<string, OnlineSourceBinding> }>('/api/mlops/online-sources', { method: 'GET' }),
    getSettings: () => requestJson<MlopsSettings>('/api/mlops/settings', { method: 'GET' }),
    updateSettings: (input: MlopsSettings) =>
      requestJson<MlopsSettings>('/api/mlops/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    bindOnlineSource: (modelId: string, input: OnlineSourceBinding) =>
      requestJson<{ model_id: string; source: OnlineSourceBinding }>(
        `/api/mlops/online-sources/${encodeURIComponent(modelId)}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(input),
        },
      ),
    unbindOnlineSource: (modelId: string) =>
      requestJson<void>(`/api/mlops/online-sources/${encodeURIComponent(modelId)}`, { method: 'DELETE' }),
    triggerRetrain: (modelVersion: string, force = false) =>
      requestJson<RetrainResponse>(
        `/api/mlops/retrain?model_version=${encodeURIComponent(modelVersion)}&force=${String(force)}`,
        { method: 'POST' },
      ),
  }
}

export const mlopsApi = createMlopsApi({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
