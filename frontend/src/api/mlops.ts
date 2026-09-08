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
  AUC?: number
  '预测延迟 (ms)'?: number | null
  调用次数?: number | null
  健康度?: string
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
