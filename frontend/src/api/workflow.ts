import { authorizedFetch, createRequestJson, normalizeApiBaseUrl, streamSseJson } from './client'

export interface WorkflowRetrieveInput {
  user_query: string
  target_col: string
}

export interface WorkflowRetrieveResponse {
  status: string
  task_id?: string
  sql_executed?: string
  thought_process?: string
  eda_summary?: Record<string, unknown>
  data_shape?: number[]
  msg?: string
}

export interface WorkflowFeatureInput {
  task_id: string
  eda_report_text?: string
}

export interface WorkflowFeatureAction {
  action?: string
  column?: string
  reason?: string
  params?: Record<string, unknown>
  [key: string]: unknown
}

export interface WorkflowFeatureResponse {
  status: string
  task_id: string
  strategy: { actions?: WorkflowFeatureAction[]; [key: string]: unknown }
  actions_count: number
}

export interface WorkflowTrainEvent {
  status: 'RUNNING' | 'DONE' | 'ERROR' | 'HEARTBEAT' | string
  trial?: number
  value?: number | null
  best_value?: number | null
  params?: Record<string, unknown>
  msg?: string
}

export interface WorkflowTaskStatus {
  task_id: string
  status: string
  files?: string[]
  step1?: string
  step2?: string
  step3?: string
  best_auc?: number
  feature_columns?: string[]
}

export interface WorkflowImportance {
  feature: string
  importance: number
}

export interface WorkflowModelRecord {
  task_id: string
  model_path?: string
}

export interface WorkflowPredictInput {
  model_version: string
  features: Record<string, unknown>
}

export interface WorkflowPredictResponse {
  status: string
  model_version?: string
  prediction: number
  probability: number
  input_features?: Record<string, unknown>
}

interface WorkflowApiOptions {
  baseUrl: string
  fetcher?: typeof fetch
}

export function parseSseEvent(raw: string): WorkflowTrainEvent | null {
  const dataLine = raw
    .replaceAll('\\n', '\n')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line.startsWith('data:'))
  if (!dataLine) return null
  return JSON.parse(dataLine.slice(5).trim()) as WorkflowTrainEvent
}

export function createWorkflowApi(options: WorkflowApiOptions) {
  const baseUrl = normalizeApiBaseUrl(options.baseUrl)
  const fetcher = options.fetcher ?? authorizedFetch
  const requestJson = createRequestJson({ ...options, label: 'Workflow' })

  function trainStreamUrl(taskId: string, nTrials = 30, cvFolds = 5): string {
    return `${baseUrl}/api/workflow/step3_train_stream?task_id=${encodeURIComponent(taskId)}&n_trials=${nTrials}&cv_folds=${cvFolds}`
  }

  return {
    retrieveData: (input: WorkflowRetrieveInput) =>
      requestJson<WorkflowRetrieveResponse>('/api/workflow/step1_retrieve_data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    generateFeatures: (input: WorkflowFeatureInput) =>
      requestJson<WorkflowFeatureResponse>('/api/workflow/step2_generate_features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ eda_report_text: '', ...input }),
    }),
    trainStreamUrl,
    streamTraining: (
      taskId: string,
      nTrials: number,
      cvFolds: number,
      onEvent: (event: WorkflowTrainEvent) => void | Promise<void>,
      signal?: AbortSignal,
    ) => streamSseJson<WorkflowTrainEvent>(trainStreamUrl(taskId, nTrials, cvFolds), { fetcher, signal, onEvent }),
    getTaskStatus: (taskId: string) =>
      requestJson<WorkflowTaskStatus>(`/api/workflow/task/${encodeURIComponent(taskId)}/status`, { method: 'GET' }),
    getImportance: (taskId: string) =>
      requestJson<WorkflowImportance[]>(`/api/workflow/task/${encodeURIComponent(taskId)}/importance`, { method: 'GET' }),
    listModels: () => requestJson<{ models: WorkflowModelRecord[] }>('/api/workflow/models', { method: 'GET' }),
    predict: (input: WorkflowPredictInput) =>
      requestJson<WorkflowPredictResponse>('/api/workflow/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
  }
}

export const workflowApi = createWorkflowApi({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
