import { authorizedFetch, createRequestJson, normalizeApiBaseUrl, streamSseJson } from './client'
import type { ProblemSpec, ProblemType } from './problem'

export type AutoMLStatus = 'queued' | 'running' | 'cancelling' | 'cancelled' | 'completed' | 'failed'
export type AutoMLMetric = 'auto' | 'roc_auc' | 'accuracy' | 'f1' | 'rmse' | 'mae' | 'r2'
export type AutoMLModel =
  | 'lightgbm'
  | 'xgboost'
  | 'catboost'
  | 'random_forest'
  | 'logistic_regression'
  | 'linear_regression'

export interface AutoMLDecisionSummary {
  rows?: number
  columns?: number
  target_col?: string
  feature_count?: number
  numeric_features?: number
  categorical_features?: number
  missing_rate?: number
  candidate_models?: AutoMLModel[]
  recommendation_reason?: string
  recommendation_source?: string
  llm_status?: string
  business_context?: string
  [key: string]: unknown
}

export interface AutoMLFeatureSelectionInfo {
  enabled: boolean
  applied: boolean
  threshold?: string
  original_feature_count?: number
  selected_feature_count?: number
  reason?: string
}

export interface AutoMLSamplingInfo {
  enabled: boolean
  applied: boolean
  negative_ratio: number
  original_rows: number
  sampled_rows: number
  positive_rows: number
  negative_rows: number
  kept_negative_rows?: number
}

export interface AutoMLJob {
  job_id: string
  status: AutoMLStatus
  progress: number
  current_trial: number
  n_trials: number
  cv_folds: number
  optimization_metric: AutoMLMetric
  target_col: string
  decision_summary: AutoMLDecisionSummary
  problem_spec: ProblemSpec
  early_stopping_rounds?: number
  min_improvement?: number
  enable_feature_selection?: boolean
  enable_sampling?: boolean
  sampling_negative_ratio?: number
  error?: string | null
  artifact_path?: string | null
}

export interface AutoMLTrialEvent {
  event: 'trial' | 'status' | 'done' | 'error'
  status?: AutoMLStatus
  trial_number?: number
  value?: number
  best_value?: number
  params?: Record<string, unknown>
  model_type?: AutoMLModel
  problem_spec?: ProblemSpec
  progress?: number
  message?: string
  timestamp?: string
}

export interface AutoMLResult {
  job_id: string
  status: AutoMLStatus
  best_model: AutoMLModel
  best_params: Record<string, unknown>
  best_value: number
  metrics: Record<string, number>
  trial_history: AutoMLTrialEvent[]
  decision_summary: AutoMLDecisionSummary
  feature_columns: string[]
  problem_spec: ProblemSpec
  feature_importance?: Array<{ feature: string; importance: number }>
  early_stopped?: boolean
  feature_selection?: AutoMLFeatureSelectionInfo | null
  sampling?: AutoMLSamplingInfo | null
}

export interface AutoMLSaveResult {
  status: 'saved'
  job_id: string
  artifact_path: string
}

export interface CreateAutoMLJobInput {
  file: File
  targetCol?: string
  problemType?: ProblemType
  nTrials: number
  cvFolds: number
  candidateModels?: AutoMLModel[]
  optimizationMetric?: AutoMLMetric
  useLlm?: boolean
  businessContext?: string
  earlyStoppingRounds?: number
  minImprovement?: number
  enableFeatureSelection?: boolean
  enableSampling?: boolean
  samplingNegativeRatio?: number
}

interface AutoMLApiOptions {
  baseUrl: string
  fetcher?: typeof fetch
}

export interface AutoMLApi {
  createJob: (input: CreateAutoMLJobInput) => Promise<AutoMLJob>
  getJob: (jobId: string) => Promise<AutoMLJob>
  getResult: (jobId: string) => Promise<AutoMLResult>
  cancelJob: (jobId: string) => Promise<AutoMLJob>
  saveResult: (jobId: string) => Promise<AutoMLSaveResult>
  streamJob: (
    jobId: string,
    onEvent: (event: AutoMLTrialEvent) => void | Promise<void>,
    signal?: AbortSignal,
    cursor?: number,
  ) => Promise<void>
}

export function buildAutoMLJobForm(input: CreateAutoMLJobInput): FormData {
  const form = new FormData()
  form.append('file', input.file)
  if (input.targetCol?.trim()) {
    form.append('target_col', input.targetCol.trim())
  }
  if (input.problemType) {
    form.append('problem_type', input.problemType)
  }
  form.append('n_trials', String(input.nTrials))
  form.append('cv_folds', String(input.cvFolds))
  if (input.candidateModels?.length) {
    form.append('candidate_models', JSON.stringify(input.candidateModels))
  }
  form.append('optimization_metric', input.optimizationMetric ?? 'auto')
  form.append('use_llm', String(input.useLlm ?? true))
  if (input.businessContext?.trim()) {
    form.append('business_context', input.businessContext.trim())
  }
  form.append('early_stopping_rounds', String(input.earlyStoppingRounds ?? 0))
  form.append('min_improvement', String(input.minImprovement ?? 0.001))
  form.append('enable_feature_selection', String(input.enableFeatureSelection ?? false))
  form.append('enable_sampling', String(input.enableSampling ?? false))
  form.append('sampling_negative_ratio', String(input.samplingNegativeRatio ?? 1.0))
  return form
}

export function createAutoMLApi(options: AutoMLApiOptions): AutoMLApi {
  const baseUrl = normalizeApiBaseUrl(options.baseUrl)
  const fetcher = options.fetcher ?? authorizedFetch
  const requestJson = createRequestJson({ ...options, label: 'AutoML' })

  return {
    createJob: (input) =>
      requestJson<AutoMLJob>('/api/automl/jobs', {
        method: 'POST',
        body: buildAutoMLJobForm(input),
      }),
    getJob: (jobId) => requestJson<AutoMLJob>(`/api/automl/jobs/${jobId}`, { method: 'GET' }),
    getResult: (jobId) => requestJson<AutoMLResult>(`/api/automl/jobs/${jobId}/result`, { method: 'GET' }),
    cancelJob: (jobId) => requestJson<AutoMLJob>(`/api/automl/jobs/${jobId}/cancel`, { method: 'POST' }),
    saveResult: (jobId) => requestJson<AutoMLSaveResult>(`/api/automl/jobs/${jobId}/save`, { method: 'POST' }),
    streamJob: (jobId, onEvent, signal, cursor = 0) =>
      streamSseJson<AutoMLTrialEvent>(
        `${baseUrl}/api/automl/jobs/${encodeURIComponent(jobId)}/stream?cursor=${cursor}`,
        { fetcher, signal, onEvent },
      ),
  }
}

export const automlApi = createAutoMLApi({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
