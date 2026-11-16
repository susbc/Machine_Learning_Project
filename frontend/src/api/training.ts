import { createRequestJson } from './client'
import type { ProblemCapabilitiesResponse, ProblemSpec, ProblemType } from './problem'

export type TrainingStatus = 'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'cancelled'
export type TrainingAlgorithm =
  | 'lightgbm'
  | 'xgboost'
  | 'catboost'
  | 'random_forest'
  | 'logistic_regression'
  | 'linear_regression'

export interface TrainingDataset {
  dataset_id: string
  filename: string
  rows: number
  columns: number
  target_col: string
  problem_spec: ProblemSpec
  created_at: string
  quality?: DataQualitySummary
  preview: Record<string, unknown>[]
}

export interface DataQualitySummary {
  duplicate_rows: number
  missing_cells: number
  missing_rate: number
  infinite_values: number
  constant_columns: string[]
  high_missing_columns: string[]
}

export interface TrainingField {
  field_name: string
  field_type: 'key' | 'label' | 'num' | 'str'
  dtype: string
  missing_rate: number
  unique_count: number
  encoding?: string | null
  scaling?: string | null
}

export interface TrainingEda {
  dataset_id: string
  summary: Record<string, number | string>
  columns: TrainingField[]
  quality?: DataQualitySummary
  correlations: Record<string, Record<string, number>>
  outliers?: EdaOutlierRow[]
  significance?: EdaSignificanceRow[]
  high_correlation_pairs?: EdaCorrelationPair[]
}

export interface EdaOutlierRow {
  field: string
  count: number
  rate: number
  lower: number
  upper: number
}

export interface EdaSignificanceRow {
  field: string
  method: string
  statistic: number | null
  p_value: number | null
  target_association: number | null
}

export interface EdaCorrelationPair {
  field_a: string
  field_b: string
  correlation: number
}

export interface TrainingJob {
  job_id: string
  dataset_id: string
  algorithm: TrainingAlgorithm
  status: TrainingStatus
  progress: number
  error?: string | null
  artifact_path?: string | null
  problem_spec: ProblemSpec
}

export interface TrainingResult {
  job_id: string
  dataset_id: string
  algorithm: TrainingAlgorithm
  params: Record<string, unknown>
  metrics: Record<string, number>
  confusion_matrix: number[][] | null
  feature_importance: Array<{ feature: string; importance: number }>
  train_size: number
  test_size: number
  feature_columns: string[]
  problem_spec: ProblemSpec
  threshold_calibration?: {
    threshold: number
    source: 'training_calibration_split' | 'default'
    calibration_size: number
  } | null
}

export interface UploadTrainingDatasetInput {
  file: File
  targetCol?: string
  problemType?: ProblemType
}

export interface CreateTrainingJobInput {
  dataset_id: string
  algorithm: TrainingAlgorithm
  params: Record<string, unknown>
  test_size?: number
  random_state?: number
  problemType?: ProblemType
}

interface TrainingApiOptions {
  baseUrl: string
  fetcher?: typeof fetch
}

export function buildTrainingDatasetForm(input: UploadTrainingDatasetInput): FormData {
  const form = new FormData()
  form.append('file', input.file)
  if (input.targetCol?.trim()) {
    form.append('target_col', input.targetCol.trim())
  }
  if (input.problemType) {
    form.append('problem_type', input.problemType)
  }
  return form
}

export function createTrainingApi(options: TrainingApiOptions) {
  const requestJson = createRequestJson({ ...options, label: 'Training' })

  return {
    uploadDataset: (input: UploadTrainingDatasetInput) =>
      requestJson<TrainingDataset>('/api/training/datasets', {
        method: 'POST',
        body: buildTrainingDatasetForm(input),
      }),
    getFieldConfig: (datasetId: string) =>
      requestJson<{ dataset_id: string; fields: TrainingField[] }>(`/api/training/datasets/${datasetId}/field-config`, {
        method: 'GET',
      }),
    getCapabilities: () => requestJson<ProblemCapabilitiesResponse>('/api/training/capabilities', { method: 'GET' }),
    runEda: (datasetId: string) =>
      requestJson<TrainingEda>('/api/training/eda', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: datasetId }),
      }),
    createJob: (input: CreateTrainingJobInput) => {
      const { problemType, ...payload } = input
      return requestJson<TrainingJob>('/api/training/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...payload,
          ...(problemType ? { problem_type: problemType } : {}),
        }),
      })
    },
    getJob: (jobId: string) => requestJson<TrainingJob>(`/api/training/jobs/${jobId}`, { method: 'GET' }),
    getResult: (jobId: string) => requestJson<TrainingResult>(`/api/training/jobs/${jobId}/result`, { method: 'GET' }),
    cancelJob: (jobId: string) =>
      requestJson<TrainingJob>(`/api/training/jobs/${jobId}/cancel`, { method: 'POST' }),
    saveResult: (jobId: string) =>
      requestJson<{ status: 'saved'; job_id: string; artifact_path: string }>(`/api/training/jobs/${jobId}/save`, {
        method: 'POST',
      }),
  }
}

export const trainingApi = createTrainingApi({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
