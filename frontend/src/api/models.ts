import { createRequestJson } from './client'
import type { ProblemSpec, ResolvedProblemType } from './problem'

export interface ModelRegistryRecord {
  model_id: string
  source: 'automl' | 'manual_training' | 'agent_workflow' | 'legacy' | string
  name: string
  artifact_path: string
  metrics: Record<string, unknown>
  problem_type: ResolvedProblemType | null
  problem_spec: ProblemSpec | null
  legacy: boolean
  metadata_error: unknown | null
  updated_at: number
}

export interface ModelPredictionInput {
  features: Record<string, unknown>
}

export interface ModelPredictionResponse {
  model_id: string
  source: string
  input_features: Record<string, unknown>
  prediction: number
  encoded_prediction: number | null
  probability: number | null
}

export interface ModelBatchPredictionInput {
  rows: Record<string, unknown>[]
}

export interface ModelBatchPredictionRow {
  index: number
  features: Record<string, unknown>
  prediction: number
  encoded_prediction: number | null
  probability: number | null
}

export interface ModelBatchPredictionResponse {
  model_id: string
  source: string
  summary: {
    count: number
    positive_rate: number | null
    average_probability: number | null
  }
  rows: ModelBatchPredictionRow[]
}

interface ModelsApiOptions {
  baseUrl: string
  fetcher?: typeof fetch
}

export function createModelsApi(options: ModelsApiOptions) {
  const requestJson = createRequestJson({ ...options, label: 'Models' })

  return {
    listModels: () => requestJson<{ models: ModelRegistryRecord[] }>('/api/models', { method: 'GET' }),
    getModel: (modelId: string) =>
      requestJson<ModelRegistryRecord>(`/api/models/${encodeURIComponent(modelId)}`, { method: 'GET' }),
    predict: (modelId: string, input: ModelPredictionInput) =>
      requestJson<ModelPredictionResponse>(`/api/models/${encodeURIComponent(modelId)}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    batchPredict: (modelId: string, input: ModelBatchPredictionInput) =>
      requestJson<ModelBatchPredictionResponse>(`/api/models/${encodeURIComponent(modelId)}/batch-predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
  }
}

export const modelsApi = createModelsApi({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
