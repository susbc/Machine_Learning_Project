import { createRequestJson } from './client'
import type { ProblemSpec, ResolvedProblemType } from './problem'

export type PredictionValue = string | number

export interface PredictionDiagnostics {
  unknown_input_keys: string[]
  missing_columns_imputed: string[]
}

export interface ModelMetadataError {
  code: string
  field: string
  message: string
}

export interface ModelRegistryRecord {
  model_id: string
  source: 'automl' | 'manual_training' | 'agent_workflow' | 'legacy' | string
  name: string
  artifact_path: string
  metrics: Record<string, unknown>
  optimization_metric: string | null
  best_metric: string | null
  problem_type: ResolvedProblemType | null
  problem_spec: ProblemSpec | null
  legacy: boolean
  metadata_error: ModelMetadataError | null
  updated_at: number
}

export interface ModelPredictionInput {
  features: Record<string, unknown>
}

export interface ModelPredictionResponse {
  model_id: string
  source: string
  input_features: Record<string, unknown>
  prediction: PredictionValue
  encoded_prediction: number | null
  probability: number | null
  diagnostics: PredictionDiagnostics
}

export interface ModelBatchPredictionInput {
  rows: Record<string, unknown>[]
}

export interface ModelBatchPredictionRow {
  index: number
  features: Record<string, unknown>
  prediction: PredictionValue
  encoded_prediction: number | null
  probability: number | null
  diagnostics: PredictionDiagnostics
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

export interface PredictionExportRow {
  index: number
  features: Record<string, unknown>
  prediction: PredictionValue
  encoded_prediction: number | null
  probability: number | null
  diagnostics: PredictionDiagnostics | null
}

export type PredictionExportTarget = 'file' | 'database'

export interface PredictionExportInput {
  model_id: string
  target: PredictionExportTarget
  rows: PredictionExportRow[]
  output_dir?: string
  connection_id?: string
  table_name?: string
}

export interface PredictionExportResponse {
  target: PredictionExportTarget
  exported: boolean
  status: 'ok' | 'permission_denied' | string
  warnings: string[]
  row_count: number
  path: string | null
  connection_id: string | null
  table_name: string | null
  table_created: boolean | null
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
    batchPredict: async (modelId: string, input: ModelBatchPredictionInput): Promise<ModelBatchPredictionResponse> => {
      let combined: ModelBatchPredictionResponse | undefined
      for (let offset = 0; offset < Math.max(input.rows.length, 1); offset += 1000) {
        const rows = input.rows.slice(offset, offset + 1000)
        const batch = await requestJson<ModelBatchPredictionResponse>(`/api/models/${encodeURIComponent(modelId)}/batch-predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rows }),
        })
        if (batch.rows.length !== rows.length || batch.summary.count !== rows.length) {
          throw new Error('批量预测返回行数不一致，请重试')
        }
        batch.rows = batch.rows.map((row, index) => ({ ...row, index: offset + index }))
        if (!combined) combined = batch
        else {
          const previousCount = combined.summary.count
          const count = previousCount + batch.summary.count
          for (const key of ['positive_rate', 'average_probability'] as const) {
            const previous = combined.summary[key]
            const current = batch.summary[key]
            combined.summary[key] = previous == null || current == null ? null
              : (previous * previousCount + current * batch.summary.count) / count
          }
          combined.summary.count = count
          combined.rows.push(...batch.rows)
        }
      }
      return combined!
    },
    exportPredictions: (input: PredictionExportInput) =>
      requestJson<PredictionExportResponse>('/api/prediction-exports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
  }
}

export const modelsApi = createModelsApi({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
