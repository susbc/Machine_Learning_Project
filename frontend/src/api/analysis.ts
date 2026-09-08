import { createRequestJson } from './client'

export type RiskLevel = 'info' | 'low' | 'medium' | 'high' | 'critical'

export interface AnalysisEvidence {
  name: string
  value: unknown
  detail?: string
  source?: string
}

export interface AnalysisRisk {
  title: string
  severity: RiskLevel
  reason?: string
  evidence?: string[]
  recommendation?: string
}

export interface AnalysisRecommendation {
  title: string
  action: string
  priority: RiskLevel
  reason?: string
  expected_impact?: string
}

export interface AnalysisResult {
  analysis_type: string
  summary: string
  risk_level: RiskLevel
  confidence: number
  evidence: AnalysisEvidence[]
  risks: AnalysisRisk[]
  recommendations: AnalysisRecommendation[]
  next_actions: string[]
  raw_llm_response?: string
  metadata?: Record<string, unknown>
  created_at?: string
}

export interface DatasetAnalysisInput {
  file: File
  targetCol?: string
}

export interface TrainingAnalysisInput {
  metrics: Record<string, unknown>
  params?: Record<string, unknown>
  feature_importance?: Array<[string, number]>
}

export interface PredictionAnalysisInput {
  predictions: number[]
  threshold?: number
  reference_predictions?: number[]
}

export interface CompositeAnalysisInput {
  title?: string
  results: AnalysisResult[]
  llm_sections?: Record<string, string>
  metadata?: Record<string, unknown>
  save_report?: boolean
}

export interface CompositeAnalysisResponse {
  result: AnalysisResult
  report_path: string | null
}

export interface AnalysisReportRecord {
  analysis_id: string
  created_at?: string
  analysis_type: string
  risk_level: RiskLevel
  confidence?: number
  summary?: string
  report_path?: string
  metadata?: Record<string, unknown>
}

interface AnalysisApiOptions {
  baseUrl: string
  fetcher?: typeof fetch
}

export function buildDatasetAnalysisForm(input: DatasetAnalysisInput): FormData {
  const form = new FormData()
  form.append('file', input.file)
  if (input.targetCol?.trim()) {
    form.append('target_col', input.targetCol.trim())
  }
  return form
}

export function createAnalysisApi(options: AnalysisApiOptions) {
  const requestJson = createRequestJson({ ...options, label: 'Analysis' })

  return {
    analyzeDataset: (input: DatasetAnalysisInput) =>
      requestJson<AnalysisResult>('/api/analysis/dataset', {
        method: 'POST',
        body: buildDatasetAnalysisForm(input),
      }),
    analyzeTraining: (input: TrainingAnalysisInput) =>
      requestJson<AnalysisResult>('/api/analysis/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    analyzePrediction: (input: PredictionAnalysisInput) =>
      requestJson<AnalysisResult>('/api/analysis/prediction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    createCompositeReport: (input: CompositeAnalysisInput) =>
      requestJson<CompositeAnalysisResponse>('/api/analysis/composite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    listReports: () =>
      requestJson<{ reports: AnalysisReportRecord[] }>('/api/analysis/reports', {
        method: 'GET',
      }),
  }
}

export const analysisApi = createAnalysisApi({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
