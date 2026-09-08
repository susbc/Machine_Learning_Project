/** Canonical frontend mirror of the backend task-resolution contract. */
export type ProblemType = 'auto' | 'binary_classification' | 'multiclass_classification' | 'regression'
export type ResolvedProblemType = Exclude<ProblemType, 'auto'>
export type MetricDirection = 'minimize' | 'maximize'
export type CanonicalProblemSpecSource = 'auto_inferred' | 'user_override'
export type PersistedProblemSpecSource = CanonicalProblemSpecSource | 'legacy_compatibility'

export interface ProblemCapabilities {
  algorithms: string[]
  metrics: string[]
  default_metric: string
  metric_directions: Record<string, MetricDirection>
}

export interface ProblemSpec {
  schema_version: 1
  requested_type: ProblemType
  resolved_type: ResolvedProblemType
  source: PersistedProblemSpecSource
  reason: string
  explanation: string
  target: {
    column: string
    dtype: string
    valid_rows: number
    unique_count: number
    unique_ratio: number
    integer_like: boolean
  }
  capabilities: ProblemCapabilities
}

export interface ProblemCapabilitiesResponse {
  problem_types: Record<ResolvedProblemType, ProblemCapabilities>
}
