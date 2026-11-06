import { createRequestJson } from './client'

export type DatabaseType = 'sqlite' | 'mysql' | 'postgresql' | 'sqlserver'

export interface DatabaseConfig {
  db_type: DatabaseType
  host?: string
  port?: number
  database: string
  user?: string
  password?: string
  password_masked?: string
  has_password?: boolean
  credential_status?: 'ok' | 'unreadable' | 'not_set'
  extra_params?: Record<string, unknown>
}

export interface DatabaseConnection {
  id: string
  name: string
  db_type: DatabaseType
  description?: string
  tags?: string[]
  created_at?: string
  last_used?: string
  use_count?: number
  is_active?: boolean
  config: DatabaseConfig
}

export interface DatabaseStatistics {
  total_connections: number
  active_connection?: string | null
  db_types: DatabaseType[]
}

export interface DatabaseConnectionListResponse {
  connections: DatabaseConnection[]
  active_connection: DatabaseConnection | null
  statistics: DatabaseStatistics
}

export interface DatabaseConnectionPayload {
  name?: string
  config?: DatabaseConfig
  description?: string
  tags?: string[]
  is_active?: boolean
}

export interface DatabaseTestResponse {
  connection_id?: string
  ok: boolean
  message: string
}

export interface DatabaseTestConfigInput {
  config: DatabaseConfig
  connection_id?: string
}

export interface TableSample {
  columns?: string[]
  rows: Array<Record<string, unknown>>
  row_count?: number
}

export interface TableColumn {
  name: string
  type: string
  nullable: boolean
}

export interface DatabaseTableSchema {
  name: string
  columns: TableColumn[]
  sample?: TableSample
}

export interface DatabaseSchemaResponse {
  connection: DatabaseConnection
  tables: DatabaseTableSchema[]
}

export interface QueryPreviewInput {
  connection_id: string
  sql: string
  limit?: number
}

export interface QueryPreviewResponse {
  connection_id?: string
  columns: string[]
  rows: Array<Record<string, unknown>>
  row_count: number
  limited: boolean
  limit?: number
}

interface DatabaseApiOptions {
  baseUrl: string
  fetcher?: typeof fetch
}

export const databaseTypeOptions: Array<{ label: string; value: DatabaseType; defaultPort?: number }> = [
  { label: 'SQLite', value: 'sqlite' },
  { label: 'MySQL', value: 'mysql', defaultPort: 3306 },
  { label: 'PostgreSQL', value: 'postgresql', defaultPort: 5432 },
  { label: 'SQL Server', value: 'sqlserver', defaultPort: 1433 },
]

export function createDatabaseApi(options: DatabaseApiOptions) {
  const requestJson = createRequestJson({ ...options, label: 'Database' })

  return {
    listConnections: () => requestJson<DatabaseConnectionListResponse>('/api/database/connections', { method: 'GET' }),
    createConnection: (input: DatabaseConnectionPayload & { name: string; config: DatabaseConfig }) =>
      requestJson<DatabaseConnection>('/api/database/connections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    updateConnection: (connectionId: string, input: DatabaseConnectionPayload) =>
      requestJson<DatabaseConnection>(`/api/database/connections/${encodeURIComponent(connectionId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    deleteConnection: (connectionId: string) =>
      requestJson<void>(`/api/database/connections/${encodeURIComponent(connectionId)}`, { method: 'DELETE' }),
    testConnection: (connectionId: string) =>
      requestJson<DatabaseTestResponse>(`/api/database/connections/${encodeURIComponent(connectionId)}/test`, {
        method: 'POST',
      }),
    testConfig: (input: DatabaseTestConfigInput) =>
      requestJson<DatabaseTestResponse>('/api/database/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    getSchema: (connectionId: string, sampleLimit = 5) =>
      requestJson<DatabaseSchemaResponse>(
        `/api/database/connections/${encodeURIComponent(connectionId)}/schema?sample_limit=${sampleLimit}`,
        { method: 'GET' },
      ),
    previewQuery: (input: QueryPreviewInput) =>
      requestJson<QueryPreviewResponse>('/api/database/query-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...input, limit: input.limit ?? 100 }),
      }),
  }
}

export const databaseApi = createDatabaseApi({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
