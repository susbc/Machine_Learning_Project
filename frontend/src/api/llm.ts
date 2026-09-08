import { createRequestJson } from './client'

export type LlmProviderType = 'openai' | 'anthropic' | 'qwen' | 'zhipu' | 'baidu'

export interface LlmProviderConfig {
  api_key?: string
  api_key_masked?: string
  has_api_key?: boolean
  base_url?: string
  model?: string
  secret_key?: string
  secret_key_masked?: string
  has_secret_key?: boolean
}

export interface LlmProvider {
  id: string
  name: string
  type: LlmProviderType
  description?: string
  enabled: boolean
  is_active: boolean
  created_at?: string
  config: LlmProviderConfig
}

export interface LlmProviderStatistics {
  total_providers: number
  enabled_providers: number
  active_provider?: string | null
  provider_types: LlmProviderType[]
}

export interface LlmProviderListResponse {
  providers: LlmProvider[]
  active_provider: LlmProvider | null
  statistics: LlmProviderStatistics
}

export interface LlmProviderPayload {
  name: string
  type?: LlmProviderType
  description?: string
  enabled?: boolean
  config?: LlmProviderConfig
}

export interface LlmTestResponse {
  provider_id: string
  ok: boolean
  message: string
}

export interface LlmActivateResponse {
  active_provider: LlmProvider
  statistics?: LlmProviderStatistics
}

interface LlmApiOptions {
  baseUrl: string
  fetcher?: typeof fetch
}

export const providerTypeOptions: Array<{ label: string; value: LlmProviderType; defaultModel: string; defaultBaseUrl?: string }> = [
  { label: 'OpenAI / 兼容接口', value: 'openai', defaultModel: 'gpt-4o-mini', defaultBaseUrl: 'https://api.openai.com/v1' },
  { label: 'Anthropic Claude', value: 'anthropic', defaultModel: 'claude-3-5-sonnet-latest' },
  { label: '通义千问 Qwen', value: 'qwen', defaultModel: 'qwen-plus', defaultBaseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1' },
  { label: '智谱 GLM', value: 'zhipu', defaultModel: 'glm-4-plus', defaultBaseUrl: 'https://open.bigmodel.cn/api/paas/v4' },
  { label: '百度文心', value: 'baidu', defaultModel: 'ernie-4.0-turbo-8k' },
]

export function createLlmApi(options: LlmApiOptions) {
  const requestJson = createRequestJson({ ...options, label: 'LLM' })

  return {
    listProviders: () => requestJson<LlmProviderListResponse>('/api/llm/providers', { method: 'GET' }),
    createProvider: (input: LlmProviderPayload & { type: LlmProviderType }) =>
      requestJson<LlmProvider>('/api/llm/providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    updateProvider: (providerId: string, input: LlmProviderPayload) =>
      requestJson<LlmProvider>(`/api/llm/providers/${encodeURIComponent(providerId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      }),
    deleteProvider: (providerId: string) =>
      requestJson<void>(`/api/llm/providers/${encodeURIComponent(providerId)}`, { method: 'DELETE' }),
    testProvider: (providerId: string) =>
      requestJson<LlmTestResponse>(`/api/llm/providers/${encodeURIComponent(providerId)}/test`, { method: 'POST' }),
    activateProvider: (providerId: string) =>
      requestJson<LlmActivateResponse>(`/api/llm/providers/${encodeURIComponent(providerId)}/activate`, {
        method: 'POST',
      }),
  }
}

export const llmApi = createLlmApi({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
