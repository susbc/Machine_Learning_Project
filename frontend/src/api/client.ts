export interface ApiClientOptions {
  baseUrl: string
  fetcher?: typeof fetch
}

export interface HealthStatus {
  status: string
  scheduler_running?: boolean
  version?: string
}

export interface ApiClient {
  getHealth: () => Promise<HealthStatus>
}

export function normalizeApiBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '')
}

const API_KEY_STORAGE_KEY = 'agentic_ml_studio_api_key'

/**
 * 读取配置的 API 密钥：优先使用构建期注入的 VITE_API_KEY，
 * 其次使用运行时通过 setApiKey() 写入 localStorage 的值。
 * 后端未启用 API_KEY 鉴权时返回空串，请求不携带密钥头。
 */
export function getApiKey(): string {
  const fromEnv = import.meta.env.VITE_API_KEY
  if (typeof fromEnv === 'string' && fromEnv) {
    return fromEnv
  }
  try {
    return localStorage.getItem(API_KEY_STORAGE_KEY) ?? ''
  } catch {
    return ''
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) {
      localStorage.setItem(API_KEY_STORAGE_KEY, key)
    } else {
      localStorage.removeItem(API_KEY_STORAGE_KEY)
    }
  } catch {
    // localStorage 不可用时静默降级为不携带密钥
  }
}

/** 与 fetch 同构的包装：后端启用 API_KEY 鉴权时自动附加 X-API-Key 头。 */
export const authorizedFetch: typeof fetch = (input, init) => {
  const key = getApiKey()
  if (!key) {
    return fetch(input, init)
  }
  const headers = new Headers(init?.headers)
  headers.set('X-API-Key', key)
  return fetch(input, { ...init, headers })
}

interface FastApiValidationError {
  loc?: Array<string | number>
  msg?: string
  type?: string
}

export function formatApiErrorDetail(detail: unknown): string {
  if (typeof detail === 'string') {
    return detail
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === 'object') {
          const error = item as FastApiValidationError
          const loc = (error.loc ?? []).filter((part) => part !== 'body' && part !== 'query').join('.')
          return loc ? `${loc}: ${error.msg ?? JSON.stringify(item)}` : (error.msg ?? JSON.stringify(item))
        }
        return String(item)
      })
      .join('；')
  }
  if (detail && typeof detail === 'object') {
    const record = detail as { message?: string; msg?: string }
    return record.message ?? record.msg ?? JSON.stringify(detail)
  }
  return String(detail)
}

export async function readApiErrorDetail(response: Response): Promise<string> {
  let detail = `${response.status} ${response.statusText}`
  try {
    const payload = (await response.json()) as { detail?: unknown; message?: unknown }
    detail = formatApiErrorDetail(payload.detail ?? payload.message ?? detail)
  } catch {
    // Keep HTTP status text.
  }
  return detail
}

export interface SseJsonStreamOptions<T> {
  fetcher?: typeof fetch
  signal?: AbortSignal
  onEvent: (event: T) => void | Promise<void>
}

/** Read a JSON SSE response through fetch so API-key headers remain available. */
export async function streamSseJson<T>(url: string, options: SseJsonStreamOptions<T>): Promise<void> {
  const fetcher = options.fetcher ?? authorizedFetch
  const response = await fetcher(url, {
    method: 'GET',
    headers: { Accept: 'text/event-stream' },
    signal: options.signal,
  })
  if (!response.ok) {
    throw new Error(`SSE request failed: ${await readApiErrorDetail(response)}`)
  }
  if (!response.body) {
    throw new Error('SSE response body is unavailable')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  async function consume(block: string): Promise<void> {
    const data = block
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n')
    if (data) {
      await options.onEvent(JSON.parse(data) as T)
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    buffer = (buffer + decoder.decode(value, { stream: !done })).replaceAll('\r\n', '\n')
    let boundary = buffer.indexOf('\n\n')
    while (boundary >= 0) {
      await consume(buffer.slice(0, boundary))
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf('\n\n')
    }
    if (done) break
  }
  if (buffer.trim()) {
    await consume(buffer)
  }
}

export interface RequestJsonOptions {
  baseUrl: string
  fetcher?: typeof fetch
  /** 模块标签，用于错误消息前缀（如 "LLM"、"Database"）。 */
  label: string
}

/**
 * 各 API 模块共享的 JSON 请求工厂：统一错误格式、204 处理和鉴权头注入。
 */
export function createRequestJson(options: RequestJsonOptions) {
  const baseUrl = normalizeApiBaseUrl(options.baseUrl)
  const fetcher = options.fetcher ?? authorizedFetch

  return async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetcher(`${baseUrl}${path}`, init)
    if (!response.ok) {
      throw new Error(`${options.label} API request failed: ${await readApiErrorDetail(response)}`)
    }
    if (response.status === 204) {
      return undefined as T
    }
    return response.json() as Promise<T>
  }
}

export function createApiClient(options: ApiClientOptions): ApiClient {
  const baseUrl = normalizeApiBaseUrl(options.baseUrl)
  const fetcher = options.fetcher ?? authorizedFetch

  async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetcher(`${baseUrl}${path}`, {
      ...init,
      method: init.method ?? 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(init.headers ?? {}),
      },
    })

    if (!response.ok) {
      throw new Error(`API request failed: ${await readApiErrorDetail(response)}`)
    }

    return response.json() as Promise<T>
  }

  return {
    getHealth: () => request<HealthStatus>('/health'),
  }
}

export const apiClient = createApiClient({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
})
