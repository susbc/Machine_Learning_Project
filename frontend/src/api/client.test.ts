import { describe, expect, it, vi } from 'vitest'

import {
  authorizedFetch,
  createApiClient,
  createRequestJson,
  formatApiErrorDetail,
  normalizeApiBaseUrl,
  setApiKey,
  streamSseJson,
} from './client'

describe('api client', () => {
  it('normalizes API base URLs by removing trailing slashes', () => {
    expect(normalizeApiBaseUrl('http://localhost:8000///')).toBe('http://localhost:8000')
    expect(normalizeApiBaseUrl('')).toBe('')
  })

  it('fetches FastAPI health status from the configured base URL', async () => {
    const fetcher = vi.fn(async () => {
      return new Response(JSON.stringify({ status: 'healthy', version: '3.0.0' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    })
    const client = createApiClient({ baseUrl: 'http://localhost:8000/', fetcher })

    const health = await client.getHealth()

    expect(fetcher).toHaveBeenCalledWith('http://localhost:8000/health', expect.objectContaining({ method: 'GET' }))
    expect(health.status).toBe('healthy')
    expect(health.version).toBe('3.0.0')
  })

  it('uses relative URLs when the base URL is empty', async () => {
    const fetcher = vi.fn(async () => {
      return new Response(JSON.stringify({ status: 'healthy' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    })
    const client = createApiClient({ baseUrl: '', fetcher })

    await client.getHealth()

    expect(fetcher).toHaveBeenCalledWith('/health', expect.objectContaining({ method: 'GET' }))
  })

  it('attaches the configured API key to outgoing requests', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetcher)
    setApiKey('topsecret')

    await authorizedFetch('/api/llm/providers', { method: 'GET' })

    const headers = new Headers(fetcher.mock.calls[0][1]?.headers)
    expect(headers.get('X-API-Key')).toBe('topsecret')
    setApiKey('')
  })

  it('omits the API key header when no key is configured', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetcher)
    setApiKey('')

    await authorizedFetch('/api/llm/providers', { method: 'GET', headers: { 'Content-Type': 'application/json' } })

    const headers = new Headers(fetcher.mock.calls[0][1]?.headers)
    expect(headers.get('X-API-Key')).toBeNull()
    expect(headers.get('Content-Type')).toBe('application/json')
  })

  it('createRequestJson prefixes errors with the module label', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => new Response(JSON.stringify({ detail: 'boom' }), { status: 500 }))
    const requestJson = createRequestJson({ baseUrl: '', fetcher, label: 'LLM' })

    await expect(requestJson('/api/llm/providers', { method: 'GET' })).rejects.toThrow('LLM API request failed: boom')
  })

  it('createRequestJson returns undefined for 204 responses', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => new Response(null, { status: 204 }))
    const requestJson = createRequestJson({ baseUrl: 'http://x.test/', fetcher, label: 'Database' })

    const result = await requestJson<string | undefined>('/api/database/connections/abc', { method: 'DELETE' })

    expect(result).toBeUndefined()
    expect(fetcher).toHaveBeenCalledWith('http://x.test/api/database/connections/abc', expect.anything())
  })

  it('streams SSE through authenticated fetch and parses multiple JSON events', async () => {
    const fetcher = vi.fn<typeof fetch>(async () =>
      new Response('data: {"status":"RUNNING"}\r\n\r\ndata: {"status":"DONE"}\r\n\r\n', {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      }),
    )
    vi.stubGlobal('fetch', fetcher)
    setApiKey('stream-secret')
    const events: Array<{ status: string }> = []

    await streamSseJson<{ status: string }>('/api/stream', {
      onEvent: (event) => {
        events.push(event)
      },
    })

    const headers = new Headers(fetcher.mock.calls[0][1]?.headers)
    expect(headers.get('X-API-Key')).toBe('stream-secret')
    expect(headers.get('Accept')).toBe('text/event-stream')
    expect(events).toEqual([{ status: 'RUNNING' }, { status: 'DONE' }])
    setApiKey('')
  })

  it('formats FastAPI validation detail arrays into readable messages', () => {
    expect(
      formatApiErrorDetail([
        {
          loc: ['body', 'config', 'base_url'],
          msg: 'base_url 必须是 http/https URL',
          type: 'value_error',
        },
        {
          loc: ['body', 'name'],
          msg: 'Field required',
        },
      ]),
    ).toBe('config.base_url: base_url 必须是 http/https URL；name: Field required')
  })
})
