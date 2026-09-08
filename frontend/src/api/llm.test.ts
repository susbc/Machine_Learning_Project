import { describe, expect, it, vi } from 'vitest'

import { createLlmApi } from './llm'

describe('llm api', () => {
  it('uses stable LLM provider management endpoints without plain key reads', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const body = init?.body ? JSON.parse(String(init.body)) : null

      if (url === '/api/llm/providers' && init?.method === 'GET') {
        return new Response(
          JSON.stringify({
            providers: [
              {
                id: 'p1',
                name: 'Qwen',
                type: 'openai',
                enabled: true,
                is_active: false,
                config: { model: 'qwen-max', api_key_masked: 'sk-...cret', has_api_key: true },
              },
            ],
            active_provider: null,
            statistics: { total_providers: 1, enabled_providers: 1, provider_types: ['openai'] },
          }),
          { status: 200 },
        )
      }

      if (url === '/api/llm/providers' && init?.method === 'POST') {
        expect(body.config.api_key).toBe('sk-secret')
        return new Response(
          JSON.stringify({
            id: 'p2',
            name: body.name,
            type: body.type,
            enabled: body.enabled,
            is_active: false,
            config: { model: body.config.model, api_key_masked: 'sk-...cret', has_api_key: true },
          }),
          { status: 201 },
        )
      }

      if (url === '/api/llm/providers/p2' && init?.method === 'PATCH') {
        expect(body.config.api_key).toBe('')
        return new Response(
          JSON.stringify({
            id: 'p2',
            name: body.name,
            type: 'openai',
            enabled: true,
            is_active: false,
            config: { model: body.config.model, api_key_masked: 'sk-...cret', has_api_key: true },
          }),
          { status: 200 },
        )
      }

      if (url === '/api/llm/providers/p2/test' && init?.method === 'POST') {
        return new Response(JSON.stringify({ provider_id: 'p2', ok: false, message: '连接测试失败' }), { status: 200 })
      }

      if (url === '/api/llm/providers/p2/activate' && init?.method === 'POST') {
        return new Response(JSON.stringify({ active_provider: { id: 'p2', name: 'Qwen', is_active: true } }), {
          status: 200,
        })
      }

      if (url === '/api/llm/providers/p2' && init?.method === 'DELETE') {
        return new Response(null, { status: 204 })
      }

      throw new Error(`unexpected request: ${url}`)
    })
    const api = createLlmApi({ baseUrl: '', fetcher })

    const list = await api.listProviders()
    expect(list.providers[0].config.api_key_masked).toBe('sk-...cret')
    expect(JSON.stringify(list)).not.toContain('sk-secret')

    const created = await api.createProvider({
      name: 'Qwen',
      type: 'openai',
      enabled: true,
      config: { api_key: 'sk-secret', base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', model: 'qwen-max' },
    })
    expect(created.id).toBe('p2')

    const updated = await api.updateProvider('p2', {
      name: 'Qwen tuned',
      config: { api_key: '', model: 'qwen-max-latest' },
    })
    expect(updated.config.model).toBe('qwen-max-latest')
    expect((await api.testProvider('p2')).ok).toBe(false)
    expect((await api.activateProvider('p2')).active_provider.id).toBe('p2')
    await expect(api.deleteProvider('p2')).resolves.toBeUndefined()
  })

  it('formats FastAPI validation errors from provider forms', async () => {
    const fetcher = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          detail: [{ loc: ['body', 'config', 'base_url'], msg: 'base_url 必须是 http/https URL' }],
        }),
        { status: 422, statusText: 'Unprocessable Entity' },
      )
    })
    const api = createLlmApi({ baseUrl: '', fetcher })

    await expect(
      api.createProvider({
        name: 'bad',
        type: 'openai',
        config: { api_key: 'secret', base_url: 'file:///tmp/key', model: 'gpt-4o-mini' },
      }),
    ).rejects.toThrow('config.base_url: base_url 必须是 http/https URL')
  })
})
