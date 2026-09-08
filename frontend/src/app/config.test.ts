import { describe, expect, it } from 'vitest'

import { resolveAppConfig } from './config'

describe('app config', () => {
  it('uses relative API paths by default so Vite or deployment proxies can route requests', () => {
    const config = resolveAppConfig({})
    expect(config.apiBaseUrl).toBe('')
  })

  it('accepts Vite environment overrides', () => {
    const config = resolveAppConfig({ VITE_API_BASE_URL: 'http://127.0.0.1:9000/' })
    expect(config.apiBaseUrl).toBe('http://127.0.0.1:9000')
  })
})
