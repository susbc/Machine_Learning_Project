import { describe, expect, it, vi } from 'vitest'

import { createDatabaseApi } from './database'

describe('database api', () => {
  it('uses stable database endpoints for connections, schema and read-only preview', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const body = init?.body ? JSON.parse(String(init.body)) : null

      if (url === '/api/database/connections' && init?.method === 'GET') {
        return new Response(
          JSON.stringify({
            connections: [{ id: 'c1', name: 'SQLite', db_type: 'sqlite', config: { database: 'sample.db' } }],
            active_connection: null,
            statistics: { total_connections: 1, db_types: ['sqlite'] },
          }),
          { status: 200 },
        )
      }
      if (url === '/api/database/connections' && init?.method === 'POST') {
        expect(body.config.password).toBe('secret')
        return new Response(
          JSON.stringify({
            id: 'c2',
            name: body.name,
            db_type: body.config.db_type,
            config: { ...body.config, password: undefined, password_masked: 'sec...cret', has_password: true },
          }),
          { status: 201 },
        )
      }
      if (url === '/api/database/connections/c2' && init?.method === 'PATCH') {
        return new Response(JSON.stringify({ id: 'c2', name: body.name, db_type: 'sqlite', config: body.config }), {
          status: 200,
        })
      }
      if (url === '/api/database/connections/c2/test' && init?.method === 'POST') {
        return new Response(JSON.stringify({ connection_id: 'c2', ok: true, message: '连接测试通过' }), { status: 200 })
      }
      if (url === '/api/database/connections/c2/schema?sample_limit=5' && init?.method === 'GET') {
        return new Response(
          JSON.stringify({ tables: [{ name: 'customers', columns: [{ name: 'id', type: 'INTEGER' }], sample: { rows: [] } }] }),
          { status: 200 },
        )
      }
      if (url === '/api/database/query-preview' && init?.method === 'POST') {
        expect(body.limit).toBe(2)
        return new Response(JSON.stringify({ columns: ['id'], rows: [{ id: 1 }], row_count: 1, limited: true }), {
          status: 200,
        })
      }
      if (url === '/api/database/connections/c2' && init?.method === 'DELETE') {
        return new Response(null, { status: 204 })
      }
      throw new Error(`unexpected request: ${url}`)
    })
    const api = createDatabaseApi({ baseUrl: '', fetcher })

    expect((await api.listConnections()).connections[0].id).toBe('c1')
    expect(
      (
        await api.createConnection({
          name: 'SQLite',
          config: { db_type: 'sqlite', database: 'sample.db', password: 'secret' },
        })
      ).config.password_masked,
    ).toBe('sec...cret')
    expect((await api.updateConnection('c2', { name: 'SQLite tuned', config: { db_type: 'sqlite', database: 'sample.db' } })).id).toBe(
      'c2',
    )
    expect((await api.testConnection('c2')).ok).toBe(true)
    expect((await api.getSchema('c2')).tables[0].name).toBe('customers')
    expect((await api.previewQuery({ connection_id: 'c2', sql: 'SELECT * FROM customers', limit: 2 })).row_count).toBe(1)
    await expect(api.deleteConnection('c2')).resolves.toBeUndefined()
  })
})
