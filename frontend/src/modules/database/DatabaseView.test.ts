import { mount, flushPromises } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import DatabaseView from './views/DatabaseView.vue'

const api = vi.hoisted(() => ({
  listConnections: vi.fn(async () => ({
    connections: [{ id: 'db1', name: 'Saved MySQL', db_type: 'mysql', is_active: true,
      config: { db_type: 'mysql', host: 'localhost', port: 3306, database: 'original', user: 'root', password_masked: '****' } }],
    active_connection: null, statistics: { total_connections: 1, db_types: ['mysql'] },
  })),
  testConfig: vi.fn(async () => ({ ok: true, message: '连接测试通过' })),
  updateConnection: vi.fn(), createConnection: vi.fn(),
}))
vi.mock('@/api/database', () => ({ databaseApi: api, databaseTypeOptions: [
  { label: 'MySQL', value: 'mysql', defaultPort: 3306 }, { label: 'SQLite', value: 'sqlite' },
] }))

describe('Database editor connection tests', () => {
  beforeEach(() => vi.clearAllMocks())
  it('tests unsaved fields with the stored credential reference without saving', async () => {
    const wrapper = mount(DatabaseView, { global: { plugins: [ElementPlus] } })
    await flushPromises()
    await wrapper.get('input[placeholder="SQLite 文件路径或数据库名"]').setValue('edited')
    await wrapper.get('[data-testid="test-current-connection"]').trigger('click')
    await flushPromises()
    expect(api.testConfig).toHaveBeenCalledWith(expect.objectContaining({
      connection_id: 'db1', config: expect.objectContaining({ database: 'edited', password: '' }),
    }))
    expect(api.updateConnection).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('连接测试通过')
    await wrapper.get('input[placeholder="SQLite 文件路径或数据库名"]').setValue('changed-again')
    expect(wrapper.find('.test-result').exists()).toBe(false)
    wrapper.unmount()
  })
})
