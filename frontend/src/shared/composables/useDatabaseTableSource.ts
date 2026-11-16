import { computed, ref, watch } from 'vue'

import { databaseApi, type DatabaseConnection } from '@/api/database'

export interface TableImportResult {
  file?: File
  error?: string
}

/**
 * 数据库表导入共享逻辑：默认激活连接、仅启用连接可选、按连接加载表、
 * 取行后生成 CSV 文件供既有上传链路复用。
 */
export function useDatabaseTableSource() {
  const connections = ref<DatabaseConnection[]>([])
  const connectionId = ref('')
  const tableName = ref('')
  const tableOptions = ref<string[]>([])
  const importLimit = ref(1000)
  const loadingConnections = ref(false)
  const loadingTables = ref(false)

  const canImportTable = computed(() => Boolean(connectionId.value && tableName.value))

  async function loadConnections(): Promise<string | null> {
    if (connections.value.length) return null
    loadingConnections.value = true
    try {
      const payload = await databaseApi.listConnections()
      connections.value = payload.connections.filter((connection) => connection.is_active)
      if (!connectionId.value) {
        connectionId.value = payload.active_connection?.id ?? connections.value[0]?.id ?? ''
      }
      return null
    } catch (error) {
      return error instanceof Error ? error.message : '数据库连接列表加载失败'
    } finally {
      loadingConnections.value = false
    }
  }

  async function loadTableOptions(): Promise<string | null> {
    tableOptions.value = []
    tableName.value = ''
    if (!connectionId.value) return null
    loadingTables.value = true
    try {
      const schema = await databaseApi.getSchema(connectionId.value, 1)
      tableOptions.value = schema.tables.map((table) => table.name)
      return null
    } catch (error) {
      return error instanceof Error ? error.message : '数据表列表加载失败'
    } finally {
      loadingTables.value = false
    }
  }

  function rowsToCsv(columns: string[], rows: Array<Record<string, unknown>>): string {
    const escape = (value: unknown) => {
      if (value === null || value === undefined) return ''
      const text = String(value)
      return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
    }
    return [
      columns.map(escape).join(','),
      ...rows.map((row) => columns.map((column) => escape(row[column])).join(',')),
    ].join('\r\n')
  }

  async function importTableAsFile(): Promise<TableImportResult> {
    if (!canImportTable.value) return {}
    try {
      const payload = await databaseApi.getTableRows(connectionId.value, tableName.value, importLimit.value)
      if (!payload.rows.length) return { error: '数据表没有可导入的行' }
      return {
        file: new File([rowsToCsv(payload.columns, payload.rows)], `${tableName.value}.csv`, { type: 'text/csv' }),
      }
    } catch (error) {
      return { error: error instanceof Error ? error.message : '数据库表数据导入失败' }
    }
  }

  watch(connectionId, () => {
    void loadTableOptions()
  })

  return {
    connections,
    connectionId,
    tableName,
    tableOptions,
    importLimit,
    loadingConnections,
    loadingTables,
    canImportTable,
    loadConnections,
    loadTableOptions,
    importTableAsFile,
  }
}
