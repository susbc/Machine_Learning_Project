<template>
  <section class="workspace-theme database-workspace">
    <div class="section-heading database-heading">
      <div>
        <p class="eyebrow">Phase 5</p>
        <h2>数据库连接</h2>
        <p>管理数据库连接、浏览 Schema、查看样本数据，并通过只读 SQL 预览数据。</p>
      </div>
      <div class="heading-actions">
        <el-button :icon="Refresh" :loading="loading" @click="loadConnections">刷新</el-button>
        <el-button type="primary" :icon="Plus" @click="startCreate">新增连接</el-button>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" />

    <section class="db-status-strip">
      <div class="status-item">
        <span>连接总数</span>
        <strong>{{ statistics?.total_connections ?? 0 }}</strong>
      </div>
      <div class="status-item">
        <span>当前激活</span>
        <strong>{{ activeConnection?.name ?? '-' }}</strong>
      </div>
      <div class="status-item">
        <span>数据库类型</span>
        <strong>{{ dbTypesText }}</strong>
      </div>
      <div class="status-item">
        <span>当前表数</span>
        <strong>{{ schemaTables.length }}</strong>
      </div>
    </section>

    <div class="database-grid">
      <section class="panel connection-panel">
        <div class="panel-title">
          <h3>连接列表</h3>
          <span>{{ connections.length }} 项</span>
        </div>

        <el-empty v-if="!loading && connections.length === 0" description="暂无数据库连接" />

        <div v-else class="connection-list">
          <article
            v-for="connection in connections"
            :key="connection.id"
            class="connection-row"
            :class="{ selected: connection.id === selectedConnectionId }"
            @click="selectConnection(connection)"
          >
            <div class="connection-main">
              <div class="connection-identity">
                <strong>{{ connection.name }}</strong>
                <span>{{ connection.config.database }}</span>
              </div>
              <el-tag v-if="connection.is_active" class="connection-badge" type="success" effect="light">激活</el-tag>
              <el-tag v-else class="connection-badge" effect="light">{{ dbTypeLabel(connection.db_type) }}</el-tag>
            </div>
            <div class="connection-meta">
              <span>{{ connection.config.host || 'local file' }}</span>
              <span>{{ connection.config.user || 'no user' }}</span>
              <span :class="{ 'credential-warning': connection.config.credential_status === 'unreadable' }">
                {{ connection.config.password_masked || '无密码' }}
              </span>
            </div>
            <div class="connection-actions" @click.stop>
              <el-tooltip content="测试连接" placement="top">
                <el-button :icon="Connection" circle plain :loading="testingId === connection.id" @click="testDbConnection(connection)" />
              </el-tooltip>
              <el-tooltip content="加载 Schema" placement="top">
                <el-button :icon="Tickets" circle plain @click="loadSchema(connection)" />
              </el-tooltip>
              <el-tooltip content="删除" placement="top">
                <el-button :icon="Delete" circle plain type="danger" @click="removeConnection(connection)" />
              </el-tooltip>
            </div>
          </article>
        </div>
      </section>

      <section class="panel editor-panel">
        <div class="panel-title">
          <h3>{{ editingConnection ? '编辑连接' : '新增连接' }}</h3>
          <span>{{ editingConnection ? editingConnection.id : 'encrypted credential' }}</span>
        </div>

        <el-form label-position="top" class="connection-form" @submit.prevent>
          <div class="form-grid">
            <el-form-item label="名称">
              <el-input v-model="form.name" placeholder="例如：本地 SQLite" />
            </el-form-item>
            <el-form-item label="类型">
              <el-select v-model="form.dbType" @change="applyDbDefaults">
                <el-option v-for="option in databaseTypeOptions" :key="option.value" :label="option.label" :value="option.value" />
              </el-select>
            </el-form-item>
          </div>

          <el-form-item :label="form.dbType === 'sqlite' ? '数据库文件路径' : '数据库名'">
            <el-input v-model="form.database" placeholder="SQLite 文件路径或数据库名" />
          </el-form-item>

          <div v-if="form.dbType !== 'sqlite'" class="form-grid">
            <el-form-item label="Host">
              <el-input v-model="form.host" placeholder="localhost" />
            </el-form-item>
            <el-form-item label="Port">
              <el-input-number v-model="form.port" :min="1" :max="65535" controls-position="right" />
            </el-form-item>
            <el-form-item label="User">
              <el-input v-model="form.user" />
            </el-form-item>
            <el-form-item :label="editingConnection ? 'Password（留空则不覆盖）' : 'Password'">
              <el-input v-model="form.password" type="password" show-password autocomplete="new-password" />
              <p
                v-if="editingConnection?.config.password_masked"
                class="field-hint"
                :class="{ 'credential-warning': editingConnection.config.credential_status === 'unreadable' }"
              >
                当前密码：{{ editingConnection.config.password_masked }}
              </p>
            </el-form-item>
          </div>

          <el-form-item label="描述">
            <el-input v-model="form.description" type="textarea" :rows="3" />
          </el-form-item>

          <div class="form-footer">
            <el-switch v-model="form.isActive" active-text="设为激活" inactive-text="普通连接" />
            <div class="form-actions">
              <el-button @click="resetForm">清空</el-button>
              <el-button :icon="Connection" :loading="testingForm" :disabled="saving" data-testid="test-current-connection" @click="testCurrentConfig">测试连接</el-button>
              <el-button type="primary" :loading="saving" @click="saveConnection">{{ editingConnection ? '保存修改' : '创建连接' }}</el-button>
            </div>
          </div>
        </el-form>

        <el-alert
          v-if="testResult"
          class="test-result"
          :type="testResult.ok ? 'success' : 'warning'"
          :title="testResult.message"
          show-icon
          :closable="false"
        />
      </section>
    </div>

    <section class="panel schema-panel">
      <div class="panel-title">
        <h3>Schema 浏览</h3>
        <span>{{ selectedConnection?.name ?? '未选择连接' }}</span>
      </div>
      <el-table :data="schemaTables" height="300">
        <el-table-column prop="name" label="表名" width="220" />
        <el-table-column label="字段">
          <template #default="{ row }">
            <el-tag v-for="column in row.columns" :key="column.name" class="column-tag" effect="plain">
              {{ column.name }} · {{ column.type }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="样本" width="120">
          <template #default="{ row }">{{ row.sample?.row_count ?? 0 }} 行</template>
        </el-table-column>
      </el-table>
    </section>

    <section class="panel preview-panel">
      <div class="panel-title">
        <h3>只读 SQL 预览</h3>
        <div class="preview-actions">
          <el-input-number v-model="queryLimit" :min="1" :max="1000" controls-position="right" />
          <el-button type="primary" :loading="previewing" @click="runPreview">执行预览</el-button>
        </div>
      </div>
      <el-input v-model="querySql" type="textarea" :rows="4" placeholder="SELECT * FROM table_name" />
      <el-table :data="previewRows" height="260" class="preview-table">
        <el-table-column v-for="column in previewColumns" :key="column" :prop="column" :label="column" min-width="140" />
      </el-table>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { Connection, Delete, Plus, Refresh, Tickets } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus/es/components/message/index.mjs'
import { ElMessageBox } from 'element-plus/es/components/message-box/index.mjs'

import {
  databaseApi,
  databaseTypeOptions,
  type DatabaseConnection,
  type DatabaseStatistics,
  type DatabaseTableSchema,
  type DatabaseTestResponse,
  type DatabaseType,
} from '@/api/database'

interface ConnectionForm {
  name: string
  dbType: DatabaseType
  database: string
  host: string
  port: number
  user: string
  password: string
  description: string
  isActive: boolean
}

const connections = ref<DatabaseConnection[]>([])
const activeConnection = ref<DatabaseConnection | null>(null)
const statistics = ref<DatabaseStatistics | null>(null)
const selectedConnection = ref<DatabaseConnection | null>(null)
const selectedConnectionId = ref('')
const editingConnection = ref<DatabaseConnection | null>(null)
const schemaTables = ref<DatabaseTableSchema[]>([])
const previewColumns = ref<string[]>([])
const previewRows = ref<Array<Record<string, unknown>>>([])
const loading = ref(false)
const saving = ref(false)
const previewing = ref(false)
const testingId = ref('')
const testingForm = ref(false)
const errorMessage = ref('')
const testResult = ref<DatabaseTestResponse | null>(null)
const querySql = ref('SELECT * FROM customers')
const queryLimit = ref(50)

const form = reactive<ConnectionForm>({
  name: '',
  dbType: 'sqlite',
  database: '',
  host: 'localhost',
  port: 3306,
  user: '',
  password: '',
  description: '',
  isActive: false,
})

watch(form, () => { testResult.value = null })

const dbTypesText = computed(() => {
  const types = statistics.value?.db_types ?? []
  return types.length ? types.map(dbTypeLabel).join(' / ') : '-'
})

async function loadConnections() {
  loading.value = true
  errorMessage.value = ''
  try {
    const response = await databaseApi.listConnections()
    connections.value = response.connections
    activeConnection.value = response.active_connection
    statistics.value = response.statistics
    if (!editingConnection.value && response.connections.length > 0) {
      selectConnection(response.connections[0])
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '数据库连接加载失败'
  } finally {
    loading.value = false
  }
}

function startCreate() {
  editingConnection.value = null
  selectedConnection.value = null
  selectedConnectionId.value = ''
  schemaTables.value = []
  resetForm()
}

function selectConnection(connection: DatabaseConnection) {
  editingConnection.value = connection
  selectedConnection.value = connection
  selectedConnectionId.value = connection.id
  form.name = connection.name
  form.dbType = connection.db_type
  form.database = connection.config.database
  form.host = connection.config.host ?? 'localhost'
  form.port = connection.config.port ?? defaultPort(connection.db_type)
  form.user = connection.config.user ?? ''
  form.password = ''
  form.description = connection.description ?? ''
  form.isActive = Boolean(connection.is_active)
  testResult.value = null
}

function resetForm() {
  form.name = ''
  form.dbType = 'sqlite'
  form.database = ''
  form.host = 'localhost'
  form.port = 3306
  form.user = ''
  form.password = ''
  form.description = ''
  form.isActive = false
  testResult.value = null
}

function applyDbDefaults() {
  form.port = defaultPort(form.dbType)
  if (form.dbType === 'sqlite') {
    form.host = ''
    form.user = ''
    form.password = ''
  } else if (!form.host) {
    form.host = 'localhost'
  }
}

async function saveConnection() {
  if (!form.name.trim() || !form.database.trim()) {
    ElMessage.warning('请填写连接名称和数据库')
    return
  }

  saving.value = true
  errorMessage.value = ''
  try {
    const config = buildFormConfig()
    if (editingConnection.value) {
      const updated = await databaseApi.updateConnection(editingConnection.value.id, {
        name: form.name.trim(),
        description: form.description.trim(),
        is_active: form.isActive,
        config,
      })
      ElMessage.success('连接已更新')
      await loadConnections()
      selectConnection(updated)
    } else {
      const created = await databaseApi.createConnection({
        name: form.name.trim(),
        description: form.description.trim(),
        config,
        is_active: form.isActive,
      })
      ElMessage.success('连接已创建')
      await loadConnections()
      selectConnection(created)
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '数据库连接保存失败'
  } finally {
    saving.value = false
  }
}

function buildFormConfig() {
  return {
    db_type: form.dbType,
    database: form.database.trim(),
    host: form.dbType === 'sqlite' ? undefined : form.host.trim(),
    port: form.dbType === 'sqlite' ? undefined : form.port,
    user: form.dbType === 'sqlite' ? undefined : form.user.trim(),
    password: form.dbType === 'sqlite' ? undefined : form.password,
  }
}

async function testDbConnection(connection: DatabaseConnection) {
  testingId.value = connection.id
  errorMessage.value = ''
  try {
    testResult.value = await databaseApi.testConnection(connection.id)
    if (testResult.value.ok) {
      ElMessage.success(testResult.value.message)
    } else {
      ElMessage.warning(testResult.value.message)
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '连接测试失败'
  } finally {
    testingId.value = ''
  }
}

// 测试编辑面板里当前填写的配置（无需先保存）；
// 编辑现有连接且密码留空时，后端复用已存储的凭证
async function testCurrentConfig() {
  if (form.dbType !== 'sqlite' && !form.host.trim()) {
    ElMessage.warning('请先填写 Host')
    return
  }
  if (!form.database.trim()) {
    ElMessage.warning('请先填写数据库')
    return
  }
  testingForm.value = true
  testResult.value = null
  errorMessage.value = ''
  try {
    testResult.value = await databaseApi.testConfig({
      config: buildFormConfig(),
      connection_id: editingConnection.value?.id,
    })
    if (testResult.value.ok) {
      ElMessage.success(testResult.value.message)
    } else {
      ElMessage.warning(testResult.value.message)
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '连接测试失败'
  } finally {
    testingForm.value = false
  }
}

async function loadSchema(connection = selectedConnection.value) {
  if (!connection) {
    ElMessage.warning('请先选择数据库连接')
    return
  }
  errorMessage.value = ''
  try {
    const response = await databaseApi.getSchema(connection.id)
    selectedConnection.value = connection
    selectedConnectionId.value = connection.id
    schemaTables.value = response.tables
    if (response.tables[0]) {
      querySql.value = `SELECT * FROM ${response.tables[0].name}`
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'Schema 加载失败'
  }
}

async function runPreview() {
  if (!selectedConnection.value) {
    ElMessage.warning('请先选择数据库连接')
    return
  }
  previewing.value = true
  errorMessage.value = ''
  try {
    const response = await databaseApi.previewQuery({
      connection_id: selectedConnection.value.id,
      sql: querySql.value,
      limit: queryLimit.value,
    })
    previewColumns.value = response.columns
    previewRows.value = response.rows
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '查询预览失败'
  } finally {
    previewing.value = false
  }
}

async function removeConnection(connection: DatabaseConnection) {
  try {
    await ElMessageBox.confirm(`删除数据库连接「${connection.name}」？`, '确认删除', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
    await databaseApi.deleteConnection(connection.id)
    ElMessage.success('连接已删除')
    if (selectedConnectionId.value === connection.id) {
      startCreate()
    }
    await loadConnections()
  } catch (error) {
    if (error !== 'cancel') {
      errorMessage.value = error instanceof Error ? error.message : '删除连接失败'
    }
  }
}

function dbTypeLabel(type: string) {
  return databaseTypeOptions.find((option) => option.value === type)?.label ?? type
}

function defaultPort(type: DatabaseType) {
  return databaseTypeOptions.find((option) => option.value === type)?.defaultPort ?? 3306
}

onMounted(loadConnections)
</script>

<style scoped>
.database-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.database-heading,
.panel-title,
.connection-main,
.connection-actions,
.form-footer,
.form-actions,
.preview-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.database-heading,
.panel-title,
.connection-main,
.form-footer {
  justify-content: space-between;
}

.heading-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: flex-end;
}

.db-status-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.status-item,
.panel {
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
}

.status-item {
  min-width: 0;
  padding: 16px;
}

.status-item span,
.connection-meta,
.panel-title span,
.field-hint {
  color: #64748b;
  font-size: 12px;
}

.status-item strong {
  display: block;
  overflow: hidden;
  margin-top: 5px;
  color: #0f172a;
  font-size: 19px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.database-grid {
  display: grid;
  grid-template-columns: minmax(360px, 1fr) minmax(360px, 0.9fr);
  gap: 18px;
}

.panel {
  padding: 18px;
}

.panel-title {
  margin-bottom: 14px;
}

.panel-title h3 {
  margin: 0;
  color: #111827;
  font-size: 16px;
}

.connection-list {
  display: grid;
  gap: 10px;
}

.connection-row {
  position: relative;
  display: grid;
  gap: 10px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 14px;
  cursor: pointer;
}

.connection-badge {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.connection-row.selected {
  border-color: #2f6f73;
  background: #f3faf8;
}

.connection-identity { min-width: 0; overflow-wrap: anywhere; }
.connection-identity strong,
.connection-identity > span {
  display: block;
}

.connection-identity > span {
  color: #64748b;
  font-size: 13px;
}

.connection-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
}

.connection-actions {
  justify-content: flex-end;
}

.connection-form {
  display: grid;
  gap: 2px;
}

.form-footer, .form-actions { flex-wrap: wrap; }
.form-actions .el-button + .el-button { margin-left: 0; }

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.field-hint {
  margin: 6px 0 0;
}

.credential-warning {
  color: #b45309 !important;
  font-weight: 600;
}

.test-result,
.preview-table {
  margin-top: 14px;
}

.column-tag {
  margin: 3px 6px 3px 0;
}

@media (max-width: 1080px) {
  .database-grid,
  .db-status-strip,
  .form-grid {
    grid-template-columns: 1fr;
  }

  .database-heading,
  .form-footer,
  .panel-title {
    align-items: flex-start;
    flex-direction: column;
  }

  .heading-actions,
  .form-actions,
  .preview-actions {
    justify-content: flex-start;
  }
}
</style>
