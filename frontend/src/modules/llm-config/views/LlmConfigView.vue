<template>
  <section class="workspace-theme llm-workspace">
    <div class="section-heading llm-heading">
      <div>
        <p class="eyebrow">Phase 5</p>
        <h2>大模型配置</h2>
        <p>管理 LLM Provider、连接测试、启用停用和当前激活模型；API Key 仅写入，不回显明文。</p>
      </div>
      <div class="heading-actions">
        <el-button :icon="Refresh" :loading="loading" @click="loadProviders">刷新</el-button>
        <el-button type="primary" :icon="Plus" @click="startCreate">新增 Provider</el-button>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" />

    <section class="llm-status-strip">
      <div class="status-item">
        <span>配置总数</span>
        <strong>{{ statistics?.total_providers ?? 0 }}</strong>
      </div>
      <div class="status-item">
        <span>已启用</span>
        <strong>{{ statistics?.enabled_providers ?? 0 }}</strong>
      </div>
      <div class="status-item">
        <span>当前激活</span>
        <strong>{{ activeProvider?.name ?? '-' }}</strong>
      </div>
      <div class="status-item">
        <span>Provider 类型</span>
        <strong>{{ providerTypesText }}</strong>
      </div>
    </section>

    <div class="llm-grid">
      <section class="panel provider-panel">
        <div class="panel-title">
          <h3>Provider 列表</h3>
          <span>{{ providers.length }} 项</span>
        </div>

        <el-empty v-if="!loading && providers.length === 0" description="暂无大模型配置" />

        <div v-else class="provider-list">
          <article
            v-for="provider in providers"
            :key="provider.id"
            class="provider-row"
            :class="{ selected: provider.id === selectedProviderId }"
            @click="selectProvider(provider)"
          >
            <div class="provider-main">
              <div class="provider-identity">
                <strong>{{ provider.name }}</strong>
                <span>{{ provider.config.model ?? provider.type }}</span>
              </div>
              <el-tag v-if="provider.is_active" class="provider-badge" type="success" effect="light">激活</el-tag>
              <el-tag v-else-if="provider.enabled" class="provider-badge" effect="light">启用</el-tag>
              <el-tag v-else class="provider-badge" type="info" effect="light">停用</el-tag>
            </div>
            <div class="provider-meta">
              <span>{{ providerTypeLabel(provider.type) }}</span>
              <span>{{ provider.config.api_key_masked || '未设置密钥' }}</span>
            </div>
            <div class="provider-actions" @click.stop>
              <el-tooltip content="连接测试" placement="top">
                <el-button :icon="Connection" circle plain :loading="testingId === provider.id" @click="testConnection(provider)" />
              </el-tooltip>
              <el-tooltip content="设为激活" placement="top">
                <el-button
                  :icon="Check"
                  circle
                  plain
                  type="success"
                  :disabled="provider.is_active || !provider.enabled"
                  @click="activate(provider)"
                />
              </el-tooltip>
              <el-tooltip content="删除" placement="top">
                <el-button :icon="Delete" circle plain type="danger" @click="removeProvider(provider)" />
              </el-tooltip>
            </div>
          </article>
        </div>
      </section>

      <section class="panel editor-panel">
        <div class="panel-title">
          <h3>{{ editingProvider ? '编辑 Provider' : '新增 Provider' }}</h3>
          <span>{{ editingProvider ? editingProvider.id : 'write-only key' }}</span>
        </div>

        <form class="provider-form" @submit.prevent="saveProvider">
          <div class="form-grid">
            <label class="field-row">
              <span>名称</span>
              <el-input data-testid="llm-provider-name-input" v-model="form.name" placeholder="例如：Qwen Max" />
            </label>
            <label class="field-row">
              <span>类型</span>
              <el-select data-testid="llm-provider-type-select" v-model="form.type" @change="applyTypeDefaults">
                <el-option v-for="option in providerTypeOptions" :key="option.value" :label="option.label" :value="option.value" />
              </el-select>
            </label>
          </div>

          <label class="field-row">
            <span>模型名称</span>
            <el-input data-testid="llm-provider-model-input" v-model="form.model" placeholder="例如：qwen-max / gpt-4o-mini" />
          </label>

          <label class="field-row">
            <span>Base URL</span>
            <el-input data-testid="llm-provider-base-url-input" v-model="form.baseUrl" placeholder="兼容 OpenAI 的服务地址，可留空" />
          </label>

          <label class="field-row">
            <span>{{ editingProvider ? 'API Key（留空则不覆盖）' : 'API Key' }}</span>
            <el-input
              data-testid="llm-provider-api-key-input"
              v-model="form.apiKey"
              type="password"
              show-password
              autocomplete="new-password"
              placeholder="仅保存到后端配置文件"
            />
            <p v-if="editingProvider?.config.api_key_masked" class="field-hint">当前密钥：{{ editingProvider.config.api_key_masked }}</p>
          </label>

          <label class="field-row">
            <span>描述</span>
            <el-input
              data-testid="llm-provider-description-input"
              v-model="form.description"
              type="textarea"
              :rows="3"
              placeholder="用途、额度、适用模块等"
            />
          </label>

          <div class="form-footer">
            <el-switch v-model="form.enabled" active-text="启用" inactive-text="停用" />
            <div class="form-actions">
              <el-button @click="resetForm">清空</el-button>
              <el-button type="primary" :loading="saving" @click="saveProvider">{{ editingProvider ? '保存修改' : '创建 Provider' }}</el-button>
            </div>
          </div>
        </form>

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
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { Check, Connection, Delete, Plus, Refresh } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus/es/components/message/index.mjs'
import { ElMessageBox } from 'element-plus/es/components/message-box/index.mjs'

import {
  llmApi,
  providerTypeOptions,
  type LlmProvider,
  type LlmProviderStatistics,
  type LlmProviderType,
  type LlmTestResponse,
} from '@/api/llm'

interface ProviderForm {
  name: string
  type: LlmProviderType
  model: string
  baseUrl: string
  apiKey: string
  description: string
  enabled: boolean
}

const providers = ref<LlmProvider[]>([])
const activeProvider = ref<LlmProvider | null>(null)
const statistics = ref<LlmProviderStatistics | null>(null)
const selectedProviderId = ref('')
const editingProvider = ref<LlmProvider | null>(null)
const loading = ref(false)
const saving = ref(false)
const testingId = ref('')
const errorMessage = ref('')
const testResult = ref<LlmTestResponse | null>(null)

const form = reactive<ProviderForm>({
  name: '',
  type: 'openai',
  model: 'gpt-4o-mini',
  baseUrl: 'https://api.openai.com/v1',
  apiKey: '',
  description: '',
  enabled: true,
})

const providerTypesText = computed(() => {
  const types = statistics.value?.provider_types ?? []
  return types.length ? types.map(providerTypeLabel).join(' / ') : '-'
})

async function loadProviders() {
  loading.value = true
  errorMessage.value = ''
  try {
    const response = await llmApi.listProviders()
    providers.value = response.providers
    activeProvider.value = response.active_provider
    statistics.value = response.statistics
    if (!editingProvider.value && response.providers.length > 0) {
      selectProvider(response.providers[0])
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '大模型配置加载失败'
  } finally {
    loading.value = false
  }
}

function startCreate() {
  editingProvider.value = null
  selectedProviderId.value = ''
  resetForm()
}

function selectProvider(provider: LlmProvider) {
  editingProvider.value = provider
  selectedProviderId.value = provider.id
  form.name = provider.name
  form.type = provider.type
  form.model = provider.config.model ?? defaultOption(provider.type).defaultModel
  form.baseUrl = provider.config.base_url ?? defaultOption(provider.type).defaultBaseUrl ?? ''
  form.apiKey = ''
  form.description = provider.description ?? ''
  form.enabled = provider.enabled
  testResult.value = null
}

function resetForm() {
  const defaults = defaultOption(form.type)
  form.name = ''
  form.model = defaults.defaultModel
  form.baseUrl = defaults.defaultBaseUrl ?? ''
  form.apiKey = ''
  form.description = ''
  form.enabled = true
  testResult.value = null
}

function applyTypeDefaults() {
  const defaults = defaultOption(form.type)
  form.model = defaults.defaultModel
  form.baseUrl = defaults.defaultBaseUrl ?? ''
}

async function saveProvider() {
  if (!form.name.trim()) {
    ElMessage.warning('请填写 Provider 名称')
    return
  }
  if (!editingProvider.value && !form.apiKey.trim()) {
    ElMessage.warning('新增 Provider 需要填写 API Key')
    return
  }

  saving.value = true
  errorMessage.value = ''
  try {
    const config = {
      api_key: form.apiKey,
      base_url: form.baseUrl || undefined,
      model: form.model || undefined,
    }
    if (editingProvider.value) {
      const updated = await llmApi.updateProvider(editingProvider.value.id, {
        name: form.name.trim(),
        type: form.type,
        description: form.description.trim(),
        enabled: form.enabled,
        config,
      })
      ElMessage.success('Provider 已更新')
      await loadProviders()
      selectProvider(updated)
    } else {
      const created = await llmApi.createProvider({
        name: form.name.trim(),
        type: form.type,
        description: form.description.trim(),
        enabled: form.enabled,
        config,
      })
      ElMessage.success('Provider 已创建')
      await loadProviders()
      selectProvider(created)
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'Provider 保存失败'
  } finally {
    saving.value = false
  }
}

async function testConnection(provider: LlmProvider) {
  testingId.value = provider.id
  errorMessage.value = ''
  try {
    testResult.value = await llmApi.testProvider(provider.id)
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

async function activate(provider: LlmProvider) {
  errorMessage.value = ''
  try {
    await llmApi.activateProvider(provider.id)
    ElMessage.success('已切换激活 Provider')
    await loadProviders()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '激活 Provider 失败'
  }
}

async function removeProvider(provider: LlmProvider) {
  try {
    await ElMessageBox.confirm(`删除 Provider「${provider.name}」？`, '确认删除', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
    await llmApi.deleteProvider(provider.id)
    ElMessage.success('Provider 已删除')
    if (selectedProviderId.value === provider.id) {
      startCreate()
    }
    await loadProviders()
  } catch (error) {
    if (error !== 'cancel') {
      errorMessage.value = error instanceof Error ? error.message : '删除 Provider 失败'
    }
  }
}

function defaultOption(type: LlmProviderType) {
  return providerTypeOptions.find((option) => option.value === type) ?? providerTypeOptions[0]
}

function providerTypeLabel(type: string) {
  return providerTypeOptions.find((option) => option.value === type)?.label ?? type
}

onMounted(loadProviders)
</script>

<style scoped>
.llm-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.llm-heading,
.panel-title,
.provider-main,
.provider-actions,
.form-footer,
.form-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.llm-heading,
.panel-title,
.provider-main,
.form-footer {
  justify-content: space-between;
}

.heading-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: flex-end;
}

.llm-status-strip {
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
.provider-meta,
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

.llm-grid {
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

.provider-list {
  display: grid;
  gap: 10px;
}

.provider-row {
  display: grid;
  gap: 10px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 14px;
  cursor: pointer;
}

.provider-row.selected {
  border-color: #2f6f73;
  background: #f3faf8;
}

.provider-identity {
  min-width: 0;
  flex: 1;
}

.provider-badge {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 44px;
}

.provider-identity > strong,
.provider-identity > span {
  display: block;
  overflow-wrap: anywhere;
}

.provider-identity > span {
  color: #64748b;
  font-size: 13px;
}

.provider-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
}

.provider-actions {
  justify-content: flex-end;
}

.provider-form {
  display: grid;
  gap: 14px;
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.field-row {
  display: grid;
  min-width: 0;
  gap: 6px;
}

.field-row > span {
  color: #475569;
  font-size: 13px;
  font-weight: 650;
}

.field-row :deep(.el-input),
.field-row :deep(.el-select),
.field-row :deep(.el-textarea) {
  width: 100%;
}

.field-hint {
  margin: 6px 0 0;
}

.test-result {
  margin-top: 14px;
}

@media (max-width: 1080px) {
  .llm-grid,
  .llm-status-strip,
  .form-grid {
    grid-template-columns: 1fr;
  }

  .llm-heading,
  .form-footer {
    align-items: flex-start;
    flex-direction: column;
  }

  .heading-actions,
  .form-actions {
    justify-content: flex-start;
  }
}
</style>
