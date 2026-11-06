<template>
  <section class="workspace-theme analysis-workspace">
    <div class="section-heading analysis-heading">
      <div>
        <p class="eyebrow">Phase 2</p>
        <h2>AI 智能分析</h2>
        <p>汇总数据、训练、预测、MLOps 和 LLM 上下文，形成可追踪的模型诊断与综合报告。</p>
      </div>
      <div class="heading-actions">
        <el-button plain :loading="contextLoading" @click="loadAnalysisContext">刷新上下文</el-button>
        <el-button type="primary" plain :disabled="selectedResults.length === 0" :loading="savingComposite" @click="saveComposite">
          保存综合报告
        </el-button>
      </div>
    </div>

    <section class="context-strip" data-testid="analysis-context-strip">
      <div class="context-heading">
        <div>
          <h3>跨模块上下文</h3>
          <p>把已训练模型、MLOps 健康、LLM Provider 和历史报告接到同一诊断入口。</p>
        </div>
        <el-tag :type="contextRiskTag" effect="light">{{ contextRiskText }}</el-tag>
      </div>
      <div class="context-grid">
        <article class="context-card">
          <span>可用模型</span>
          <strong>{{ modelRecords.length }}</strong>
          <p>{{ primaryModelId }}</p>
        </article>
        <article class="context-card">
          <span>模型健康</span>
          <strong>{{ healthRows.length }}</strong>
          <p>{{ primaryHealthText }}</p>
        </article>
        <article class="context-card">
          <span>LLM Provider</span>
          <strong>{{ llmProviderName }}</strong>
          <p>{{ llmProviderHint }}</p>
        </article>
        <article class="context-card">
          <span>最近报告</span>
          <strong>{{ reports.length }}</strong>
          <p>{{ latestReportSummary }}</p>
        </article>
      </div>
      <div class="context-actions">
        <el-button data-testid="analysis-import-health-button" plain :disabled="healthRows.length === 0" @click="importHealthForTraining">
          导入模型健康诊断
        </el-button>
        <el-button plain :disabled="reports.length === 0" @click="activeTab = 'reports'">查看报告历史</el-button>
      </div>
    </section>

    <el-tabs v-model="activeTab" class="analysis-tabs">
      <el-tab-pane label="数据集分析" name="dataset">
        <section class="analysis-panel">
          <div class="panel-title">
            <h3>数据规则诊断</h3>
            <span>{{ datasetFile?.name ?? '未选择数据' }}</span>
          </div>
          <el-upload
            drag
            accept=".csv,.xlsx,.xls"
            :auto-upload="false"
            :limit="1"
            :on-change="handleDatasetFile"
            :on-remove="removeDatasetFile"
          >
            <div class="upload-copy">
              <strong>选择 CSV / Excel 数据集</strong>
              <span>用于缺失、重复、目标泄漏、类别不平衡等规则诊断</span>
            </div>
          </el-upload>
          <div class="inline-form">
            <el-input v-model="targetCol" placeholder="目标列，留空自动识别" clearable />
            <el-button type="primary" :disabled="!datasetFile" :loading="loading.dataset" @click="runDatasetAnalysis">
              分析数据集
            </el-button>
          </div>
        </section>
      </el-tab-pane>

      <el-tab-pane label="训练分析" name="training">
        <section class="analysis-panel">
          <div class="panel-title">
            <h3>训练指标诊断</h3>
            <span>{{ trainingInputSource }}</span>
          </div>
          <div class="helper-bar">
            <span>可从 MLOps 模型健康导入，也可手动调整 JSON。</span>
            <el-button size="small" plain :disabled="healthRows.length === 0" @click="importHealthForTraining">导入健康数据</el-button>
          </div>
          <el-input v-model="trainingJson" type="textarea" :rows="10" />
          <el-button type="primary" :loading="loading.training" @click="runTrainingAnalysis">分析训练结果</el-button>
        </section>
      </el-tab-pane>

      <el-tab-pane label="预测分析" name="prediction">
        <section class="analysis-panel">
          <div class="panel-title">
            <h3>预测分布诊断</h3>
            <span>JSON 输入</span>
          </div>
          <el-input v-model="predictionJson" type="textarea" :rows="10" />
          <el-button type="primary" :loading="loading.prediction" @click="runPredictionAnalysis">分析预测结果</el-button>
        </section>
      </el-tab-pane>

      <el-tab-pane label="报告历史" name="reports">
        <section class="analysis-panel">
          <div class="panel-title">
            <h3>Registry</h3>
            <el-button plain @click="loadReports">刷新</el-button>
          </div>
          <el-table :data="reports" height="360">
            <el-table-column prop="analysis_id" label="ID" width="130" />
            <el-table-column prop="analysis_type" label="类型" width="180" />
            <el-table-column label="风险" width="100">
              <template #default="{ row }">
                <el-tag :type="riskTagType(row.risk_level)" effect="light">{{ row.risk_level }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="summary" label="摘要" />
            <el-table-column prop="created_at" label="时间" width="180" />
          </el-table>
        </section>
      </el-tab-pane>
    </el-tabs>

    <el-alert v-if="errorMessage" class="analysis-error" type="error" :title="errorMessage" show-icon :closable="false" />

    <section class="result-grid">
      <article v-for="item in selectedResults" :key="item.analysis_type" class="result-card">
        <div class="panel-title">
          <h3>{{ resultTitle(item.analysis_type) }}</h3>
          <el-tag :type="riskTagType(item.risk_level)" effect="light">{{ item.risk_level }}</el-tag>
        </div>
        <p class="summary">{{ item.summary }}</p>
        <div class="mini-metrics">
          <span>置信度 {{ item.confidence?.toFixed(2) ?? '-' }}</span>
          <span>风险 {{ item.risks?.length ?? 0 }}</span>
          <span>证据 {{ item.evidence?.length ?? 0 }}</span>
        </div>

        <div v-if="item.risks?.length" class="list-section">
          <h4>风险</h4>
          <div v-for="risk in item.risks" :key="risk.title" class="risk-row">
            <strong>{{ risk.title }}</strong>
            <span>{{ risk.reason || risk.recommendation }}</span>
          </div>
        </div>

        <div v-if="item.recommendations?.length" class="list-section">
          <h4>建议</h4>
          <div v-for="rec in item.recommendations" :key="rec.title" class="risk-row">
            <strong>{{ rec.title }}</strong>
            <span>{{ rec.action }}</span>
          </div>
        </div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import type { UploadFile } from 'element-plus'
import { ElMessage } from 'element-plus/es/components/message/index.mjs'

import {
  analysisApi,
  type AnalysisReportRecord,
  type AnalysisResult,
  type RiskLevel,
} from '@/api/analysis'
import { llmApi } from '@/api/llm'
import { mlopsApi, type ModelHealthRow } from '@/api/mlops'
import { workflowApi, type WorkflowModelRecord } from '@/api/workflow'

const activeTab = ref('dataset')
const datasetFile = ref<File | null>(null)
const targetCol = ref('')
const datasetResult = ref<AnalysisResult | null>(null)
const trainingResult = ref<AnalysisResult | null>(null)
const predictionResult = ref<AnalysisResult | null>(null)
const reports = ref<AnalysisReportRecord[]>([])
const modelRecords = ref<WorkflowModelRecord[]>([])
const healthRows = ref<ModelHealthRow[]>([])
const llmProviderName = ref('-')
const llmProviderHint = ref('未加载')
const errorMessage = ref('')
const savingComposite = ref(false)
const contextLoading = ref(false)
const trainingInputSource = ref('手动 JSON 输入')
const loading = reactive({ dataset: false, training: false, prediction: false })

const trainingJson = ref(
  JSON.stringify(
    {
      metrics: { val_auc: 0.58, val_f1: 0.4, train_size: 80, n_features: 30 },
      params: { model_type: 'lightgbm' },
    },
    null,
    2,
  ),
)
const predictionJson = ref(
  JSON.stringify(
    {
      predictions: [0.99, 0.98, 0.97, 0.96, 0.99, 0.98],
      threshold: 0.5,
      reference_predictions: [0.05, 0.2, 0.5, 0.8, 0.95],
    },
    null,
    2,
  ),
)

const selectedResults = computed(() =>
  [datasetResult.value, trainingResult.value, predictionResult.value].filter(Boolean) as AnalysisResult[],
)
const primaryModelId = computed(() => modelRecords.value[0]?.task_id ?? '暂无可用模型')
const primaryHealth = computed(() => healthRows.value[0] ?? null)
const primaryHealthText = computed(() => {
  if (!primaryHealth.value) return '暂无健康数据'
  const auc = typeof primaryHealth.value.AUC === 'number' ? primaryHealth.value.AUC.toFixed(4) : '-'
  return `${primaryHealth.value.模型 ?? '未知模型'} / AUC ${auc} / ${primaryHealth.value.健康度 ?? '未评级'}`
})
const latestReportSummary = computed(() => reports.value[0]?.summary || '暂无历史报告')
const contextRiskText = computed(() => {
  if (healthRows.value.some((row) => row.健康度 === '异常' || row.健康度 === '需优化')) return '需关注'
  if (modelRecords.value.length === 0) return '待接入'
  return '已接入'
})
const contextRiskTag = computed(() => {
  if (contextRiskText.value === '需关注') return 'warning'
  if (contextRiskText.value === '待接入') return 'info'
  return 'success'
})

function handleDatasetFile(uploadFile: UploadFile) {
  datasetFile.value = uploadFile.raw ?? null
}

function removeDatasetFile() {
  datasetFile.value = null
}

async function runDatasetAnalysis() {
  if (!datasetFile.value) return
  await runAnalysis('dataset', async () => {
    datasetResult.value = await analysisApi.analyzeDataset({ file: datasetFile.value!, targetCol: targetCol.value })
  })
}

async function runTrainingAnalysis() {
  await runAnalysis('training', async () => {
    trainingResult.value = await analysisApi.analyzeTraining(JSON.parse(trainingJson.value))
  })
}

async function runPredictionAnalysis() {
  await runAnalysis('prediction', async () => {
    predictionResult.value = await analysisApi.analyzePrediction(JSON.parse(predictionJson.value))
  })
}

async function runAnalysis(kind: keyof typeof loading, action: () => Promise<void>) {
  loading[kind] = true
  errorMessage.value = ''
  try {
    await action()
    ElMessage.success('分析完成')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '分析失败'
  } finally {
    loading[kind] = false
  }
}

async function saveComposite() {
  savingComposite.value = true
  errorMessage.value = ''
  try {
    await analysisApi.createCompositeReport({
      title: 'AI 智能分析综合诊断',
      results: selectedResults.value,
      save_report: true,
      metadata: { source: 'vue_phase_2' },
    })
    ElMessage.success('综合报告已保存')
    await loadReports()
    activeTab.value = 'reports'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '保存综合报告失败'
  } finally {
    savingComposite.value = false
  }
}

async function loadReports() {
  try {
    reports.value = (await analysisApi.listReports()).reports
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '报告历史加载失败'
  }
}

async function loadAnalysisContext() {
  contextLoading.value = true
  errorMessage.value = ''
  try {
    const [modelsResponse, healthResponse, llmResponse] = await Promise.all([
      workflowApi.listModels(),
      mlopsApi.getModelHealth(),
      llmApi.listProviders(),
    ])
    modelRecords.value = modelsResponse.models
    healthRows.value = healthResponse.models
    const provider = llmResponse.active_provider ?? llmResponse.providers.find((item) => item.enabled) ?? llmResponse.providers[0]
    llmProviderName.value = provider?.name ?? '-'
    llmProviderHint.value = provider?.config.model ?? '暂无 Provider'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '跨模块上下文加载失败'
  } finally {
    contextLoading.value = false
  }
}

function importHealthForTraining() {
  const row = primaryHealth.value
  if (!row) {
    ElMessage.warning('暂无 MLOps 模型健康数据')
    return
  }
  trainingJson.value = JSON.stringify(
    {
      metrics: {
        val_auc: Number(row.AUC ?? 0),
        prediction_latency_ms: row['预测延迟 (ms)'] ?? null,
        call_count: row.调用次数 ?? 0,
      },
      params: {
        source: 'mlops_health',
        model_version: row.模型 ?? primaryModelId.value,
        health_status: row.健康度 ?? '未评级',
      },
    },
    null,
    2,
  )
  trainingInputSource.value = '来自 MLOps 健康'
  activeTab.value = 'training'
}

function resultTitle(type: string): string {
  const labels: Record<string, string> = {
    dataset_rules: '数据集规则诊断',
    training_rules: '训练规则诊断',
    prediction_rules: '预测规则诊断',
    composite_diagnosis: '综合诊断',
  }
  return labels[type] ?? type
}

function riskTagType(level?: RiskLevel) {
  if (level === 'critical' || level === 'high') return 'danger'
  if (level === 'medium') return 'warning'
  if (level === 'low') return 'success'
  return 'info'
}

onMounted(async () => {
  await Promise.all([loadReports(), loadAnalysisContext()])
})
</script>

<style scoped>
.analysis-workspace {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.analysis-heading,
.heading-actions,
.panel-title {
  display: flex;
  align-items: flex-start;
  gap: 14px;
}

.analysis-heading,
.panel-title {
  justify-content: space-between;
}

.heading-actions {
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.context-strip {
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #ffffff;
  padding: 18px;
  box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
}

.context-heading,
.context-actions,
.helper-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.context-heading h3 {
  margin: 0;
  color: #111827;
  font-size: 16px;
}

.context-heading p,
.context-card p,
.helper-bar span {
  margin: 4px 0 0;
  color: #64748b;
  font-size: 13px;
}

.context-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.context-card {
  min-width: 0;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
  padding: 12px;
}

.context-card span {
  color: #64748b;
  font-size: 12px;
}

.context-card strong {
  display: block;
  overflow: hidden;
  margin-top: 5px;
  color: #0f172a;
  font-size: 18px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.context-card p {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.context-actions {
  justify-content: flex-start;
  margin-top: 14px;
}

.analysis-panel,
.result-card {
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #ffffff;
  padding: 18px;
  box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
}

.panel-title {
  align-items: center;
  margin-bottom: 14px;
}

.panel-title h3 {
  margin: 0;
  color: #111827;
  font-size: 16px;
}

.panel-title span {
  color: #64748b;
  font-size: 12px;
}

.upload-copy {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #334155;
}

.upload-copy span {
  color: #64748b;
  font-size: 13px;
}

.inline-form {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  margin-top: 14px;
}

.helper-bar {
  margin-bottom: 12px;
}

.analysis-panel > .el-button {
  margin-top: 14px;
}

.analysis-error {
  margin-top: 2px;
}

.result-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.summary {
  min-height: 52px;
  color: #334155;
  line-height: 1.7;
}

.mini-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0;
}

.mini-metrics span {
  border: 1px solid #e2e8f0;
  border-radius: 999px;
  padding: 5px 9px;
  color: #475569;
  background: #f8fafc;
  font-size: 12px;
}

.list-section {
  margin-top: 14px;
}

.list-section h4 {
  margin: 0 0 8px;
  color: #111827;
  font-size: 14px;
}

.risk-row {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 0;
  border-top: 1px solid #eef2f7;
}

.risk-row strong {
  color: #0f172a;
}

.risk-row span {
  color: #64748b;
  line-height: 1.6;
}

@media (max-width: 1100px) {
  .result-grid,
  .inline-form,
  .context-grid {
    grid-template-columns: 1fr;
  }

  .analysis-heading,
  .context-heading,
  .context-actions,
  .helper-bar {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
