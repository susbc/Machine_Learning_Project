import { mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it, vi } from 'vitest'

import ManualTrainingView from './views/ManualTrainingView.vue'

const trainingMocks = vi.hoisted(() => ({
  cancelJob: vi.fn(async () => ({
    job_id: 'job_1',
    dataset_id: 'dataset_1',
    algorithm: 'logistic_regression',
    status: 'cancelled',
    progress: 60,
  })),
}))

vi.mock('@/api/training', () => ({
  trainingApi: {
    uploadDataset: vi.fn(),
    getFieldConfig: vi.fn(),
    runEda: vi.fn(),
    createJob: vi.fn(),
    getJob: vi.fn(),
    getResult: vi.fn(),
    saveResult: vi.fn(),
    cancelJob: trainingMocks.cancelJob,
  },
}))

describe('ManualTrainingView', () => {
  it('validates parameter JSON before creating a job', async () => {
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    const vm = wrapper.vm as unknown as { paramsJson: string }

    vm.paramsJson = '[]'
    await wrapper.vm.$nextTick()

    expect(wrapper.text()).toContain('模型参数必须是 JSON 对象')
    wrapper.unmount()
  })

  it('sends cancellation for a running job', async () => {
    const wrapper = mount(ManualTrainingView, { global: { plugins: [ElementPlus] } })
    const vm = wrapper.vm as unknown as {
      job: Record<string, unknown>
      cancelJob: () => Promise<void>
    }
    vm.job = {
      job_id: 'job_1',
      dataset_id: 'dataset_1',
      algorithm: 'logistic_regression',
      status: 'running',
      progress: 60,
    }

    await vm.cancelJob()

    expect(trainingMocks.cancelJob).toHaveBeenCalledWith('job_1')
    expect(vm.job.status).toBe('cancelled')
    wrapper.unmount()
  })
})
