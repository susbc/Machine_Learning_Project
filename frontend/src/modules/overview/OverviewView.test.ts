import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import OverviewView from './views/OverviewView.vue'

vi.mock('@/shared/charts/ReadinessChart.vue', () => ({
  default: { template: '<div data-testid="readiness-chart" />' },
}))

describe('OverviewView', () => {
  it('shows the TypeScript migration replacement status across modules', () => {
    const wrapper = mount(OverviewView)

    expect(wrapper.text()).toContain('前端替代状态看板')
    expect(wrapper.text()).toContain('生产托管')
    expect(wrapper.text()).toContain('AutoML 训练')
    expect(wrapper.text()).toContain('灰度可用')
    expect(wrapper.text()).toContain('自动迭代训练')
    expect(wrapper.text()).toContain('数据管理')
    expect(wrapper.text()).not.toContain('待迁移')
    expect(wrapper.text()).not.toContain('Phase 0 迁移目标')
  })
})
