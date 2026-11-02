import { mount, flushPromises } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { createPinia } from 'pinia'
import { createMemoryHistory, createRouter } from 'vue-router'
import { describe, expect, it, vi } from 'vitest'

import AppShell from './AppShell.vue'

vi.mock('@/shared/composables/useModuleNav', () => ({
  useModuleNav: () => ({
    navItems: [
      { name: 'overview', path: '/', label: '总览', navGroup: 'core' },
      { name: 'llm-config', path: '/llm-config', label: '大模型配置', navGroup: 'settings' },
    ],
  }),
}))

describe('AppShell', () => {
  it('does not show the Phase 0 initialization title on every module page', async () => {
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [{ path: '/', component: { template: '<section>当前模块</section>' } }],
    })
    await router.push('/')
    await router.isReady()

    const wrapper = mount(AppShell, {
      global: {
        plugins: [createPinia(), router, ElementPlus],
      },
    })
    await flushPromises()

    expect(wrapper.text()).not.toContain('Phase 0')
    expect(wrapper.text()).not.toContain('前端工程初始化')
    expect(wrapper.text()).not.toContain('FastAPI 已连接')
    expect(wrapper.text()).not.toContain('刷新后端状态')
  })

  it('zooms the brand logo to fullscreen and back on click', async () => {
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [{ path: '/', component: { template: '<section>当前模块</section>' } }],
    })
    await router.push('/')
    await router.isReady()

    const wrapper = mount(AppShell, {
      global: {
        plugins: [createPinia(), router, ElementPlus],
      },
    })
    await flushPromises()

    expect(wrapper.find('.brand-lightbox').exists()).toBe(false)

    await wrapper.find('.brand-mark-button').trigger('click')
    const lightbox = wrapper.find('.brand-lightbox')
    expect(lightbox.exists()).toBe(true)
    expect(lightbox.find('img').attributes('alt')).toContain('溯算编程')

    await lightbox.trigger('click')
    expect(wrapper.find('.brand-lightbox').exists()).toBe(false)
  })
})
