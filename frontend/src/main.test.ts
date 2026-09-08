import { describe, expect, it, vi } from 'vitest'
import { elementPlusComponents } from './app/element-plus'

const mainMocks = vi.hoisted(() => {
  const app = {
    mount: vi.fn(),
    use: vi.fn(),
  }
  app.use.mockReturnValue(app)
  return {
    app,
    pinia: { name: 'pinia' },
    router: { name: 'router' },
  }
})

vi.mock('vue', () => ({
  createApp: vi.fn(() => mainMocks.app),
}))

vi.mock('./App.vue', () => ({
  default: { name: 'App' },
}))

vi.mock('@/app/pinia', () => ({
  createAppPinia: vi.fn(() => mainMocks.pinia),
}))

vi.mock('@/app/router', () => ({
  createAppRouter: vi.fn(() => mainMocks.router),
}))

describe('main bootstrap', () => {
  it('installs only the Element Plus components used by the application', async () => {
    await import('./main')

    expect(mainMocks.app.use).toHaveBeenCalledWith(mainMocks.pinia)
    expect(mainMocks.app.use).toHaveBeenCalledWith(mainMocks.router)
    elementPlusComponents.forEach((component) => {
      expect(mainMocks.app.use).toHaveBeenCalledWith(component)
    })
    expect(mainMocks.app.mount).toHaveBeenCalledWith('#app')
  })
})
