import { describe, expect, it } from 'vitest'

import { moduleRoutes } from './router'

describe('module routes', () => {
  it('defines the Phase 0 navigation shell routes', () => {
    const paths = moduleRoutes.map((route) => route.path)
    expect(paths).toEqual([
      '/',
      '/automl',
      '/auto-iteration',
      '/ai-analysis',
      '/mlops',
      '/data-management',
      '/manual-training',
      '/llm-config',
      '/database',
      '/agent-workflow',
      '/model-registry',
      '/prediction-lab',
    ])
  })

  it('uses stable route names for migration modules', () => {
    const names = moduleRoutes.map((route) => route.name)
    expect(names).toContain('automl')
    expect(names).toContain('auto-iteration')
    expect(names).toContain('ai-analysis')
    expect(names).toContain('data-management')
    expect(names).toContain('agent-workflow')
    expect(names).toContain('model-registry')
    expect(names).toContain('prediction-lab')
  })
})
