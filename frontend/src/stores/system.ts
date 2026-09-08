import { defineStore } from 'pinia'

import { apiClient, type HealthStatus } from '@/api/client'

interface SystemState {
  health: HealthStatus | null
  loading: boolean
  error: string
}

export const useSystemStore = defineStore('system', {
  state: (): SystemState => ({
    health: null,
    loading: false,
    error: '',
  }),
  getters: {
    isHealthy: (state) => state.health?.status === 'healthy',
  },
  actions: {
    async refreshHealth() {
      this.loading = true
      this.error = ''
      try {
        this.health = await apiClient.getHealth()
      } catch (error) {
        this.health = null
        this.error = error instanceof Error ? error.message : '无法连接后端服务'
      } finally {
        this.loading = false
      }
    },
  },
})
