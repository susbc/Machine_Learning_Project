import { normalizeApiBaseUrl } from '@/api/client'

export interface AppConfig {
  apiBaseUrl: string
}

export interface AppEnv {
  VITE_API_BASE_URL?: string
}

export function resolveAppConfig(env: AppEnv = import.meta.env as AppEnv): AppConfig {
  return {
    apiBaseUrl: normalizeApiBaseUrl(env.VITE_API_BASE_URL ?? ''),
  }
}

export const appConfig = resolveAppConfig()
