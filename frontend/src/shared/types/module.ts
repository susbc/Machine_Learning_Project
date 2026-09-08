import type { RouteRecordRaw } from 'vue-router'

export interface ModuleNavItem {
  name: NonNullable<RouteRecordRaw['name']>
  path: string
  label: string
  navGroup: string
}
