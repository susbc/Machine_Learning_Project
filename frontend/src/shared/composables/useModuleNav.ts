import { computed } from 'vue'

import { moduleRoutes } from '@/app/router'
import type { ModuleNavItem } from '@/shared/types/module'

export function useModuleNav() {
  const navItems = computed<ModuleNavItem[]>(() =>
    moduleRoutes.map((route) => ({
      name: route.name ?? route.path,
      path: route.path,
      label: String(route.meta?.label ?? route.name ?? route.path),
      navGroup: String(route.meta?.navGroup ?? 'core'),
    })),
  )

  return { navItems }
}
