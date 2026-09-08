<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">ML</div>
        <div>
          <strong>Agentic AutoML</strong>
          <span>TypeScript Frontend</span>
        </div>
      </div>

      <nav class="nav-list" aria-label="主导航">
        <RouterLink
          v-for="item in navItems"
          :key="String(item.name)"
          :to="item.path"
          class="nav-item"
        >
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>
    </aside>

    <main class="workspace">
      <header class="topbar">
        <section class="status-strip" :class="{ healthy: system.isHealthy, degraded: system.error }">
          <span class="status-dot" />
          <span v-if="system.isHealthy">FastAPI 已连接：{{ system.health?.version ?? 'unknown' }}</span>
          <span v-else-if="system.error">{{ system.error }}</span>
          <span v-else>等待后端状态检查</span>
        </section>
        <el-button :loading="system.loading" plain @click="system.refreshHealth">
          刷新后端状态
        </el-button>
      </header>

      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'

import { useModuleNav } from '@/shared/composables/useModuleNav'
import { useSystemStore } from '@/stores/system'

const { navItems } = useModuleNav()
const system = useSystemStore()

onMounted(() => {
  void system.refreshHealth()
})
</script>
