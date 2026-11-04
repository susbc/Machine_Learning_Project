<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <button
          type="button"
          class="brand-mark-button"
          aria-label="放大查看溯算编程品牌标识"
          @click="logoZoomed = true"
        >
          <img class="brand-mark" src="@/assets/brand-logo.png" alt="溯算编程" />
        </button>
        <div>
          <strong>Agentic AutoML</strong>
          <span>溯算编程</span>
        </div>
      </div>

      <nav class="nav-list" aria-label="主导航">
        <RouterLink
          v-for="item in navItems"
          :key="String(item.name)"
          :to="item.path"
          class="nav-item"
        >
          <component :is="navIcon(item.name)" class="nav-icon" :size="17" :stroke-width="1.9" />
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>

      <img
        class="sidebar-deco"
        src="@/assets/brand-logo.png"
        alt=""
        aria-hidden="true"
      />

      <footer class="sidebar-footer">© 2026 溯算编程 susuancode</footer>
    </aside>

    <main class="workspace">
      <RouterView />
    </main>

    <div
      v-if="logoZoomed"
      class="brand-lightbox"
      role="dialog"
      aria-modal="true"
      aria-label="溯算编程品牌标识，点击任意处关闭"
      @click="logoZoomed = false"
    >
      <img src="@/assets/brand-logo-full.webp" alt="溯算编程 SUSUAN PROGRAMMING" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, type Component } from 'vue'
import {
  Box,
  Boxes,
  ChartLine,
  Database,
  FlaskConical,
  House,
  Layers,
  Link,
  Menu,
  Monitor,
  Play,
  Share2,
} from 'lucide-vue-next'

import { useModuleNav } from '@/shared/composables/useModuleNav'

const { navItems } = useModuleNav()

const NAV_ICONS: Record<string, Component> = {
  overview: House,
  automl: Layers,
  'ai-analysis': ChartLine,
  mlops: Monitor,
  'data-management': Database,
  'manual-training': Play,
  'llm-config': Box,
  database: Link,
  'agent-workflow': Share2,
  'model-registry': Boxes,
  'prediction-lab': FlaskConical,
}

function navIcon(name: unknown): Component {
  return NAV_ICONS[String(name)] ?? Menu
}

const logoZoomed = ref(false)

const closeOnEscape = (event: KeyboardEvent) => {
  if (event.key === 'Escape') {
    logoZoomed.value = false
  }
}

onMounted(() => window.addEventListener('keydown', closeOnEscape))
onBeforeUnmount(() => window.removeEventListener('keydown', closeOnEscape))
</script>
