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
          <span>溯算编程 · susuancode</span>
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
import { onBeforeUnmount, onMounted, ref } from 'vue'

import { useModuleNav } from '@/shared/composables/useModuleNav'

const { navItems } = useModuleNav()

const logoZoomed = ref(false)

const closeOnEscape = (event: KeyboardEvent) => {
  if (event.key === 'Escape') {
    logoZoomed.value = false
  }
}

onMounted(() => window.addEventListener('keydown', closeOnEscape))
onBeforeUnmount(() => window.removeEventListener('keydown', closeOnEscape))
</script>
