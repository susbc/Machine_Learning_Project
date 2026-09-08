import { fileURLToPath, URL } from 'node:url'

import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [vue()],
  build: {
    chunkSizeWarningLimit: 1300,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/echarts/') || id.includes('/node_modules/zrender/')) return 'charts'
          if (id.includes('/node_modules/element-plus/') || id.includes('/node_modules/@element-plus/')) return 'element'
          if (id.includes('/node_modules/vue') || id.includes('/node_modules/pinia/')) return 'vue'
        },
      },
    },
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/openapi.json': 'http://localhost:8000',
    },
  },
})
