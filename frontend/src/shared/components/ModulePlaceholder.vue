<template>
  <section class="page-section">
    <div class="section-heading">
      <p class="eyebrow">{{ eyebrow }}</p>
      <h2>{{ title }}</h2>
      <p>{{ copy }}</p>
    </div>

    <div class="panel">
      <h3>当前状态</h3>
      <p>Phase 0 已建立路由、布局、状态管理和 API 基础设施。该模块将在对应阶段接入真实后端接口。</p>
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  eyebrow: string
  title: string
  copy: string
}>()
</script>
